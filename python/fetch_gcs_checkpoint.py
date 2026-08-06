"""Pre-populate openpi's checkpoint cache from GCS, with parallel range reads.

`serve_policy --policy.dir gs://...` downloads on first use, single-threaded. From
Beijing that measured ~3.5 MB/s, i.e. about an hour of dead time at the head of
the first evaluation. This fetches the same objects into the same cache path
(`$OPENPI_DATA_HOME/<netloc>/<path>`, matching openpi's `maybe_download`), using
concurrent range requests, so the eval job finds a warm cache and starts at once.

    python python/fetch_gcs_checkpoint.py gs://openpi-assets/checkpoints/pi05_libero

Idempotent: files already present with the right size are skipped.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import pathlib
import time
import urllib.parse

CHUNK = 64 << 20  # 64MB range requests


def fetch_one(fs, remote: str, local: pathlib.Path, size: int, chunk_workers: int) -> tuple[str, int, float]:
    started = time.monotonic()
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.exists() and local.stat().st_size == size:
        return (remote, 0, 0.0)

    partial = local.with_suffix(local.suffix + ".partial")
    if size <= CHUNK or chunk_workers <= 1:
        with fs.open(remote, "rb") as src, open(partial, "wb") as dst:
            while True:
                block = src.read(8 << 20)
                if not block:
                    break
                dst.write(block)
    else:
        ranges = [(off, min(off + CHUNK, size)) for off in range(0, size, CHUNK)]
        with open(partial, "wb") as dst:
            dst.truncate(size)

        def grab(span):
            start, end = span
            data = fs.cat_file(remote, start=start, end=end)
            # Each worker opens its own handle; pwrite keeps the writes independent.
            fd = os.open(str(partial), os.O_WRONLY)
            try:
                os.pwrite(fd, data, start)
            finally:
                os.close(fd)
            return len(data)

        with concurrent.futures.ThreadPoolExecutor(max_workers=chunk_workers) as pool:
            for _ in pool.map(grab, ranges):
                pass

    got = partial.stat().st_size
    if got != size:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"{remote}: got {got} bytes, expected {size}")
    os.replace(partial, local)
    return (remote, size, time.monotonic() - started)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="gs:// URL of the checkpoint directory")
    parser.add_argument("--cache-dir", default=os.environ.get("OPENPI_DATA_HOME", "/mnt/cpfs/PeterX/data/openpi_data"))
    parser.add_argument("--file-workers", type=int, default=4)
    parser.add_argument("--chunk-workers", type=int, default=8)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import gcsfs

    parsed = urllib.parse.urlparse(args.url)
    if parsed.scheme != "gs":
        raise SystemExit(f"only gs:// URLs are supported, got {args.url}")
    remote_root = f"{parsed.netloc}/{parsed.path.strip('/')}"
    # Same layout as openpi.shared.download.maybe_download, so serve_policy hits it.
    local_root = pathlib.Path(args.cache_dir).expanduser().resolve() / parsed.netloc / parsed.path.strip("/")

    fs = gcsfs.GCSFileSystem(token="anon")
    listing = fs.find(remote_root, detail=True)
    files = {name: info for name, info in listing.items() if info.get("type") == "file"}
    total = sum(info["size"] for info in files.values())
    logging.info("%d files, %.2f GB -> %s", len(files), total / 1e9, local_root)

    jobs = []
    for name, info in sorted(files.items(), key=lambda kv: -kv[1]["size"]):
        relative = name[len(remote_root) :].lstrip("/")
        jobs.append((name, local_root / relative, info["size"]))

    started = time.monotonic()
    downloaded = 0
    skipped = 0
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.file_workers) as pool:
        futures = {
            pool.submit(fetch_one, fs, remote, local, size, args.chunk_workers): remote
            for remote, local, size in jobs
        }
        for done in concurrent.futures.as_completed(futures):
            try:
                remote, size, elapsed = done.result()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{futures[done]}: {exc}")
                logging.error("FAIL %s: %s", futures[done], exc)
                continue
            if size == 0:
                skipped += 1
                continue
            downloaded += size
            logging.info(
                "%-56s %7.1f MB in %5.1fs (%.1f MB/s) | 总计 %.2f/%.2f GB",
                remote.split("/")[-1], size / 1e6, elapsed, size / 1e6 / max(elapsed, 1e-6),
                downloaded / 1e9, total / 1e9,
            )

    elapsed = time.monotonic() - started
    logging.info(
        "完成: 下载 %.2f GB, 跳过 %d 个已有文件, 用时 %.1f 分钟 (%.1f MB/s)",
        downloaded / 1e9, skipped, elapsed / 60.0, downloaded / 1e6 / max(elapsed, 1e-6),
    )
    if errors:
        for line in errors:
            print("ERROR", line)
        return 1

    on_disk = sum(f.stat().st_size for f in local_root.rglob("*") if f.is_file())
    print(f"cache: {local_root}")
    print(f"本地字节 {on_disk} / 远端 {total}  {'一致 ✓' if on_disk == total else '不一致 ✗'}")
    return 0 if on_disk == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
