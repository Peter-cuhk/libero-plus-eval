"""Construct every LIBERO-plus environment once and report the ones that fail.

A full evaluation is 10,030 environments built from filenames that LIBERO-plus
decodes with regex into camera poses, robot variants, textures and init states.
A single task with a missing asset or an unparseable name shows up hours into a
GPU run as one silent failed episode. This sweep finds them up front, on CPU,
with no policy server and no GPU.

    MUJOCO_GL=osmesa python python/validate_envs.py --workers 8

Writes env_validation.json: per-task status plus a summary by dimension.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import multiprocessing
import os
import pathlib
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import libero_plus_common as common  # noqa: E402

_SUITE_CACHE: dict = {}


def _get_task_suite(suite: str):
    if suite not in _SUITE_CACHE:
        from libero.libero import benchmark as libero_benchmark

        with contextlib.redirect_stdout(io.StringIO()):
            _SUITE_CACHE[suite] = libero_benchmark.get_benchmark_dict()[suite]()
    return _SUITE_CACHE[suite]


def check_task(payload):
    """Build one env, apply its init state, take one step, look at the frame."""
    suite, task_index, steps, resolution = payload
    started = time.monotonic()
    try:
        import numpy as np
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        task_suite = _get_task_suite(suite)
        task = task_suite.get_task(task_index)
        init_states = task_suite.get_task_init_states(task_index)
        bddl_path = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        # str(), not Path: LIBERO-plus does string .split() on this argument.
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_path), camera_heights=resolution, camera_widths=resolution
        )
        env.seed(7 + task_index)
        env.reset()
        obs = env.set_init_state(init_states[0])
        for _ in range(steps):
            obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
        agent = np.asarray(obs["agentview_image"])
        wrist = np.asarray(obs["robot0_eye_in_hand_image"])
        env.close()
        return {
            "suite": suite,
            "task_id": task_index,
            "ok": bool(agent.std() > 1.0 and wrist.std() > 1.0),
            "n_init_states": int(len(init_states)),
            "agent_std": round(float(agent.std()), 2),
            "wrist_std": round(float(wrist.std()), 2),
            "prompt": str(task.language),
            "elapsed_s": round(time.monotonic() - started, 2),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - collecting failures is the point
        return {
            "suite": suite,
            "task_id": task_index,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=4),
            "elapsed_s": round(time.monotonic() - started, 2),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--split", default=None, help="Restrict to a split file (default: all 10,030)")
    parser.add_argument("--out", default="env_validation.json")
    parser.add_argument("--resume", action="store_true", help="Skip tasks already present in --out")
    parser.add_argument("--benchmark-root", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = pathlib.Path(args.benchmark_root) if args.benchmark_root else common.libero_plus_root()
    classification = common.load_task_classification(root)
    common.check_alignment(classification, root)

    if args.split:
        shard = common.load_shard(args.split)
    else:
        shard = {s: list(range(common.EXPECTED_TASK_COUNTS[s])) for s in common.SUITES}

    out_path = pathlib.Path(args.out)
    done = {}
    if args.resume and out_path.is_file():
        for record in json.loads(out_path.read_text()).get("tasks", []):
            done[(record["suite"], record["task_id"])] = record
        logging.info("resume: %d tasks already checked", len(done))

    work = [
        (suite, task_id, args.steps, args.resolution)
        for suite in common.SUITES
        for task_id in shard[suite]
        if (suite, task_id) not in done
    ]
    logging.info("checking %d environments with %d workers", len(work), args.workers)

    results = list(done.values())
    started = time.monotonic()
    failures = 0
    if work:
        with multiprocessing.get_context("spawn").Pool(processes=min(args.workers, len(work))) as pool:
            for i, record in enumerate(pool.imap_unordered(check_task, work, chunksize=4), start=1):
                results.append(record)
                if not record["ok"]:
                    failures += 1
                    logging.warning(
                        "FAIL %s[%d] %s", record["suite"], record["task_id"],
                        (record.get("error") or "black frame")[:160],
                    )
                if i % 200 == 0 or i == len(work):
                    elapsed = time.monotonic() - started
                    rate = i / elapsed
                    logging.info(
                        "%d/%d | fail %d | %.1f env/s | ETA %.1f min",
                        i, len(work), failures, rate, (len(work) - i) / rate / 60.0,
                    )

    lookup = {(s, i): r for s in common.SUITES for i, r in enumerate(classification[s])}
    by_category: dict = {}
    for record in results:
        meta = lookup.get((record["suite"], record["task_id"]))
        category = meta["category"] if meta else "?"
        bucket = by_category.setdefault(category, {"ok": 0, "failed": 0})
        bucket["ok" if record["ok"] else "failed"] += 1

    bad = [r for r in results if not r["ok"]]
    common.write_json(
        out_path,
        {
            "checked": len(results),
            "failed": len(bad),
            "by_category": by_category,
            "failures": bad[:200],
            "tasks": results,
        },
    )

    print(f"\n{'=' * 66}")
    print(f"环境构造检查: {len(results) - len(bad)}/{len(results)} OK, {len(bad)} 失败")
    print(f"{'维度':<24}{'OK':>8}{'失败':>8}")
    for category in common.CATEGORIES:
        bucket = by_category.get(category)
        if bucket:
            print(f"{category:<24}{bucket['ok']:>8}{bucket['failed']:>8}")
    print("=" * 66)
    for record in bad[:20]:
        print(f"[FAIL] {record['suite']}[{record['task_id']}] {(record.get('error') or 'black frame')[:150]}")
    print(f"\n详情写入 {out_path}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
