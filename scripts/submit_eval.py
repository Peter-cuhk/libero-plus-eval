#!/usr/bin/env python3
"""Shard a LIBERO-plus evaluation across N single-GPU Pro5000 DLC jobs.

评测算力 2026-09 从 H20/北京 迁到 Pro5000/乌兰察布（pai-toolkit 的 `pro5000.py`,
configs/5kpro.yaml）。Pro5000 与 PPU 同 region、挂同一组数据源，所以提交端和
job 端看到的是**同一份** /mnt/cpfs/PeterX 与 /mnt/oss/PeterX——ckpt 不再需要
跨区搬运。

前置条件（ckpt 写完整、benchmark 在位、评测 venv 建好）仍然编译进 job 命令，
在头几秒就失败，而不是跑一小时才发现。

Always inspect a --dry-run before submitting for real.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shlex
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "python"))

import libero_plus_common as common  # noqa: E402
import make_shards  # noqa: E402

PAI_TOOLKIT = "/mnt/cpfs/PeterX/skills/pai-toolkit"
DEFAULT_REPO = "/mnt/cpfs/PeterX/repos/libero-plus-eval"


def build_command(args, split_path: str, shard_index: int, shard_out: str) -> str:
    """The shell command the DLC job runs. Preconditions first, then the shard."""
    checks = [
        f"test -d {shlex.quote(args.benchmark_root)} || {{ echo 'MISSING benchmark: {args.benchmark_root}'; exit 1; }}",
        f"test -x {shlex.quote(args.eval_python)} || {{ echo 'MISSING eval venv: {args.eval_python}'; exit 1; }}",
        f"test -f {shlex.quote(split_path)} || {{ echo 'MISSING split file: {split_path}'; exit 1; }}",
    ]
    if not args.ckpt.startswith("gs://"):
        checks.append(
            f"test -f {shlex.quote(args.ckpt + '/params/_METADATA')} || "
            f"{{ echo 'CHECKPOINT NOT COMMITTED (写盘未完成?): {args.ckpt}'; exit 1; }}"
        )
        if args.expect_ckpt_bytes:
            checks.append(
                f"actual=$(du -sb {shlex.quote(args.ckpt)} | cut -f1); "
                f"test \"$actual\" = '{args.expect_ckpt_bytes}' || "
                f"{{ echo \"CHECKPOINT SIZE MISMATCH: expected {args.expect_ckpt_bytes} got $actual\"; exit 1; }}"
            )

    env_assignments = " ".join(
        f"{key}={shlex.quote(value)}"
        for key, value in [
            ("CONFIG_NAME", args.config),
            ("CHECKPOINT_DIR", args.ckpt),
            ("BENCHMARK", args.benchmark),
            ("LIBERO_PLUS_ROOT", args.benchmark_root) if args.benchmark == "plus" else ("LIBERO_CLEAN_ROOT", args.benchmark_root),
            ("OPENPI_REPO", args.openpi_repo),
            (
                "PYTHONPATH",
                f"{args.openpi_repo}:{args.openpi_repo}/src:{args.openpi_repo}/packages/openpi-client/src",
            ),
            ("EVAL_PYTHON", args.eval_python),
            ("NUM_WORKERS", str(args.num_workers)),
            ("SEED", str(args.seed)),
            ("SHARD_INDEX", str(shard_index)),
            ("NUM_SHARDS", str(args.shards)),
        ]
        + ([("LANGUAGE_SWAP_FILE", args.language_swap_file)] if args.language_swap_file else [])
        + ([("RECORD_FIRST_CHUNK", "1")] if args.record_first_chunk else [])
        + ([("NUM_TRIALS_PER_TASK", str(args.num_trials_per_task))] if args.num_trials_per_task else [])
    )
    run = (
        f"{env_assignments} bash {shlex.quote(args.repo)}/scripts/run_eval.sh "
        f"{shlex.quote(split_path)} {shlex.quote(shard_out)}"
    )
    # DLC runs UserCommand under /bin/sh (dash), which rejects `-o pipefail`
    # ("Illegal option -o pipefail") and kills the job before anything starts.
    # The command is a plain && chain, so `set -eu` is all it needs; run_eval.sh
    # itself is invoked through bash and keeps its own `set -euo pipefail`.
    return " && ".join(["set -eu", *checks, run])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, help="Split file, e.g. splits/dev_v1.json")
    parser.add_argument(
        "--split-remote",
        default="",
        help="Job 侧的 split 路径（默认与本地路径相同：两端共用同一套 CPFS）",
    )
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--exp", required=True, help="Experiment id, e.g. B0b-pi05-libero-official")
    parser.add_argument("--out", required=True, help="输出根目录，例如 /mnt/oss/PeterX/outputs/<exp>/eval/full-YYYYMMDD")
    parser.add_argument("--config", default="pi05_libero")
    parser.add_argument("--ckpt", default="gs://openpi-assets/checkpoints/pi05_libero")
    parser.add_argument("--expect-ckpt-bytes", default="", help="断言 ckpt 的 `du -sb` 字节数（拷贝完整性自检）")
    parser.add_argument("--benchmark", choices=["plus", "clean"], default="plus")
    parser.add_argument("--benchmark-root", default="/mnt/cpfs/PeterX/repos/LIBERO-plus")
    parser.add_argument("--openpi-repo", default="/mnt/cpfs/PeterX/policy/openpi-ar")
    parser.add_argument("--eval-python", default="/mnt/cpfs/PeterX/env/libero-plus-eval-py38/bin/python")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="job 侧的 libero-plus-eval checkout（共享 CPFS 上的同一份）")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--num-trials-per-task", type=int, default=0, help="0 = protocol default (plus:1, clean:50)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--language-swap-file", default="",
                        help="语言 swap 自检的替换表（见 docs/SWAP_PROTOCOL.md）；路径按 job 侧解析")
    parser.add_argument("--record-first-chunk", action="store_true",
                        help="记录每条 episode 的首个 action chunk（swap 自检的 PSD 指标需要）")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--cpu", type=int, default=16)
    parser.add_argument("--memory", default="128Gi")
    parser.add_argument("--template", default="jobs/5kpro/debug-1gpu.yaml")
    parser.add_argument("--entrypoint", default="pro5000.py",
                        help="pai-toolkit 入口脚本；H20 关停后默认 pro5000.py")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-only", "--write-shards-only", dest="print_only", action="store_true",
                        help="Print the job commands and submit nothing")
    args = parser.parse_args()

    # The split file is committed and slicing is deterministic, so each job
    # derives its own slice from it. 这是 H20 时代两套 CPFS 不通留下的设计；
    # 现在两端共享同一套 CPFS，写分片文件也能工作，但保持原样——分片纯由
    # (split, shards, shard_index) 决定，成绩才可复现。
    split = common.load_shard(args.split)
    pieces = make_shards.make_shards(split, args.shards)
    split_remote = args.split_remote or args.split
    if not split_remote.startswith("/"):
        split_remote = f"{args.repo}/{split_remote}"
    print(f"split {args.split} -> {split_remote} ({common.shard_size(split)} tasks over {args.shards} shard(s))")

    commands = []
    for index, piece in enumerate(pieces):
        name = f"{args.exp}-s{index:02d}"[:60]
        command = build_command(args, split_remote, index, f"{args.out}/shard_{index:02d}")
        commands.append((name, command, common.shard_size(piece)))

    print(f"\n{'=' * 78}")
    for name, command, size in commands:
        print(f"[{name}] {size} tasks")
        print(f"  {command}\n")
    print("=" * 78)

    if args.print_only:
        print("\n--print-only: nothing submitted")
        return 0

    env = dict(os.environ)
    failures = 0
    for name, command, _ in commands:
        argv = [
            "uv", "run", args.entrypoint, "submit",
            "-n", name,
            "-c", command,
            "-t", args.template,
            "--gpu", str(args.gpu),
            "--cpu", str(args.cpu),
            "--memory", args.memory,
        ]
        if args.dry_run:
            argv.append("--dry-run")
        print(f"\n>>> {' '.join(shlex.quote(a) for a in argv[:8])} ...")
        result = subprocess.run(argv, cwd=PAI_TOOLKIT, env=env)
        failures += int(result.returncode != 0)
    if failures:
        print(f"\n{failures}/{len(commands)} submissions failed", file=sys.stderr)
        return 1
    print(f"\n{len(commands)} job(s) {'previewed' if args.dry_run else 'submitted'}")
    if not args.dry_run:
        print(f"aggregate when done:\n  python python/aggregate.py --run-dir {args.out} --split {args.split} --exp-id {args.exp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
