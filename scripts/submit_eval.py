#!/usr/bin/env python3
"""Shard a LIBERO-plus evaluation across N single-GPU H20 DLC jobs.

Submission happens from the PPU dev box, but the jobs run in Beijing against a
different CPFS. Nothing about the Beijing side can be stat'd from here, so every
precondition (checkpoint committed, benchmark present, eval venv built) is
compiled into the job command itself and fails the job in its first seconds
rather than after an hour of rollouts.

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
            f"{{ echo 'CHECKPOINT NOT COMMITTED (cross-region copy incomplete?): {args.ckpt}'; exit 1; }}"
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
            ("EVAL_PYTHON", args.eval_python),
            ("NUM_WORKERS", str(args.num_workers)),
            ("SEED", str(args.seed)),
            ("SHARD_INDEX", str(shard_index)),
            ("NUM_SHARDS", str(args.shards)),
        ]
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
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--exp", required=True, help="Experiment id, e.g. B0b-pi05-libero-official")
    parser.add_argument("--out", required=True, help="Beijing output root, e.g. /mnt/oss/PeterX/outputs/<exp>/eval/full-YYYYMMDD")
    parser.add_argument("--config", default="pi05_libero")
    parser.add_argument("--ckpt", default="gs://openpi-assets/checkpoints/pi05_libero")
    parser.add_argument("--expect-ckpt-bytes", default="", help="Assert `du -sb` of the checkpoint matches (cross-region integrity)")
    parser.add_argument("--benchmark", choices=["plus", "clean"], default="plus")
    parser.add_argument("--benchmark-root", default="/mnt/cpfs/PeterX/repos/LIBERO-plus")
    parser.add_argument("--openpi-repo", default="/mnt/cpfs/PeterX/policy/openpi-ar")
    parser.add_argument("--eval-python", default="/mnt/cpfs/PeterX/env/libero-plus-eval-py38/bin/python")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="libero-plus-eval checkout on the Beijing CPFS")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--num-trials-per-task", type=int, default=0, help="0 = protocol default (plus:1, clean:50)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--cpu", type=int, default=16)
    parser.add_argument("--memory", default="128Gi")
    parser.add_argument("--template", default="jobs/h20/debug-1gpu.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-only", "--write-shards-only", dest="print_only", action="store_true",
                        help="Print the job commands and submit nothing")
    args = parser.parse_args()

    # The split file is committed and present in both regions, and slicing is
    # deterministic, so each job derives its own slice. Nothing per-job needs to
    # be written here -- the two CPFS filesystems are not shared, so a shard file
    # written next to this script would simply not exist where the job runs.
    split = common.load_shard(args.split)
    pieces = make_shards.make_shards(split, args.shards)
    split_remote = args.split
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
            "uv", "run", "pai.py", "submit",
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
