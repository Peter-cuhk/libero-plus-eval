"""Generate the frozen LIBERO-plus dev split (stratified by perturbation dimension).

The AutoResearch plan calls for ~220 episodes per dimension (~1,540 total) so a
screening run costs ~15% of the full 10,030-episode benchmark. Within each
dimension, tasks are drawn proportionally from the four suites so the suite mix
mirrors the full set.

Generated once with a fixed seed and then frozen: re-running with the same
--seed and --per-category reproduces the identical file, and the committed
splits/dev_v1.json is what every screening run must use.
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import libero_plus_common as common  # noqa: E402


def build_split(classification, per_category: int, seed: int):
    rng = random.Random(seed)
    by_category = collections.defaultdict(lambda: collections.defaultdict(list))
    for suite in common.SUITES:
        for index, record in enumerate(classification[suite]):
            by_category[record["category"]][suite].append(index)

    selected = {suite: [] for suite in common.SUITES}
    stats = {}
    for category in common.CATEGORIES:
        per_suite = by_category[category]
        total = sum(len(v) for v in per_suite.values())
        take = min(per_category, total)

        # Largest-remainder allocation keeps the suite mix proportional and the
        # total exactly equal to `take`.
        exact = {suite: len(ids) * take / total for suite, ids in per_suite.items()}
        quota = {suite: int(value) for suite, value in exact.items()}
        remainder = take - sum(quota.values())
        for suite, _ in sorted(exact.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=True)[:remainder]:
            quota[suite] += 1

        chosen = {}
        for suite, ids in per_suite.items():
            picked = rng.sample(sorted(ids), quota[suite])
            selected[suite].extend(picked)
            chosen[suite] = len(picked)
        stats[category] = {"pool": total, "selected": take, "per_suite": chosen}

    for suite in common.SUITES:
        selected[suite] = sorted(selected[suite])
        if len(set(selected[suite])) != len(selected[suite]):
            raise RuntimeError(f"{suite}: duplicate task ids in split")
    return selected, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-category", type=int, default=220)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--out", default=None, help="default: splits/dev_v1.json next to this repo")
    parser.add_argument("--benchmark-root", default=None)
    parser.add_argument("--full", action="store_true", help="also (re)write splits/full.json")
    args = parser.parse_args()

    repo = pathlib.Path(__file__).resolve().parent.parent
    root = pathlib.Path(args.benchmark_root) if args.benchmark_root else common.libero_plus_root()
    classification = common.load_task_classification(root)
    common.check_alignment(classification, root)

    if args.full:
        full = {suite: list(range(common.EXPECTED_TASK_COUNTS[suite])) for suite in common.SUITES}
        common.write_json(repo / "splits" / "full.json", full)
        print(f"wrote splits/full.json ({common.shard_size(full)} tasks)")

    selected, stats = build_split(classification, args.per_category, args.seed)
    out = pathlib.Path(args.out) if args.out else repo / "splits" / "dev_v1.json"
    payload = dict(selected)
    payload["_meta"] = {
        "seed": args.seed,
        "per_category": args.per_category,
        "total": common.shard_size(selected),
        "libero_plus_commit": "4976dc3",
        "stats": stats,
        "note": "FROZEN. Regenerating with the same seed/per_category reproduces this file.",
    }
    common.write_json(out, payload)

    print(f"wrote {out} ({common.shard_size(selected)} tasks)")
    print(f"{'dimension':24s} {'pool':>6s} {'picked':>7s}  per-suite")
    for category, entry in stats.items():
        mix = " ".join(f"{s.replace('libero_', '')}={n}" for s, n in entry["per_suite"].items())
        print(f"{category:24s} {entry['pool']:6d} {entry['selected']:7d}  {mix}")
    for suite in common.SUITES:
        print(f"  {suite:16s} {len(selected[suite]):5d} / {common.EXPECTED_TASK_COUNTS[suite]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
