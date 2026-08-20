"""Aggregate shard results into the seven-dimension LIBERO-plus score sheet.

Refuses to print a score unless the episode set is complete and duplicate-free.
A partially-finished run must fail loudly rather than quietly report a number
that looks like a real result -- the AutoResearch ledger only accepts figures
computed from complete eval output.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import libero_plus_common as common  # noqa: E402


def wilson_halfwidth(successes: int, total: int, z: float = 1.96) -> float:
    """Half-width of the 95% Wilson interval, in percentage points."""
    if total == 0:
        return float("nan")
    p = successes / total
    denom = 1 + z * z / total
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return 100.0 * margin


def collect(run_dir: pathlib.Path):
    episodes = []
    sources = sorted(run_dir.glob("**/episodes.jsonl"))
    for path in sources:
        episodes.extend(common.read_episodes(path))
    return episodes, sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Directory containing shard_*/episodes.jsonl")
    parser.add_argument("--split", required=True, help="The split file the run was supposed to cover")
    parser.add_argument("--exp-id", default="", help="Experiment id for the EXPERIMENTS.csv row")
    parser.add_argument("--benchmark-root", default=None)
    parser.add_argument("--out", default=None, help="Write report.md / summary.json here (default: run-dir)")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Report anyway when episodes are missing. Result is NOT ledger-eligible.",
    )
    parser.add_argument(
        "--no-categories",
        action="store_true",
        help="Plain LIBERO regression: no task_classification.json, so report overall + per-suite only",
    )
    args = parser.parse_args()

    if args.no_categories:
        classification = None
    else:
        root = pathlib.Path(args.benchmark_root) if args.benchmark_root else common.libero_plus_root()
        classification = common.load_task_classification(root)
        common.check_alignment(classification, root)

    expected_shard = common.load_shard(args.split)
    expected_keys = {(suite, task_id) for suite in common.SUITES for task_id in expected_shard[suite]}
    with open(args.split) as split_file:
        split_payload = json.load(split_file)
    split_protocol = split_payload.get("_meta", {}).get("protocol", "")

    run_dir = pathlib.Path(args.run_dir)
    episodes, sources = collect(run_dir)
    if not episodes:
        raise SystemExit(f"no episodes found under {run_dir}")

    seen = collections.Counter((e["suite"], e["task_id"], e["trial"]) for e in episodes)
    duplicates = [key for key, count in seen.items() if count > 1]
    got_keys = {(e["suite"], e["task_id"]) for e in episodes}
    missing = expected_keys - got_keys
    unexpected = got_keys - expected_keys
    errored = [e for e in episodes if e.get("error")]

    print(f"sources: {len(sources)} episodes.jsonl file(s)")
    print(f"episodes: {len(episodes)}  tasks covered: {len(got_keys)} / {len(expected_keys)} expected")
    if duplicates:
        print(f"DUPLICATES: {len(duplicates)} (suite,task,trial) keys appear more than once, e.g. {duplicates[:3]}")
    if unexpected:
        print(f"UNEXPECTED: {len(unexpected)} tasks not in the split, e.g. {sorted(unexpected)[:3]}")
    if missing:
        print(f"MISSING: {len(missing)} tasks never ran, e.g. {sorted(missing)[:3]}")
    if errored:
        print(f"ERRORED: {len(errored)} episodes ended on an exception (counted as failures)")

    incomplete = bool(missing or duplicates or unexpected)
    if incomplete and not args.allow_incomplete:
        raise SystemExit(
            "\nRefusing to report a score from an incomplete or inconsistent run.\n"
            "Re-run the missing shards (eval_libero_plus.py --resume), or pass --allow-incomplete\n"
            "if you explicitly want a diagnostic number that must NOT go in the ledger."
        )

    lookup = {}
    if classification is not None:
        for suite in common.SUITES:
            for index, record in enumerate(classification[suite]):
                lookup[(suite, index)] = record

    weight_lookup = {}
    if split_protocol == "quick1000_v1":
        population_strata = collections.Counter()
        sample_strata = collections.Counter()
        for suite in common.SUITES:
            for record in classification[suite]:
                level = record.get("difficulty_level")
                population_strata[(suite, record["category"], "unlabeled" if level is None else str(level))] += 1
            for task_id in expected_shard[suite]:
                record = classification[suite][task_id]
                level = record.get("difficulty_level")
                sample_strata[(suite, record["category"], "unlabeled" if level is None else str(level))] += 1
        uncovered = set(population_strata) - set(sample_strata)
        if uncovered:
            raise SystemExit(f"quick split leaves population strata uncovered: {sorted(uncovered)[:3]}")
        weight_lookup = {key: population_strata[key] / sample_strata[key] for key in population_strata}

    by_category = collections.defaultdict(lambda: [0, 0])
    by_suite = collections.defaultdict(lambda: [0, 0])
    by_difficulty = collections.defaultdict(lambda: [0, 0])
    total = [0, 0]
    weighted_total = [0.0, 0.0]
    for episode in episodes:
        record = lookup.get((episode["suite"], episode["task_id"]))
        if record is None and classification is not None:
            continue
        hit = int(bool(episode["success"]))
        weight = 1.0
        if weight_lookup:
            level = record.get("difficulty_level")
            key = (episode["suite"], record["category"], "unlabeled" if level is None else str(level))
            weight = weight_lookup[key]
        weighted_total[0] += hit * weight
        weighted_total[1] += weight
        buckets = [by_suite[episode["suite"]], total]
        if record is not None:
            buckets.append(by_category[record["category"]])
            # ~121 of the 10,030 tasks carry no difficulty label.
            buckets.append(by_difficulty[record.get("difficulty_level") or "unlabeled"])
        for bucket in buckets:
            bucket[0] += hit
            bucket[1] += 1

    def rate(bucket):
        return 100.0 * bucket[0] / bucket[1] if bucket[1] else float("nan")

    micro = rate(total)
    weighted_micro = 100.0 * weighted_total[0] / weighted_total[1]
    scored_categories = [c for c in common.CATEGORIES if by_category[c][1]]
    macro = sum(rate(by_category[c]) for c in scored_categories) / len(scored_categories) if scored_categories else micro

    title = "净版 LIBERO 回归报告" if args.no_categories else "LIBERO-plus 评测报告"
    lines = []
    lines.append(f"# {title}{(' — ' + args.exp_id) if args.exp_id else ''}\n")
    lines.append(f"- split: `{args.split}`")
    lines.append(f"- run_dir: `{run_dir}`")
    trials_per_task = total[1] / len(got_keys) if got_keys else 0
    lines.append(
        f"- tasks: **{len(got_keys)} / {len(expected_keys)}** expected; "
        f"episodes: **{total[1]}** ({trials_per_task:.0f} trial/task), errored {len(errored)}"
    )
    lines.append(f"- **overall (micro) = {micro:.2f}%**  ±{wilson_halfwidth(*total):.2f}pt (95% Wilson)")
    if weight_lookup:
        lines.append(f"- **selection score (population-weighted) = {weighted_micro:.2f}%**")
    if scored_categories:
        lines.append(f"- overall (macro over 7 dims) = {macro:.2f}%")
    if incomplete:
        lines.append("\n> ⚠️ **不完整/不一致的运行，此数字不得进台账。**")
    if scored_categories:
        lines.append("\n## 七维\n")
        lines.append("| 维度 | 成功 | 总数 | 成功率 | ±95% |")
        lines.append("|---|---:|---:|---:|---:|")
        for category in common.CATEGORIES:
            bucket = by_category[category]
            if not bucket[1]:
                continue
            lines.append(
                f"| {category} | {bucket[0]} | {bucket[1]} | {rate(bucket):.2f}% | ±{wilson_halfwidth(*bucket):.2f} |"
            )
    lines.append("\n## 按 suite\n")
    lines.append("| suite | 成功 | 总数 | 成功率 |")
    lines.append("|---|---:|---:|---:|")
    for suite in common.SUITES:
        bucket = by_suite[suite]
        if bucket[1]:
            lines.append(f"| {suite} | {bucket[0]} | {bucket[1]} | {rate(bucket):.2f}% |")
    if by_difficulty:
        lines.append("\n## 按难度\n")
        lines.append("| level | 成功 | 总数 | 成功率 |")
        lines.append("|---|---:|---:|---:|")
        for level in sorted(by_difficulty, key=lambda x: (isinstance(x, str), x)):
            bucket = by_difficulty[level]
            label = level if isinstance(level, str) else f"L{level}"
            lines.append(f"| {label} | {bucket[0]} | {bucket[1]} | {rate(bucket):.2f}% |")

    lines.append("\n## EXPERIMENTS.csv 片段\n")
    if scored_categories:
        csv_order = ["Objects Layout", "Camera Viewpoints", "Robot Initial States", "Language Instructions",
                     "Light Conditions", "Background Textures", "Sensor Noise"]
        csv_cells = ",".join(f"{rate(by_category[c]):.2f}" if by_category[c][1] else "" for c in csv_order)
        lines.append("`layout,camera,robot_init,language,light,background,noise` 七列：\n")
        lines.append(f"```\n{csv_cells}\n```")
    lines.append(f"\noverall（micro）= `{micro:.2f}`" + ("  → `clean_libero` 列\n" if args.no_categories else "\n"))

    report = "\n".join(lines) + "\n"
    out_dir = pathlib.Path(args.out) if args.out else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    common.write_json(
        out_dir / "aggregate.json",
        {
            "exp_id": args.exp_id,
            "split": str(args.split),
            "complete": not incomplete,
            "episodes": total[1],
            "expected": len(expected_keys),
            "errored": len(errored),
            "overall_micro": round(micro, 4),
            "overall_weighted": round(weighted_micro, 4) if weight_lookup else None,
            "selection_score": round(weighted_micro if weight_lookup else micro, 4),
            "overall_macro": round(macro, 4),
            "by_category": {c: {"successes": by_category[c][0], "episodes": by_category[c][1],
                                "rate": round(rate(by_category[c]), 4)}
                            for c in common.CATEGORIES if by_category[c][1]},
            "by_suite": {s: {"successes": by_suite[s][0], "episodes": by_suite[s][1],
                             "rate": round(rate(by_suite[s]), 4)}
                         for s in common.SUITES if by_suite[s][1]},
        },
    )
    print()
    print(report)
    print(f"wrote {out_dir / 'report.md'} and {out_dir / 'aggregate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
