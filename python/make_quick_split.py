#!/usr/bin/env python3
"""Build the frozen, stratified 1,000-task AutoResearch screen."""

from __future__ import annotations

import argparse
import collections
import hashlib
import math
import pathlib
import subprocess

import libero_plus_common as common


DEFAULT_SEED = 20260820
DEFAULT_SIZE = 1000


def largest_remainder(counts: dict, total: int) -> dict:
    population = sum(counts.values())
    exact = {key: total * count / population for key, count in counts.items()}
    allocated = {key: math.floor(value) for key, value in exact.items()}
    order = sorted(counts, key=lambda key: (-(exact[key] - allocated[key]), str(key)))
    for key in order[: total - sum(allocated.values())]:
        allocated[key] += 1
    return allocated


def covered_allocation(counts: dict, total: int) -> dict:
    """Allocate proportionally while retaining at least one item per stratum."""
    if total < len(counts):
        raise ValueError(f"quota {total} cannot cover {len(counts)} non-empty strata")
    allocated = {key: 1 for key in counts}
    remaining = total - len(counts)
    capacities = {key: count - 1 for key, count in counts.items()}
    if remaining:
        extras = largest_remainder(capacities, remaining)
        for key, count in extras.items():
            allocated[key] += count
    if any(allocated[key] > counts[key] for key in counts):
        raise RuntimeError("allocation exceeds a stratum population")
    return allocated


def stable_key(seed: int, suite: str, task_id: int) -> bytes:
    return hashlib.sha256(f"{seed}:{suite}:{task_id}".encode()).digest()


def difficulty_label(record: dict) -> str:
    level = record.get("difficulty_level")
    return "unlabeled" if level is None else str(level)


def git_commit(repo: pathlib.Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def build(classification: dict, *, size: int, seed: int) -> dict:
    suite_counts = {suite: len(classification[suite]) for suite in common.SUITES}
    suite_quotas = largest_remainder(suite_counts, size)
    result = {suite: [] for suite in common.SUITES}
    population_strata = {}
    sample_strata = {}

    for suite in common.SUITES:
        groups = collections.defaultdict(list)
        for task_id, record in enumerate(classification[suite]):
            key = (record["category"], difficulty_label(record))
            groups[key].append(task_id)
        counts = {key: len(task_ids) for key, task_ids in groups.items()}
        allocation = covered_allocation(counts, suite_quotas[suite])
        for key, task_ids in groups.items():
            chosen = sorted(task_ids, key=lambda task_id: stable_key(seed, suite, task_id))[: allocation[key]]
            result[suite].extend(chosen)
            stratum = f"{suite}|{key[0]}|{key[1]}"
            population_strata[stratum] = len(task_ids)
            sample_strata[stratum] = len(chosen)
        result[suite].sort()

    if common.shard_size(result) != size:
        raise RuntimeError(f"generated {common.shard_size(result)} tasks, expected {size}")
    result["_meta"] = {
        "protocol": "quick1000_v1",
        "generator": "python/make_quick_split.py",
        "seed": seed,
        "size": size,
        "suite_population": suite_counts,
        "suite_sample": suite_quotas,
        "population_strata": population_strata,
        "sample_strata": sample_strata,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    benchmark_root = pathlib.Path(args.benchmark_root)
    classification = common.load_task_classification(benchmark_root)
    common.check_alignment(classification, benchmark_root)
    payload = build(classification, size=args.size, seed=args.seed)
    payload["_meta"]["libero_plus_commit"] = git_commit(benchmark_root)
    common.write_json(args.out, payload)
    print(f"wrote {args.out}: {common.shard_size(payload)} tasks")
    for suite in common.SUITES:
        print(f"  {suite}: {len(payload[suite])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
