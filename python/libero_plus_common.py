"""Shared helpers for LIBERO-plus evaluation.

Deliberately dependency-free (stdlib only) so it can be imported from either the
py38 evaluation venv or any modern interpreter used for split/aggregation work.
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Dict, List, Sequence, Tuple

# The four suites that make up the LIBERO-plus benchmark. libero_90 is part of
# the LIBERO-100 lineage but is not scored by LIBERO-plus.
SUITES: Tuple[str, ...] = ("libero_spatial", "libero_object", "libero_goal", "libero_10")

# Expected task counts, verified against LIBERO-plus @ 4976dc3.
EXPECTED_TASK_COUNTS: Dict[str, int] = {
    "libero_spatial": 2402,
    "libero_object": 2518,
    "libero_goal": 2591,
    "libero_10": 2519,
}
EXPECTED_TOTAL = 10030

# The seven perturbation dimensions, in the order used by the LIBERO-plus paper.
CATEGORIES: Tuple[str, ...] = (
    "Objects Layout",
    "Camera Viewpoints",
    "Robot Initial States",
    "Language Instructions",
    "Light Conditions",
    "Background Textures",
    "Sensor Noise",
)

# Rollout budget per suite. Copied verbatim from openpi examples/libero so that
# our numbers stay comparable with the published pi05_libero results.
MAX_STEPS: Dict[str, int] = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def libero_plus_root() -> pathlib.Path:
    root = os.environ.get("LIBERO_PLUS_ROOT")
    if not root:
        raise RuntimeError("LIBERO_PLUS_ROOT is not set")
    return pathlib.Path(root)


def load_task_classification(libero_root: pathlib.Path | str | None = None) -> Dict[str, List[dict]]:
    """Load task_classification.json: suite -> list of {id,name,category,difficulty_level}.

    The list index (0-based) is the same task index the LIBERO benchmark object
    hands out, so `records[i]` describes `benchmark_dict[suite]().get_task(i)`.
    This alignment is asserted by `check_alignment`.
    """
    root = pathlib.Path(libero_root) if libero_root else libero_plus_root()
    path = root / "libero" / "libero" / "benchmark" / "task_classification.json"
    with open(path) as f:
        data = json.load(f)
    for suite in SUITES:
        expected = EXPECTED_TASK_COUNTS[suite]
        if len(data[suite]) != expected:
            raise RuntimeError(f"{suite}: task_classification has {len(data[suite])} entries, expected {expected}")
    return data


def check_alignment(task_classification: Dict[str, List[dict]], libero_root: pathlib.Path | str | None = None) -> None:
    """Assert task_classification[suite][i]['name'] == libero_task_map[suite][i].

    The seven-dimension breakdown is only meaningful if this holds. It is cheap,
    so every entry point checks it rather than trusting a one-off verification.
    """
    root = pathlib.Path(libero_root) if libero_root else libero_plus_root()
    path = root / "libero" / "libero" / "benchmark" / "libero_suite_task_map.py"
    namespace: dict = {}
    exec(compile(open(path).read(), str(path), "exec"), namespace)  # noqa: S102
    task_map = namespace["libero_task_map"]
    for suite in SUITES:
        names = task_map[suite]
        records = task_classification[suite]
        if len(names) != len(records):
            raise RuntimeError(f"{suite}: task_map has {len(names)} tasks but classification has {len(records)}")
        for index, (name, record) in enumerate(zip(names, records)):
            if name != record["name"]:
                raise RuntimeError(f"{suite}[{index}]: task_map={name!r} != classification={record['name']!r}")


def load_shard(path: pathlib.Path | str) -> Dict[str, List[int]]:
    """Load a split/shard file: {"libero_spatial": [task_id, ...], ...}."""
    with open(path) as f:
        data = json.load(f)
    shard = {}
    for suite in SUITES:
        ids = data.get(suite, [])
        if len(set(ids)) != len(ids):
            raise RuntimeError(f"{suite}: shard contains duplicate task ids")
        shard[suite] = sorted(int(i) for i in ids)
    if not any(shard.values()):
        raise RuntimeError(f"{path}: shard selects no tasks at all")
    return shard


def shard_size(shard: Dict[str, Sequence[int]]) -> int:
    return sum(len(ids) for ids in shard.values())


def write_json(path: pathlib.Path | str, payload) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write("\n")
    os.replace(tmp, path)


def read_episodes(path: pathlib.Path | str) -> List[dict]:
    """Read an episodes.jsonl, tolerating a truncated final line from a killed job."""
    path = pathlib.Path(path)
    if not path.is_file():
        return []
    episodes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                episodes.append(json.loads(line))
            except json.JSONDecodeError:
                # Only ever legitimate for the last line of a hard-killed run.
                continue
    return episodes
