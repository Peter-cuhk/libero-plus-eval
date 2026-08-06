"""Generate the frozen language-swap split and substitution map.

See docs/SWAP_PROTOCOL.md. Each task A is paired with B = (A + 5) mod 10 inside
the same suite, so the pairing is a fixed-point-free involution and needs no
random seed. The environment (and therefore the success predicate) stays A's;
only the prompt is replaced with B's instruction.

Verifies before writing that every suite really is a single shared scene with
pairwise-distinct goals -- if that stopped holding, a swapped instruction could
be satisfiable by the very goal we are checking, and the metric would be void.

    python python/make_swap_split.py --libero-root <clean LIBERO checkout>
"""

from __future__ import annotations

import argparse
import collections
import os
import pathlib
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import libero_plus_common as common  # noqa: E402

# Only libero_goal qualifies. Verified against the bddl files:
#   libero_goal    - all 10 tasks share one object set {akita_black_bowl_1,
#                    cream_cheese_1, plate_1, wine_bottle_1} and have 10 pairwise
#                    distinct goal predicates.                          -> USE
#   libero_object  - each task instantiates only 7 of the 10 grocery items and the
#                    sets differ per task, so a swapped instruction often names an
#                    object that is not even in the scene.              -> EXCLUDE
#   libero_spatial - shares the scene but ALL 10 tasks carry the same predicate
#                    (On akita_black_bowl_1 plate_1); disambiguation lives purely
#                    in the language, so a swap can be satisfied by accident.
#                                                                       -> EXCLUDE
SWAP_SUITES = ("libero_goal",)
PAIR_OFFSET = 5


def parse_bddl(path: pathlib.Path):
    """Extract fixtures / object instances / goal / language from a bddl file.

    Blocks are delimited by the next `(:tag` rather than by indentation -- an
    indentation-based regex silently returns no match on some files, and a
    caller that treats "no match" as an empty string then finds every task
    identical. That mistake makes the scene check pass vacuously, so the
    extraction here fails loudly instead.
    """
    text = path.read_text()

    def block(tag: str) -> str:
        m = re.search(rf"\(:{tag}\b(.*?)(?=\n\s*\(:|\n\s*\)\s*$)", text, re.S)
        if m is None:
            raise SystemExit(f"{path}: 解析不出 (:{tag} ...) 块，无法核实场景是否共享")
        return " ".join(m.group(1).split())

    # Compare object *sets*, not the raw text: the declaration order differs
    # between files even when the scene is identical.
    objects = frozenset(re.findall(r"([a-z_]+_\d+)\s+-", block("objects")))
    language = re.search(r"\(:language\s+([^)]*)\)", text)
    return {
        "scene": (block("fixtures"), objects),
        "goal": block("goal"),
        "language": language.group(1).strip() if language else "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--libero-root",
        default="/mnt/cpfs/PeterX/policy/openpi-ar/third_party/libero",
        help="Plain LIBERO checkout (the swap test runs on clean LIBERO by design)",
    )
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    root = pathlib.Path(args.libero_root) / "libero" / "libero"
    namespace: dict = {}
    task_map_path = root / "benchmark" / "libero_suite_task_map.py"
    exec(compile(task_map_path.read_text(), str(task_map_path), "exec"), namespace)  # noqa: S102
    task_map = namespace["libero_task_map"]

    split: dict = {suite: [] for suite in common.SUITES}
    swap_map: dict = {}
    for suite in SWAP_SUITES:
        names = task_map[suite]
        if len(names) != 10:
            raise SystemExit(f"{suite}: expected 10 tasks, got {len(names)}")
        info = [parse_bddl(root / "bddl_files" / suite / f"{n}.bddl") for n in names]

        scenes = {i["scene"] for i in info}
        if len(scenes) != 1:
            raise SystemExit(
                f"{suite}: 10 个任务分属 {len(scenes)} 个不同场景（物体集合不同）；"
                "swap 要求同一场景，否则换来的指令会点名场上没有的物体"
            )
        goals = [i["goal"] for i in info]
        if len(set(goals)) != len(goals):
            dupes = [g for g, n in collections.Counter(goals).items() if n > 1]
            raise SystemExit(f"{suite}: goal predicates are not pairwise distinct ({len(dupes)} repeated)")

        split[suite] = list(range(10))
        swap_map[suite] = {}
        for a in range(10):
            b = (a + PAIR_OFFSET) % 10
            if info[a]["goal"] == info[b]["goal"]:
                raise SystemExit(f"{suite}[{a}] and [{b}] share a goal; pairing is void")
            swap_map[suite][str(a)] = {
                "swap_from": b,
                "original": info[a]["language"],
                "swapped": info[b]["language"],
            }
        shared = sorted(next(iter(scenes))[1])
        print(f"{suite}: 单一场景 ✓  10 个 goal 两两不同 ✓  配对 A <-> (A+{PAIR_OFFSET})%10")
        print(f"   共享物体: {' '.join(shared)}")

    repo = pathlib.Path(__file__).resolve().parent.parent
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else repo / "splits"
    split["_meta"] = {
        "note": "language-swap self-check, clean LIBERO only. See docs/SWAP_PROTOCOL.md",
        "suites": list(SWAP_SUITES),
        "total": common.shard_size(split),
    }
    common.write_json(out_dir / "swap_v1.json", split)
    swap_map["_meta"] = {
        "pair_offset": PAIR_OFFSET,
        "libero_root": str(args.libero_root),
        "note": "FROZEN. Changing the pairing requires a new v2 and a re-run of the baseline.",
    }
    common.write_json(out_dir / "swap_v1_map.json", swap_map)

    print(f"\nwrote {out_dir/'swap_v1.json'} ({common.shard_size(split)} tasks) and {out_dir/'swap_v1_map.json'}")
    for suite in SWAP_SUITES:
        a0 = swap_map[suite]["0"]
        print(f"  {suite}[0] 原指令: {a0['original']}")
        print(f"  {suite}[0] 换成  : {a0['swapped']}  (来自任务 {a0['swap_from']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
