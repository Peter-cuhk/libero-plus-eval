"""Benchmark-side smoke test: build LIBERO-plus envs without any policy server.

Picks N tasks from each of the seven perturbation dimensions, constructs the
environment, applies the init state, steps a few random actions and dumps one
agentview frame per task. This exercises every filename-decoding branch in
LIBERO-plus (`_view_`, `_initstate_`, `_noise_`, `_language_`, `_table_`,
`_light_`, `_add_`) plus the MountedPandaN robot variants.

Runs anywhere a renderer exists, including CPU-only OSMesa, so the LIBERO-plus
install can be validated before spending GPU time.
"""

from __future__ import annotations

import argparse
import collections
import logging
import os
import pathlib
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import libero_plus_common as common  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-per-category", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--out", default="smoke_out")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--benchmark-root", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = pathlib.Path(args.benchmark_root) if args.benchmark_root else common.libero_plus_root()

    classification = common.load_task_classification(root)
    common.check_alignment(classification, root)
    logging.info("task_id <-> category alignment verified (%d tasks)", common.EXPECTED_TOTAL)

    # Import after the alignment check so a bad LIBERO_CONFIG_PATH fails fast.
    import imageio
    import numpy as np
    from libero.libero import benchmark as libero_benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    picks: collections.OrderedDict = collections.OrderedDict((c, []) for c in common.CATEGORIES)
    for suite in common.SUITES:
        for index, record in enumerate(classification[suite]):
            bucket = picks[record["category"]]
            if len(bucket) < args.tasks_per_category and not any(s == suite for s, _, _ in bucket):
                bucket.append((suite, index, record))
    # Top up any dimension that could not find one task per suite.
    for suite in common.SUITES:
        for index, record in enumerate(classification[suite]):
            bucket = picks[record["category"]]
            if len(bucket) < args.tasks_per_category and (suite, index, record) not in bucket:
                bucket.append((suite, index, record))

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for category, bucket in picks.items():
        for suite, index, record in bucket:
            label = f"{category} | {suite}[{index}] {record['name'][:70]}"
            started = time.monotonic()
            try:
                task_suite = libero_benchmark.get_benchmark_dict()[suite]()
                task = task_suite.get_task(index)
                init_states = task_suite.get_task_init_states(index)
                bddl_path = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
                # str(), not Path: LIBERO-plus does string .split() on this argument.
                env = OffScreenRenderEnv(
                    bddl_file_name=str(bddl_path),
                    camera_heights=256,
                    camera_widths=256,
                )
                env.seed(args.seed + index)
                env.reset()
                obs = env.set_init_state(init_states[0])
                for _ in range(args.steps):
                    action = np.random.uniform(-0.2, 0.2, size=7).tolist()
                    obs, _, _, _ = env.step(action)
                frame = np.ascontiguousarray(obs["agentview_image"][::-1])
                png = out_dir / f"{category.replace(' ', '_')}__{suite}_{index}.png"
                imageio.imwrite(str(png), frame)
                env.close()
                nonblack = float(np.asarray(frame).std())
                ok = nonblack > 1.0
                results.append((ok, label, f"std={nonblack:.1f} prompt={task.language[:60]!r}"))
                logging.info("%s %s (%.1fs)", "OK  " if ok else "BLACK", label, time.monotonic() - started)
            except Exception:  # noqa: BLE001 - the whole point is to collect failures
                results.append((False, label, traceback.format_exc(limit=3)))
                logging.error("FAIL %s\n%s", label, traceback.format_exc(limit=3))

    ok_count = sum(1 for ok, _, _ in results if ok)
    print(f"\n{'=' * 70}\nsmoke: {ok_count}/{len(results)} tasks OK, frames in {out_dir}\n{'=' * 70}")
    for ok, label, detail in results:
        if not ok:
            print(f"[FAIL] {label}\n{detail}")
    return 0 if ok_count == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
