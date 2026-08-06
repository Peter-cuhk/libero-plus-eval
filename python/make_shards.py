"""Split a split-file into N balanced shards, one per evaluation job.

Tasks are interleaved (`global_index % num_shards`) rather than chunked so every
shard gets a similar suite mix. That matters because the per-suite step budget
ranges from 220 (spatial) to 520 (libero_10); a contiguous chunking would leave
the libero_10 shards running hours after the others finished.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import libero_plus_common as common  # noqa: E402


def make_shards(shard, num_shards: int):
    ordered = [(suite, task_id) for suite in common.SUITES for task_id in shard[suite]]
    shards = [{suite: [] for suite in common.SUITES} for _ in range(num_shards)]
    for position, (suite, task_id) in enumerate(ordered):
        shards[position % num_shards][suite].append(task_id)
    return shards


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    if args.shards < 1:
        raise SystemExit("--shards must be >= 1")

    shard = common.load_shard(args.split)
    total = common.shard_size(shard)
    pieces = make_shards(shard, args.shards)

    out_dir = pathlib.Path(args.out_dir)
    written = 0
    for index, piece in enumerate(pieces):
        payload = dict(piece)
        payload["_meta"] = {"source_split": str(args.split), "shard": index, "num_shards": args.shards}
        common.write_json(out_dir / f"shard_{index:02d}.json", payload)
        written += common.shard_size(piece)
        print(f"shard_{index:02d}.json  {common.shard_size(piece):5d} tasks  " +
              " ".join(f"{s.replace('libero_', '')}={len(piece[s])}" for s in common.SUITES))

    if written != total:
        raise SystemExit(f"BUG: sharding lost tasks ({written} != {total})")
    print(f"\n{args.shards} shards, {total} tasks total, no task lost or duplicated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
