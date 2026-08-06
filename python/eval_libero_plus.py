"""LIBERO-plus evaluation client for an openpi policy server.

Derived from openpi's `examples/libero/main_parallel.py` (Apache-2.0). The
rollout loop — 180 degree image rotation, resize_with_pad(224), replan every 5
steps, 10 settling steps, 256px render resolution, per-suite step budgets — is
kept byte-for-byte equivalent so results stay comparable with the published
pi05_libero numbers. What is new here is sharding, per-episode JSONL output and
resume.

Run against a policy server started by `scripts/run_eval.sh`.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import dataclasses
import io
import json
import logging
import math
import multiprocessing
import os
import pathlib
import random
import sys
import time
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import libero_plus_common as common  # noqa: E402
import make_shards  # noqa: E402

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render openpi's LIBERO training data


@dataclasses.dataclass(frozen=True)
class EvalConfig:
    host: str
    port: int
    server_handshake_timeout_s: float
    resize_size: int
    replan_steps: int
    num_steps_wait: int
    num_trials_per_task: int
    seed: int
    out_dir: str
    video_dir: Optional[str]
    record_first_chunk: bool = False


def _quat2axisangle(quat):
    """Copied from robosuite transform_utils, via openpi examples/libero.

    Kept identical to openpi's version down to `math.isclose(den, 0.0)` (whose
    default abs_tol of 0 makes it an exact-zero test) so the state vector fed to
    the policy matches the one the published pi05_libero numbers were produced
    with. The only deviation is copying the array instead of clipping the
    observation in place.
    """
    import numpy as np

    quat = np.asarray(quat, dtype=float).copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


_BENCHMARK_CACHE: dict = {}


def _get_task_suite(suite: str):
    """Build (once per worker process) the LIBERO benchmark object for a suite.

    Two reasons this is cached rather than rebuilt per task:
      - constructing it materialises ~2,500 Task tuples;
      - LIBERO-plus prints the entire task-order list on every construction,
        which is ~15KB of stdout each time. Suppress that too: at 10,030 tasks
        it would otherwise bury the run's real log lines under 150MB of noise.
    """
    if suite not in _BENCHMARK_CACHE:
        from libero.libero import benchmark as libero_benchmark

        with contextlib.redirect_stdout(io.StringIO()):
            _BENCHMARK_CACHE[suite] = libero_benchmark.get_benchmark_dict()[suite]()
    return _BENCHMARK_CACHE[suite]


def _make_env(suite: str, task_index: int, seed: int):
    """Build the LIBERO-plus environment for one task index.

    NOTE: the bddl path must be passed as `str`, not `pathlib.Path`. LIBERO-plus
    calls `"_view_" in bddl_file_name` and `bddl_file_name.split(...)` on this
    argument to decode camera / robot-init perturbations out of the filename;
    a Path raises TypeError there.
    """
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_suite = _get_task_suite(suite)
    task = task_suite.get_task(task_index)
    init_states = task_suite.get_task_init_states(task_index)

    bddl_path = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
    )
    # IMPORTANT: the seed affects object placement even with a fixed init state.
    env.seed(seed)
    return env, task, init_states


def _write_video(path: pathlib.Path, frames) -> None:
    import imageio
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    # Undo the 180 degree rotation so the saved clip is upright for humans.
    imageio.mimwrite(str(path), [np.asarray(frame[:, ::-1]) for frame in frames], fps=20)


def run_task(payload: Tuple[str, int, dict, bool, Optional[str]]) -> List[dict]:
    """Roll out every trial of one task. Returns one record per episode."""
    suite, task_index, config_dict, record_video, swapped_language = payload
    config = EvalConfig(**config_dict)

    import numpy as np
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy

    np.random.seed(config.seed + task_index)
    max_steps = common.MAX_STEPS[suite]

    env, task, init_states = _make_env(suite, task_index, config.seed + task_index)
    # `open_timeout` exists in some openpi forks but not in upstream, where the
    # constructor is (host, port, api_key). Stay compatible with both.
    try:
        client = websocket_client_policy.WebsocketClientPolicy(
            config.host, config.port, open_timeout=config.server_handshake_timeout_s
        )
    except TypeError:
        client = websocket_client_policy.WebsocketClientPolicy(config.host, config.port)

    # LIBERO-plus collapses some perturbations to a single init state, so never
    # ask for more trials than the benchmark actually provides.
    num_trials = min(config.num_trials_per_task, len(init_states))
    # Language-swap self-check: the environment (and its success predicate) stays
    # this task's; only the prompt is replaced with another task's instruction.
    # See docs/SWAP_PROTOCOL.md.
    task_description = swapped_language if swapped_language is not None else task.language

    records = []
    for trial in range(num_trials):
        started = time.monotonic()
        env.reset()
        obs = env.set_init_state(init_states[trial])
        action_plan: collections.deque = collections.deque()
        frames = [] if record_video and trial == 0 else None

        step = 0
        done = False
        error = None
        infer_calls = 0
        infer_ms_total = 0.0
        first_chunk = None
        while step < max_steps + config.num_steps_wait:
            try:
                # Let objects settle before handing control to the policy.
                if step < config.num_steps_wait:
                    obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                    step += 1
                    continue

                # IMPORTANT: rotate 180 degrees to match openpi's LIBERO training preprocessing.
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(img, config.resize_size, config.resize_size)
                )
                wrist_img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist_img, config.resize_size, config.resize_size)
                )
                if frames is not None:
                    frames.append(img)

                if not action_plan:
                    element = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        ),
                        "prompt": str(task_description),
                    }
                    infer_started = time.monotonic()
                    response = client.infer(element)
                    infer_ms_total += (time.monotonic() - infer_started) * 1000.0
                    infer_calls += 1
                    action_chunk = response["actions"]
                    if config.record_first_chunk and first_chunk is None:
                        # The first inference happens after the fixed settling
                        # steps, so both swap conditions see an identical
                        # observation/state/seed here -- the only difference is
                        # the prompt. That makes this chunk a perfectly
                        # controlled probe of prompt sensitivity.
                        first_chunk = np.asarray(action_chunk, dtype=float).round(6).tolist()
                    if len(action_chunk) < config.replan_steps:
                        raise RuntimeError(
                            f"policy returned {len(action_chunk)} actions, need {config.replan_steps}"
                        )
                    action_plan.extend(action_chunk[: config.replan_steps])

                obs, _, done, _ = env.step(action_plan.popleft().tolist())
                if done:
                    break
                step += 1
            except Exception as exc:  # noqa: BLE001 - one bad episode must not kill the shard
                error = f"{type(exc).__name__}: {exc}"
                logging.error("[%s/%d] %s", suite, task_index, error)
                break

        if frames and config.video_dir:
            outcome = "success" if done else "failure"
            try:
                _write_video(
                    pathlib.Path(config.video_dir) / f"{suite}_task{task_index:05d}_{outcome}.mp4",
                    frames,
                )
            except Exception as exc:  # noqa: BLE001 - video is a nicety, never fail the episode on it
                logging.warning("video write failed for %s/%d: %s", suite, task_index, exc)

        records.append(
            {
                "suite": suite,
                "task_id": task_index,
                "task_name": task.name,
                "language": str(task_description),
                "language_original": str(task.language),
                "swapped": swapped_language is not None,
                "trial": trial,
                "success": bool(done),
                "steps": step,
                "seed": config.seed + task_index,
                "infer_calls": infer_calls,
                "infer_ms_mean": round(infer_ms_total / infer_calls, 2) if infer_calls else None,
                "elapsed_s": round(time.monotonic() - started, 2),
                "first_chunk": first_chunk,
                "error": error,
            }
        )

    env.close()
    return records


def build_worklist(shard, done_keys, video_tasks):
    """Expand a shard into per-task work items, skipping already-finished tasks."""
    work = []
    skipped = 0
    for suite in common.SUITES:
        for task_index in shard[suite]:
            if (suite, task_index) in done_keys:
                skipped += 1
                continue
            work.append((suite, task_index, (suite, task_index) in video_tasks))
    return work, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-file", required=True, help="Split/shard JSON {suite: [task_id, ...]}")
    parser.add_argument(
        "--shard-index", type=int, default=0,
        help="Which slice of --shard-file to run. Slicing is deterministic (global_index %% num_shards), "
             "so every job derives its own slice from the committed split file -- no per-job shard files "
             "have to be shipped to the other region.",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--out", required=True, help="Output directory for episodes.jsonl / summary.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--server-handshake-timeout-s", type=float, default=300.0)
    parser.add_argument("--num-trials-per-task", type=int, default=1, help="LIBERO-plus protocol is 1")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--save-videos", type=int, default=12, help="Number of tasks to record video for")
    parser.add_argument(
        "--language-swap-file", default=None,
        help="Language-swap self-check: JSON {suite: {task_id: {swapped: '...'}}}. The environment and its "
             "success predicate stay this task's; only the prompt is replaced. See docs/SWAP_PROTOCOL.md",
    )
    parser.add_argument(
        "--record-first-chunk", action="store_true",
        help="Store the first predicted action chunk per episode (needed for the swap test's PSD metric)",
    )
    parser.add_argument("--resume", action="store_true", help="Skip tasks already present in episodes.jsonl")
    parser.add_argument(
        "--benchmark-root",
        default=None,
        help="LIBERO-plus checkout (defaults to $LIBERO_PLUS_ROOT). Point at plain LIBERO for the clean regression.",
    )
    parser.add_argument(
        "--skip-alignment-check",
        action="store_true",
        help="Only for plain LIBERO, which has no task_classification.json",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    root = pathlib.Path(args.benchmark_root) if args.benchmark_root else common.libero_plus_root()
    if not args.skip_alignment_check:
        common.check_alignment(common.load_task_classification(root), root)
        logging.info("task_id <-> category alignment verified for all %d tasks", common.EXPECTED_TOTAL)

    shard = common.load_shard(args.shard_file)
    if args.num_shards > 1:
        if not 0 <= args.shard_index < args.num_shards:
            raise SystemExit(f"--shard-index must be in [0, {args.num_shards}), got {args.shard_index}")
        full_size = common.shard_size(shard)
        shard = make_shards.make_shards(shard, args.num_shards)[args.shard_index]
        logging.info(
            "shard %d/%d of %s: %d of %d tasks",
            args.shard_index, args.num_shards, args.shard_file, common.shard_size(shard), full_size,
        )
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = out_dir / "episodes.jsonl"

    done_keys = set()
    if args.resume:
        for record in common.read_episodes(episodes_path):
            done_keys.add((record["suite"], record["task_id"]))
        logging.info("resume: %d tasks already recorded in %s", len(done_keys), episodes_path)

    swap_map = {}
    if args.language_swap_file:
        with open(args.language_swap_file) as f:
            raw = json.load(f)
        for suite in common.SUITES:
            for task_id, entry in raw.get(suite, {}).items():
                swap_map[(suite, int(task_id))] = entry["swapped"]
        missing = [k for suite in common.SUITES for k in shard[suite] if (suite, k) not in swap_map]
        if missing:
            raise SystemExit(
                f"--language-swap-file 缺少 {len(missing)} 个任务的替换指令，例如 {missing[:3]}；"
                "swap 必须整批替换，漏一个就会把普通评测混进 swapped 条件"
            )
        logging.info("language swap: %d 个任务的提示词将被替换（环境与判据不变）", len(swap_map))

    all_tasks = [(suite, i) for suite in common.SUITES for i in shard[suite]]
    rng = random.Random(args.seed)
    video_tasks = set(rng.sample(all_tasks, min(args.save_videos, len(all_tasks)))) if args.save_videos else set()

    work, skipped = build_worklist(shard, done_keys, video_tasks)
    logging.info("shard=%d tasks, skipped=%d, to run=%d", len(all_tasks), skipped, len(work))
    if not work:
        logging.info("nothing to do")
        return 0

    config = EvalConfig(
        host=args.host,
        port=args.port,
        server_handshake_timeout_s=args.server_handshake_timeout_s,
        resize_size=args.resize_size,
        replan_steps=args.replan_steps,
        num_steps_wait=args.num_steps_wait,
        num_trials_per_task=args.num_trials_per_task,
        seed=args.seed,
        out_dir=str(out_dir),
        video_dir=str(out_dir / "videos") if args.save_videos else None,
        record_first_chunk=args.record_first_chunk,
    )
    config_dict = dataclasses.asdict(config)
    payloads = [
        (suite, task_index, config_dict, record_video, swap_map.get((suite, task_index)))
        for suite, task_index, record_video in work
    ]

    started = time.monotonic()
    last_report = started
    completed = 0
    successes = 0
    failures_hard = 0
    # The parent is the only writer, so episodes.jsonl never interleaves.
    with open(episodes_path, "a", buffering=1) as sink:
        with multiprocessing.get_context("spawn").Pool(processes=min(args.num_workers, len(payloads))) as pool:
            for records in pool.imap_unordered(run_task, payloads):
                for record in records:
                    sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                    successes += int(record["success"])
                    failures_hard += int(record["error"] is not None)
                completed += 1
                # Also log on a timer: the clean-LIBERO regression runs only 10
                # tasks per shard (50 trials each), so a pure every-25-tasks rule
                # would print nothing at all until the shard finished.
                now = time.monotonic()
                if completed % 25 == 0 or completed == len(payloads) or now - last_report >= 300:
                    last_report = now
                    elapsed = now - started
                    rate = completed / elapsed if elapsed else 0.0
                    eta = (len(payloads) - completed) / rate if rate else float("nan")
                    logging.info(
                        "%d/%d tasks | success %d | errors %d | %.2f task/s | ETA %.1f min",
                        completed, len(payloads), successes, failures_hard, rate, eta / 60.0,
                    )

    summary = {
        "shard_file": str(args.shard_file),
        "tasks_in_shard": len(all_tasks),
        "tasks_run": completed,
        "tasks_skipped_resume": skipped,
        "episodes_succeeded": successes,
        "episodes_errored": failures_hard,
        "num_trials_per_task": args.num_trials_per_task,
        "seed": args.seed,
        "language_swap_file": args.language_swap_file or "",
        "wall_clock_s": round(time.monotonic() - started, 1),
    }
    common.write_json(out_dir / "summary.json", summary)
    logging.info("wrote %s", out_dir / "summary.json")
    if failures_hard:
        logging.warning("%d episodes ended on an exception - inspect the 'error' field", failures_hard)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
