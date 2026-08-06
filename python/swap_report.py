"""Compute the language-swap self-check metrics. See docs/SWAP_PROTOCOL.md.

Takes the two paired runs -- same tasks, same init states, same seed, differing
only in the prompt -- and reports:

  LFR = 1 - S_swapped / S_correct
        Outcome level. S_swapped is how often the policy still completed THIS
        task's goal while being told to do a different one. LFR near 0 means the
        prompt made no difference to the outcome.

  PSD = mean ||chunk_correct - chunk_swapped|| / ||chunk_correct||
        Behaviour level, measured on the first predicted action chunk. Both runs
        reach that point with an identical observation, state and seed, so the
        prompt is the only difference; PSD near 0 is direct evidence that the
        policy did not read the prompt at the decision point.

    python python/swap_report.py --correct-dir <...> --swapped-dir <...>
"""

from __future__ import annotations

import argparse
import collections
import math
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import libero_plus_common as common  # noqa: E402


def load(run_dir: pathlib.Path):
    episodes = []
    for path in sorted(run_dir.glob("**/episodes.jsonl")):
        episodes.extend(common.read_episodes(path))
    return {(e["suite"], e["task_id"], e["trial"]): e for e in episodes}


def wilson(successes: int, total: int, z: float = 1.96) -> float:
    if total == 0:
        return float("nan")
    p = successes / total
    denom = 1 + z * z / total
    return 100.0 * z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correct-dir", required=True, help="Run with each task's own instruction")
    parser.add_argument("--swapped-dir", required=True, help="Run with the paired task's instruction")
    parser.add_argument("--exp-id", default="")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    correct = load(pathlib.Path(args.correct_dir))
    swapped = load(pathlib.Path(args.swapped_dir))

    keys = sorted(set(correct) & set(swapped))
    if not keys:
        raise SystemExit("两个目录没有共同的 (suite, task, trial)，无法配对比较")
    only_c, only_s = len(set(correct) - set(swapped)), len(set(swapped) - set(correct))
    if only_c or only_s:
        raise SystemExit(
            f"两次运行的 episode 集合不一致（correct 独有 {only_c}，swapped 独有 {only_s}）。"
            "swap 必须严格配对，否则 LFR 的分子分母不是同一批 episode。"
        )

    # Guard against pointing this at two identical runs.
    swapped_flagged = sum(1 for k in keys if swapped[k].get("swapped"))
    if swapped_flagged != len(keys):
        raise SystemExit(
            f"--swapped-dir 里只有 {swapped_flagged}/{len(keys)} 条 episode 标了 swapped=true；"
            "该目录多半不是 swap 条件的输出"
        )
    if any(correct[k].get("swapped") for k in keys):
        raise SystemExit("--correct-dir 里出现了 swapped=true 的 episode，两个目录传反了？")

    n_c = sum(correct[k]["success"] for k in keys)
    n_s = sum(swapped[k]["success"] for k in keys)
    s_correct = n_c / len(keys)
    s_swapped = n_s / len(keys)
    lfr = 1 - s_swapped / s_correct if s_correct > 0 else float("nan")

    # PSD on the first action chunk.
    psd_values = []
    missing_chunk = 0
    for k in keys:
        a, b = correct[k].get("first_chunk"), swapped[k].get("first_chunk")
        if not a or not b:
            missing_chunk += 1
            continue
        flat_a = [x for row in a for x in row]
        flat_b = [x for row in b for x in row]
        if len(flat_a) != len(flat_b):
            continue
        num = math.sqrt(sum((x - y) ** 2 for x, y in zip(flat_a, flat_b)))
        den = math.sqrt(sum(x * x for x in flat_a))
        if den > 0:
            psd_values.append(num / den)
    psd = sum(psd_values) / len(psd_values) if psd_values else float("nan")

    by_task = collections.defaultdict(lambda: [0, 0, 0])
    for k in keys:
        b = by_task[(k[0], k[1])]
        b[0] += correct[k]["success"]
        b[1] += swapped[k]["success"]
        b[2] += 1

    lines = [f"# 语言 swap 自检{(' — ' + args.exp_id) if args.exp_id else ''}\n"]
    lines.append(f"- 配对 episode：**{len(keys)}**（correct / swapped 各一份，任务·init state·seed 全部一致）")
    lines.append(f"- S_correct（给对指令时完成本任务）= **{100*s_correct:.2f}%** ±{wilson(n_c, len(keys)):.2f}")
    lines.append(f"- S_swapped（被告知做别的、仍完成本任务）= **{100*s_swapped:.2f}%** ±{wilson(n_s, len(keys)):.2f}")
    lines.append(f"\n## LFR = **{lfr:.3f}**   （0 = 完全无视语言，1 = 完全跟随语言）")
    if psd_values:
        lines.append(f"\n## PSD = **{psd:.4f}**   基于 {len(psd_values)} 对首个 action chunk"
                     f"{f'（{missing_chunk} 对缺 chunk，跑时忘了 --record-first-chunk？）' if missing_chunk else ''}")
    else:
        lines.append("\n## PSD = 无数据（两次运行都要加 --record-first-chunk）")

    verdict = ("完全无视语言" if lfr < 0.15 else "部分跟随" if lfr < 0.6 else "明显跟随语言")
    behav = "" if not psd_values else (
        "；且首个 action chunk 几乎不随提示词变化，**模型在决策点上没有读提示词**"
        if psd < 0.02 else "；首个 action chunk 确实随提示词变化"
    )
    lines.append(f"\n**判读：{verdict}{behav}。**")

    lines.append("\n## 按任务\n")
    lines.append("| suite | task | 原指令完成 | 被换指令后仍完成 | n |")
    lines.append("|---|---:|---:|---:|---:|")
    for (suite, task_id), (c, s, n) in sorted(by_task.items()):
        lines.append(f"| {suite} | {task_id} | {100*c/n:.1f}% | {100*s/n:.1f}% | {n} |")

    report = "\n".join(lines) + "\n"
    out_dir = pathlib.Path(args.out) if args.out else pathlib.Path(args.swapped_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "swap_report.md").write_text(report, encoding="utf-8")
    common.write_json(out_dir / "swap_report.json", {
        "exp_id": args.exp_id, "episodes": len(keys),
        "s_correct": round(s_correct, 4), "s_swapped": round(s_swapped, 4),
        "lfr": round(lfr, 4), "psd": round(psd, 5) if psd_values else None,
    })
    print(report)
    print(f"wrote {out_dir/'swap_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
