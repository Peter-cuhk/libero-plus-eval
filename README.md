# libero-plus-eval

对 openpi 系 checkpoint 跑 **LIBERO-plus**（10,030 episodes / 七个扰动维度）与**净版 LIBERO** 回归评测。

服务于 `/mnt/cpfs/PeterX/autoresearch/PLAN.md` 的 AutoResearch 主循环：
训出 checkpoint → 评测 → 七维成绩进 `EXPERIMENTS.csv`。

## Pin 的版本

| 组件 | 版本 |
|---|---|
| LIBERO-plus | `sylvestf/LIBERO-plus` @ **`4976dc3`** |
| dev split | `splits/dev_v1.json`，seed **20260806**，每维 220 条，共 **1,540** |
| 评测端代码 | **只读**。改了 LIBERO-plus 任何一行，成绩作废 |

## 集群分工

* **训练**在 PPU / 乌兰察布（`ppu.py`）。
* **评测**在 H20 / 北京（`pai.py`）——PPU 节点没有 `libEGL`、没有 `/dev/dri`，MuJoCo 渲染跑不了。
* 两套 CPFS **不互通**，checkpoint 必须跨区搬到北京并校验后才能开评测。
* 本机 PPU 只能跑 `python/smoke_env.py`（OSMesa 软件渲染、不接策略），用来验证 LIBERO-plus 装对了。

## 用法

```bash
# 0. 一次性：北京侧备齐 LIBERO-plus / openpi-ar / py38 评测环境
bash scripts/bootstrap_h20.sh

# 1. 分片提交（先 dry-run 核对卡数/路径/镜像）
python scripts/submit_eval.py --split splits/dev_v1.json --shards 4 \
    --exp E101-xiaomi-dynamic --ckpt /mnt/oss/PeterX/outputs/E101-xiaomi-dynamic/checkpoints/.../30000 \
    --out /mnt/oss/PeterX/outputs/E101-xiaomi-dynamic/eval/dev-20260806 --dry-run

# 2. 聚合出七维成绩（分片不全会直接报错退出，不会给半份成绩）
python python/aggregate.py --run-dir /mnt/oss/PeterX/outputs/E101-xiaomi-dynamic/eval/dev-20260806 \
    --split splits/dev_v1.json --exp-id E101
```

净版 LIBERO 回归：`--benchmark clean --benchmark-root <原版 LIBERO>`（trials/task 自动变 50）。

## 目录

```
python/libero_plus_common.py   共享常量与校验（suite/维度/步数预算/split IO）
python/eval_libero_plus.py     评测客户端：分片、逐 episode JSONL、断点续跑
python/smoke_env.py            不接策略的环境冒烟（可在 CPU/OSMesa 上跑）
python/make_dev_split.py       生成冻结的 dev split
python/make_shards.py          split → N 个均衡分片
python/aggregate.py            分片 → 七维表 + overall + EXPERIMENTS.csv 片段
scripts/probe_h20.sh           北京侧环境探测（0 卡 DLC job）
scripts/bootstrap_h20.sh       北京侧一次性基建
scripts/run_eval.sh            单分片：起 policy server + 跑 client
scripts/submit_eval.py         分片提交到 H20 DLC
docs/RUNBOOK.md                踩坑清单与 benchmark 行为备忘（**先读这个**）
```

## 出处

`python/eval_libero_plus.py` 的 rollout 循环裁剪自
[openpi](https://github.com/Physical-Intelligence/openpi) 的 `examples/libero/main_parallel.py`（Apache-2.0）。
图像 180° 旋转、`resize_with_pad(224)`、每 5 步 replan、10 步静置、256px 渲染、每 suite 步数预算
全部保持一致，否则成绩无法与公开的 `pi05_libero` 数字对比。
