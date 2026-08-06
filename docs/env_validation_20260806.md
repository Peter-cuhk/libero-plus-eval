# 全量环境构造检查 — 2026-08-06

`python/validate_envs.py`，本机 PPU（8 worker，`MUJOCO_GL=osmesa`，无 GPU、无策略服务器）。
每个任务：构造 `OffScreenRenderEnv` → `set_init_state(init_states[0])` → 走 1 步 → 检查
agentview 与 wrist 两路观测都不是黑帧。

## 结果：10,030 / 10,030 通过，0 失败

| 维度 | OK | 失败 |
|---|---:|---:|
| Objects Layout | 1525 | 0 |
| Camera Viewpoints | 1599 | 0 |
| Robot Initial States | 1550 | 0 |
| Language Instructions | 1537 | 0 |
| Light Conditions | 1142 | 0 |
| Background Textures | 1076 | 0 |
| Sensor Noise | 1601 | 0 |

各维计数与 `task_classification.json` 完全一致，说明 `task_id → 维度` 的映射在全量上成立，
不只是抽样成立。

单个环境构造耗时（OSMesa 软件渲染、8 并发）：中位 **5.0s**，p95 **8.1s**；
整轮约 110 分钟。H20 上有 EGL 硬件渲染，会快很多。

## 附带发现：init_states 数量是两极分布

| `len(init_states)` | 任务数 |
|---|---:|
| 1 | 1,525 |
| 50 | 8,505 |

**1,525 恰好等于 Objects Layout 的任务数**——`get_task_init_states` 对 `_add_`/`_level`
分支会 `reshape(1, -1)`，只给一个 init state。其余 8,505 个任务各带 50 个。

这条对协议有实际影响：LIBERO-plus 规定 `num_trials_per_task = 1`，所有任务统一取
`init_states[0]`，没有问题。但**如果有人把 trials 调大**，8,505 个任务会跑更多 trial 而
Objects Layout 那 1,525 个跑不了，各维权重就被悄悄改掉了。客户端里的
`num_trials = min(config.num_trials_per_task, len(init_states))` 只保证不越界，
**不会**把协议拉回平衡——所以 LIBERO-plus 档位就别动 trials。

## 无害噪音（不用管）

* `DeprecationWarning: The binary mode of fromstring is deprecated`
  — LIBERO-plus `env_wrapper.py:52` 里 wand → cv2 的转换，Sensor Noise 维每个任务都会打。
* `Gym has been unmaintained since 2022 ...` — `gym==0.25.2` 的固定告警。
* `[robosuite WARNING] No private macro file found!` — 不影响评测。
