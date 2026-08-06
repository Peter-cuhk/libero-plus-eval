# RUNBOOK — LIBERO-plus 评测踩坑清单

跑评测之前先读这一页。每条都是实测结论，不是猜测。

---

## 一、必须原样复刻、**不要"修好"**的 benchmark 行为

### 1. prompt 里带着扰动后缀（七维中的六维）

`libero/libero/benchmark/__init__.py:46 grab_language_from_filename()` 对**不含 `_language_` 的**任务，
直接把文件名的下划线换成空格当作指令。于是策略实际收到的 prompt 是：

| 维度 | 实际 prompt |
|---|---|
| Background Textures | `pick up the black bowl ... on the plate **table 13**` |
| Robot Initial States | `... on the plate **view 0 0 100 0 0 initstate 121**` |
| Camera Viewpoints | `... on the plate **view 13 15 100 0 0 initstate 0**` |
| Sensor Noise | `... on the plate **view 0 0 100 0 0 initstate 0 noise 6**` |
| Objects Layout | `... on the plate **add 13**` |
| Light Conditions | `... on the plate **light 12**` |
| Language Instructions | 走另一分支，从 bddl 读 LLM 改写后的干净指令 |

**这就是官方 leaderboard 产出成绩时的行为**（README 明确说"只需把 `num_trials_per_task` 从 50 改成 1，
其余代码不用动"）。把后缀清掉会让我们的数字与所有公开数字不可比。**保持原样。**

> 顺带：这也解释了为什么公开榜上 Camera / Robot 两维普遍很低。若以后要做「prompt 清洗」的消融，
> 必须单独标注为 unfair 对照，不能进主榜。

### 2. `num_trials_per_task = 1`

LIBERO-plus 的扰动已经把 init state 编进任务名，每个任务只有 1 个 init state
（`_add_`/`_level` 分支甚至 `reshape(1, -1)`）。客户端里有
`num_trials = min(config.num_trials_per_task, len(init_states))` 兜底，但协议就是 1。

### 3. bddl 文件"缺失"是正常的

`_view_` 类任务（各 suite 约 1500 个，共 6,287 个）在 `bddl_files/` 里**没有对应文件**。
`envs/env_wrapper.py:207` 会把 `_view_<h>_<v>_<scale>_<rot>_<vert>_initstate_<n>[_noise_<m>]`
从路径里切掉，还原成基础 bddl，再把这些数值作为相机/机器人参数传进去。
已验证：**所有缺失的 bddl 恰好只是 `_view_` 那一类，没有真缺文件。**

### 4. bddl 路径必须是 `str`，不能是 `pathlib.Path`

`env_wrapper.py` 对这个参数做 `"_view_" in bddl_file_name` 和 `.split()`。
传 `Path` 会 `TypeError`。openpi 原版 `main.py` 传的是 `Path`，所以直接照抄会炸。

### 5. `LIBERO_CONFIG_PATH` 要在 import 之前设好

`benchmark/__init__.py` 在**模块导入时**就为 1,537 个 `_language_` 任务调用
`get_libero_path("bddl_files")` 并解析 bddl。配置没就位的话 import 直接失败；
另外这也意味着每个 worker 进程 import 一次要几秒。

---

## 二、环境依赖的坑

| 坑 | 症状 | 处理 |
|---|---|---|
| **ImageMagick** | `import libero...env_wrapper` 直接失败 | `env_wrapper.py:15` 在模块层 `from wand.api import library`。装 `libmagickwand-dev`（DLC 里 apt）或走 conda-forge。**所有维度都需要，不只是 Sensor Noise** |
| **opencv GUI 版** | 无头环境里 import 拉 `libGLX` 失败 | LIBERO 钉的是 `opencv-python==4.6.0.66`，装完要换成 `opencv-python-headless==4.6.0.66` |
| **EGL vendor 清单** | `MUJOCO_GL=egl` 找不到驱动 | CUDA 镜像不一定有 `/usr/share/glvnd/egl_vendor.d/`。`run_eval.sh` 自己写一个 `10_nvidia.json` 并用 `__EGL_VENDOR_LIBRARY_FILENAMES` 指过去 |
| **glib 系列 .so** | `libgthread-2.0.so.0` 找不到 | 用 `EXTRA_LIB_DIR` 指到一个含这些 .so 的目录，`run_eval.sh` 会软链进 `LD_LIBRARY_PATH` |
| **python 版本** | 依赖解析失败 | LIBERO 钉 `numpy==1.22.4` / `transformers==4.21.1`，跟着 openpi 的做法用 **py3.8** |
| **`uv run` 挂死** | 命令无输出、几分钟后超时 | 没 `source /mnt/cpfs/PeterX/env/env.sh` 时 uv 会走 pypi.org。**先 source 再跑任何 `uv run`** |

---

## 三、PPU 侧为什么不能评测（实测）

```
/usr/lib/x86_64-linux-gnu/libEGL*        → 不存在
/usr/share/glvnd/egl_vendor.d/           → 不存在
/dev/dri                                 → 不存在
libOSMesa.so.8                           → 有（纯软件渲染）
repos/openpi-icl/.venv 的 jax.devices()  → [CpuDevice(id=0)]
```

渲染和推理两头都不行。本机只用来跑 `smoke_env.py`（OSMesa，不接策略），验证装对没有。

---

## 四、成绩口径

* **overall 用 micro**（总成功数 / 总 episode 数）。七维任务数天然不等
  （Noise 1601 / Camera 1599 / Robot 1550 / Language 1537 / Layout 1525 / Light 1142 / Background 1076），
  micro 就等价于「按维度任务数加权」，对齐官方 leaderboard 的 Total 列。macro 也一并输出但不作主指标。
* `aggregate.py` 在 episode 缺失、重复或超出 split 时**直接报错退出**。要诊断数字必须显式加
  `--allow-incomplete`，那种数字**不得进台账**。
* dev split 的统计噪声（95% Wilson 半宽，实测）：单维 220 条时 **±3.3 ~ ±6.6pt**（成功率越接近 50% 越宽），
  overall 1,540 条时 **±2.2pt**。screening 的 keep/kill 阈值是 +1.5pt——
  **单维 dev 数字不足以判生死，只能看 overall，且要标注置信区间。**

---

## 五、已验证的事实（可以直接引用）

* LIBERO-plus @ `4976dc3`：spatial 2402 + object 2518 + goal 2591 + libero_10 2519 = **10,030**。
* `benchmark_dict[suite]()` 的 0-based `task_id` **精确对应** `task_classification.json[suite][task_id]`
  （`task_order_index=0` 是恒等序，10,030 条 name 零错位）。七维统计靠这个映射。
  `libero_plus_common.check_alignment()` 每次运行都会重新验证，不依赖这份记录。
* assets.zip 解压后 **448,799 个文件 / 8.3GiB**，与 zip 内条目数完全一致。
