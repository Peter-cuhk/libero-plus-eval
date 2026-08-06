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
| **ImageMagick** | `import libero...env_wrapper` 直接失败 | `env_wrapper.py:15` 在模块层 `from wand.api import library`。**所有维度都需要，不只是 Sensor Noise**。用 micromamba 装进评测 prefix（不动系统 apt），运行时设 `MAGICK_HOME=<prefix>` |
| **glib 系列 .so** | `libgthread-2.0.so.0` 找不到 | DLC 镜像里没有。同一个 conda prefix 的 `lib/` 就带了 `libglib-2.0.so.0` / `libgthread-2.0.so.0`，把它加进 `LD_LIBRARY_PATH` 即可，不用额外找 |
| **opencv GUI 版** | 无头环境里 import 拉 `libGLX` 失败 | LIBERO 钉的是 `opencv-python==4.6.0.66`，要换成 `opencv-python-headless==4.6.0.66` |
| **EGL vendor 清单** | `MUJOCO_GL=egl` 枚举到 0 个设备 | 见 §3.2。`run_eval.sh` 自动生成 `10_nvidia.json` 并用 `__EGL_VENDOR_LIBRARY_FILENAMES` 指过去 |
| **robomimic → torchvision** | `Failed to build torchvision==0.25.0+v0.1.0.ppu2.1.0` | 集群内部 pypi 镜像里有 PPU 专用的 torchvision 源码包，构建时找不到 torch 就炸。`robomimic` 只被 `libero/lifelong` 用到，**评测不需要，直接不装** |
| **`bddl` 缺 `future`** | `ModuleNotFoundError: No module named 'future'` | `bddl/backend_abc.py` 导入 `future.utils` 但没声明依赖，要手动装 `future==0.18.2` |
| **内部 pypi 镜像没有 cp38 的 torch** | `no version of torch==1.11.0` | 集群默认索引 `aiext-pypi.mirrors.aliyuncs.com` 一个 cp38 torch wheel 都没有。装依赖时用 `--default-index https://mirrors.aliyun.com/pypi/simple/` |
| **python 版本** | 依赖解析失败 | LIBERO 钉 `numpy==1.22.4`，跟着 openpi 的做法用 **py3.8** |
| **`uv run` 挂死** | 命令无输出、几分钟后超时 | 没 `source /mnt/cpfs/PeterX/env/env.sh` 时 uv 会走 pypi.org。**先 source 再跑任何 `uv run`** |

---

## 三、北京 H20 侧的实测事实

探测 job：`dlcrzmz9mv5h7ats`（0 卡）、`dlcoe0op6ptpeu6g`（1 卡）、`dlc1pki9zen2ygzj`（EGL 修复验证）。

### 3.1 **0 卡 job 探不出渲染栈**

`--gpu 0` 的容器不会注入 NVIDIA 运行时，`libEGL*` / `libGL*` / `nvidia-smi` 全部显示缺失。
**任何关于 EGL 的结论都必须用 `--gpu >= 1` 的 job 得出**，否则是假阴性。

### 3.2 EGL：有驱动，但没有 ICD 清单

1 卡 job 上：

```
nvidia-smi        NVIDIA H20-3e, 143771 MiB, driver 570.133.20
libEGL.so.1       /usr/lib/x86_64-linux-gnu/libEGL.so.1          ✓
libEGL_nvidia     libEGL_nvidia.so.0 / .570.133.20               ✓
egl_vendor.d      MISSING                                        ✗ ← 病根
/dev/dri          MISSING（EGL device platform 不需要它）
libglib/libgthread MISSING（评测环境的 conda prefix 提供）
```

没有 ICD 清单时，libEGL 枚举到 **0 个设备**，报错：

```
RuntimeError: The MUJOCO_EGL_DEVICE_ID environment variable must be an integer
between 0 and -1 (inclusive), got 0.
```

补上清单后**实测渲染成功**（`shape=(128,128,3) mean=135.3 std=75.9`）：

```bash
printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}\n' > 10_nvidia.json
export __EGL_VENDOR_LIBRARY_FILENAMES=$PWD/10_nvidia.json MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
```

`run_eval.sh` 会自动生成这个清单。程序退出时 `EGLError: <exception str() failed>` 是
mujoco 析构器的已知噪音，不影响结果。

### 3.3 网络与存储

| 项 | 结论 |
|---|---|
| `github.com` / `huggingface.co` | http=200，**直连可达** → LIBERO-plus 与 assets 在北京直接下，不必跨区搬 |
| `storage.googleapis.com` | http=400（无 bucket 的正常应答，不是超时）→ `gs://openpi-assets` 大概率可直取 |
| `/mnt/cpfs/PeterX` | 已存在（policy/openpi-icl、env、data、tools/mamba 都在），10T 盘剩 1.5T |
| `/mnt/oss/PeterX` | 可写，512T |
| `apt-get install libmagickwand-dev` | 容器里可用（装的是 ImageMagick 6.9）。但我们走 conda 的 IM7，两个 region 保持一致 |
| micromamba / conda | 镜像里没有；用 CPFS 上的 `tools/mamba/micromamba` |
| `gs://openpi-assets` | **可直取**（gcsfs `token="anon"`）。`pi05_libero` = 16 个文件 / 12.44GB。**带宽是总量受限，不是单连接受限**：单流 3.5 MB/s，32 路并发 range 请求也只有约 3.8 MB/s 聚合——12GB 就是要花约 1 小时。用 `python/fetch_gcs_checkpoint.py` 提前预热，别让它卡在评测 job 头上。（Peter 自己训练的 ckpt 走跨区 rclone，不受此影响） |

### 3.4 `/mnt/oss` 是 ossfs2，不能当工作目录

实测：`ln -s` 直接报 `Operation not supported`。append 写与 rename 同样不可依赖，
而 `episodes.jsonl` 正是以 append 模式打开、且是成绩的唯一事实来源。

**所以 `run_eval.sh` 把全部实时写入放在北京 CPFS 的 `WORK_ROOT`
（默认 `/mnt/cpfs/PeterX/train/libero-plus-eval/<...>`），跑完再整份 `cp` 发布到
OSS 输出目录。** 发布只用整文件复制，绝不 link/move 进 ossfs。
断点续跑读的是 CPFS 上的工作目录，所以跨 job 重启也有效。

### 3.5 uv 的托管 Python 必须落在 CPFS

openpi 的 `.python-version` 要 **3.11**，而 DLC 镜像自带 3.12，所以 `uv sync` 会下载一个
托管解释器。默认装到 `$HOME/.local/share/uv/python` —— **DLC job 里的 `$HOME` 是容器本地的，
job 一结束就没了**，留下 `.venv/bin/python` 指向不存在的路径。下一个 job 看到的现象是：
`.venv/` 目录在、里面文件齐全，但 `[ -e .venv/bin/python ]` 为假（悬空链接）。

```bash
export UV_PYTHON_INSTALL_DIR=/mnt/cpfs/PeterX/tools/uv_pythons   # 必须
```

**检查 venv 是否可用要执行它，不能只看目录在不在**：`.venv/bin/python -c ''`。

### 3.6 DLC 提交的两个硬限制

* **UserCommand 上限 65,536 字节**（超了报 `The job parameters length(69665) exceeds limit(65536)`）。
  北京 CPFS 上还没有本仓库时，可以把 tar.gz base64 内联进命令送过去（整个仓库约 58KB base64，刚好够）；
  再大就得先落一次盘，之后只补送单个文件。
* **0 卡 job 拿不到 NVIDIA 运行时**，见 §3.1。

### 3.7 openpi 的 checkpoint 缓存布局

`openpi.shared.download.maybe_download` 把 `gs://<netloc>/<path>` 缓存到
`$OPENPI_DATA_HOME/<netloc>/<path>`，即
`gs://openpi-assets/checkpoints/pi05_libero` → `/mnt/cpfs/PeterX/data/openpi_data/openpi-assets/checkpoints/pi05_libero`。
预先按这个布局放好文件，`serve_policy --policy.dir gs://...` 会直接命中缓存、不再下载。

---

## 四、PPU 侧为什么不能评测（实测）

```
/usr/lib/x86_64-linux-gnu/libEGL*        → 不存在
/usr/share/glvnd/egl_vendor.d/           → 不存在
/dev/dri                                 → 不存在
libOSMesa.so.8                           → 有（纯软件渲染）
repos/openpi-icl/.venv 的 jax.devices()  → [CpuDevice(id=0)]
```

渲染和推理两头都不行。本机只用来跑 `smoke_env.py`（OSMesa，不接策略），验证装对没有。

---

## 五、成绩口径

* **overall 用 micro**（总成功数 / 总 episode 数）。七维任务数天然不等
  （Noise 1601 / Camera 1599 / Robot 1550 / Language 1537 / Layout 1525 / Light 1142 / Background 1076），
  micro 就等价于「按维度任务数加权」，对齐官方 leaderboard 的 Total 列。macro 也一并输出但不作主指标。
* `aggregate.py` 在 episode 缺失、重复或超出 split 时**直接报错退出**。要诊断数字必须显式加
  `--allow-incomplete`，那种数字**不得进台账**。
* dev split 的统计噪声（95% Wilson 半宽，实测）：单维 220 条时 **±3.3 ~ ±6.6pt**（成功率越接近 50% 越宽），
  overall 1,540 条时 **±2.2pt**。screening 的 keep/kill 阈值是 +1.5pt——
  **单维 dev 数字不足以判生死，只能看 overall，且要标注置信区间。**

---

## 六、已验证的事实（可以直接引用）

* LIBERO-plus @ `4976dc3`：spatial 2402 + object 2518 + goal 2591 + libero_10 2519 = **10,030**。
* `benchmark_dict[suite]()` 的 0-based `task_id` **精确对应** `task_classification.json[suite][task_id]`
  （`task_order_index=0` 是恒等序，10,030 条 name 零错位）。七维统计靠这个映射。
  `libero_plus_common.check_alignment()` 每次运行都会重新验证，不依赖这份记录。
* assets.zip 解压后 **448,799 个文件 / 8.3GiB**，与 zip 内条目数完全一致。
* **全量 10,030 个环境构造检查：10,030/10,030 通过，0 失败**（`validate_envs.py`，本机 OSMesa）。
  七维计数与 `task_classification.json` 逐项吻合，详见 `docs/env_validation_20260806.md`。
  附带发现：**1,525 个任务只有 1 个 init state**（恰好是 Objects Layout 全部，`_add_` 分支
  会 `reshape(1,-1)`），其余 8,505 个各有 50 个。所以 **LIBERO-plus 档位不要动
  `num_trials_per_task`**——调大只会让 8,505 个任务多跑，各维权重被悄悄改掉。
* **PPU 本机 OSMesa 冒烟：14/14 通过**（七维各 2 个 task，`smoke_env.py`）。
  Camera Viewpoints 出图确为偏移视角，Sensor Noise 出图确有 wand 施加的模糊/退化，
  说明 `_view_/_initstate_/_noise_/_language_/_table_/_light_/_add_` 全部分支与
  `MountedPandaN` 变体机器人都工作正常。import 一次约 13 秒（模块层解析 1,537 个 language bddl）。
