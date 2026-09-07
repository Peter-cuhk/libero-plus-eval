# RUNBOOK — LIBERO-plus 评测踩坑清单

跑评测之前先读这一页。每条都是实测结论，不是猜测。

> **2026-09 算力变更**：H20 / 北京已关停，评测迁到 **Pro5000 / 乌兰察布**。
> 现行事实看 [§三点五](#三点五pro5000--乌兰察布侧的实测事实现行)；§三 是 H20 的历史记录，
> 保留是因为病根（EGL ICD 缺失、ossfs2 不能当工作目录、uv 托管 python 要落 CPFS）
> 在新集群上一模一样。

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

## 三、北京 H20 侧的实测事实（历史；H20 已于 2026-09 关停）

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
* **UserCommand 是 `/bin/sh`（dash）执行的，不是 bash。** 命令开头写 `set -euo pipefail`
  会直接 `Illegal option -o pipefail` 并让 job 秒失败。要么用 POSIX 的 `set -eu`，
  要么把脚本投递到文件里再用 `bash` 跑（本仓库两种都用到了）。

### 3.7 openpi 的 checkpoint 缓存布局

`openpi.shared.download.maybe_download` 把 `gs://<netloc>/<path>` 缓存到
`$OPENPI_DATA_HOME/<netloc>/<path>`，即
`gs://openpi-assets/checkpoints/pi05_libero` → `/mnt/cpfs/PeterX/data/openpi_data/openpi-assets/checkpoints/pi05_libero`。
预先按这个布局放好文件，`serve_policy --policy.dir gs://...` 会直接命中缓存、不再下载。

---

## 三点五、Pro5000 / 乌兰察布侧的实测事实（现行）

### 3.5.1 集群坐标

| 项 | 值 |
|---|---|
| 入口 | `skills/pai-toolkit` → **`pro5000.py`**（`configs/5kpro.yaml`） |
| region | `cn-wulanchabu`（**和 PPU 同一个**） |
| workspace | `293248`（`ai_platform_ws_5kpro`） |
| 配额 | `quotam3rsay5sm3t`（`ai_pai_quota_5kpro`），**24 GPU / 765c / 4410Gi**，ECS 型 |
| 镜像 | `registry.cn-wulanchabu.aliyuncs.com/pai-dlc/pytorch-training:2.4.0-gpu-py3.10-cu12.5-ngc24.06-ubuntu22.04` |
| 数据源 | `d-r8j0kpemv8hxx00ucr`→`/mnt/cpfs/`、`d-6f7a61uo2kuq1xbdh7`→`/mnt/oss/` |

`cn-beijing` 下已经**查不到任何配额**（`pai.py status` 直接 404），H20 是真的没了。

### 3.5.2 和 PPU 共用同一套存储 —— 跨区搬运整条流程作废

上表两个数据源 id 与 `configs/ppu.yaml` **逐字相同**：训练 job 和评测 job 看到的
`/mnt/cpfs/PeterX`、`/mnt/oss/PeterX` 是同一份。

* ckpt 训完**直接开评**，不再需要 `coco-transfer-team-data` / rclone 跨区拷贝与校验。
* `--expect-ckpt-bytes` 从"跨区完整性校验"降级为普通的写盘完整性自检，可选。
* 官方 `pi05_libero` 已经落在 CPFS：`/mnt/cpfs/PeterX/data/openpi_data/openpi-assets/checkpoints/pi05_libero`
  （12G，`params/_METADATA` 齐全）。**不用再从 `gs://` 拉那一个小时。**

### 3.5.3 EGL：和 H20 同一个病根，同一个解法

Pro5000 上实测渲染成功的组合（探测 job `dlcytozkacvpipym` = envprobe、
`dlcyjgyru2jp3mod` = render-preflight2，都 Succeeded）：

```bash
printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}\n' > 10_nvidia.json
export __EGL_VENDOR_LIBRARY_FILENAMES=$PWD/10_nvidia.json
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export EGL_PLATFORM=surfaceless        # ← H20 时代没有这一行
```

envprobe 日志末尾是 `DEBUG_MAKE_ENV_OK KITCHEN_SCENE4_..._noise_10 50 13.38`，
说明 **LIBERO-plus 的环境在 Pro5000 上能构造并渲染**，用的就是现有的
`/mnt/cpfs/PeterX/env/libero-plus-eval-py38`（共享 CPFS，不用重建）。

`run_eval.sh` 已经把这一套接进去：ICD 清单照旧自动生成，多出来的
`EGL_PLATFORM` 由 `EGL_PLATFORM_MODE`（默认 `surfaceless`）控制，置空即可关掉。
退出时的 `EGLError: <exception str() failed>` 仍然是 mujoco 析构器噪音，不影响结果。

### 3.5.4 卡本身

1 卡 job（`dlc1bqy8lm851dj4`）实测：

```
NVIDIA RTX PRO 5000 72GB      73415 MiB
driver 580.126.09             CUDA 13.0
```

对照 H20-3e 的 143771 MiB —— **显存只有一半**，切分片和调 `NUM_WORKERS` 时要记得。

### 3.5.5 建 venv 必须走国内镜像，**别用 `uv sync --frozen`**

这个坑很贵：直接 `uv sync --frozen` 会**卡到天荒地老**（实测提了个 DLC job
`dlc1bqy8lm851dj4`，跑满 24 小时都没下完 torch/cudnn，最后手动停掉）。

病根：

* `openpi-ar/uv.lock` 把每个 wheel 的下载地址钉死成 **`https://files.pythonhosted.org/...`**。
  `--frozen` 照着 lock 里的 URL 取包，`env.sh` 设的 `UV_DEFAULT_INDEX`（阿里云镜像）
  **完全被绕过**。
* 从乌兰察布到 `files.pythonhosted.org` 慢到没法用（实测 DSW 24 KB/s、DLC 节点 350 KB/s），
  2.5G 的 torch+CUDA 依赖根本下不完。

解法：**导出钉死版本再从国内镜像装**。镜像上的 wheel 路径带的是同一个 sha256
（例如 torch 那个 `e5/94/34b8…`），是**字节相同**的包，换源不改版本、不改 hash，
`uv pip install` 会做 hash 校验，**完全可复现**。实测同区镜像下 torch wheel：

| 源 | 速度 |
|---|---|
| pythonhosted（lock 默认） | 0.35 MB/s |
| 阿里云 `mirrors.aliyun.com/pypi` | ~20 MB/s |
| 清华 `pypi.tuna.tsinghua.edu.cn` | ~140 MB/s |

整套建 venv 从 24h+ 降到约 10 分钟。配方（在**哪台机器上跑都行**——venv 落在共享
CPFS，Pro5000 评测 job 直接用；只有最后验 `jax.devices()` 必须在 Pro5000 上，
因为 PPU DSW 的卡 jax 认不出来）：

```bash
cd /mnt/cpfs/PeterX/policy/openpi-ar
export UV_CACHE_DIR=/mnt/cpfs/uv_cache UV_PYTHON_INSTALL_DIR=/mnt/cpfs/PeterX/tools/uv_pythons UV_LINK_MODE=copy
rm -rf .venv && uv venv --python 3.11 .venv
# 导出钉死版本 + hash（排除本地项目），再从清华镜像装
uv export --frozen --no-emit-project --no-editable -o /tmp/openpi-ar-reqs.txt
uv pip install --python .venv/bin/python -r /tmp/openpi-ar-reqs.txt \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple/ \
    --extra-index-url https://mirrors.aliyun.com/pypi/simple/
# 本地项目单独装（无下载）
uv pip install --python .venv/bin/python --no-deps -e . -e packages/openpi-client
```

配套细节：

* `uv` 在 pai-dlc 镜像里不一定有，CPFS 上放了一份 `/mnt/cpfs/PeterX/tools/uv`，
  DLC job 里 `export PATH=/mnt/cpfs/PeterX/tools:$PATH` 即可。
* `UV_PYTHON_INSTALL_DIR` 一定要指到 CPFS，否则 uv 下的托管解释器落在容器本地，
  job 一结束 `.venv/bin/python` 就变成断链（见 §3.5）。
* `AI 工具代理`（`/mnt/cpfs/tools/ai-proxy`）**帮不上这里**：实测它到 pythonhosted
  只有 14 KB/s（比直连还慢）、且不转发 github，它只服务 codex/claude 那些 AI 端点。

### 3.5.6 policy server 的 JAX 在 Blackwell 上：**已验证能跑**（jax 0.5.3 + cu126）

2026-09-08，1 卡 Pro5000 probe job `dlcxsz8juhxuwgrb`，跑现建的 `openpi-ar/.venv`
（jax/jaxlib 0.5.3，torch 2.7.1+cu126）：

```
NVIDIA RTX PRO 5000 72GB Blackwell, 73415 MiB
default_backend: gpu
devices: [CudaDevice(id=0)]
MATMUL_OK 8589934592.0 on {CudaDevice(id=0)}          # 2048^3，结果正确
```

**结论：openpi-ar 锁的 stock jax 直接吃到 Blackwell，不需要 ttt-vla 那套 cu128 overlay。**
（openpi-ar 恰好也锁 `jax-cuda12-plugin==0.5.3`，与 ttt-vla 验证过的同版本。）

一条**良性告警**，别当错误：

```
W ... ptxas does not support CC 12.0
W ... ptxas too old. Falling back to the driver to compile.
```

bundled ptxas 是 CUDA 12.6 的，不认 sm_120（Blackwell = CC 12.0），XLA 自动回退到
**驱动**（580.126.09 / CUDA 13.0，支持 sm_120）做 PTX 编译——结果照跑，只是首次编译
稍慢。要消掉它得升到 cu128 的 jaxlib，没必要。

* venv 用国内镜像现建（§3.5.5），`.venv/bin/python` 已指向 CPFS 的托管解释器
  `/mnt/cpfs/PeterX/tools/uv_pythons/...`，不是断链。

**换机器/换环境后仍然：先 smoke（`splits/smoke_v1.json`、1 卡）跑通策略链路，
再跑净版 LIBERO 2,000ep 回归对齐 97.10**，对不上就别信任何 Pro5000 上的新数字。

### 3.5.7 评测客户端 torch 冲突：DLC 镜像的 py3.10 torch 抢了 py38 的

2026-09-08 首个 Pro5000 smoke（`dlcj3m5u2bg69mya`）**Failed** 在这上面：

```
py38 评测环境 import torch
→ ImportError: /usr/local/lib/python3.10/dist-packages/torch/lib/libtorch_python.so:
  undefined symbol: PyObject_GET_WEAKREFS_LISTPTR
```

`pai-dlc/pytorch-training:2.4.0-...-py3.10-...` 镜像自带一个 **python3.10** 的 torch，
并把它的 `torch/lib` 放进了容器默认 `LD_LIBRARY_PATH`。评测 env 是 **py38**（torch 1.11），
它的 `torch/_C` 用 DT_RUNPATH 找 `libtorch_python.so`——RUNPATH 排在 LD_LIBRARY_PATH
之后，于是 3.10 的那个被抢先加载，ABI 对不上就炸。H20 的 mmld 镜像没这问题。

修法（`run_eval.sh` 已做）：把评测 env 自己的 `lib/python*/site-packages/torch/lib`
**前置**到客户端 `LD_LIBRARY_PATH`，让它稳赢。改完 smoke 通过：
`14/14 tasks | success 10 | errors 0`（对齐 H20 smoke 的 9/14，errors 0）。
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

## 四点五、吞吐实测（决定切几片）

2026-08-06，1×H20-3e，8 worker，smoke_v1 14 个任务（`dlchhz8qy2kffxzz`）：

| 项 | 实测 |
|---|---|
| 成功 | 9/14，0 错误 |
| 客户端总时长 | 283.7s（另加约 2.6 min 服务端加载 12.4GB 权重）|
| 单任务 worker 时间 | 平均约 47s |
| 推理占 worker 时间 | **44%**，其余是仿真与环境构造 |
| 推理延迟 | 首次约 1546ms（JIT 预热）→ 稳态 **83–148ms** |

据此（每任务约 46 次推理、稳态约 120ms/次）：

| 档 | 规模 | 分片 | 单片预计 |
|---|---|---|---|
| dev | 1,540 task | 4 | 40–60 min |
| full | 10,030 task | 8 | 2–3 h |
| 净版回归 | 40 task × 50 trial | 4 | 约 45 min |

PLAN 里「full 单卡 8–20h」的估算可以用这组实测数字替换。
注意 CPU 不是瓶颈（16 核的 job 平均只用到约 2 核），**瓶颈是单卡策略服务器的串行推理**——
所以加 worker 收益有限，加分片（=加卡）才有效。

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

## 五点五、已验证的基线（2026-08-06，官方 pi05_libero）

| 协议 | 实测 | 对照 | job |
|---|---|---|---|
| 净版 LIBERO 2,000ep（40task×50trial） | **97.10** ±0.74 | openpi 公布 96.85 | `dlctzml8qgziw5j1` 等 4 片 |
| LIBERO-plus dev 1,540ep | **84.61** ±1.8 | — | `dlczjegwapvyd9ri` 等 4 片 |
| LIBERO-plus **full 10,030ep** | **84.20** ±0.71 | 见下 | `dlcpa64uieqca2bl` 等 8 片，0 错误 |

full 七维：Layout 85.70 / **Camera 71.04** / **Robot init 75.55** / Language 85.04 /
Light 97.11 / Background 96.00 / Noise 86.32。macro 85.25。
**dev 的每一维都落在自身置信区间内命中 full 的值** —— dev_v1 作为筛选工具是可信的。

**公开榜 Total 的口径已核实为 micro**（按维度任务数加权）：用 OpenVLA-OFT+ 的七维反推得
micro 79.56，与 README 公布的 79.6 吻合（macro 会得 80.66）。`aggregate.py` 主报 micro 正确。

净版逐 suite：spatial 99.20 / object 98.40 / goal 97.60 / libero_10 93.20
（公布 98.8 / 98.2 / 98.0 / 92.4）——四项全部落在 1pt 内，**整条管线的预处理已被证明正确**。

dev 七维：Layout 85.00 / **Camera 70.45** / Robot init 75.00 / Language 87.73 /
Light 96.36 / Background 95.91 / Noise 81.82。最弱的相机视角与机器人初始位姿
与 LIBERO-plus 论文的核心结论一致。难度梯度 L1 94.44 → L5 65.11，单调。

**这两个数字是回归基准**：以后动了评测端、换了环境、升了依赖，先重跑净版对 97.10。

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
