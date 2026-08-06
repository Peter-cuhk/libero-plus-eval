#!/usr/bin/env bash
# One-time Beijing/H20 setup: LIBERO-plus + assets, an openpi serving repo, and
# the python 3.8 evaluation environment.
#
# Self-contained on purpose: runs as a DLC job command with nothing but the
# Beijing CPFS available. Every step is idempotent, so a failed run can simply
# be resubmitted.
#
# Verified by the probe jobs (dlcrzmz9mv5h7ats / dlcoe0op6ptpeu6g):
#   - github.com and huggingface.co are directly reachable from Beijing DLC,
#     so nothing has to be copied across regions.
#   - the DLC images have libEGL_nvidia but no /usr/share/glvnd/egl_vendor.d,
#     which run_eval.sh works around with its own ICD manifest.
#   - micromamba is not in the image; we use the copy on CPFS.

set -euo pipefail

PETERX=/mnt/cpfs/PeterX
LIBERO_PLUS_COMMIT=4976dc30028e805ff8094b55501d532c48fec182
OPENPI_COMMIT=${OPENPI_COMMIT:-15a9616}
LIBERO_PLUS_ROOT=$PETERX/repos/LIBERO-plus
OPENPI_AR=$PETERX/policy/openpi-ar
EVAL_ENV=$PETERX/env/libero-plus-eval-py38
REPO=$PETERX/repos/libero-plus-eval
MICROMAMBA=$PETERX/tools/mamba/micromamba
ASSETS_URL=https://huggingface.co/datasets/Sylvest/LIBERO-plus/resolve/main/assets.zip
ASSETS_BYTES=6395849578
ASSETS_PREFIX=inspire/hdd/project/embodied-multimodality/public/syfei/libero_new/release/dataset/LIBERO-plus-0/assets

step() { printf '\n\033[1m>>> %s\033[0m\n' "$*"; }

step "0. 前置检查"
region=$(df /mnt/cpfs 2>/dev/null | tail -1)
echo "  /mnt/cpfs -> $region"
case "$region" in
    *cn-beijing*) echo "  region: 北京 ✓" ;;
    *) echo "  拒绝执行：这不是北京的 CPFS。评测基建只装在 H20 侧。" >&2; exit 1 ;;
esac
test -d "$PETERX" || { echo "  $PETERX 不存在" >&2; exit 1; }
export UV_CACHE_DIR=${UV_CACHE_DIR:-/mnt/cpfs/uv_cache}
export MAMBA_ROOT_PREFIX=$PETERX/tools/mamba

step "1. LIBERO-plus @ ${LIBERO_PLUS_COMMIT:0:7}"
if [[ ! -d "$LIBERO_PLUS_ROOT/.git" ]]; then
    git clone https://github.com/sylvestf/LIBERO-plus.git "$LIBERO_PLUS_ROOT"
fi
git -C "$LIBERO_PLUS_ROOT" fetch --quiet origin
git -C "$LIBERO_PLUS_ROOT" checkout --quiet "$LIBERO_PLUS_COMMIT"
echo "  HEAD = $(git -C "$LIBERO_PLUS_ROOT" rev-parse --short HEAD)"

step "2. assets (约 6.0GB 下载, 8.3GB 解压后 448,799 个文件)"
assets_link=$LIBERO_PLUS_ROOT/libero/libero/assets
if [[ -e "$assets_link" && -d "$assets_link/textures" ]]; then
    echo "  已就位，跳过 ($(find "$assets_link/" -maxdepth 1 -type d | wc -l) 个子目录)"
else
    zip=$PETERX/data/libero_plus/assets.zip
    mkdir -p "$(dirname "$zip")"
    have=$( [[ -f "$zip" ]] && stat -c %s "$zip" || echo 0 )
    if [[ "$have" != "$ASSETS_BYTES" ]]; then
        # huggingface.co only issues a 302 to a CloudFront/Xet host; a plain
        # `curl -L` that cannot reach that host silently writes the 1042-byte
        # redirect body instead. Prefer huggingface_hub, which speaks Xet, and
        # always verify the byte count before trusting the file.
        echo "  下载 assets.zip (本地 $have B, 期望 $ASSETS_BYTES B)"
        rm -f "$zip"
        if command -v uv >/dev/null && uv venv --python 3.11 /tmp/hfdl >/dev/null 2>&1 && \
           uv pip install --python /tmp/hfdl/bin/python \
                --default-index https://mirrors.aliyun.com/pypi/simple/ \
                "huggingface_hub[hf_xet]" >/dev/null 2>&1; then
            /tmp/hfdl/bin/python - "$zip" <<'PY' || true
import shutil, sys
from huggingface_hub import hf_hub_download
path = hf_hub_download("Sylvest/LIBERO-plus", "assets.zip", repo_type="dataset")
shutil.copyfile(path, sys.argv[1])
print("  huggingface_hub 下载完成")
PY
        fi
        if [[ ! -f "$zip" || "$(stat -c %s "$zip")" != "$ASSETS_BYTES" ]]; then
            echo "  huggingface_hub 未拿到完整文件，退回 curl"
            curl -fL --retry 5 --retry-delay 10 -o "$zip" "$ASSETS_URL"
        fi
    fi
    got=$(stat -c %s "$zip")
    if [[ "$got" != "$ASSETS_BYTES" ]]; then
        echo "  assets.zip 大小不对: 拿到 $got, 期望 $ASSETS_BYTES" >&2
        echo "  （$got 约等于 1042 说明只收到了 HF 的 302 重定向体，CDN 主机不可达）" >&2
        exit 1
    fi
    echo "  assets.zip 校验通过 ($got B)"
    echo "  解压 -> $LIBERO_PLUS_ROOT/libero/libero/"
    unzip -q -o "$zip" -d "$LIBERO_PLUS_ROOT/libero/libero/"
    ln -sfn "$LIBERO_PLUS_ROOT/libero/libero/$ASSETS_PREFIX" "$assets_link"
fi
n_assets=$(find -L "$assets_link/" -type f | wc -l)
echo "  assets 文件数 = $n_assets (PPU 侧实测 448799)"
[[ "$n_assets" -ge 448000 ]] || { echo "  assets 不完整" >&2; exit 1; }

step "3. openpi 服务端仓库 @ $OPENPI_COMMIT"
if [[ ! -d "$OPENPI_AR/.git" ]]; then
    git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git "$OPENPI_AR"
fi
git -C "$OPENPI_AR" fetch --quiet origin
git -C "$OPENPI_AR" checkout --quiet "$OPENPI_COMMIT"
git -C "$OPENPI_AR" submodule update --init --recursive --quiet
echo "  HEAD = $(git -C "$OPENPI_AR" rev-parse --short HEAD)"
grep -q 'name="pi05_libero"' "$OPENPI_AR/src/openpi/training/config.py" || {
    echo "  这个 openpi 里没有 pi05_libero 配置" >&2; exit 1; }
echo "  pi05_libero 配置存在 ✓"

step "4. openpi 服务端 venv (uv sync, 首次约 15-30 分钟)"
( cd "$OPENPI_AR" && UV_CACHE_DIR="$UV_CACHE_DIR" uv sync --frozen )
echo "  $("$OPENPI_AR/.venv/bin/python" -c 'import jax; print("jax", jax.__version__, jax.devices())' 2>&1 | tail -1)"

step "5. 评测环境 py3.8 + ImageMagick"
if [[ ! -x "$EVAL_ENV/bin/python" ]]; then
    test -x "$MICROMAMBA" || chmod +x "$MICROMAMBA"
    "$MICROMAMBA" create -y -p "$EVAL_ENV" -c conda-forge python=3.8 imagemagick pkg-config
fi
# The public mirror is required: the cluster's default index carries no cp38 torch.
uv pip install --python "$EVAL_ENV/bin/python" -r "$REPO/requirements/client-py38.in" \
    --default-index https://mirrors.aliyun.com/pypi/simple/
uv pip install --python "$EVAL_ENV/bin/python" -e "$OPENPI_AR/packages/openpi-client"
for lib in libMagickWand-7.Q16HDRI.so libglib-2.0.so.0 libgthread-2.0.so.0; do
    test -e "$EVAL_ENV/lib/$lib" || { echo "  评测环境缺 $lib" >&2; exit 1; }
done
echo "  ImageMagick + glib 就位 ✓"

step "6. 自检：LIBERO-plus 能否 import 且任务数正确"
cfg=/tmp/libero-config-bootstrap
mkdir -p "$cfg/datasets"
printf '%s\n' \
    "benchmark_root: $LIBERO_PLUS_ROOT/libero/libero" \
    "bddl_files: $LIBERO_PLUS_ROOT/libero/libero/bddl_files" \
    "init_states: $LIBERO_PLUS_ROOT/libero/libero/init_files" \
    "datasets: $cfg/datasets" \
    "assets: $LIBERO_PLUS_ROOT/libero/libero/assets" \
    > "$cfg/config.yaml"
MAGICK_HOME="$EVAL_ENV" LD_LIBRARY_PATH="$EVAL_ENV/lib" \
LIBERO_CONFIG_PATH="$cfg" PYTHONPATH="$LIBERO_PLUS_ROOT" \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
"$EVAL_ENV/bin/python" - <<'PY' | tail -8
from libero.libero import benchmark
d = benchmark.get_benchmark_dict()
total = 0
for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
    n = d[suite]().n_tasks
    total += n
    print(f"  {suite}: {n}")
print("  TOTAL =", total)
assert total == 10030, f"expected 10030 tasks, got {total}"
print("  自检通过 ✓")
PY

printf '\n\033[1m===== bootstrap 完成 =====\033[0m\n'
echo "  LIBERO-plus : $LIBERO_PLUS_ROOT ($(git -C "$LIBERO_PLUS_ROOT" rev-parse --short HEAD))"
echo "  openpi      : $OPENPI_AR ($(git -C "$OPENPI_AR" rev-parse --short HEAD))"
echo "  eval python : $EVAL_ENV/bin/python"
