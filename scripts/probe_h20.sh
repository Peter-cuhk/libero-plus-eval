#!/usr/bin/env bash
# Read-only reconnaissance of the Beijing/H20 side. Runs as a 0-GPU DLC job.
#
# The two CPFS filesystems (Wulanchabu vs Beijing) are separate, so nothing about
# the Beijing side can be observed from the PPU dev box. Everything this script
# prints is a fact we would otherwise have to guess at while bootstrapping.
#
# Never writes outside /tmp and its own probe file, which it removes.

set -uo pipefail   # deliberately NOT -e: a failing probe is a result, not an abort

section() { printf '\n\033[1m===== %s =====\033[0m\n' "$*"; }
probe()   { printf '  %-46s %s\n' "$1" "$2"; }

section "0. 身份与基本环境"
probe "hostname" "$(hostname)"
probe "whoami" "$(whoami)"
probe "pwd" "$(pwd)"
probe "nproc" "$(nproc)"
probe "python3" "$(command -v python3 || echo MISSING) $(python3 --version 2>&1)"
probe "uv" "$(command -v uv || echo MISSING) $(uv --version 2>/dev/null)"
probe "UV_CACHE_DIR" "${UV_CACHE_DIR:-<unset>}"
probe "HF_HOME" "${HF_HOME:-<unset>}"
probe "http_proxy" "${http_proxy:+<set>}${http_proxy:-<unset>}"

section "1. 北京 CPFS: /mnt/cpfs/PeterX 有什么"
if [[ -d /mnt/cpfs/PeterX ]]; then
    probe "/mnt/cpfs/PeterX" "EXISTS"
    ls -la /mnt/cpfs/PeterX 2>&1 | head -40
    for d in policy repos env data train skills autoresearch tools; do
        if [[ -d "/mnt/cpfs/PeterX/$d" ]]; then
            probe "  $d/" "$(ls "/mnt/cpfs/PeterX/$d" 2>/dev/null | tr '\n' ' ' | cut -c1-160)"
        else
            probe "  $d/" "MISSING"
        fi
    done
    probe "LIBERO-plus" "$([[ -d /mnt/cpfs/PeterX/repos/LIBERO-plus ]] && echo EXISTS || echo MISSING)"
    probe "openpi-ar" "$([[ -d /mnt/cpfs/PeterX/policy/openpi-ar ]] && echo EXISTS || echo MISSING)"
    probe "libero-plus-eval" "$([[ -d /mnt/cpfs/PeterX/repos/libero-plus-eval ]] && echo EXISTS || echo MISSING)"
    probe "eval py38 venv" "$([[ -x /mnt/cpfs/PeterX/env/libero-plus-eval-py38/bin/python ]] && echo EXISTS || echo MISSING)"
    probe "df /mnt/cpfs" "$(df -h /mnt/cpfs 2>/dev/null | tail -1)"
else
    probe "/mnt/cpfs/PeterX" "MISSING  <-- 北京侧要从零建"
    ls -la /mnt/cpfs 2>&1 | head -20
fi
probe "/mnt/cpfs/tools/ai-proxy" "$([[ -f /mnt/cpfs/tools/ai-proxy/bootstrap.sh ]] && echo EXISTS || echo MISSING)"
probe "/mnt/cpfs/uv_cache" "$([[ -d /mnt/cpfs/uv_cache ]] && echo EXISTS || echo MISSING)"

section "2. 北京 OSS 可写性"
probe "/mnt/oss/PeterX" "$([[ -d /mnt/oss/PeterX ]] && echo EXISTS || echo MISSING)"
ls /mnt/oss/PeterX 2>&1 | head -10
_probe_file=/mnt/oss/PeterX/.libero_plus_probe.$$
if mkdir -p /mnt/oss/PeterX 2>/dev/null && echo probe > "$_probe_file" 2>/dev/null; then
    probe "write test" "OK"
    rm -f "$_probe_file"
else
    probe "write test" "FAILED"
fi
probe "df /mnt/oss" "$(df -h /mnt/oss 2>/dev/null | tail -1)"

section "3. 渲染栈 (LIBERO 需要 EGL)"
probe "nvidia-smi" "$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -2 | tr '\n' ';')"
probe "libEGL.so.1" "$(ls /usr/lib/x86_64-linux-gnu/libEGL.so.1 2>/dev/null || echo MISSING)"
probe "libEGL_nvidia" "$(ls /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so* 2>/dev/null | tr '\n' ' ' || echo MISSING)"
probe "egl_vendor.d" "$(ls /usr/share/glvnd/egl_vendor.d/ 2>/dev/null | tr '\n' ' ' || echo MISSING)"
probe "/dev/dri" "$(ls /dev/dri 2>/dev/null | tr '\n' ' ' || echo MISSING)"
probe "libOSMesa" "$(ls /usr/lib/x86_64-linux-gnu/libOSMesa.so* 2>/dev/null | head -1 || echo MISSING)"
for lib in libgthread-2.0.so.0 libglib-2.0.so.0 libGLdispatch.so.0 libGL.so.1; do
    probe "  $lib" "$(ldconfig -p 2>/dev/null | grep -m1 "$lib" | sed 's/^\s*//' || echo MISSING)"
done

section "4. 网络可达性"
for url in https://mirrors.aliyun.com/pypi/simple/ https://github.com https://huggingface.co https://storage.googleapis.com; do
    code=$(timeout 15 curl -s -o /dev/null -w '%{http_code}' -L "$url" 2>/dev/null)
    probe "$url" "http=${code:-TIMEOUT}"
done
probe "gsutil/gcloud" "$(command -v gsutil || command -v gcloud || echo MISSING)"

section "5. apt 能否装 ImageMagick (wand 是 LIBERO-plus 的硬依赖)"
probe "MagickWand.h" "$(ls /usr/include/ImageMagick*/wand/MagickWand.h 2>/dev/null | head -1 || echo MISSING)"
probe "libMagickWand.so" "$(ldconfig -p 2>/dev/null | grep -m1 libMagickWand | sed 's/^\s*//' || echo MISSING)"
if timeout 300 apt-get update -qq 2>&1 | tail -2 && \
   timeout 600 apt-get install -y --no-install-recommends libmagickwand-dev 2>&1 | tail -4; then
    probe "apt install libmagickwand-dev" "OK"
    probe "  libMagickWand.so now" "$(ldconfig -p 2>/dev/null | grep -m1 libMagickWand | sed 's/^\s*//' || echo STILL-MISSING)"
else
    probe "apt install libmagickwand-dev" "FAILED  <-- 需要走 CPFS 携带 .so 的方案"
fi

section "6. micromamba / conda 是否可用（apt 的备选）"
probe "micromamba" "$(command -v micromamba || echo MISSING)"
probe "conda" "$(command -v conda || echo MISSING)"

printf '\n\033[1m===== 探测结束 =====\033[0m\n'
