#!/usr/bin/env bash
# Rendering-stack probe that MUST run with --gpu >= 1.
#
# A 0-GPU DLC job gets no NVIDIA runtime injected, so it reports libEGL/libGL as
# missing even when a GPU job would have them. Anything about EGL learned from a
# 0-GPU job is meaningless; this script exists to get the real answer.

set -uo pipefail
probe() { printf '  %-44s %s\n' "$1" "$2"; }

printf '\n\033[1m===== GPU 与渲染栈 =====\033[0m\n'
probe "nvidia-smi" "$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | head -2 | tr '\n' ';')"
probe "libEGL.so.1" "$(ls /usr/lib/x86_64-linux-gnu/libEGL.so.1 2>/dev/null || echo MISSING)"
probe "libEGL_nvidia" "$(ls /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so* 2>/dev/null | tr '\n' ' ' || echo MISSING)"
probe "libGLX_nvidia" "$(ls /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so* 2>/dev/null | tr '\n' ' ' || echo MISSING)"
probe "egl_vendor.d" "$(ls /usr/share/glvnd/egl_vendor.d/ 2>/dev/null | tr '\n' ' ' || echo MISSING)"
probe "/dev/dri" "$(ls /dev/dri 2>/dev/null | tr '\n' ' ' || echo MISSING)"
for lib in libGL.so.1 libGLdispatch.so.0 libOpenGL.so.0 libgthread-2.0.so.0 libglib-2.0.so.0; do
    probe "  $lib" "$(ldconfig -p 2>/dev/null | grep -m1 "$lib" | sed 's/^\s*//' || echo MISSING)"
done

printf '\n\033[1m===== 端到端 EGL 渲染实测 =====\033[0m\n'
# Build a throwaway env in /tmp and actually render a MuJoCo frame off-screen.
# This is the only probe that proves EGL works rather than merely looks present.
export UV_CACHE_DIR=/mnt/cpfs/uv_cache
if uv venv --python 3.11 /tmp/eglcheck >/dev/null 2>&1 && \
   uv pip install --python /tmp/eglcheck/bin/python \
        --default-index https://mirrors.aliyun.com/pypi/simple/ \
        "mujoco==3.2.3" numpy >/dev/null 2>&1; then
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 \
    /tmp/eglcheck/bin/python - <<'PY'
import numpy as np, mujoco
xml = """
<mujoco><worldbody>
  <light pos="0 0 3"/>
  <geom type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>
  <body pos="0 0 0.5"><geom type="box" size="0.2 0.2 0.2" rgba="1 0 0 1"/></body>
</worldbody></mujoco>
"""
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=128, width=128)
mujoco.mj_forward(model, data)
renderer.update_scene(data)
frame = renderer.render()
print(f"  EGL render OK  shape={frame.shape} mean={frame.mean():.1f} std={frame.std():.1f}")
print("  VERDICT:", "RENDER WORKS" if frame.std() > 1.0 else "BLACK FRAME - EGL broken")
PY
    echo "  exit=$?"
else
    echo "  could not build the probe venv (uv/network problem), EGL untested"
fi

printf '\n\033[1m===== 探测结束 =====\033[0m\n'
