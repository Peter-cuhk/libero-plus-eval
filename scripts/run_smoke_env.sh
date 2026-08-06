#!/usr/bin/env bash
# Benchmark-side smoke with no policy server: sets up the LIBERO config and the
# EGL vendor manifest, then runs python/smoke_env.py.
#
# Usage:  run_smoke_env.sh [output-dir]
#
# Set MUJOCO_GL_BACKEND=osmesa to run it on a machine without a GPU (the PPU dev
# box); the default `egl` is what the H20 evaluation nodes use.

set -euo pipefail

out_dir="${1:-/tmp/libero-plus-smoke}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${LIBERO_PLUS_ROOT:=/mnt/cpfs/PeterX/repos/LIBERO-plus}"
: "${EVAL_PYTHON:=/mnt/cpfs/PeterX/env/libero-plus-eval-py38/bin/python}"
: "${MUJOCO_GL_BACKEND:=egl}"
: "${EGL_VENDOR_LIBRARY:=libEGL_nvidia.so.0}"
: "${EGL_LOADER_LIBRARY:=/usr/lib/x86_64-linux-gnu/libEGL.so.1}"
: "${TASKS_PER_CATEGORY:=2}"
: "${STEPS:=20}"

[[ -d "$LIBERO_PLUS_ROOT" ]] || { echo "Missing LIBERO-plus: $LIBERO_PLUS_ROOT" >&2; exit 1; }
[[ -x "$EVAL_PYTHON" ]] || { echo "Missing eval python: $EVAL_PYTHON" >&2; exit 1; }
eval_prefix="$(dirname "$(dirname "$EVAL_PYTHON")")"

mkdir -p "$out_dir/frames" "$out_dir/libero-config/datasets" "$out_dir/runtime-libs"
printf '%s\n' \
    "benchmark_root: $LIBERO_PLUS_ROOT/libero/libero" \
    "bddl_files: $LIBERO_PLUS_ROOT/libero/libero/bddl_files" \
    "init_states: $LIBERO_PLUS_ROOT/libero/libero/init_files" \
    "datasets: $out_dir/libero-config/datasets" \
    "assets: $LIBERO_PLUS_ROOT/libero/libero/assets" \
    > "$out_dir/libero-config/config.yaml"

egl_manifest=""
if [[ "$MUJOCO_GL_BACKEND" == "egl" ]]; then
    [[ -e "$EGL_LOADER_LIBRARY" ]] || { echo "Missing EGL loader: $EGL_LOADER_LIBRARY" >&2; exit 1; }
    ln -sfn "$EGL_LOADER_LIBRARY" "$out_dir/runtime-libs/libEGL.so"
    egl_manifest="$out_dir/runtime-libs/10_nvidia.json"
    printf '{"file_format_version":"1.0.0","ICD":{"library_path":"%s"}}\n' "$EGL_VENDOR_LIBRARY" > "$egl_manifest"
fi

echo "renderer=$MUJOCO_GL_BACKEND  benchmark=$LIBERO_PLUS_ROOT  out=$out_dir"
env \
    LIBERO_PLUS_ROOT="$LIBERO_PLUS_ROOT" \
    LIBERO_CONFIG_PATH="$out_dir/libero-config" \
    PYTHONPATH="$LIBERO_PLUS_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    MAGICK_HOME="$eval_prefix" \
    LD_LIBRARY_PATH="$out_dir/runtime-libs:$eval_prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    ${egl_manifest:+__EGL_VENDOR_LIBRARY_FILENAMES="$egl_manifest"} \
    MUJOCO_GL="$MUJOCO_GL_BACKEND" \
    PYOPENGL_PLATFORM="$MUJOCO_GL_BACKEND" \
    MUJOCO_EGL_DEVICE_ID=0 \
    "$EVAL_PYTHON" "$repo_root/python/smoke_env.py" \
        --tasks-per-category "$TASKS_PER_CATEGORY" \
        --steps "$STEPS" \
        --out "$out_dir/frames"
