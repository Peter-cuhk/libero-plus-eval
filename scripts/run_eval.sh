#!/usr/bin/env bash
# Run one LIBERO-plus (or clean LIBERO) evaluation shard: start an openpi policy
# server, wait for it, drive it with eval_libero_plus.py, tear the server down.
#
# Usage:
#   run_eval.sh <shard-file> <output-dir>
#
# Configured entirely through environment variables (see the defaults below) so
# the same script serves smoke runs, the clean-LIBERO regression and full 10,030
# episode shards without editing anything.
#
# The EGL handling here is lifted from openpi-icl's run_pi05_libero_eval.sh,
# which is the version that has actually rendered LIBERO on these H20 nodes.

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <shard-file> <output-dir>" >&2
    exit 2
fi

shard_file="$1"
output_dir="$2"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- what to evaluate -------------------------------------------------------
: "${CONFIG_NAME:=pi05_libero}"
: "${CHECKPOINT_DIR:=gs://openpi-assets/checkpoints/pi05_libero}"
: "${BENCHMARK:=plus}"                 # plus | clean
: "${NUM_TRIALS_PER_TASK:=}"           # default depends on BENCHMARK
: "${NUM_WORKERS:=16}"
: "${SEED:=7}"
: "${SAVE_VIDEOS:=12}"
: "${RESUME:=1}"

# --- where things live ------------------------------------------------------
: "${LIBERO_PLUS_ROOT:=/mnt/cpfs/PeterX/repos/LIBERO-plus}"
: "${LIBERO_CLEAN_ROOT:=/mnt/cpfs/PeterX/policy/openpi-ar/third_party/libero}"
: "${OPENPI_REPO:=/mnt/cpfs/PeterX/policy/openpi-ar}"
: "${EVAL_PYTHON:=/mnt/cpfs/PeterX/env/libero-plus-eval-py38/bin/python}"
: "${OPENPI_DATA_HOME:=/mnt/cpfs/PeterX/data/openpi_data}"
: "${UV_CACHE_DIR:=/mnt/cpfs/uv_cache}"

# --- server / runtime knobs -------------------------------------------------
: "${SERVER_PORT:=8000}"
: "${SERVER_START_TIMEOUT:=1800}"      # first run must download the checkpoint
: "${SERVER_HANDSHAKE_TIMEOUT_S:=600}"
: "${SKIP_UV_SYNC:=0}"
: "${JAX_COMPILATION_CACHE_DIR:=/tmp/libero-plus-jax-cache}"
: "${EGL_VENDOR_LIBRARY:=libEGL_nvidia.so.0}"
: "${EGL_LOADER_LIBRARY:=/usr/lib/x86_64-linux-gnu/libEGL.so.1}"
: "${MUJOCO_GL_BACKEND:=egl}"
: "${EXTRA_LIB_DIR:=}"                 # optional dir of .so files to prepend to LD_LIBRARY_PATH

case "$BENCHMARK" in
    plus)
        benchmark_root="$LIBERO_PLUS_ROOT"
        : "${NUM_TRIALS_PER_TASK:=1}"          # LIBERO-plus protocol: exactly one trial per task
        alignment_flag=()
        ;;
    clean)
        benchmark_root="$LIBERO_CLEAN_ROOT"
        : "${NUM_TRIALS_PER_TASK:=50}"         # original LIBERO protocol
        alignment_flag=(--skip-alignment-check) # plain LIBERO ships no task_classification.json
        ;;
    *)
        echo "BENCHMARK must be 'plus' or 'clean', got: $BENCHMARK" >&2
        exit 2
        ;;
esac

for path in "$shard_file" "$benchmark_root" "$OPENPI_REPO"; do
    [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 1; }
done
[[ -x "$EVAL_PYTHON" ]] || { echo "Missing eval python: $EVAL_PYTHON (run bootstrap_h20.sh)" >&2; exit 1; }
[[ -f "$OPENPI_REPO/uv.lock" ]] || { echo "Not an openpi repo: $OPENPI_REPO" >&2; exit 1; }
# A local checkpoint must be fully committed; a gs:// URI is fetched by the server.
if [[ "$CHECKPOINT_DIR" != gs://* && ! -f "$CHECKPOINT_DIR/params/_METADATA" ]]; then
    echo "Checkpoint is not committed: $CHECKPOINT_DIR (missing params/_METADATA)" >&2
    exit 1
fi

mkdir -p "$output_dir" "$JAX_COMPILATION_CACHE_DIR"
run_libs="$output_dir/runtime-libs"
libero_config_dir="$output_dir/libero-config"
mkdir -p "$run_libs" "$libero_config_dir" "$output_dir/libero-datasets"

# LIBERO resolves every benchmark path through this config file. Writing our own
# and pointing LIBERO_CONFIG_PATH at it avoids touching the shared ~/.libero.
printf '%s\n' \
    "benchmark_root: $benchmark_root/libero/libero" \
    "bddl_files: $benchmark_root/libero/libero/bddl_files" \
    "init_states: $benchmark_root/libero/libero/init_files" \
    "datasets: $output_dir/libero-datasets" \
    "assets: $benchmark_root/libero/libero/assets" \
    > "$libero_config_dir/config.yaml"

# glvnd needs an ICD manifest to find the NVIDIA EGL driver; the CUDA images do
# not always ship /usr/share/glvnd/egl_vendor.d, so supply one of our own.
egl_manifest=""
if [[ "$MUJOCO_GL_BACKEND" == "egl" ]]; then
    [[ -e "$EGL_LOADER_LIBRARY" ]] || { echo "Missing EGL loader: $EGL_LOADER_LIBRARY" >&2; exit 1; }
    ln -sfn "$EGL_LOADER_LIBRARY" "$run_libs/libEGL.so"
    egl_manifest="$run_libs/10_nvidia.json"
    printf '{"file_format_version":"1.0.0","ICD":{"library_path":"%s"}}\n' "$EGL_VENDOR_LIBRARY" > "$egl_manifest"
fi
if [[ -n "$EXTRA_LIB_DIR" && -d "$EXTRA_LIB_DIR" ]]; then
    for so in "$EXTRA_LIB_DIR"/*.so*; do
        [[ -e "$so" ]] && ln -sfn "$so" "$run_libs/$(basename "$so")"
    done
fi

# The evaluation environment is a micromamba prefix that ships both ImageMagick
# (required by `wand`, which env_wrapper.py imports at module level) and glib
# (libgthread/libglib, absent from the DLC images). Put its lib dir on the
# search path and point wand at it.
eval_prefix="$(dirname "$(dirname "$EVAL_PYTHON")")"
if [[ ! -e "$eval_prefix/lib/libMagickWand-7.Q16HDRI.so" ]]; then
    echo "Evaluation env is missing ImageMagick: $eval_prefix/lib (run bootstrap_h20.sh)" >&2
    exit 1
fi

echo "=== LIBERO-plus eval ==="
echo "  benchmark    : $BENCHMARK ($benchmark_root)"
echo "  config/ckpt  : $CONFIG_NAME  <-  $CHECKPOINT_DIR"
echo "  shard        : $shard_file"
echo "  output       : $output_dir"
echo "  trials/task  : $NUM_TRIALS_PER_TASK   workers: $NUM_WORKERS   seed: $SEED"
echo "  renderer     : $MUJOCO_GL_BACKEND"

# ---------------------------------------------------------------- policy server
cd "$OPENPI_REPO"
if [[ "$SKIP_UV_SYNC" != "1" ]]; then
    UV_CACHE_DIR="$UV_CACHE_DIR" uv sync --frozen
fi
uv_args=(--frozen)
[[ "$SKIP_UV_SYNC" == "1" ]] && uv_args+=(--no-sync)

server_log="$output_dir/policy-server.log"
OPENPI_DATA_HOME="$OPENPI_DATA_HOME" \
UV_CACHE_DIR="$UV_CACHE_DIR" \
JAX_COMPILATION_CACHE_DIR="$JAX_COMPILATION_CACHE_DIR" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run "${uv_args[@]}" scripts/serve_policy.py \
    --env LIBERO \
    --port "$SERVER_PORT" \
    policy:checkpoint \
    --policy.config "$CONFIG_NAME" \
    --policy.dir "$CHECKPOINT_DIR" \
    > "$server_log" 2>&1 &
server_pid=$!
cleanup() { kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

ready=0
for ((elapsed = 0; elapsed < SERVER_START_TIMEOUT; elapsed += 5)); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "Policy server exited before becoming ready:" >&2
        tail -n 120 "$server_log" >&2
        exit 1
    fi
    if (exec 3<>"/dev/tcp/127.0.0.1/$SERVER_PORT") 2>/dev/null; then exec 3>&-; ready=1; break; fi
    sleep 5
done
if [[ "$ready" -ne 1 ]]; then
    echo "Policy server not ready after ${SERVER_START_TIMEOUT}s:" >&2
    tail -n 120 "$server_log" >&2
    exit 1
fi
echo "policy server ready on :$SERVER_PORT (pid $server_pid)"

# ---------------------------------------------------------------------- client
client_args=(
    --shard-file "$shard_file"
    --out "$output_dir"
    --host 127.0.0.1
    --port "$SERVER_PORT"
    --server-handshake-timeout-s "$SERVER_HANDSHAKE_TIMEOUT_S"
    --num-trials-per-task "$NUM_TRIALS_PER_TASK"
    --num-workers "$NUM_WORKERS"
    --seed "$SEED"
    --save-videos "$SAVE_VIDEOS"
    --benchmark-root "$benchmark_root"
)
[[ "$RESUME" == "1" ]] && client_args+=(--resume)
client_args+=("${alignment_flag[@]+"${alignment_flag[@]}"}")

env \
    LIBERO_PLUS_ROOT="$benchmark_root" \
    LIBERO_CONFIG_PATH="$libero_config_dir" \
    PYTHONPATH="$benchmark_root:$OPENPI_REPO/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}" \
    MAGICK_HOME="$eval_prefix" \
    LD_LIBRARY_PATH="$run_libs:$eval_prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    ${egl_manifest:+__EGL_VENDOR_LIBRARY_FILENAMES="$egl_manifest"} \
    MUJOCO_GL="$MUJOCO_GL_BACKEND" \
    PYOPENGL_PLATFORM="$MUJOCO_GL_BACKEND" \
    MUJOCO_EGL_DEVICE_ID=0 \
    "$EVAL_PYTHON" "$repo_root/python/eval_libero_plus.py" "${client_args[@]}"

printf '%s\n' \
    "benchmark: $BENCHMARK" \
    "config: $CONFIG_NAME" \
    "checkpoint: $CHECKPOINT_DIR" \
    "shard_file: $shard_file" \
    "num_trials_per_task: $NUM_TRIALS_PER_TASK" \
    "seed: $SEED" \
    > "$output_dir/evaluation.yaml"
touch "$output_dir/_SUCCESS"
echo "=== shard done: $output_dir ==="
