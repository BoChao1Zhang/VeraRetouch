#!/bin/bash
# =============================================================================
# launch_reasoning.sh — data-parallel local vLLM replicas for canonical
# Responses annotation. vGate discovers these replicas and exposes the single
# broker URL configured in databuild.toml; databuild never addresses a replica.
#
# Model: Qwen3.5-35B-A3B-FP8 (MoE, multimodal, fp8). The user asked for a
#        "Qwen3.6-35B-A3B" model; that name was empty on disk when this build
#        was specified, so this script uses the available, verified complete
#        35B-A3B-FP8 replica. (A Qwen3.6-35B-A3B-FP8 dir later appeared but uses
#        a non-standard layers-N.safetensors / outside / mtp weight layout that
#        vLLM nightly does not load via the standard path; 3.5-FP8 uses the
#        standard model.safetensors-NNNNN sharding and loads natively.)
#        Arch Qwen3_5MoeForConditionalGeneration / model_type qwen3_5_moe is
#        supported by vllm/vllm-openai:nightly (vLLM 0.20.1rc1 + transformers 5.7).
#
# Topology — DATA PARALLEL, one full replica per card (NOT tensor-parallel):
#   reason_g0 -> GPU0 -> :8001
#   reason_g1 -> GPU1 -> :8002
#   => two independent replicas behind vGate. Each uses tensor parallel size 1.
#
# PERSISTENT COMPILE / CUDA-GRAPH CACHE (the key ask):
#   Host dir $CACHE is mounted into BOTH containers at /root/.cache/vllm and
#   VLLM_CACHE_ROOT is pinned there, so vLLM's torch.compile / inductor +
#   CUDA-graph artifacts are written to the host on the FIRST launch and REUSED
#   on every later launch — container restarts skip recompilation (first launch
#   compiles+caches and is slow; subsequent launches reuse and start fast).
#
# Modes:
#   start (default) -> launch both replicas + wait until /v1/models serves the
#                      model on :8001 and :8002 (LONG timeout: first compile is slow).
#   stop            -> docker rm -f reason_g0 reason_g1.
#
# Usage: bash dataset_build/docker/launch_reasoning.sh [start|stop]
# =============================================================================
set -uo pipefail

MODE="${1:-start}"

IMG="vllm/vllm-openai:nightly"
MODELS="/home/bc/data/models"
MODEL_DIR="/models/Qwen3.5-35B-A3B-FP8"          # in-container path (see -v below)
SERVED="qwen3_5-35b-a3b"                          # MUST equal annotation.local.model in TOML
MAXLEN=32768

# Persistent compile/CUDA-graph cache. Created once on the host; mounted into
# each container at /root/.cache/vllm and pinned via VLLM_CACHE_ROOT so the
# torch.compile/inductor + CUDA-graph artifacts persist across restarts.
CACHE="/home/bc/data/vllm_cache"
mkdir -p "$CACHE"

# --- start one replica (explicit per-container command; zsh-safe, no word-split) ---
start_replica () {  # name device hostport
  local name="$1" dev="$2" port="$3"
  docker rm -f "$name" >/dev/null 2>&1
  echo "[reason] launching $name : Qwen3.5-35B-A3B-FP8 on GPU$dev -> :$port (served=$SERVED)"
  docker run -d --runtime nvidia --gpus "\"device=$dev\"" \
    -v "$MODELS":/models \
    -v "$CACHE":/root/.cache/vllm \
    -e VLLM_CACHE_ROOT=/root/.cache/vllm \
    -p "$port":8000 --ipc=host --name "$name" "$IMG" \
    --model "$MODEL_DIR" \
    --served-model-name "$SERVED" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --max-model-len "$MAXLEN" \
    --limit-mm-per-prompt '{"image": 2}' \
    --trust-remote-code >/dev/null
  # NOTE: quant fp8 is auto-detected from config.json's quantization_config;
  # do NOT also pass a conflicting --dtype (it would error against fp8 weights).
}

# --- wait until a replica serves the model (first compile is slow: ~30min cap) ---
wait_replica () {  # hostport
  local port="$1"
  echo -n "[reason] waiting :$port for '$SERVED' (first launch compiles + caches; up to ~30min) "
  for i in $(seq 1 180); do   # 180 * 10s = 1800s = 30min
    if curl -s "http://localhost:$port/v1/models" 2>/dev/null | grep -q "$SERVED"; then
      echo "READY (~$((i*10))s)"; return 0
    fi
    sleep 10; echo -n "."
  done
  echo "TIMEOUT"; return 1
}

case "$MODE" in
  start)
    start_replica reason_g0 0 8001
    start_replica reason_g1 1 8002
    wait_replica 8001 || { echo "[reason] reason_g0 failed to come up; see: docker logs reason_g0"; exit 1; }
    wait_replica 8002 || { echo "[reason] reason_g1 failed to come up; see: docker logs reason_g1"; exit 1; }
    echo "[reason] BOTH replicas READY for vGate discovery:"
    echo "         reason_g0  GPU0  http://localhost:8001/v1"
    echo "         reason_g1  GPU1  http://localhost:8002/v1"
    echo "[reason] compile/CUDA-graph cache persisted at $CACHE (restarts reuse it)."
    echo "[reason] databuild.toml annotation.local.model must be '$SERVED'."
    echo "[reason] stop with: bash dataset_build/docker/launch_reasoning.sh stop"
    ;;
  stop)
    echo "[reason] stopping reason_g0 reason_g1"
    docker rm -f reason_g0 reason_g1 2>/dev/null
    echo "[reason] stopped (compile cache at $CACHE is preserved)."
    ;;
  start-one)
    # Elastic per-card launch used by core/broker/supervisor.py. NON-blocking:
    # `docker run -d` returns immediately and the vGate broker detects readiness
    # via its own /v1/models polling (first compile is slow but cache-backed).
    #   usage: launch_reasoning.sh start-one <name> <gpu_index> <host_port>
    name="${2:?name}"; dev="${3:?gpu_index}"; port="${4:?host_port}"
    start_replica "$name" "$dev" "$port"
    echo "[reason] launched $name on GPU$dev -> :$port (detached; broker will discover when ready)"
    ;;
  stop-one)
    #   usage: launch_reasoning.sh stop-one <name>
    name="${2:?name}"
    docker rm -f "$name" 2>/dev/null
    echo "[reason] stopped $name (compile cache at $CACHE is preserved)."
    ;;
  *)
    echo "unknown mode: $MODE (start|stop|start-one|stop-one)"; exit 2;;
esac
