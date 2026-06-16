#!/bin/bash
# =============================================================================
# DEPRECATED (vGate cutover, 2026-06-16). Use the canonical, decoupled path:
#   1. broker     :  bash dataset_build/core/broker/launch_broker.sh start
#   2. replica(s) :  bash dataset_build/docker/launch_reasoning.sh start
#                    (or the elastic supervisor: python -m dataset_build.core.broker.supervisor --apply)
#   3. shards     :  bash dataset_build/launch_build.sh pilot|full|fast-degrade
#
# Why deprecated: this launcher (a) serves the stale qwen3-vl-8b model on
# :8001/:8002 — the build now expects the 35B 'qwen3_5-35b-a3b' (launch_reasoning.sh),
# so the served-name no longer matches config.yaml; (b) sed's per-port configs
# (mk_cfg) and hardcodes /2 — both obsolete now that every shard points at the
# broker (:8003) and shard count is decoupled from replica count. Kept only for
# reference / rollback. See dataset_build/core/broker/README.md.
# -----------------------------------------------------------------------------
# Dual-GPU "mirror" launcher for the VeraRetouch Direction-A dataset build.
#   GPU0: vLLM qwen3-vl-8b @:8001 + renderer (orchestrator --shard 0/2)
#   GPU1: vLLM qwen3-vl-8b @:8002 + renderer (orchestrator --shard 1/2)
# Modes:
#   smoke            -> 1 vLLM(GPU0:8001) + 1 worker(GPU1), --pilot 60 --stream S1,S5,S6,S7 --out-suffix _smoke
#   fast-degrade     -> CPU-only S1/S7 sharded build; no vLLM, no renderer.
#   pilot [N]        -> 2 vLLM + 2 sharded workers, --pilot N(=2700) --stream S1,S5,S6,S7
#   full  [STREAMS]  -> 2 vLLM + 2 sharded workers, --full --stream STREAMS(=S1,S5,S6,S7) --resume
# Usage: bash dataset_build/launch_dual.sh smoke|fast-degrade|pilot|full [arg]
set -uo pipefail
cd /home/bc/VeraRetouch
echo "[launch_dual] DEPRECATED — prefer launch_broker.sh + launch_reasoning.sh + launch_build.sh (see header)." >&2
MODE="${1:-smoke}"; ARG="${2:-}"
PY=/home/bc/miniconda3/bin/python            # base env: llava + torch + renderer
IMG=vllm/vllm-openai:nightly
MODELS=/home/bc/data/models
CFG=dataset_build/config.yaml
LOGDIR=/home/bc/data/datasets/vera_directionA_1M/logs; mkdir -p "$LOGDIR"
MEMUTIL=0.55    # vLLM KV; leaves renderer/headroom on the same H100. Higher values
                # can starve the native-res teacher render during mirror runs.
MAXLEN=32768

launch_vllm () {  # name device port
  local name=$1 dev=$2 port=$3
  docker rm -f "$name" >/dev/null 2>&1
  echo "[vllm] launching $name on GPU$dev :$port"
  docker run -d --runtime nvidia --gpus "\"device=$dev\"" \
    -v "$MODELS":/models -p "$port":8000 --ipc=host --name "$name" "$IMG" \
    --model /models/Qwen3-VL-8B-Instruct --served-model-name qwen3-vl-8b \
    --tensor-parallel-size 1 --gpu-memory-utilization "$MEMUTIL" \
    --max-model-len "$MAXLEN" --limit-mm-per-prompt '{"image": 2}' \
    --dtype bfloat16 --trust-remote-code >/dev/null
}
wait_vllm () {  # port
  local port=$1
  echo -n "[vllm] waiting :$port "
  for i in $(seq 1 90); do
    if curl -s "http://localhost:$port/v1/models" 2>/dev/null | grep -q qwen3-vl; then echo "READY (${i}0s)"; return 0; fi
    sleep 10; echo -n "."
  done
  echo "TIMEOUT"; return 1
}
mk_cfg () {  # port -> config path (only vllm.base_url differs)
  local port=$1
  local out="dataset_build/config_g${port}.yaml"
  sed "s#http://localhost:8001/v1#http://localhost:${port}/v1#" "$CFG" > "$out"; echo "$out"
}
cleanup () { echo "[launch] (vLLM containers left running: docker rm -f qwen_g0 qwen_g1 to stop)"; }
trap cleanup EXIT

case "$MODE" in
  smoke)
    launch_vllm qwen_g0 0 8001; wait_vllm 8001 || exit 1
    echo "[run] SMOKE: renderer GPU1 -> vLLM:8001, --pilot 60 --stream S1,S5,S6,S7"
    CUDA_VISIBLE_DEVICES=1 $PY -m dataset_build.run --config "$CFG" \
      --pilot 60 --stream S1,S5,S6,S7 --out-suffix _smoke 2>&1 | tee "$LOGDIR/smoke.log"
    ;;
  fast-degrade)
    STREAMS="${ARG:-S1,S7}"
    RUNFLAGS="--full --stream $STREAMS --resume"
    echo "[run] FAST-DEGRADE flags='$RUNFLAGS' (CPU-only: shard 0/2 + 1/2, no vLLM containers)"
    CUDA_VISIBLE_DEVICES="" $PY -m dataset_build.run --config "$CFG" $RUNFLAGS --shard 0/2 \
      2>&1 | tee "$LOGDIR/fast_degrade_g0.log" &
    P0=$!
    CUDA_VISIBLE_DEVICES="" $PY -m dataset_build.run --config "$CFG" $RUNFLAGS --shard 1/2 \
      2>&1 | tee "$LOGDIR/fast_degrade_g1.log" &
    P1=$!
    echo "[run] workers pid w0=$P0 w1=$P1 ; waiting..."
    wait $P0; wait $P1
    ;;
  pilot|full)
    if [ "$MODE" = pilot ]; then N="${ARG:-2700}"; STREAMS=S1,S5,S6,S7
        RUNFLAGS="--pilot $N --stream $STREAMS"
    else STREAMS="${ARG:-S1,S5,S6,S7}"; RUNFLAGS="--full --stream $STREAMS --resume"; fi
    launch_vllm qwen_g0 0 8001
    launch_vllm qwen_g1 1 8002
    wait_vllm 8001 || exit 1
    wait_vllm 8002 || exit 1
    CFG0=$(mk_cfg 8001); CFG1=$(mk_cfg 8002)
    echo "[run] MODE=$MODE flags='$RUNFLAGS' (mirror: shard 0/2 on GPU0, 1/2 on GPU1)"
    CUDA_VISIBLE_DEVICES=0 $PY -m dataset_build.run --config "$CFG0" $RUNFLAGS --shard 0/2 \
      2>&1 | tee "$LOGDIR/${MODE}_g0.log" &
    P0=$!
    CUDA_VISIBLE_DEVICES=1 $PY -m dataset_build.run --config "$CFG1" $RUNFLAGS --shard 1/2 \
      2>&1 | tee "$LOGDIR/${MODE}_g1.log" &
    P1=$!
    echo "[run] workers pid g0=$P0 g1=$P1 ; waiting..."
    wait $P0; wait $P1
    ;;
  *) echo "unknown mode: $MODE (smoke|fast-degrade|pilot|full)"; exit 2;;
esac
echo "[launch] $MODE done. Shards/manifests under /home/bc/data/datasets/vera_directionA_1M/"
