#!/bin/bash
# =============================================================================
# launch_build.sh — CANONICAL orchestrator launcher (post-vGate).
#
# The single normalized startup path for the build's sharded workers. Unlike the
# deprecated launch_dual.sh, it does NOT launch vLLM, does NOT sed per-port
# configs (mk_cfg), and does NOT hardcode /2: every shard uses the one
# config.yaml whose vllm.base_url already points at the vGate broker (:8003),
# and the broker balances across whatever replicas are live. Shard count is
# independent of replica count.
#
# Topology (start these first; each is independently managed):
#   1. vGate broker   ->  dataset_build/core/broker/launch_broker.sh  (or vgate.service)
#   2. vLLM replica(s)->  dataset_build/docker/launch_reasoning.sh    (or supervisor.py)
#   3. THIS launcher  ->  N renderer shards, all consuming the broker.
#
# Each shard i runs its in-process teacher renderer on GPU ${GPUS[i]} and sends
# all VLM work to the broker. Shards = number of GPUs listed in $VERA_GPUS.
#
# Modes:
#   pilot [N]        -> --pilot N(=2700) over $STREAMS
#   full  [STREAMS]  -> --full --stream STREAMS(=S1,S5,S6,S7) --resume
#   fast-degrade [S] -> CPU-only S1/S7 sharded build; no vLLM, no renderer, no broker
#
# Env: VERA_GPUS="0 1" (shard<->GPU map) · VGATE_URL=http://localhost:8003 ·
#      VERA_CONFIG=dataset_build/config.yaml · STREAMS=S1,S5,S6,S7
#
# Usage: bash dataset_build/launch_build.sh pilot|full|fast-degrade [arg]
# =============================================================================
set -uo pipefail
cd /home/bc/VeraRetouch

PY=/home/bc/miniconda3/bin/python
CFG="${VERA_CONFIG:-dataset_build/config.yaml}"
BROKER="${VGATE_URL:-http://localhost:8003}"
GPUS_STR="${VERA_GPUS:-0 1}"
STREAMS_DEFAULT="${STREAMS:-S1,S5,S6,S7}"
LOGDIR=/home/bc/data/datasets/vera_directionA_1M/logs; mkdir -p "$LOGDIR"

MODE="${1:-pilot}"; ARG="${2:-}"
read -r -a GPUS <<< "$GPUS_STR"
N=${#GPUS[@]}

assert_broker () {
  if ! curl -sf "$BROKER/healthz" >/dev/null 2>&1; then
    echo "[build] vGate broker not healthy at $BROKER" >&2
    echo "[build] start it first: bash dataset_build/core/broker/launch_broker.sh start" >&2
    exit 1
  fi
  echo "[build] broker OK at $BROKER ; config=$CFG (base_url must point here)"
}

run_shards () {  # $1=RUNFLAGS ; launches N shards, shard i on GPU ${GPUS[i]}
  local flags="$1" pids=()
  for i in "${!GPUS[@]}"; do
    echo "[build] shard $i/$N on GPU ${GPUS[$i]} : $flags --shard $i/$N"
    CUDA_VISIBLE_DEVICES="${GPUS[$i]}" $PY -m dataset_build.run --config "$CFG" \
      $flags --shard "$i/$N" 2>&1 | tee "$LOGDIR/build_g${GPUS[$i]}.log" &
    pids+=($!)
  done
  echo "[build] shard pids: ${pids[*]} ; waiting..."
  local rc=0; for p in "${pids[@]}"; do wait "$p" || rc=$?; done
  return $rc
}

case "$MODE" in
  pilot)
    assert_broker
    run_shards "--pilot ${ARG:-2700} --stream $STREAMS_DEFAULT"
    ;;
  full)
    assert_broker
    run_shards "--full --stream ${ARG:-$STREAMS_DEFAULT} --resume"
    ;;
  fast-degrade)
    # CPU-only S1/S7 (no vLLM, no renderer, no broker dependency).
    local_streams="${ARG:-S1,S7}"
    pids=()
    for i in "${!GPUS[@]}"; do
      echo "[build] fast-degrade shard $i/$N (CPU): --full --stream $local_streams --resume"
      CUDA_VISIBLE_DEVICES="" $PY -m dataset_build.run --config "$CFG" \
        --full --stream "$local_streams" --resume --shard "$i/$N" \
        2>&1 | tee "$LOGDIR/fast_degrade_$i.log" &
      pids+=($!)
    done
    rc=0; for p in "${pids[@]}"; do wait "$p" || rc=$?; done
    exit $rc
    ;;
  *)
    echo "unknown mode: $MODE (pilot|full|fast-degrade)"; exit 2;;
esac
echo "[build] $MODE done. Shards/manifests under /home/bc/data/datasets/vera_directionA_1M/"
