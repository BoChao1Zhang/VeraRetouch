#!/bin/bash
# =============================================================================
# launch_broker.sh — start the vGate vLLM broker (Phase 0).
#
# A thin, out-of-process OpenAI-compatible reverse proxy in front of the 1-2
# stock `vllm serve` replicas (reason_g0:8001 / reason_g1:8002). Business code
# points its base_url at this broker (:8003) and gets replica discovery,
# least-outstanding routing, a global admission budget, and class fairness for
# free — no business code changes (see UNIFIED_CONCURRENCY_DESIGN_v2). Pure
# Python (fastapi/uvicorn/httpx); no torch, no GPU.
#
# Modes:
#   start (default) -> run in the foreground (use systemd / nohup to daemonize).
#   stop            -> kill any running broker on $VGATE_PORT.
#
# Rollback: stop the broker and point base_url back at a replica
#   (config.yaml vllm.base_url -> :8001/:8002 ; SOURCE_QA_VLLM -> :8002).
#
# Usage: bash dataset_build/core/broker/launch_broker.sh [start|stop]
# Env (all optional): VGATE_PORT(8003) VGATE_REPLICA_PORTS("8001,8002")
#   VGATE_SERVED_NAME(qwen3_5-35b-a3b) VGATE_REPLICA_CAP(32) VGATE_LOG_LEVEL(info)
# =============================================================================
set -uo pipefail

cd /home/bc/VeraRetouch                       # so `dataset_build` resolves
PY=/home/bc/miniconda3/bin/python             # base env: fastapi/uvicorn/httpx

MODE="${1:-start}"
PORT="${VGATE_PORT:-8003}"

case "$MODE" in
  start)
    echo "[vgate] starting broker on :$PORT (replicas=${VGATE_REPLICA_PORTS:-8001,8002}, served=${VGATE_SERVED_NAME:-qwen3_5-35b-a3b}, cap=${VGATE_REPLICA_CAP:-32})"
    exec "$PY" -m dataset_build.core.broker.app \
      --host "${VGATE_HOST:-0.0.0.0}" \
      --port "$PORT" \
      --replica-ports "${VGATE_REPLICA_PORTS:-8001,8002}" \
      --served-name "${VGATE_SERVED_NAME:-qwen3_5-35b-a3b}" \
      --cap "${VGATE_REPLICA_CAP:-32}" \
      --log-level "${VGATE_LOG_LEVEL:-info}"
    ;;
  stop)
    pids=$(pgrep -f "dataset_build.core.broker.app" || true)
    if [ -n "$pids" ]; then
      echo "[vgate] stopping broker pids: $pids"
      kill $pids
    else
      echo "[vgate] no broker running"
    fi
    ;;
  *)
    echo "usage: $0 [start|stop]"; exit 2
    ;;
esac
