#!/usr/bin/env bash
# Qwen3-VL-4B Arm B base SFT launcher (spec 7.2: 2 GPUs, ZeRO-3).
#
# THIS SCRIPT DOES NOT RUN AS PART OF S0-TRAIN. The task card forbids starting
# the real training; it exists so the launch is one reviewed command later.
#
# D-20 submission discipline (CLAUDE.md) is implemented below:
#   1. rm -f the target log first (zsh noclobber makes `> existing` fail silently)
#   2. prove liveness with `ps -p $PID`, never pgrep
#   3. tail the log for substantive output
#   4. only then write job.marker
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/bc/VeraRetouch}"
PYTHON="${PYTHON:-/home/bc/envs/q3vl_sft/bin/python}"
CONFIG="${CONFIG:-${REPO_ROOT}/q3vl/train/configs/sft_base.yaml}"
RUN_DIR="${RUN_DIR:-/mnt/nfs/bc/runs/q3vl_base_sft_20260804}"
LOG="${LOG:-${RUN_DIR}/train.log}"
NPROC="${NPROC:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
# ZeRO-3 + a 4B model on 2x H100 leaves plenty of room; keep NCCL chatty enough
# to diagnose a hang without flooding the log.
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

mkdir -p "${RUN_DIR}"

# --- D-20 step 1: the log must not already exist -----------------------------
rm -f "${LOG}"

CMD=(
  "${PYTHON}" -m torch.distributed.run
  --nproc_per_node="${NPROC}"
  --master_port="${MASTER_PORT:-29517}"
  -m q3vl.train.train_sft
  --config "${CONFIG}"
  "$@"
)

echo "launching: ${CMD[*]}"
cd "${REPO_ROOT}"
nohup "${CMD[@]}" > "${LOG}" 2>&1 &
PID=$!

# --- D-20 step 2: prove it is alive (ps -p, never pgrep) ---------------------
sleep 20
if ! ps -p "${PID}" > /dev/null 2>&1; then
  echo "FAILED: process ${PID} is not alive. Log tail:" >&2
  tail -n 40 "${LOG}" >&2 || true
  exit 1
fi

# --- D-20 step 3: the log must contain substantive output --------------------
for _ in $(seq 1 60); do
  if grep -qE 'special token ids|trainable params|shard verification PASSED' "${LOG}"; then
    break
  fi
  if ! ps -p "${PID}" > /dev/null 2>&1; then
    echo "FAILED: process died during startup. Log tail:" >&2
    tail -n 40 "${LOG}" >&2
    exit 1
  fi
  sleep 10
done
if ! grep -qE 'special token ids|trainable params|shard verification PASSED' "${LOG}"; then
  echo "FAILED: no substantive output after 10 minutes." >&2
  tail -n 40 "${LOG}" >&2
  exit 1
fi

# --- D-20 step 4: only now write the marker ----------------------------------
cat > "${RUN_DIR}/job.marker" <<EOF
{
  "pid": ${PID},
  "command": "${CMD[*]}",
  "log": "${LOG}",
  "config": "${CONFIG}",
  "started_at": "$(date -Iseconds)",
  "host": "$(hostname)",
  "cuda_visible_devices": "${CUDA_VISIBLE_DEVICES}"
}
EOF
echo "OK: pid=${PID} log=${LOG} marker=${RUN_DIR}/job.marker"
tail -n 20 "${LOG}"
