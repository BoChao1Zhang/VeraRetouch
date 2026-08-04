#!/usr/bin/env bash
# S0-JOINT preflight smoke launcher (NOT the real training launcher).
#
# Real training uses q3vl/train/scripts/launch_sft.sh with the production config.
# This one only ever writes into a smoke RUN_DIR and is used by the joint
# preflight to exercise the 2-GPU ZeRO-3 path on the real dataset.
#
# D-20 discipline (CLAUDE.md): rm -f log first (zsh noclobber), prove liveness
# with `ps -p $PID` (never pgrep), tail for substantive output, only then write
# job.marker.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/bc/VeraRetouch}"
PYTHON="${PYTHON:-/home/bc/envs/q3vl_sft/bin/python}"
CONFIG="${CONFIG:-${REPO_ROOT}/q3vl/train/configs/sft_base.yaml}"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
LOG="${LOG:-${RUN_DIR}/train.log}"
NPROC="${NPROC:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

case "${RUN_DIR}" in
  */q3vl_base_sft_20260804*)
    echo "REFUSING: ${RUN_DIR} is the production output path" >&2; exit 2 ;;
esac

mkdir -p "${RUN_DIR}"

# --- D-20 step 1 -------------------------------------------------------------
rm -f "${LOG}"

CMD=(
  "${PYTHON}" -m torch.distributed.run
  --nproc_per_node="${NPROC}"
  --master_port="${MASTER_PORT:-29517}"
  -m q3vl.train.train_sft
  --config "${CONFIG}"
  --output_dir "${RUN_DIR}"
  "$@"
)

echo "launching: ${CMD[*]}"
cd "${REPO_ROOT}"
nohup "${CMD[@]}" > "${LOG}" 2>&1 &
PID=$!
echo "PID=${PID}"

# --- D-20 step 2: ps -p, never pgrep -----------------------------------------
sleep 25
if ! ps -p "${PID}" > /dev/null 2>&1; then
  echo "FAILED: process ${PID} is not alive. Log tail:" >&2
  tail -n 60 "${LOG}" >&2 || true
  exit 1
fi

# --- D-20 step 3: substantive output -----------------------------------------
for _ in $(seq 1 90); do
  if grep -qE 'shard verification PASSED|special token ids|data manifest' "${LOG}"; then
    break
  fi
  if ! ps -p "${PID}" > /dev/null 2>&1; then
    echo "FAILED: process died during startup. Log tail:" >&2
    tail -n 60 "${LOG}" >&2
    exit 1
  fi
  sleep 10
done
if ! grep -qE 'shard verification PASSED|special token ids|data manifest' "${LOG}"; then
  echo "FAILED: no substantive output after 15 minutes." >&2
  tail -n 60 "${LOG}" >&2
  exit 1
fi

# --- D-20 step 4 -------------------------------------------------------------
cat > "${RUN_DIR}/job.marker" <<EOF
{
  "pid": ${PID},
  "command": "${CMD[*]}",
  "log": "${LOG}",
  "config": "${CONFIG}",
  "started_at": "$(date -Iseconds)",
  "host": "$(hostname)",
  "cuda_visible_devices": "${CUDA_VISIBLE_DEVICES}",
  "role": "S0-JOINT preflight smoke (not production training)"
}
EOF
echo "OK: pid=${PID} log=${LOG} marker=${RUN_DIR}/job.marker"
tail -n 25 "${LOG}"
