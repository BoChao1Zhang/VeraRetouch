#!/usr/bin/env bash
# Qwen3-VL-4B v2seg base SFT launcher -- SINGLE card, ZeRO-3 (2026-08-14).
#
# Differences from launch_sft.sh:
#   * --nproc_per_node=1 and a fresh --master_port (29531), so it can coexist
#     with a 2-GPU run on 29517 without a rendezvous port clash.
#   * CONFIG/RUN_DIR point at the v2seg pair.
#   * NO nohup / setsid / background `&` and no job.marker: this file is a
#     QUEUE PAYLOAD. qjob.sh supervises the process and supplies the D-20
#     discipline (log truncation, liveness, --ready gate, marker). Detaching
#     here would free the queue slot instantly and make `q cancel` unable to
#     kill the trainer -- the exact failure waves/where_b_arm.sh documents.
#   * CUDA_VISIBLE_DEVICES is NOT set: the queue wrapper pins the card, and the
#     payload always talks to cuda:0 after that remapping.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/bc/VeraRetouch}"
PYTHON="${PYTHON:-/home/bc/envs/q3vl_sft/bin/python}"
CONFIG="${CONFIG:-${REPO_ROOT}/q3vl/train/configs/sft_base_v2seg.yaml}"
# Must stay equal to `training.output_dir` in CONFIG: this variable only decides
# where run artefacts are created, the trainer takes the checkpoint path from
# the config.
RUN_DIR="${RUN_DIR:-/home/bc/data/runs/q3vl_base_sft_v2seg_20260814}"
NPROC="${NPROC:-1}"

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
# pueued hands its children a 1024 fd soft limit; ZeRO-3 + 4 dataloader workers
# + safetensors shards run past it. Raise it in the entrypoint (CLAUDE.md).
ulimit -n 65536 2>/dev/null || true

mkdir -p "${RUN_DIR}"

CMD=(
  "${PYTHON}" -m torch.distributed.run
  --nproc_per_node="${NPROC}"
  --master_port="${MASTER_PORT:-29531}"
  -m q3vl.train.train_sft
  --config "${CONFIG}"
  "$@"
)

echo "launch_sft_v2seg: exec ${CMD[*]}"
echo "launch_sft_v2seg: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset, queue-provided>}"
cd "${REPO_ROOT}"
exec "${CMD[@]}"
