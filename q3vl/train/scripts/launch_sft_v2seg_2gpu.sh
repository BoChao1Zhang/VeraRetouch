#!/usr/bin/env bash
# Qwen3-VL-4B v2seg base SFT launcher -- TWO cards, ZeRO-3 (2026-08-14).
#
# This is the original spec-7.2 two-GPU geometry (micro 4 x GAS 4 x world_size 2
# = global batch 32) applied to the v2seg target template. It replaces the
# single-card variant launcher (launch_sft_v2seg.sh, nproc 1, GAS 8).
#
# Differences from launch_sft.sh:
#   * CONFIG/RUN_DIR point at the v2seg 2-GPU pair.
#   * --master_port=29533, distinct from 29517 (launch_sft.sh) and 29531
#     (launch_sft_v2seg.sh), so a rendezvous port clash is impossible.
#   * CUDA_VISIBLE_DEVICES is set UNCONDITIONALLY to 0,1. The queue wrapper
#     exports a single card number (CUDA_VISIBLE_DEVICES=0) into the payload
#     env; a `${CUDA_VISIBLE_DEVICES:-0,1}` default would inherit that and
#     silently start a 2-rank job on one card. This job owns BOTH cards, so the
#     gpu1 queue group must be paused before submitting.
#   * ulimit -n 65536: pueued hands its children a 1024 fd soft limit; ZeRO-3 +
#     4 dataloader workers + safetensors shards run past it (CLAUDE.md).
#   * NO nohup / setsid / background `&` and no job.marker: this file is a
#     QUEUE PAYLOAD. qjob.sh supervises the process and supplies the D-20
#     discipline (log truncation, liveness, --ready gate, marker). Detaching
#     here would free the queue slot instantly and make `q cancel` unable to
#     kill the trainer.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/bc/VeraRetouch}"
PYTHON="${PYTHON:-/home/bc/envs/q3vl_sft/bin/python}"
CONFIG="${CONFIG:-${REPO_ROOT}/q3vl/train/configs/sft_base_v2seg_2gpu.yaml}"
# Must stay equal to `training.output_dir` in CONFIG: this variable only decides
# where run artefacts are created, the trainer takes the checkpoint path from
# the config.
RUN_DIR="${RUN_DIR:-/home/bc/data/runs/q3vl_base_sft_v2seg_20260814}"
NPROC="${NPROC:-2}"

# Unconditional: overrides whatever single card the queue wrapper injected.
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
ulimit -n 65536 2>/dev/null || true

mkdir -p "${RUN_DIR}"

CMD=(
  "${PYTHON}" -m torch.distributed.run
  --nproc_per_node="${NPROC}"
  --master_port="${MASTER_PORT:-29533}"
  -m q3vl.train.train_sft
  --config "${CONFIG}"
  "$@"
)

echo "launch_sft_v2seg_2gpu: exec ${CMD[*]}"
echo "launch_sft_v2seg_2gpu: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (forced, both cards)"
cd "${REPO_ROOT}"
exec "${CMD[@]}"
