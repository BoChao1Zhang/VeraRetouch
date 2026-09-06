#!/bin/bash
# 冒烟 5：双卡全参可行性——GKD，学生 tuner_type full（fp32 master + AdamW、grad ckpt、mb1）在 gpu1；
# 教师在 gpu0：默认经 --teacher_model_server（须先在 gpu0 起 `swift deploy`/vLLM 取 logprobs），
# 或 TEACHER_MODE=local 时同进程加载独立冻结拷贝（同卡，8.3 GiB bf16）。
set -euo pipefail; source "$(dirname "$0")/smoke_common.sh"
OUT=${OUT:-$ROOT/gkd_full}; mkdir -p "$OUT"
TEACHER_MODE=${TEACHER_MODE:-server}
if [ "$TEACHER_MODE" = server ]; then TEACHER=(--teacher_model_server "${TEACHER_URL:-http://127.0.0.1:8100}" --gkd_logits_topk "${TOPK:-64}"); else TEACHER=(--teacher_model "$MODEL"); fi
VR_PROBE_OUT="$OUT/probe_segments.jsonl" CUDA_VISIBLE_DEVICES=${GPU:-1} swift rlhf --rlhf_type gkd "${COMMON_ARGS[@]}" --tuner_type full --torch_dtype float32 --bf16 true --fp16 false \
  "${TEACHER[@]}" --beta 0 --lmbda 1.0 --sft_alpha 0 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
  --output_dir "$OUT" "$@" 2>&1 | tee "$OUT/run.log"
