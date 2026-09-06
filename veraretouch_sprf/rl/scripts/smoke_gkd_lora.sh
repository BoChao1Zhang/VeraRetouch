#!/bin/bash
# 冒烟 3：OPSD-GKD（学生 LoRA r16，教师 = 同权重 disable_adapter 固定教师；beta 0 = forward KL；lmbda 1 全 on-policy）
set -euo pipefail; source "$(dirname "$0")/smoke_common.sh"
OUT=${OUT:-$ROOT/gkd_lora}; mkdir -p "$OUT"
VR_PROBE_OUT="$OUT/probe_segments.jsonl" CUDA_VISIBLE_DEVICES=${GPU:-1} swift rlhf --rlhf_type gkd "${COMMON_ARGS[@]}" "${LORA_ARGS[@]}" \
  --teacher_model "$MODEL" --beta 0 --lmbda 1.0 --sft_alpha 0 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
  --output_dir "$OUT" "$@" 2>&1 | tee "$OUT/run.log"
