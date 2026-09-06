#!/bin/bash
# 冒烟 4：GRPO + 教师（OPD-RL；无 reward_funcs = 纯蒸馏优势），HF 生成（use_vllm false），G=2
# generation_batch_size = per_device × grad_accum 必须被 num_generations 整除 → per_device 1 × grad_accum 2
set -euo pipefail; source "$(dirname "$0")/smoke_common.sh"
OUT=${OUT:-$ROOT/grpo_lora}; mkdir -p "$OUT"
VR_PROBE_OUT="$OUT/probe_segments.jsonl" CUDA_VISIBLE_DEVICES=${GPU:-1} swift rlhf --rlhf_type grpo "${COMMON_ARGS[@]}" "${LORA_ARGS[@]}" \
  --teacher_model "$MODEL" --num_generations 2 --use_vllm false \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 2 \
  --output_dir "$OUT" "$@" 2>&1 | tee "$OUT/run.log"
