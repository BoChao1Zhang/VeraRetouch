#!/bin/bash
# gpu1：OPSD 全参训练（GKD 路径，rollout=vLLM server@gpu0，教师=logprob server@gpu0）。
# 用法：STEPS=20 PB=1 GA=8 bash train_opsd_full.sh [透传 swift rlhf 参数]
set -euo pipefail; source "$(dirname "$0")/../smoke_common.sh"
DATA=${DATA_FULL:-/data/runs/epr052_rl/data/opsd50k/opsd_train.jsonl}
OUT=${OUT:-/data/runs/epr052_rl/opsd_full/run_$(date +%Y%m%d_%H%M%S)}; mkdir -p "$OUT"
if [ -n "${STEPS:-}" ]; then LEN=(--max_steps "$STEPS"); else LEN=(--num_train_epochs "${EPOCHS:-1}"); fi
VR_PROBE_OUT="$OUT/probe_segments.jsonl" CUDA_VISIBLE_DEVICES=${GPU:-1} swift rlhf --rlhf_type gkd \
  --model "$MODEL" --model_type qwen3_vl --template qwen3_vl \
  --dataset "$DATA" --split_dataset_ratio 0 --dataset_num_proc 4 --dataloader_num_workers 1 \
  --tuner_type full --freeze_vit true --freeze_aligner true --torch_dtype bfloat16 --bf16 true --fp16 false \
  --optim adamw_bnb_8bit --adam_beta1 0.9 --adam_beta2 0.999 --learning_rate "${LR:-5e-6}" \
  --teacher_model_server "${TEACHER_URL:-http://127.0.0.1:8100}" --gkd_logits_topk "${TOPK:-64}" \
  --beta 0 --lmbda 1.0 --sft_alpha 0 --temperature 1.0 \
  --max_length "${MAXLEN:-4096}" --max_completion_length 2048 --truncation_strategy delete \
  --gradient_checkpointing true --attn_impl sdpa \
  --per_device_train_batch_size "${PB:-1}" --gradient_accumulation_steps "${GA:-8}" \
  --use_vllm true --vllm_mode server --vllm_server_host 127.0.0.1 --vllm_server_port "${ROLLOUT_PORT:-8000}" \
  "${LEN[@]}" --logging_steps 1 --save_steps "${SAVE:-200}" --save_total_limit "${KEEP:-2}" --save_only_model true \
  --external_plugins /workspace/VeraRetouch/veraretouch_sprf/rl/plugins/gkd_probe_v7.py \
  --report_to none --seed 20260906 --output_dir "$OUT" "$@" 2>&1 | tee "$OUT/run.log"
