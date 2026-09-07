#!/bin/bash
# EPR-052 C4：GRPO + 双奖励（latent 余弦 + 执行器 linf8），纯奖励 RL（**不设 teacher_model**，--beta 0 无参考模型）。
# gpu1 训练；gpu0 = swift rollout（vLLM server）。用法：STEPS=200 PB=8 GA=1 bash train_grpo_c4.sh [透传参数]
set -euo pipefail; source "$(dirname "$0")/../smoke_common.sh"
DATA=${DATA_FULL:-/data/runs/epr052_rl/data/opsd50k/opsd_train.jsonl}
OUT=${OUT:-/data/runs/epr052_rl/c4/run_$(date +%Y%m%d_%H%M%S)}; mkdir -p "$OUT"
if [ -n "${STEPS:-}" ]; then LEN=(--max_steps "$STEPS"); else LEN=(--num_train_epochs "${EPOCHS:-1}"); fi
VR_REWARD_OUT="$OUT/reward_log.jsonl" VR_PROBE_OUT="$OUT/probe_segments.jsonl" \
CUDA_VISIBLE_DEVICES=${GPU:-1} swift rlhf --rlhf_type grpo \
  --model "$MODEL" --model_type qwen3_vl --template qwen3_vl \
  --dataset "$DATA" --split_dataset_ratio 0 --dataset_num_proc 4 --dataloader_num_workers 1 \
  --tuner_type full --freeze_vit true --freeze_aligner true --torch_dtype bfloat16 --bf16 true --fp16 false \
  --optim adamw_bnb_8bit --adam_beta1 0.9 --adam_beta2 0.999 --learning_rate "${LR:-5e-6}" --warmup_steps "${WARMUP:-100}" \
  --external_plugins /workspace/VeraRetouch/veraretouch_sprf/rl/plugins/c4_reward.py \
  --reward_funcs vr_latent vr_executor --reward_weights "${W_LAT:-1.0}" "${W_EXE:-1.0}" \
  --num_generations "${G:-8}" --loss_type "${LOSS_TYPE:-grpo}" --scale_rewards "${SCALE:-group}" \
  --beta 0 --temperature 1.0 --top_p 1.0 \
  --max_length "${MAXLEN:-4096}" --max_completion_length 2048 --truncation_strategy delete \
  --gradient_checkpointing true --attn_impl sdpa \
  --per_device_train_batch_size "${PB:-8}" --gradient_accumulation_steps "${GA:-1}" \
  --use_vllm true --vllm_mode server --vllm_server_host 127.0.0.1 --vllm_server_port "${ROLLOUT_PORT:-8000}" \
  --async_generate "${ASYNC:-false}" --log_completions true --log_rollout_offpolicy_metrics true \
  "${LEN[@]}" --logging_steps 1 --save_steps "${SAVE:-50}" --save_total_limit "${KEEP:-2}" --save_only_model true \
  --report_to none --seed 20260907 --output_dir "$OUT" "$@" 2>&1 | tee "$OUT/run.log"
