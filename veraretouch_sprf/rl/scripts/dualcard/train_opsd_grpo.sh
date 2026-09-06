#!/bin/bash
# gpu1：OPSD 经 GRPO 路径（OPD-RL）：优势 = 教师 k1 log-ratio（teacher_kl_coef），num_generations 1，beta 0（无参考模型），
# rollout = vLLM server@gpu0（swift rollout，async_generate 生成/训练重叠），教师 = logprob server@gpu0（固定 S1F，采样 token logp）。
# 与 GKD 全词表口径差异：GKD 用教师全词表（或 top-k）分布做 forward KL；此路径只用采样 token 的 log π_T − log π_S（k1，反向 KL 梯度估计）。
# 用法：PB=12 GA=1 STEPS=200 bash train_opsd_grpo.sh [透传参数]
set -euo pipefail; source "$(dirname "$0")/../smoke_common.sh"
DATA=${DATA_FULL:-/data/runs/epr052_rl/data/opsd50k/opsd_train.jsonl}
OUT=${OUT:-/data/runs/epr052_rl/opsd_full/run_$(date +%Y%m%d_%H%M%S)}; mkdir -p "$OUT"
if [ -n "${STEPS:-}" ]; then LEN=(--max_steps "$STEPS"); else LEN=(--num_train_epochs "${EPOCHS:-1}"); fi
VR_PROBE_OUT="$OUT/probe_segments.jsonl" CUDA_VISIBLE_DEVICES=${GPU:-1} swift rlhf --rlhf_type grpo \
  --model "$MODEL" --model_type qwen3_vl --template qwen3_vl \
  --dataset "$DATA" --split_dataset_ratio 0 --dataset_num_proc 4 --dataloader_num_workers 1 \
  --tuner_type full --freeze_vit true --freeze_aligner true --torch_dtype bfloat16 --bf16 true --fp16 false \
  --optim adamw_bnb_8bit --adam_beta1 0.9 --adam_beta2 0.999 --learning_rate "${LR:-5e-6}" \
  --teacher_model_server "${TEACHER_URL:-http://127.0.0.1:8100}" --teacher_kl_coef "${TKL:-1.0}" \
  --num_generations 1 --beta 0 --temperature 1.0 \
  --max_length "${MAXLEN:-4096}" --max_completion_length 2048 --truncation_strategy delete \
  --gradient_checkpointing true --attn_impl sdpa \
  --per_device_train_batch_size "${PB:-12}" --gradient_accumulation_steps "${GA:-1}" \
  --use_vllm true --vllm_mode server --vllm_server_host 127.0.0.1 --vllm_server_port "${ROLLOUT_PORT:-8000}" \
  --async_generate "${ASYNC:-true}" --log_completions true --log_rollout_offpolicy_metrics true \
  "${LEN[@]}" --logging_steps 1 --save_steps "${SAVE:-200}" --save_total_limit "${KEEP:-2}" --save_only_model true \
  --external_plugins /workspace/VeraRetouch/veraretouch_sprf/rl/plugins/gkd_segment_probe.py \
  --report_to none --seed 20260906 --output_dir "$OUT" "$@" 2>&1 | tee "$OUT/run.log"
