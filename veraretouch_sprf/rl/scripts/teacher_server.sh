#!/bin/bash
# 冒烟 5 教师端：gpu0 起 vLLM 服务返回 top-k logprobs（swift deploy），供 --teacher_model_server 取教师 logprobs。
# 显存共存：vllm_gpu_memory_utilization 0.45（≈43 GB），与 headline 作业共存 < 65 GB。
set -euo pipefail; source "$(dirname "$0")/smoke_common.sh"
PORT=${PORT:-8100}; UTIL=${UTIL:-0.45}; MAXLEN=${MAXLEN:-6144}
CUDA_VISIBLE_DEVICES=${GPU:-0} swift deploy --model "$MODEL" --model_type qwen3_vl --template qwen3_vl \
  --infer_backend vllm --port "$PORT" --max_logprobs "${TOPK:-64}" \
  --vllm_gpu_memory_utilization "$UTIL" --vllm_max_model_len "$MAXLEN" --max_length "$MAXLEN" \
  --vllm_limit_mm_per_prompt '{"image": 1}' --torch_dtype bfloat16 "$@"
