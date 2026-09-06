#!/bin/bash
# gpu0：学生 rollout vLLM 服务（swift rollout）。用法：UTIL=0.47 PORT=8000 bash rollout_server.sh
set -euo pipefail; source "$(dirname "$0")/../smoke_common.sh"
CUDA_VISIBLE_DEVICES=${GPU:-0} swift rollout --model "$MODEL" --model_type qwen3_vl --template qwen3_vl \
  --port "${PORT:-8000}" --vllm_gpu_memory_utilization "${UTIL:-0.47}" --vllm_max_model_len "${MAXLEN:-4096}" \
  --vllm_limit_mm_per_prompt '{"image": 1}' --torch_dtype bfloat16 "$@"
