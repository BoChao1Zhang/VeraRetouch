# EPR-052 ENG-2 冒烟公共变量（容器内）。source 之。
export PYTHONPATH=/workspace/VeraRetouch
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MODELSCOPE_CACHE=/data/modelscope_cache HF_HOME=/data/hf_home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
MODEL=${MODEL:-/data/runs/epr052_rl/s1f_epoch1_merged}          # S1F-FULL ckpt_epoch1（单目录，含 6 行阶段 token）
DATA=${DATA:-/data/runs/epr052_rl/smoke/opsd_smoke8.jsonl}      # 8 条 train 记录（spec-5 几何 y 图）
ROOT=${ROOT:-/data/runs/epr052_rl/smoke}
COMMON_ARGS=(
  --model "$MODEL" --model_type qwen3_vl --template qwen3_vl
  --dataset "$DATA" --split_dataset_ratio 0 --dataset_num_proc 1 --dataloader_num_workers 1
  --torch_dtype bfloat16 --attn_impl sdpa --gradient_checkpointing true
  --freeze_vit true --freeze_aligner true
  --max_length 4096 --max_completion_length 2048 --truncation_strategy delete
  --temperature 1.0 --max_steps ${STEPS:-2} --logging_steps 1 --save_steps 1000 --save_only_model true
  --report_to none --seed 20260906 --learning_rate 1e-5
  --external_plugins /workspace/VeraRetouch/veraretouch_sprf/rl/plugins/gkd_segment_probe.py
)
LORA_ARGS=(--tuner_type lora --lora_rank 16 --lora_alpha 16 --lora_dropout 0 --target_modules q_proj k_proj v_proj o_proj)
