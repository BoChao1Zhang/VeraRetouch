#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=/home/bc/miniconda3/envs/llm_factory/lib
export PYTHONPATH=/home/bc/VeraRetouch
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
cd /home/bc/VeraRetouch
sixstage_python=/home/bc/envs/q3vl_sft/bin/python
sixstage_root=/home/bc/nfsvfs/bc/data/runs/epr072_sixstage_public_20260920/full
"$sixstage_python" -m tools.epr072_eval.public_six_stage \
  --bench artedit fivek ppr10k --out "$sixstage_root" --mem-fraction .28
for bench in artedit fivek ppr10k; do
  "$sixstage_python" -m tools.epr072_eval.check_public_six_stage "$sixstage_root/$bench"
done
