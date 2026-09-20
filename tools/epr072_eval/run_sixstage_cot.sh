#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=/home/bc/miniconda3/envs/llm_factory/lib
export PYTHONPATH=/home/bc/VeraRetouch
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
cd /home/bc/VeraRetouch
sixstage_python=/home/bc/envs/q3vl_sft/bin/python
sixstage_root=/home/bc/nfsvfs/bc/data/runs/epr072_sixstage_compare_20260921
paired_root=/home/bc/nfsvfs/bc/data/runs/epr072_sixstage_public_20260920/full
"$sixstage_python" -m tools.epr072_eval.public_six_stage --bench artedit \
  --out "$sixstage_root/with_cot" --text-mode cot --paired-root "$paired_root"
"$sixstage_python" -m tools.epr072_eval.check_public_six_stage "$sixstage_root/with_cot/artedit"
systemd-run --user --unit=epr072-sixstage-cot-judge --collect \
  --property=MemoryMax=8G --property=WorkingDirectory=/home/bc/VeraRetouch \
  --setenv=PYTHONPATH=/home/bc/VeraRetouch \
  --setenv=LD_LIBRARY_PATH=/home/bc/miniconda3/envs/llm_factory/lib \
  "$sixstage_python" -u -m tools.epr072_eval.six_stage_score --mode cot --lane viescore
"$sixstage_python" -m tools.epr072_eval.six_stage_journals --out "$sixstage_root/journals"
"$sixstage_python" -m tools.epr072_eval.public_six_stage --bench fivek ppr10k \
  --out "$sixstage_root/with_cot" --text-mode cot --paired-root "$paired_root" \
  --journal-root "$sixstage_root/journals"
for bench in fivek ppr10k; do
  "$sixstage_python" -m tools.epr072_eval.check_public_six_stage "$sixstage_root/with_cot/$bench"
done
