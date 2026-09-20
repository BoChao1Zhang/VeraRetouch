#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=/home/bc/VeraRetouch
export CUDA_VISIBLE_DEVICES=0
export HF_TOKEN=
export HUGGING_FACE_HUB_TOKEN=
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export PYTHONUNBUFFERED=1
cd /home/bc/VeraRetouch
for mode in nocot cot; do
  if [ "$mode" = cot ]; then
    cot_summary=/home/bc/nfsvfs/bc/data/runs/epr072_sixstage_compare_20260921/with_cot/artedit/summary.json
    deadline=$((SECONDS + 21600))
    while [ ! -f "$cot_summary" ]; do
      if [ "$SECONDS" -ge "$deadline" ]; then
        echo 'CoT render did not finish within six hours' >&2
        exit 1
      fi
      sleep 60
    done
  fi
  /home/bc/data/external/qalign/venv/bin/python -m tools.epr072_eval.six_stage_score --mode "$mode" --lane qalign
  /home/bc/data/external/qalign/venv/bin/python -m tools.epr072_eval.six_stage_score --mode "$mode" --lane deqa
  /home/bc/data/external/artimuse/venv/bin/python -m tools.epr072_eval.six_stage_score --mode "$mode" --lane artimuse
done
