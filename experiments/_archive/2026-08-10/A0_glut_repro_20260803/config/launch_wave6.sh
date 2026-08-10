#!/bin/bash
# Wave 6 (after wave 1): the converged rec-arm number on the FULL 75-LUT set.
# hypothesis (b) on the 9-LUT subset showed 20 ep -> 40 ep is worth ~+1 dB, so
# the headline 41.92 dB is an UNDER-CONVERGED number, not the recipe's ceiling.
# This run produces the converged headline the anchor judgement should use.
set -e
cd /home/bc/VeraRetouch
D=experiments/A0_glut_repro_20260803
while ! grep -q "ABLATE_WAVE1_DONE" $D/logs/ablate_wave1.log 2>/dev/null; do sleep 30; done
export CUDA_VISIBLE_DEVICES=0
python3 -m model.glut_repro.run_a0 \
  --lut-list $D/config/a0_luts_75.txt --arm rec --epochs 60 \
  --outdir $D/runs/rec_60ep --chunk-size 25 --gt-workers 8 --de-workers 8
echo "WAVE6_DONE"
