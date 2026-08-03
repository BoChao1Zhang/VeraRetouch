#!/bin/bash
# Wave 5 (after wave 4): the 12k-step E1 arm -- optimisation-budget control for
# hypothesis (a).  Deferred so it does not starve the wave-1 ablation.
set -e
cd /home/bc/VeraRetouch
D=experiments/A0_glut_repro_20260803
while ! grep -q "WAVE4_DONE" $D/logs/wave4.log 2>/dev/null; do sleep 30; done
export CUDA_VISIBLE_DEVICES=0
python3 -m model.glut_repro.run_e1_on_a0 \
  --lut-list $D/config/a0_luts_75.txt --steps 12000 --batch-luts 16 \
  --outdir $D/hypA --tag e1_on_a0_12k
echo "WAVE5_DONE"
