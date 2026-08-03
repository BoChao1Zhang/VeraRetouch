#!/bin/bash
# Re-run of the 12k-step E1 arm (the first attempt died on a pred_tmp collision
# with the concurrent wave-3 job; pred_tmp is now tag-scoped).
set -e
cd /home/bc/VeraRetouch
export CUDA_VISIBLE_DEVICES=0
D=experiments/A0_glut_repro_20260803
python3 -m model.glut_repro.run_e1_on_a0 \
  --lut-list $D/config/a0_luts_75.txt --steps 12000 --batch-luts 16 \
  --outdir $D/hypA --tag e1_on_a0_12k
echo "HYPA12K_DONE"
