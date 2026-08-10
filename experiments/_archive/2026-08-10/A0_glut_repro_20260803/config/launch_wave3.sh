#!/bin/bash
# Wave 3:
#  (1) capacity sweep on the A0 corpus -- what N does OUR corpus need to reach
#      the 45.47 dB that GLUT-32 reaches on theirs?  (hypothesis c, quantified)
#  (2) ablation wave-2 -- the repaired L_hc (chroma floor + gradient-calibrated
#      lambda) and the repaired full arm.
set -e
cd /home/bc/VeraRetouch
export CUDA_VISIBLE_DEVICES=0
D=experiments/A0_glut_repro_20260803
for N in 48 64; do
  python3 -m model.glut_repro.run_e1_on_a0 \
    --lut-list $D/config/a0_luts_75.txt --n $N --steps 3000 --batch-luts 16 \
    --outdir $D/hypA --tag e1_on_a0_N${N}
done
python3 -m model.glut_repro.ablate_a0 \
  --lut-list $D/config/ablate_luts_9.txt --arms hc_cal,full_fix,hc_w1 \
  --epochs 20 --outdir $D/ablate --tag ablate_fix
echo "WAVE3_DONE"
