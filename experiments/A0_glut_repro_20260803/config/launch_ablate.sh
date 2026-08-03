#!/bin/bash
# A0 gap attribution wave-1: full-arm ingredient isolation (task 1).
# 9 stratified LUTs (spanning the rec-arm PSNR range) x 20 epochs, paired.
set -e
cd /home/bc/VeraRetouch
export CUDA_VISIBLE_DEVICES=0
D=experiments/A0_glut_repro_20260803
python3 -m model.glut_repro.ablate_a0 \
  --lut-list $D/config/ablate_luts_9.txt \
  --arms rec,hc,sparse,mining,full,hc_fix,g0,g0_full \
  --epochs 20 --outdir $D/ablate --tag ablate_20ep
echo "ABLATE_WAVE1_DONE"
