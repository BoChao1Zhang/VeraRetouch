#!/bin/bash
# Hypothesis (b): epoch budget.  Same 9 stratified LUTs as the ablation, so
# rec@20ep (wave-1) / rec@40ep / rec@60ep are a paired epoch sweep.
set -e
cd /home/bc/VeraRetouch
export CUDA_VISIBLE_DEVICES=0
D=experiments/A0_glut_repro_20260803
python3 -m model.glut_repro.ablate_a0 \
  --lut-list $D/config/ablate_luts_9.txt --arms rec \
  --epochs 40 --outdir $D/ablate --tag rec_40ep
python3 -m model.glut_repro.ablate_a0 \
  --lut-list $D/config/ablate_luts_9.txt --arms rec \
  --epochs 60 --outdir $D/ablate --tag rec_60ep
echo "HYPB_DONE"
