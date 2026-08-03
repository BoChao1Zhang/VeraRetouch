#!/bin/bash
# Wave 4 (starts after wave 3): bounded-opacity arms.
# The dossier-2.4 ruling `opacity = raw parameter + clamp[0,1]` makes opacity a
# one-way trapdoor (torch clamp backward is 0 outside the range), so primitives
# that fall below 0 are dead forever.  Measured alive_frac in the rec arm is
# 0.854.  These arms replace it with a bounded sigmoid (init logit 4.0).
set -e
cd /home/bc/VeraRetouch
D=experiments/A0_glut_repro_20260803
while ! grep -q "WAVE3_DONE" $D/logs/wave3.log 2>/dev/null; do sleep 30; done
export CUDA_VISIBLE_DEVICES=0
python3 -m model.glut_repro.ablate_a0 \
  --lut-list $D/config/ablate_luts_9.txt --arms opac_sig,opac_sig_g0 \
  --epochs 20 --outdir $D/ablate --tag ablate_opacity
echo "WAVE4_DONE"
