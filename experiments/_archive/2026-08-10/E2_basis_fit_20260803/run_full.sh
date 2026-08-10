#!/bin/bash
set -e
cd /home/bc/VeraRetouch/experiments/E2_basis_fit_20260803
echo "[$(date +%F\ %T)] prep start"
CUDA_VISIBLE_DEVICES=0 python3 prep_data.py
echo "[$(date +%F\ %T)] fit start"
nice -n 10 python3 run_fit.py --workers 36
echo "[$(date +%F\ %T)] analyze start"
python3 analyze.py
echo "[$(date +%F\ %T)] ALL DONE"
touch FULL_RUN_DONE
