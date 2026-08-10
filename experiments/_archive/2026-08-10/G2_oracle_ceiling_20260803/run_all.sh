#!/usr/bin/env bash
# G2 全量长任务驱动（**只用卡 1**）。顺序：解析法三档 → MLP 容量探针 → GLUT 实验法互证 → viz。
set -u
cd /home/bc/VeraRetouch
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/home/bc/VeraRetouch
E=experiments/G2_oracle_ceiling_20260803
L=$E/logs
M=$E/metrics
IDX_L=$E/index/dsftl_sval_normal.jsonl
IDX_G=$E/index/dsftg_sval_normal.jsonl
N_REAL=${N_REAL:-600}
N_CTRL=${N_CTRL:-600}
N_XCHECK=${N_XCHECK:-100}

step() { echo "=== [$(date +%H:%M:%S)] $* ==="; }
upd() { python3 - "$1" "$2" <<'PY'
import json,sys,os,datetime
p=os.environ["E"]+"/STATUS.md"
open(p,"a").write(f"- {datetime.datetime.now():%Y-%m-%d %H:%M:%S}  {sys.argv[1]}: {sys.argv[2]}\n")
PY
}
export E

step "① D-CONSTRUCT(S-val) 全 8 级"; upd construct start
python3 tools/ceiling/run_analytic.py --track construct --out $M \
  --donor-index $IDX_L > $L/analytic_construct.log 2>&1
upd construct "rc=$?"

step "② 真实档 D-SFT-L(S-val,normal) n=$N_REAL"; upd real start
python3 tools/ceiling/run_analytic.py --track real --index $IDX_L \
  --donor-index $IDX_L --limit $N_REAL --out $M > $L/analytic_real.log 2>&1
upd real "rc=$?"

step "③ 对照档 D-SFT-G(S-val,normal) + 移植掩膜 n=$N_CTRL"; upd control start
python3 tools/ceiling/run_analytic.py --track control --index $IDX_G \
  --donor-index $IDX_L --limit $N_CTRL --out $M > $L/analytic_control.log 2>&1
upd control "rc=$?"

step "D0-4 MLP 容量探针 (L1/L4 各 3)"; upd mlp start
python3 tools/ceiling/mlp_probe.py --steps 6000 \
  --out $M/mlp_probe.json > $L/mlp_probe.log 2>&1
upd mlp "rc=$?"

step "实验法互证 GLUT 三臂 n=$N_XCHECK"; upd xcheck start
python3 tools/ceiling/fit_crosscheck.py --index $IDX_L --limit $N_XCHECK \
  --steps 1200 --out $M/glut_crosscheck.json > $L/glut_crosscheck.log 2>&1
upd xcheck "rc=$?"

step "viz"; upd viz start
python3 tools/ceiling/viz.py \
  --per-image $M/per_image_construct.jsonl $M/per_image_real.jsonl $M/per_image_control.jsonl \
  --labels "D-CONSTRUCT(S-val)" "D-SFT-L(S-val,normal) 真实档" "D-SFT-G(S-val) 对照" \
  --index $IDX_L --out-dir $E/viz > $L/viz.log 2>&1
upd viz "rc=$?"
python3 tools/ceiling/viz.py \
  --per-image $M/per_image_construct.jsonl $M/per_image_real.jsonl $M/per_image_control.jsonl \
  --labels "D-CONSTRUCT(S-val)" "D-SFT-L(S-val,normal) 真实档" "D-SFT-G(S-val) 对照" \
  --index $IDX_L --out-dir $E/viz --key delta >> $L/viz.log 2>&1

step "汇总 metrics.json"; upd summary start
python3 tools/ceiling/summarize.py --exp-dir $E > $L/summarize.log 2>&1
upd summary "rc=$?"
step "ALL DONE"; upd all done
