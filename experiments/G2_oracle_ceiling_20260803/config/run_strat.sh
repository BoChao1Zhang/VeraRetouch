#!/usr/bin/env bash
# G2 补批（D-34）：**只改抽样**——按 build 分层均衡取样，覆盖 l1..l6 / g1..g4 全部。
# 判据、估计器（逐箱最小二乘仿射，D-24）、donor 池构造一律沿用原批次，不动一行。
# 只用卡 1（CUDA_VISIBLE_DEVICES=1）。
set -u
cd /home/bc/VeraRetouch
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/home/bc/VeraRetouch
E=experiments/G2_oracle_ceiling_20260803
L=$E/logs
M=$E/metrics_strat
IDX_L=$E/index/dsftl_sval_normal.jsonl
IDX_G=$E/index/dsftg_sval_normal.jsonl
PB_REAL=${PB_REAL:-100}      # 6 build × 100 = 600（判据要求 ≥600，每 build ≥80）
PB_CTRL=${PB_CTRL:-150}      # 4 build × 150 = 600
mkdir -p "$M" "$L"

export E
step() { echo "=== [$(date +%H:%M:%S)] $* ==="; }
upd() { python3 - "$1" "$2" <<'PY'
import sys, os, datetime
open(os.environ["E"] + "/STATUS.md", "a").write(
    f"- {datetime.datetime.now():%Y-%m-%d %H:%M:%S}  [补批] {sys.argv[1]}: {sys.argv[2]}\n")
PY
}

step "① 真实档 D-SFT-L 分层 per_build=$PB_REAL"; upd strat_real start
python3 tools/ceiling/run_analytic.py --track real --index $IDX_L \
  --donor-index $IDX_L --per-build $PB_REAL --tag _strat --out $M \
  > $L/strat_real.log 2>&1
upd strat_real "rc=$?"

step "② 对照档 D-SFT-G 分层 per_build=$PB_CTRL"; upd strat_control start
python3 tools/ceiling/run_analytic.py --track control --index $IDX_G \
  --donor-index $IDX_L --per-build $PB_CTRL --tag _strat --out $M \
  > $L/strat_control.log 2>&1
upd strat_control "rc=$?"

step "③ 补齐 O-3 的附录口径案例图（--key delta，原批次 per_image）"; upd viz_delta start
python3 tools/ceiling/viz.py \
  --per-image $E/metrics/per_image_construct.jsonl \
              $E/metrics/per_image_real.jsonl \
              $E/metrics/per_image_control.jsonl \
  --labels "D-CONSTRUCT(S-val)" "D-SFT-L(S-val,normal) 真实档" "D-SFT-G(S-val) 对照" \
  --index $IDX_L --out-dir $E/viz/appendixA_delta --key delta \
  > $L/strat_viz_delta.log 2>&1
upd viz_delta "rc=$?"

step "ALL DONE"; upd all done
