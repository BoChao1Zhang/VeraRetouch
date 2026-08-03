#!/usr/bin/env bash
# 补批附加 viz：**新覆盖的 l5/l6** 的成功/失败案例（证明零样本 build 上的图像形态正常）。
set -eu
cd /home/bc/VeraRetouch
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/home/bc/VeraRetouch
E=experiments/G2_oracle_ceiling_20260803
T=$E/metrics_strat/per_image_real_l5l6.jsonl
python3 - "$E/metrics_strat/per_image_real_strat.jsonl" "$T" <<'PY'
import json, sys
keep = ("prod-l5-local17k-20260801", "prod-l6-local17k-20260801")
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
sub = [r for r in rows if r["build"] in keep]
with open(sys.argv[2], "w", encoding="utf-8") as f:
    for r in sub:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"l5/l6 子集 n={len(sub)}")
PY
python3 tools/ceiling/viz.py \
  --per-image $E/metrics/per_image_construct.jsonl "$T" $E/metrics_strat/per_image_control_strat.jsonl \
  --labels "D-CONSTRUCT(S-val)" "D-SFT-L(S-val,normal) 真实档 l5/l6" "D-SFT-G(S-val) 对照 分层" \
  --index $E/index/dsftl_sval_normal.jsonl --out-dir $E/viz/strat_l5l6 --n-samples 2
