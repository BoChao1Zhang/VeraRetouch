#!/bin/bash
# held-out d6 headline 提交：1,464 键双卡生成（vlmsft/submit_heldout_headline.sh 的主线包版，只改脚本路径）。
# usage: submit_heldout_headline.sh <adapt_ckpt> <tag_prefix> <mem_peak_gib>
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
S=$(cd "$(dirname "$0")" && pwd)
R=${VR_RUNS_VLM:-/home/bc/data/runs/epr051_vlmsft}
SP=$R/heldout_d6_split
ADAPT=${1:?adapt ckpt}; TAG=${2:?tag prefix}; MEM=${3:-22}
OUT=$R/heldout_$TAG
read u0 u1 < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')
for g in 0 1; do
  u=$([ $g -eq 0 ] && echo $u0 || echo $u1)
  gib=$(echo "scale=2; $u/1024" | bc)
  ok=$(echo "$gib < 65 && $gib + $MEM <= 80" | bc)
  echo "[gate] gpu$g used=${gib}GiB declared=${MEM}GiB pass=$ok"
  [ "$ok" = "1" ] || { echo "FATAL: gpu$g fails the memory gate"; exit 1; }
done
set +o noclobber
rm -f $OUT/a.log $OUT/a.log.pid $OUT/b.log $OUT/b.log.pid
mkdir -p "$OUT"
q submit SPRF_VLMADAPT_${TAG}_A 0 "$OUT/a.log" --desc "heldout-d6 half A (732) gen_batch16 max_new2048" \
  --mem-peak "$MEM" --ready 'dump:START' --ready-timeout 1800 \
  -- bash $S/dump_half.sh $SP/half_a.json ${TAG}a $OUT/out_a $ADAPT 16 2048
q submit SPRF_VLMADAPT_${TAG}_B 1 "$OUT/b.log" --desc "heldout-d6 half B (732) gen_batch16 max_new2048" \
  --mem-peak "$MEM" --ready 'dump:START' --ready-timeout 1800 \
  -- bash $S/dump_half.sh $SP/half_b.json ${TAG}b $OUT/out_b $ADAPT 16 2048
echo "[submit] both halves queued; when BOTH are Done run:"
echo "  bash $S/heldout_final.sh $OUT/out_a/gencache_${TAG}a.pt $OUT/out_b/gencache_${TAG}b.pt ${TAG}m $OUT/final $ADAPT $SP/manifest.json 2048"
