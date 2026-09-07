#!/bin/bash
# EPR-052 OPSD pilot 驱动 v2（新文件；v1 不动）。
# 训练在 gpu1；每 SAVE(=50) 步落一个滚动 ckpt（save_total_limit=2，仅供在线判据用），
# 每 EVAL_EVERY(=50) 步在 gpu0 跑 32 键在线余弦；每 KEEP_EVERY(=200) 步把该 ckpt 复制进 keep/ 长期保留（只留最近 KEEP_N 个）。
# usage: STEPS=200 bash pilot_opsd_v2.sh <run_name> [EXTRA...]
set -uo pipefail
NAME=${1:?run name}; shift || true
ROOT=/home/bc/data/runs/epr052_rl/opsd_full/$NAME; mkdir -p "$ROOT/online" "$ROOT/keep"
S=/home/bc/VeraRetouch/veraretouch_sprf/rl/scripts/dualcard
SAVE=${SAVE:-50}; EVAL_EVERY=${EVAL_EVERY:-50}; KEEP_EVERY=${KEEP_EVERY:-200}; KEEP_N=${KEEP_N:-2}
cd /home/bc/VeraRetouch/docker/ms-swift
(docker compose exec -T -e GPU=1 -e OUT=/data/runs/epr052_rl/opsd_full/$NAME -e STEPS=${STEPS:-} -e EPOCHS=${EPOCHS:-1} \
   -e SAVE="$SAVE" -e KEEP=2 -e PB=${PB:-8} -e GA=${GA:-1} -e LR=${LR:-} -e DATA_FULL=${DATA_FULL:-} \
   swift bash /workspace/VeraRetouch/veraretouch_sprf/rl/scripts/dualcard/${TRAIN_SCRIPT:-train_opsd_full_v7.sh} "$@" >| "$ROOT/train_host.log" 2>&1 &)
(nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader -l 15 >| "$ROOT/nvsmi.log" 2>&1 &)
echo "[pilot2] $NAME launched (save/eval every $SAVE/$EVAL_EVERY, keep every $KEEP_EVERY)"
done_ck=""
while true; do
  for ck in $(find "$ROOT" -maxdepth 3 -type d -name 'checkpoint-*' 2>/dev/null | sort -t- -k2 -n); do
    n=$(basename "$ck" | cut -d- -f2)
    case " $done_ck " in *" $n "*) continue;; esac
    [ -f "$ck/config.json" ] && ls "$ck"/*.safetensors >/dev/null 2>&1 || continue
    [ $((n % EVAL_EVERY)) -eq 0 ] || { done_ck="$done_ck $n"; continue; }
    sleep 25   # 等权重写完
    if [ $((n % KEEP_EVERY)) -eq 0 ]; then
      cp -r "$ck" "$ROOT/keep/checkpoint-$n" 2>/dev/null && echo "[pilot2] kept checkpoint-$n"
      ls -1dt "$ROOT"/keep/checkpoint-* 2>/dev/null | tail -n +$((KEEP_N+1)) | xargs -r rm -rf
    fi
    bash $S/online_eval.sh "$ck" "step$n" "$ROOT/online" 2>&1 | grep ONLINE_EVAL
    done_ck="$done_ck $n"
  done
  if ! docker compose exec -T swift bash -c "ps -eo cmd | grep -q '[o]utput_dir /data/runs/epr052_rl/opsd_full/$NAME'" 2>/dev/null; then
    sleep 60
    docker compose exec -T swift bash -c "ps -eo cmd | grep -q '[o]utput_dir /data/runs/epr052_rl/opsd_full/$NAME'" 2>/dev/null || { echo "[pilot2] trainer exited"; break; }
  fi
  sleep 45
done
for p in $(pgrep -f 'nvidia-smi --query-gpu=[t]imestamp'); do kill $p; done
echo "[pilot2] done"
