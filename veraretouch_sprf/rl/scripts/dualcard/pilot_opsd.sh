#!/bin/bash
# OPSD 全参双卡 pilot 驱动（宿主）：起容器内训练（gpu1），每出现 checkpoint-N 即在 gpu0 跑在线判据并追加一行摘要。
# usage: STEPS=100 SAVE=25 bash pilot_opsd.sh <run_name>
set -uo pipefail
NAME=${1:?run name}; ROOT=/home/bc/data/runs/epr052_rl/opsd_full/$NAME; mkdir -p "$ROOT/online"
S=/home/bc/VeraRetouch/veraretouch_sprf/rl/scripts/dualcard
cd /home/bc/VeraRetouch/docker/ms-swift
(docker compose exec -T -e GPU=1 -e OUT=/data/runs/epr052_rl/opsd_full/$NAME -e STEPS=${STEPS:-} -e EPOCHS=${EPOCHS:-1} -e MAXLEN=${MAXLEN:-4096} -e SAVE=${SAVE:-200} -e KEEP=${KEEP:-2} -e PB=${PB:-1} -e GA=${GA:-8} swift bash /workspace/VeraRetouch/veraretouch_sprf/rl/scripts/dualcard/${TRAIN_SCRIPT:-train_opsd_grpo.sh} >| "$ROOT/train_host.log" 2>&1 &)
(nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,noheader -l 10 >| "$ROOT/nvsmi.log" 2>&1 &)
echo "[pilot] training launched; monitoring checkpoints under $ROOT"
done_ck=""
# step-0 基线（S1F 原权重）
[ -f "$ROOT/online/online_eval.jsonl" ] && grep -q '"tag": "step0"' "$ROOT/online/online_eval.jsonl" || bash $S/online_eval.sh /home/bc/data/runs/epr052_rl/s1f_epoch1_merged step0 "$ROOT/online" 2>&1 | grep ONLINE_EVAL
while true; do
  for ck in $(find "$ROOT" -maxdepth 3 -type d -name 'checkpoint-*' | sort -t- -k2 -n); do
    n=$(basename "$ck" | cut -d- -f2)
    case " $done_ck " in *" $n "*) continue;; esac
    [ $((n % ${EVAL_EVERY:-200})) -eq 0 ] || { done_ck="$done_ck $n"; continue; }
    [ -f "$ck/config.json" ] && ls "$ck"/*.safetensors >/dev/null 2>&1 || continue
    sleep 30   # 等 safetensors 写完
    bash $S/online_eval.sh "$ck" "step$n" "$ROOT/online" 2>&1 | grep ONLINE_EVAL
    done_ck="$done_ck $n"
  done
  if ! docker compose exec -T swift pgrep -f "output_dir /data/runs/epr052_rl/opsd_full/$NAME" >/dev/null 2>&1; then
    sleep 60; if ! docker compose exec -T swift pgrep -f "output_dir /data/runs/epr052_rl/opsd_full/$NAME" >/dev/null 2>&1; then echo "[pilot] trainer exited"; break; fi
  fi
  sleep 60
done
pkill -f 'nvidia-smi --query-gpu=timestamp' ; echo "[pilot] done"
