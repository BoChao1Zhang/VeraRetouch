#!/bin/bash
# EPR-052 诊断臂驱动：跑一个臂（默认 60 步），逐步监控探针；若前 30 步内六段有序率连续 5 步 0/8 则提前停并记停点。
# usage: ARM=arm1_lr1e6 STEPS=60 EXTRA="--learning_rate 1e-6 --warmup_steps 100" bash run_arm.sh
set -uo pipefail
ARM=${ARM:?arm name}; STEPS=${STEPS:-60}; PB=${PB:-8}; GA=${GA:-1}
TRAIN=${TRAIN_SCRIPT:-train_opsd_full.sh}
R=/home/bc/data/runs/epr052_rl/arms/$ARM; mkdir -p "$R"
cd /home/bc/VeraRetouch/docker/ms-swift
(nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader -l 5 >| "$R/nvsmi.log" 2>&1 &)
(docker compose exec -T -e GPU=1 -e OUT=/data/runs/epr052_rl/arms/$ARM -e STEPS=$STEPS -e SAVE=100000 -e PB=$PB -e GA=$GA -e DATA_FULL="${DATA_FULL:-}" -e LR="${LR:-}" \
   swift bash /workspace/VeraRetouch/veraretouch_sprf/rl/scripts/dualcard/$TRAIN ${EXTRA:-} >| "$R/host.log" 2>&1 &)
echo "[arm $ARM] launched: STEPS=$STEPS PB=$PB EXTRA=${EXTRA:-none} TRAIN=$TRAIN"
P="$R/probe_segments.jsonl"; zero=0; last=-1; stop=""
while true; do
  if [ -f "$P" ]; then
    n=$(grep -c '"six_complete_rows"' "$P" 2>/dev/null || echo 0)   # 只数 seg 行（v4+ 每步另写 loss/log/grad 行）
    if [ "$n" -gt "$last" ]; then
      last=$n
      z=$(python3 -c "
import json,sys
rows=[json.loads(l) for l in open('$P') if '"six_complete_rows"' in l]
run=0; stop=None
for r in rows:
    run = run+1 if r.get('six_complete_rows')==0 else 0
    if run>=5 and r['step']<30 and stop is None: stop=r['step']
print(stop if stop is not None else -1)")
      if [ "$z" != "-1" ]; then stop=$z; echo "[arm $ARM] EARLY STOP: 六段有序率连续 5 步 0/8，停点 step=$z"; break; fi
      [ "$n" -ge "$STEPS" ] && { echo "[arm $ARM] reached $STEPS steps"; break; }
    fi
  fi
  docker compose exec -T swift pgrep -f "[r]lhf.py" >/dev/null 2>&1 || { sleep 30; docker compose exec -T swift pgrep -f "[r]lhf.py" >/dev/null 2>&1 || { echo "[arm $ARM] trainer exited"; break; }; }
  sleep 30
done
docker compose exec -T swift bash -c 'for p in $(pgrep -f "[r]lhf.py"); do kill $p; done' >/dev/null 2>&1
sleep 8
for p in $(pgrep -f 'nvidia-smi --query-gpu=[t]imestamp'); do kill $p; done
echo "${stop:-none}" > "$R/early_stop_step.txt"
echo "[arm $ARM] finished; seg rows: $(grep -c '"six_complete_rows"' "$P" 2>/dev/null || echo 0); early_stop=${stop:-none}"
