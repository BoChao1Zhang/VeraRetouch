#!/usr/bin/env bash
# Two-card production of the generated <where>/<color> context (WT-J10 / Where-B S2).
#
#   usage:  genctx_dual.sh <checkpoint_path>
#
# One driver, one shard queue per GPU.  Each queue entry is a separate
# `genctx_shard.py` process that publishes its own atomic shard root; when every
# shard of a split is complete the driver merges them into the single root the
# consumers read (`merge_genctx.py`, CPU only).
#
# D-20 is applied PER SHARD JOB, not once for the driver:
#   1. `rm -f` the log first          (zsh noclobber turns `> existing.log` into a
#      failed redirection and the process never starts, silently)
#   2. `ps -p $PID` proves liveness   (NEVER pgrep: `pgrep -f <pat>` always matches
#      the shell running the grep -- four separate incidents in this campaign)
#   3. the log must contain real output (the producer's setup JSON, i.e. weights
#      loaded AND the shard's dataset opened) -- an existing file is not enough
#   4. only then is `<job>.job.marker` written
# `wait $pid` afterwards gives the job's exact rc, because the python process is
# a direct child of its GPU worker subshell (no setsid needed: the driver itself
# is what gets nohup'd).
#
# Resume: a shard whose published root already has `manifest.json` with
# `"status": "complete"` is skipped.  A half-written root cannot exist (the
# publisher stages under a hidden dir and renames), but if one is ever found the
# job refuses rather than deleting it (CLAUDE.md: back up, never delete).
#
# --- how to submit it (the driver itself also gets the four steps) ------------
#   cd /home/bc/VeraRetouch/experiments/Q3VL_metacanvas_where_what_20260804/where_b/genctx_jobs
#   rm -f logs/driver_two_segment.log
#   nohup setsid bash genctx_dual.sh /home/bc/data/runs/q3vl_base_sft_20260804/checkpoint-4976 \
#         > logs/driver_two_segment.log 2>&1 &
#   echo $! > logs/driver_two_segment.pid
#   ps -p "$(cat logs/driver_two_segment.pid)" -o pid,etime,cmd --no-headers
#   grep -q PROBE-OK logs/driver_two_segment.log && tail -n 40 logs/driver_two_segment.log
#
# `set -e` is deliberately absent: a failing shard must still record its rc and
# let the other card carry on.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CKPT="${1:-}"
if [ -z "${CKPT}" ]; then
  sed -n '2,8p' "${BASH_SOURCE[0]}"; echo; echo "ERROR: <checkpoint_path> is required"; exit 2
fi

# --- knobs -------------------------------------------------------------------
MODE="${MODE:-two_segment}"                 # two_segment | forced_color
# two_segment  : Where-B (train + V_where) and Stage-What T01-T08/C03-C04 (train + V_what)
# forced_color : Stage-What C01/C02 only    (train + V_what)
if [ "${MODE}" = "forced_color" ]; then
  SPLITS="${SPLITS:-V_what train}"
else
  SPLITS="${SPLITS:-V_where V_what train}"
fi
GPUS="${GPUS:-0 1}"
NUM_SHARDS="${NUM_SHARDS:-8}"               # shards for a big split (train)
SMALL_SPLIT_MAX="${SMALL_SPLIT_MAX:-5000}"  # below this, use SMALL_NUM_SHARDS
SMALL_NUM_SHARDS="${SMALL_NUM_SHARDS:-2}"
SHARD_MODE="${SHARD_MODE:-interleave}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"     # whereb/config.GEN_MAX_NEW_TOKENS
LIMIT="${LIMIT:-}"                          # calibration only; applies to the whole split
DO_MERGE="${DO_MERGE:-1}"
DRY_RUN="${DRY_RUN:-0}"
FORCE_BUSY_GPU="${FORCE_BUSY_GPU:-0}"
GPU_FREE_MIB="${GPU_FREE_MIB:-2000}"
PROBE_TIMEOUT="${PROBE_TIMEOUT:-900}"       # seconds to wait for the setup JSON

PY="${PY:-/home/bc/envs/q3vl_sft/bin/python}"
REPO="${REPO:-/home/bc/VeraRetouch}"
STAGE_ROOT="${STAGE_ROOT:-/mnt/nfs/bc/data/datasets/where_b-20260805/genwhere/_shards}"
GENCTX_DIR="${GENCTX_DIR:-/mnt/nfs/bc/data/datasets/where_b-20260805/genwhere}"
SPLIT_DIR="${SPLIT_DIR:-/mnt/nfs/bc/data/datasets/sft2seg-20260804/splits}"
LOG_DIR="${LOG_DIR:-${HERE}/logs}"
REPORT_ROOT="${REPORT_ROOT:-${HERE}/reports}"

# sqlite3 in the campaign env needs conda's libstdc++ (CXXABI_1.3.15) -- campaign
# bug R6; every published-shard read goes through it.
export LD_LIBRARY_PATH="/home/bc/miniconda3/envs/llm_factory/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TOKENIZERS_PARALLELISM=false

ts() { date -Is; }
say() { printf '[%s] %s\n' "$(ts)" "$*"; }
DRIVER_RC=0
mkdir -p "${LOG_DIR}" "${REPORT_ROOT}"

leaf_of() { if [ "${MODE}" = "two_segment" ]; then echo "$1"; else echo "$1-${MODE}"; fi; }
shards_for() { if [ "$1" -lt "${SMALL_SPLIT_MAX}" ]; then echo "${SMALL_NUM_SHARDS}"; else echo "${NUM_SHARDS}"; fi; }
n_samples_of() { wc -l < "${SPLIT_DIR}/$1.index.jsonl" | tr -d ' '; }

# how many samples shard <i> of <n> must hold for <split>, mirroring
# genctx_shard.shard_indices (and honouring LIMIT, which applies to the whole
# split before the partition).  Used to catch the one trap the resume logic
# would otherwise walk into: a LIMIT= calibration run leaves "complete" shard
# roots at the same paths, and a later full run would skip them as done.
expected_shard_n() {
  local split="$1" i="$2" n="$3" tot
  tot=$(n_samples_of "${split}")
  if [ -n "${LIMIT}" ] && [ "${LIMIT}" -lt "${tot}" ]; then tot="${LIMIT}"; fi
  if [ "${SHARD_MODE}" = "interleave" ]; then
    echo $(( (tot - i + n - 1) / n ))
  else
    echo $(( (tot * (i + 1)) / n - (tot * i) / n ))
  fi
}

# ---------------------------------------------------------------- probe ------
probe() {
  say "MODE=${MODE} SPLITS='${SPLITS}' GPUS='${GPUS}' batch=${BATCH_SIZE} max_new=${MAX_NEW_TOKENS}"
  say "checkpoint=${CKPT}"
  [ -x "${PY}" ] || { say "no interpreter at ${PY}"; return 1; }
  [ -d "${REPO}/q3vl/whereb" ] || { say "no repo at ${REPO}"; return 1; }
  local shards; shards=$(ls "${CKPT}"/model-*.safetensors 2>/dev/null | wc -l)
  [ -d "${CKPT}" ] && [ "${shards}" -gt 0 ] || { say "checkpoint unusable: ${CKPT} (${shards} weight shards)"; return 1; }
  say "checkpoint weight shards: ${shards}"

  local s n final active=""
  for s in ${SPLITS}; do
    [ -f "${SPLIT_DIR}/${s}.index.jsonl" ] || { say "missing split index ${SPLIT_DIR}/${s}.index.jsonl"; return 1; }
    n=$(n_samples_of "${s}")
    final="${GENCTX_DIR}/$(leaf_of "${s}")"
    if [ -e "${final}" ] || [ -L "${final}" ]; then
      # publication is atomic, so a manifest that says complete IS complete:
      # that split is already delivered and is dropped from this run.  Anything
      # else is an ambiguous leftover and stops the whole driver.
      if [ -f "${final}/manifest.json" ] && grep -q '"status": "complete"' "${final}/manifest.json"; then
        say "split ${s}: final root ${final} is already complete -- dropped from this run"
        continue
      fi
      say "FINAL ROOT EXISTS BUT IS NOT A COMPLETE PUBLICATION: ${final}"
      say "  move it aside (do NOT delete -- CLAUDE.md long-job discipline) before rerunning"
      return 1
    fi
    active="${active}${s} "
    say "split ${s}: ${n} samples, $(shards_for "${n}") shards, final -> ${final}"
  done
  SPLITS="${active% }"
  [ -n "${SPLITS}" ] || { say "every requested split is already published -- nothing to do"; return 1; }

  # GPUs must be idle.  The one thing this script must never do is start on a
  # card that still holds Base SFT or a chain job.  DRY_RUN only prints the plan,
  # so there a busy card is a warning -- that is the state the plan is written in.
  local g used
  for g in ${GPUS}; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${g}" 2>/dev/null | tr -d ' ')
    [ -n "${used}" ] || { say "cannot query GPU ${g}"; return 1; }
    say "GPU ${g}: ${used} MiB in use"
    if [ "${used}" -ge "${GPU_FREE_MIB}" ]; then
      if [ "${DRY_RUN}" = "1" ]; then
        say "GPU ${g} is busy (${used} MiB >= ${GPU_FREE_MIB}) -- DRY_RUN, planning anyway"
      elif [ "${FORCE_BUSY_GPU}" != "1" ]; then
        say "GPU ${g} is busy (${used} MiB >= ${GPU_FREE_MIB}); refusing to start (FORCE_BUSY_GPU=1 overrides)"
        return 1
      else
        say "GPU ${g} is busy (${used} MiB) but FORCE_BUSY_GPU=1 -- starting anyway"
      fi
    fi
  done

  mkdir -p "${STAGE_ROOT}/${MODE}" || return 1
  say "PROBE-OK genctx_dual ready: interpreter, checkpoint, ${SPLITS} indexes, free GPUs, stage root"
}

# ------------------------------------------------------------- one shard -----
# run_shard <gpu> <split> <shard-i> <num-shards>
run_shard() {
  local gpu="$1" split="$2" i="$3" n="$4"
  local tag; tag=$(printf 'shard%02dof%02d' "${i}" "${n}")
  local leaf; leaf=$(leaf_of "${split}")
  local name="${MODE}_${split}_${tag}"
  local log="${LOG_DIR}/${name}.log"
  local out_root="${STAGE_ROOT}/${MODE}/${split}"
  local published="${out_root}/${tag}/${leaf}"

  if [ -f "${published}/manifest.json" ]; then
    if grep -q '"status": "complete"' "${published}/manifest.json"; then
      local have want
      have=$(grep -o '"sample_count": *[0-9]*' "${published}/manifest.json" | head -1 | grep -o '[0-9]*$')
      want=$(expected_shard_n "${split}" "${i}" "${n}")
      if [ "${have:-0}" != "${want}" ]; then
        say "[gpu${gpu}] ${name}: complete but holds ${have} samples, not ${want}"
        say "  (a LIMIT= calibration run publishes to these same paths -- move it aside, do not delete)"
        printf '90\n' > "${LOG_DIR}/${name}.rc"; return 90
      fi
      say "[gpu${gpu}] ${name}: already complete (${have} samples) -- skipped"
      printf '0\n' > "${LOG_DIR}/${name}.rc"; return 0
    fi
    say "[gpu${gpu}] ${name}: ${published} exists but is not complete -- refusing (move it aside, do not delete)"
    printf '90\n' > "${LOG_DIR}/${name}.rc"; return 90
  elif [ -d "${published}" ]; then
    say "[gpu${gpu}] ${name}: ${published} exists without a manifest -- refusing (move it aside, do not delete)"
    printf '90\n' > "${LOG_DIR}/${name}.rc"; return 90
  fi

  local extra=()
  [ "${MODE}" = "forced_color" ] && extra+=(--forced-color-prefix)
  [ -n "${LIMIT}" ] && extra+=(--limit "${LIMIT}")

  say "[gpu${gpu}] ${name}: START -> ${published}"
  rm -f "${log}" "${LOG_DIR}/${name}.rc" "${LOG_DIR}/${name}.job.marker"   # step 1
  # NOT wrapped in `( ... ) &`: that would make $! the subshell's pid, so `ps -p`
  # would prove the *wrapper* alive and `kill` would orphan python on the card
  # (exactly review nit N7 on run_where_b.sh).  A simple command with an
  # assignment prefix is forked-and-exec'd, so $! is python itself.  The worker
  # already cd'd to the repo.
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" "${HERE}/genctx_shard.py" \
    --split "${split}" --shard "${i}" --num-shards "${n}" \
    --shard-mode "${SHARD_MODE}" \
    --out-root "${out_root}" \
    --report-dir "${REPORT_ROOT}/${MODE}/${split}" \
    --checkpoint "${CKPT}" \
    --batch-size "${BATCH_SIZE}" --max-new-tokens "${MAX_NEW_TOKENS}" \
    "${extra[@]}" > "${log}" 2>&1 &
  local pid=$!

  sleep 5                                                                  # step 2
  if ! ps -p "${pid}" -o pid,etime,cmd --no-headers; then
    say "[gpu${gpu}] ${name}: FAILED to start (no pid ${pid})"; sed -n '1,40p' "${log}"
    printf '91\n' > "${LOG_DIR}/${name}.rc"; return 91
  fi

  local waited=0                                                           # step 3
  until grep -q '"vlm"' "${log}" 2>/dev/null; do
    sleep 10; waited=$((waited + 10))
    if ! ps -p "${pid}" > /dev/null; then
      say "[gpu${gpu}] ${name}: died after ${waited}s before printing the setup JSON"
      tail -n 40 "${log}"; wait "${pid}"; local drc=$?
      # exiting 0 without ever printing the setup JSON is still a failure: the
      # shard was not generated, and "it exited cleanly" must not read as done
      [ "${drc}" -eq 0 ] && drc=92
      printf '%s\n' "${drc}" > "${LOG_DIR}/${name}.rc"; return "${drc}"
    fi
    if [ "${waited}" -ge "${PROBE_TIMEOUT}" ]; then
      say "[gpu${gpu}] ${name}: no setup JSON after ${waited}s -- refusing to claim success"
      tail -n 40 "${log}"
      # it is still holding the card, and the next queue entry targets the same
      # card: kill it by the PID we own (never pkill/pattern matching)
      kill "${pid}" 2>/dev/null; sleep 10; kill -9 "${pid}" 2>/dev/null; wait "${pid}"
      printf '93\n' > "${LOG_DIR}/${name}.rc"; return 93
    fi
  done
  say "[gpu${gpu}] ${name}: RUNNING pid=${pid} (setup JSON after ${waited}s)"

  printf 'pid=%s\ngpu=%s\nmode=%s\nsplit=%s\nshard=%s/%s\ncheckpoint=%s\nbatch_size=%s\nmax_new_tokens=%s\nout_root=%s\nlog=%s\nstarted_at=%s\nverified=/"vlm"/ after %ss\n' \
    "${pid}" "${gpu}" "${MODE}" "${split}" "${i}" "${n}" "${CKPT}" "${BATCH_SIZE}" \
    "${MAX_NEW_TOKENS}" "${published}" "${log}" "$(ts)" "${waited}" \
    > "${LOG_DIR}/${name}.job.marker"                                      # step 4

  wait "${pid}"; local rc=$?
  printf '%s\n' "${rc}" > "${LOG_DIR}/${name}.rc"
  say "[gpu${gpu}] ${name}: END rc=${rc}"
  [ "${rc}" -eq 0 ] || tail -n 25 "${log}"
  return "${rc}"
}

# ------------------------------------------------------------ gpu worker -----
# gpu_worker <gpu> <task...>   where task = "<split>:<i>:<n>"
gpu_worker() {
  local gpu="$1"; shift
  local worker_rc=0 t split i n
  cd "${REPO}" || { say "[gpu${gpu}] cannot cd ${REPO}"; return 1; }
  for t in "$@"; do
    IFS=':' read -r split i n <<< "${t}"
    run_shard "${gpu}" "${split}" "${i}" "${n}" || worker_rc=$?
  done
  say "[gpu${gpu}] worker done rc=${worker_rc}"
  return "${worker_rc}"
}

# ----------------------------------------------------------------- main ------
say "DRIVER START pid=$$ host=$(hostname)"
if ! probe; then say "PROBE FAILED -- nothing started"; printf '1\n' > "${LOG_DIR}/genctx_dual_${MODE}.rc"; exit 1; fi

# build the task list: shards round-robin over the cards, small splits first so
# the cheap consumers (V_where / V_what boards) are unblocked within the hour
declare -a TASKS=()
for split in ${SPLITS}; do
  n=$(shards_for "$(n_samples_of "${split}")")
  for ((i = 0; i < n; i++)); do TASKS+=("${split}:${i}:${n}"); done
done
read -r -a GPU_ARR <<< "${GPUS}"
NG=${#GPU_ARR[@]}
if [ "${NG}" -lt 1 ] || [ "${NG}" -gt 4 ]; then
  say "GPUS='${GPUS}' -> ${NG} cards; this driver holds 1..4 queues"; exit 2
fi
declare -a QUEUE_0=() QUEUE_1=() QUEUE_2=() QUEUE_3=()
for ((k = 0; k < ${#TASKS[@]}; k++)); do
  eval "QUEUE_$((k % NG))+=(\"\${TASKS[k]}\")"
done

say "PLAN: ${#TASKS[@]} shard jobs over ${NG} card(s)"
for ((g = 0; g < NG; g++)); do
  eval "q=(\"\${QUEUE_${g}[@]}\")"
  say "  GPU ${GPU_ARR[g]} <- ${q[*]}"
done

if [ "${DRY_RUN}" = "1" ]; then
  say "DRY_RUN=1 -- plan printed, no process started, no GPU touched"
  printf '0\n' > "${LOG_DIR}/genctx_dual_${MODE}.rc"; exit 0
fi

declare -a WPIDS=()
for ((g = 0; g < NG; g++)); do
  eval "q=(\"\${QUEUE_${g}[@]}\")"
  gpu_worker "${GPU_ARR[g]}" "${q[@]}" &
  WPIDS+=("$!")
done
say "workers: ${WPIDS[*]}"
for p in "${WPIDS[@]}"; do wait "${p}" || DRIVER_RC=$?; done
say "all workers finished; worst worker rc=${DRIVER_RC}"

# ---------------------------------------------------------------- merge ------
if [ "${DO_MERGE}" = "1" ] && [ "${DRIVER_RC}" -eq 0 ]; then
  for split in ${SPLITS}; do
    n=$(shards_for "$(n_samples_of "${split}")")
    mlog="${LOG_DIR}/merge_${MODE}_${split}.log"
    say "MERGE ${MODE}/${split} (${n} shards) -> ${mlog}"
    rm -f "${mlog}"
    ( cd "${REPO}" && "${PY}" "${HERE}/merge_genctx.py" \
        --split "${split}" --mode "${MODE}" \
        --shard-root "${STAGE_ROOT}/${MODE}/${split}" \
        --num-shards "${n}" --shard-mode "${SHARD_MODE}" ) > "${mlog}" 2>&1
    rc=$?
    printf '%s\n' "${rc}" > "${LOG_DIR}/merge_${MODE}_${split}.rc"
    say "MERGE ${MODE}/${split} rc=${rc}"
    [ "${rc}" -eq 0 ] || { tail -n 25 "${mlog}"; DRIVER_RC="${rc}"; }
  done
elif [ "${DO_MERGE}" = "1" ]; then
  say "skipping the merge: at least one shard failed (rc=${DRIVER_RC}); "\
"fix or rerun the failed shards (completed ones are skipped) and then run merge_genctx.py"
fi

say "DRIVER EXIT rc=${DRIVER_RC}"
printf '%s\n' "${DRIVER_RC}" > "${LOG_DIR}/genctx_dual_${MODE}.rc"
exit "${DRIVER_RC}"
