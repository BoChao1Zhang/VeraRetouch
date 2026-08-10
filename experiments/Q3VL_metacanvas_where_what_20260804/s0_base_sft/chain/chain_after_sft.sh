#!/usr/bin/env bash
# =============================================================================
# EXEC-1 -- zero-idle GPU hand-off from the Base SFT to the next three jobs.
#
# Waits for the two Base SFT rank processes to exit, adjudicates whether the run
# actually completed, and -- only then -- dispatches, in one go:
#
#   GPU0  gpu0_ckpt_verify     offline verification of checkpoint-2488 /
#                              checkpoint-4976 (spec 8.3 structural rates,
#                              deferred offline by ruling D-J3) + the
#                              assistant-only eval_loss book-keeping table
#   GPU1  gpu1_wherea_s2       Where-A S2: protocol 14 items 4/5/6 GPU preflight
#                              on the real checkpoint + the D5 radius/eps sweep
#   CPU   cpu_sync_maskviews   SYNC_PROTECTED_TO_NFS.sh (durability first),
#                              then Where-A S1 mask-view packing for 5 splits
#
# Every dispatch follows D-20 in order: rm -f the target log -> capture $! and
# prove liveness with `ps -p $PID` -> wait for substantive output (the job's own
# PROBE-OK line, which carries measured values) -> only then write job.marker
# and report.  `pgrep` is never used, for liveness or for anything else: any
# `pgrep -f <pattern>` matches the shell running the grep.  `ps -p $PID` with an
# optional cmdline match on that one pid has no such failure mode, and it also
# closes the PID-reuse hole.
#
# After ALL-DISPATCHED the script keeps running as a monitor and appends
# `CHAIN: <job> DONE|FAILED` (and per-phase lines) as jobs finish.
#
# Submitting it (itself D-20):
#     rm -f  .../chain/logs/chain.log
#     nohup bash .../chain/chain_after_sft.sh > .../chain/logs/chain.log 2>&1 &
#     ps -p $!            # liveness
#     tail .../chain/logs/chain.log
#
# Dry run (no GPU, no real output dirs -- see dryrun/run_dryrun.sh):
#     CHAIN_STUB_JOBS=1 CHAIN_RUN_DIR=<fake> CHAIN_RANK_PIDS="..." \
#     CHAIN_RANK_PATTERN=mock_train_sft CHAIN_POLL_SECONDS=2 \
#     bash chain_after_sft.sh
# =============================================================================

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REPO="${REPO:-/home/bc/VeraRetouch}"
RUN_DIR="${CHAIN_RUN_DIR:-/home/bc/data/runs/q3vl_base_sft_20260804}"
TRAIN_LOG="${CHAIN_TRAIN_LOG:-${RUN_DIR}/train.log}"
RANK_PIDS="${CHAIN_RANK_PIDS:-3395226 3395227}"
LAUNCHER_PID="${CHAIN_LAUNCHER_PID:-3395099}"
RANK_PATTERN="${CHAIN_RANK_PATTERN:-q3vl.train.train_sft}"
FINAL_STEP="${CHAIN_FINAL_STEP:-4976}"
LOG_DIR="${CHAIN_LOG_DIR:-${HERE}/logs}"
POLL="${CHAIN_POLL_SECONDS:-60}"
SETTLE="${CHAIN_SETTLE_SECONDS:-45}"
PROBE_TIMEOUT="${CHAIN_PROBE_TIMEOUT:-600}"
WAIT_MAX_HOURS="${CHAIN_WAIT_MAX_HOURS:-8}"
MONITOR_MAX_HOURS="${CHAIN_MONITOR_MAX_HOURS:-24}"
STUB_JOBS="${CHAIN_STUB_JOBS:-0}"
JOBS="${CHAIN_JOBS:-gpu0 gpu1 cpu}"
HEARTBEAT_EVERY="${CHAIN_HEARTBEAT_EVERY:-10}"    # in poll intervals

mkdir -p "${LOG_DIR}"
export JOB_DIR_IN="${LOG_DIR}"
export REPO

ts()   { date -Is; }
info() { printf '[chain %s] %s\n' "$(ts)" "$*"; }
stat_line() { printf 'CHAIN: %s  at=%s\n' "$*" "$(ts)"; }

# ---------------------------------------------------------------------------
# liveness.  NEVER pgrep.
#   `ps -p <pid> -o args=` prints only that pid's command line, so the grep that
#   filters it cannot match itself the way `pgrep -f` always does.  The optional
#   pattern also means a recycled pid does not read as "still training".
# ---------------------------------------------------------------------------
alive() {   # alive <pid> [cmdline-substring]
  local pid="$1" pat="${2:-}"
  ps -p "${pid}" > /dev/null 2>&1 || return 1
  if [ -n "${pat}" ]; then
    ps -p "${pid}" -o args= 2>/dev/null | grep -a -F -q -- "${pat}" || return 1
  fi
  return 0
}

n_ranks_alive() {
  local n=0 pid
  for pid in ${RANK_PIDS}; do
    alive "${pid}" "${RANK_PATTERN}" && n=$((n + 1))
  done
  printf '%s' "${n}"
}

# ---------------------------------------------------------------------------
# 1. wait for the Base SFT ranks
# ---------------------------------------------------------------------------
wait_for_sft() {
  local deadline i n
  deadline=$(( $(date +%s) + WAIT_MAX_HOURS * 3600 ))
  i=0
  info "waiting for Base SFT ranks [${RANK_PIDS}] (pattern '${RANK_PATTERN}'), poll=${POLL}s"
  while :; do
    n="$(n_ranks_alive)"
    if [ "${n}" -eq 0 ]; then
      info "all Base SFT ranks have exited"
      break
    fi
    if [ "$(date +%s)" -ge "${deadline}" ]; then
      stat_line "CHAIN-ABORT reason=\"waited ${WAIT_MAX_HOURS}h and ${n} rank(s) are still alive\""
      return 1
    fi
    if [ $(( i % HEARTBEAT_EVERY )) -eq 0 ]; then
      local prog=""
      if [ -f "${TRAIN_LOG}" ]; then
        prog="$(tail -c 4000 "${TRAIN_LOG}" 2>/dev/null | tr '\r' '\n' \
                | grep -aoE "[0-9]+/${FINAL_STEP}" | tail -1)"
      fi
      info "still training: ranks_alive=${n} progress=${prog:-unknown}"
    fi
    i=$(( i + 1 ))
    sleep "${POLL}"
  done

  # the launcher exits right after its ranks; give it a bounded grace so the GPUs
  # are genuinely released before anything else claims them.
  local waited=0
  while alive "${LAUNCHER_PID}" "" && [ "${waited}" -lt 600 ]; do
    sleep 5; waited=$(( waited + 5 ))
  done
  alive "${LAUNCHER_PID}" "" && info "WARNING: launcher pid ${LAUNCHER_PID} still alive after ${waited}s"
  info "settling ${SETTLE}s for the final save to flush"
  sleep "${SETTLE}"
  return 0
}

# ---------------------------------------------------------------------------
# 2. did the run actually finish?
# ---------------------------------------------------------------------------
REQUIRED_CKPT_FILES=(
  config.json generation_config.json model.safetensors.index.json
  tokenizer.json tokenizer_config.json special_tokens_map.json added_tokens.json
  preprocessor_config.json chat_template.jinja trainer_state.json
)
BAD_LOG_PATTERNS=(
  "Traceback (most recent call last)"
  "CUDA out of memory"
  "OutOfMemoryError"
  "Killed"
  "Segmentation fault"
  "terminate called after throwing"
  "KeyboardInterrupt"
  "torch.distributed.elastic.multiprocessing.errors"
  "Signal 9"
)

adjudicate() {
  local -a reasons=()
  local ck="${RUN_DIR}/checkpoint-${FINAL_STEP}"

  info "adjudicating completion: run_dir=${RUN_DIR} final_step=${FINAL_STEP}"

  if [ ! -d "${ck}" ]; then
    reasons+=("checkpoint-${FINAL_STEP} directory is absent (${ck})")
  else
    local f
    for f in "${REQUIRED_CKPT_FILES[@]}"; do
      [ -f "${ck}/${f}" ] || reasons+=("checkpoint-${FINAL_STEP}/${f} is missing")
    done
    local n_shards
    n_shards=$(find "${ck}" -maxdepth 1 -type f \
                 \( -name 'model-*.safetensors' -o -name 'model.safetensors' \) 2>/dev/null | wc -l)
    [ "${n_shards}" -ge 1 ] || reasons+=("checkpoint-${FINAL_STEP} carries no safetensors weight file")
    info "  checkpoint weight files: ${n_shards}"
    if [ -f "${ck}/trainer_state.json" ]; then
      local gs
      gs=$(jq -r '.global_step' "${ck}/trainer_state.json" 2>/dev/null)
      info "  trainer_state.global_step=${gs} (expected ${FINAL_STEP})"
      [ "${gs}" = "${FINAL_STEP}" ] || \
        reasons+=("checkpoint-${FINAL_STEP}/trainer_state.json global_step=${gs} != ${FINAL_STEP}")
    fi
  fi

  # save_model()+save_state() at the output_dir root are the last thing
  # train_sft.py does; SYNC_PROTECTED_TO_NFS.sh also refuses without it.
  if [ ! -f "${RUN_DIR}/trainer_state.json" ]; then
    reasons+=("${RUN_DIR}/trainer_state.json absent -- the final save_state() never ran")
  fi

  if [ ! -f "${TRAIN_LOG}" ]; then
    reasons+=("train log ${TRAIN_LOG} is absent")
  else
    local tailfile="${LOG_DIR}/train_log_tail.txt"
    tail -c 400000 "${TRAIN_LOG}" > "${tailfile}" 2>/dev/null
    local pat
    for pat in "${BAD_LOG_PATTERNS[@]}"; do
      if grep -a -F -q -- "${pat}" "${tailfile}"; then
        reasons+=("train log tail contains '${pat}'")
      fi
    done
    if grep -a -F -q -- "training finished:" "${tailfile}"; then
      info "  train log tail carries the 'training finished:' line"
    else
      info "  NOTE: 'training finished:' not seen in the last 400 KiB of the log (informational)"
    fi
  fi

  if [ "${#reasons[@]}" -gt 0 ]; then
    stat_line "CHAIN-ABORT reasons=${#reasons[@]}"
    local r
    for r in "${reasons[@]}"; do
      stat_line "CHAIN-ABORT reason: ${r}"
    done
    stat_line "CHAIN-ABORT no job was started; the GPUs were left untouched"
    return 1
  fi
  info "adjudication passed: the run completed normally"
  return 0
}

# ---------------------------------------------------------------------------
# 3. dispatch (D-20, four steps, per job)
# ---------------------------------------------------------------------------
declare -A JOB_PID JOB_LOG JOB_T0
declare -A PHASE_SEEN

dispatch() {   # dispatch <name> <log> <cmd...>
  local name="$1" log="$2"; shift 2
  local rc_file="${LOG_DIR}/${name}.rc"

  rm -f "${log}"                                        # D-20 step 1
  rm -f "${rc_file}"
  info "dispatching ${name}: $*"
  nohup "$@" > "${log}" 2>&1 &
  local pid=$!
  sleep 3

  if ! ps -p "${pid}" > /dev/null 2>&1 && [ ! -f "${rc_file}" ]; then   # D-20 step 2
    stat_line "${name} FAILED-TO-START pid=${pid} log=${log}"
    info "${name} first 40 log lines:"
    tail -n 40 "${log}" 2>/dev/null | sed 's/^/    /'
    return 1
  fi

  local t0 ok=0                                          # D-20 step 3
  t0=$(date +%s)
  while [ $(( $(date +%s) - t0 )) -lt "${PROBE_TIMEOUT}" ]; do
    if grep -a -q 'PROBE-OK' "${log}" 2>/dev/null; then ok=1; break; fi
    if ! ps -p "${pid}" > /dev/null 2>&1; then
      grep -a -q 'PROBE-OK' "${log}" 2>/dev/null && ok=1
      break
    fi
    sleep 2
  done
  if [ "${ok}" -ne 1 ]; then
    stat_line "${name} FAILED-TO-START pid=${pid} log=${log} reason=no-substantive-output-in-${PROBE_TIMEOUT}s"
    info "${name} last 40 log lines:"
    tail -n 40 "${log}" 2>/dev/null | sed 's/^/    /'
    return 1
  fi

  {                                                      # D-20 step 4
    printf 'pid=%s\n' "${pid}"
    printf 'cmd=%s\n' "$*"
    printf 'log=%s\n' "${log}"
    printf 'rc_file=%s\n' "${rc_file}"
    printf 'phase_dir=%s\n' "${LOG_DIR}/phase_${name}"
    printf 'started_at=%s\n' "$(ts)"
    printf 'liveness_ps_p=%s\n' "$(ps -p "${pid}" > /dev/null 2>&1 && echo true || echo already-exited)"
    printf 'dispatched_by=%s\n' "${BASH_SOURCE[0]}"
  } > "${LOG_DIR}/${name}.job.marker"

  JOB_PID["${name}"]="${pid}"
  JOB_LOG["${name}"]="${log}"
  JOB_T0["${name}"]="$(date +%s)"
  stat_line "${name} STARTED pid=${pid} log=${log}"
  info "${name} substantive output:"
  grep -a -m 20 -B 6 'PROBE-OK' "${log}" 2>/dev/null | sed 's/^/    /'
  return 0
}

scan_phases() {   # scan_phases <name>
  local name="$1" d="${LOG_DIR}/phase_${name}" f p rc
  [ -d "${d}" ] || return 0
  for f in "${d}"/*.rc; do
    [ -f "${f}" ] || continue
    p="$(basename "${f}" .rc)"
    [ -n "${PHASE_SEEN[${name}/${p}]:-}" ] && continue
    rc="$(cat "${f}" 2>/dev/null)"
    PHASE_SEEN["${name}/${p}"]=1
    if [ "${rc}" = "0" ]; then
      stat_line "${name}/${p} DONE rc=0"
    else
      stat_line "${name}/${p} FAILED rc=${rc}"
    fi
  done
}

monitor() {
  local deadline i=0 name pid rc status running t0
  deadline=$(( $(date +%s) + MONITOR_MAX_HOURS * 3600 ))
  local -a live=()
  [ "${#JOB_PID[@]}" -gt 0 ] && live=("${!JOB_PID[@]}")
  while [ "${#live[@]}" -gt 0 ]; do
    local -a still=()
    for name in "${live[@]}"; do
      scan_phases "${name}"
      pid="${JOB_PID[${name}]}"
      if ps -p "${pid}" > /dev/null 2>&1; then
        still+=("${name}")
        continue
      fi
      sleep 1                       # let the EXIT trap land its rc file
      scan_phases "${name}"
      if [ -f "${LOG_DIR}/${name}.rc" ]; then
        rc="$(cat "${LOG_DIR}/${name}.rc" 2>/dev/null)"
        [ "${rc}" = "0" ] && status=DONE || status=FAILED
      else
        rc="unknown"; status=FAILED
      fi
      t0="${JOB_T0[${name}]}"
      stat_line "${name} ${status} rc=${rc} elapsed_s=$(( $(date +%s) - t0 )) log=${JOB_LOG[${name}]}"
      if [ "${status}" = FAILED ]; then
        info "${name} last 30 log lines:"
        tail -n 30 "${JOB_LOG[${name}]}" 2>/dev/null | sed 's/^/    /'
      fi
    done
    live=()
    [ "${#still[@]}" -gt 0 ] && live=("${still[@]}")
    [ "${#live[@]}" -eq 0 ] && break
    if [ "$(date +%s)" -ge "${deadline}" ]; then
      stat_line "MONITOR-TIMEOUT after ${MONITOR_MAX_HOURS}h; still running: ${live[*]}"
      return 1
    fi
    if [ $(( i % HEARTBEAT_EVERY )) -eq 0 ]; then
      running="${live[*]}"
      info "running: ${running}"
    fi
    i=$(( i + 1 ))
    sleep "${POLL}"
  done
  return 0
}

# ---------------------------------------------------------------------------
main() {
  info "chain_after_sft.sh pid=$$ host=$(hostname)"
  info "  repo=${REPO}"
  info "  run_dir=${RUN_DIR}"
  info "  train_log=${TRAIN_LOG}"
  info "  rank_pids=${RANK_PIDS} launcher=${LAUNCHER_PID} pattern=${RANK_PATTERN}"
  info "  final_step=${FINAL_STEP} poll=${POLL}s settle=${SETTLE}s probe_timeout=${PROBE_TIMEOUT}s"
  info "  log_dir=${LOG_DIR} jobs='${JOBS}' stub_jobs=${STUB_JOBS}"
  info "  git_commit=$(git -C "${REPO}" rev-parse HEAD 2>/dev/null || echo unknown)"

  wait_for_sft || return 2
  adjudicate   || return 2

  local rc_any=0
  local stub="${HERE}/dryrun/stub_job.sh"
  for j in ${JOBS}; do
    case "${j}" in
      gpu0)
        if [ "${STUB_JOBS}" = "1" ]; then
          dispatch gpu0_ckpt_verify "${LOG_DIR}/gpu0_ckpt_verify.log" \
            bash "${stub}" gpu0_ckpt_verify "${CHAIN_STUB_RC_GPU0:-0}" "${CHAIN_STUB_SECONDS:-8}" || rc_any=1
        else
          dispatch gpu0_ckpt_verify "${LOG_DIR}/gpu0_ckpt_verify.log" \
            bash "${HERE}/job_gpu0_ckpt_verify.sh" || rc_any=1
        fi
        ;;
      gpu1)
        if [ "${STUB_JOBS}" = "1" ]; then
          dispatch gpu1_wherea_s2 "${LOG_DIR}/gpu1_wherea_s2.log" \
            bash "${stub}" gpu1_wherea_s2 "${CHAIN_STUB_RC_GPU1:-0}" "${CHAIN_STUB_SECONDS:-8}" || rc_any=1
        else
          dispatch gpu1_wherea_s2 "${LOG_DIR}/gpu1_wherea_s2.log" \
            bash "${HERE}/job_gpu1_wherea_s2.sh" || rc_any=1
        fi
        ;;
      cpu)
        if [ "${STUB_JOBS}" = "1" ]; then
          dispatch cpu_sync_maskviews "${LOG_DIR}/cpu_sync_maskviews.log" \
            bash "${stub}" cpu_sync_maskviews "${CHAIN_STUB_RC_CPU:-0}" "${CHAIN_STUB_SECONDS:-8}" || rc_any=1
        else
          dispatch cpu_sync_maskviews "${LOG_DIR}/cpu_sync_maskviews.log" \
            bash "${HERE}/job_cpu_sync_maskviews.sh" || rc_any=1
        fi
        ;;
      *) info "unknown job selector '${j}' -- ignored" ;;
    esac
  done

  stat_line "ALL-DISPATCHED started=${#JOB_PID[@]} failed_to_start=${rc_any}"
  monitor || rc_any=1

  local done_n=0 fail_n=0 name rc
  if [ "${#JOB_PID[@]}" -gt 0 ]; then
    for name in "${!JOB_PID[@]}"; do
      rc="$(cat "${LOG_DIR}/${name}.rc" 2>/dev/null || echo unknown)"
      if [ "${rc}" = "0" ]; then done_n=$(( done_n + 1 )); else fail_n=$(( fail_n + 1 )); fi
    done
  fi
  stat_line "ALL-DONE done=${done_n} failed=${fail_n}"
  [ "${fail_n}" -eq 0 ] && [ "${rc_any}" -eq 0 ] && return 0
  return 1
}

main "$@"
rc=$?
info "chain_after_sft.sh exiting rc=${rc}"
exit "${rc}"
