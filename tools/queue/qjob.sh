#!/usr/bin/env bash
# =============================================================================
# qjob.sh -- the per-task wrapper that pueue actually executes.
#
# One of these wraps every queued job.  It is the piece that carries the
# campaign's hard-won discipline into the queue:
#
#   * artifact gate  -- a job does not start because a clock said so, it starts
#                       because the files it needs are on disk (the chain
#                       script's "produce gate, not log-string gate" rule).
#                       Gates WAIT (bounded) rather than fail fast, so a queued
#                       arm can be parked behind a job that is still running
#                       outside the queue and still hand over with zero idle.
#   * D-20           -- 1) rm -f the log (zsh noclobber kills the whole
#                          redirection if the target exists, and the process
#                          then never starts, invisibly)
#                       2) prove liveness with `ps -p $PID`.  NEVER pgrep: any
#                          `pgrep -f <pat>` matches the shell running the grep.
#                       3) wait for substantive output in the log, not merely
#                          for the file to exist
#                       4) only then write job.marker
#   * post-mortem    -- on exit, scan the log for the known training failure
#                       signatures and record rc + last 30 lines into the status
#                       json, so `qstatus.sh` can explain a failure without
#                       anyone opening a log.
#
# It deliberately does NOT setsid the payload: pueue kills a task by signalling
# its process group, and a setsid child would survive `pueue kill` as an orphan
# holding a GPU.  Staying in the group is what makes cancellation work.
#
# Usage:
#   qjob.sh --name W03 --gpu 0 --log /path/train.log \
#           [--gate /path/that/must/exist]...  [--gate-cmd 'shell test']... \
#           [--gate-timeout-hours 24] [--ready 'grep -E pattern'] \
#           [--ready-timeout 1800] [--poll 60] \
#           -- <command> [args...]
#
# Exit codes: the payload's own rc, or
#   78  gate never opened within --gate-timeout-hours (job skipped, queue moves on)
#   79  payload failed to start / died before printing the --ready pattern
# =============================================================================
set -uo pipefail

QUEUE_HOME="${QUEUE_HOME:-/home/bc/data/queue}"
STATUS_DIR="${QUEUE_HOME}/status"

NAME=""
GPU="-"
LOG=""
READY=""
READY_TIMEOUT=1800
GATE_TIMEOUT_HOURS=24
POLL=60
GATES=()
GATE_CMDS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --name)               NAME="$2"; shift 2 ;;
    --gpu)                GPU="$2"; shift 2 ;;
    --log)                LOG="$2"; shift 2 ;;
    --gate)               GATES+=("$2"); shift 2 ;;
    --gate-cmd)           GATE_CMDS+=("$2"); shift 2 ;;
    --gate-timeout-hours) GATE_TIMEOUT_HOURS="$2"; shift 2 ;;
    --ready)              READY="$2"; shift 2 ;;
    --ready-timeout)      READY_TIMEOUT="$2"; shift 2 ;;
    --poll)               POLL="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "qjob: unknown option '$1'" >&2; exit 2 ;;
  esac
done

[ -n "${NAME}" ] || { echo "qjob: --name is required" >&2; exit 2; }
[ -n "${LOG}"  ] || { echo "qjob: --log is required"  >&2; exit 2; }
[ $# -gt 0 ]     || { echo "qjob: no command given after --" >&2; exit 2; }

CMD=("$@")
CMD_STR="$(printf '%q ' "${CMD[@]}")"
STATUS_JSON="${STATUS_DIR}/${NAME}.json"
MARKER="$(dirname "${LOG}")/job.marker"
mkdir -p "${STATUS_DIR}" "$(dirname "${LOG}")"

ts()  { date -Is; }
now() { date +%s; }
say() { printf '[qjob %s %s] %s\n' "${NAME}" "$(ts)" "$*"; }

T_ENQUEUED="$(ts)"
T_START_EPOCH="$(now)"
PHASE="gate-wait"
PID=""
RC="null"
READY_AT="null"
STARTED_AT="null"
ENDED_AT="null"
NOTE=""

# --------------------------------------------------------------------------
# status json -- the single machine-readable record for this job.  Written
# through a temp file so a reader never sees a half-written document.
# --------------------------------------------------------------------------
write_status() {
  local tail_json='[]' gates_json='[]' sigs_json="${SIGS_JSON:-[]}"
  if [ -f "${LOG}" ]; then
    tail_json="$(tail -n 30 "${LOG}" 2>/dev/null | tr -d '\000' \
                 | jq -R -s 'split("\n") | map(select(length > 0))' 2>/dev/null || echo '[]')"
  fi
  local g
  gates_json='[]'
  for g in ${GATES+"${GATES[@]}"}; do
    local ok=false
    [ -e "${g}" ] && ok=true
    gates_json="$(jq -c --arg p "${g}" --argjson ok "${ok}" \
                    '. + [{path: $p, ok: $ok}]' <<<"${gates_json}")"
  done

  jq -n \
    --arg   name    "${NAME}" \
    --arg   gpu     "${GPU}" \
    --arg   phase   "${PHASE}" \
    --arg   log     "${LOG}" \
    --arg   cmd     "${CMD_STR}" \
    --arg   note    "${NOTE}" \
    --arg   enq     "${T_ENQUEUED}" \
    --argjson pid   "${PID:-null}" \
    --argjson rc    "${RC}" \
    --argjson started "${STARTED_AT}" \
    --argjson ready   "${READY_AT}" \
    --argjson ended   "${ENDED_AT}" \
    --argjson elapsed "$(( $(now) - T_START_EPOCH ))" \
    --argjson gates "${gates_json}" \
    --argjson sigs  "${sigs_json}" \
    --argjson tail  "${tail_json}" \
    --arg   updated "$(ts)" \
    '{name:$name, gpu:$gpu, phase:$phase, pid:$pid, rc:$rc, log:$log,
      cmd:$cmd, note:$note, enqueued_at:$enq, started_at:$started,
      ready_at:$ready, ended_at:$ended, elapsed_s:$elapsed,
      gates:$gates, failure_signatures:$sigs, log_tail:$tail,
      updated_at:$updated}' \
    > "${STATUS_JSON}.tmp" 2>/dev/null && mv -f "${STATUS_JSON}.tmp" "${STATUS_JSON}"
}

# --------------------------------------------------------------------------
# failure signatures -- the four ways a training job on this box dies.
# --------------------------------------------------------------------------
SIGS_JSON='[]'
scan_signatures() {
  [ -f "${LOG}" ] || return 0
  local labels=(traceback cuda_oom nan_loss killed cuda_error dataloader)
  local pats=(
    'Traceback \(most recent call last\)'
    'CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED'
    '[Nn]a[Nn] detected|loss (is|=|: ?)[[:space:]]*[Nn]a[Nn]|loss=nan'
    'Killed|Segmentation fault|Bus error|signal (9|11)'
    'CUDA error|device-side assert|NCCL error|cuDNN error'
    'KeyError|FileNotFoundError|AssertionError'
  )
  local i line
  SIGS_JSON='[]'
  for i in "${!labels[@]}"; do
    line="$(grep -a -m1 -E "${pats[$i]}" "${LOG}" 2>/dev/null | tr -d '\000' | cut -c1-400)"
    if [ -n "${line}" ]; then
      SIGS_JSON="$(jq -c --arg s "${labels[$i]}" --arg l "${line}" \
                     '. + [{signature: $s, line: $l}]' <<<"${SIGS_JSON}")"
    fi
  done
}

finish() {
  scan_signatures
  ENDED_AT="\"$(ts)\""
  write_status
}

on_signal() {
  say "received termination signal -- killing payload pid=${PID:-none}"
  [ -n "${PID}" ] && kill -TERM "${PID}" 2>/dev/null
  sleep 5
  [ -n "${PID}" ] && kill -KILL "${PID}" 2>/dev/null
  PHASE="killed"; RC=143; NOTE="terminated by signal (pueue kill / shutdown)"
  finish
  exit 143
}
trap on_signal TERM INT

write_status

# --------------------------------------------------------------------------
# 0. artifact gate.  Waits; does not fail fast.  A bounded wait means a dead
#    upstream eventually releases the slot instead of wedging the card forever.
# --------------------------------------------------------------------------
gate_open() {
  local g c
  for g in ${GATES+"${GATES[@]}"}; do
    [ -e "${g}" ] || { echo "${g}"; return 1; }
    if [ -f "${g}" ] && [ ! -s "${g}" ]; then echo "${g} (empty)"; return 1; fi
  done
  for c in ${GATE_CMDS+"${GATE_CMDS[@]}"}; do
    bash -c "${c}" >/dev/null 2>&1 || { echo "cmd: ${c}"; return 1; }
  done
  return 0
}

if [ ${#GATES[@]} -gt 0 ] || [ ${#GATE_CMDS[@]} -gt 0 ]; then
  gate_deadline=$(( $(now) + GATE_TIMEOUT_HOURS * 3600 ))
  i=0
  while :; do
    if blocker="$(gate_open)"; then
      say "gate open after $(( $(now) - T_START_EPOCH ))s"
      NOTE="gate opened"
      break
    fi
    if [ "$(now)" -ge "${gate_deadline}" ]; then
      PHASE="gate-timeout"; RC=78
      NOTE="gate never opened in ${GATE_TIMEOUT_HOURS}h; blocked on ${blocker}"
      say "GATE TIMEOUT: ${NOTE}"
      finish
      exit 78
    fi
    if [ $(( i % 10 )) -eq 0 ]; then
      say "waiting on gate: ${blocker}"
      NOTE="waiting on ${blocker}"
      write_status
    fi
    i=$(( i + 1 ))
    sleep "${POLL}"
  done
fi

# --------------------------------------------------------------------------
# 1. D-20 step 1: rm -f the log.  Under zsh's noclobber a `> existing.log`
#    makes the whole redirection fail and the process never starts.
# --------------------------------------------------------------------------
PHASE="starting"
rm -f "${LOG}" "${LOG}.pid" "${MARKER}"
write_status

if [ "${GPU}" != "-" ]; then
  export CUDA_VISIBLE_DEVICES="${GPU}"
  say "CUDA_VISIBLE_DEVICES=${GPU}"
fi
say "exec: ${CMD_STR}"

"${CMD[@]}" > "${LOG}" 2>&1 &
PID=$!
STARTED_AT="\"$(ts)\""
echo "${PID}" > "${LOG}.pid"

# --------------------------------------------------------------------------
# 2. D-20 step 2: prove liveness with `ps -p`.  NEVER pgrep.
# --------------------------------------------------------------------------
sleep 5
if ! ps -p "${PID}" -o pid,etime,args --no-headers 2>/dev/null; then
  wait "${PID}"; early=$?
  if [ "${early}" -eq 0 ]; then
    say "payload exited 0 within 5s (short job) -- accepting"
    PHASE="done"; RC=0; NOTE="completed before the liveness probe"
    finish
    exit 0
  fi
  PHASE="start-failed"; RC=79
  NOTE="pid ${PID} was gone 5s after launch (payload rc=${early})"
  say "FAILED TO START: ${NOTE}"
  finish
  exit 79
fi
PHASE="running"
write_status

# --------------------------------------------------------------------------
# 3. D-20 step 3: substantive output, not merely a file that exists.
# --------------------------------------------------------------------------
if [ -n "${READY}" ]; then
  waited=0
  until grep -a -q -E "${READY}" "${LOG}" 2>/dev/null; do
    if ! ps -p "${PID}" >/dev/null 2>&1; then
      wait "${PID}"; rc=$?
      if [ "${rc}" -eq 0 ]; then
        say "payload finished rc=0 without ever printing /${READY}/ -- accepting, but noting it"
        PHASE="done"; RC=0
        NOTE="never printed the readiness pattern /${READY}/ yet exited 0"
        finish
        exit 0
      fi
      PHASE="failed"; RC="${rc}"
      NOTE="died after ${waited}s before printing /${READY}/"
      say "DIED EARLY: ${NOTE}"
      finish
      exit "${rc}"
    fi
    if [ "${waited}" -ge "${READY_TIMEOUT}" ]; then
      say "no /${READY}/ after ${waited}s -- killing rather than claiming success"
      kill -TERM "${PID}" 2>/dev/null; sleep 10; kill -KILL "${PID}" 2>/dev/null
      wait "${PID}" 2>/dev/null
      PHASE="failed"; RC=79
      NOTE="alive but silent: no /${READY}/ within ${READY_TIMEOUT}s"
      finish
      exit 79
    fi
    sleep 5
    waited=$(( waited + 5 ))
    if [ $(( waited % 300 )) -eq 0 ]; then write_status; fi
  done
  READY_AT="\"$(ts)\""
  say "readiness confirmed: /${READY}/ after ${waited}s"
fi

# --------------------------------------------------------------------------
# 4. D-20 step 4: only now is the job real.
# --------------------------------------------------------------------------
{
  printf 'name=%s\npid=%s\ngpu=%s\nlog=%s\nstatus_json=%s\n' \
    "${NAME}" "${PID}" "${GPU}" "${LOG}" "${STATUS_JSON}"
  printf 'started_at=%s\nverified=/%s/\ncmd=%s\n' \
    "$(ts)" "${READY:-<none>}" "${CMD_STR}"
} > "${MARKER}"
say "job.marker written: ${MARKER}"
write_status

# --------------------------------------------------------------------------
# 5. supervise to completion, refreshing status so qstatus stays live
# --------------------------------------------------------------------------
# The sleep granularity is the hand-off latency: the pueue slot -- and with it
# the card -- stays occupied until this loop notices the payload is gone.  Poll
# the process every 5s and merely *report* every ${POLL}s, so a finished arm
# releases its GPU in seconds rather than in a minute.
tick=0
while ps -p "${PID}" >/dev/null 2>&1; do
  sleep 5
  tick=$(( tick + 5 ))
  if [ $(( tick % POLL )) -lt 5 ]; then write_status; fi
done
wait "${PID}"; rc=$?
RC="${rc}"
if [ "${rc}" -eq 0 ]; then PHASE="done"; NOTE="completed"; else PHASE="failed"; NOTE="payload exited ${rc}"; fi
say "payload exited rc=${rc}"
finish
exit "${rc}"
