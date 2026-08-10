#!/usr/bin/env bash
# Shared helpers for the post-SFT chain jobs.  Sourced, never executed.
#
# Contract with chain_after_sft.sh (D-20):
#   * the job prints a line containing PROBE-OK as soon as it has proved, with
#     real measured values, that its interpreter / GPU / inputs are usable.
#     The chain waits for that token before it accepts the job as started --
#     "the log file exists" is explicitly not enough (CLAUDE.md D-20 step 3).
#   * every phase writes  $PHASE_DIR/<phase>.rc  containing its exit code.
#   * the job writes      $JOB_DIR/<JOB_NAME>.rc  on exit (EXIT trap), so the
#     chain can tell "finished non-zero" from "died without a trace".
#
# NOTE: `set -e` is deliberately NOT used.  A failing phase must still record
# its rc and let the following phases decide for themselves.

set -uo pipefail

ts() { date -Is; }
say() { printf '[%s] %s\n' "$(ts)" "$*"; }

_job_exit_trap() {
  local rc=$?
  if [ "${JOB_RC}" -ne 0 ]; then rc="${JOB_RC}"; fi
  printf '%s\n' "${rc}" > "${JOB_DIR}/${JOB_NAME}.rc"
  say "JOB ${JOB_NAME} EXIT rc=${rc}"
}

job_init() {   # job_init <job-name> <job-dir>
  JOB_NAME="$1"
  JOB_DIR="$2"
  PHASE_DIR="${JOB_DIR}/phase_${JOB_NAME}"
  mkdir -p "${PHASE_DIR}"
  JOB_RC=0
  # a stale rc from an earlier attempt would make the chain call a running job
  # finished, so clear both the job rc and this job's phase records up front.
  rm -f "${JOB_DIR}/${JOB_NAME}.rc"
  rm -f "${PHASE_DIR}"/*.rc 2>/dev/null || true
  trap _job_exit_trap EXIT
  say "JOB ${JOB_NAME} START pid=$$ host=$(hostname)"
  say "  job_dir=${JOB_DIR}"
  say "  phase_dir=${PHASE_DIR}"
}

phase() {      # phase <phase-name> <cmd> [args...]
  local name="$1"; shift
  local t0 rc
  say "PHASE ${name} BEGIN: $*"
  t0=$(date +%s)
  "$@"
  rc=$?
  printf '%s\n' "${rc}" > "${PHASE_DIR}/${name}.rc"
  say "PHASE ${name} END rc=${rc} seconds=$(( $(date +%s) - t0 ))"
  if [ "${rc}" -ne 0 ] && [ "${JOB_RC}" -eq 0 ]; then JOB_RC="${rc}"; fi
  return "${rc}"
}

probe_ok() {   # probe_ok <one-line summary>
  say "PROBE-OK $*"
}
