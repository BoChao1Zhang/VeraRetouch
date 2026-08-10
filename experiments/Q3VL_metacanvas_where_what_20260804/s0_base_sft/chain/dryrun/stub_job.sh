#!/usr/bin/env bash
# Dry-run stand-in for a chain job.  Speaks the exact same protocol as the real
# wrappers (PROBE-OK line, per-phase .rc files, job .rc via the EXIT trap), so
# the chain's D-20 dispatch and its monitor are exercised for real without a GPU
# and without touching any production output directory.
#
#   stub_job.sh <job-name> <final-rc> <seconds>

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../joblib.sh
source "${HERE}/../joblib.sh"

NAME="${1:?job name}"
RC="${2:-0}"
SECONDS_TOTAL="${3:-8}"
JOB_DIR_IN="${JOB_DIR_IN:-${HERE}/../logs}"

job_init "${NAME}" "${JOB_DIR_IN}"

probe() {
  say "stub job, no GPU, no production path touched"
  say "python=$(command -v python3) uname=$(uname -sr)"
  probe_ok "stub ${NAME} ready (final rc will be ${RC}, runtime ${SECONDS_TOTAL}s)"
}

phase probe probe

half=$(( SECONDS_TOTAL / 2 ))
[ "${half}" -lt 1 ] && half=1

phase work_a bash -c "sleep ${half}; echo 'stub phase work_a did something'"
phase work_b bash -c "sleep ${half}; echo 'stub phase work_b did something'; exit ${RC}"

exit "${JOB_RC}"
