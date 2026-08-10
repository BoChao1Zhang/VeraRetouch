#!/usr/bin/env bash
# =============================================================================
# One Where-B main arm, in the exact form the queue should launch it.
#
# This is the payload that qjob.sh supervises.  It deliberately does NOT call
# `run_where_b.sh train`: that verb runs its own `submit`, which nohup+setsid's
# the trainer into a new session.  Under the queue that would detach the real
# process from the pueue task -- the slot would free itself instantly, both
# cards would look idle, and `pueue kill` could never stop the training.  So we
# reproduce what `run_where_b.sh train` sets up (LD_LIBRARY_PATH, interpreter,
# repo cwd, checkpoint) and `exec` python directly, letting qjob.sh supply the
# D-20 discipline that `submit` would otherwise have supplied.
#
# ---------------------------------------------------------------------------
# The wave-pinned micro batch
# ---------------------------------------------------------------------------
# run_where_b.sh states the contract:
#
#   "A wave pins ONE --micro-batch for both of its arms: BalancedContextSampler
#    chunks the same seeded permutation, so a different micro-batch changes
#    len(sampler), total_optimizer_steps and therefore the LR schedule -- the
#    two arms would no longer be the paired comparison protocol 11 asks for."
#
# Both arms of a wave run on different cards and cannot both probe and agree.
# Naming one of them "the prober" up front does not work either: the cards drift
# apart, so the designated follower can perfectly well reach the head of its
# queue hours before the designated prober reaches the head of the other -- and
# would then sit waiting for a partner that has not even started.
#
# So the role is claimed, not assigned: `mkdir` of a per-wave lock directory is
# atomic, so whichever arm starts FIRST becomes the prober regardless of card
# or arm number.  It probes, then publishes the value its own run_setup.json
# records; the other arm reads that file and pins the same number.  Skew in
# either direction is harmless, and there is no ordering assumption left.
#
# If the prober never publishes (it died during startup), the follower REFUSES
# to run rather than silently producing an unpaired comparison -- an idle card
# is recoverable, a quietly invalid wave is not.  The queue moves on to the next
# wave either way.  An explicit --micro-batch bypasses the whole mechanism.
#
# Usage:
#   where_b_arm.sh W03 --wave w2 [--pair-timeout 7200] [extra args...]
#   where_b_arm.sh W03 --micro-batch 4          # explicit, no coordination
# Exit codes: 80 = the wave partner never published a micro batch.
# =============================================================================
set -uo pipefail

REPO="${REPO:-/home/bc/VeraRetouch}"
RUNS="${WHERE_B_RUNS:-/home/bc/data/runs/where_b}"
PY="${WHERE_B_PY:-/home/bc/envs/q3vl_sft/bin/python}"
CKPT="${CKPT:-/home/bc/data/runs/q3vl_base_sft_20260804/checkpoint-4976}"

# The training env's sqlite3 needs conda's libstdc++ (CXXABI_1.3.15); the mask
# locator opens a build catalog, so without this the data path cannot start.
export LD_LIBRARY_PATH=/home/bc/miniconda3/envs/llm_factory/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

ARM="${1:?usage: where_b_arm.sh <W01..W08> [--wave NAME] [args...]}"; shift
WAVE=""
PAIR_TIMEOUT=7200
EXPLICIT_MB=0
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --wave)         WAVE="$2"; shift 2 ;;
    --pair-timeout) PAIR_TIMEOUT="$2"; shift 2 ;;
    --micro-batch)  EXPLICIT_MB=1; EXTRA+=("$1" "$2"); shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

echo "where_b_arm: arm=${ARM} wave=${WAVE:-<none>} ckpt=${CKPT}"
mkdir -p "${RUNS}"

if [ -n "${WAVE}" ] && [ "${EXPLICIT_MB}" -eq 0 ]; then
  LOCK="${RUNS}/.wave_${WAVE}.prober.lock"
  MBFILE="${RUNS}/.wave_${WAVE}.micro_batch"
  SETUP="${RUNS}/${ARM}/run_setup.json"

  if [ -s "${MBFILE}" ]; then
    # Wave already settled (a rerun, or the partner got here well ahead).
    mb="$(cat "${MBFILE}")"
    echo "where_b_arm: wave ${WAVE} already pinned micro_batch=${mb}"
    EXTRA=(--micro-batch "${mb}" ${EXTRA+"${EXTRA[@]}"})
  elif mkdir "${LOCK}" 2>/dev/null; then
    # We got here first: we probe, and we publish for the partner.
    echo "${ARM}" > "${LOCK}/arm"
    echo "where_b_arm: claimed the PROBER role for wave ${WAVE}; will publish to ${MBFILE}"
    # Runs beside the trainer (same process group, so `pueue kill` gets it too)
    # and turns our own run_setup.json into the wave's pinned value.
    (
      w=0
      while [ ! -s "${SETUP}" ] && [ "${w}" -lt "${PAIR_TIMEOUT}" ]; do
        sleep 10; w=$(( w + 10 ))
      done
      if [ -s "${SETUP}" ]; then
        v="$(jq -r '.train.micro_batch // empty' "${SETUP}" 2>/dev/null)"
        if [ -n "${v}" ] && [ "${v}" != "null" ]; then
          printf '%s' "${v}" > "${MBFILE}.tmp" && mv -f "${MBFILE}.tmp" "${MBFILE}"
          echo "where_b_arm(publisher): wave ${WAVE} micro_batch=${v} published"
        fi
      else
        echo "where_b_arm(publisher): ${SETUP} never appeared; partner will refuse"
      fi
    ) &
  else
    # The partner is probing.  Wait for its number.
    claimed="$(cat "${LOCK}/arm" 2>/dev/null || echo '?')"
    echo "where_b_arm: wave ${WAVE} is being probed by ${claimed}; waiting for ${MBFILE}"
    waited=0
    while [ ! -s "${MBFILE}" ]; do
      if [ "${waited}" -ge "${PAIR_TIMEOUT}" ]; then
        echo "where_b_arm: REFUSING TO RUN -- wave partner ${claimed} published no" \
             "micro batch within ${PAIR_TIMEOUT}s (${MBFILE})."
        echo "where_b_arm: re-enqueue with an explicit --micro-batch N once the" \
             "partner's value is known; running unpinned would break the paired" \
             "comparison (protocol 11)."
        exit 80
      fi
      [ $(( waited % 300 )) -eq 0 ] && echo "where_b_arm: still waiting on ${claimed} (${waited}s)"
      sleep 20
      waited=$(( waited + 20 ))
    done
    mb="$(cat "${MBFILE}")"
    echo "where_b_arm: wave ${WAVE} pinned micro_batch=${mb} after ${waited}s"
    EXTRA=(--micro-batch "${mb}" ${EXTRA+"${EXTRA[@]}"})
  fi
fi

cd "${REPO}" || exit 1
echo "where_b_arm: exec ${PY} -m q3vl.whereb.scripts.run_where_b --arm ${ARM}" \
     "--checkpoint ${CKPT}" ${EXTRA+"${EXTRA[@]}"}
exec "${PY}" -m q3vl.whereb.scripts.run_where_b \
     --arm "${ARM}" --checkpoint "${CKPT}" ${EXTRA+"${EXTRA[@]}"}
