#!/usr/bin/env bash
# =============================================================================
# Enqueue Where-B waves W2, W3, W4 -- six arms, two cards, unattended.
#
# Wave layout (protocol METACANVAS ... 2026-08-04, table at "| Wave | GPU 0 |
# GPU 1 |"):  W1 = W01/W02, W2 = W03/W04, W3 = W05/W06, W4 = W07/W08, odd arm
# on GPU 0 and even arm on GPU 1.  W1 is NOT enqueued here: EXEC-3 launches it
# directly.  These six pick up the moment it lands.
#
# Ordering does not need dependencies.  Each card is a group with parallelism 1,
# so the gpu0 queue is exactly W03 -> W05 -> W07 and gpu1 is W04 -> W06 -> W08.
# A failed arm therefore does not block its successors -- which is the intent:
# one dead arm must not cost the campaign a card for a day.
#
# Only the head of each card carries an artifact gate, and it names the arm that
# is running *outside* the queue right now:
#
#     W03 (gpu0) waits for arm_W01.json      W04 (gpu1) waits for arm_W02.json
#
# That is the same "gate on the product, never on a log string" rule the
# chain_after_sft.sh hand-off used.  The gate WAITS (bounded, default 48h)
# instead of failing, so the card changes hands within seconds of W1 finishing.
#
# Two deliberate omissions in the gates:
#
#   * NFS.  The oracle latents, mask views and generated <where> context live
#     under /mnt/nfs (hard mount).  A `test -e` on a hard mount with a dead
#     server blocks in D state forever and cannot be killed, so no gate here
#     touches /mnt/nfs.  Those preconditions are not skipped -- run_where_b.py
#     asserts every one of them itself (assert_genctx_coverage, OracleStore
#     coverage, load_basis), and it does so inside the job where a hang is
#     visible and killable.
#   * The wave's shared micro batch.  See where_b_arm.sh: whichever arm of a
#     wave starts first claims the prober role and publishes its value for the
#     other, so the pairing contract holds without anyone guessing a number now
#     and without assuming which card frees up first.
#
# Usage:
#   tools/queue/waves/enqueue_where_b_w2_w4.sh            # enqueue
#   tools/queue/waves/enqueue_where_b_w2_w4.sh --stashed  # enqueue held
#   tools/queue/waves/enqueue_where_b_w2_w4.sh --dry-run  # print, change nothing
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
Q="${HERE}/../q"
ARM_SH="${HERE}/where_b_arm.sh"

REPORT_DIR="${WHERE_B_REPORT_DIR:-/home/bc/VeraRetouch/experiments/Q3VL_metacanvas_where_what_20260804/where_b}"
RUNS="${WHERE_B_RUNS:-/home/bc/data/runs/where_b}"
CKPT="${CKPT:-/home/bc/data/runs/q3vl_base_sft_20260804/checkpoint-4976}"

GATE_TIMEOUT_HOURS="${GATE_TIMEOUT_HOURS:-48}"
READY_TIMEOUT="${READY_TIMEOUT:-7200}"     # VLM load + genctx coverage + probe
PAIR_TIMEOUT="${PAIR_TIMEOUT:-7200}"

DRY=0; STASHED=""
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    --stashed) STASHED="--stashed" ;;
    *) echo "unknown flag $a" >&2; exit 2 ;;
  esac
done

# arm | gpu | wave (shared micro batch) | gate artifact (empty = checkpoint only)
WAVES=(
  "W03|0|w2|${REPORT_DIR}/arm_W01.json"
  "W04|1|w2|${REPORT_DIR}/arm_W02.json"
  "W05|0|w3|"
  "W06|1|w3|"
  "W07|0|w4|"
  "W08|1|w4|"
)

echo "Where-B W2-W4: 6 arms, checkpoint ${CKPT}"
[ -n "${MICRO_BATCH:-}" ] && echo "micro batch: PINNED to ${MICRO_BATCH} for all six arms"
echo

# Enqueueing a wave means starting it fresh, so its coordination state must go.
# A lock left by an arm that was killed before it published would otherwise make
# the *next* run of that arm a follower waiting on a prober that never runs --
# it would sit there for the whole --pair-timeout and then refuse with rc=80.
# (Observed for real: tearing down a queued W05 left .wave_w3.prober.lock behind.)
if [ "${DRY}" -eq 0 ]; then
  for w in w2 w3 w4; do
    if [ -e "${RUNS}/.wave_${w}.prober.lock" ] || [ -e "${RUNS}/.wave_${w}.micro_batch" ]; then
      echo "clearing stale wave state for ${w}"
      rm -rf "${RUNS}/.wave_${w}.prober.lock" "${RUNS}/.wave_${w}.micro_batch"
    fi
  done
fi

for row in "${WAVES[@]}"; do
  IFS='|' read -r arm gpu wave gate <<<"${row}"
  log="${RUNS}/${arm}/train.log"

  opts=(--ready 'total_optimizer_steps' --ready-timeout "${READY_TIMEOUT}"
        --gate "${CKPT}" --gate-timeout-hours "${GATE_TIMEOUT_HOURS}" --poll 30)
  [ -n "${gate}" ] && opts+=(--gate "${gate}")

  # MICRO_BATCH=N pins every arm explicitly and skips the wave coordination
  # entirely -- this is what EXEC-3 did for W1 (`--micro-batch 8`).  Unset, each
  # wave settles on its own first-arm-probes value.
  if [ -n "${MICRO_BATCH:-}" ]; then
    payload=(bash "${ARM_SH}" "${arm}" --micro-batch "${MICRO_BATCH}")
  else
    payload=(bash "${ARM_SH}" "${arm}" --wave "${wave}" --pair-timeout "${PAIR_TIMEOUT}")
  fi

  if [ "${DRY}" -eq 1 ]; then
    printf '%s -> gpu%s  wave=%s  gate=[%s]\n    %s\n' \
      "${arm}" "${gpu}" "${wave}" "${gate:-checkpoint only}" "${payload[*]}"
  else
    bash "${Q}" add "${arm}" "${gpu}" "${log}" ${STASHED} "${opts[@]}" -- "${payload[@]}"
  fi
done

echo
[ "${DRY}" -eq 1 ] && echo "(dry run -- nothing was enqueued)" || bash "${Q}" status
