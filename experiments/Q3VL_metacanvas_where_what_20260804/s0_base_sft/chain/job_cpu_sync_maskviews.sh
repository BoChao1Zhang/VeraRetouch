#!/usr/bin/env bash
# CPU / IO job: durability first, then the Where-A S1 mask-view packing.
#
#   phase sync              SYNC_PROTECTED_TO_NFS.sh -- rsync checkpoint-2488 /
#                           checkpoint-4976 / the root artefacts to the NFS
#                           durable path and record both sides' sha256 digests
#                           (main-agent ruling D-J1).  This runs FIRST: until it
#                           finishes the only copies of the protected
#                           checkpoints live on one local disk.
#   phase maskviews_<split> Where-A S1 -- publish the GT mask views as indexed
#                           tar shards for the five splits
#                           (PREFLIGHT_WHERE_A_PENDING.md §S1).
#
# Split order: V_where first as a cheap smoke (896 rows) before the train split
# (159,215 rows, ~75.5k eligible after the local-build + eligibility filter)
# commits hours of IO.  The pending doc lists train first; the loop has no
# internal dependency, and burning hours before discovering a packer problem is
# the failure this reorder buys out.  Recorded in chain/NOTES.md.
#
# `low` winner_confidence is INCLUDED by default: D1 put it into the Where-A
# calibration population, and q3vl/where/config.py already carries
# EXCLUDE_WINNER_CONFIDENCE_LOW = False, which makes extract_maskviews'
# eligibility filter keep it without --include-low.
#
# No GPU: CUDA_VISIBLE_DEVICES is emptied so nothing here can steal a card.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=joblib.sh
source "${HERE}/joblib.sh"

REPO="${REPO:-/home/bc/VeraRetouch}"
PY="${PY:-/home/bc/envs/q3vl_sft/bin/python}"
RUN_DIR="${RUN_DIR:-/home/bc/data/runs/q3vl_base_sft_20260804}"
NFS_DIR="${NFS_DIR:-/mnt/nfs/bc/runs/q3vl_base_sft_20260804}"
SYNC_SH="${SYNC_SH:-${REPO}/experiments/Q3VL_metacanvas_where_what_20260804/s0_base_sft/config/SYNC_PROTECTED_TO_NFS.sh}"
SPLITS="${SPLITS:-V_where V_what T_final T_lut_unseen train}"
SKIP_SYNC="${SKIP_SYNC:-0}"
SKIP_MASKVIEWS="${SKIP_MASKVIEWS:-0}"
JOB_DIR_IN="${JOB_DIR_IN:-${HERE}/logs}"

export LD_LIBRARY_PATH="/home/bc/miniconda3/envs/llm_factory/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUDA_VISIBLE_DEVICES=""
export TOKENIZERS_PARALLELISM=false

job_init "cpu_sync_maskviews" "${JOB_DIR_IN}"

probe() {
  say "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
  df -h "${RUN_DIR}" /mnt/nfs || return 1
  du -sh "${RUN_DIR}/checkpoint-2488" "${RUN_DIR}/checkpoint-4976" 2>&1 || true
  cd "${REPO}" || return 1
  "${PY}" - <<'PY' || return 1
import sqlite3, pathlib
from q3vl.where.config import MASKVIEW_DIR, LOCAL_BUILDS, EXCLUDE_WINNER_CONFIDENCE_LOW
from q3vl.where.maskdata import split_index_path
print(f"sqlite3={sqlite3.sqlite_version}")
print(f"maskview_dir={MASKVIEW_DIR} exists={MASKVIEW_DIR.exists()}")
print(f"local_builds={LOCAL_BUILDS} exclude_winner_confidence_low={EXCLUDE_WINNER_CONFIDENCE_LOW}")
for s in ("V_where", "V_what", "T_final", "T_lut_unseen", "train"):
    p = split_index_path(s)
    print(f"split_index {s}: {p} exists={p.is_file()}")
    if not p.is_file():
        raise SystemExit(f"missing split index for {s}")
PY
  probe_ok "cpu/io ready; split indexes present, NFS reachable"
}

phase probe probe || { say "probe failed -- not starting sync/maskviews"; exit "${JOB_RC}"; }

if [ "${SKIP_SYNC}" = "1" ]; then
  say "SKIP_SYNC=1 -- skipping the durability sync"
else
  phase sync env REPO_ROOT="${REPO}" LOCAL_DIR="${RUN_DIR}" NFS_DIR="${NFS_DIR}" \
    bash "${SYNC_SH}"
fi

if [ "${SKIP_MASKVIEWS}" = "1" ]; then
  say "SKIP_MASKVIEWS=1 -- skipping the Where-A S1 packing"
else
  cd "${REPO}" || exit 1
  for split in ${SPLITS}; do
    phase "maskviews_${split}" "${PY}" -m q3vl.where.scripts.extract_maskviews --split "${split}"
  done
fi

say "deliverables:"
ls -la "${NFS_DIR}" 2>&1 | head -20 || true
ls -la "${REPO}/experiments/Q3VL_metacanvas_where_what_20260804/where_a/maskviews" 2>&1 | head -30 || true

exit "${JOB_RC}"
