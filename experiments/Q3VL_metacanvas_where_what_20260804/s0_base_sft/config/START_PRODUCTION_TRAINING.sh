#!/usr/bin/env bash
# =============================================================================
# Qwen3-VL-4B-Instruct base SFT -- PRODUCTION LAUNCH
#
# MAIN-AGENT RULINGS, 2026-08-05 (joint preflight decisions closed):
#   D-J1 -> LOCAL DISK during training.
#           output_dir = /home/bc/data/runs/q3vl_base_sft_20260804
#           Measured basis (PREFLIGHT_JOINT.md 4.2/4.3): the training images
#           (23.68 GB) share the NFS mount the old output path used, and a
#           ZeRO-3 checkpoint write saturates that mount for ~9.3 min -- an
#           `ls`/`du` on it blocked >60 s during the smoke. Coupling saves to
#           dataloader reads was judged unacceptable. Local free 1.5 TB against
#           a 339 GiB peak resident footprint.
#           The 0.5-epoch (step 2488) and 1.0-epoch (step 4976) protected
#           checkpoints plus the final deliverable are rsync'd to the NFS
#           durable path AFTER training by SYNC_PROTECTED_TO_NFS.sh, which
#           records both sides' paths and sha256 digests.
#   D-J3 -> generation diagnostics stay OFF (gen_diag_samples: 0); the spec 8.3
#           structural rates are computed offline afterwards.
#
# Starts a ~8.2 hour, 2-GPU job that writes ~614 GiB of checkpoints.
#
# Preconditions enforced before launching (the three original ones retained):
#   1. both H100s are idle;
#   2. the production output dir does not already contain a checkpoint
#      (spec 8.1: a fresh run, never a smoke directory's training state);
#   3. the terminal manifest digest still matches the one this run was
#      resolved against;
#   plus two added with the D-J1 retarget:
#   4. RUN_DIR equals the config's output_dir -- launch_sft.sh derives only the
#      LOG path from RUN_DIR while the trainer writes checkpoints to the
#      config's output_dir, so a divergence would split the run across two
#      filesystems;
#   5. the local volume has room for the peak resident footprint.
#
# D-20 discipline (rm -f log, `ps -p $PID`, tail for substantive output, only
# then job.marker) lives in the reviewed launcher
# q3vl/train/scripts/launch_sft.sh, which this script calls.
# =============================================================================
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/bc/VeraRetouch}"
RUN_DIR="${RUN_DIR:-/home/bc/data/runs/q3vl_base_sft_20260804}"
NFS_DURABLE="${NFS_DURABLE:-/mnt/nfs/bc/runs/q3vl_base_sft_20260804}"
CONFIG="${CONFIG:-${REPO_ROOT}/q3vl/train/configs/sft_base.yaml}"
MANIFEST=/mnt/nfs/bc/data/datasets/sft2seg-20260804/manifest/terminal_manifest.json
EXPECT_DIGEST=9278d721bb1e234c0a6e7ad94f8f8ae1eab10844602a26a1cfb0e20cab7ac840
EXPECT_N_EFFECTIVE=159215
NEED_GIB=400   # 339 GiB peak resident + headroom

# --- 1. GPUs idle ------------------------------------------------------------
BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
if [ "${BUSY}" -ne 0 ]; then
  echo "REFUSING: ${BUSY} compute process(es) already on the GPUs:" >&2
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv >&2
  exit 1
fi

# --- 2. fresh output dir -----------------------------------------------------
if compgen -G "${RUN_DIR}/checkpoint-*" > /dev/null; then
  echo "REFUSING: ${RUN_DIR} already holds checkpoints. Production training must" >&2
  echo "start fresh; move the old directory aside first." >&2
  exit 1
fi

# --- 3. manifest still the one this config was resolved against --------------
GOT_DIGEST=$(jq -r .digest "${MANIFEST}")
GOT_N=$(jq -r .n_effective "${MANIFEST}")
if [ "${GOT_DIGEST}" != "${EXPECT_DIGEST}" ] || [ "${GOT_N}" != "${EXPECT_N_EFFECTIVE}" ]; then
  echo "REFUSING: terminal manifest changed." >&2
  echo "  digest expected ${EXPECT_DIGEST} got ${GOT_DIGEST}" >&2
  echo "  n_effective expected ${EXPECT_N_EFFECTIVE} got ${GOT_N}" >&2
  echo "Re-resolve the config (steps_per_epoch / protected steps) before launching." >&2
  exit 1
fi

# --- 4. RUN_DIR must be the config's output_dir ------------------------------
CFG_OUT=$(grep -E '^[[:space:]]+output_dir:' "${CONFIG}" | head -1 \
          | sed -E 's/^[[:space:]]*output_dir:[[:space:]]*//' | tr -d "\"' ")
if [ "${CFG_OUT}" != "${RUN_DIR}" ]; then
  echo "REFUSING: RUN_DIR and the config's output_dir disagree." >&2
  echo "  RUN_DIR           = ${RUN_DIR}" >&2
  echo "  config output_dir = ${CFG_OUT}   (${CONFIG})" >&2
  echo "The log would land on one filesystem and the checkpoints on another." >&2
  exit 1
fi

# --- 5. local capacity -------------------------------------------------------
mkdir -p "${RUN_DIR}"
AVAIL_GIB=$(df -BG --output=avail "${RUN_DIR}" | tail -1 | tr -dc '0-9')
if [ "${AVAIL_GIB}" -lt "${NEED_GIB}" ]; then
  echo "REFUSING: ${RUN_DIR} has ${AVAIL_GIB} GiB free, need >= ${NEED_GIB} GiB" >&2
  echo "(peak resident = 6 x 55.03 GiB checkpoints + the final 8.27 GiB model)." >&2
  exit 1
fi

echo "preconditions OK"
echo "  run dir (local): ${RUN_DIR}   [${AVAIL_GIB} GiB free]"
echo "  nfs durable    : ${NFS_DURABLE}   (populated post-training by SYNC_PROTECTED_TO_NFS.sh)"
echo "  config         : ${CONFIG}"
echo "  manifest digest: ${GOT_DIGEST}"
echo "  N_effective    : ${GOT_N}  -> steps_per_epoch 4976, protected 2488 / 4976"
echo "  expected wall  : ~8.2 h (5.078 s/step measured x 4976 + 11 evals + 11 local saves)"
echo

RUN_DIR="${RUN_DIR}" CONFIG="${CONFIG}" exec bash "${REPO_ROOT}/q3vl/train/scripts/launch_sft.sh" "$@"
