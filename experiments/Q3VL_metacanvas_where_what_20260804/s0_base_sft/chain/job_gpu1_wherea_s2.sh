#!/usr/bin/env bash
# GPU1 job: Where-A step S2 -- the GPU-level preflight plus the D5 sweep.
# Definition of S2: experiments/.../where_a/PREFLIGHT_WHERE_A_PENDING.md §S2.
#
#   phase preflight        protocol 14 items 4/5/6 on the real checkpoint
#                          -> WA-P4b frozen-vision-leak check (SKIP until now),
#                             WA-P4e real F_pre / guided-upsample high-res path,
#                             WA-P5 basis conditioning + the L-BFGS throughput
#                             field that S5's wall clock depends on
#                          -> experiments/.../where_a/preflight_where_a.json
#   phase sweep_upsample   D5: radius_low x eps, lexicographic selection
#                          (domain gate S_OOD_FRAC_MAX first, then soft-IoU)
#                          -> experiments/.../where_a/d5_upsample_sweep.json
#
# F_pre is invariant to the SFT (patch_embed / pos_embed / all 24 vision blocks
# are frozen -- q3vl/train/freeze.py, echoed in q3vl/where/config.py), so this
# can use checkpoint-4976 without waiting for any checkpoint-selection verdict.
#
# DECISION (recorded in chain/NOTES.md): the sweep runs even if the preflight
# exits non-zero.  WA-P4e can legitimately fail on the pre-registered
# S_OOD_FRAC_MAX gate, and the sweep is the very tool that re-fixes D5; both
# rcs are recorded separately so nothing is hidden.
#
# sqlite3 in the campaign env needs conda's libstdc++ (CXXABI_1.3.15); without
# LD_LIBRARY_PATH the mask locator cannot open a build catalog at all.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=joblib.sh
source "${HERE}/joblib.sh"

REPO="${REPO:-/home/bc/VeraRetouch}"
PY="${PY:-/home/bc/envs/q3vl_sft/bin/python}"
CKPT="${CKPT:-/home/bc/data/runs/q3vl_base_sft_20260804/checkpoint-4976}"
PREFLIGHT_LIMIT="${PREFLIGHT_LIMIT:-32}"
SWEEP_LIMIT="${SWEEP_LIMIT:-24}"
JOB_DIR_IN="${JOB_DIR_IN:-${HERE}/logs}"
PROBE_ONLY="${PROBE_ONLY:-0}"     # smoke test: stop after the probe

export LD_LIBRARY_PATH="/home/bc/miniconda3/envs/llm_factory/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
# `-` not `:-`: CHAIN_GPU1="" must mean "no GPU at all" for the no-GPU smoke test.
export CUDA_VISIBLE_DEVICES="${CHAIN_GPU1-1}"
export TOKENIZERS_PARALLELISM=false

job_init "gpu1_wherea_s2" "${JOB_DIR_IN}"

probe() {
  say "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  say "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
  cd "${REPO}" || return 1
  "${PY}" - "${CKPT}" <<'PY' || return 1
import pathlib, sqlite3, sys, torch
from q3vl.where.config import MODEL_DIR, S_DOMAIN, S_OOD_FRAC_MAX, GUIDED_PARAMS_PROVISIONAL
ckpt = pathlib.Path(sys.argv[1])
print(f"sqlite3={sqlite3.sqlite_version}")          # CXXABI check: no LD_LIBRARY_PATH, no mask locator
print(f"torch={torch.__version__}")
print(f"base_model_dir={MODEL_DIR} exists={MODEL_DIR.is_dir()}")
shards = sorted(ckpt.glob('model-*.safetensors'))
print(f"checkpoint={ckpt} exists={ckpt.is_dir()} shards={len(shards)}")
if not ckpt.is_dir() or not shards:
    raise SystemExit("checkpoint unusable")
# the s-cache consumption contract: state the expected domain out loud
print(f"S_DOMAIN={S_DOMAIN} S_OOD_FRAC_MAX={S_OOD_FRAC_MAX} "
      f"GUIDED_PARAMS_PROVISIONAL={GUIDED_PARAMS_PROVISIONAL}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("no CUDA device visible -- refusing to start the GPU job")
p = torch.cuda.get_device_properties(0)
print(f"cuda_device_0={p.name} total_mem_gib={p.total_memory/2**30:.1f}")
PY
  probe_ok "gpu1 ready; sqlite3 + checkpoint + base model reachable"
}

phase probe probe || { say "probe failed -- not starting Where-A S2"; exit "${JOB_RC}"; }
[ "${PROBE_ONLY}" = "1" ] && { say "PROBE_ONLY=1 -- stopping before Where-A S2"; exit "${JOB_RC}"; }

cd "${REPO}" || exit 1

phase preflight "${PY}" -m q3vl.where.preflight \
  --device cuda --limit "${PREFLIGHT_LIMIT}" --checkpoint "${CKPT}"
pf_rc=$?
say "preflight rc=${pf_rc} (sweep runs regardless -- see the header note)"

phase sweep_upsample "${PY}" -m q3vl.where.scripts.sweep_upsample \
  --limit "${SWEEP_LIMIT}" --checkpoint "${CKPT}"

say "deliverables:"
ls -la "${REPO}/experiments/Q3VL_metacanvas_where_what_20260804/where_a/preflight_where_a.json" \
       "${REPO}/experiments/Q3VL_metacanvas_where_what_20260804/where_a/d5_upsample_sweep.json" 2>&1 || true

exit "${JOB_RC}"
