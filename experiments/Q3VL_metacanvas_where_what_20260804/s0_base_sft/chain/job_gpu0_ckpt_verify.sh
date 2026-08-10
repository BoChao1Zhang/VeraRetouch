#!/usr/bin/env bash
# GPU0 job: offline verification of the two protected checkpoints.
#
# Produces
#   experiments/.../s0_base_sft/CHECKPOINT_VERIFICATION.md
#   experiments/.../s0_base_sft/checkpoint_verification.json
#   experiments/.../s0_base_sft/checkpoint_verification_samples.jsonl
#
# Single GPU, no DeepSpeed -- main-agent ruling D-J3 moved the spec 8.3
# structural rates offline precisely because ZeRO-3 `generate` was never
# validated.  ~9 GiB of weights per checkpoint, bf16, batched greedy decoding
# with KV cache.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=joblib.sh
source "${HERE}/joblib.sh"

REPO="${REPO:-/home/bc/VeraRetouch}"
PY="${PY:-/home/bc/envs/q3vl_sft/bin/python}"
RUN_DIR="${RUN_DIR:-/home/bc/data/runs/q3vl_base_sft_20260804}"
STEPS="${STEPS:-2488,4976}"
N_SAMPLES="${N_SAMPLES:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-448}"   # GT target max measured at 326 tokens
BATCH_SIZE="${BATCH_SIZE:-8}"
SEED="${SEED:-42}"
OUT_DIR="${OUT_DIR:-${REPO}/experiments/Q3VL_metacanvas_where_what_20260804/s0_base_sft}"
JOB_DIR_IN="${JOB_DIR_IN:-${HERE}/logs}"
PROBE_ONLY="${PROBE_ONLY:-0}"     # smoke test: stop after the probe

# `-` not `:-` on purpose: CHAIN_GPU0="" must mean "no GPU at all" (that is how
# the no-GPU smoke test is run).  With `:-` an empty value silently fell back to
# card 0, which is exactly how a supposedly GPU-free test grabbed a training GPU.
export CUDA_VISIBLE_DEVICES="${CHAIN_GPU0-0}"
export TOKENIZERS_PARALLELISM=false

job_init "gpu0_ckpt_verify" "${JOB_DIR_IN}"

probe() {
  say "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  "${PY}" - "${RUN_DIR}" "${STEPS}" <<'PY' || return 1
import sys, pathlib, torch, transformers
run, steps = pathlib.Path(sys.argv[1]), sys.argv[2].split(",")
print(f"python={sys.executable}")
print(f"torch={torch.__version__} transformers={transformers.__version__}")
# file checks come first so a no-GPU smoke test still exercises every path
for s in steps:
    d = run / f"checkpoint-{s.strip()}"
    shards = sorted(d.glob("model-*.safetensors")) + sorted(d.glob("model.safetensors"))
    gib = sum(p.stat().st_size for p in shards) / 2**30
    print(f"checkpoint-{s.strip()}: exists={d.is_dir()} weight_shards={len(shards)} weight_gib={gib:.2f}")
    if not d.is_dir() or not shards:
        raise SystemExit(f"checkpoint-{s.strip()} unusable")
print(f"cuda_available={torch.cuda.is_available()} n_devices={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    raise SystemExit("no CUDA device visible -- refusing to start the GPU job")
p = torch.cuda.get_device_properties(0)
print(f"cuda_device_0={p.name} total_mem_gib={p.total_memory/2**30:.1f}")
PY
  probe_ok "gpu0 ready; checkpoints readable"
}

phase probe probe || { say "probe failed -- not starting the verification"; exit "${JOB_RC}"; }
[ "${PROBE_ONLY}" = "1" ] && { say "PROBE_ONLY=1 -- stopping before the verification"; exit "${JOB_RC}"; }

phase verify env PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PY}" "${HERE}/verify_checkpoints.py" \
    --run-dir "${RUN_DIR}" \
    --steps "${STEPS}" \
    --n "${N_SAMPLES}" \
    --seed "${SEED}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --batch-size "${BATCH_SIZE}" \
    --device cuda:0 \
    --out-dir "${OUT_DIR}"

say "deliverables:"
ls -la "${OUT_DIR}/CHECKPOINT_VERIFICATION.md" \
       "${OUT_DIR}/checkpoint_verification.json" \
       "${OUT_DIR}/checkpoint_verification_samples.jsonl" 2>&1 || true

exit "${JOB_RC}"
