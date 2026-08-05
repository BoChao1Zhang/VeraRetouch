#!/usr/bin/env bash
# Dry-run of the genctx sharding + merge path.  CPU only; loads no weights and
# touches no GPU (CUDA_VISIBLE_DEVICES is emptied for every child process).
#
#   bash dryrun/run_dryrun.sh          # -> dryrun/dryrun.log, exit 0 = all pass
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/home/bc/envs/q3vl_sft/bin/python}"
LOG="${HERE}/dryrun.log"

export LD_LIBRARY_PATH="/home/bc/miniconda3/envs/llm_factory/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUDA_VISIBLE_DEVICES=""
export TOKENIZERS_PARALLELISM=false
export PY

rm -f "${LOG}"                       # D-20 step 1 (zsh noclobber)
"${PY}" "${HERE}/test_shard_and_merge.py" 2>&1 | tee "${LOG}"
rc=${PIPESTATUS[0]}
echo "exit_code=${rc}" | tee -a "${LOG}"
exit "${rc}"
