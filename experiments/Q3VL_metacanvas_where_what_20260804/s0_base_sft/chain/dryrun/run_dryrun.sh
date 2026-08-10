#!/usr/bin/env bash
# =============================================================================
# Dry-run harness for chain_after_sft.sh.
#
# Every case runs the REAL chain script end to end against a fake run directory
# and mock rank processes, with the three jobs replaced by stub_job.sh.  No GPU
# is touched, no production directory is read or written, and the production
# chain/logs/ directory is not used (each case gets its own log dir).
#
#   bash run_dryrun.sh            # all cases
#   bash run_dryrun.sh happy      # one case
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHAIN="${HERE}/../chain_after_sft.sh"
CASES_DIR="${HERE}/cases"
RESULT_MD="${HERE}/DRYRUN_RESULT.md"
DEAD_PID=4194304          # above /proc/sys/kernel/pid_max: never a live process

PASS=0
FAIL=0
declare -a REPORT=()

# --------------------------------------------------------------------------
make_run_dir() {   # make_run_dir <dir> <global_step> <root_state:0|1> <log:clean|traceback|oom> <ckpt:1|0>
  local dir="$1" step="$2" root="$3" logkind="$4" mkckpt="$5"
  rm -rf "${dir}"
  mkdir -p "${dir}"
  if [ "${mkckpt}" = "1" ]; then
    local ck="${dir}/checkpoint-4976"
    mkdir -p "${ck}"
    for f in config.json generation_config.json tokenizer.json tokenizer_config.json \
             special_tokens_map.json added_tokens.json preprocessor_config.json \
             chat_template.jinja; do
      printf '{"mock": true}\n' > "${ck}/${f}"
    done
    printf '{"weight_map": {"a": "model-00001-of-00002.safetensors"}}\n' \
      > "${ck}/model.safetensors.index.json"
    : > "${ck}/model-00001-of-00002.safetensors"
    : > "${ck}/model-00002-of-00002.safetensors"
    printf '{"global_step": %s, "max_steps": 4976, "epoch": 1.0, "log_history": []}\n' \
      "${step}" > "${ck}/trainer_state.json"
  fi
  [ "${root}" = "1" ] && printf '{"global_step": 4976, "log_history": []}\n' \
    > "${dir}/trainer_state.json"
  {
    echo "  99%|#########9| 4970/4976 [8:10:00<00:30,  5.10s/it]"
    case "${logkind}" in
      clean)     echo "training finished: {'train_runtime': 29000.0}" ;;
      traceback) echo "Traceback (most recent call last):"; echo "RuntimeError: boom" ;;
      oom)       echo "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB" ;;
    esac
  } > "${dir}/train.log"
}

start_mock_ranks() {   # start_mock_ranks <seconds> -> echoes the pids
  # The mocks MUST NOT inherit this function's stdout: when it is called inside
  # `$( )` the substitution's pipe stays open until every holder closes it, so
  # an unredirected background child makes the caller block for the full sleep
  # and the "ranks still alive" branch would never be exercised.
  local secs="$1" p1 p2
  bash "${HERE}/mock_train_sft.sh" "${secs}" > /dev/null 2>&1 < /dev/null & p1=$!
  bash "${HERE}/mock_train_sft.sh" "${secs}" > /dev/null 2>&1 < /dev/null & p2=$!
  printf '%s %s' "${p1}" "${p2}"
}

check() {   # check <case> <description> <0=expect-present|1=expect-absent> <pattern> <file>
  local case_name="$1" desc="$2" mode="$3" pat="$4" file="$5" hit
  if grep -a -F -q -- "${pat}" "${file}" 2>/dev/null; then hit=0; else hit=1; fi
  if [ "${hit}" = "${mode}" ]; then
    PASS=$(( PASS + 1 ))
    REPORT+=("| ${case_name} | ${desc} | \`${pat}\` | PASS |")
    printf '    PASS  %s\n' "${desc}"
  else
    FAIL=$(( FAIL + 1 ))
    REPORT+=("| ${case_name} | ${desc} | \`${pat}\` | **FAIL** |")
    printf '    FAIL  %s   (pattern: %s)\n' "${desc}" "${pat}"
  fi
}

check_rc() {   # check_rc <case> <expected> <actual>
  local case_name="$1" want="$2" got="$3"
  if [ "${want}" = "${got}" ]; then
    PASS=$(( PASS + 1 ))
    REPORT+=("| ${case_name} | chain exit code | expected ${want} | PASS |")
    printf '    PASS  chain exit code == %s\n' "${want}"
  else
    FAIL=$(( FAIL + 1 ))
    REPORT+=("| ${case_name} | chain exit code | expected ${want}, got ${got} | **FAIL** |")
    printf '    FAIL  chain exit code: expected %s got %s\n' "${want}" "${got}"
  fi
}

run_chain() {   # run_chain <case> <run_dir> <rank_pids> [extra env assignments...]
  local case_name="$1" run_dir="$2" rank_pids="$3"; shift 3
  local logdir="${CASES_DIR}/${case_name}/logs"
  mkdir -p "${logdir}"
  rm -f "${logdir}/chain.log"                 # D-20 step 1, even in the dry run
  env CHAIN_RUN_DIR="${run_dir}" \
      CHAIN_RANK_PIDS="${rank_pids}" \
      CHAIN_RANK_PATTERN=mock_train_sft \
      CHAIN_LAUNCHER_PID="${DEAD_PID}" \
      CHAIN_LOG_DIR="${logdir}" \
      CHAIN_POLL_SECONDS=2 \
      CHAIN_SETTLE_SECONDS=1 \
      CHAIN_PROBE_TIMEOUT=60 \
      CHAIN_HEARTBEAT_EVERY=1 \
      CHAIN_WAIT_MAX_HOURS=1 \
      CHAIN_MONITOR_MAX_HOURS=1 \
      CHAIN_STUB_JOBS=1 \
      CHAIN_STUB_SECONDS=4 \
      "$@" \
      bash "${CHAIN}" > "${logdir}/chain.log" 2>&1
  printf '%s' "$?"
}

# --------------------------------------------------------------------------
case_happy() {
  local d="${CASES_DIR}/happy"; local run="${d}/run"
  echo "== case happy: ranks still alive, then a clean finish =="
  make_run_dir "${run}" 4976 1 clean 1
  local pids; pids="$(start_mock_ranks 6)"
  local rc; rc="$(run_chain happy "${run}" "${pids}")"
  local log="${d}/logs/chain.log"
  check happy "waited while the ranks were alive"        0 "still training: ranks_alive=2" "${log}"
  check happy "noticed the ranks exiting"                0 "all Base SFT ranks have exited" "${log}"
  check happy "adjudication passed"                      0 "the run completed normally" "${log}"
  check happy "did NOT abort"                            1 "CHAIN-ABORT" "${log}"
  check happy "gpu0 STARTED line"                        0 "CHAIN: gpu0_ckpt_verify STARTED pid=" "${log}"
  check happy "gpu1 STARTED line"                        0 "CHAIN: gpu1_wherea_s2 STARTED pid=" "${log}"
  check happy "cpu STARTED line"                         0 "CHAIN: cpu_sync_maskviews STARTED pid=" "${log}"
  check happy "ALL-DISPATCHED"                           0 "CHAIN: ALL-DISPATCHED started=3 failed_to_start=0" "${log}"
  check happy "per-phase line surfaced"                  0 "CHAIN: gpu0_ckpt_verify/work_a DONE rc=0" "${log}"
  check happy "gpu0 DONE"                                0 "CHAIN: gpu0_ckpt_verify DONE rc=0" "${log}"
  check happy "gpu1 DONE"                                0 "CHAIN: gpu1_wherea_s2 DONE rc=0" "${log}"
  check happy "cpu DONE"                                 0 "CHAIN: cpu_sync_maskviews DONE rc=0" "${log}"
  check happy "ALL-DONE"                                 0 "CHAIN: ALL-DONE done=3 failed=0" "${log}"
  check happy "job.marker written for gpu0"              0 "pid=" "${d}/logs/gpu0_ckpt_verify.job.marker"
  check_rc happy 0 "${rc}"
}

case_abort_step() {
  local d="${CASES_DIR}/abort_step"; local run="${d}/run"
  echo "== case abort_step: checkpoint-4976 carries global_step 3500 =="
  make_run_dir "${run}" 3500 1 clean 1
  local rc; rc="$(run_chain abort_step "${run}" "${DEAD_PID}")"
  local log="${d}/logs/chain.log"
  check abort_step "aborted"          0 "CHAIN: CHAIN-ABORT reason: checkpoint-4976/trainer_state.json global_step=3500 != 4976" "${log}"
  check abort_step "no job started"   1 "STARTED pid=" "${log}"
  check abort_step "GPUs untouched"   0 "the GPUs were left untouched" "${log}"
  check_rc abort_step 2 "${rc}"
}

case_abort_traceback() {
  local d="${CASES_DIR}/abort_traceback"; local run="${d}/run"
  echo "== case abort_traceback: a Traceback in the train log tail =="
  make_run_dir "${run}" 4976 1 traceback 1
  local rc; rc="$(run_chain abort_traceback "${run}" "${DEAD_PID}")"
  local log="${d}/logs/chain.log"
  check abort_traceback "aborted on the traceback" 0 "train log tail contains 'Traceback (most recent call last)'" "${log}"
  check abort_traceback "no job started"           1 "STARTED pid=" "${log}"
  check_rc abort_traceback 2 "${rc}"
}

case_abort_oom() {
  local d="${CASES_DIR}/abort_oom"; local run="${d}/run"
  echo "== case abort_oom: CUDA OOM in the train log tail =="
  make_run_dir "${run}" 4976 1 oom 1
  local rc; rc="$(run_chain abort_oom "${run}" "${DEAD_PID}")"
  local log="${d}/logs/chain.log"
  check abort_oom "aborted on the OOM"  0 "train log tail contains 'CUDA out of memory'" "${log}"
  check abort_oom "no job started"      1 "STARTED pid=" "${log}"
  check_rc abort_oom 2 "${rc}"
}

case_abort_missing_ckpt() {
  local d="${CASES_DIR}/abort_missing_ckpt"; local run="${d}/run"
  echo "== case abort_missing_ckpt: checkpoint-4976 never appeared =="
  make_run_dir "${run}" 4976 1 clean 0
  local rc; rc="$(run_chain abort_missing_ckpt "${run}" "${DEAD_PID}")"
  local log="${d}/logs/chain.log"
  check abort_missing_ckpt "aborted"        0 "CHAIN-ABORT reason: checkpoint-4976 directory is absent" "${log}"
  check abort_missing_ckpt "no job started" 1 "STARTED pid=" "${log}"
  check_rc abort_missing_ckpt 2 "${rc}"
}

case_abort_no_root_state() {
  local d="${CASES_DIR}/abort_no_root_state"; local run="${d}/run"
  echo "== case abort_no_root_state: final save_state() never ran =="
  make_run_dir "${run}" 4976 0 clean 1
  local rc; rc="$(run_chain abort_no_root_state "${run}" "${DEAD_PID}")"
  local log="${d}/logs/chain.log"
  check abort_no_root_state "aborted"        0 "the final save_state() never ran" "${log}"
  check abort_no_root_state "no job started" 1 "STARTED pid=" "${log}"
  check_rc abort_no_root_state 2 "${rc}"
}

case_missing_file() {
  local d="${CASES_DIR}/abort_missing_file"; local run="${d}/run"
  echo "== case abort_missing_file: checkpoint-4976 has no processor config =="
  make_run_dir "${run}" 4976 1 clean 1
  rm -f "${run}/checkpoint-4976/preprocessor_config.json"
  local rc; rc="$(run_chain abort_missing_file "${run}" "${DEAD_PID}")"
  local log="${d}/logs/chain.log"
  check abort_missing_file "aborted"        0 "checkpoint-4976/preprocessor_config.json is missing" "${log}"
  check abort_missing_file "no job started" 1 "STARTED pid=" "${log}"
  check_rc abort_missing_file 2 "${rc}"
}

case_job_failure() {
  local d="${CASES_DIR}/job_failure"; local run="${d}/run"
  echo "== case job_failure: gpu1 exits non-zero =="
  make_run_dir "${run}" 4976 1 clean 1
  local rc; rc="$(run_chain job_failure "${run}" "${DEAD_PID}" CHAIN_STUB_RC_GPU1=3)"
  local log="${d}/logs/chain.log"
  check job_failure "gpu1 phase failure surfaced" 0 "CHAIN: gpu1_wherea_s2/work_b FAILED rc=3" "${log}"
  check job_failure "gpu1 FAILED"                 0 "CHAIN: gpu1_wherea_s2 FAILED rc=3" "${log}"
  check job_failure "gpu0 still DONE"             0 "CHAIN: gpu0_ckpt_verify DONE rc=0" "${log}"
  check job_failure "cpu still DONE"              0 "CHAIN: cpu_sync_maskviews DONE rc=0" "${log}"
  check job_failure "ALL-DONE counts"             0 "CHAIN: ALL-DONE done=2 failed=1" "${log}"
  check_rc job_failure 1 "${rc}"
}

case_pid_reuse() {
  local d="${CASES_DIR}/pid_reuse"; local run="${d}/run"
  echo "== case pid_reuse: the rank pid is alive but is a different program =="
  make_run_dir "${run}" 4976 1 clean 1
  sleep 120 & local other=$!
  local rc; rc="$(run_chain pid_reuse "${run}" "${other}")"
  kill "${other}" 2>/dev/null
  local log="${d}/logs/chain.log"
  check pid_reuse "recycled pid does not read as training" 0 "all Base SFT ranks have exited" "${log}"
  check pid_reuse "chain proceeded to dispatch"            0 "CHAIN: ALL-DISPATCHED" "${log}"
  check_rc pid_reuse 0 "${rc}"
}

# --------------------------------------------------------------------------
ALL_CASES=(happy abort_step abort_traceback abort_oom abort_missing_ckpt
           abort_no_root_state missing_file job_failure pid_reuse)

mkdir -p "${CASES_DIR}"
SELECT=("$@")
[ "${#SELECT[@]}" -eq 0 ] && SELECT=("${ALL_CASES[@]}")

for c in "${SELECT[@]}"; do
  case "${c}" in
    happy)               case_happy ;;
    abort_step)          case_abort_step ;;
    abort_traceback)     case_abort_traceback ;;
    abort_oom)           case_abort_oom ;;
    abort_missing_ckpt)  case_abort_missing_ckpt ;;
    abort_no_root_state) case_abort_no_root_state ;;
    missing_file)        case_missing_file ;;
    job_failure)         case_job_failure ;;
    pid_reuse)           case_pid_reuse ;;
    *) echo "unknown case ${c}"; FAIL=$(( FAIL + 1 )) ;;
  esac
done

{
  echo "# chain_after_sft.sh dry-run record"
  echo
  echo "生成于 \`$(date -Is)\`，主机 \`$(hostname)\`。"
  echo
  echo "每个 case 都跑**真实的** \`chain_after_sft.sh\`：真实的等待循环、真实的完成判定、"
  echo "真实的 D-20 四步派发与监控。只有三个作业被换成 \`dryrun/stub_job.sh\`（同一套"
  echo "PROBE-OK / phase-rc / job-rc 协议），rank 进程被换成 \`dryrun/mock_train_sft.sh\`，"
  echo "run 目录是 \`dryrun/cases/<case>/run\` 下的假目录。**未使用任何 GPU，未读写任何生产目录。**"
  echo
  echo "| case | 检查 | 断言 | 结果 |"
  echo "|---|---|---|---|"
  printf '%s\n' "${REPORT[@]}"
  echo
  echo "**合计：${PASS} passed / ${FAIL} failed**"
  echo
  echo "每个 case 的完整 chain.log 在 \`dryrun/cases/<case>/logs/chain.log\`。"
} > "${RESULT_MD}"

echo
echo "==== ${PASS} passed / ${FAIL} failed ===="
echo "record: ${RESULT_MD}"
[ "${FAIL}" -eq 0 ]
