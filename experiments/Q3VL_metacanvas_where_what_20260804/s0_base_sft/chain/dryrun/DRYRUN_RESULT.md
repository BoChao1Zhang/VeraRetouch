# chain_after_sft.sh dry-run record

生成于 `2026-08-05T10:14:35+08:00`，主机 `h3c`。

每个 case 都跑**真实的** `chain_after_sft.sh`：真实的等待循环、真实的完成判定、
真实的 D-20 四步派发与监控。只有三个作业被换成 `dryrun/stub_job.sh`（同一套
PROBE-OK / phase-rc / job-rc 协议），rank 进程被换成 `dryrun/mock_train_sft.sh`，
run 目录是 `dryrun/cases/<case>/run` 下的假目录。**未使用任何 GPU，未读写任何生产目录。**

| case | 检查 | 断言 | 结果 |
|---|---|---|---|
| happy | waited while the ranks were alive | `still training: ranks_alive=2` | PASS |
| happy | noticed the ranks exiting | `all Base SFT ranks have exited` | PASS |
| happy | adjudication passed | `the run completed normally` | PASS |
| happy | did NOT abort | `CHAIN-ABORT` | PASS |
| happy | gpu0 STARTED line | `CHAIN: gpu0_ckpt_verify STARTED pid=` | PASS |
| happy | gpu1 STARTED line | `CHAIN: gpu1_wherea_s2 STARTED pid=` | PASS |
| happy | cpu STARTED line | `CHAIN: cpu_sync_maskviews STARTED pid=` | PASS |
| happy | ALL-DISPATCHED | `CHAIN: ALL-DISPATCHED started=3 failed_to_start=0` | PASS |
| happy | per-phase line surfaced | `CHAIN: gpu0_ckpt_verify/work_a DONE rc=0` | PASS |
| happy | gpu0 DONE | `CHAIN: gpu0_ckpt_verify DONE rc=0` | PASS |
| happy | gpu1 DONE | `CHAIN: gpu1_wherea_s2 DONE rc=0` | PASS |
| happy | cpu DONE | `CHAIN: cpu_sync_maskviews DONE rc=0` | PASS |
| happy | ALL-DONE | `CHAIN: ALL-DONE done=3 failed=0` | PASS |
| happy | job.marker written for gpu0 | `pid=` | PASS |
| happy | chain exit code | expected 0 | PASS |
| abort_step | aborted | `CHAIN: CHAIN-ABORT reason: checkpoint-4976/trainer_state.json global_step=3500 != 4976` | PASS |
| abort_step | no job started | `STARTED pid=` | PASS |
| abort_step | GPUs untouched | `the GPUs were left untouched` | PASS |
| abort_step | chain exit code | expected 2 | PASS |
| abort_traceback | aborted on the traceback | `train log tail contains 'Traceback (most recent call last)'` | PASS |
| abort_traceback | no job started | `STARTED pid=` | PASS |
| abort_traceback | chain exit code | expected 2 | PASS |
| abort_oom | aborted on the OOM | `train log tail contains 'CUDA out of memory'` | PASS |
| abort_oom | no job started | `STARTED pid=` | PASS |
| abort_oom | chain exit code | expected 2 | PASS |
| abort_missing_ckpt | aborted | `CHAIN-ABORT reason: checkpoint-4976 directory is absent` | PASS |
| abort_missing_ckpt | no job started | `STARTED pid=` | PASS |
| abort_missing_ckpt | chain exit code | expected 2 | PASS |
| abort_no_root_state | aborted | `the final save_state() never ran` | PASS |
| abort_no_root_state | no job started | `STARTED pid=` | PASS |
| abort_no_root_state | chain exit code | expected 2 | PASS |
| abort_missing_file | aborted | `checkpoint-4976/preprocessor_config.json is missing` | PASS |
| abort_missing_file | no job started | `STARTED pid=` | PASS |
| abort_missing_file | chain exit code | expected 2 | PASS |
| job_failure | gpu1 phase failure surfaced | `CHAIN: gpu1_wherea_s2/work_b FAILED rc=3` | PASS |
| job_failure | gpu1 FAILED | `CHAIN: gpu1_wherea_s2 FAILED rc=3` | PASS |
| job_failure | gpu0 still DONE | `CHAIN: gpu0_ckpt_verify DONE rc=0` | PASS |
| job_failure | cpu still DONE | `CHAIN: cpu_sync_maskviews DONE rc=0` | PASS |
| job_failure | ALL-DONE counts | `CHAIN: ALL-DONE done=2 failed=1` | PASS |
| job_failure | chain exit code | expected 1 | PASS |
| pid_reuse | recycled pid does not read as training | `all Base SFT ranks have exited` | PASS |
| pid_reuse | chain proceeded to dispatch | `CHAIN: ALL-DISPATCHED` | PASS |
| pid_reuse | chain exit code | expected 0 | PASS |

**合计：43 passed / 0 failed**

每个 case 的完整 chain.log 在 `dryrun/cases/<case>/logs/chain.log`。
