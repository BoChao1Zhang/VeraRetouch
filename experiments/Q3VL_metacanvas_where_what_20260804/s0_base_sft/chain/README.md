# SFT 收尾链（EXEC-1）

Base SFT 一退出，两张 H100 立刻接上后续作业。判定训练是否正常结束、再按 D-20 四步逐个派工，
之后常驻监控并把每个作业的状态写进 `logs/chain.log`。

```
chain_after_sft.sh          等待 → 完成判定 → 派三个作业 → 监控（本目录唯一的入口）
joblib.sh                   作业侧公共协议：PROBE-OK / phase rc / job rc
job_gpu0_ckpt_verify.sh     GPU0：两个 protected checkpoint 的离线验证
verify_checkpoints.py         └ 实际干活的代码（生成诊断 + eval_loss 汇总 + 报告）
job_gpu1_wherea_s2.sh       GPU1：Where-A S2（preflight + D5 sweep）
job_cpu_sync_maskviews.sh   CPU/IO：先 NFS 同步，后 Where-A S1 五个 split 打包
dryrun/run_dryrun.sh        链逻辑的 dry-run 测试（9 个 case / 43 条断言）
dryrun/DRYRUN_RESULT.md     测试记录
NOTES.md                    核实记录 / 假设 / 待主 agent 决策 / 墙钟预估
logs/                       chain.log、各作业日志、job.marker、rc 文件
```

## 监控

```bash
tail -f  experiments/Q3VL_metacanvas_where_what_20260804/s0_base_sft/chain/logs/chain.log
grep '^CHAIN:' experiments/Q3VL_metacanvas_where_what_20260804/s0_base_sft/chain/logs/chain.log
```

状态行格式见 NOTES.md §7。`CHAIN-ABORT` 出现即代表**一个作业都没起**，两张卡保持空闲，
需要人来看训练到底怎么了。

## 产出落点

| 作业 | 产出 |
|---|---|
| gpu0 | `s0_base_sft/CHECKPOINT_VERIFICATION.md`、`checkpoint_verification.json`、`checkpoint_verification_samples.jsonl` |
| gpu1 | `where_a/preflight_where_a.json`、`where_a/d5_upsample_sweep.json` |
| cpu | `s0_base_sft/checkpoint_sync_record.json`、`/mnt/nfs/bc/runs/q3vl_base_sft_20260804/`、`/mnt/nfs/bc/data/datasets/where_a-20260805/maskviews/<split>/`、`where_a/maskviews/<split>.report.json` |

## 重跑单个作业

三个 wrapper 都可以脱离链单独跑，环境变量都有默认值：

```bash
bash job_gpu0_ckpt_verify.sh                       # 默认 GPU0，两个 checkpoint，64 条
N_SAMPLES=128 bash job_gpu0_ckpt_verify.sh         # 加大抽样
PROBE_ONLY=1 CHAIN_GPU0="" bash job_gpu0_ckpt_verify.sh   # 无 GPU 冒烟
SKIP_SYNC=1 bash job_cpu_sync_maskviews.sh         # 只做 maskviews
```

⚠ `extract_maskviews` 的发布是原子的：目标目录已存在会硬报错。重跑必须先把旧目录**挪走**，
不要先删。
