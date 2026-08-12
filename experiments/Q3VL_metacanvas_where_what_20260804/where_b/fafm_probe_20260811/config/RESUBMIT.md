# FAFM_PROBE 重提交命令(逐字复用)

> 背景:2026-08-11,P1 因 bug 重启导致队列次序颠倒,FAFM_PROBE 抢在 AMORT_P1 之前
> 进入 gpu0 并停在 `gate-wait`(占着卡槽但零算力),后在 P3' 停臂窗口真正开跑约 3 分钟。
> 经版本指纹判定其加载的源码早于实现审阅 B3–B8 修复(日志缺 B4 新增的
> `c* domain asserted` 行),已 `q cancel`;AMORT_P1 随即接卡,次序自然恢复。
>
> **已于 2026-08-11 02:5x 用下方命令重新提交(id 18 → reorder 后为队首)**,
> 代码含 U5/U6 + B3–B9 全部修复,`pytest` 702 项全过。
>
> **重提交必须逐字复用下面的参数。** 三个参数任一改动都会静默改变行为:
> `--gate` 决定它等哪个产物、`--ready` 决定「真的起来了」怎么判、payload 决定跑什么。

```bash
q submit FAFM_PROBE 0 /home/bc/data/runs/where_b/fafm_probe_20260811.log \
  --desc 'Where-B proposal B FAFM probe: 6 arms, 9 pre-registered criteria, CFG rule (Gate 0 PASS)' \
  --gate /home/bc/data/runs/where_b/fafm_cache_20260811/manifest.json \
  --ready 'M params' --ready-timeout 10800 \
  -- bash /home/bc/agent-gpu-queue/waves/fafm_probe.sh 0
```

## 参数为什么是这些

| 参数 | 值 | 理由 |
|---|---|---|
| 卡 | `0` | 与 Gate 0 同卡无要求,任一卡可;改卡只需改末尾的 `fafm_probe.sh <n>` **和** submit 的卡号,**两处必须一致** |
| `--gate` | 训练缓存的 `manifest.json` | manifest **只在预计算全部完成时才写出**,所以它等价于「缓存就绪」。本地路径,不在 `/mnt/nfs` 下(队列纪律:`test -e` 打死挂载会进 D 态) |
| `--ready` | `M params` | 匹配日志里的 `[fafm] 32.0M params, ...`,证明训练**真的起来了**而不只是进程存在 |
| payload | `waves/fafm_probe.sh 0` | 脚本自己 `export LD_LIBRARY_PATH`(`q` 用 `env -i` 清环境,不会继承);脚本 `exec` python,**不 nohup/setsid**,以便 `q cancel` 能真正停掉它 |

## 重提交时的状态

预计算 manifest 预计 **2026-08-11 ~01:50** 落盘(17000/20000 时 ETA 11 min,0 失败)。
若重提交发生在此之后,**gate 立即满足,不会有 gate-wait 占槽**;
若之前,它会照常 gate-wait —— 所以**建议排在 AMORT_P1 之后再提交**,这也正是本次调整的目的。

## 无需改动的东西

代码、wave 脚本、两份缓存、九条预注册判据、CFG 规则、判据 6 的天花板归一化与颈部税列
**全部已就位且已冒烟验证**,重提交不涉及任何代码变更。
