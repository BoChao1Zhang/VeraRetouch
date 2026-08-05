# genctx 分片/合并 dry-run 记录

```bash
bash dryrun/run_dryrun.sh      # -> dryrun/dryrun.log
```

**85/85 断言通过，exit 0。全程 `CUDA_VISIBLE_DEVICES=""`，不加载权重、不碰 GPU。**
（唯一加载真权重的是 `cpu_smoke_real_producer.log`，`--device cpu`。）

## 覆盖的内容

| 组 | 条数 | 断言 |
|---|---|---|
| 分片划分 | 26 | 6 组规模 × 2 种切法：并集恰为 `range(n)`、两两不交、最大最小差 ≤ 1；含真实 train 规模 159,215×8 与 n<k 的退化情形；4 种非法参数被拒 |
| 强制隔离 | 7 | `--out-root`/`--report-dir` 都被追加 `shard<i>of<n>`；透传参数原样保留；`--out-root`/`--report-dir`/`--split` 透传被拒；8 个分片 → 8 个互不相同的 out-root |
| CLI | 6 | 三个入口 `--help` 均 exit 0；确认生产脚本**确实没有** `--shard`；`--plan-only` 0.1 s 出计划（连 torch 都没导）；驱动缺 checkpoint 参数 exit 2 |
| 端到端合并 | 14 | 用**真实 V_where 索引（896 条）** + **真实 `publish_generated`** 发两个分片 → `merge_genctx.py` 合并 → `GenContextStore` 读回 896/896、manifest `complete`、报告里覆盖率 1.0 且 checkpoint 唯一、两段统计齐全 |
| CX-1 兼容软链 | 7 | `two_segment` 自链与 `forced_color` 跨链都建立且解析正确；**真实 `ColorGenContextStore` + `assert_covers` 走软链读通**；mode 不匹配照样硬停；两种 mode 落在不同的真实根上 |
| 合并的拒绝路径 | 7 | 目标根已存在 / 分片缺失 / 分片不是它该分到的那一片（`--shard-mode` 用错）/ 一个 split 混了两个 checkpoint / 记录 mode 不是要的那个 —— 全部非零退出且**没有发布任何字节**；`--check-only` 只校验不发布 |
| 驱动 | 18 | DRY_RUN 计划正确（12 个分片作业 = 2+2+8，两卡 6/6，小 split 排在前）；读到真实 split 规模；卡不空闲时拒绝启动、DRY_RUN 下只告警；已完整发布的 split 被摘掉；不完整的最终根让驱动整体停下；`forced_color` wrapper 正确翻转 MODE/SPLITS（10 个作业） |
| 续跑 | 4 | 已完成的分片被跳过；**样本数对不上的"已完成"分片被拒**（`LIMIT=` 标定跑会落在同一批路径上，这是唯一会被误当成正式产物的情况） |
| D-20 机制 | 3 | 用替身解释器实跑一个分片作业：`job.marker` 里记的 PID **等于该进程自己的 `$$`**（证明 `$!` 抓的是工作进程本身、不是包一层的子 shell —— run_where_b.sh 审阅 nit N7 那个坑）；`.rc` 落盘；marker 记录 checkpoint/batch/out_root |

## dry-run 抓到的两个真 bug（已修）

1. **`import q3vl` 会失败**：`python /abs/path/x.py` 把**脚本目录**放进 `sys.path`，`cd $REPO` 不管用
   （只有 `-m` 才把 cwd 加进去）。后果是分片作业在"已上报启动成功"之后才死。
   两个脚本现在都显式把仓库根插进 `sys.path`。
2. **`publish_generated` 自己会调 `genwhere_payload`**，合并时再包一层 payload 会 `TypeError`。

## 真生产脚本的 CPU 实跑（`cpu_smoke_real_producer.log`）

`make_generated_context` 本体 + wrapper，`--split V_where --limit 4 --num-shards 2`，
base 权重、`--device cpu --dtype float32 --attn eager --max-new-tokens 8`，两个分片都 exit 0：

* 分片 0 = `refs[0::2]`、分片 1 = `refs[1::2]`，**实测与计划逐条相符、互不相交、并集 = 4 条**；
* 两个 `manifest.status=complete`、`sample_count=2`、`schema=q3vl.where_b.genwhere/2`、`mode=two_segment`；
* 两份报告落在**不同**目录（`reports/V_where/shard00of02/` 与 `shard01of02/`）——
  没有这个隔离，两个文件同名同路径，后跑的静默覆盖先跑的。
