# T5 · s 缓存服务（INF-5）+ oracle 目录 — 自检报告

日期：2026-08-02 ｜ 规格：EXPERIMENTS_v3 §INF-5 ｜ 代码：`tools/scache/`（全部新建，未动现有代码）

## 目标

读出臂与渲染臂解耦的 s 缓存层：`s_cache/<arm>/<img_id>__<instr_hash>.npy`（float16，默认
32×32，可配原生分辨率）+ 同名 `.meta.json`（层号/归一化参数/生成时间/arm 版本）；
oracle 目录从 l 系 shards 的 C_GT 掩膜面积加权下采样构建，meta 存原掩膜 (shard, offset) 引用；
上采样用 `kornia.filters.guided_blur`（guide=原图）。

## 设置

- 数据：`prod-l1-local17k-20260731`，shard 迭代序（batch-0000 起）前 **100 组**（不同
  group_id），共 **185 个候选条目**（每组 1–2 个带 C_GT 的候选进了 shards）。
- 指令哈希：journal 归档 `sft.jsonl` 的 instruction，md5[:12]；oracle img_id=candidate_id。
- 下采样：PIL BOX（面积加权，float32 'F'）→ 32×32 float16（单文件 2176 B ≈ 2KB/图，合 spec）。
- 上采样：bilinear → `guided_blur`，kernel=13（auto，~scale/4）、eps=1e-4、subsample=1、RGB guide
  （输入图 resize 到掩膜尺寸）；参数经扫描选定（k∈{9,13,25,49,97}×eps{1e-4,3e-4,1e-3}×sub{1,8}×guide{rgb,gray}）。
- IoU：双方 0.5 阈值二值化后 |∩|/|∪|。设备 cuda:0；写 8.0s + 往返 32.6s / 185 条。

## 预注册判据 vs 实测

| 判据 | 预注册 | 实测 | 结果 |
|---|---|---|---|
| 写→读→上采样往返跑通（100 组） | 跑通 | 185/185 条无错 | ✅ |
| IoU@0.5 vs 原 C_GT（mean 口径） | > 0.95 | **0.9695** | ✅ |

分布（n=185）：median **0.9966**，min **0.5800**，max 0.9995；逐条 >0.95 占 85.4%，>0.90 占
88.6%；逐组均值的均值 0.9701。纯 bilinear 对照（无引导）：mean 0.9694 / median 0.9977 —
guided 与 bilinear 在均值上基本打平，guided 在小掩膜上略优（worst 案例 0.580 vs 0.576）。

判据口径说明：任务卡未写明 mean 还是逐张；按 mean 判定（见 tools/scache/NOTES.md 待决策 #4），
逐张达标率一并公布如上。

## 失败案例归因（viz/roundtrip_worst_*.png）

IoU 随掩膜面积单调劣化——**32×32 对小面积软边掩膜是表示极限**，非实现缺陷：

| 掩膜面积占比 | n | mean IoU | min |
|---|---|---|---|
| < 2% | 3 | 0.645 | 0.580 |
| 2–5% | 5 | 0.843 | 0.759 |
| 5–10% | 16 | 0.876 | 0.681 |
| 10–30% | 39 | 0.973 | 0.853 |
| > 30% | 122 | 0.994 | 0.932 |

最差案例（area 1.3%，重度羽化的坐姿人物）：原生分辨率对照显示即使缓存开到 128×128，
bilinear 往返也只到 0.857——羽化坡缓 + 面积小，0.5 等值线对数值误差极敏感。guided filter
无法救回：C_GT 是羽化的胖掩膜，与光度边缘并不重合，强引导（大核/小 eps）反而把质量摊薄、
IoU 更差（k=97 时 worst 掉到 0.356）——这也是 auto kernel 取小核（~scale/4）的依据。

## 结论与建议

1. **判据达成**（mean 0.9695 > 0.95），INF-5 缓存服务可交付使用。
2. 对小掩膜任务（本样本 area<5% 的候选约 4%）建议渲染臂改用 `resolution=64` 或原生分辨率
   目录（API 已支持，`SCache(root, arm, resolution=None)`）；或在 EXPERIMENTS_v3 INF-5 行
   注明"32×32 缓存的适用域：掩膜面积 ≥5%"。
3. guided 上采样在 C_GT oracle 上相对 bilinear 增益甚微（掩膜羽化、与光度边缘不重合）；
   对读出臂产出的 s 场（对齐语义边界）预计增益更大，建议 RO 臂接入后复测同款报告。

## 产物清单

- `metrics.json` — 185 条 per-entry IoU + 汇总（机器可读）。
- `viz/roundtrip_{best,median,worst}_iou*.png` — 三列拼图：原掩膜 / 32×32 缓存 / 上采样恢复。
- `s_cache/oracle/` — 前 100 组 oracle 缓存（185×{npy,meta.json}）。
- 代码：`tools/scache/{api.py,oracle.py,upsample.py,selfcheck.py,README.md,NOTES.md}`。
- 环境：python 3.13（/home/bc/miniconda3）、torch 2.6.0+cu124、kornia 0.8.2、numpy 2.4.6、
  零新增依赖；git 分支 lens-exp。
