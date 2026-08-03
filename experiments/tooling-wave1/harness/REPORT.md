# REPORT — T3 统一评测 harness（INF-1）

日期：2026-08-02　代码：`tools/harness/`（全部新建，未动仓库既有代码）

## 目标

所有实验臂共用的评测与监控件（EXPERIMENTS_v3 §1 INF-1）：masked PSNR 三分、全图 PSNR
（round×255 口径）、ΔE00（D65）、SSIM、LPIPS（懒加载）、Δ_const/Δ_shuffle 塌陷探针、
σ_s 分布统计（M3）、配对 ΔPSNR + bootstrap 95% CI + 符号检验、metrics.json → markdown 总榜。

## 判据 vs 实测

| 预注册判据（任务卡 T3） | 实测 | 结论 |
|---|---|---|
| 合成自检通过（已知答案对拍到 1e-6） | `selfcheck.py` **42/42 PASS**（解析类 1e-6，恒等/计数类 1e-12），退出码 0 | 达标 |
| README 接口文档齐 | `tools/harness/README.md`：五模块全接口 + 口径表 + metrics.json schema + 已知边界 | 达标 |
| 真实数据小样本跑通（CLAUDE.md 交付规范） | `smoke_real.py` 在 mini30-v52-20260730 shard 的 6 个带 `.cgt.png` 样本上全链路跑通（含 LPIPS） | 达标 |

### 自检覆盖（合成已知答案，全部取值在 1/255 整数格上使 round×255 量化零损失）

- 全图 PSNR：64×64 常量底图 + 32×32 矩形掩膜内 +10/255 偏移 → 解析 34.151404 dB，对拍 1e-6。
- masked 三分（band_px=2）：三区期望 PSNR 用**独立闭式距离公式**（矩形内 min 轴距 / 外
  clamp 欧氏距）算出，逐像素区域图与 `region_partition` 完全一致；in=28.1308 / band=31.3184 /
  out=cap；三区互斥且覆盖全图。
- ΔE00：恒等=0（1e-12）；中性灰对的解析简化 ΔE00=ΔL′/S_L（独立实现 sRGB→L* 与
  CIEDE2000 中性色简化）对拍 1e-6（实测偏差 1.6e-8，来源为 skimage 白点数值精度）。
- SSIM：常量图对解析值 (2ab+C1)/(a²+b²+C1) 对拍 1e-6；恒等=1。
- stats：退化 bootstrap CI=(v,v)；符号检验精确 p 对拍解析值（全正 n=20 → 2⁻¹⁹；混合
  3+/1−/1×0 → 10/16）。
- collapse probes：8 样本玩具渲染器，Δ_const 与 Δ_shuffle 的期望 PSNR 全解析；置换的
  derangement 性质（无不动点）在 seed 0–4 逐一验证（用 spy render_fn 记录实收 s）。
- σ_s 统计：已知 20% 贴上界 → frac_at_upper=0.2；M3 红线（>80%）触发/不触发两例。
- leaderboard：缺 Δ_shuffle 行标 `N/A ⚠` + stderr 警告；psnr_in 降序。

### 真实数据冒烟（mini30-v52-20260730 / shard-00000，6 样本）

输入(.in.jpg) vs GT(.jpg)，掩膜=.cgt.png（软边单通道），band_px=3：

| 指标 | 均值 | 备注 |
|---|---|---|
| PSNR_in / band / out | 23.60 / 28.90 / 38.07 dB | in < band < out，符合"编辑集中在掩膜内"的预期，效应量放大方向正确 |
| PSNR_full | 26.42 dB | 介于三者之间 |
| SSIM / ΔE00 / LPIPS | 0.903 / 4.21 / 0.106 | LPIPS alex 权重本机可用 |
| Δ_const / Δ_shuffle | +7.25 / +5.54 dB | oracle 混合渲染器（s=cgt 下采 32×32）上显著为正，探针方向正确 |
| 配对 ΔPSNR（oracle vs 恒等） | +12.80 dB，CI95 [+10.13, +15.58]，sign p=0.0312 | n=6 全正 → 精确 p=2·0.5⁶=0.03125 ✓ |

产物：`metrics.json`（schema 见 README）、`leaderboard_demo.md`、
`viz/success_smoke_{8975b1e9,bd776045}.png`、`viz/failure_smoke_7b8fdb4c.png`
（psnr_in 最差样本：大面积软渐变掩膜 + 强低光重打光，in-mask 仅 17.2 dB——这类"掩膜≈半图
软渐变"的样本三分区退化为对角切分，band 统计意义变弱，正式实验按掩膜面积分层报告更稳）。

## 设置

- 环境：/home/bc/miniconda3（numpy 2.4.6 / scipy 1.16.3 / skimage 0.25.2 / torch 2.6.0 /
  pillow 12.2.0 / lpips 已装）。**零新增依赖**。
- 数据：仅 QA 用途的小 build mini30-v52-20260730（符合数据纪律：fresh*/mini* 只做 QA）。
- 外部事实核实与假设清单：`tools/harness/NOTES.md`。

## 待主 agent 决策（详见 NOTES.md §四）

1. **band_px 默认 3** 为保守占位——正式实验前应在 EXPERIMENTS_v3 定死一个 k 并全局沿用。
2. **Δ_const 的 s_∅ 口径**：默认全评测集均值常量场；模型自带（condition dropout 学出的）
   s_∅ 时用 `s_null=` 传入；`delta_const_mode` 字段留了口径追溯。
3. **烘焙一致性评测器**（INF-1 原文含）不在任务卡 T3 的 5 文件范围内，未实现——需另开
   任务卡（依赖 INF-3 cube 工具链）。
4. metrics.json schema 为本交付新定（文档无既定 schema），leaderboard 按此解析；如需改
   key 映射集中在 `leaderboard.py` COLUMNS 一处。

## 建议下一步

- 实现审阅后，接首个消费者（A0 GLUT 复现的 480p 评测 + RO 臂 s 缓存质量报告）压接口。
- 若 W1 内做烘焙一致性评测器，可复用本 harness 的 PSNR/ΔE 口径，仅需加 .cube 回读器。
