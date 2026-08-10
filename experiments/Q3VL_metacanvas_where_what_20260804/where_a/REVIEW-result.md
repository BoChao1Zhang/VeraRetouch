# REVIEW-result · Stage-Where-A basis calibration 四臂（BA-0/1/2/3）

审阅日期：2026-08-06 ｜ 审阅角色：结果审阅 agent（只读交付文件夹与权威文档，未读实现代码，未与实现者沟通）
交付物：`/home/bc/data/runs/where_a/{BA-0-Fixed,BA-1-Band,BA-2-CBand12,BA-3-Joint}/`、
`/mnt/nfs/bc/data/datasets/where_a-20260805/{basis,oracle,maskviews}/`、
`experiments/Q3VL_metacanvas_where_what_20260804/where_a/`
权威文档：`docs/METACANVAS_WHERE_WHAT_FINAL_EXPERIMENT_PROTOCOL_2026-08-04.md` §4.4 / §5.6 / §10.2 / §16 / §17.2(A-5)；`CLAUDE.md`「AUC 全实验禁用」「空间场可视化纪律」「s 缓存消费契约」

**总判定：科学结论成立且可信，交付形式不合规，且有 3 个 blocker 会在 Where-B 消费时静默出错。**
四臂全部跑完（339 min，串行单卡），拟合成功率 99.99%+，分层如实。
但 **REPORT.md / metrics.json / viz/ / config/ 四项交付物全部缺失**（CLAUDE.md 交付物规范），
且 §5.6 gate 需要的三个分母里有两个 Where-A 没产出、一个口径未绑定。

---

## 一、本次审阅自算的证据

`eval_V_where.json` 只有聚合量，**无法回答"配对增益"**。审阅侧改从
`/mnt/nfs/bc/data/datasets/where_a-20260805/oracle/<arm>/V_where/shards/shard-00000.tar`
读逐样本 `.oracle.json`（四臂各 400 条，**sample_id 完全一致，交集 400/400**），
用项目约定的最近秩百分位（`idx = clip(ceil(q*k)−1, 0, k−1)`）自行复算。

聚合量与 `eval_V_where.json` 逐位对得上（median/mean/p10 四臂 × 2 readout × 2 分辨率全部一致），
**交付的聚合数字本身可信**。以下所有 Δ 与 p 值均为本审阅新算，交付里没有。

---

## 二、四臂天花板并排表（问题 1）

口径：soft-IoU = **min/max 形式**（`soft_iou_minmax`）；headline = `winner_confidence == normal`，**n = 224**；
`low` = `F_pre` 网格（32×48 量级），`hi` = 交付分辨率（短边 512）。

| Arm | readout | 分辨率 | median | mean | p10 | p90 | min |
|---|---|---|---:|---:|---:|---:|---:|
| BA-0-Fixed | band | low | 0.9669 | 0.9302 | 0.8015 | 0.9865 | 0.6416 |
| BA-0-Fixed | band | **hi** | **0.9611** | 0.9177 | 0.7843 | 0.9884 | 0.1758 |
| BA-0-Fixed | cband12 | low | 0.9749 | 0.9378 | 0.8266 | 0.9881 | 0.6493 |
| BA-0-Fixed | cband12 | **hi** | **0.9714** | 0.9264 | 0.8088 | 0.9881 | 0.2743 |
| BA-1-Band | band | low | 0.9724 | 0.9448 | 0.8549 | 0.9874 | 0.6176 |
| BA-1-Band | band | **hi** | **0.9623** | 0.9284 | 0.8308 | 0.9869 | 0.3287 |
| BA-1-Band | cband12 | low | 0.9822 | 0.9522 | 0.8598 | 0.9917 | 0.6185 |
| BA-1-Band | cband12 | **hi** | **0.9718** | 0.9348 | 0.8264 | 0.9890 | 0.2341 |
| BA-2-CBand12 | band | low | 0.9724 | 0.9445 | 0.8484 | 0.9871 | 0.6036 |
| BA-2-CBand12 | band | **hi** | **0.9623** | 0.9277 | 0.8213 | 0.9869 | 0.1169 |
| BA-2-CBand12 | cband12 | low | 0.9827 | 0.9513 | 0.8513 | 0.9915 | 0.5954 |
| BA-2-CBand12 | cband12 | **hi** | **0.9736** | 0.9345 | 0.8288 | 0.9888 | 0.1712 |
| **BA-3-Joint** | band | low | 0.9727 | 0.9453 | 0.8547 | 0.9869 | 0.5590 |
| **BA-3-Joint** | band | **hi** | **0.9624** | 0.9297 | 0.8258 | 0.9870 | 0.3534 |
| **BA-3-Joint** | cband12 | low | 0.9822 | 0.9527 | 0.8612 | 0.9916 | 0.6648 |
| **BA-3-Joint** | cband12 | **hi** | **0.9724** | 0.9355 | 0.8324 | 0.9896 | 0.2853 |

**要点**：四臂的 median 落在 0.961–0.983 的一个 0.02 宽的带子里；
**hi < low 是普遍现象**（median 落差 −0.003 ～ −0.008），说明 guided upsample 在交付分辨率上
略有损耗，不像 D5 sweep 在 24 样本上看到的"cband12 落差为负"。D5 的那条结论**在全量上不成立**，
但落差量级很小、不影响使用。

---

## 三、配对增益：B 校准到底买到了什么（问题 1 后半 + 问题 2）

同一 400 张图、同一 GT mask、同一 `F_pre`，四臂唯一差别是 `B`，因此可以做严格配对。
`Δmean`/`Δmedian` = 逐样本差的均值/中位；`p` = 2000 次 sign-flip 置换检验（对配对差的均值）；
`CI95` = 2000 次配对 bootstrap 的中位差 95% 区间；`win%` = Δ>0 的样本比例。headline n=224。

| 对比 | readout | 分辨率 | Δmean | **Δmedian** | CI95(median) | p | win% |
|---|---|---|---:|---:|---|---:|---:|
| **BA-3 − BA-0** | band | low | +0.0151 | **+0.0027** | [+0.0016, +0.0035] | 0.0005 | 73.2 |
| **BA-3 − BA-0** | band | **hi** | +0.0120 | **−0.0000** | [−0.0003, +0.0003] | 0.0005 | **49.6** |
| **BA-3 − BA-0** | cband12 | low | +0.0150 | **+0.0064** | [+0.0053, +0.0074] | 0.0005 | 86.2 |
| **BA-3 − BA-0** | cband12 | **hi** | +0.0091 | **+0.0030** | [+0.0021, +0.0039] | 0.0005 | 73.7 |
| BA-1 − BA-0 | band | low | +0.0146 | +0.0023 | [+0.0017, +0.0033] | 0.0005 | 75.4 |
| BA-1 − BA-0 | cband12 | low | +0.0144 | +0.0053 | [+0.0041, +0.0066] | 0.0005 | 82.1 |
| BA-2 − BA-0 | band | low | +0.0143 | +0.0021 | [+0.0015, +0.0032] | 0.0005 | 73.2 |
| BA-2 − BA-0 | cband12 | low | +0.0135 | +0.0056 | [+0.0042, +0.0073] | 0.0005 | 79.5 |
| **BA-3 − BA-1** | band | low | +0.0005 | +0.0002 | [+0.0000, +0.0004] | **0.51** | 56.7 |
| **BA-3 − BA-1** | cband12 | hi | +0.0008 | +0.0002 | [−0.0002, +0.0007] | **0.45** | 53.6 |
| **BA-3 − BA-2** | cband12 | low | +0.0015 | +0.0003 | [−0.0002, +0.0011] | **0.062** | 54.0 |
| **BA-3 − BA-2** | band | hi | +0.0020 | −0.0000 | [−0.0002, +0.0001] | **0.17** | 49.6 |
| **BA-1 − BA-2** | band | low | +0.0003 | −0.0000 | [−0.0003, +0.0002] | **0.77** | 49.1 |
| **BA-1 − BA-2** | cband12 | hi | +0.0003 | +0.0002 | [−0.0002, +0.0007] | **0.82** | 53.1 |

（完整 24 行在审阅脚本输出里；上表取全部有结论意义的行。）

### 三个可以写进论文的归因结论

**(1) 校准是"修尾巴"，不是"抬中位数"。**
`Δmean`（+0.009 ~ +0.015）比 `Δmedian`（−0.000 ~ +0.006）大 2–50 倍，说明增益全部集中在下分位。
p10 的提升才是真实收益：

| | band/hi p10 | cband12/hi p10 |
|---|---:|---:|
| BA-0（未校准） | 0.7843 | 0.8088 |
| BA-3（联合校准） | **0.8258 (+0.0415)** | **0.8324 (+0.0236)** |

**band 在交付分辨率上的中位增益是 0.0000，win% 49.6%——就是掷硬币。**
任何"校准提升了 band 天花板"的说法只能限定在 p10 / mean，不能说中位。

**(2) 收益来自"训练 B 这件事本身"，与哪个 readout 驱动、是否共享无关。**
- BA-1（只被 R-Band 反传）把 **cband12** 的 low 中位抬了 +0.0053；
  BA-2（只被 R-CBand12 反传）把 **band** 的 low 中位抬了 +0.0021。**跨 readout 迁移几乎无损。**
- BA-1 vs BA-2 在四个格子上全部统计不可区分（p = 0.32 ~ 0.82，|Δmedian| ≤ 0.0002）。
- BA-3 vs BA-1 / BA-2 全部统计不可区分（p = 0.062 ~ 0.51，|Δmedian| ≤ 0.0006）。

**几何证据（审阅从 `B.npy` 直接算）**：

| | ‖B−B₀‖_F / ‖B₀‖_F | 与 B₀ 子空间最小主余弦 | 正交误差 | cond |
|---|---:|---:|---:|---:|
| BA-0 | 0 | 1.000 | 4.3e-7 | 1.0000 |
| BA-1 | 0.236 | 0.6228 | 0.0710 | 1.783 |
| BA-2 | 0.240 | 0.6238 | 0.0714 | 1.934 |
| BA-3 | **0.264** | **0.5756** | 0.0824 | 2.002 |

| 两两之间 | ‖Δ‖_F | 最小主余弦 | 中位主余弦 |
|---|---:|---:|---:|
| BA-1 vs BA-2 | 1.032 | **0.9427** | 0.9969 |
| BA-1 vs BA-3 | 0.954 | **0.9424** | 0.9974 |
| BA-2 vs BA-3 | 0.819 | **0.9557** | 0.9973 |

三个校准臂各自把 64 维子空间从起点转开了不少（最小主余弦 0.58–0.62），
**但彼此几乎落在同一个子空间里（最小主余弦 0.94–0.96，中位 0.997）**。
**这就是"共享 projector 的收益来自哪里"的答案：来自校准信号本身是 readout-无关的，
三条路径收敛到同一个解；联合校准（BA-3）没有额外贡献。**

**(3) 训练侧与评测侧一致，无过拟合、也无更大空间。**
训练 loss（`1 − softIoU`，train 全量 75,544 含 low）：
BA-1 0.1260→0.1071、BA-2 0.1280→0.1129、BA-3 0.1262→0.1108（前 50 步均值 → 后 50 步均值）。
即 train 侧 mean soft-IoU +0.015 ~ +0.019，与 V_where 的 `Δmean` +0.009 ~ +0.015 同量级。
一个 epoch 的 cosine 已退火到 `lr=0`、`grad_norm` 降到 0.044–0.071，**继续训不会有质变**。

### 对问题 4 的直接回答：诚实的写法

> **"F_pre + Phi-64 在 `V_where` 上的 oracle mask 表达力已经接近饱和：
> 未校准的 seeded-orthogonal projector 就能达到交付分辨率中位 soft-IoU 0.961(band)/0.971(cband12)；
> 用 75,544 个 local 样本校准 `B` 之后，中位数只再涨 0.000(band)/0.003(cband12)，
> 真实收益集中在下分位（p10 +0.042/+0.024）。
> 三种校准目标（单 Band / 单 CBand12 / 联合）在统计上不可区分，且收敛到同一个 64 维子空间
> （两两最小主余弦 ≥ 0.94），说明校准信号是 readout-无关的通用信号，
> 联合校准的'共享'成分不带来额外收益。"**

这是有效的科学结论，比"校准有效"更有信息量——它把"Where 的上限不在 basis 上"这件事钉死了，
从而把后续 Where-B 的失败归因空间收窄。**不要**把 `Δmean` 单独拿出来写成"校准提升 1.5%"，
那会让审稿人一算 median 就发现被误导。

---

## 四、预注册判据 vs 实测（问题 3）

**先说一个结构性问题：协议 §4.4 没有给 Where-A 任何数值 gate。**
它只规定了四臂的归因角色、`BA-3-Joint` 是预注册主方案且"不根据 Where-B 结果事后切换"。
因此下表里能称为"预注册"的只有实现侧 pre-register 的门槛与 §10.2 的配置。
**这本身是计划的缺口**（见 §八建议 R1）。

| 预注册项 | 出处 | 门槛 | 实测 | 判定 |
|---|---|---|---|---|
| 四臂全量跑完，不做短跑筛选 | §4.4 / §16 | 4 arms | 4/4 完成，串行单卡 339 min | **PASS** |
| 1 epoch over all eligible local train samples | §10.2 | 75,544 | 三个训练臂 `n_seen = 75,544`，`sample_gap = 0`，`actual_steps = planned = 2,361` | **PASS** |
| AdamW / lr 1e-4 / wd 0.01 / warmup 3% / cosine | §10.2 | — | 全部一致，`warmup_steps=71`（=0.03×2361），`final_lr=0` | **PASS** |
| L-BFGS float64 + 多起点 + 失败显式 rejection | §10.2 | 不静默换零向量 | `dtype=float64`，训练侧 band 10 起点 / cband 5，评测侧 band **18** / cband **9**；rejection 逐条落盘 | **PASS** |
| 拟合失败率 | §10.2（无数字） | — | BA-1 2/75,544 (2.6e-5)；BA-2 6/75,544 (7.9e-5)；BA-3 3/151,088 (2.0e-5)；**评测侧四臂 3200/3200 全部 ok，0 失败起点** | **PASS** |
| s 越域中位 ≤ 1%（`S_OOD_FRAC_MAX`） | NOTES D5/N-20 | median ≤ 0.01 | 四臂 × 2 readout 的 `frac_ood` **中位与 p90 全部 = 0**；最大单样本 band 0.034–0.116、cband12 ≤ 0.0042；`frac>1%` 的样本 band 2–3 个 / cband12 0 个 | **PASS** |
| 越域上报在 clamp **之前** | CLAUDE.md s 缓存契约 | 必须 | 逐样本 `s_domain` 带 `raw_min/raw_max/frac_above/frac_below/frac_out_of_domain`，`clamped:true`，`domain:[−3,3]` 为**整臂常量** | **PASS** |
| headline = normal 层，low 单独分层（D1） | NOTES D1 | — | `headline_winner_confidence:["normal"]`，n=224；`by_winner_confidence_*` 单列 | **PASS** |
| `image.upscaled` 分层 | 任务卡 | — | `strata.upscaled` 有（True 54 / False 346 张图） | **PASS（但 n 记法有误，见 §六 N-2）** |
| 退化（灰度）样本保留不剔除 | NOTES §四之三裁定 | — | 裁定写在 NOTES；**证据只有 32 样本 preflight（2/32）**，全量无归档 | **PARTIAL（见 §六 B-3）** |
| maskview shard 命中率 | NOTES §四之五 | 0 misses | train 75,544/0、V_where 400/0，四臂零退回 live 解码 | **PASS** |
| guided filter 参数定档 | NOTES D5 / N-27 | S4 后用 BA-3 的 B 复跑 sweep 再翻 `provisional=False` | 四臂 `eval.upsample.provisional` 仍为 **true**，`d5_upsample_sweep.json` 的 `note` 仍写着待办 | **未闭合** |

---

## 五、失败案例归因

### 5.1 训练期 rejection（11 条 / 302,176 次拟合，全部 `loss_above_threshold`）

| 臂 | 条数 | build 分布 | winner_confidence |
|---|---:|---|---|
| BA-1-Band | 2 | l2, l5 | normal 1 / low 1 |
| BA-2-CBand12 | 6 | l5×3, l2, l4×2 | low 6 |
| BA-3-Joint | 3（全 cband12） | l2, l5, l1 | low 2 / normal 1 |

全部是 `soft_iou_minmax ≈ 0.02–0.10`（即 `1−softIoU ≥ 0.9` 的拒收阈值），
`n_failed_starts = 0`、`start_errors = []`——**不是数值失败，是这些样本的 GT mask 在
`[geo5, L, S, sem64]` 张成的方向上真的不可线性分离**。
两个样本（`sft_5a33232e…` l2、`sft_a381dbc0…` l5）在 BA-2 与 BA-3 上**同时**被拒，
说明是样本属性而非随机性。**归因正确、无需处理**，量级 3.6e-5 可忽略。

### 5.2 评测期最差样本（BA-3-Joint，hi，normal 层）

| sample_id | band hi | cband12 hi | band low | build / region | BA-0 band hi |
|---|---:|---:|---:|---|---:|
| `sft_4fd87f21a7528d3c0d9dfaf4ae8961de` | **0.3534** | **0.2853** | 0.5590 | l5 / center | 0.1758 |
| `sft_1d863dc5430143027a0d8e1afc82b74a` | 0.4770 | 0.5012 | **0.8507** | l3 / lower | 0.3601 |
| `sft_b78fab3865c76bb0cb40dd60d6faf9f2` | 0.5379 | 0.4402 | **0.8428** | l5 / lower | 0.4210 |
| `sft_9e2c10ec2fc4892fc7897c92ab187a4c` | 0.5168 | 0.5036 | 0.7219 | l6 / center | 0.3366 |

**两类失败，性质不同：**
1. **`4fd87f21` 型（低分辨率就拟合不上）**：low 0.559，hi 0.285–0.353。
   basis 表达力不足，属真实天花板，校准把它从 0.176 抬到 0.353 但仍然很低。
2. **`1d863dc5` / `b78fab38` 型（low 0.85 → hi 0.44–0.50，掉了 0.35–0.40）**：
   **低分辨率拟合得很好，是 guided upsample 在交付分辨率上崩掉的。**
   这是**最值得追的一类**，因为它指向 D5 的 `radius_low=1, eps=1e-2` 在某些图上不合适，
   而 D5 的定档只用了 24 个样本、且用的是**未校准的 B**（N-27 未闭）。

**低→高落差分布（normal 层中位）**：BA-0 band −0.0027 / cband12 −0.0026；
BA-3 band −0.0061 / cband12 −0.0076；p10 达 −0.034 ~ −0.036。
**校准后落差反而变大了**——校准让低分辨率的场更"锐"，guided upsample 跟不上。
这条没有任何交付文件提到。

### 5.3 参数饱和（有信息量的诊断，不是失败）

| 臂 | band `pi` 顶到 1.0 | cband12 `c` 顶到边界 | cband12 `o` 顶到边界 |
|---|---:|---:|---:|
| BA-0 | 0 | 7 | 0 |
| BA-1 | 25 | 26 | 1 |
| BA-2 | 25 | 28 | 2 |
| BA-3 | 28 | 30 | 0 |

`pi=1` = L-BFGS 选定了硬极性（`m(z)` 完全良定义），符合 NOTES §四之三 WA-P6b 的裁定。
校准后饱和数从 0 → 25–28，说明校准后的 `s` 轴更容易做出干净的单极性划分。可入报告。

---

## 六、Blocker 与 nit

### **B-1（blocker，影响 Where-B）**：`soft_iou_prod` 是掩膜软度的函数，不是质量指标，**绝不能进任何判据表或 gate**

交付的每个样本同时给了 `soft_iou_minmax` 与 `soft_iou_prod`，两者差 0.22–0.25 绝对值。
审阅侧检验（BA-3-Joint，hi，normal n=224）：

- 一个**完美预测**（pred ≡ GT）在 prod 形式下的自身上限 `E[m²]/(2E[m]−E[m²])`：
  中位 **0.7858**、p10 0.4413。
- 实测 `soft_iou_prod` 中位 **0.7475**、p10 0.4345 —— 即**已经处在它自己的上限的 98.4%**。
- **`corr(soft_iou_prod, 自身上限) = 0.955`；`corr(soft_iou_prod, soft_iou_minmax) = −0.003`。**

也就是说 `soft_iou_prod` **几乎完全由 GT mask 的软度决定，与拟合质量零相关**。
四臂 × 2 readout × 2 分辨率的 prod 中位全部卡在 0.739–0.748 的一条线上，正是这个原因。
**风险**：§5.6 gate 第一行写 `median soft-IoU ≥ 0.75`。若 Where-B 的 `soft-IoU` 用的是 prod 形式，
这个 gate 会变成"GT mask 够不够硬"的检查，与红线痛斥的 AUC 是同一类错误
（一个零参数基线就能拿高分）。**必须在协议里把 soft-IoU 定义钉成 min/max 形式**。

### **B-2（blocker，静默）**：V_where oracle shard 没有声明 `cband_normalization`，而 11% 的样本正好落在两种归一化会分道扬镳的地方

NOTES §四之二 N-25 明确要求"每样本 `oracle.json` 加 `cband_normalization`，Where-B 必须用同一约定
重算 `R(z;ρ_pred)`，否则 `L_curve` 两侧不是同一个函数"。
**实测：`run_calibration` 产出的 `oracle/<arm>/V_where` shard 的 `.oracle.json` 里
没有 `cband_normalization` 字段，也没有 `curve` / `r*(z)` 的 257 点采样。**
（该字段与曲线应由 S5 `make_oracle_latents` 产出，但 S5 目前仍在跑，见 B-4。）

**这不是理论风险**：NOTES §D10 记录了字面 `eps` 形式在 σ 取下界时分母塌陷到恒等 0。
审阅侧实测 BA-3-Joint 的 400 个 cband12 latent：
**44/400（11.0%）至少有一个 `σ` 顶在 0.02500 的下界上**，123/400（30.8%）有 σ ≤ 0.05。
Where-B 若用 `eps` 形式重渲染这些 `ρ*`，**在 11% 的样本上会得到与 Where-A 不同的目标曲线，
而且不会报错**——正是 CLAUDE.md「s 缓存消费契约」里说的第二类静默失败。

### **B-3（blocker，交付形式）**：CLAUDE.md 规定的交付物缺四项

`experiments/Q3VL_metacanvas_where_what_20260804/where_a/` 现有：NOTES.md、4 个
`calibration_*.json`、preflight/sweep/mask-validation/maskview-verify JSON、`maskviews/`。

**缺失**：
- **`REPORT.md`（含强制三行）** —— CLAUDE.md 交付物规范写明"审阅见到缺失即打回"；
- **`metrics.json`** —— harness 机器可读输出（`eval_V_where.json` 在 `/home/bc/data/runs/` 下，
  没有汇总到交付文件夹，也没有四臂并排结构）；
- **`viz/success_*` 与 `viz/failure_*`** —— **一张图都没有**。规范写明"'没有失败案例' = 没找够，
  审阅直接打回"。§5.2 已经点名了四个可视化价值极高的失败样本；
- **`config/`** —— 配置快照 / seed / git commit / 环境。
  这些字段散在 `calibration_*.json.setup` 里（有 git_commit、torch 版本、seed、
  完整超参），**内容其实齐全，只是没有按规范落成 `config/`**。

另：退化（灰度）样本的处理**裁定**写在 NOTES §四之三，但**全量证据没有归档**：
NOTES 引用的"V_where 3.75% / train 3.25%"在整个交付文件夹里 grep 不到；
只有 `preflight_where_a.json` 的 32 样本档案（2/32 = 6.25%，两个 sample_id）。
`eval_V_where.json` 也没有 degenerate 分层。
（审阅侧用"`w_raw` 的 S 分量恰为 0"做代理只识别出 3 个，是弱代理，无法证伪也无法证实 3.75%。
这 3 个样本的 hi 中位 band 0.9327 / cband12 0.9373，低于全体的 0.9651 / 0.9759，
**说明灰度样本确实更难，值得单列一层**。）

### **B-4（阻塞 Where-B 开跑，非本次结果问题）**：S5 train oracle latents 未交付

`/mnt/nfs/bc/data/datasets/where_a-20260805/oracle/BA-3-Joint/s5/` 下只有
`.V_where.partial.3172235.…`，`shard-00000.tar.tmp` 与 `shard-00000.idx.jsonl.tmp` **均为 0 字节**。
PID 3172235 **仍存活**（`ps -p` 实证），所以是在跑而非失败——但它跑的是 **V_where**，
**train split 的 oracle latents 一条都没有**。§5.5 的 `L_s` / `L_curve` / `L_dir` 全靠它。

### N-1（nit）：四臂不是同一个 git commit，且 BA-0/BA-1 的 commit 记录是脏的

`calibration_*.json.setup.env.git_commit`：BA-0 与 BA-1 = `04e6a06`，BA-2 与 BA-3 = `2c41e3c`。
`git log 04e6a06..2c41e3c -- q3vl/where/` 只有一条 `08992d2`（BA-0 空转 epoch + 序列脚本判据修复），
**Where-A 核心库（calibrate/oracle/readout/basis/phi/upsample）零改动**，四臂科学上可比。
但 `08992d2` 的提交时间是 20:27:15，而 BA-0 启动于 **20:16:34**、BA-1 于 **20:23:13** ——
两臂都在该 commit 之前启动，却已经带着它的修复（BA-0 的 `schedule.json` 有
`epoch_skipped:true`，这个字段就是该 commit 加的）。
**结论：跑的是未提交的工作树，`git_commit` 字段不足以复现，且没有 `dirty` 标记。**
建议 `run_setup.json` 增记 `git diff --stat` 摘要或 worktree hash。

### N-2（nit）：`strata` 把两个 readout 池在一起，n 翻倍

`eval_V_where.json.strata` 报 `upscaled.False n=692 / True n=108`、
`winner_confidence.normal n=448 / low n=352`——这是 400 张图 × 2 readout。
描述统计没错，但 **n 不是独立样本数**，直接引用会高估精度。
真实图像数：normal 224 / low 176 / upscaled 54 / not-upscaled 346。
分层表应按 readout 拆开重报（审阅侧已按 readout 拆开重算，见附录）。

### N-3（nit）：BA-0 的 `maskview_stats.V_where.hits = 401`，而 `n_samples = 400`

多一次命中（很可能是 fit-pool self-check 的那次真实拟合）。不影响任何数字，但计数口径应对齐。

### N-4（nit）：`eval_V_where.json.elapsed_s` 的语义在四臂之间不一致

BA-0 = 222.1 s（只有评测，epoch 被跳过）；BA-1/2/3 = 6581.1 / 5803.8 / 7870.9 s（整臂墙钟）。
NOTES §四之六把 222.1 s 写成"评测耗时"对 BA-0 成立，但同名字段在其余三臂不是这个意思。

### N-5（nit）：训练侧与评测侧的多起点预算不同，且都低于 NOTES 引用的口径

训练侧 `n_random=2, max_iter=40` → band 10 起点 / cband12 5 起点；
评测侧 band **18** / cband12 **9**。NOTES §3.1 写"起点数（band 12 / cband 6）"、
D3 墙钟用 `n_random=3, max_iter=80`。天花板是拟合预算的函数，
**交付里没有任何"天花板 vs 起点数/迭代数"的敏感性检查**。
评测用更强预算是对的（天花板本就该用强拟合测），但需要一句话说明，
否则读者无法判断 0.972 是不是还能再高。

---

## 七、天花板数字能否当 §5.6 gate 的分母（问题 5）

§5.6（A-5 修订后）十行 gate 里有四行需要 Where-A 提供分母或对齐口径：

| gate 行 | 需要的分母 | 交付里有吗 | 判定 |
|---|---|---|---|
| `local .cgt median soft-IoU ≥ 0.75` | 无（绝对阈值） | — | **口径未绑定**：必须声明 soft-IoU = min/max 形式（B-1） |
| **`相对逐图 oracle 的 soft-IoU ≥ 85%`** | 逐样本 oracle soft-IoU | **有**（400 条 × 2 readout × 2 分辨率，按 sample_id 对齐） | **可用，但须绑定 (a) min/max 形式、(b) 分辨率档** |
| **`grid 级 boundary F1 / oracle ≥ 75%`** | 逐样本 oracle 的 **grid 级 boundary F1** | **没有**。交付只有 soft_iou_minmax / soft_iou_prod / mae / mse | **缺分母 → 该 gate 行当前无法评估** |
| `std(s_pred)/std(s*) 中位 ≥ 0.60` | 逐样本 `std(s*)` | **没有**（只有 `s_low_range` / `s_hi_range` 的 min/max） | **缺分母** |

**好消息**：后两个分母**不需要重训**。逐样本 latent（`w0, w_dir(71), alpha, rho, rho_raw`）
和冻结的 `B.npy` 都已交付，Where-B 侧可以自己渲染 `s*` 与 oracle mask 再算。
**但前提是 B-2 被修掉**（cband 归一化约定必须显式声明），否则重渲染出的 oracle 与 Where-A 报的不是同一个东西。

**分辨率必须选 `low`。** A-5（§17.2「换成了什么」）明写三列"全部在 `F_pre` 网格上、
阈值化统一为匹配 GT 面积的 top-k"。因此分母取 `eval.low.soft_iou_minmax`。
按 BA-3-Joint 的 headline（normal, n=224）：

| readout | oracle median (low) | 0.85 × oracle | 与绝对门槛 0.75 取大 |
|---|---:|---:|---:|
| R-Band | 0.9727 | 0.8268 | **0.8268** |
| R-CBand12 | 0.9822 | 0.8348 | **0.8348** |

即 **相对 oracle 那一行才是真正约束住的那一行**（0.827/0.835 高于绝对门槛 0.75），
Where-B 的 8 个臂需要在 generated-context 上做到 low-res median soft-IoU ≥ 0.83 才过门。
这是个**很高**的门槛，主 agent 应当在 Where-B 开跑前就知道。

**BA-3 的 B 作为固定 projector 的适格性**：**适格**。
理由：(i) §4.4 预注册 BA-3 为主方案且禁止事后切换，本次结果没有任何理由推翻它；
(ii) BA-3 在 8 个 headline 格子里有 6 个是最好或并列最好；
(iii) 三个校准臂统计不可区分且落在同一子空间（最小主余弦 ≥ 0.94），**换成 BA-1 或 BA-2 也没差**，
所以维持预注册选择是零成本的诚实做法；
(iv) `B.npy` 有 sha256 + digest，四臂互不覆盖，S5 命令行已打印。
**唯一保留**：guided filter 参数仍是 `provisional=true`（N-27 未闭），
而 §5.2 显示交付分辨率的落差在校准后**变大**，这个参数值得在 Where-B 开跑前用 BA-3 的 B 复扫一次。

---

## 八、对计划的具体修改建议

**R1 —— 给 §4.4 补一个 Where-A 的验收段落（协议第 210–223 行之后）。**
现在 §4.4 只定义了四臂角色，没有任何数值判据，导致"校准收益有限"既不能算 PASS 也不能算 FAIL。
建议追加一句预注册：「Where-A 的验收 = (a) 四臂全部完成且 rejection 率 < 1e-3；
(b) `V_where` headline 的逐样本 oracle 天花板在 `low` 与 `hi` 两档均落盘；
(c) BA-3 相对 BA-0 的配对 Δ（median + mean + p10 + sign-flip p）落盘。
**天花板本身不设阈值——它是 §5.6 的分母，不是被考核的量。**」

**R2 —— 在 §17.2「换成了什么」的判据表里把 soft-IoU 的定义钉死（协议第 991–994 行）。**
现表只写「soft-IoU / hard-IoU」。补一行脚注：
「soft-IoU 一律指 `Σmin(a,b)/Σmax(a,b)`。**禁用积形式** `Σab/Σ(a+b−ab)`——
Where-A 实测该形式与拟合质量相关系数 −0.003、与 GT mask 软度相关系数 0.955，
其完美预测上限中位只有 0.786，阈值 0.75 在该形式下无意义。」
理由与「AUC 全实验禁用」同源：一个零信息量的统计量能拿到看起来合格的分数。

**R3 —— 在 §5.6 gate 表（协议第 329–340 行）给每一行标注分母来源与分辨率档。**
具体：第 2 行（相对 oracle soft-IoU）标注「分母 = `oracle/BA-3-Joint/V_where` 逐样本
`eval.low.soft_iou_minmax`，按 readout 匹配」；
第 4 行（grid boundary F1 / oracle）标注「分母需 Where-B 侧从交付 latent + `B.npy` 重渲染 oracle mask 后自算」；
第 8 行（`std(s_pred)/std(s*)`）同上。
并把 §4.4 第 223 行「逐图 oracle 参数只作为监督和 ceiling」扩写为
「…并作为 §5.6 第 2/4/8 行的分母；重渲染必须使用与 Where-A 相同的 CBand 归一化约定」。

**R4（blocker，须在 Where-B 开跑前落地）—— oracle 载荷补三个字段。**
`cband_normalization`（= `logsumexp`）、`r*(z)` 的 257 点采样、`std(s*)`。
前两项 N-25/B-7 已经为 `make_oracle_latents` 要求过，但 `run_calibration` 产出的
`V_where` shard 没带。**11% 的样本 σ 顶在下界，这是会真的发散的。**
成本：CPU 重放已冻结的 latent，无需重训。

**R5 —— D5/N-27 闭环再做一次，并把"低→高落差"列进 sweep 的选择规则。**
当前 sweep 用 24 样本 + 未校准 B 选出 `r=1, eps=1e-2`，而全量实测：
校准后 hi−low 中位落差从 −0.0027 恶化到 −0.0061(band)/−0.0076(cband12)，
p10 落差 −0.034 ~ −0.036，且有 `low 0.85 → hi 0.44` 这种单样本崩塌。
建议：用 `--basis .../BA-3-Joint/B.npy` 在 ≥100 个样本上复扫，
字典序改为「先过越域门槛 → 再比 hi 档 **p10** → 再比中位」（现在只比中位，
而问题恰恰全在尾部）。翻 `GUIDED_PARAMS_PROVISIONAL=False` 之前不得进 Where-B。

**R6 —— 补交付物：`REPORT.md`（强制三行）、`metrics.json`（四臂并排 + 本审阅的配对 Δ 表）、
`viz/`（§5.2 点名的 4 个 failure + 对应 success，每张含 输入 / 预测 mask / GT / `s` 场并排）、`config/`。**
可视化必须遵守 CLAUDE.md「空间场可视化纪律」：禁逐图 min-max 着色、色标只取有效格、
叠回原图用严格逆映射而非 resize、进入判据的数字用未归一化原始场。
**特别提醒**：`s` 场的色标要用整臂常量 `[−3, 3]`（生产方已声明的域），不要用逐图 min-max。

**R7 —— `eval_V_where.json` 增加 degenerate（灰度）分层，并把全量统计归档。**
NOTES 引用的 3.75%/3.25% 在交付里查不到；审阅侧代理证据显示灰度样本的 hi 中位
比全体低 0.03–0.04。这是 D8/§四之三裁定「保留样本，只在报告侧剔除退化维」的**报告侧义务**，
现在只做了一半。

---

## 九、数字是否可写进最终报告

| 数字 | 判定 | 必须同时写明的限定 |
|---|---|---|
| 四臂 headline 天花板 `soft_iou_minmax`（low 与 hi，median/mean/p10） | **可写** | (1) soft-IoU = min/max 形式；(2) headline = `winner_confidence=normal`，n=224；(3) hi = 交付分辨率（短边 512），low = `F_pre` 网格；(4) guided filter 参数仍为 provisional |
| BA-3 vs BA-0 配对 Δ（本审阅算） | **可写** | 必须 median + mean + p10 三个一起给，并说明"增益集中在下分位"。**只给 mean 会误导** |
| BA-3 vs BA-1 / BA-2 的差异 | **只能写"统计上不可区分"** | 给出 p 值（0.062–0.51）与 CI。**禁止**写"联合校准更优" |
| B 子空间主角度（本审阅算） | **可写** | 这是"收益来自校准本身而非共享"的最强证据 |
| 拟合成功率 / rejection 率 / s 越域 | **可写** | 训练侧 11/302,176；评测侧 3200/3200；越域中位 0 |
| `soft_iou_prod` | **不可单独出现** | 若要写，必须同时给"完美预测在该形式下的上限"（中位 0.786）并说明它与质量零相关 |
| `strata` 里 n=692/108/448/352 | **不可直接引用** | 是 400 图 × 2 readout 的池化计数；真实图像数 346/54/224/176 |
| NOTES 里"V_where 灰度 3.75%" | **暂不可引用** | 交付内无归档，须先落盘（R7） |
| D5 sweep 的"cband12 落差为负" | **不可引用** | 24 样本 + 未校准 B 的结论；全量上四臂八格全部为负落差（hi < low） |
| §5.6 分母 0.9727(band)/0.9822(cband12) 与 gate 线 0.827/0.835 | **可写，但须先修 B-2** | 否则重渲染的 oracle 与 Where-A 报的不是同一个函数 |

---

## 附录：按 readout 拆开的分层表（审阅侧重算，median `soft_iou_minmax`）

| 分层 | 图像数 | Arm | band/low | band/hi | cband12/low | cband12/hi |
|---|---:|---|---:|---:|---:|---:|
| normal | 224 | BA-0 | 0.9669 | 0.9611 | 0.9749 | 0.9714 |
| normal | 224 | BA-1 | 0.9724 | 0.9623 | 0.9822 | 0.9718 |
| normal | 224 | BA-2 | 0.9724 | 0.9623 | 0.9827 | 0.9736 |
| normal | 224 | **BA-3** | **0.9727** | **0.9624** | **0.9822** | **0.9724** |
| wc=low | 176 | BA-0 | 0.9728 | 0.9686 | 0.9781 | 0.9787 |
| wc=low | 176 | BA-1 | 0.9748 | 0.9693 | 0.9851 | 0.9781 |
| wc=low | 176 | BA-2 | 0.9755 | 0.9699 | 0.9852 | 0.9794 |
| wc=low | 176 | **BA-3** | 0.9747 | 0.9686 | 0.9842 | 0.9792 |
| upscaled=True | 54 | BA-0 | 0.9742 | 0.9683 | 0.9815 | 0.9787 |
| upscaled=True | 54 | **BA-3** | 0.9757 | 0.9711 | 0.9856 | 0.9798 |
| upscaled=False | 346 | BA-0 | 0.9705 | 0.9626 | 0.9754 | 0.9720 |
| upscaled=False | 346 | **BA-3** | 0.9727 | 0.9644 | 0.9832 | 0.9742 |

**分层结论**：`winner_confidence=low` 的天花板**不低于** normal（band/hi 0.9686 vs 0.9624），
`upscaled=True` 也**不低于** `False`（band/hi 0.9711 vs 0.9644）。
两条都支持 D1 的裁定（low 与 normal 在"候选区域在哪"这件事上正交），
也说明 upscaled 不是天花板的风险因子。**这两条可以直接写进论文的数据纪律小节。**
