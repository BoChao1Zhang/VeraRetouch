# VeraRetouch 局部精修战役 · 进度汇报简报

> **⚠️ 日期快照声明（2026-08-11 追加）：本文数字截至 2026-08-10 20:35，最新状态见
> `docs/WHERE_STATE_2026-08-11.md`**（where 侧唯一权威入口）。本文为该时点的汇报存档，不随后续
> 实验更新；引用数字前请先对入口文档核对。

> 汇报时点：2026-08-12（周三）19:00　｜　素材截止：**2026-08-10 20:35**
> 纪律：每个数字带出处路径；失败如实呈现，不粉饰；尚未产出的数字留占位并标注「周二更新」。

---

## 1. 战役总览时间线

| 日期 | 阶段 | 一行成果 | 墙钟 | 出处 |
|---|---|---|---|---|
| 08-04 | 立项 | 冻结实验协议（METACANVAS Where/What）与 Base SFT 规格 | — | `docs/METACANVAS_WHERE_WHAT_FINAL_EXPERIMENT_PROTOCOL_2026-08-04.md`、`docs/QWEN3_VL_BASE_SFT_SPEC_2026-08-04.md` |
| 08-05 02:12 | 实现审阅 S0 | 4 个 blocker 全部集中在**从未被执行过**的 ZeRO-3 启动路径，清零后才放行训练 | — | `docs/reviews/REVIEW-impl-S0.md` |
| 08-05 03:32→11:48 | **Base SFT** | Qwen3-VL-4B 官方 Arm B、两卡 ZeRO-3 训完 1 epoch，两段式 `<where>`/`<color>` 格式立住 | **8h14m47s** | `/home/bc/data/runs/q3vl_base_sft_20260804/{job.marker,train.log}` |
| 08-05 11:49 | 离线验收 | 两个 protected checkpoint 自由生成结构率 10/10 项满分 | 生成 636/640 s | `.../s0_base_sft/CHECKPOINT_VERIFICATION.md` |
| 08-05 20:10→20:28 | genctx | `<where>`+`<color>` 生成上下文全量落盘：train 159,215 / V_where 896 / V_what 897，零截断零格式失败 | — | `.../where_b/genctx_{train,V_where,V_what}.json` |
| 08-05 → 08-06 | **Where-A（成功）** | 71 维 basis 的表达上界实测 **0.972**（交付分辨率中位），四臂归因完成，mentor 展示图 28 张 | 训练 ≈3.6 h/臂 × 3 臂 + V_where 重拟合 1.6–2.2 h/臂 | `.../where_a/REPORT.md`、`calibration_throughput_w38.json`、`calibration_BA-*.json` |
| 08-07 ~ 08-09 | （暂停） | 战役目录 / docs / q3vl 三处**零文件变更**，无产物 | — | `find ... -newermt "2026-08-06 23:00" ! -newermt "2026-08-10 00:00"` = 0 |
| 08-10 11:39→19:02 | **Where-B W1 波（失败）** | W01/W02 双卡并行跑满 1 epoch（4975 步），十项 gate **过 4 项**，`WHERE-GATE-FAILED` | W01 **7h22m46s** / W02 **7h11m43s** | `/home/bc/data/runs/where_b/W0{1,2}/job.marker` + `where_b_final.pt` mtime |
| 08-10 下午 | 判定与转向 | 用户裁定**设计失败**，W03–W08 六臂取消；终局结果审阅 + Where 头重设计需求文档落盘 | — | `.../where_b/REVIEW-result-W1.md`、`docs/WHERE_HEAD_REQUIREMENTS_2026-08-10.md` |
| 08-10 15:10→15:21 | 基础设施 | 本地分片缓存 + 有序预取上线：数据供给 **2.216 → 0.063 s/step（35×）** | — | commit `93f46ba`/`c978b10`、`q3vl/whereb/scripts/perf_acceptance.py` |
| 08-10 19:0x→20:0x | 新方案探针 | attention 读出路线（E1）与 merger 相似度路线（PW5）两枚探针出结果 | E1 前向 5.9 min / PW5 全集 24 s | `.../where_b/probe_e1_whereattn_20260810/REPORT.md`、`probe_pw5_fpresim_20260810/REPORT.md` |
| 08-10 20:10 | **Stage-What C 波起跑** | C01–C04 控制臂入队开跑（NoWhere 下界 + OracleWhere 上界） | 预计 08-11 05:00 前后全部完成 | `tools/queue/q status`、`.../what/config/EXEC4_launch.json` |

---

## 2. Base SFT —— 地基已立住

**配置**：规格冻结的**唯一训练臂「官方 Arm B」**（vision blocks 冻结，4 个 merger + Language 塔可训），
2×H100 DeepSpeed ZeRO-3（`world_size=2`），159,215 样本 × 1 epoch = 4,976 optimizer step，global batch 32，bf16。
出处：`docs/QWEN3_VL_BASE_SFT_SPEC_2026-08-04.md` §2.2/§7.1、`q3vl/train/configs/sft_base.yaml`、
`/home/bc/data/runs/q3vl_base_sft_20260804/job.marker`。

**墙钟**：`train_runtime = 29,686.5 s` = **8h14m47s**（03:32:43 起 → 11:48:36 止），5.363 样本/秒。
出处：`.../q3vl_base_sft_20260804/train.log` 末行。

**收敛**（book-keeping，红线禁止用 val loss 选 checkpoint）：

| step | eval_loss | assistant_acc | where_acc | color_acc |
|---:|---:|---:|---:|---:|
| 500 | **0.821242** | 0.7374 | 0.9033 | 0.7143 |
| 2488（0.5 ep，protected） | 0.705572 | 0.7649 | 0.9111 | 0.7446 |
| 4976（1.0 ep，protected） | **0.664593** | 0.7762 | 0.9128 | 0.7572 |

**自由生成的结构率（真正的验收项）**：离线单卡、无 DeepSpeed、V_where 同一批 64 条（seed=42）贪心生成，
两个 checkpoint 的 **10 项结构指标全部满分**——标签完整率 1.00、顺序正确率 1.00、两段非空率 1.00、
旧七标签泄漏率 0.00、重复段率 0.00、EOS 终止率 1.00、截断率 0.00。

**语言侧把区域「说」对了**：`<where>` 段 token-F1 **p50 = 1.0000**，**53.12%** 的样本与 GT 逐字一致；
`<color>` 段预测行数恰为 6 行的比例 1.0000、与 GT 行数一致率 1.0000。

> 出处全部为 `experiments/Q3VL_metacanvas_where_what_20260804/s0_base_sft/CHECKPOINT_VERIFICATION.md` §3–§5。
> 为什么必须离线补这一步：裁定 D-J3 关掉了训练内生成诊断（ZeRO-3 `generate` 未验证），
> 不补则「推理时两段格式是否成立」只有训练 loss 可拿。

---

## 3. Where-A（成功）—— basis 的表达上界够用

**要回答的问题**：Qwen3-VL merger 前特征 `F_pre` 经共享 64 维投影器 + 71 维解析 basis，
够不够表达 local 编辑掩膜？够，则 Where-B 只需预测一组全局参数而不必生成逐像素场。

**结论：够。** V_where headline（`winner_confidence=normal`，n=224）、交付分辨率（hi）档：

| 臂 | readout | hi 中位 | hi p10 |
|---|---|---:|---:|
| `BA-0-Fixed`（不训练，随机正交投影） | cband12 | 0.9714 | 0.8088 |
| `BA-2-CBand12` | cband12 | **0.9736** | 0.8288 |
| `BA-3-Joint`（预注册主方案） | cband12 | 0.9724 | **0.8324** |
| `BA-0-Fixed` | band | 0.9611 | 0.7843 |
| `BA-3-Joint` | band | 0.9624 | 0.8258 |

**四臂归因的三条结论**（这本身就是论文要写的一句话）：

1. **校准 `B` 的收益是二阶的**：中位数上四臂相差 ≤ 0.002，未校准的随机正交投影已拿到 0.9714——指标饱和，
   天花板由 `F_pre` 与 71 维 basis 的结构决定，投影器方向几乎不改变它。
2. **收益集中在尾部，不在中位数**：band 的 hi **p10 从 0.7843 提到 0.8258–0.8324（+0.04 量级）**，
   min 从 0.1169–0.1758 提到 0.3287–0.3534。**只看中位数会完全看不到这件事。**
3. **cband12 稳定优于 band**（中位 +0.010，四臂一致）：单个高斯覆盖不了环形/平顶区域。

**预注册判据全部达成**：oracle 拟合成功率 **100.0%**（四臂 × 两 readout，0 拒绝，预注册 ≥90%）；
残差化正交误差 7.60e-7（<1e-4）；s 越域比例中位 **0**（≤1%）；GPU preflight **8/8 PASS**。

**失败案例已如实分列两类**（交付规范要求必须有）：单活跃基元的上采样窄带塌陷（30.8% 的样本落在单活跃基元解上，
其低→高落差中位 +0.0123，是多基元解的两倍以上）；以及一例**真天花板**（low 档本身只有 0.665，
GT 形状不在单轴 basis 表达范围内）。两类性质不同，混报会同时掩盖两个问题。

**mentor 展示图已有**：`where_a/viz_mentor/` **28 张**（V_where 固定 seed 随机抽样，commit `0695e0a`），
外加 `viz/` 3 成功 + 3 失败并排面板。

> 出处：`.../where_a/REPORT.md`、`metrics.json`、`calibration_BA-*.json`、`viz_mentor/index.md`。

---

## 4. Where-B（失败，但归因精确）

### 4.1 终局判据：十项 gate 过 4 项

主榜 = V_where / generated context / local，n = 400；终局 = step 4975。全部数字由结果审阅从
`eval_final/per_sample.jsonl` **独立重算并逐位复现**。

| # | gate | 阈值 | W01(band) | W02(cband12) | 判定 |
|---:|---|---:|---:|---:|:--|
| 1 | local median soft-IoU | ≥ 0.75 | **0.5565** | **0.5617** | ✗ |
| 2 | 相对逐图 oracle | ≥ 85% | 57.94% | 58.19% | ✗ |
| 3 | p10 | ≥ 0.55 | 0.2358 | 0.2965 | ✗ |
| 4 | grid 边界 F1 / oracle | ≥ 75% | **32.14%** | **33.14%** | ✗ |
| 5–6 | 中心先验配对 Δ(hard-IoU) 及 p | >0, ≤0.05 | +0.038, 1e-4 | +0.057, 1e-4 | ✓ |
| 7 | instruction shuffle 降幅 | ≥ 0.20 | 0.1445 | 0.1259 | ✗ |
| 8 | `std(s_pred)/std(s*)` | ≥ 0.60 | 0.3863 | 0.4487 | ✗ |
| 9 | global mask soft-IoU | ≥ 0.98 | 0.999995 | 0.999990 | ✓ |
| 10 | GT/generated context gap | ≤ 0.05 | 0.0187 | 0.0384 | ✓ |
| | **合计** | | **4/10** | **4/10** | `WHERE-GATE-FAILED` |

**通过的四项没有一项是正面证据**：第 9 行是全 1 掩膜的平凡解（global 样本 GT 恒为 1）；
第 10 行说明 teacher 与 generated 两种上下文**一样差**；第 5/6 行整臂通过，但分层后含义被改写（见 4.3）。

### 4.2 「指令只承载尺度」——本次最有价值的精确画像

模型**确实在读指令**，但那条通路只承载得动一个标量。

| context | `corr(pred_mean, gt_mean)` Spearman（W01 / W02） |
|---|---|
| generated（真实指令） | **0.665 / 0.691** |
| shuffled（**同一张图**的另一条真实指令） | **0.188 / 0.150** |
| null（无语言上下文） | 0.002 / −0.078（场退化成近常数，`pred_std` 中位 **0.016**） |

**扣掉面积以后，指令条件性 ≈ 0**：同图配对差分、中心先验校准后、**面积均衡子集**（面积比 ∈[0.5,2]）：
**W01 Δ = +0.0032（p = 0.460，CI95 跨 0）/ W02 Δ = +0.0113（p = 0.028）**。
全部配对口径下是 +0.0239 / +0.0312（p 1e-4）——**两个口径的差，就是「面积」这一项**。

配套三条负控制齐备：shuffled 降幅 0.145/0.126、fixed_phrase 0.227/0.220、irrelevant_words 0.246/0.400；
antonym 翻转后 `|Δ(hard-IoU)|` 中位 **0.0000**（场不偷读颜色方向词，PASS）。
**即：错的不是读了不该读的，是该读的（主体身份 / 位置）没读出来。**

### 4.3 机制不匹配，而非容量到顶（措辞边界）

- **形状列全程没动**：grid 边界 F1 在 4975 步里 W01 只从 0.2898 走到 0.3186（+0.029），
  同期 median soft-IoU 涨了 **+0.153** ——**增益全部来自面积标定**；W02 末段 4000→4975 净变化 **−0.019**。
  按各自斜率外推到 gate 要的 0.75 需 **29 / 6.6 个 epoch**。
- **零参数归一化后模型只补上多少**：中心先验 → oracle 的差距，覆盖列补上 **6.7% / 10.1%**、形状列 **15.1% / 16.6%**；
  且 **21.5% / 17.5%** 的样本得分**低于「整幅全 1」**这个不看图不看指令的掩膜。
- **readout ×12 无效**：Band → CBand12（表达力约 12 倍）中位只买到 **+0.005**、形状 +0.012，
  只改善尾部（长尾 65→43）与 s 塌缩比例（33.5%→3.5%）。证明**瓶颈在 readout 上游的映射**，不在读出函数族。
- **中心先验必须分层看**：整臂 Δ>0 几乎全部由「中心先验本来就打不中」的类贡献
  （偏心 n=76 基线 0.175、小面积 n=52 基线 0.139、复杂边界 n=95 基线 0.323，那里 Δ = +0.12~+0.15）；
  而在中心先验本来就强的三类上，W01 **三类全部打平**（p 0.61–0.67）。

**审阅裁定的正式措辞**：证据支持「**机制不匹配**」（文本 hidden → 全局 71 维线性方向这条通路，
只承载得动目标的尺度，承载不了目标的身份与形状），**不支持**「容量到顶」——MC16 三种结构 × 2 readout 共 6 臂**从未跑过**。

### 4.4 失败的价值：知道为什么失败 + 六项机制发现指路

这次失败**不是白跑**，它交付了三样东西：

1. **一份可被引用的正式答案**（协议 §15 问题 2）：「不能」，且**限定配置写全版**——冻结边界、只读最后一层、
   MC8-Joint、全局 71 维、单 seed、单 lr、1 epoch，逐条列出结论的边界。
2. **新方案的最小可证伪判据**：主判据改用**面积均衡的同图配对校准 Δ**（现方案 +0.003 / +0.011），
   而不是 median soft-IoU——后者可以靠面积标定单独刷到 0.56 而形状毫无进展。**这比 median 便宜，也更难作弊。**
3. **六项已确立的机制发现**（`docs/WHERE_HEAD_REQUIREMENTS_2026-08-10.md` §2.4），成为新方案的信号地图：
   pad/sink 涌现与排除、层头差分、浅层词-视觉对齐最强、主体/中心先验陷阱、生成式空间 logits 两次坍缩、
   本次失败画像。

**同时诚实记录仍未排除的替代解释**（不能写进立项材料当已知）：canvas 容量（6 臂未跑）、
lr/clip（只跑过一档 `lr=2e-4`，实测 `grad_norm` 稳态 2.1–5.5 而 clip=1.0，**整个训练都在裁剪区**，从未扫过）、
`H_where` 只取最后一层（与自家发现三直接冲突，从未测过浅层）、
`L_mask` 中 `0.25×balanced_BCE`（正类权 ∝1/面积）对小目标过覆盖的 loss 侧解释（未做判别实验）。

> 出处：`.../where_b/REVIEW-result-W1.md` §1–§8、`docs/WHERE_HEAD_REQUIREMENTS_2026-08-10.md` §1.4/§2.4/§3。
> 附一条流程战果：结果审阅发现需求文档里三条支撑数字取自 **step-1500/2500 的中途快照**，
> 在终局已变形甚至**反号**（「居中类输给中心先验 Δ=−0.05」终局实为**打平** −0.005 / +0.019），
> 需求文档已按 R1–R12 修订清单改正。**照抄中途数字写立项材料会被审稿人一击即破。**

---

## 5. 当前进行中

### 5.1 Stage-What 控制臂 C01–C04 —— What 可行域的首次定量

08-10 20:10 入队起跑，**双卡并行 + 两波串接**：

| 波 | 卡 | 臂 | 角色 | gate |
|---|---|---|---|---|
| C1 | gpu0 / gpu1 | **C01 / C02**（NoWhere，FG48 / SB48） | **下界**：完全不给 where 信息，配色能做到多少 | checkpoint-4976 |
| C2 | gpu0 / gpu1 | **C03 / C04**（OracleWhere，FG48 / SB48） | **上界**：给 oracle 掩膜 + latent，配色天花板在哪 | 前序臂 `what_final.pt` |

共 4,975 optimizer step / 臂，effective batch 32（micro 16 × accum 2）。
20:31 实测 C01 在 step 360、**3.205 s/step** ⇒ 单臂 ≈ 4h26m，C1 波约 08-11 00:40 完成、C2 波约 **08-11 05:05** 完成。

**这一对上下界是本战役第一次定量回答「Where 的质量到底值多少 What 分」**——Where-B 失败后，
它直接决定新 Where 方案要达到什么水平才有意义。

> **结果：周二更新**（预注册判据 vs 实测数字并排表、`ΔE00`/`PSNR_in`/`var_ratio`/烘焙一致性四列、
> C01↔C03 与 C02↔C04 的配对差）。
> 出处：`tools/queue/q status`、`.../what/config/{EXEC4_launch.json,arm_matrix.json,run_setup_C0{1,2}.json}`。

### 5.2 新 Where 方案探针（另一会话在跑，截至 08-10 20:35 的读数）

| 探针 | 问的问题 | 当前读数 |
|---|---|---|
| **E1 · attention 读出** | `<where>` token → image attention 经 sink 排除后是不是现成的指令条件定位场？ | **否定**。三个 query 池全部同时踩 P2/P3 两条作废线：vs 中心先验 Δ = **−0.029 ~ −0.049**（p_FWER ≤ 5e-4，**反向输**）、Δ_shuffle ≈ **+0.002**（对指令几乎无反应）。补件 P4 有一条**通过**的登记判据（`where_content` 小目标 Δ=+0.131，p=7.7e-05）——但它测的是「有没有指代表达」，不是「指代哪一个」：**该通道是物体性闸门，不是 grounding**。 |
| **PW5 · merger 输出相似度** | merger 输出（2560 维）与 `<where>` 目标名词 embedding 的相似度带不带词特异空间信息？ | **四种读法（全格/剔离群 × dot/cosine）词特异 Δ>0 且 p<0.05 全部 PASS**，最弱一档 p=1.0e-05；成本 400 样本 **24 秒**、无需 LLM 前向。 |

顺带一条**前提性事实**：Qwen3-VL 是 native-res、无 `expand2square` 黑边，但 **sink 依然存在且极强**——
按进契约的合取规则（n=400）：sink 格占比中位 **9.90%**、吃掉图上注意力质量 **45.4%**、
逐（层,头）argmax 落在 sink 上 **96.3%**、**400/400 样本的图像块第一个 token 就是 sink**；
代价是 oracle 天花板从 0.787 压到 **0.729**。RO-9c 在旧模型上的 pad 现象以纯 sink 形态完整复现，
**sink 排除是任何 attention 读出路线的第一前提，不是后处理选项**。

> **方案定档与下一步：周二更新**（探针仍在推进中）。
> 出处：`.../where_b/probe_e1_whereattn_20260810/{REPORT.md,REVIEW-result.md}`、`.../probe_pw5_fpresim_20260810/REPORT.md`。

---

## 6. 基础设施成果

三条线同时把「无人值守跑实验」从口号做成了可复核的事实。**无人值守 GPU 队列**（`pueue` + 产物 gate +
事件流 + systemd 自愈，已独立成仓库 `/home/bc/agent-gpu-queue`，`tools/queue` 为 symlink）：`q status` 一条命令给出每卡任务 /
实测 GPU 利用率 / 排队 gate 满足情况 / 失败签名，`q events` 是给 agent 的增量收件箱（没有任何机制能中断一个会话去通知它
「W03 半夜挂了」），并且**如实写明了自己的可见边界**——手工按 D-20 起的 W01/W02 不在队列里，
所以「`q events` 是干净的」只能推出「排过队的任务没挂」，推不出「什么都没挂」（`docs/QUEUE_USAGE.md`）。
**IO 提速 35×**：本地分片缓存 + 顺序保持的样本预取 + fd 池，把 NFS 延迟移出训练线程，
数据供给从 **2.216 s/step 降到 0.063 s/step**（`q3vl/whereb/scripts/perf_acceptance.py` 的预注册验收表，
由 `bench_supply` 于 08-10 实测；单次成员读 RAM 0.02 ms / 本地 ext4 6.5 ms / nfs-ro 17 ms），
主张口径是「**不改变任何权重**」并配等价性测试。**如实标注**：W01/W02 于 11:39 起跑、PERF-1 于 15:10 才合入，
所以这两臂**没有吃到**该优化（其 7h+ 墙钟包含约 2.2 s/step 的数据供给）；C01/C02 因缓存 manifest 每进程 memoise 一次、
预热 20:23 才完成，也不吃新缓存，**C03/C04 才是第一批受益的臂**（`.../what/NOTES.md §12.8`）。
**审阅流程战果**：四份实现审阅累计 **27 个 blocker 在上卡前被抓住**（S0 4 / Where-A 7 / Where-B 9 / What 7），
三个最有代表性的：

1. **ZeRO-3 静默缩表**（`REVIEW-impl-S0` B-2/B-3）：ZeRO-3 的 `zero.Init()` 在 `from_pretrained` 阶段就切片参数，
   使 `param.shape/numel` 变成 **0**，于是 `prepare_embeddings` 读到切片形状会把词表从 151,936 **缩到 151,680**
   （与代码自己文档承诺的「never shrink」相反），同一根因还让 `FreezeReport` 的参数量全部塌成 0、
   SPEC §9 项 2/3 的冻结数量断言在正式训练中**静默失效**。全部 preflight 模型侧证据都是单卡采的，
   这条路径**从未被执行过**；审阅期间一次两卡 smoke 74 秒崩溃，崩溃点与审阅独立推导**逐字一致**。
2. **oracle namespace 错配**（`REVIEW-impl-WhereB` §6.6）：`<arm>/train` 不存在会**响亮**失败，
   而 `<arm>/V_where` **存在**却缺 `curve`/`cband_normalization` ⇒ 会**悄悄**用另一次 run 的 latent 当 eval 监督。
   两个 split 只有一个报错，正是最难发现的形态。修法是 `assert_oracle_contract` 显式声明期望的归一化约定与 z 网格并断言
   生数据确实住在里面——正是 CLAUDE.md「s 缓存消费契约」要求的消费侧断言。
3. **W-B1 假保护**（`REVIEW-impl-WhereB` 第 6 轮）：D-B17 号称把读路径改到 `/mnt/nfs-ro`，实际只改了索引文件与三类已发布派生物；
   `records`/`images`——训练循环里 **100% 的流式 IO**——**仍走 hard 挂载 `/mnt/nfs`**，
   因为冻结索引里 `shard` 字段是绝对路径、`ShardStore` 只对相对路径拼 `shard_root`，
   **而启动断言会为它打绿灯**。判决：W03–W08 在修好前不得按「已受保护」的前提启动。

---

## 7. 数字附录（全部关键数字带出处）

出处根：`E = experiments/Q3VL_metacanvas_where_what_20260804/`，`R = /home/bc/data/runs/`。

| # | 数字 | 值 | 出处路径 |
|---:|---|---|---|
| 1 | Base SFT 墙钟 / 步数 / 样本 | 29,686.5 s = 8h14m47s / 4,976 步 / 159,215 | `R/q3vl_base_sft_20260804/train.log`、`job.marker` |
| 2 | eval_loss 首末 | 0.821242 @500 → **0.664593** @4976 | `E/s0_base_sft/CHECKPOINT_VERIFICATION.md` §3 |
| 3 | 结构率（n=64，两 ckpt） | 10/10 项满分（标签完整 1.00、泄漏 0.00、EOS 1.00、截断 0.00） | 同上 §4 |
| 4 | `<where>` token-F1 p50 / 逐字一致率 | **1.0000** / 0.5312 | 同上 §5 |
| 5 | Where-A oracle 上界（hi 中位，n=224） | BA-0 **0.9714** → BA-2 **0.9736** / BA-3 0.9724（cband12） | `E/where_a/REPORT.md`、`metrics.json` |
| 6 | Where-A 尾部收益 | band hi p10 0.7843 → **0.8258–0.8324**（+0.04 量级） | 同上 |
| 7 | Where-A 拟合成功率 / 越域中位 / preflight | 100.0%（0 拒绝）/ 0 / 8/8 PASS | 同上；`E/where_a/preflight_where_a.json` |
| 8 | Where-A 单基元脆弱性 | 30.8% 样本单活跃基元，低→高落差中位 +0.0123（多基元 +0.0057） | `E/where_a/REPORT.md` 失败案例节 |
| 9 | mentor 展示图 | **28 张**（V_where 固定 seed 抽样） | `E/where_a/viz_mentor/`（实测计数） |
| 10 | Where-A 吞吐 | 5.529 wall s/step @38 workers × 2,361 步 ≈ 3.6 h/臂；串行基线 ~104 s/step | `E/where_a/calibration_throughput_w38.json`；commit `3a834c7` |
| 11 | Where-B 墙钟 | W01 **7h22m46s**（11:39:31→19:02:17）/ W02 **7h11m43s**（11:50:34→19:02:17） | `R/where_b/W0{1,2}/job.marker` + `where_b_final.pt` mtime |
| 12 | Where-B gate | **4/10**，`WHERE-GATE-FAILED` | `E/where_b/REVIEW-result-W1.md` §1 |
| 13 | Where-B 终局 median soft-IoU | **0.5565 / 0.5617**（vs gate 0.75） | 同上 §1、§2.1（审阅逐位复现） |
| 14 | 相对逐图 oracle / p10 | 57.94% / 58.19%（gate 85%）；0.2358 / 0.2965（gate 0.55） | 同上 |
| 15 | grid 边界 F1 / oracle | **32.14% / 33.14%**（gate 75%） | 同上 |
| 16 | 面积 Spearman 三档 | 0.665/0.691（真实）→ 0.188/0.150（同图换指令）→ 0.002/−0.078（无语言） | 同上 §5.1 |
| 17 | 面积均衡同图配对校准 Δ | **+0.0032（p 0.460）/ +0.0113（p 0.028）** | 同上 §5.2 |
| 18 | 形状列全程变化 | W01 0.2898→0.3186（+0.029），同期 median +0.153 | 同上 §3.1 |
| 19 | 模型补上「中心先验→oracle」差距 | 覆盖 6.7%/10.1%、形状 15.1%/16.6%；21.5%/17.5% 不如全 1 掩膜 | 同上 §3.1 |
| 20 | 中心先验分层（居中类 n=147） | W01 **−0.005（p 0.674）** / W02 +0.019（p 0.119）＝打平 | 同上 §3.3 |
| 21 | readout ×12 的买到量 | 中位 +0.005、形状 +0.012；长尾 65→43、s 塌缩 33.5%→3.5% | 同上 §3.4 |
| 22 | 梯度裁剪区（未排除的替代解释） | `grad_norm` 稳态 2.1–5.5（W01 早期 13–27），`clip=1.0`，未扫过 lr/clip | 同上 §3.5 第 2 条 |
| 23 | 评测板合规性偏离 | 主榜含 **44%** `winner_confidence=low`（D-B14 裁定保留）；normal-only median **0.5081/0.5406**（更低） | 同上 §2.2 |
| 24 | genctx 规模 / 质量 | train 159,215 + V_where 896 + V_what 897，coverage 1.0；零截断零格式失败（V_where 主榜 896/896 `stop_reason=closed`） | `E/where_b/genctx_*.json`；`docs/WHERE_HEAD_REQUIREMENTS_2026-08-10.md` §2.2；`E/where_b/REVIEW-result-W1.md` §1 |
| 25 | IO 供给提速 | **2.216 → 0.063 s/step（35×）**；单次成员读 RAM 0.02 / ext4 6.5 / nfs-ro 17 ms | `q3vl/whereb/scripts/perf_acceptance.py` docstring；commit `c978b10` |
| 26 | 本地分片缓存规模 | Where-B 23 条目 / 27.5 GiB；What 补预热后 32 条目 / 32 GB | `E/what/NOTES.md` §12.8 |
| 27 | 审阅 blocker 累计 | **27**（S0 4 / Where-A 7 / Where-B 9 / What 7） | `docs/reviews/REVIEW-impl-{S0,WhereA,WhereB,What}.md` |
| 28 | C 波配置 | C01–C04，4,975 步/臂，effective batch 32（16×2），3.205 s/step 实测 | `E/what/config/EXEC4_launch.json`；`R/what/C01/train.log` |
| 29 | E1 探针终局 | 三池全踩线：vs 中心先验 Δ −0.029~−0.049（p_FWER ≤5e-4）、Δ_shuffle ≈ +0.002 | `E/where_b/probe_e1_whereattn_20260810/REPORT.md` §3、§7 |
| 30 | Qwen3-VL sink 普查（合取规则，进契约行） | sink 格 **9.90%**、吃掉注意力质量 **45.4%**、argmax 落 sink **96.3%**、400/400 首 token 即 sink；天花板 0.787→**0.729** | 同上 §2「两条规则的数字对照」 |
| 31 | PW5 探针 | 四种读法词特异 Δ>0 全 PASS（最弱 p=1.0e-05）；400 样本 24 s | `E/where_b/probe_pw5_fpresim_20260810/REPORT.md` §2 |

**判据纪律声明**：本简报涉及的全部空间场判据为 soft/hard-IoU + grid 级边界 F1 + 中心先验列三件套，
**全程无任何 AUC**（CLAUDE.md 红线）；阈值化一律用匹配 GT 面积的 top-k，无逐场调阈值。

**待周二更新的占位**：§5.1 Stage-What C01–C04 的判据表与上下界数字（预计 08-11 05:00 前后产出）；
§5.2 新 Where 方案的定档结论与最小验证探针清单。
