## 要验证的结论

如果这个实验成功，我们就能说「RO-9（GL token 读出）拿不到 where 只是**层选错了**，换到早期层
就能拿到指令条件的区域场」；失败就不能说这句话，只能说「GL token 的 attention 在**任何一层**
都不随指令改指区域，RO-9 作为 where 读出**判死**」。

## 为什么需要验证它

D-30 已判 canonical 读出 FAIL，但 D-32 留了最后一条线索：早期层 ρ_region_opp 显著更低
（L0=0.045 / L4=0.270）。低相关有「真随指令变」和「场不可复现」两解，不判别就无法回答审稿人
「你只测了 L8–15，凭什么说整个模型里没有 where」—— 这条不关掉，RO-9 的死因就永远可争议，
H2（VLM 里有 where）的后续臂（RO-1/RO-3/RO-2/RO-5）也失去干净的起点。

## 怎么验的

对 G1 已落盘的三 token 逐层 s 场（`run_regfull` 214 源 × {reg_a 点名主体, reg_b 点名补集}，
**零额外前向**）逐层算三组数：目标区域 AUC（reg_a→SAM3 主体掩膜 / reg_b→其补集）、
ρ_region_opp、以及空间自相关 Moran's I 与有效秩（各配同支撑空间置换零模型），
按预注册规则对 24 层逐层判「救回 / 判死」。

---

## 一、设置

| 项 | 值 |
|---|---|
| 实验编号 | RO9-L（D-32 派生；顺带完成 D-22） |
| 输入 | `experiments/G1_s_identifiability_20260803/run_{regfull,regsmoke20,full,smoke30,shufctrl}/stacks`（1388 npz） |
| 数据 split | 区域对立批 214 源（S-val 源，`config/g1_region_opp.json`）→ **有效 212**（2 源主体掩膜 16×16 占比为 0，AUC 无定义，见 NOTES §〇.1）；主批 300 源；错位对照 60 源 |
| 分层 | region_b_kind：**background 165**（reg_b 字面 = 主体补集，**主判据口径**）/ spatial 47（「左半/右半」，并列口径） |
| 标签 | SAM3 主体掩膜银行 `cache/subject/.subject.png`（D-MASKBANK），16×16 格覆盖 ≥0.5 判正，只取 valid（非 pad）格；主体面积16 中位 0.138 |
| 读出 | pre-softmax head-mean attention logit（D-0 修复后），逐层不聚合；主叙事 token = `<retouch_light>`（U5），另两 token 为 D-22 附录 |
| 计算 | 纯 CPU numpy，wall-clock **107 s**，seed 20260804，空间置换零模型 32 次/场/层；**未占用 GPU** |
| 环境/commit | `config/env.json`（`0e5d04a`, branch `lens-exp`, numpy 2.5.1 / scipy 1.17.1） |

## 二、预注册判据 vs 实测数字

判别规则在跑数前写死在 `analyze_layers.py:VERDICT_RULE`：
**救回 ⟺ ∃层 L 使得 AUC_target(L) ≥ 0.65 且 ρ_region_opp(L) < 0.30 且 该层非白噪声。**

| # | 预注册判据 | 实测 | 判定 |
|---|---|---|---|
| ① | 存在层使 **AUC_target ≥ 0.65** | **24/24 层不过线**；全层区间 **0.479–0.526**，最好层 L20 = **0.526**，bootstrap CI95 **[0.390, 0.630]**（含 0.5） | **FAIL** |
| ② | 该层 **ρ_region_opp < 0.30** | 只有 **L0 (0.045)** 与 **L4 (0.270)** 满足；两层的 AUC_target = **0.487 / 0.495** | 条件满足但①不满足 |
| ③ | 该层**非白噪声** | L0/L4 均**非**白噪声（Moran's I 0.446/0.232 ≫ 零模型 −0.007；erank 5.99/7.92 < 零模型 8.98） | 条件满足但①不满足 |
| ④ | 救回层集合 | **空集** | — |
| **裁决** | AUC 接近 0.5 或场是白噪声 → **判死** | **AUC 全层 ≈ 0.5** | **RO-9 判死（DEAD）** |

**逐层三条曲线（light token；完整 24 行见 `metrics.json.layers`）**

| L | AUC_target(bg) | AUC(s_a,M) | AUC(s_b,M) | ΔAUC 配对 | ρ_region_opp | ρ_syn | sep | Moran's I / 零模型 | erank / 零模型 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.487 | 0.506 | 0.533 | −0.014 | **0.045** | 0.025 | −0.020 | 0.446 / −0.007 | 5.99 / 8.98 |
| 3 | 0.484 | 0.471 | 0.492 | −0.021 | 0.426 | 0.352 | −0.073 | 0.394 / −0.005 | 8.03 / 8.97 |
| 4 | 0.495 | 0.537 | 0.547 | −0.011 | **0.270** | 0.229 | −0.041 | 0.232 / −0.008 | 7.92 / 8.99 |
| 11 | 0.494 | 0.812 | 0.813 | −0.002 | 0.995 | 0.994 | −0.001 | 0.493 / −0.006 | 7.68 / 8.88 |
| 20 | **0.526** | 0.799 | 0.776 | +0.006 | 0.944 | 0.946 | +0.002 | 0.425 / −0.006 | 8.12 / 8.89 |
| 22 | 0.479 | 0.834 | 0.822 | +0.006 | 0.967 | 0.967 | +0.001 | 0.460 / −0.004 | 7.81 / 8.88 |
| L8–15 均值（canonical） | 0.504 | 0.648 | 0.642 | +0.0005 | 0.884 | — | — | — | — |

全 24 层：**max|ΔAUC 配对| = 0.0255**（L14）；**max sep = ρ_syn − ρ_region_opp = +0.0024**（L21）
—— 即**没有任何一层**上「换区域」比「换同义说法」更能改变 s 场。

## 三、逐案例口径（`metrics.json.per_case_rule_pass`）

| 量 | 值 |
|---|---|
| 至少一个早期层（L0/L3/L4）满足 ρ<0.3 的源 | 196 / 212 |
| 在那些层上**两侧 AUC_target 都 ≥0.65** 的源 | **3 / 212 = 1.4%** |
| 逐层通过数（两侧都 ≥0.65） | L0: 2, L3: 0, L4: 1, **L11: 0, L22: 0** |
| 逐层 min(两侧 AUC_target) 中位 | L0 0.422, L3 0.421, L4 0.429, L11 0.187, L22 0.178 |

3 个通过源全部落在早期层（那里逐源 AUC 方差最大），canonical 与后期层 **0/212** —— 与
「纯偶然命中」一致。唯一被标 `success_early` 的源（`src_d891d170b448df60`）在 L4 刚好压线
（0.71 / 0.65，救回分恰为 0.65），其 L4 场目视仍是行状条带（`viz/success_early_sfield_*.png`）。

## 四、结论

1. **RO-9 判死为 where 读出，且死因从「层没选对」收敛到「这个读法本身没有 where」。**
   判据量 AUC_target 在 **24/24 层**贴着 0.5（0.479–0.526），配对 ΔAUC 在**任何层**都
   ≤0.026。注意区分两个数：`AUC(s_a, M)` 在后期层可达 **0.834**（L22），但同一层
   `AUC(s_b, M) = 0.822` —— 指令从「点名主体」换成「点名背景」，s 场几乎不动。
   **s 是图像驱动的主体显著场，不是指令条件的区域场**（`viz/layer_curves_auc_rho.png` 面板 A
   两条曲线逐层重合，是本报告最直观的一张图）。

2. **早期层的低 ρ 是「场不可复现」，不是「随指令改指」——但也不是白噪声。**
   L0 上 ρ_region_opp=0.045 的同时 **ρ_syn=0.025**（同义改述）、**ρ_shuf=−0.007**（跨图错位
   指令）：任何措辞改动都让 L0 的场彻底变掉，所以低 ρ 不含指令信息。
   结构性测量则显示 L0/L4 **不是白噪声**（Moran's I 0.446/0.232 vs 零模型 −0.007；
   erank 5.99/7.92 vs 零模型 8.98）—— 场是**行状条带**（见 `*_sfield_*.png` 的 L0/L3/L4 两行），
   空间上有结构、语义上无内容。预注册规则里的「非白噪声」子句因此**未成为承重项**：
   救回在判据①上就已出局。

3. **早期层低 ρ 的成因：长度/位置假说被否定（null 结果）。**
   `Spearman(ρ_region_opp, |Δtoken数|)` = **−0.036 (L0) / −0.028 (L4)**（reg_a 与 reg_b 的
   token 数差中位 12）。本实验只**排除**了这个解释，未给出正面机制。
   见 `viz/diag_prompt_length_NULL.png` 与 NOTES §三.1。

4. **D-31 的措辞纪律在这里再次应验**：ρ_region_opp 在 L0「过线」（0.045 < 0.3）
   完全没有正面价值 —— 未触发死刑 ≠ 获得正面证据。本实验就是把这条线索用一个
   **正面证据量（AUC）** 关掉的。

5. **D-22（顺带完成）**：
   - **跨 token 两两空间 ρ（canonical, reg_a 条件）**：light~colortemp **0.936**、
     colortemp~colormixer **0.900**、light~colormixer **0.810** → 三个 special token 的
     attention 图高度同质，**U5 选 light 不损失信息**；逐层曲线见
     `metrics.json.d22_cross_token_rho_layer`。
   - **三 token 各自 AUC（canonical）**：`AUC(s_a,M)` light **0.648** / colortemp **0.649** /
     colormixer **0.606**（luma 基线 0.465）；判据量 `AUC_target` 分别为 **0.507 / 0.510 /
     0.504**，配对 ΔAUC **0.0005 / 0.0094 / 0.0000**。→ **换 token 救不回 RO-9**，
     三条 token 的结论完全一致，light 作主叙事、另两为附录的分工成立。

## 五、建议下一步

1. **`EXPERIMENTS_v3` 的 RO-9 行**：状态从「待判」改为 **淘汰（where 读出）**，
   并注明降级用途 = 「图像驱动主体显著性」的 analysis 素材（推荐层 L20–L22，
   `AUC(s,M)` 0.78–0.83，**但必须同时标注它与指令无关**，禁止出现在任何主张 where 的论证链里）。
   D-33 的第一梯队（RO-1 / RO-3 / RO-2 / RO-5）按原优先级推进，**不受本判死影响**——
   被证伪的仍然只是「GL token 的 attention 这个读法」，不是 H2。
2. **给 RO-3（全层全头扫描）加一行预期**：本实验是 **head-mean** 口径。head-mean 把
   24×14 = 336 个头压成 24 条曲线，若某几个头单独携带区域信息会被平均掉。
   RO-3 应把本报告的 **AUC_target + 配对 ΔAUC** 直接作为逐头判据量复用（阈值沿用 0.65 / 0.30），
   这样 RO-9 与 RO-3 的结论可以并排读。
3. **早期层行状条带**单列一条待查项（NOTES §三.1）：来源是位置编码、单头结构还是 D-0
   插值残留，目前未知；它不影响任何判据，但会影响 RO-3 对早期层结果的解读。
4. **本报告的判据量建议提升为 RO 系通用判据**：`AUC_target`（reg_a→区域 / reg_b→补集混池）
   比 ρ_region_opp 更抗「平凡低相关」—— ρ 低有噪声和真信号两解，AUC 只有一解。
   建议写进 `EXPERIMENTS_v3` 的 RO 系公共判据行。

## 六、交付物

```
experiments/RO9_layer_verdict_20260804/
  REPORT.md                     # 本文件
  NOTES.md                      # 落盘核对声明 / 方法学决策 / 待决策项 / 红线自查
  metrics.json                  # 逐层 24 行 + 逐案例 212 行 + D-22 + viz 选例规则
  analyze_layers.py             # 一次跑出全部数字与图（CPU, 107 s）
  config/env.json               # commit / seed / 环境 / 输入清单 / 预注册规则
  config/g1_region_opp.snapshot.json
  config/run_analyze.log
  viz/layer_curves_auc_rho.png          # 三面板：A 两指令 AUC 重合 / B 判据量 AUC_target+ΔAUC / C ρ 与结构量
  viz/diag_prompt_length_NULL.png       # 长度假说的 null 结果（如实登记）
  viz/success_early_sfield_*.png        # 1 张：唯一逐案例压线过的源（L4, 0.71/0.65）
  viz/bestcase_early_sfield_*.png       # 3 张：早期层最有利但未过线的源
  viz/failure_early_sfield_*.png        # 4 张：早期层典型失败（条带噪声）
  viz/success_saliency_sfield_*.png     # 3 张：RO-9 降级用途成功案（主体显著性对上，但 reg_b 同样对上）
  viz/failure_saliency_sfield_*.png     # 3 张：连主体显著性都不成立的源
```

每张 `*_sfield_*.png` 为 2 行 × 6 列：上行 reg_a、下行 reg_b，列为
[原图/SAM3 掩膜, L0, L3, L4, L11(canonical 带内), L22]，每格标注该条件下的 **AUC_target**。
成功/失败两端用同一条主体面积筛（0.04–0.60，172/212 源入选），规则落在
`metrics.json.viz_case_selection`。
