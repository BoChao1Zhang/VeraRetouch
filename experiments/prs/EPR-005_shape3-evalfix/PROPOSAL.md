PROPOSED

# EPR-005 · SHAPE3 补跑 eval（接线 `shape_residual`）

> **作业已排队但已 `q hold` 冻结**（pueue id 44 / 45 / 46），等本提案批准后 `q release`。
> 不训练、不改权重，只重打分。

## 1. 目标指标与基线

- **目标指标**：**形状残差 `shape_residual`** = `1 − IoU(pred, 该族解析形状的最小二乘最佳拟合)`
  （`q3vl/whereb/edgequal.py:311`），**不是 mIoU**。
- **基线值**：**不存在**——这正是本 EPR 的存在理由。
  `shape_residual` 全仓**零调用点**：`rg shape_residual` 只命中定义(:311)、导出(:32) 与散文；
  `/home/bc/data/runs/` 与 `experiments/` 下所有产物文件中该字符串命中数 **0**。
  两个 SHAPE3 臂因此是**用 IoU 判的**，而 IoU 恰是该臂声明「不能只看」的列。
- 附带参照（同批产出，非目标指标）：mIoU normal-only，SHAPE3_A **0.73467** / SHAPE3_B **0.75414** /
  P3'(1200) **0.74172**。

## 2. 预注册数字（冻结）

| 门 | 数字 | 处置 |
|---|---|---|
| 层-3 采纳 | `shape_residual(A)` **显著低于**对照，且 mIoU 不降 >0.01 | 采纳距离场重参数化 |
| 证伪 | A 的形状残差 ≥ 对照 | eikonal 重参数化不解决形状规整性，关停该层 |
| 可读性守卫 | GT 自拟合列必须 ≈0（同批产出） | 该列显著大于 0 ⇒ 度量在这批数据上失效，全表不可读 |
| 度量判别力 | 软场上：真椭圆 ≈0.05、非二次团 ≈0.27 | 已单测锁定（见 §6） |

**「对照」是谁——本 EPR 的核心修正**：原判据写的是 SHAPE3_B。实读 `config/run_setup.json`：
**A 从零训 1200 步**（`resume=null`），**B 从 P3'(1200) 热启动再训 1200 步**（累计 2400，且 `arm=P3prime`）。
A vs B **起点与步数双重不匹配**，违反本项目自己的 U4 步数匹配纪律。
⇒ **A 的步数匹配对照是 P3'(1200)**（同样从零训 1200 步）。三个臂都重打分，让形状列有匹配基线。

## 3. 假设一句话

成功则能说：**把场重参数化成距离场，能让预测的等值线成为一个连贯的形状族（形状残差显著下降），
而不只是让场变光滑**；失败则不能说光滑度与形状规整性可以由参数化一起解决。

## 4. 输入

- 评测：V_where local **400** → headline normal-only **224**；形状列只在**解析族**
  （radial / linear / band）上有定义，semantic 族无解析成员可拟合，**显式排除**（不是记 0 分）
- 权重：三个已落盘 checkpoint，**逐位不动**
  - A：`amort_SHAPE3_A_20260811/amort_step1200.pt`（其板的 `selected`）
  - B：`amort_SHAPE3_B_20260811/amort_step800.pt`（其板的 `selected`）
  - 对照：`amort_P3prime_20260810/amort_final.pt`
- 全量档（非探针）：eval 走完整 400。

## 5. 与既有产物的唯一差异

**只加一列**：eval 每样本多算 `shape_residual(pred)` 与 `shape_residual(gt)`（GT 自拟合对照），
其余评测流程逐字不变。**不训练**（`--eval-only`），所以新板的 mIoU 必须与旧板一致——
这本身就是本次重打分正确性的自检。

## 6. 判据表（含运行时断言）

| 列 | 规则 | 运行时断言 |
|---|---|---|
| shape_residual | 逐样本，k 取匹配 GT 面积的 top-k（禁逐场调阈值） | **`assert_criteria_ran(board,"SHAPE3")`：登记表要求该列，计数为 0 直接 `AssertionError`，不许出板** |
| shape_residual_gt | GT 自拟合对照，与预测同 k | 与主列同生共死（同一函数产出） |
| 族分层 | radial / linear / band 分别报 | `by_family` 字段 |
| mIoU 一致性 | 新板 mIoU 应复现旧板 | 人工对表（重打分自检） |
| **AUC** | 禁 | 不产出 |

**度量已验判别力**（单测 `test_shape_residual_is_wired_and_discriminative`）：
软场上真椭圆 **0.000**、软斜坡 **0.000**、非二次团 **0.222**、椭圆+噪声 **0.051**
（对照 HANDOFF 公布的 0.047 / 0.040 / 0.268）。
⚠️ 拟合在 **logit 空间**做最小二乘，**硬 0/1 掩膜恒得 ~0.29** 无论形状多正确；
预测与软 GT 都是 sigmoid 软场，正是该度量的适用域。

## 7. 资源预算与回退

- 预算：**≈0.6 GPU·h**（3 × eval-only，每次 6 上下文 × 400 样本，无训练）
- 回退：若 GT 自拟合列不 ≈0 ⇒ 判 metric 在本数据上不可读，出「不可判」而非硬判；
  若 A 与对照差异不显著 ⇒ 记「不可判定」，不得改判据凑结论。

## 8. 状态流转

`**PROPOSED**（待主 agent 批准）→ APPROVED → RUNNING（`q release 44 45 46`）→ SETTLED`

## Changelog

- 2026-08-12：建档。相对 HANDOFF §3.3 的原判据有**一处实质修正**：对照臂由 SHAPE3_B
  改为 P3'(1200)（步数与起点匹配），B 仍重打分并列出，但不作为 A 的主对照。
