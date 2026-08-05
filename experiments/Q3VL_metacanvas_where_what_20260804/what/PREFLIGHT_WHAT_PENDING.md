# Stage-What preflight：已完成项与待 GPU / 待作业清单

协议 §14 共 15 项。Stage-What 负责其中 **项 8 的后半（`Q_color` 不读 `H_where`）、项 9、项 12、项 13、项 14**，
外加四条本仓库特有的前置。其余各项归 Base SFT / Where-A / Where-B。

## 一、已完成（CPU，`preflight/preflight_what.json`，11/11 pass）

在战役环境 `/home/bc/envs/q3vl_sft/bin/python`（transformers 4.57.1 / torch 2.10.0+cu128，
git `fef9f93`）下运行 `python -m q3vl.what.preflight --limit 400`：

| 检查 id | 协议条款 | 结论 |
|---|---|---|
| `WT-P8-no-h-where` | §14.8 后半 | pass。AST 标识符扫描：`color.py` / `attention.py` 中不存在含 `where` 的标识符（`torch.where` 按限定名单独放行，且测试验证放行不会掩盖真的 `h_where`）；`ColorConnector.forward` / `ColorStack.forward` 的形参集合被断言封闭；`ColorStack` 的参数名里无 `where` |
| `WT-P9-no-target-leak` | §14.9 | pass。`WhatModel.forward` 形参 == `MODEL_INPUT_KEYS` 白名单；`WhereSignals` 无 `i_tar`/`baked`/`target`/`lut` 字段；`META_KEYS` 不含 baked locator；oracle 输入只到达 C03/C04 |
| `WT-P12-gaussian-constraints` | §14.12 | pass。μ∈cube、σ>0（SPD）、`Σq_i ≤ 1`、opacity/existence∈(0,1)、前向与两组 raw 输出的梯度全有限 |
| `WT-P-zero-init-identity` | §7.6 红线 | pass。零初始化 `T(x)=x` 最大误差 **4.8e-7**；同时记录了协议字面公式的实测 `mean T(x)/x = 2.000`（见 NOTES D-W1） |
| `WT-P13-lab-units-and-grad-norms` | §14.13 | pass。归一化 Lab 绝对值 ≤1.5；raw/norm 放大倍数实测 a/b = 128.0、L = 100.0；`grad_norm(L_func)` 与 `grad_norm(10·L_hc)` 均有限并入库 |
| `WT-P14-bake-readback` | §14.14 | pass。① 读回格点自身误差 **0.0**；② 仿射函数 analytic→33³→四面体 最大误差 **4.8e-7**（关闭 clamp，因为 clamp 不是仿射）；③ 随机 Gaussian 混合的重采样代价实测 MAE **2.19e-4** / p99 **4.01e-3** / max **1.30e-2**（见下方风险 R1） |
| `WT-W1-gt-lut-resolves` | 本仓库前置 | pass（每 split 抽 400 条）。五个 split 的 `lut_id` 全部解析到真实文件，0 缺失 |
| `WT-W2-lut-unseen-disjoint` | §2.2 | pass（全量 index 扫描，非抽样）。`T_lut_unseen` 259 个 LUT 与 train(3149) / V_where(530) / V_what(531) / T_final(577) 交集**全为 0** |
| `WT-W3-srht-deterministic` | §7.1 | pass。跨实例逐位相同；距离比均值 1.00、区间 [0.75, 1.25]；`z_gt` 单位范数 |
| `WT-W4-param-match` | §7.5 | pass。FG 61,394,716 vs SB 61,406,641（generator 侧），相对差 **0.019%**；整模型六对配对相对差 **0.013%** |
| `WT-P-arm-matrix` | §8 | pass。12 臂全部可构造，4 WC × 2 generator + 4 控制臂 |

## 二、待 GPU（Base SFT 释放两卡后才能做，本次全程未用 GPU）

| id | 内容 | 依赖 | 为什么不能在 CPU 上代替 |
|---|---|---|---|
| `WT-G1` | `H_color` 抽取契约在真实 Qwen3-VL 上的验证：`hidden_states` 层数、post-final-RMSNorm 口径、`<color>` 段切片位置与 `n_color_tokens` 对齐 | Base SFT checkpoint | 需要真实权重与真实 tokenizer；Where-B 已在真机上验证过 `H_where` 的同一契约（`q3vl/whereb/tests/test_hiddens.py`），Stage-What 只是换了切片区间，但**必须自己跑一遍** |
| `WT-G2` | 证明 `where_prefix=True/False` 只改变序列而不改变其它任何输入（T01 vs C01 的唯一差别） | 同上 | 需要真实 forward 才能对比两条序列的 `H_color` |
| `WT-G3` | `F_pre` 真实形状 / 宽高比 / 与 `rgb_low` 网格对齐（§14.4 的 What 侧复核） | 同上 | 需要真实 vision tower |
| `WT-G4` | 冻结 Where checkpoint 接入：digest 校验、`m_low`/`m_hi`/`canvas_axis`/`canvas_rho`/`w`/`rho` 六路信号形状与取值域断言 | **Where-B 选出并冻结一个 checkpoint**（§5.6） | 现在还没有 checkpoint |
| `WT-G5` | 单 batch 显存 / 吞吐 / micro-batch 探测，使 effective batch = 32（§14.15 的 What 侧） | 两卡空闲 | 只能在 H100 上测 |
| `WT-G6` | bf16 下的数值复核：确认 renderer / bake / 四面体 / loss 全部在 float32（trainer 已用 `autocast(enabled=False)` 包住，但要在真机上验证没有外层 autocast 泄漏）。**并入审阅 N-11**：另需记录 `out.params`（在 autocast 区内解码，带 bf16 舍入 ~4e-3）与 `aligned_pool` 内 Mahalanobis 的**实测 dtype**，不要只查外层泄漏 | 同上 | 这是 Where-B review blocker B4 的同类问题，必须实测 |
| `WT-G7` | LPIPS 后端接入（`image_metrics` 目前在没有后端时报 `nan` 并显式列在此处，不做静默替代） | 需要预训练网络 | §12.2 要求 LPIPS |
| `WT-G8` | 33³ bake 延迟与 VLM 后增量延迟测量（§7.5 要求与质量并列报告） | 两卡空闲 | |

## 三、待作业（IO 密集，脚本已写好，**未执行**）

| id | 脚本 | 产物 | 备注 |
|---|---|---|---|
| `WT-J1` | `q3vl/what/scripts/pack_gt_luts.py` | `/mnt/nfs/bc/data/datasets/what-20260805/gtluts/`（indexed tar shards） | 已 `--dry-run` 过两个小 split（790 lut_id / 0 冲突 / 0 缺失）。全量约 3.4k 表、8.79 GiB 文本 → 约 1.5 GiB float32 二进制 |
| `WT-J2` | `q3vl/what/scripts/make_zgt_center.py` | `zgt_center.npz` + `zgt.jsonl` + `center_report.json` | **`mean_train_u` 只能用 train 的 lut_id**，否则 `T_lut_unseen` 的"未见"说法作废。脚本已把这条写死并记录进 report |
| `WT-J3` | `WT-W1` 的**全量**版本（现在是每 split 抽 400 条） | 全 5 个 split 的 `lut_id → preset_path` 完整映射 + 冲突报告 | 需要读 159,215 条 record，属重 IO，等排期 |
| `WT-J4` | §12.3 的 image-shuffle / instruction-shuffle 评测批次构造 | shuffle 索引 | 依赖 Where-B 的 `ShuffleIndex` 口径，等 Where 定档后对齐。instruction-shuffle 的**配对**差值已实现（`arm_metrics`），image-shuffle 的批次构造未实现 |
| `WT-J5` | **§12.3 的 `WC-1/2/3` 相对 `WC-0` 的 paired improvement**（审阅 N-9） | 跨臂配对表 | 需要 4 个 WC 臂在同一批 `V_what` 样本上的 per-sample 记录；实现是一个跨 `per_sample.jsonl` 的 join，等 T1–T4 wave 跑完 |
| `WT-J6` | **paired bootstrap 95% CI**（§10.4 / §12.4 / §13.1，审阅 N-9） | bootstrap 脚本 | 主差异的置信区间；`across-seed range` 需要 top-2 的多 seed 复跑先完成 |
| `WT-J7` | **33³ baked render 的最终图像指标与可视化**（§12.1 / §13，审阅 N-10） | 每样本第二组图像指标 | 现在 `sample_row` 只渲染 analytic；§13 的联图要求 `pred analytic render` 与 `pred 33³ render` 并排，交付前必须补 |
| `WT-J8` | 12 臂 `run_setup.json` 的 Where digest 一致性巡检 | 一次全量扫描 | `provenance.assert_where_consistency` 在每个臂启动时已强制，但全部跑完后应再做一次总巡检并写进 REPORT |

## 四、风险与预注册说明

### R1 — 33³ bake gate 可能很紧

§12.1 的 gate 是 `mean RGB MAE ≤ 1e-4` 且 `p99 ≤ 5e-4`。preflight 实测：**随机参数**的 48-Gaussian 混合，
analytic vs 33³ 四面体回读的 MAE 是 2.19e-4、p99 是 4.01e-3，即分别是 gate 的 **2.2 倍**与 **8 倍**。

这不是实现缺陷（仿射函数的回读误差是 4.8e-7，格点回读误差是 0.0，说明插值器本身无损），而是**各向异性 Gaussian
重采样到均匀 33³ 格点的固有代价**。`L_bake`（权重 0.10）正是为把它压下去而存在的，但**这条 gate 能否达成是一个实验
结果，不是实现问题**。请主 agent 预知：如果所有 12 臂都过不了 bake gate，正确的动作是按 §15 分阶段报告
（"解析 renderer 成立、33³ 交付不成立"），而不是事后放宽 gate。

参考：旧 RD-G 的 bake 判据是 ΔE00 p99 < 2（`experiments/RDG_transformer_20260803/tools/bake_check.py`），
与本轮的 RGB MAE 口径不可直接换算，但说明"接近零"这个说法在 ΔE00 口径下成立、在 1e-4 RGB 口径下需要实测。

### R2 — 90M 的 adapter 参数量

12 臂各约 91–94M 可训练参数，大头是 ModLN（`Linear(1024, 1024)` × 4/block × 6 blocks = 25.2M）。
这是 §7.3「`z_style` 生成的 ModLN 调制」+ 每子层独立参数的直接后果。若显存或时间不允许，唯一无损的削减点是
ModLN 投影共享或 `z_style` 先降维——**属于改结构，需主 agent 决策**。

### R3 — base conda 环境不能跑 Where-B 测试

`/home/bc/miniconda3` 的 transformers 是 4.36.0，没有 Qwen3-VL 类，因此 `q3vl/whereb/tests/test_hiddens.py` 等
在该环境下 error / fail。这与本次实现**无关**（`git status` 证实 `q3vl/whereb/` 未改动），但值得记进档：
**所有正式作业必须用 `/home/bc/envs/q3vl_sft/bin/python`**。

### R4 — `n_active` 与 `z_effective_rank` 的早期读数

见 NOTES 第五节：opacity 初值 `sigmoid(z−2)≈0.12`，所以 §12.3 的"激活数分布"在训练早期恒为 0，是设计而非坍缩；
`z_effective_rank` 必须在 ≥32 样本的评测集上算，micro-batch 上的读数无意义。

### R5 — `--out` 默认值已移出交付目录（审阅 N-16 / 审阅人自身事故）

审阅期间有人用默认参数跑了一次 `--no-data` preflight，把交付的 `preflight_what.json` 静默覆盖成 9 pass / 2 skip 的版本，
而覆盖后的文件**仍然 `ok: true`**，只有 `complete` 与 `skipped` 变了——正是本战役 s 缓存契约里说的「第二种失败模式是静默的」。

现在：默认 `--out` 是 `RUN_ROOT/preflight`（scratch），要发布必须显式给交付路径；并且把 `complete: true` 覆盖成 `false`
会被**拒绝**，`--force` 时先把旧报告备份成 `preflight_what.json.superseded`。

## 五、一句话结论

Stage-What 的全部 12 臂代码、§9 全配方 loss（含 amendment A-2 的 `d_func` 口径与 A-3 的统一 natural 采样）、
33³ 烘焙与四面体回读、以及协议 §14 的项 8b/9/12/13/14 preflight 均已实现并在 CPU 上通过
（11/11 preflight、**191 个单测**、T01/T08 mock 闭环 loss 单调下降）。
REVIEW-impl-What 的 **6 个 BLOCKER 全部清零**，各配回归测试。
**未启动任何训练，未占用 GPU，未执行任何重 IO 作业。** 进入正式训练还差：Where-B 定档一个冻结 checkpoint、
两卡释放后跑完 `WT-G1`–`WT-G8`、以及 `WT-J1`/`WT-J2` 两个数据派生物作业。
