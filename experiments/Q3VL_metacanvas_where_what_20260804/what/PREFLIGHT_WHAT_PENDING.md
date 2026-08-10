# Stage-What preflight：已完成项与待 GPU / 待作业清单

> **2026-08-10 更新（任务卡 EXEC-4）**：本文档「待 GPU」与「待作业」两节里，
> **C 波（C01–C04）需要的全部条目已完成**，实测数字见 `NOTES.md` 第十二节。
> 一句话状态：
>
> | 条目 | 状态 |
> |---|---|
> | `WT-J1` GT-LUT 打包 | **done** — 3,408 表 / 6,816 成员 / 3 shard / 2.4 GB / 0 冲突 0 缺失 |
> | `WT-J2` `mean_train_u` + `C` | **done** — 中心只用 3,149 个 train lut_id；`C = 29.4131` |
> | `WT-J3` 全量 `lut_id → preset_path` | **done**（`WT-J1` 的 collect 阶段就是全量，5 split 全部 record） |
> | `WT-J9` `<color>` 边界全语料扫描 | **done** — 162,359 条、超界 0、全语料 max 361（余量仅 21，见 D-EXEC4-3） |
> | `WT-J10` generated context | **done（本次补完合并）** — `forced_color` 的 train/V_what 合并从未跑过，已补；两 mode 覆盖率均 1.0 |
> | `WT-G1/G2/G3/G5/G6/G7/G8/G9` | **done** — `preflight/preflight_what_gpu.json`，7 pass / 1 skip(`WT-G4`) / 1 warn(`WT-G7` LPIPS 未装) / 0 fail |
> | `WT-G4` 冻结 Where 接入 | **skip（C 波不适用）**，理由落盘在报告里；**T01–T08 开跑前必须补做** |
> | `WT-J4`–`WT-J8` | 未动（评测/选择期作业，与 C 波开跑无关） |
>
> C 波已于 2026-08-10 20:11 上队列：C1 = C01(gpu0) + C02(gpu1)，C2 = C03/C04 同卡排后并 gate 在前一臂的
> `what_final.pt`。四臂共用 `--micro-batch 16`（`WT-G5` 实测）、`--keep-last none`（审阅 N-26）。
> 偏离 **D-EXEC4**（A-3 统一 natural 采样在无 Where checkpoint 时的声明式替代）见 `NOTES.md` §12.3。

协议 §14 共 15 项。Stage-What 负责其中 **项 8 的后半（`Q_color` 不读 `H_where`）、项 9、项 12、项 13、项 14**，
外加四条本仓库特有的前置。其余各项归 Base SFT / Where-A / Where-B。

## 一、已完成（CPU，`preflight/preflight_what.json`，11/11 pass）

在战役环境 `/home/bc/envs/q3vl_sft/bin/python`（transformers 4.57.1 / torch 2.10.0+cu128）下运行
`python -m q3vl.what.preflight --limit 400 --out <交付目录> --force`（**12 项**，amendment A-4 新增一项）：

| 检查 id | 协议条款 | 结论 |
|---|---|---|
| `WT-P7-color-context-flows` | §5.4（经 amendment A-4 转置到 `<color>`） | pass。四条：generated builder 的签名**不可能**拿到 GT 文本；缺闭合标签→截断并标记（非回退）；micro-batch 2/4/8 全部恰好 50/50 且奇数被拒；`C01`/`C02` → forced-prefix、其余 → with-where-prefix |
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
| `WT-G2` | 证明 `where_prefix=True/False` 只改变序列而不改变其它任何输入（T01 vs C01 的唯一差别）。**并入 amendment A-4（审阅者点名）**：同时测出 **GT vs generated 两条序列的 `H_color`** —— 差异范数、逐 token 对齐、以及 `<color>` 段起止位置在两条序列下的一致性。这是 A-4 需要的第一个数字，也是「generated 主榜比 teacher 差多少」的下界解释 | 同上 | 需要真实 forward 才能对比两条序列的 `H_color` |
| `WT-G3` | `F_pre` 真实形状 / 宽高比 / 与 `rgb_low` 网格对齐（§14.4 的 What 侧复核） | 同上 | 需要真实 vision tower |
| `WT-G4` | 冻结 Where checkpoint 接入：digest 校验、`m_low`/`m_hi`/`canvas_axis`/`canvas_rho`/`w`/`rho` 六路信号形状与取值域断言 | **Where-B 选出并冻结一个 checkpoint**（§5.6） | 现在还没有 checkpoint |
| `WT-G5` | 单 batch 显存 / 吞吐 / micro-batch 探测，使 effective batch = 32（§14.15 的 What 侧） | 两卡空闲 | 只能在 H100 上测 |
| `WT-G6` | bf16 下的数值复核：确认 renderer / bake / 四面体 / loss 全部在 float32（trainer 已用 `autocast(enabled=False)` 包住，但要在真机上验证没有外层 autocast 泄漏）。**并入审阅 N-11**：另需记录 `out.params`（在 autocast 区内解码，带 bf16 舍入 ~4e-3）与 `aligned_pool` 内 Mahalanobis 的**实测 dtype**，不要只查外层泄漏 | 同上 | 这是 Where-B review blocker B4 的同类问题，必须实测 |
| `WT-G7` | LPIPS 后端接入（`image_metrics` 目前在没有后端时报 `nan` 并显式列在此处，不做静默替代） | 需要预训练网络 | §12.2 要求 LPIPS |
| `WT-G8` | 33³ bake 延迟与 VLM 后增量延迟测量（§7.5 要求与质量并列报告） | 两卡空闲 | |
| `WT-G9` | **在线评测（`eval_fn`）的真实墙钟** | 两卡空闲 | NOTES 第十一节给了实测（CPU、What 侧 39.4 ms/sample）+ 外推（GPU 一次 eval ≈ 25 s，占比 0.2–0.4%）。**VLM 那一项是从 Base SFT 的 5.1 s/step 外推的，不是直接测的**。`make_eval_fn` 每次把 `eval_seconds` 写进 `eval.jsonl`，第一次真实 eval 之后即为实测 |

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
| `WT-J9` | **`<color>` 段 token 边界的全语料校验** —— **已按审阅 N-24 提升为开跑前硬前置** | `preflight/color_boundary_scan.json` | 384 来自 3,745 条抽样（max 324）；`gt_color_context` 超界**抛错**，一条超界样本会让某个臂在训练中途崩。脚本 `scripts/scan_color_boundary.py`（纯读 record，不解码图像、不 tokenise，用 build 已写好的 `tokens.color`）；门 `boundary.require_color_boundary_scan` 由 `run_what.py` 在**任何昂贵操作之前**调用，缺失/schema 过期/边界不符/未覆盖全部 split/有超界样本/有缺字段记录，六种都硬停 |
| `WT-J10` | **generated `<color>` context 生成作业**（amendment A-4，**WB-IMPL 负责**） | 每 split × 每 mode 一套 `genwhere/2` shards | Stage-What 的**硬前置**：`run_what.py` 在任何昂贵操作前 `assert_covers` 全 split，缺一条即拒跑。需要两套：`with_where_prefix`（T01-T08 + C03/C04）与 `forced_color_prefix`（C01/C02）。Base SFT 本就一次生成两段，v2 只是把 `<color>` 段留下 |

### 三-bis、amendment A-4 带来的排期依赖

| 依赖 | 谁负责 | 阻塞什么 |
|---|---|---|
| `genwhere/2` schema（v1 + `<color>` 段 + `mode` 字段） | WB-IMPL | 全部 12 臂的训练 |
| forced-`<color>`-prefix CLI 模式 | WB-IMPL | `C01`/`C02` 两臂 |
| 每 split × 每 mode 的生成作业执行 | 排期（需 GPU） | 全部 12 臂的训练 |

Stage-What 侧的消费契约已实现并有测试（`q3vl/what/stores.py` + `test_a4_color_context.py`）：
schema 非 v2、缺字段、`mode` 不匹配、覆盖不全，四种情况都在**第一步之前**硬停，不会在训练中途才发现。

## 四、风险与预注册说明

### R1 — 33³ bake gate 可能很紧

§12.1 的 gate 是 `mean RGB MAE ≤ 1e-4` 且 `p99 ≤ 5e-4`。preflight 实测：**随机参数**的 48-Gaussian 混合，
analytic vs 33³ 四面体回读的 MAE 是 2.19e-4、p99 是 4.01e-3，即分别是 gate 的 **2.2 倍**与 **8 倍**。

这不是实现缺陷（仿射函数的回读误差是 4.8e-7，格点回读误差是 0.0，说明插值器本身无损），而是**各向异性 Gaussian
重采样到均匀 33³ 格点的固有代价**。`L_bake`（权重 0.10）正是为把它压下去而存在的，但**这条 gate 能否达成是一个实验
结果，不是实现问题**。请主 agent 预知：如果所有 12 臂都过不了 bake gate，正确的动作是按 §15 分阶段报告
（"解析 renderer 成立、33³ 交付不成立"），而不是事后放宽 gate。

参考：旧 RD-G 的 bake 判据是 ΔE00 p99 < 2（`experiments/_archive/2026-08-10/RDG_transformer_20260803/tools/bake_check.py`），
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

### R6 — **战役环境里 `import torch` 会毒掉 `import sqlite3`（campaign-wide，非 Stage-What 特有）**

落地 A-4 时在战役环境 `/home/bc/envs/q3vl_sft` 实测到：

```
import sqlite3; import torch   -> 正常
import torch;   import sqlite3 -> ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6:
                                  version `CXXABI_1.3.15' not found
                                  (required by .../llm_factory/lib/.../libicui18n.so.78)
```

torch 加载的 libstdc++ 遮蔽了 `_sqlite3` 依赖链（libicui18n）所需的那一份。**本战役所有读已发布 shard 的
store 都经 `q3vl.data.shardio` 触到 sqlite3**，因此任何先 import torch 的进程之后都打不开 shard——
`run_what.py` 在构造 `ColorGenContextStore` 时正会撞上。

**已实测确认 `q3vl/whereb/stores.py` 在同一条链上**：

```
import torch; import q3vl.whereb.stores   -> 同样的 ImportError
```

即 **Where-B 的 `run_where_b.py` 有同样暴露**，这不是 A-4 或 Stage-What 引入的。

**Stage-What 侧已修**：三个入口脚本（`run_what.py` / `pack_gt_luts.py` / `make_zgt_center.py`）在最顶部
先 `import sqlite3`，一行成本，整个进程免疫；并加了 AST 单测 `test_every_entry_point_imports_sqlite3_before_torch`
钉住 import 顺序，防止被「整理」掉。库模块（如 `preflight.py`）**不加**该 guard——它们在 torch 之后才被 import，
guard 反而会让 import 本身失败。

**需要主 agent 决策**（越出「不改 whereb」边界，故未动）：Where-B 的入口脚本
（`run_where_b.py` / `make_generated_context.py` / `make_oracle_latents.py`）是否也加同样一行。
`WT-J10`（A-4 的 generated context 生成作业）正是 WB-IMPL 要跑的作业之一，**它会读写 shard**，
所以这条对 A-4 的排期是直接前置。

### R5 — `--out` 默认值已移出交付目录（审阅 N-16 / 审阅人自身事故）

审阅期间有人用默认参数跑了一次 `--no-data` preflight，把交付的 `preflight_what.json` 静默覆盖成 9 pass / 2 skip 的版本，
而覆盖后的文件**仍然 `ok: true`**，只有 `complete` 与 `skipped` 变了——正是本战役 s 缓存契约里说的「第二种失败模式是静默的」。

现在：默认 `--out` 是 `RUN_ROOT/preflight`（scratch），要发布必须显式给交付路径；并且把 `complete: true` 覆盖成 `false`
会被**拒绝**，`--force` 时先把旧报告备份成 `preflight_what.json.superseded`。

### 三-ter、评测的两个入口（NF-2 路线 (a)+离线互补）

> **2026-08-10 更新（任务卡 EXEC-5）**：离线入口已**入队**（`eval_C01`–`eval_C04`，
> pueue id 10–13，各自 gate 在本臂 `what_final.pt`，排在 C1/C2 训练波之后）。
> 接线时修掉三个会让它直接失败或静默出错的问题，其中最重的是
> **`I_tar` 读的是 `image.baked` = `I_in` 的副本**（详见 `NOTES.md` §13.3）。
> `WT-G7` 的 `still_open`（「evaluate_what 仍传 `lpips_fn=None`」）是**过期陈述**——
> 该接线 2026-08-05 的 `80a5078` 就在了；本次只是把 lambda 提成具名函数并补单测。

| | 在线 | 离线 |
|---|---|---|
| 入口 | `run_what.py` 构造的 `eval_fn`（`evalloop.make_eval_fn`） | `scripts/evaluate_what.py`（**已入队，2026-08-10**） |
| 数据 | `V_what` 固定 256 条确定性分层子集（清单落盘且进 `config_digest`） | 完整 `V_what`，双 context |
| 指标 | LUT function 级 + bake gate | §12.1 全套 + §12.2 图像分区分层 + §12.3 |
| 产出 | 每 500 步两行 `arm_metrics` + `gap`，写 `eval.jsonl` / `eval_per_sample.jsonl` | `main_board` / `ceiling_board` / `context_report` + `per_sample_*.jsonl` |
| 作用 | 保住将被选中的 checkpoint 文件（激活 B-2）、落实 `eval_steps` | **最终选择依据** |

`I_tar` 只在离线入口出现（已加测试断言：全包只有 `data.py` 定义与 `evaluate_what.py` 调用两处提及
`load_target_image`）。

## 五、一句话结论

Stage-What 的全部 12 臂代码、§9 全配方 loss（含 amendment A-2 的 `d_func` 口径与 A-3 的统一 natural 采样）、
33³ 烘焙与四面体回读、以及协议 §14 的项 8b/9/12/13/14 preflight 均已实现并在 CPU 上通过
（**12/12** preflight、**254 个单测**、T01/T08 mock 闭环 loss 单调下降且每步 50/50 teacher/generated）。
REVIEW-impl-What 的 **6 个初审 BLOCKER、复审的 NF-1、三审的 NF-2 全部清零**，各配回归测试——
其中 NF-2 的回归测试**不再依赖 `_EvalStub`**：既有对 `run_what.py` 调用点的 AST 断言（`eval_fn` 是否真的传了、
边界门是否在 trainer 之前），也有用**真实 `make_eval_fn`** 驱动真实 trainer 的 B-2 保护验证。
**未启动任何训练，未占用 GPU，未执行任何重 IO 作业。** 进入正式训练还差：Where-B 定档一个冻结 checkpoint、
两卡释放后跑完 `WT-G1`–`WT-G8`、`WT-J1`/`WT-J2` 两个数据派生物作业，以及 **`WT-J10`（WB-IMPL 的 `genwhere/2`
生成作业，amendment A-4 的硬前置）**。
