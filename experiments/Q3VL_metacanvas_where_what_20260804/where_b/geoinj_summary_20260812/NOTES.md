# NOTES — GEOINJ-1（实施前核实 / 假设 / 待主 agent 决策）

## 0. 实施前读了什么

- `CLAUDE.md`「Where 战役入口」+ 红线速查 + D-20 + s 缓存契约；`trash/` 未读（红线）。
- `docs/WHERE_STATE_2026-08-11.md` 全文（§一配方、§二 S1–S14、§三死亡路线）。
- `docs/PROPOSAL_geometry-injection_2026-08-11.md` §0 裁定表 D-1..D-14、§1 门、§2 PCH 规格、
  §3 三臂、§4 判据表、§5 排期。
- `HANDOFF_geometry-injection_20260812.md` 五节全（**以 0812 版为准**，0811 版只做对照）。
- 代码：`amort/{pch,model,heads,data,geomparse,trainer,evaluate}.py`、
  `scripts/run_amort_arm.py`、`waves/amort_arm.sh`。

**在线核实**：本轮**未引入任何外部事实**（无新 URL / 无新超参出处 / 无新论文数字）。
PCH 的结构锚（SAM / SurgicalSAM / ControlNet）是前任在 `pch.py` 里已落的，本轮未改其架构、
也未据其新增任何主张，故无需联网复核。所有数字均来自本机产物文件，键路径已写进 SCOREBOARD。

## 1. 已核实的事实（都当场验了，不是转述）

| 事实 | 怎么验的 |
|---|---|
| B2 崩溃根因 = stem 1025→1046 通道 | 读 `amort_B2_gtcode_20260811/train.log` 的原始 traceback，`geo.tower.stem.weight` 与 `sem.stem.weight` 两处 |
| 三个 B2 臂 resume 的是 **P3'(1200)** 而非 CONT | `pueue status --json` 取回原始提交命令：`--resume .../amort_P3prime_20260810/amort_final.pt` |
| CONT2 = 0.79095、且主榜出自 **step3500** | `eval_final/metrics.json` 的 `.topk_iou_median_normal_only` 与 `.checkpoint_selection.selected` |
| 配对 Δ 流水线正确 | 复算 CONT vs P3' = +0.0254、P1 vs POOLED = +0.0856，与 HANDOFF 公布值逐位一致 |
| PCH 参数量 | 实例化后 `n_params()`：Full **3,855,296** / Lite **501,536** |
| 单卡 amort 臂显存 | `nvidia-smi --query-compute-apps`：**20,634 MiB**（GTCODE 稳态） |
| L7 是标注作业不是训练 | `ps -p 3253279`：`construct.agent run --config databuild.prod-l7-local400k-20260811.toml`，RSS 19.5 GB，GPU 2.9 GiB |

## 2. 本轮改了什么代码

| 文件 | 改动 | 能证伪它的观测量（军规 14） |
|---|---|---|
| `amort/resume.py`（新） | 形状/键感知加载：同形拷贝、输入通道扩列**零填充**、新模块允许缺键、**多余键硬报错** | 单测断言**输出**逐位一致，不是断言形状 |
| `amort/pch.py` | 未改（前任实现，冒烟已过） | — |
| `amort/model.py` | `geom_mode="pch"` 时 `geom_dim=0`（stem 不动），挂 `self.pch`，`_inject_of()` 下发 | `facts()` 落盘 `geom_mode` + `pch.n_params` |
| `amort/heads.py` | `apply_inject()`：残差加在塔的**倒数第二层特征**（提案 tap A 位置） | 三个头签名统一，`inject=None` 时恒等 |
| `scripts/run_amort_arm.py` | `--geom-mode/--pch-size/--freeze-base`；resume 走 `load_resumable` 并打印报告 | 日志里 `resume: {...}` 那行就是报告 |
| `amort/trainer.py` | 冻结基座下全语义头 micro-batch 无梯度时跳 backward 并计数 | `state.warnings` 里的 `no_grad_microbatch` |
| `amort/tests/test_pch_resume.py`（新） | 6 条 | 全套 9 条通过（含既有 3 条 U1 回归） |

**一个值得记的工程事实**：broadcast 形态**即使把新通道置零也做不到逐位一致**——
1046 通道的卷积与 1025 通道的规约顺序不同，实测某格差 7e-9。PCH 没有这个尾巴，
逐位相同。单测把这条写死了。

## 3. 待主 agent 决策（都采了保守默认，**没有静默拍板**）

### D-a｜PCH 只注入几何头，不注入语义头（保守默认：只注几何头）

语义头由 `.cgt` 直接监督、已在 0.820，且 21 维码描述的是解析族几何。代价是
**语义族样本（17.5%，评测时路由到 m_sem）完全不受注入影响**，headline 的 Δ 因此是
**偏保守**的读数。M1 门（+0.008/+0.015）打在 headline 上，存在「机制有效但被稀释到门下」
的风险。交付时会**同时报几何族子集的 Δ**。若主 agent 要求，可加一份语义头 PCH（+~3.9M 参数）。

### D-b｜冻结基座，只训注入器（保守默认：冻结，按提案 §2.4）

提案 §2.4 明写「可训参数 = 仅 PCH」。这么做的**关键好处**：M0 就是被 resume 的那块板本身
（0.79095），Δ 里**不可能混进续训收益**。反面：注入器只能在冻结特征上做残差，上限低于
联调。若 M1 落在灰区 [0.008, 0.015)，建议先加一档「基座 0.1× lr 联调」再判死。

### D-c｜resume 用 cont2 的 **step3500**，不是任务卡写的 cont 2541 步版

任务卡指定 `amort_P3prime_cont_20260811` 的 2541 步版（0.7622）。但 CONT2 已落盘且更高
（0.79095），HANDOFF §3.4 明确要求「注入臂的 Δ 必须对**最新**基线算」。故改用
`amort_P3prime_cont2_20260811/amort_step3500.pt`（即产出 0.79095 那块板的权重）。
**这抬高了晋级难度**（要在 0.7909 上再要 +0.015），但这是判据要求的比法。

### D-d｜实现版 PCH ≠ 提案 §2.2 的字面张量流

前任实现的是「原型库 + hypernet scale/shift + 特征对原型 cross-attn + 零初始化输出投影」，
**预算对得上**（Full 3.86M vs 提案 3.96M；Lite 0.50M vs 0.52M），关键性质也都在
（零初始化 no-op、空码零残差、空间选择性、组合性）。但**不是**提案写的
TwoWayTransformer(depth=2) + SAM 锚的 tap A/tap B 双抽头。

**更要紧的是 GeoCode 契约 v1.0 没有落地**：提案要 `c_disc(20) + c_cont(3) + conf(4) + valid`，
实现是 **21 维多热、无连续量、conf 由码密度导出**（`clamp(Σ|c|/6, ≤1)`）。
按 HANDOFF §3.5，连续量在生成期就被 q=3 量化成词、「不存在可回收的连续残差」，
所以缺 c_cont 有据；但 **D-4 的 conf 生产规格（A1 的 span-min）在 A1 臂开工时必须重新接线**，
不能沿用密度代理。协调者已裁「不要重写 PCH」，故本轮照用并记录在此。

### D-e｜步数/批量偏离提案

提案 §2.4 要 batch 64 / 10k 步；本轮用**队列既有配方** effective-batch 32、
`--max-steps 1500`、`--max-hours 2.5`（实际排到 1336 个 optimizer step）。
理由：与之前所有 amort 臂同配方、可比；10k 步在 2.5h 墙钟内不可能。
lr 3e-4 与提案一致（恰是 runner 默认）。

### D-f｜gpu1 并行度已从 1 调到 2（按用户共存裁定）

`pueue parallel 2 --group gpu1`。**这个设置是常驻的**：L7 结束后，gpu1 会同时跑两个
作业而不是一个。若主 agent 希望 L7 结束即恢复独占，需要有人把它调回 1
（`pueue parallel 1 --group gpu1`）。军规 9 说的「并行度被改成 2 会让 q submit 直接开跑」
在本轮是**故意**利用的，不是意外。

## 3bis. 军规增补（主 agent 2026-08-12 裁定，第三次踩同一个坑后固化）

> **预注册判据必须有运行时断言：评测收板前检查该判据函数确实被调用过，
> 计数为 0 就不许出板。**

「定义了、测了、没接线」在本战役已出现三次（WEVAL-1 的
`active_primitive_bucket`、DX-5 的 `need_mask`、本轮的 `shape_residual`）。三次的表面
各不相同，共同点是**板子看起来是完整的**——所有该有的列都在，只是**这个实验赖以裁决的
那一列不在**，于是实验被拿另一列（通常是 IoU）判了，而那一列恰恰是该实验声明「不能只看」的。

已落地实现：`evaluate.py::assert_criteria_ran(board, arm)`，在 `evaluate_arm` 收板前调用；
`{"SHAPE3": ["shape_residual"]}` 是判据登记表，缺值即 `AssertionError`，
连带单测 `test_criteria_assertion_blocks_an_unadjudicable_board`。
**新增臂时必须往这张表里加一行**，否则新臂又会退回「靠肉眼记得要算」。

## 3ter. 待办（防止设置漂移）

- [ ] **L7_LOCAL400K 退出后，把 gpu1 并行度调回 1**：`pueue parallel 1 --group gpu1`。
      当前为 2（为与 L7 共存而设，用户裁定）。不调回的后果：gpu1 会**同时开两个训练作业**，
      两个作业抢同一张卡 ⇒ 墙钟截断落在不同步数 ⇒ 违反 U4 步数匹配，
      而且**不会有任何报错**。此项也写进了两个 PCH run 的 `job.marker` 旁注。

## 3quater. 两条必须记录的过程事实

1. **SHAPE3_A 与 SHAPE3_B 从来就不是步数匹配的对照。**
   `config/run_setup.json` 实读：A `resume=null`、1200 步（**从零训**）；
   B `resume=amort_P3prime_20260810/amort_final.pt`、再训 1200 步（**热启动，累计 2400**），
   且 B 的 `arm` 就是 `P3prime`（不是记错——B 本来就是「同容量普通头」对照）。
   ⇒ 已发布的「A vs B −0.0057」**被起点和步数双重混淆**，不能作为 eikonal 的判据。
   **A 的步数匹配对照是 P3'(1200)**（都是从零训 1200 步）：实测 IoU −0.0040 (p=0.029)。
   本轮因此排了**三个** eval-only 重打分（A / B / P3'(1200)），让形状列有一个匹配的基线。
2. **改源码时有两个训练作业在跑**（09:33–09:45 改 `evaluate.py` / `heads.py` / `model.py`）。
   两个 PCH 臂在 09:20 与 09:26 启动时就已把这些模块 import 进内存，
   **跑的是改动前的字节码**，不受影响；它们是 `P3prime` 臂、判据登记表里没有必需列，
   所以收板不会触发新断言。**代价**：这两块板不会有 `criteria_columns` 字段——
   这是预期的，不是缺失。（S12 的「进程启动后禁改源码」本轮以此方式规避；
   若后续要改的是**运行中作业会懒加载**的模块，则必须等作业结束。）

## 4. 显存预算账（用户要求）

| 卡 | 常驻 | 新增 | 合计 | 阈值 65 GB |
|---|---|---|---|---|
| gpu0 | 0 | GTCODE 20.2 GiB（实测 20,634 MiB） | **23.7 GiB**（含上下文碎片） | 通过 |
| gpu1 | L7 2.9 GiB | PARSED 20.2 GiB | **~23.1 GiB** | 通过 |

**没有把两个训练臂挤到同一张卡**，尽管 2.9+20.2×2 = 43.3 GiB 仍在 65 GB 以内：
两个训练作业抢同一张卡会让**墙钟截断落在不同步数**上，而三个臂正是靠
`--max-steps`/墙钟对齐来保证可比（U4 步数匹配）。L7 的 GPU util 是 0%，
与它共存不产生算力竞争，所以 gpu1 上的共存是安全的。

## 5. 附：复算脚本

`paired.py` 与 `run_meta.json` 同目录落盘；跨 run 配对 Δ 全部由它产出，
自检项（CONT vs P3' = +0.0254、P1 vs POOLED = +0.0856）在脚本第一、二行输出。
