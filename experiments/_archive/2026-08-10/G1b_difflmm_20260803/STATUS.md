# STATUS — G1b / RO-D（DiffLMM attend-and-segment 读出重跑 G1）

## 排卡与资源（硬约束逐条核实）

| 项 | 约束 | 实测 / 采用 |
|---|---|---|
| 卡 | 卡 1 | ✅ 全部批次 `CUDA_VISIBLE_DEVICES=1 --device cuda:0` |
| 显存 | ≤ 20 GB | 实测 **2.5 GB / worker**；`set_per_process_memory_fraction(0.05)` = 4.9 GB 上限 × 4 worker = **≤ 19.6 GB** |
| 起跑前卡 1 占用 | — | 73.7 GB / 97.9 GB（RO-1/2/3/W + RD-G 的 G-Base） |
| 别人的进程 | **严禁 kill** | ✅ 只按 PID 停过**本实验自己**的 worker（`config/restart_shards.sh` 用 `[r]od_...` 括号模式，且只匹配本实验的 `--plan` 命令行）；**未 kill / 未 SIGSTOP 任何他人进程** |
| CPU worker | ≤ 4 | 4 个 worker × `torch.set_num_threads(3)`，`OMP/MKL_NUM_THREADS=4`；**无 dataloader** |
| 提交方式 | nohup 后台 + `job.marker` | ✅ `job.marker`（PID + 完整启动命令 + 日志路径 + 计划） |
| 断点续跑 | 已有结果自动 SKIP | ✅ `--skip-existing`（按 `stacks/<img_id>__<instr_hash>.npz`）+ tmp→`os.replace` 原子写 |

## 批次计划（优先级排序；算力紧张时从后往前砍）

主 agent 追加要求：**优先保 2×2**（{prompt 顺序} × {读出方式}），而不是把单一条件跑到更大 n。
计划据此重排过一次（第 3 版）。

| # | 批 | prompt 顺序 | 源 × 指令 | 读出数 | 承重的判据 |
|---|---|---|---|---|---|
| 1 | `run_region` | image_first（部署默认） | 120 × {reg_a, reg_b} | 240 | **主判据 ρ_region_opp** + AUC + AUC_target + 差分场 |
| 2 | `run_floor` | image_first | 120 × {reg_a_para} | 120 | **匹配噪声地板** ρ_floor |
| 3 | `run_region_ifirst` | **instr_first** | 120 × {reg_a, reg_b} | 240 | **2×2 第二列** |
| 4 | `run_floor_ifirst` | **instr_first** | 120 × {reg_a_para} | 120 | 2×2 第二列的地板 |
| 5 | `run_shuf` | image_first | 60 × {shuf} | 60 | **负控制（任务卡必跑）** |
| 6 | `run_syn` | image_first | 60 × {syn_a, syn_b} | 120 | ρ_syn 配对基线 + 同区域差分场对照 |
| 7–8 | n-boost（可省） | image_first | 剩余 94 源 | 282 | 只提升 n |

**必需批（1–6）= 900 次读出**；一次前向同时导出 **8 个读出变体 × 3 个 special token**，
消融梯子零额外 GPU 成本。

## 冒烟与吞吐

| 阶段 | 结果 |
|---|---|
| 机械冒烟（2 源） | 24 层 pre/post 捕获、image span 断言、展开长度断言全过；npz 338 KB |
| **读出链路正确性** | **`canon` 变体 vs G1 落盘 npz 逐格对拍：n=40，Pearson 中位 0.99999992，max_abs_diff 中位 0.0039**（logit 值域 ~[−13,1]）⇒ 与 G1 同一条链路 |
| 数值冒烟（n=21 源） | 8 变体全部产出合理数字（见 NOTES §5）；`aas` 的 ρ_region_opp 显著低于 `canon`，但**地板同步走低**——这正是必须有匹配地板才能判读的情形 |
| 吞吐 | 4 worker 聚合 **≈ 160 次读出/小时**（单次 t_gen ≈ 75 s，host load ~120/48 核） |

## 时间线

- 16:0x 首跑成功；修 bug 1（`np.savez_compressed` 自动补 `.npz` 致原子写路径错）。
- 16:1x **性能返工**：t_gen 139.6 s/样本 → 定位 CPU 线程超订（进程内 121 线程 vs load 141/48 核）
  → `OMP/MKL=4` + `torch threads=3` 降到 53 s → 4 路分片并行，聚合 ≈160/h。
- 16:2x 提交第 2 版计划（1176 次读出）。
- 16:4x **收到主 agent 关于 RO-2 的追加消息** → 加入 `instr_first` prompt 变体，
  重排为第 3 版计划（2×2 优先）。实测 prompt 顺序两种写法的 token 布局：
  `image_first` img_span=(14,270)、指令在图之后；`instr_first` img_span=(64,320)、指令在图之前。
  **前者与 RO-2 报的 span 逐位一致。**
- 16:47 第 3 版计划起跑（4 worker），必需批 ETA ≈ 5.3 h。

## 已知偏离与限制（如实记）

- **image_first 列的 n 从 G1 的 214 降到 120**：换取 `instr_first` 列，遵主 agent
  「优先保 2×2」。代价可接受，因为 **canonical × image_first 这一格已被证明与 G1 逐格等价**
  （Pearson 0.9999999），该格可直接引用 G1 全量 n=214 的数字。
- `instr_first` 是**分布外输入**（模型按 image_first 做的 SFT），必须同时看 fallback 率与
  gen_len；若模型在该条件下不再产出 retouch token，比较会被「模型崩了」混淆——已逐样本记录。
- 批 7–8（n-boost）能否跑完取决于机器负载，跑不完如实报 n。

## 追加时间线（第 3→4 版计划）

- 17:07 **再次重排批次**（第 4 版）：把 `run_region_ifirst` 提到 `run_floor` **之前**。
  理由：2×2 的**主判据**（ρ_region_opp + AUC_target，只需 region 批）信息量最高，
  匹配地板是解读辅助；万一必须提前收尾，先拿到完整 2×2 主判据比拿到左列地板更有价值。
  最终批次顺序：`region(if) → region(ifirst) → floor(if) → floor(ifirst) → shuf → syn → [n-boost]`。
- 17:1x 资源实测：4 worker 合计 **12.0 GB** GPU（预算 20 GB），host load ~142/48 核。
- **`instr_first` 已单独冒烟验证**（`run_ifirst_smoke/`，2 样本）：
  `fallback=False`、`gen_len=312`（与 image_first 的 311/322 同量级）、
  `post_img_mass=0.210`（image_first 为 0.204）、全部断言通过。
  实测 `img_span`：`instr_first` = **[72,328] / [63,319]**（起点随指令长度浮动 ⇒ 指令确实在图之前）；
  `image_first` = **[14,270]**（恒定）。⇒ 该条件**不是**「模型直接崩了」，比较有效。
- 收尾流水线 `config/finalize.sh` 已写好并逐段验证过：
  全量分析（含 viz）→ 导出 scache arm → 打印判决摘要。

## 已验证的工具链（跑批期间用部分数据逐段验过）

| 环节 | 验证方式 | 结果 |
|---|---|---|
| 读出正确性 | `canon` vs G1 落盘 npz 逐格 | **Pearson 0.99999992**，max_abs_diff 0.0039 |
| 判据口径 | `canon` 的 AUC_target 对比 RO-9 `canonical_L8_15` | 我方 0.507（部分批） vs RO-9 0.5067（n=214）——**口径一致** |
| viz | 部分数据出图 | 2×5 面板（源图/GT/reg_a/reg_b/**地板**/差分），成功与失败两端各出 |
| scache 往返 | 写→读→除以 global_scale 复原 | 最大相对误差中位 **2.1e-4**；`meta.norm.domain` 存在、`per_image_normalization=false` |

## 终态（2026-08-03 19:2x）

- **全部 8 个批次自然跑完，1182 次读出，0 次失败**，worker 全部自然退出（`pgrep` 空）。
  `region 428/428（=G1 全量 214 源）· floor 214/214 · region_ifirst 240 · floor_ifirst 120 ·
  syn 120 · shuf 60`。n-boost 批（7、8）也跑完了，故 **image_first 列的 n 与 G1 完全一致（214）**，
  此前 STATUS 里「n 降到 120」的偏离**已不再成立**。
- 交付齐备：`REPORT.md`（含强制前三行）/ `metrics.json` / `NOTES.md` / `STATUS.md` /
  `viz/` 6 张（成功与失败两端各 3）/ `config/`（含 seed、批次计划、C_GT 与掩膜缓存、复现脚本）。
- scache arm：`/var/cache/veradata/scache/difflmm-light-L0-23/`（822 条）。
- **复现命令**（纯 CPU，幂等）：`bash config/finalize.sh`
  （= 全量分析 + viz + 导出 arm + 打印判决摘要）。
- **全程未 kill / 未 SIGSTOP 任何不属于本实验的进程**；卡 1 峰值占用 12.0 GB（预算 20 GB）。
