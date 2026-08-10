# Where 头重设计 · 需求与资源盘点（2026-08-10）

> 背景一句话：现方案（MetaCanvas query → cross-attn(H_where, F_pre) → 全局 (w, ρ)）在 159k 全量训练下语义崩塌
> （W01/W02 @53% 训练 mIoU 0.51/0.53、增速 +0.01/500 步、过覆盖 3.7×、居中类输给零参数中心先验 Δ=−0.05 p=1e-4），
> 用户裁定设计失败，W03-W08 容量变体已取消。本文只写**要实现什么**与**手上有什么**，不预设方案。

---

## 一、要实现的功能（需求）

### 1.1 核心功能

**输入**（推理时只有这两样，红线）：
- `I_in`（原图，短边 512 / 等比 / 32 对齐 / 长边 ≤2048）
- `instruction`（用户编辑指令文本）
- 允许使用 SFT VLM 前向的一切中间产物：`<where>` 推理文本（自回归生成）、任意层 hidden、任意 attention、F_pre 视觉特征
- **禁止**：`I_tar`、GT mask、GT LUT、oracle latent 进入推理输入

**输出**：
- 软掩膜 `m_pred(p) ∈ [0,1]`，交付分辨率 = 原图尺寸，指定「这条指令要编辑的语义区域」
- （供 Stage-What 消费）掩膜之外至少还需：`F_roi/F_bg`（m 加权的 F_pre 池化，现成公式）；可选一个全局空间 latent（原 z_where 位）

### 1.2 质量目标（预注册，V_where generated-context 主榜）

| 指标 | 目标 | 现方案实测（参照系） |
|---|---|---|
| local soft-IoU (min/max) 中位 | ≥ 0.75（协议 §5.6 gate） | 0.51/0.53 @53% 训练 |
| 相对 oracle 上界（0.97）比值 | ≥ 85% | 54-56% |
| p10 | ≥ 0.55 | 0.20-0.23 |
| 指令条件性 | 同图不同指令产生不同区域（配对 Δ>0）；shuffle 指令 IoU 显著降 | 未达（过覆盖=铺全局） |
| global 样本（全图编辑） | soft-IoU ≥ 0.98 | 可达（全 1 mask 平凡解） |

日常监控主读数已简化为一个：**soft_iou_vs_oracle**（预测 vs oracle mask 的 min/max soft-IoU，直接值）。
完整板（七 context/负控制/中心先验/分层）只在选型时离线跑一次。

### 1.3 训练约束

- 监督信号：GT mask（`.cgt.png` 软边掩膜，local train 75,544 条）；oracle latents（可选辅助）
- VLM 冻结边界：现协议冻结整个 SFT VLM。**若新方案需微调 VLM 任何部分（如 LISA 式 seg-token），须先经用户批准**——这是唯一需要用户拍板的边界
- 单卡一臂、effective batch 32、1 epoch 量级；bf16；无人值守可跑（接入现有队列）
- 评测按 WEVAL 分层工具离线归因（面积/位置/边界复杂度/软边等六维 + 长尾机制归因）

### 1.4 失败教训（新方案必须回应的三个问题）

1. **间接映射学不出来**：文本 hidden →（query 读出）→ 71 维全局线性方向 w 的映射，在有 oracle 蒸馏监督
   （L_s/L_curve/L_dir）的情况下仍然崩塌——监督不缺，是表征/机制不匹配。新方案要说明它的「文本→空间」通路为何更直接。
2. **过覆盖退化**：长尾 ~70% 主因是预测面积 3.7× 于 GT（铺成近全局掩膜，大/居中区域靠面积吃分）。
   新方案要说明什么机制阻止这个平凡解。
3. **历史坍缩**：直接空间 logits（旧 MCQ 路线）也塌过。「换个头」不等于解决——需要能定位指令条件信号
   实际存在于模型的哪个部位（历史证据见 §2.4）。

---

## 二、现有资源（全部可复用）

### 2.1 模型资产

| 资产 | 位置 | 质量证据 |
|---|---|---|
| Base SFT checkpoint-4976 | `/home/bc/data/runs/q3vl_base_sft_20260804/checkpoint-4976`（NFS 备份已同步） | 两段式结构率 64/64 全满分；`<where>` 文本 token-F1 p50=1.0、53% 样本与 GT 逐字一致——**语言侧把区域"说"对了** |
| 同 checkpoint-2488（0.5 epoch） | 同目录 | 备用对照 |
| 原始基模 Qwen3-VL-4B-Instruct | `/home/bc/data/models/Qwen3-VL-4B-Instruct` | |

### 2.2 数据资产（全部 indexed tar shards，读走 /mnt/nfs-ro）

| 资产 | 规模 | 位置 |
|---|---|---|
| sft2seg 训练/评测数据 | train 159,215（local 75,544）+ V_where 896 / V_what 897 / T_final 918 / T_lut_unseen 433 | `/mnt/nfs-ro/bc/data/datasets/sft2seg-20260804/`；**本地缓存 27.5 GiB 已预热**（`/home/bc/data/shard_cache`） |
| GT maskviews（低分辨率+交付分辨率） | 五 split 全量 | `where_a-20260805/maskviews/` |
| oracle latents（w*, ρ*, s*, r*(z) 257 点, std(s*), cband_normalization） | 五 split 全量（train 75,544×2 readout） | `where_a-20260805/oracle/BA-3-Joint/s5/` |
| genctx two_segment（`<where>`+`<color>` 生成 token ids） | train+V_where+V_what 全量，零截断零格式失败 | `where_b-20260805/genwhere/` |
| genctx forced_color | train+V_what | 同上 `-forced_color` |
| V_where 掩膜几何分类缓存 | 400 条六维几何标签 | `experiments/.../where_b/geometry_*.json` |

### 2.3 代码资产（q3vl 五包，1000+ 测试，8 轮审阅闭环）

| 资产 | 说明 |
|---|---|
| F_pre 提取管线 | `q3vl/where/fpre.py`：merger 前 H/16×W/16×1024，hook 式，真实宽高比 |
| Phi-71 basis + guided upsample | oracle 上界 0.97（Where-A 实证）；`radius=1, eps=1e-2` 终值；s 域契约与越域断言齐全 |
| 两种 readout（Band/CBand12） | 公式经逐符号审阅；CBand12 logsumexp 抗塌陷 |
| 评测器 | `--online-eval quick`（vs-oracle 主读数）/ `--final-eval full`（§5.6 完整板）；per-sample 落盘 |
| WEVAL 分层归因工具 | `q3vl/whereb/analysis/`：六维分类 mIoU + 长尾机制归因（area_mismatch/s_collapse/oracle_ceiling/...） |
| IO 层 | 本地 shard cache + 有序 prefetch + fd 池：数据供给 0.063 s/step（35×），带「不改变任何权重」等价性测试 |
| context 数据流 | teacher/generated 50/50、forced-prefix、七种 context 构造（负控制备用） |
| 训练循环 | 单卡臂、D-20、eval 防御（eval 崩溃不杀训练）、checkpoint 保护 |

### 2.4 历史证据指针（新方案设计的信号地图）

| 证据 | 位置 | 结论（口径限定） |
|---|---|---|
| RO-3：text→image attention 逐层头扫描 | `experiments/_archive/2026-08-10/RO3_layerhead_scan_20260803/REPORT.md` | base 模型上 OOF 融合与 L11H5 差分场存在指令条件空间信号（AUC 口径，限于排序）——**SFT 后（merger+LLM 全调）该信号大概率更强，未测** |
| RO-2：logit lens / emb 对齐 | `.../RO2_logitlens_20260803/REPORT.md` | 进 LLM 前目标词-视觉对齐存在；LLM 前向逐层抹掉词特异性 |
| RO9b：算子修复后的定位上限 | `.../RO9b_readout_fix_20260803/` | 主体定位可达 0.92-0.935；指令可控性不足（base 模型） |
| MCQ 坍缩史 | `.../MCQ_*/` | 直接空间 logits 生成路线塌陷的完整记录 |
| W01/W02 全量失败数据 | `/home/bc/data/runs/where_b/W0{1,2}/` + `experiments/.../where_b/analysis_W0{1,2}_step1500/` | 本次失败的逐样本归因（过覆盖机制） |

### 2.5 算力与运维

- 2×H100 95GB；W01/W02 收尾后全空
- pueue 无人值守队列（`gpu-queue` skill、事件日志、产物 gate、systemd 自愈）；`q status` 一条命令全景
- 训练墙钟参照：IO 修复后单臂（159k 样本 1 epoch，冻结 VLM 前向 + 小头训练）约 3.5-4 h

---

## 三、交付期望

新方案的立项材料应包含：机制一句话、对 §1.4 三个问题的回答、最小验证探针实验（≤半天 GPU，
用 V_where + oracle + SFT checkpoint 快速证伪核心假设）、对协议的改动清单（特别是冻结边界）。
探针通过后再谈全量臂。
