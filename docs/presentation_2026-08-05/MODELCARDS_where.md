# 模型细节卡 · Where 线

生成日期：2026-08-05
用途：mentor 会追问"几层？多宽？参数量？输入输出维度？训了多少步？"。每个数字后面都跟出处（`文件:行` 或 `json 键路径`）。**追不到出处的一律写「缺」，本文件不做估算。**

所有路径以 `/home/bc/VeraRetouch/` 为根，正文里写相对路径。

---

## 0. 四个实验共用的底座（先讲一次，后面不重复）

| 项 | 值 | 出处 |
|---|---|---|
| 基础 VLM | VeraRetouch，`model_type = llava_qwen2`，架构名 `SamanthaForCausalLLM_Unified` | `/home/bc/data/models/VeraRetouch/config.json` |
| 语言侧 | hidden **896**、**24 层**、**14 个 attention head**、**2 个 KV head**（GQA）、FFN 中间维 **4864**、vocab 151,664、max_pos 32768 | 同上 |
| 视觉侧 | vision tower `mobileclip_l_1024`，`mm_hidden_size = 3072`，projector `mlp2x_gelu`，`image_aspect_ratio = pad`，`mm_vision_select_layer = -2`；投影后为 **256 个 image token（16×16）** | 同上；256 的断言在 `experiments/MCQ_e2e_whatwhere_20260803/canvas_model.py:293-294`（`b-a != N_CANVAS` 即报错） |
| 冻结边界 | 整个 VLM 冻结（`p.requires_grad_(False)`），只训 LoRA + query + route + connector + readout | `canvas_model.py:200-201` |
| LoRA | **r=32、alpha=64（=2r）、dropout 0.05、bias=none**，target = `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` 共 7 个模块 | `canvas_model.py:202-208` |
| 层路由 | 读第 **11 / 17 / 23** 层 hidden，3 个可学习标量经 softmax 加权 | `canvas_model.py:159, 221, 283-299` |
| attention 实现 | `attn_impl = "eager"`（红线：禁 FA2/SDPA 回退） | `canvas_model.py:161, 193`；各 run 的 `.config.attn_impl` |
| connector 宽度 | **384**（`dim=384`），1 个 `nn.TransformerEncoderLayer`：d_model 384、**8 heads**、FFN **1536**（=4×384）、dropout 0.1、GELU、**norm_first**；后接 **zero-init** `Linear(384→384)` 残差 + `LayerNorm` | `canvas_model.py:77-100`（`dim_feedforward=4*dim`，`self.zero` 权重/偏置置零） |
| 位置编码 | `Fourier2D`：`n_freq=32` 可学习频率 + 可学习相位 + `Linear(64→dim)`；MetaQuery 臂的 `xy` **恒为 0**（故意不带几何） | `canvas_model.py:54-65, 68-74` |
| 数值防线 | `require_finite` 在多模态 prefix、每层 hidden、connector memory、每个预测张量上逐个断言，首个非有限值即抛错 | `canvas_model.py:27-37` |

---

## 卡 1 · MCQ-E2E / WHERE 轨（`w14`）：MetaCanvas 256 vs MetaQuery 8

**一句话在验什么**：把 E2 那 14 维基底的系数交给 VLM 去读——带二维结构的 MetaCanvas 是否比 8 个一维 query 更能读出「同一张图里指令指向的是 A 区还是 B 区」。

### 模型

| 项 | MetaCanvas 臂 | MetaQuery 臂 | 出处 |
|---|---|---|---|
| query 数与布局 | **256 个**，16×16 网格，宽度 **896**，加可学习 Fourier2D(x,y) | **8 个**，一维，宽度 896，`xy ≡ 0` | `canvas_model.py:175, 217-220`；`grid_xy` 在 n≠256 时返回 `zeros(n,2)`（L68-74） |
| connector | `PatchCanvasConnector(896→384)`，**同格 image token 融合**（`i_in`）打开 | 同一个类，但 `image=None`，不做 patch 融合 | `canvas_model.py:94-100, 290-299` |
| task readout | `W14Readout`：`QueryDecoder(n_query=8, dim=384, layers=2, heads=8)` 的 **2 层 TransformerDecoder**（FFN 1536）→ `Linear(8×384=3072 → 14)`，**零初始化** | 同左（两臂 readout 完全一致） | `canvas_model.py:142-152, 103-116` |
| 可训练参数 / 总参数 | **25,304,785 / 647,724,465 = 3.91%** | **25,082,577 / 647,502,257 = 3.87%** | `runs/w14_canvas/metrics.json → .counts`；`runs/w14_metaquery/metrics.json → .counts` |
| LoRA rank | 32 | 32 | `.config.lora_r` |

### 输入（红线核实：不含 `I_tar`、不含 GT mask）

| 张量 | 形状 | 含义 | 出处 |
|---|---|---|---|
| `image` | `[B, 3, H, W]`（processor 输出，`pad` 比例） | **只有 `I_in`**。`ConstructDataset.__getitem__` 打开的是 `R.sample_paths(...)["in"]` | `data.py:143-145` |
| `instruction` | `list[str]`，B 条 | 模板化指令；L6 另外生成 `rega`/`regb` 两条区域指令 | `data.py:104-109`；`rowlib.instruction_l6_regions` |
| `phi` | `[B, 4096, 13]` | **基底方向特征网格，64×64 空间 × 13 通道**（`x,y,P2x,P2y,xy,L,S,e1..e6`）。**由 `I_in` 单独算出，不含任何 GT** | `ROW_basis_coeff_20260803/make_phi_grid.py:3, 10, 26`；`data.py:129` |
| `mask` / `other_mask` | `[B, 4096]` 各一 | GT 目标区域 / 同图另一候选区。**只进 loss 与评测，不进模型 forward** | `data.py:130-138`；forward 签名 `model(instruction, image, image_size)`（`train.py:359`） |
| `w` | `[B, 14]` | 离线 oracle 系数标签 `[w0, alpha·w_dir]`，`alpha` 截断在 100 | `data.py:139-141`；`--alpha-cap 100` |

### 输出

| 张量 | 形状 | 含义 | 出处 |
|---|---|---|---|
| 网络原始输出 | `[B, 14]` | 归一化坐标下的 14 个系数 | `canvas_model.py:150-152` |
| `w` | `[B, 14]` | `raw × scale`，`scale` = 训练集 84/16 分位半宽的整臂常量（非逐图） | `train.py:150-153`；`data.py:167-179` |
| `s` | `[B, 4096]` | `q = w0 + phi @ w[1:]`，`s = 3·tanh(q/3)` | `train.py:156-159` |
| `m` | `[B, 4096]` | `sigmoid(6·s)`，**全局固定读出，不逐图归一化** | `train.py:159` |

### 训练

| 项 | 值 | 出处 |
|---|---|---|
| step / batch | **2,500 step，batch 8**（两臂相同） | `runs/w14_*/metrics.json → .config.steps / .batch` |
| 优化器 | `AdamW`，`betas=(0.9, 0.95)`；主组 lr **2e-4** / wd **0.05**，LoRA 组 lr **2e-5** / wd **0** | `train.py:332-335`；`.config.lr / .lora_lr / .wd` |
| 调度 | 线性 warmup **200 step**，其后 cosine 衰减到 0.01× | `train.py:336-341`；`.config.warmup` |
| 梯度裁剪 / 精度 | clip 1.0；`autocast(bf16)` | `train.py:395-397, 352` |
| 每样本采样像素数 | **不采样**，直接用完整 64×64 = 4,096 格 `phi` | `train.py:377`（`mask_from_w(wp, b["phi"])` 用整张网格） |
| 损失各项与权重 | step < 200：`reg + 0.5·direction`；step ≥ 200：`bce + 0.25·reg + 0.1·direction`；全程再加 `0.01·route_floor`。`direction`（1−cosine）从 step 50 起启用 | `train.py:369-390`；`--direction-start 50` |
| checkpoint 选择规则 | **不用 val loss**。`score = −(inner 池 AUC_target_L6_p50)`，在完整 **410 条 source-disjoint inner 池**上算（禁止前缀截断，否则只剩一组 L6 A/B） | `train.py:417-421`；`NOTES.md:107-108` |
| seed / 样本顺序 | `seed = 20260803`，两臂共用同一个显式 sampler generator（`seed+17`），保证样本顺序逐条配对 | `.config.seed`；`REPORT.md:42-44` |
| wall | canvas 1,616.9 s / metaquery 1,503.3 s | `metrics.json → .wall_sec` |

### 数据

| 项 | 值 | 出处 |
|---|---|---|
| build 代号 | **D-CONSTRUCT**（INF-2 合成八级阶梯 L0–L7；几何/语义/全局三族 × 幅度/羽化/面积档；语义掩膜来自 SAM3 subject cache） | `docs/DATA_ASSIGNMENT_2026-08-02.md:46` |
| 物理位置 | `experiments/tooling-wave1/T4_construct/sanity/{train,val}/manifest.jsonl` | `ROW_basis_coeff_20260803/rowlib.py:63, 279-282` |
| 原始行数 | train **1,600**（L0–L7 各 200）、val **192**（各 24） | `wc -l` 两个 manifest；`jq .level \| sort \| uniq -c` |
| 训练/评测池（L6 展开 A/B 后） | train **1,590**、inner **410**、val **240** | `runs/w14_*/metrics.json → .data` |
| split 权威 | train 用 **S-train** 源、val 用 **S-val** 源分别生成两套；两池 source-disjoint | `DATA_ASSIGNMENT:46`；`NOTES.md:73` |
| val 中的 L6 A/B | 24 条 `rega` + 24 条 `regb`，其中 46 条条件 AUC 有定义 | `REPORT.md:107` |

### 判据 vs 实测

| 预注册判据 | 门 | MetaCanvas | MetaQuery | 判定 | 出处 |
|---|---|---:|---:|---|---|
| L6 `AUC_target`（source-disjoint val） | **≥ 0.65** | **0.5620** | **0.6035** | **两臂皆 FAIL** | 门：`NOTES.md:106`；值：`metrics.json → .final.val.auc_target_L6_p50` |
| 同上（inner 池，选 checkpoint 用） | ≥ 0.65 | 0.6130 | 0.6486 | 两臂皆 FAIL | `.final.inner.auc_target_L6_p50` |
| 配对 instruction-shuffle IoU 降幅 | > 机器零 | 0.00333 | 0.00445 | PASS（幅度极小） | `.final.val.delta_shuffle_iou_p50` |
| query 未复制/未坍缩（effective rank） | 不得 ≈1 | **102.21 / 256** | **3.88 / 8** | PASS | `.final.val.query_effective_rank`；门：`NOTES.md:110-112` |
| 「MetaCanvas 在核心 where 指标上优于 MetaQuery」 | — | — | — | **FAIL（反了）** | `REPORT.md:119` |
| 普通 mask AUC p50（描述性） | 无门 | 0.8908 | 0.9016 | — | `.final.val.auc_p50` |
| soft-IoU p50（描述性） | 无门 | 0.3551 | 0.3450 | — | `.final.val.soft_iou_p50` |
| w14 系数 RMSE p50（描述性） | 无门 | 7.5886 | 7.5890 | — | `.final.val.w_rmse_p50` |

**这张卡的落点**：普通 AUC 0.89–0.90 看着很好，条件 `AUC_target` 只有 0.56–0.60。两者的落差就是「找到主体」和「听懂指令指的是哪个区」的落差（`REPORT.md:109-111`）。

**缺**：本实验没有 `config/` 目录、`run.json` 与 `job.marker` 里都没有 git commit → **实验时的 git commit = 缺**。

---

## 卡 2 · MCQ-L 全量 Local L1–L6：**dense 空间头**（本卡只覆盖空间头/dense 部分）

**一句话在验什么**：让 MetaCanvas 直接吐一张 16×16 的 mask logits 图（"直接预测像素"），看能不能同时给出可用的空间场与颜色参数。

**结论已定档：该路线被正式暂停。** 原话证据：

```
matrix_state.json → .pause_reason = "Replaced dense spatial prediction with basis-parameter experiments"
matrix_state.json → .paused_at    = 1785836182
matrix_state.json → .status       = "paused"
```

（文件：`experiments/MCQ_full_local_l1l6_20260804/matrix_state.json`）

方案文档里的对应判词，逐字：**「旧 MCQ 直接空间 logits 路线出现明显坍缩和边缘退化，因此本方案将 MetaCanvas 从"空间图生成器"降为"全局 basis/readout 参数生成器"」**（`docs/METACANVAS_WHERE_WHAT_FINAL_EXPERIMENT_PROTOCOL_2026-08-04.md:69`）。

### 模型

| 项 | 值 | 出处 |
|---|---|---|
| query 数与布局 | **256 个 MetaCanvas query**，16×16，宽 896，带 Fourier2D 坐标（`spatial_metaquery8` 臂例外，见下） | `NOTES.md:28`；`local_model.py:363-364` |
| connector | 与卡 0 同（384 宽、8 head、FFN 1536、zero-init 残差） | `NOTES.md:29`；`canvas_model.py:77-100` |
| **dense 空间头** | `LayerNorm(384)` → reshape 成 `[B,384,16,16]` → `Conv2d(384→128, k3)` + GELU → `Conv2d(128→128, k3)` + GELU → **`Conv2d(128→1, k1)` 零初始化**，输出 `[B,16,16]` logits | `local_model.py:113-131` |
| 参数头（4D 高斯） | `QueryDecoder(49 query, dim 384, **3 层** TransformerDecoder, 8 heads, FFN 1536)` → 48 个 primitive 各出 **27** 个 raw 值 + 1 个 global query 出 **12** 个 → 共 **1,308** 个 raw 值 → `ParamHead4D` | `local_model.py:48-88`（`raw_dim = 27 if gaussian4d`）；`canvas_model.py:103-116` |
| 可训练 / 总参数 | `config_a`、`config_b`：**28,250,091 / 650,669,771 = 4.34%**；`renderer_gaussian3d_alpha`：28,248,551 / 650,668,231 = 4.34%；`renderer_lut4d`：33,892,695 / 656,312,375 = 5.16%；`spatial_metaquery8`：**29,999,978 / 652,419,658 = 4.60%** | 各 `runs/*/metrics.json → .counts`（metaquery8 取 `runs/spatial_metaquery8/run.json → .counts`） |
| `spatial_metaquery8` 臂的空间头 | **8 个无坐标 MetaQuery** → `LayerNorm(8×384=3072)` → `Linear(3072→768)` + GELU → **零初始化 `Linear(768→256)`** → reshape `[B,16,16]`。即"用一个 8×384 的全局瓶颈直接解码整张 16×16 场" | `local_model.py:155-171` |
| LoRA rank | 32 | 各 `.config.lora_r` |

### 输入（红线核实）

| 张量 | 形状 | 含义 | 出处 |
|---|---|---|---|
| `image` | processor 输出 | **只有 `I_in`**（`preprocess_source` 读 `source_ref`） | `data.py:135, 145, 151` |
| `source_rgb` | `[B, 128, 128, 3]` uint8 | `I_in` 的 128² 版本，供渲染与 basis 用 | `data.py:14 (RENDER_SIZE=128), 75, 153` |
| `instruction` | `list[str]` | 条件消融由 `condition_inputs` 换文本 / 置零图像，**从不换 target** | `train.py:118-134` |
| `target_rgb` | `[B,128,128,3]` | **只进 loss**，不进 forward | `train.py:576-578`（forward 只收 `condition_images` 与 `source_rgb`） |
| `mask`（`.cgt`） | `[B,128,128]` | GT 区域，只进 loss 与评测 | `data.py:146-149`；`NOTES.md:11` |

`metrics.json` 里有一个 `"target_not_in_condition": true` 字段，但它是**硬编码写死的声明**（`train.py:719`），不是运行时断言；真正的证据是 `condition_inputs` 只传 `batch["image"]`（= source）与文本。

### 输出

| 张量 | 形状 | 含义 | 出处 |
|---|---|---|---|
| `mask_logits` | `[B, 16, 16]` | dense 空间头的原始 logits | `local_model.py:130` |
| 上采样后的 mask 概率 | `[B, 128, 128]` | `sigmoid(bilinear_upsample(16→128))` | `train.py:184-191` |
| renderer `s` 场 | `[B, 128, 128]` | dense 臂无独立 `s_field`，**直接复用 mask 概率当第四维坐标** | `train.py:190-191` |
| 渲染参数 | 48×27 + 12 = **1,308** raw → `ParamHead4D` 出 mu/sigma/协方差/opacity/gate/局部与全局仿射 | `local_model.py:56-58, 71-72` |

**这就是"边缘退化"的机械原因**：mask 的空间自由度只有 16×16 = 256，之后是纯双线性插值。

### 训练

| 项 | 值 | 出处 |
|---|---|---|
| step / batch | **6,000 step，batch 8**（`fit` 一次完整遍历 = 4,671 个 full batch，故预算覆盖全量后再进第二个确定性 epoch） | `.config.steps / .batch`；`NOTES.md:42-44` |
| lr | config_a **2e-4 / LoRA 2e-5**；config_b **1e-4 / LoRA 1e-5**（这是 A/B 的**唯一**差异） | `.config.lr / .lora_lr`；`NOTES.md:35-40` |
| 优化器 | `AdamW`，`betas=(0.9,0.95)`，主组 wd **0.05**，LoRA 组 wd **0** | `train.py:524-528` |
| 调度 | warmup **200 step** → cosine 到 0.01× | `train.py:530-535` |
| 梯度裁剪 / 精度 | clip 1.0；`autocast(bf16)` | `train.py:616-618, 587` |
| **每样本采样像素数** | **1,024**，按区域分层：**inside 341 / boundary 341 / outside 342**（`n//3, n//3, n-2*(n//3)`），`multinomial` 有放回 | `train.py:146-169`；`.config.pixels` |
| 损失各项与权重 | `loss = charb + **0.5**·mask_bce + **0.25**·outside + **0.01**·sparse + **0.001**·prior + **0.01**·route_floor` | `train.py:601-614`；权重默认值 `train.py:451-453` |
| 各项定义 | `charb` = 1,024 个采样像素上的 Charbonnier（`sqrt(Δ²+1e-6)`）；`mask_bce` 在 **16×16** 上算，GT 用 area 插值降采样；`outside` = GT mask ≤0.02 的像素上 \|pred−source\|；`sparse` = 128² 预测 mask 的均值 L1；`prior` = 渲染器 raw 参数平方；`route_floor = relu(0.5·log3 − 路由熵)` | `train.py:596-610` |
| 指令 dropout | **p = 0.15** 换成 `NEUTRAL_INSTRUCTION` | `train.py:581-585`；`.config.condition_dropout` |
| checkpoint 选择规则 | **不用 val loss**。`selection_score`（越小越好）= `ΔE00_p50 + 0.25·ΔE00_p90 + 50·outside_leakage − 0.01·(PSNR_in+PSNR_boundary+PSNR_out) − 0.25·AUC − 0.05·(Δconst+Δshuffle)`；`var_ratio < 0.05` 罚 +100；`Δconst`/`Δshuffle` 低于 **3 dB** 各罚 `10·(3−x)`。每 500 step 在 **384 条 select-core** 上评，终局在完整 **4,225 条 select** 上定 | `train.py:256-271, 350-358`；`NOTES.md:46-51` |
| 选中的 step | config_a **5000**、config_b **5500**、gaussian3d_alpha 5000、lut4d 5500 | 各 `metrics.json → .selected_step` |

### 数据

| 项 | 值 | 出处 |
|---|---|---|
| build 代号 | `prod-l1-local17k-20260731` / `l2`/`l3` 同日、`prod-l4/l5/l6-local17k-20260801` 六个 durable SFT build | `build_manifest.py:29-37`；manifest `.builds` |
| 派生 manifest | `metacanvas-local-l1l6-v2-20260804`（NFS durable，indexed tar），`records.jsonl` sha256 `3ad571a9…9ddb4852` | `/mnt/nfs/bc/data/datasets/derived/metacanvas-local-l1l6-v2-20260804/manifest.json → .records.sha256` |
| 过滤 | `winner_confidence = normal`、`task_type = local`、levels L1–L6 | manifest `.filters` |
| 规模 | records **46,424**、sources **15,010**；分池 **fit 37,370 / select 4,225 / test 2,381 / val 2,448**；分层 L1 7,943 / L2 7,259 / L3 7,823 / L4 7,776 / L5 7,762 / L6 7,861 | manifest `.counts` |
| split 权威 | `tools/data_splits/splits.sqlite3`，sha256 `ba99b236…58b204f5`；`fit/select` 用 `sha256("mcq-local-select-v1:"+source_id) mod 10 == 0` 冻结，且脚本显式检查 source 不跨池 | manifest `.split_authority` / `.selection_rule`；`build_manifest.py:64-68, 369-373` |
| 源图预处理 | EXIF transpose → LANCZOS 短边 1024 | manifest `.source_preprocess` |

### 判据 vs 实测（完整 4,225 条 select）

| 判据 | 门 | config_a（dense） | config_b | 3D+alpha | lut4d | metaquery8 | 出处 |
|---|---|---:|---:|---:|---:|---:|---|
| `delta_shuffle` 条件因果门 | **≥ 3 dB** | **1.901** | 1.798 | 1.857 | 1.790 | 缺 | `.final.select.delta_shuffle_db_p50`；门在 `train.py:267-270`；判词 `AB_VISUALIZATION.md:28-29` |
| `delta_const` 空间收益门 | **≥ 3 dB** | **0.780** | 0.817 | 0.428 | 1.039 | 缺 | `.final.select.delta_const_db_p50` |
| 颜色非坍缩 `var_ratio` | **≥ 0.05** | 0.317 | 0.275 | 0.205 | 0.277 | **0.00087 @step500 → 0.117 @step2500** | `.final.select.var_ratio`；metaquery8 取 `metrics_partial.json → .history[].eval.var_ratio` |
| selection score（越小越好） | 无绝对门 | **37.287** | 37.967 | 41.620 | **35.619** | 48.976 @step2500 | `.final.select.selection_score` |
| mask AUC p50 | **缺**（无预注册门） | 0.9469 | 0.9385 | 0.9463 | 0.9461 | 0.8208 @step2500 | `.final.select.auc_cgt_p50` |
| soft-IoU p50 | **缺**（明文只作描述统计） | 0.6049 | 0.5823 | 0.6099 | 0.6020 | 0.4881 @step2500 | `.final.select.soft_iou_p50`；`NOTES.md:49` |
| ΔE00 p50 | **缺** | 3.560 | 3.557 | 3.624 | 3.456 | 3.944 @step2500 | `.final.select.de00_p50` |
| mask 外泄 | **缺** | 0.00192 | 0.00173 | **0.00533** | 0.00104 | — | `.final.select.outside_leakage_p50` |

`spatial_metaquery8` 在 **2,650 / 6,000 step** 被 `SIGINT` 手动中断（日志末尾是 `KeyboardInterrupt`，`run.json → .status = "interrupted"`，`matrix_state.json → .current_arms[1].signal = "SIGINT"`），因此**没有终局 `metrics.json`**，上表所有 metaquery8 数字都是 step 2500 的周期评估值（`metrics_partial.json → .history[4]`）。

**必须诚实说明的一点**：dense 头在分数上并不输给 basis（AUC 0.947 vs 0.915，soft-IoU 0.605 vs 0.528）。停掉它的理由是结构性的——空间自由度锁死在 16×16、`metaquery8` 变体一上来就近乎坍缩（`var_ratio` 8.7e-4）——而不是"跑分更差"。

**缺**：
- 本实验没有符合协议前三行格式的 `REPORT.md`（只有 `NOTES.md` / `AB_VISUALIZATION.md` / `RENDERER_GAUSSIAN3D_ALPHA_VISUALIZATION.md`）→ **预注册判据的完整清单 = 缺**；表中的门只有 selection_score 内嵌的 3 dB / 0.05 两条。
- `config_a` / `config_b` 的 `metrics.json → .config` 里 `spatial_readout / param_readout / renderer / interaction / condition_mode` 全为 `null`（跑它们时 CLI 还没有这些参数）。结构按 `NOTES.md:30-31` 记载为「16×16 convolutional mask logits + 48 个 4D anchored Gaussian」→ **这两臂的结构配置快照 = 缺（靠 NOTES 文字追认）**。
- 无 `config/` 目录、`job.marker` 无 git commit → **git commit = 缺**。

---

## 卡 3 · E2：掩膜基底离线拟合（MB-1 + 缺口 A/B 补件 + wave 3）

**一句话在验什么**：14 维单轴基底 + 沿 s 的读出，**理论上限**能把多少种掩膜画出来；以及"薄环只有平顶带通画得出"这件事是不是靠一个自由挑选的读出凑的。

**这里没有神经网络。** 每张掩膜独立做一次 L-BFGS 拟合，报的是**表达力上限（oracle）**，不是模型分数。

### 模型（=参数化）

| 项 | 值 | 出处 |
|---|---|---|
| 基底 | `φ = [1] ⊕ [x, y, P₂(x), P₂(y), x·y]（Legendre，中心化 / 除短边） ⊕ [L, S] ⊕ [e₁..e₆]` = **1 常数 + 13 方向 = 14 维** | `e2lib.py:4-11, 46-50`；坐标定义 `e2lib.py:78-100` |
| 语义通道 `e₁..e₆` | CLIP **ViT-L/14-336** MaskCLIP 式稠密特征（输入 672²、网格 48²）→ 6 个文本锚 → guided filter（r=32, eps=1e-3）→ 逐图标准化 → **对 `[1, geo5, L, S]` 残差正交化** → 再标准化 | `config/prep.json → .clip`；`REPORT.md:37-39`；锚点文本 `e2lib.py:53-66` |
| 场 | `q = w₀ + α·(w_dir·φ_dir)`，`‖w_dir‖=1`、`α = softplus ≥ 0`；**`s = 3·tanh(q/3)`** | `e2lib.py:5-7` |
| 六种读出 | `monotone` σ(g·s+b)｜`bandpass` 双阈值 logistic 平顶带（μ 自由、k∈(1,40)、h∈(0.02,2.5)）｜`gauss` 幅值恒 1 单高斯（σ∈(0.05,3.0)）｜**`cband_norm` = 渲染器真参数化**：`ΣᵢcᵢoᵢN(s;μᵢ,σᵢ) / Σⱼoⱼ N(s;μⱼ,σⱼ)`，**M=12、μ=linspace(−3,3,12) 固定为 buffer 不进优化器、σᵢ∈[0.025,0.30] 有界 sigmoid、oᵢ/cᵢ∈(0,1)**｜`cgauss` 单基元（μ 锁网格）｜`cband_unnorm` 去分母 | `e2lib.py:13-29, 74-77`；`CONSTRAINED_AXIS = {"M":12,"mu_lo":-3,"mu_hi":3,"sig_lo":0.025,"sig_hi":0.30}`（`e2lib.py:74`） |
| **自由参数量** | 无约束平顶带 **18**（w₀ 1 + w_dir 13 + α 1 + μ/k/h 3）；约束档 **51**（15 + 12σ + 12o + 12c）——**约束档参数是自由档的 3 倍，却吃同样 120 步预算** | `REPORT.md:265-266` |
| 层数 / 宽度 / head 数 | **不适用**（无网络） | — |

### 输入 / 输出

| 项 | 形状 | 含义 | 出处 |
|---|---|---|---|
| 输入掩膜 | **512 × 512** 单通道软掩膜 | 目标 | `config/prep.json → .size` |
| 输入基底 | 拟合用 **stride-2**（256²），评测用 **512² 全分辨率** | 成本控制 | `config/fit.json → .stride`；`REPORT.md:54` |
| 语义族的 GT 来源 | S-val 真实图的 `C_GT`（`.cgt.png`，slot_id `semantic-*`），`winner_confidence=low` 全部剔除，cgt admissibility 0.02–0.85 | 注意：`C_GT` 是逐候选区域掩膜，不是配色真值 | `REPORT.md:33-34`；`CLAUDE.md` 数据纪律节 |
| 输出 | 每张掩膜一组 `(w₀, α, w_dir, 读出参数)` + 全分辨率 soft-IoU | 损失 = `1 − softIoU(min/max)` | `e2lib.py:31-33` |

### "训练"（= 拟合协议）

| 项 | 值 | 出处 |
|---|---|---|
| 优化器 | `torch.optim.LBFGS`，`strong_wolfe`，**`max_iter=120`**（官方默认 `max_eval = 150`）、`history_size=20`、`tolerance_grad=1e-9`、`tolerance_change=1e-11`、**float64** | `REPORT.md:51-53`；`config/fit.json → .max_iter` |
| 起点 | LSQ-informed + **6 个随机** + α≈0 + 质心-径向 ×2；约束档另加 2 个从无约束解构造的热启动 | `e2lib.py:304-355`（`n_random=6`，`extra_starts`）；`REPORT.md:53-54` |
| 拟合总数 | **6,217 次拟合，`n_fit_errors = 0`** | `metrics.json → .n_results / .n_fit_errors` |
| 分片 | 第一回合 4,344（`config/fit.json → .n_tasks`，wall 3,938.4 s，36 workers）+ `_gauss` 1,196 + `_constrained` + `_c2` + `_c6` + `_sweep` | `metrics.json → .shards_loaded`；`config/fit.json` |
| checkpoint 选择 | **不适用**；同一掩膜取多起点中损失最优解，1e-4 内并列时取 α 最小者（常数掩膜的退化性检查） | `e2lib.py:311-315` |
| 红线自查 | σ 一律有界 sigmoid（无裸 exp）；**全程无任何 s 轴平滑正则**；约束档 μ 是 buffer 不进优化器；soft-IoU 是 PLAN 明文规定的**离线拟合**损失而非在线训练目标 | `REPORT.md:55-57` |
| seed / git commit | seed **20260803**，git commit **`0e5d04a71b1b8fc1bd7124d888b2755e67f7a333`** | `config/prep.json` / `config/fit.json → .seed / .git_commit` |

### 数据

| 项 | 值 | 出处 |
|---|---|---|
| split 代号 | **S-val**（`tools/data_splits/splits.sqlite3` T1 旁表，33,652 源中 S-val 747 源）；`eval100-annotqa-20260727` 未触及 | `REPORT.md:32-34`；`config/prep.json → .split` |
| 实际掩膜数 | linear 200 / radial_ell 200 / **ring 200** / wedge 200 / constant 20 / **semantic 176**，合计 **996** | `config/prep.json → .families` |
| ⚠ 口径冲突 | `REPORT.md:13` 写「**1,020** 张 512² 掩膜」，但 `families` 求和是 **996**（差值来自语义族预注册 200、实际只有 176）。**讲稿写 996 或直接写"约 1,000 张"，别写 1,020。** | 两处出处见上 |
| 真实图配对 | 几何掩膜与 **100 张 S-val 真实图** round-robin 配对（干扰通道在场） | `config/prep.json → .n_pool`；`REPORT.md:33-34` |
| 语义五类 | skin 109 / foliage 22 / architecture 18 / sky 13 / water 14 | `config/prep.json → .semantic_class5` |
| 缓存 | `/var/cache/veradata/e2_basis_20260803`（本轮不新增数据） | `REPORT.md:35` |

### 判据 vs 实测

| 预注册判据（出处） | 门 | 实测（全量） | 判定 | metrics.json 键路径 |
|---|---|---:|---|---|
| 线性渐变 · 单调（PLAN L204） | ≥0.97 | **0.9866**（n=200） | PASS | `.criteria.linear_mono_min097.measured_median` |
| 径向/椭圆 · 单调（L205） | ≥0.97 | **0.9857**（n=200） | PASS | `.criteria.radial_ell_mono_min097.measured_median` |
| **环形 · 单调**（L206 左半） | **≤0.40** | **0.34227**（n=200，p10 0.266，**p90 0.4285**） | PASS（**中位数**口径） | `.criteria.ring_mono_max040.measured_median` |
| **环形 · 带通**（L206 右半） | **≥0.90** | **0.97528**（n=200，p10 0.969，min 0.473） | PASS | `.criteria.ring_band_min090_CORE.measured_median` |
| 束状楔形 · 单调（L207） | 预测窗 0.55–0.70 | **0.9422** | **预测被推翻**（未触发 <0.50 的实现 bug 下限） | `.criteria.wedge_mono_expected_055_070`（含 `prediction_refuted: true`） |
| 语义五类 · 单调（L208） | ≥0.85 | **0.8730**（n=176） | PASS | `.criteria.semantic_mono_min085.measured_median` |
| 全局常数 · 退化性（L209） | α<1e−2 且 std<0.01 | α_max **1.0e-3**、pred_std_max **1.41e-5** | PASS | `.criteria.constant_alpha_lt_1e2_std_lt_001` |
| Legendre Gram 条件数（L211） | <10 | **9.000**（单项式对照 17.281） | PASS | `.condition_checks.gram_cond_legendre6` |
| 残差化后块外 Frobenius（L211） | <0.05 | **2.85e-14**（前 1.99） | PASS（构造性满足） | `.condition_checks.offblock_corr_fro_after_max` |

**补件轮（缺口 A/B，先写后跑）——这一组是"平顶不是挑出来的"的证据，讲 P15 时必须带上：**

| 判据 | 门 | 实测 | 判定 | 键路径 |
|---|---|---:|---|---|
| **A-1 约束 s 轴（M=12、μ 固定、σ 有界、跨基元归一化）** | 中位 ≥0.90 **且** 相对无约束平顶带掉幅 ≤0.03 | 14 维 **0.97537**（n=184）/ 几何-only **0.99008**（n=60）；掉幅 **−0.00009 / −0.01348**（负 = 约束档更好）；**掉幅 >0.03 的环占比 0.0%（两档）** | **PASS** | `.criteria_supplement_gapAB.A1_ring_constrained_axis_min090_and_drop_le003` |
| **A-2 响应级平顶保真** | 平顶区最大偏差中位 ≤0.05 且 RMSE 中位 ≤0.05 | 最大偏差 **1.43e-05**（14 维）/ 1.34e-05（几何-only）；RMSE 7.92e-06 / 7.02e-06（各 n=200） | **PASS（超 3 个数量级）** | `.A2_axis_response.A2_response.main14.cband_norm` |
| **A-3 载荷硬化 cᵢ→{0,1}** | 掉幅 ≤0.02 | 0.97537 → **0.97629**（掉幅 **−0.00092**，硬化反而更好），开启基元数中位 **n_on=3** | **PASS** | `.criteria_supplement_gapAB.A3_hard_payload_grouping_drop_le002` |
| **A-4 预算反混淆** | `max_iter` 120→300 的中位提升 **<0.01** | 第二回合 n=8 **+0.01047**；**wave 3 n=40 复跑 +0.012695，bootstrap 95% CI [+0.00995, +0.01642]，95% 的环变好** | **FAIL（两轮均未过门）** | `.criteria_supplement_gapAB.A4_budget_control_gain_lt001`；`metrics_wave3.json → .A4_rerun` |
| **A-4 的预算匹配对照（wave 3 新增，非预注册）** | 无门 | 无约束档 120→300 只涨 **+0.000079**（已收敛）；**同为 300 步时约束档反超自由读出 +0.010848，CI [+0.00968, +0.01479]** | **结论倒向 A-1** | `metrics_wave3.json → .A4_budget_matched.A1_drop_at_300` |
| **B `gauss` 路径可运行** | 修复后 0 报错、六族全报 | **1,196 拟合 / 0 报错**；环形严格单高斯 **0.7781**（14 维）/ 0.7770（几何-only），**不过 0.90 门** | **PASS（且确认"单高斯只到 0.78"）** | `.criteria_supplement_gapAB.B_gauss_path_runs` |

**讲 P15 时的三句准确表述**（都能追到出处）：
1. 薄环上**单调 0.342 vs 平顶带通 0.975**（`.criteria.ring_mono_max040` / `.ring_band_min090_CORE`）。
2. **0.975 不是靠自由读出凑的**：渲染器自己那根受约束的 s 轴（M=12、μ 固定网格、σ∈[0.025,0.30]、跨基元归一化）实测 **0.97537**，掉幅 −0.0001；**预算匹配到 300 步时约束档反超 +0.0108**（`REPORT.md:500-503`）。
3. **平顶是硬需求**：幅值恒 1 的单高斯（μ/σ 自由或锁网格）只有 **0.777–0.778**，不过 0.90 门。所以论文里只能写「**具备平顶的**带通读出」，不能写「任意带通读出」（`REPORT.md:277-278`）。

**附带的机制解释（可作为一句加分话）**：等 σ 时相邻基元的归一化竞争**恰好是一条 logistic**，M=12、σ=0.1347 时 `max|wₖ(s) − 两阈值 logistic 带| = 1.5e-07`；环形实需的形状不变量 β 中位 **8.40**、p10–p90 **6.47–11.28**，而约束轴可达 **[1.65, ∞)**——需求完整落在可达区间内部（`REPORT.md:118-129`，`metrics_beta_M.json`）。

**缺**：A-5 的 **mask 级** σ_max/M 扫描未做（只做了响应级 27 格 × n=30），图里已标注（`REPORT.md:157, 260`）。

---

## 卡 4 · MCQ-basis-WHERE：Geo8 vs VLM14 全量对照

**一句话在验什么**：MetaCanvas 不再吐空间图，只吐一组全局 basis 系数；纯解析几何基（8 维）与加了 VLM 语义基（14 维）谁更好。

### 这个对照有多干净（这是它的核心价值，讲稿要点名）

两臂由同一个 `run_full_basis.py` 起在两张卡上，**唯一变量是 `--spatial-readout`**：

| 对照项 | Geo8 | VLM14 | 是否相同 |
|---|---|---|---|
| 训练脚本 | `MCQ_full_local_l1l6_20260804/train.py` | 同一个文件 | ✅ `run_full_basis.py:15` |
| manifest digest | `3ad571a9…9ddb4852` | 同 | ✅ `run.json → .manifest_digest` |
| **seed** | **20260804** | **20260804** | ✅ `.config.seed` |
| **样本顺序** | `EpochShuffleSampler(len, seed+17, epoch, offset)` 决定，与 seed 一一对应 | 同 | ✅ `train.py:570-573` |
| **step / batch / 每样本像素** | **6000 / 8 / 1024** | 同 | ✅ `.config` |
| lr / LoRA lr / wd / warmup | 2e-4 / 2e-5 / 0.05 / 200 | 同 | ✅ |
| LoRA rank / connector dim / n_gauss / renderer / param_readout | 32 / 384 / 48 / `gaussian4d` / `metaquery` | 同 | ✅ |
| loss 权重（mask 0.5 / outside 0.25 / sparse 0.01 / prior 0.001）、condition_dropout 0.15 | 同 | 同 | ✅ |
| 选中的 step | **6000** | **6000** | ✅ `.selected_step` |
| **可训练参数** | **29,809,266** | **29,819,256** | 差 **9,990** |
| 参数差的完整拆解 | — | — | `semantic_directions` = 896×6 = **5,376**；系数头输出层从 `768×8+8` 变成 `768×14+14`，多 **4,614**。5,376 + 4,614 = **9,990**，一分不差。 |

所以"Geo8 更好"这句话，可归因的差异**只有 basis 本身**。

### 模型

| 项 | 值 | 出处 |
|---|---|---|
| query 数与布局 | **8 个全局 MetaQuery，无二维坐标**（`mode="metaquery"` ⇒ `n_canvas=8`，`xy ≡ 0`）。MetaCanvas 不再输出逐位置值 | `local_model.py:363-364`；`canvas_model.py:175, 68-74`；`EXPERIMENT_PROTOCOL.md:24` |
| connector | 与卡 0 同（384 宽、1 层、8 head、FFN 1536、zero-init 残差）；metaquery 模式下 `image=None`，不做 patch 融合 | 类定义 `canvas_model.py:77-100`；`image = … if self.mode == "canvas" else None` 在 `local_model.py:475-480` |
| 空间头 | `BasisParameterSpatialHead`：`LayerNorm(8×384=3072)` → `Linear(3072→768)` + GELU → **零初始化 `Linear(768→n_basis)`**，`n_basis` = **8**（Geo8）或 **14**（VLM14） | `local_model.py:180-192` |
| 基底构造 | 图像 area-pool 到 **16×16**；`analytic = [1, x, y, P₂(x), P₂(y), x·y, L, S]`（`L=0.2126R+0.7152G+0.0722B`，`S=(max−min)/max`，两者逐图标准化）；VLM14 再加 `semantic = LN(image_tokens) @ W`（`W ∈ R^{896×6}` 可学习），标准化 → **对 analytic 做最小二乘残差正交化**（`gram + 1e-4·I` 求解）→ 再标准化 | `local_model.py:197-245` |
| 语义基的输入取自哪 | **projector 输出的 256 个 image token（进 LLM 之前的 embedding，宽 896）**，不是 LLM hidden | `local_model.py:448-454`（`rows.append(embeds[bi, start:stop])`） |
| 场的合成 | `q = Σᵢ basisᵢ·coeffᵢ`；`latent_s = 3·tanh(q/3)`；`mask_logits = 6·latent_s`；`renderer_s = (latent_s+3)/6` clamp 到 [0,1] | `local_model.py:254-258`；`EXPERIMENT_PROTOCOL.md:26-31` |
| 参数头 | 与卡 2 同：49 个 query 的 3 层 TransformerDecoder → 48×27 + 12 = **1,308** raw → `ParamHead4D` | `local_model.py:48-88` |
| 可训练 / 总参数 | Geo8 **29,809,266 / 652,228,946 = 4.57%**；VLM14 **29,819,256 / 652,238,936 = 4.57%** | `runs/*/metrics.json → .counts` |
| 基底空间头自身的参数量 | Geo8 **2,372,360**（LN 6,144 + Linear 2,360,064 + 输出 6,152）；VLM14 **2,382,350**（+ 输出多 4,614 + `semantic_directions` 5,376） | 由 `local_model.py:188-195` 的层定义逐层算出 |
| LoRA rank | 32 | `.config.lora_r` |

### 输入 / 输出

同卡 2（`I_in` + instruction；`target_rgb` 与 `.cgt` 只进 loss 与评测）。**额外**：基底空间头显式接收 `source_rgb`（`[B,128,128,3]`，`I_in` 的降采版）与 `image_tokens`（`[B,256,896]`）；`source_rgb is None` 时直接报错（`local_model.py:251-252`）。

输出多两个可解释张量：`spatial_coeff{8,14}` = `[B, n_basis]`、`spatial_basis{8,14}` = `[B, 256, n_basis]`（`local_model.py:259-266`）。可视化时保存的形状逐字为 Geo8 `spatial_basis 20×256×8` / `spatial_coeff 20×8`，VLM14 `20×256×14` / `20×14`，`pred_mask/latent_s/renderer_s` 均 `20×128×128`（`VISUALIZATION_REPORT.md:54-58`）。

**⚠ 这一版的边缘分辨率仍然是 16×16**：`geo5` buffer 就注册在 `N_CANVAS=256`（`local_model.py:197-203`），`source_rgb` 也先 area-pool 到 16×16（L210-213），场合成完再双线性上采到 128×128（`train.py:184-196`）。**改进方案（卡 5）就是冲这一点去的。**

### 训练

除 `--spatial-readout` 外与卡 2 完全一致：6,000 step、batch 8、每样本 1,024 个分层像素、AdamW(0.9,0.95)、lr 2e-4 / LoRA 2e-5、wd 0.05 / LoRA 0、warmup 200 + cosine、clip 1.0、bf16、condition dropout 0.15、每 500 step 在 384 条 select-core 上评估并按 `selection_score` 选 checkpoint（非 val loss）。wall：Geo8 **4,719.8 s**、VLM14 **4,637.5 s**（`.wall_sec`）。

### 数据

与卡 2 同一份 manifest：训练池完整 `fit` **37,370** 条，周期评估固定 **384** 条 select-core，终局在完整 **4,225** 条 select 上评（`EXPERIMENT_PROTOCOL.md:11-13`；`.data`）。split 权威 `tools/data_splits/splits.sqlite3`。

### 判据 vs 实测（完整 4,225 条 select，`.final.select.*`）

| 指标 | 预注册门 | **Geo8** | VLM14 | 键路径 |
|---|---|---:|---:|---|
| selection score ↓ | 无绝对门 | **42.4266** | 46.3194 | `.final.select.selection_score`（`metrics.json:118`） |
| AUC（`.cgt`）p50 ↑ | **缺** | **0.91516** | 0.86576 | `.final.select.auc_cgt_p50` |
| soft-IoU p50 ↑ | **缺** | **0.52774** | 0.50301 | `.final.select.soft_iou_p50` |
| ΔE00 p50 / p90 ↓ | **缺** | **3.7434** / **7.4493** | 3.7750 / 7.5855 | `.final.select.de00_p50 / .de00_p90` |
| PSNR in p50 ↑ | **缺** | **22.029** | 21.829 | `.final.select.psnr_in_p50` |
| outside leakage ↓ | **缺** | 0.0025003 | **0.0024542** | `.final.select.outside_leakage_p50` |
| `delta_const` | ≥3 dB（selection_score 内嵌） | 0.8440 | 0.6398 | `.final.select.delta_const_db_p50` |
| `delta_shuffle` | ≥3 dB（同上） | 1.3601 | 1.1845 | `.final.select.delta_shuffle_db_p50` |
| `var_ratio` | ≥0.05（同上） | 0.22559 | 0.20081 | `.final.select.var_ratio` |
| query effective rank（8 个全局 query / 49 个参数 query） | 不得 ≈1 | **2.828 / 1.779** | 2.663 / 1.872 | `.final.select.canvas_query_effective_rank` / `.metaquery_query_effective_rank` |

**结论一句话**（`VISUALIZATION_REPORT.md:14`）：完整 select 上 Geo8 全面更好；VLM14 的语义基确实让预测 mask 出现更多图像相关细节，但细节较噪，尚未转化为更好的 where 指标。

**Geo8 全压在低频几何项**的证据（P16 的系数热图）：20 条固定样本的系数表里大值集中在 `1 / y / P₂(x) / P₂(y)`（如 `P2(x) −0.84`、`y +0.75`、`1 −0.85`），而 **`L`、`S` 两列全部落在 ±0.06 以内**；VLM14 的 `e₁..e₆` 也全在 ±0.12 以内（`viz/compare/coefficients_compare20.png`，共享色标分别为 ±0.827 / ±0.560）。文字判词见 `VISUALIZATION_REPORT.md:35, 48`。

**缺**：
- **预注册的数值判据 = 缺**。`EXPERIMENT_PROTOCOL.md` 只写协议不写门；表中"门"一列只有 `train.py` 的 `selection_score` 内嵌的 3 dB / 0.05 两条，且 `delta_const`/`delta_shuffle` 两臂都远未达 3 dB。
- 本实验**没有符合协议前三行格式的 `REPORT.md`**，也没有 `metrics.json` 之外的 `NOTES.md` / `config/` 目录 → **git commit = 缺**、**假设与待决策清单 = 缺**。
- **失败案例专门可视化 = 缺**（只有 20 条固定样本的通用联图，无 `viz/failure_*`）。

---

## 卡 5 · 改进方案：Stage-Where-A + Stage-Where-B（规格，尚未实现）

**状态**：`DESIGN FROZEN / NOT IMPLEMENTED / NOT STARTED`，v1.0（`docs/METACANVAS_WHERE_WHAT_FINAL_EXPERIMENT_PROTOCOL_2026-08-04.md:3-4`）。以下所有数字都是**规格**，**没有一个是实测值**。

**⚠ 底座换了**：这一版的冻结主干是 **Qwen3-VL**（文档标题 + §4.1），不是前四卡用的 VeraRetouch / Llava-Qwen2。上游 Base SFT 规格另见 `QWEN3_VL_BASE_SFT_SPEC_2026-08-04.md`。

**一句话在验什么**：把高分辨率边缘的来源从「MetaCanvas 的 query 数」换成「merger 前视觉特征 + 解析基 + 一次 guided upsample」，看 mask 能不能过 9 条门。

### 模型

| 项 | 规格 | 出处 |
|---|---|---|
| 冻结边界 | Where-A：冻结整个 SFT VLM，只训 shared basis projector `B` 与离线逐图 oracle latent；Where-B：冻结整个 SFT VLM + 已定档的 `B`，只训 `Q_where`、Where connector、readout/output head。**What 不向 Where 反传，Where 不向 VLM 反传** | §3 表（L139-146） |
| merger 前特征 | `F_pre ∈ R^[B × H/16 × W/16 × 1024]`，取自最后一个 Vision block 之后、主 merger 之前；短边 512、保持比例、32 对齐、长边 ≤2048；**复用同一次视觉前向，不额外跑 DenseCLIP** | §4.1（L154-162） |
| Phi-64 | `geo5 = [x, y, P₂(x), P₂(y), x·y]`；`range = [L, S]`（逐图固定统计口径标准化）；`semantic_low = B(F_pre)`，`B: 1024 → 64`。64 个语义通道**逐图对 `[1, geo5, L, S]` 做最小二乘残差化再零均值/单位方差标准化** | §4.2（L164-172） |
| 方向特征维数 | `phi_dir = [geo5(5), L, S, semantic_1..64] ∈ **R^71**` | §4.2（L177） |
| 场 | `w_dir = normalize(w_raw)`；`alpha = softplus(alpha_raw)`；`s_low = 3·tanh((w0 + alpha·⟨phi_dir, w_dir⟩)/3)`；`w_dir` 符号按"绝对值最大系数为正"定死，消除 `w → −w` 的非辨识性 | §4.2（L178-182） |
| **上采样约束（协议明写的红线）** | 组合完成后**只对标量 `s_low`** 做一次 edge-aware guided upsample 得到 `s(p)`；**「不能先上采样 64 个通道再组合」**（原文逐字）。因此边缘分辨率由 `F_pre` 的 `H/16 × W/16` 网格与原图 guidance 决定，**不由 MetaCanvas 的 8×8 / 16×16 query 数决定** | §4.2（**L183**、L185） |
| readout（两种，做笛卡尔积） | `R-Band`：`b(z) = σ(k(z−μ+h)) − σ(k(z−μ−h))`，`m(z) = π·b + (1−π)(1−b)`，`h>0`、`k∈[1,40]`、`π=sigmoid(π_raw)`｜`R-CBand12`：**E2 已验证的固定中心归一化 Gaussian 竞争**，`μᵢ = linspace(−3,3,12)` 固定、`σᵢ∈[0.025,0.30]`、`oᵢ,cᵢ∈(0,1)` | §4.3（L187-208） |
| Where connector | 宽度固定 **512**，**6 个 pre-norm Transformer block、8 heads、FFN 2048**；每块顺序 `self-attn(Q) → cross-attn(Q, H_where) → cross-attn(Q, F_pre) → FFN`；各 cross-attention residual gate **零初始化**；`H_where` 与 `F_pre` 分别线性投影到 512 | §5.1（L237-246） |
| 四种 MetaCanvas 结构 | `MC8-Joint` **8×8 = 64 query**，同一 attention pool + joint head 出 `w, rho`｜`MC16-Joint` **16×16 = 256 query**｜`MC16-SplitHead` 16×16 共享 canvas，`w` 与 `rho` 各自独立 attention pool 与 head｜`MC16-DualCanvas` **两套独立 16×16 query bank / connector stream**，只共享 frozen VLM | §5.2（L248-257） |
| 主臂数 | 4 结构 × 2 readout = **8 个（W01–W08）**，不做短跑筛选、不中途淘汰。本轮**不跑 MetaQuery baseline** | §5.3（L259-273）、§5.2（L257） |
| 层数 / 宽度 / head 数 / 可训练参数量 | connector 已给（6 层 / 512 / 8 head）；**总可训练参数量 = 缺**（规格未给，需实现后实测） | — |
| basis calibration 四臂 | `BA-0-Fixed`（seeded orthogonal 1024→64，不训练，no-calibration 控制）｜`BA-1-Band`（只由 R-Band oracle 反传）｜`BA-2-CBand12`｜**`BA-3-Joint`（预注册主方案，两种 readout 独立 oracle latent、共享 `B` 联合校准）**。`BA-3-Joint` 是后续 8 个主臂的固定 projector，**不根据 Where-B 结果事后切换** | §4.4（L212-223） |

### 输入 / 输出

| 项 | 规格 | 出处 |
|---|---|---|
| 输入（推理时） | **只有 `I_in` + instruction**。`Q_where` 只读 `<where>...</where>` 全部 token hidden（`H_where`）、merger 前 `F_pre`、真实宽高比下的二维位置编码；**不读 `<color>` hidden，不读 `I_tar`** | §5.1（L231-237）、§0（L35） |
| 明令禁止作为推理输入 | `I_tar`、GT mask、GT LUT、oracle latent——**只在各自阶段作监督或评测真值** | §0（L35）；§4.4（L223）"逐图 oracle 参数只作为监督和 ceiling，推理时不存在" |
| 输出 | **只有全局标量 `w0, w_dir, alpha, rho`，不产生 dense logits** | §5.1（L246） |
| 二维坐标 | 按每张图真实宽高比生成，**不把图像压成 512×512** | §4.1（L160） |

### 训练

| 项 | Where-A | Where-B | 出处 |
|---|---|---|---|
| 优化器 | AdamW | AdamW | §10.2 / §10.3 |
| lr | projector **1.0e-4** | **2.0e-4** | 同上 |
| weight decay | 0.01 | 0.01 | 同上 |
| warmup / 调度 | 0.03 / cosine | 0.03 / cosine | 同上 |
| 梯度裁剪 | 缺 | **1.0** | §10.3（L619） |
| 精度 | `bf16_forward_fp32_oracle_fit` | bf16 | 同上 |
| batch | 缺 | **effective batch 32**（micro-batch 先做显存探测再调 gradient accumulation） | §10.3（L621, L627） |
| epoch / step | **1.0 epoch**（全部合格 local train 样本） | **1.0 epoch** | 同上 |
| eval / save | 缺 | 每 **500** step | §10.3（L623-624） |
| 每样本采样像素数 | **缺**（规格未给） | **缺** | — |
| oracle 拟合 | L-BFGS，**float64、多起点、固定容差**；失败样本进显式 rejection/fit report，**不静默换成零向量** | — | §10.2（L609） |
| context 训练 | — | 训练 batch **固定 50% teacher context + 50% generated context**；generated 缺闭合标签**不得回退到 GT**，按固定 token 边界截取并记录格式失败；**模型选择以 generated context 为主**；每个 checkpoint 必须**分开报告 GT / generated / null / shuffled 四种上下文，不能混成一个均值** | §5.4（L274-289） |
| Where Loss 主项 | — | `L_mask = (1 − softIoU) + **0.25**·balanced_BCE + **0.10**·boundary_F1_3px` | §5.5（L296-299） |
| oracle 辅助项 | — | `L_s = Huber(s_pred/3, s*/3)`；`L_curve = mean_z \|R(z;rho_pred) − r*(z)\|`，`z = linspace(−3,3,257)`；`L_dir = 1 − cos(w_dir_pred, w_dir*)` | §5.5（L301-307） |
| **两段权重 schedule** | — | 前 **30%** steps：`L_mask + 1.00·L_s + 1.00·L_curve + 0.10·L_dir`；后 **70%**：`L_mask + 0.25·L_s + 0.25·L_curve + 0.05·L_dir` | §5.5（L309-317） |
| checkpoint 选择规则 | — | 过 gate 后按 lexicographic 顺序：① generated-context local median soft-IoU ② 3px boundary F1 ③ p10 soft-IoU ④ 参数量/峰值显存/延迟。**无 arm 全过门时仍选 lexicographic best 但必须标记 `WHERE-GATE-FAILED`**，后续 What 结果不得宣称完整方法已成立 | §5.6（L337-344） |

### 数据

| 项 | 规格 | 出处 |
|---|---|---|
| 数据范围 | `global: g1–g4` + `local: l1–l6`；上游 SFT authority train **169,260** / eval **3,320**（实际以校验后的 terminal manifest 为准） | §2.1（L88-100） |
| Where 用的选择集 | **`V_where`**（三个互斥子集 `V_where` / `V_what` / `T_final` 之一，按 `source_image_id`+`lut_id`+build 分组确定性划分，目标 1:1:1）。`V_where` **不得**用来选 What checkpoint | §2.2（L104-110） |
| Where-A 训练数据 | **只用 local l1–l6 的 GT mask**，逐图多起点 L-BFGS 拟合 oracle `w*, rho*`；校准目标是 oracle mask 表达力，**不使用 instruction** | §4.4（L212, L221） |
| 隔离约束 | source image 不跨 train/select/test；同一 LUT identity 不跨 `T_lut_unseen` 与任何训练集；近重复图像与同 recipe 派生物不跨集合；split manifest 与 digest 对所有 arm 固定 | §2.2（L114-120） |
| split 权威文件 | **缺**——协议只写规则，未指明具体 sqlite/manifest 路径（前四卡用的是 `tools/data_splits/splits.sqlite3`） | — |
| 训练池 / 评测池样本量 | **缺**（协议明写"实际数量以 terminal manifest 为准"，manifest 尚未生成） | §2.1（L100） |

### 判据（预注册 gate，9 条，必须**同时**满足；在 `V_where` 的 **generated-context** 主榜上）

| # | 指标 | Gate | 实测 |
|---|---|---:|---|
| 1 | local `.cgt` median soft-IoU | ≥ **0.75** | 未实现 |
| 2 | 相对逐图 oracle 的 soft-IoU | ≥ **85%** | 未实现 |
| 3 | local soft-IoU p10 | ≥ **0.55** | 未实现 |
| 4 | `AUC_target` | ≥ **0.80** | 未实现 |
| 5 | 3px boundary F1 / oracle boundary F1 | ≥ **75%** | 未实现 |
| 6 | instruction shuffle 后 IoU 降幅 | ≥ **0.20** | 未实现 |
| 7 | `std(s_pred) / std(s*)` 中位数 | ≥ **0.60** | 未实现 |
| 8 | global mask soft-IoU | ≥ **0.98** | 未实现 |
| 9 | GT / generated context IoU gap | ≤ **0.05** | 未实现 |

（全部出自 §5.6 的表，L323-335。）

**做个尺度对照，方便讲"这一版把门抬高了多少"**：第 4 条 `AUC_target ≥ 0.80` vs MCQ-E2E 的旧门 0.65（实测 0.56–0.60）；第 6 条 shuffle 降幅 ≥ 0.20 vs MCQ-E2E 实测 **0.0033–0.0045**（差两个数量级）。

---

## 「缺」项汇总

| # | 卡 | 缺什么 |
|---|---|---|
| 1 | 卡 1 MCQ-E2E | 无 `config/`、`run.json`/`job.marker` 无 git commit → 实验时 git commit 缺 |
| 2 | 卡 2 MCQ-L | 无符合协议格式的 `REPORT.md` → 预注册判据完整清单缺（只有 selection_score 内嵌的 3 dB / 0.05 两条门） |
| 3 | 卡 2 MCQ-L | `config_a`/`config_b` 的 `.config` 里 `spatial_readout / param_readout / renderer / interaction / condition_mode` 全为 `null` → 这两臂的结构配置快照缺（靠 `NOTES.md:30-31` 文字追认） |
| 4 | 卡 2 MCQ-L | 无 `config/`、`job.marker` 无 git commit → git commit 缺 |
| 5 | 卡 2 MCQ-L | `spatial_metaquery8` 无终局 `metrics.json`（2650/6000 step 被 SIGINT）→ 该臂的 `delta_const` / `delta_shuffle` 缺 |
| 6 | 卡 3 E2 | A-5 的 **mask 级** σ_max/M 扫描未做（只有响应级 27 格 × n=30） |
| 7 | 卡 4 MCQ-basis | 预注册数值判据缺（`EXPERIMENT_PROTOCOL.md` 只写协议不写门） |
| 8 | 卡 4 MCQ-basis | 无 `REPORT.md`（协议强制三行）、无 `NOTES.md`、无 `config/` → git commit 缺、假设与待决策清单缺 |
| 9 | 卡 4 MCQ-basis | 无 `viz/failure_*` 失败案例专图 |
| 10 | 卡 5 改进方案 | Where-A 的 batch、梯度裁剪、eval/save 间隔缺 |
| 11 | 卡 5 改进方案 | 两阶段的**每样本采样像素数**缺 |
| 12 | 卡 5 改进方案 | 总可训练参数量缺（规格未给，需实现后实测） |
| 13 | 卡 5 改进方案 | split 权威文件路径缺；训练池/评测池样本量缺（terminal manifest 未生成） |

**合计 13 处「缺」。**

---

## 附：一处必须当场纠正的口径冲突

E2 的 `REPORT.md:13` 写「**1,020** 张 512² 掩膜」，但 `config/prep.json → .families` 求和是 **996**（linear 200 + radial_ell 200 + ring 200 + wedge 200 + constant 20 + semantic **176**）。差值来自语义族预注册 200、实际只拿到 176 张合格 `C_GT`。**汇报时写 996，或写"约 1,000 张"，不要写 1,020。**
</content>
