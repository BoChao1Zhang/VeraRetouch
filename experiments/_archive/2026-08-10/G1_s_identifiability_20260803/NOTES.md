# NOTES — W1c：G1 s 可辨识性体检 + RO-9 GL token 读出工具

日期：2026-08-03 ｜ 编码 subagent ｜ 分支 lens-exp

## 一、实施前核实记录（模型加载与 GL token）

### 1. checkpoint

- **路径：`/home/bc/data/models/VeraRetouch`**（2.2 GiB `model.safetensors`，2026-05-17 落盘，即原作者发布权重；本项目自己的 SFT 尚未产出 checkpoint，`/home/bc/data/models` 与 `/mnt/nfs/bc` 均无更新版）。
- config：`model_type=llava_qwen2`，`architectures=["SamanthaForCausalLLM_Unified"]`（仓库类名是 `VeraRetouchForCausalLLM_Unified`，见 `llava/model/VeraRetouch.py`；from_pretrained 按类调用无碍）。
- LLM 骨干 = Qwen2-0.5B 形状：hidden 896 / 24 层 / 14 头（head_dim 64）/ KV 头 2（GQA 7:1）/ vocab 151664 / tie_word_embeddings=true。
- 伴生解码器：`VeraRetouch.Encoder_Renderer`（RO-9 读出用不到；构造函数会建 `retouch_decoder`+`retouch_head`，权重随 safetensors 加载）。
- 推理入口：`inference.py` → `VeraRetouchForCausalLLM_Unified.from_pretrained(model_path, config_add=Box(configs/infer_config.yaml), torch_dtype=bf16)`；tokenizer = `AutoTokenizer(model_path, use_fast=False)`。

### 2. GL token（special token）id 与位置

`added_tokens.json` 实测（与 tokenizer 运行时核验一致）：

| token | id |
|---|---|
| `<retouch_light>` | **151646** |
| `<retouch_color&temp>` | 151647 |
| `<retouch_colormixer>` | 151648 |
| `<Auto/Style/Professional_Retouch_Task>` | 151649/151650/151651 |
| problem/plan 三对 start/end | 151652–151663 |

- 序列位置：模型在**生成的回复里**产出 reasoning（problem/plan 区块）后跟三个 retouch token；`_generate` 取每个 token 首次出现处的 hidden state 喂 `retouch_head`（`llava/model/VeraRetouch.py:380-424`）。
- 词表中**没有**字面 "GL" token；文档（EXPERIMENTS_v3 RO-9 行）称"我们的 special token"。三个 retouch token 中取哪个为 GL 见「待主 agent 决策 D1」。
- 本分支（lens-exp）`VeraRetouch.py` 已有现成插桩：`lens_track_spans`（记录 image token 区间）、`lens_readout_layer`、`lens_recorder`——RO-9 直接复用 `lens_track_spans`。

### 3. 视觉塔与 image token

- `mm_vision_tower=mobileclip_l_1024` = Apple MobileCLIP-L 图像塔 **FastViTHD**，输入 1024²，patch 64 → **16×16 = 256 个 image token**（`llava/model/multimodal_encoder/mobileclip/configs/mobileclip_l.json`：image_cfg.embed_dim 3072 = mm_hidden_size；`mm_projector_type=mlp2x_gelu` → 896）。
- 预处理：数据集先把原图短边缩到 512（`Infer_*_Dataset.resize2_512p`），再过 `CLIPImageProcessor(shortest_edge=1024, crop=1024, mean=0, std=1)`。`image_aspect_ratio=pad`? 实测 config 有 `"image_aspect_ratio": "pad"` 但 `mm_patch_merge_type=flat`，走 flat 分支整图 256 token。
- 序列组装：prompt 里 `<image>` 占位（IMAGE_TOKEN_INDEX=-200）在 `prepare_inputs_labels_for_multimodal` 展开为 256 个 embed；文本位置 → 展开后位置 = 原位置 + 255（单图）。

### 4. eager attention 与 pre-softmax 捕获点（DOSSIER §5.4 已核对本地安装版）

- 运行环境：`.venv-lens/bin/python`（实为 conda env llm_factory py3.12）：**torch 2.10.0+cu128 / transformers 4.57.1 / CUDA 可用**。仓库 `.venv` 无 torch。
- transformers 4.57.1 `models/qwen2/modeling_qwen2.py`：`Qwen2Attention.forward` 中 `attention_interface = eager_attention_forward`（仅当 `config._attn_implementation != "eager"` 才换 SDPA/FA2）。**pre-softmax logit 捕获 = monkeypatch 模块级 `eager_attention_forward`**（`attn_weights = q@k^T*scaling + causal_mask`，softmax 前抄走目标 query 行）。
- 守卫（红线）：加载时 `attn_implementation="eager"`；运行时断言 `config._attn_implementation=="eager"` **且** forward 返回的 attentions 非 None，否则直接 raise（不回退）。
- 显存：只截取 GL/CT/CM 三个 query 行，不存全 L×L。

### 5. G1 数据链路（DATA_ASSIGNMENT §3.1）

- S-val 源：`tools/data_splits/splits.sqlite3` `sources(split='val')` 池计数 unsplash 541 / awards 331 / ppr10k 208 → 各抽 100。
- 指令：journal 归档 `groups.jsonl`（group_id→source_id, source_path）⋈ `sft.jsonl`（group_id→instruction, instruction_short, winner_confidence, task_type）。同义对 = `instruction` vs `instruction_short`；反义 = 方向词取反构造（词表见 `tools/readout/ro9_gl_attention.py` ANTONYM_MAP）。
- 源图路径 = groups.jsonl `source_path`（/home/bc/data/datasets/...，抽样核验存在）。

## 二、假设与已核实清单

- [核实] `<retouch_light>` 等 id 与 added_tokens.json 一致（运行时再断言）。
- [核实] conv 模板 `qwen_2`（system="You are a helpful assistant."），style 模式 prompt 模板取自 `data/infer_dataset.py:141`。
- [核实] `generate(inputs_embeds=...)` 时 `sequences` 只含新生成 token（HF 行为），故 output_ids 下标即生成步。
- [假设] teacher-forced 复算与生成时注意力逐位一致（greedy、同一 KV 数学；bf16 数值差异忽略）。
- [假设] head 间 pre-softmax logit 尺度可比性有限——head-mean 是 v1 聚合；per-head 图 v1
  未落盘（stacks 只存 head-mean，容量考虑），D-5 式逐头选择留给 RO-9 与 RO-1..4 对拍阶段。
- [核实] gflow CLI 可用（gbatch/gqueue/ginfo）；GPU0 空闲（H100 96G）。

## 三、待主 agent 决策（保守默认已采用，未拍板）

| # | 决策点 | 保守默认 |
|---|---|---|
| D1 | "GL token" 指哪一个 special token（词表无字面 GL；候选 light/color&temp/colormixer） | 默认 **`<retouch_light>`**（Global Light 解读）作为 s 判定 token；同一前向顺带导出另两 token 的图作对照（零额外成本），三套 ρ 都报 |
| D2 | G1 判定的 s 聚合口径（RO-9 行没定层号） | canonical = **中层 L8–15 × head-mean 平均**（FastV 中层先验）；逐层 ρ 曲线附报，判据只在 canonical 上判 |
| D3 | 反义指令只对含可取反方向词的源可构造 | 超采样候选源，跳过不可构造者并记录跳过率；不足 300 如实报 |
| D4 | low-confidence 行的 instruction 可否用于 G1（G1 不用赢家渲染，只用指令文本） | 优先 normal，配额不足回退 low（逐条记 confidence） |
| D5 | ρ 的相关系数口径 | 判据用 **Pearson**（16×16 展平、成对有限值），Spearman 附报 |
| D6 | 自由生成未产出 `<retouch_light>` 时无 query 可读 | 在生成序列末尾追加该 token 作 query（记 fallback 标志与占比） |
| D7 | scache 全局根目录未定（selfcheck 用的是实验目录内 s_cache） | 用 **`/var/cache/veradata/scache`**（与其它数据资产同盘），arm=`ro9` |
| D8 | 区域对立对构造口径（补充批）：主批 214 有 local 行的源**没有任何源带 ≥2 个不同 subject**（逐源核实，distinct by description），任务卡首选的"现成对"不存在 | 全部模板构造：reg_a = 真实 subject.description（l 系 region 词），reg_b = 按 `local.region` 取**空间对侧**（left→right side…），center（165/214）→ "the background"（主体 vs 背景，区域必然不同且必然存在——不用 sky/foreground 以免指令命名图中不存在的区域）；方向词 brighten/darken 逐源交替、**对内相同**；模板 "Please {dir} {region}, keeping the rest of the image unchanged." 仿 l 系句式 |
| D9 | 区域批与主批的 GPU 并发（主批还有 ~80 min） | 冒烟 20 源与主批**共卡并发**（GPU0 实测 3% util / 8G/98G，0.5B 模型延迟型负载；两进程共存已被主批×build-agent 先例验证）；全量 214 源保守**链式**排在错位对照批之后（`config/run_regionopp.sh` 等 PID 3232359 退出） |

## 四、D-0 伪影三件套的落地口径（任务卡缩窄为两件）

任务卡明确"token 范数 outlier 剔除 + 插值"。实现：

1. teacher-forced 前向 `output_hidden_states=True`，取**读出层 hidden state** 的 256 个 image-token L2 范数；`norm > median + 3·MAD` → outlier（逐层独立判）。
2. outlier 格子置 NaN 后用 4-邻域均值迭代插值（最多 8 轮，兜底全局均值）。
3. 每样本记 outlier 占比；16px 周期伪影（DVT 检查）在 16×16 网格上无意义（patch=64px），仅对 s 图做 2D 功率谱峰检查作附报，test-time registers 不做（需改模型，超出零训练臂）。

## 五、运行记录

- 02:36（首轮）采样器两处返工：① 旧盘源路径全部失效 → 改经 img 银行回取（复用
  tools/bgr_check BankResolver + read_member；ppr10k 银行 member 后缀是 `.source.png`，
  BankResolver 通用后缀匹配取不到，采样器内自补 `resolve_ppr10k`，catalog 列名 `size` 非
  `length`）。终态：300/300（三池各 100），208 normal + 92 low，214 local + 86 style，
  awards 池 1 源无方向词跳过。源图暂存 `/var/cache/veradata/g1_srcimg_20260803`（320 MB）。
- 2 源机械冒烟（6 样本）：全部自然产出三 retouch token（fallback=0）；24 层捕获、
  image span=256、展开长度断言全过；logit 值域 ~[-13, 1]，无 NaN；D-0 outlier 占比 ~4%。
  单样本 ~11 s（生成 ~10 s @ 512 max_new_tokens greedy + teacher-forced 前向 ~0.1 s）。
- 30 源冒烟批后台跑（90 条，写正式缓存 arm=ro9）；期间用部分产物 dry-run 了 analyze_g1.py
  （修 2 个路径 bug）。初步信号（n=6 源时）：ρ_syn≈0.94 ✅、|ρ_Y|≈0.15 ✅、
  **ρ_opp≈0.94 ❌（反义不分离，syn−opp 差≈0）**；逐层扫描无任何层呈现 syn/opp 分离。
- 全量长任务：gbatch job 2 提交后发现两卡均被 unmanaged 生产进程（prod-g4/l5/l6 build
  agents）占用，exclusive 永远 PD → gcancel，回退 **nohup PID 3228455**（详见 STATUS.md）。

- 区域批冒烟（02:2x–）与吞吐告警：冒烟 20 源与主批共卡起跑后 t_gen 11→45 s/样本；
  排查（nvidia-smi + ps/uptime）定位为 **CPU 过载**（load 65/48 核：prod-l5/l6 build
  agent + E1/A0/run_fit 工作池），非 GPU/共卡问题。SIGSTOP 主批实验证实无效
  （冒烟仍 ~40 s）→ 已 SIGCONT 恢复共跑。详见 STATUS.md 吞吐告警节。

## 七、补充批：区域对立指令对（D-17 修订主判据）

- 背景：G1 主判据修订为**区域对立**（同图两条指令指向不同区域，预期 ρ_region_opp<0.3）；
  方向对立（同区域取反，主批 opp）降为对照。依据 = 冒烟已见 ρ_opp≈ρ_syn 的
  "位置场非方向场"模式（§六），region-where 才是 s 的真检验。
- 取材核实：l 系 sft.jsonl 每行带 `local.region`（center 78%/lower/left/right/…9 值）+
  `local.subject.description`（自然语言区域描述，中位面积 13%）。主批 300 源 ∩ 有
  local 行 = 214（unsplash 75 / awards 76 / ppr10k 63，≥150 ✓ 池近均衡）。
- 构造：见待决策 D8（模板 + 空间对侧/背景 complement + 方向词交替）。产出
  `config/g1_region_opp.json`（214 源 × {reg_a, reg_b}，brighten 107 / darken 107，
  normal 152 / low 62）+ `g1_region_opp_report.json`。图复用主批暂存，无需回银行。
- 分析：analyze_g1.py 增 `--region-json`（默认自动读）；ρ_region_opp 一律 valid-mask
  口径（pad 黑边剔除，同主批修正）；gate 切换为 syn/region_opp/y 三判据
  （`gate_keys` 字段记录口径），rho_opp 保留为对照列；viz 增
  region_success_* / region_failure_* 两类（源图 + s(reg_a) + s(reg_b) + luma16）。

## 六、预注册的失败模式解读（分析前写定，防事后合理化）

- ρ_syn 高 + ρ_opp 高 + |ρ_Y| 低 → **s 是"位置场"非"方向场"**：注意力聚焦指令命名的区域，
  但对方向词（brighten/darken）不敏感。反义构造只翻方向词、区域名词不变，故此模式下
  ρ_opp≈ρ_syn 是结构性结果。这不满足 G1 判据（判 FAIL），但与"死刑"（ρ_syn<0.5 或亮度马甲）
  不同：位置信息仍可用，方向信息需另取（如 payload/w 通路），对应 PLAN_v2 §3 gate 树的
  "换读出（D-2 logit lens 绝对刻度 → D-6 探针）"支线，以及「VLM 负责 where、参数通路负责
  how much」的降级叙事。
- ρ_syn 与 ρ_opp 同高还需排除一个更糟的解释：**s 根本不依赖指令**（纯图像驱动）。
  区分测试（全量分析补做）：跨图 shuffle 对照——corr(s(img_i, instr_i), s(img_i, instr_j))
  其中 instr_j 来自别的源（区域名词不同）。若跨指令相关仍 ≈ ρ_syn，则 s 连位置条件性都没有，
  RO-9 直接降级 analysis（EXPERIMENTS_v3 RO-9 行失败判据）。
- **对齐修正（重要）**：核对 `llava/mm_utils.py:186 process_images_` 发现预处理是
  `expand2square` **黑边 pad 到方形**（image_mean=0 → 填黑），不是 center crop——
  初版 luma16 用了 crop 口径（错位），且黑边格是全指令共享的常量带，会同时抬高
  ρ_syn/ρ_opp/ρ_Y。修正：工具输出 `valid16`（格子真实图像覆盖 ≥0.5），分析器一律从源图
  重算 luma+valid（兼容旧批 stacks），**所有 ρ 只在有效格上计算**（allcells 版留作敏感性附报）。
  实测影响（n=15）：ρ_syn 0.940→0.872，ρ_opp 0.947→0.883（差值仍 ≈0，判定方向不变）；
  valid_frac 均值 0.70。

## 八、D-21 交付形态修复追记（2026-08-03，修复 agent；主批 PID 3228455 未动）

依据 REVIEW-result.md「交付形态需修」三条 + DECISIONS D-21：

1. **analyze_g1.py 判据同步 D-17**（DATA_ASSIGNMENT §3.1 G1 行 2026-08-03 修订）：
   方向对立（rho_opp）改为 `verdict="CONTROL"` 只报不判、不触发 FAIL、不进 gate；
   `gate_keys` 恒为 syn/region_opp/y；ρ_region_opp 缺席（区域对立批未落地）时
   `verdict.rho_region_opp="PENDING"`、`gate="PENDING"`（仅 syn/Y 死刑判据可判 DEAD）。
   CRITERIA 删除 `rho_opp_pass` 阈值（留 `rho_opp_note` 说明对照属性）。
   顺带清 Pyright：pyrightconfig.json 补 `experiments/G1_s_identifiability_20260803`
   executionEnvironment（extraPaths 镜像运行时 sys.path.insert：tools/readout+tools/scache）；
   spearmanr `.statistic` 走 getattr 绕 scipy 私有存根类型；两处 viz `im` possibly-unbound
   改 `im=None` 初始化 + colorbar 判空。pyright 0 errors。
2. **删除陈旧 viz**：`viz/success_src_1b2e4569e770f504.png`（02:05 valid-mask 修正前
   dry-run 产物，标题 ρ=0.93/0.91 为 allcells 口径，与判定口径不一致，REVIEW §一点名）。
   现存 8 张 02:07 图为 valid-mask 口径 n=15 快照；整目录重刷仍按 REVIEW 留待全量分析
   （本次中间重算以 `--viz-n 0` 跑，不产中间态混杂图）。
3. **metrics.json 中间版重算**（02:50，仅现有 npz：run_smoke30 + run_full 已落盘部分）：
   **n=41，gate=PENDING** ✓；ρ_syn 0.865（PASS 方向）、|ρ_Y| 0.203（PASS 方向）、
   ρ_opp 0.893（CONTROL 只报）、ρ_region_opp 缺席（PENDING）；fallback 9.8%。
4. **REPORT.md 判定表**换 D-17 口径四行（同义/亮度/方向对照/区域对立-PENDING），
   注明 n=15→当前中间快照属性。

**D-22 TODO（区域对立批落地后，随补充批分析一并做）**：补两个度量回答 ⚑U5
「哪个 token 最像主体感知」——
(a) **跨 token 两两空间 ρ**：同一 (img, instr) 下 light/color&temp/colormixer 三张 s 图
    两两 Pearson（valid-mask 口径），落 metrics（REPORT「三 token 图高度同质」需此数字背书）；
(b) **三 token 对 SAM3 掩膜 AUC**：各 token s 图对 D-SFT-L(S-val) SAM3 主体掩膜
    （及 D-CONSTRUCT L1 GT）的 ROC-AUC，对齐 DATA_ASSIGNMENT L82 统一评分集口径。
数据已全在 scache（ro9 npz 存 3 token 逐层 stack），**零额外前向**，纯分析侧扩展。
→ **已落地，见 §九**。

## 九、终版重聚合（2026-08-03 下午，重聚合 agent）

### 9.1 并批口径

- `analyze_g1.py --run-dirs run_smoke30 run_full run_regsmoke20 run_regfull run_shufctrl`
  → 主批 **300/300 源全到齐**（run_smoke30 49 npz + run_full 851 npz = 900 = 300×3），
  区域批 **214/214**（run_regsmoke20 8 + run_regfull 420 = 428 = 214×2），错位对照 **60/60**。
  三批跑批进程均已自然退出（`ps` 空、GPU0 0 MiB），无未落盘样本。
- 之前顶层 `metrics.json` 是 02:50 的 n=41 中间版（`shuffle_control`/`region_opposition`
  为 null）；本次全量覆写，`gate` 由 PENDING → **FAIL**。

### 9.2 D-22(b) 掩膜数据源核实（本轮唯一的外部事实依赖）

任务卡写「SAM3/C_GT 掩膜」，落地前逐条核实：

| 事实 | 核实方式与结果 |
|---|---|
| SAM3 主体掩膜在哪 | DATA_ASSIGNMENT §2 D-MASKBANK 写「SAM3 subject cache」。journal `groups.jsonl` 里的 `cgt_path` 指向 **`/mnt/ramstage/...`（已不存在）**，不可用。改走 veradata 银行索引 `/var/cache/veradata/global.sqlite3`：`groups` 表有 `cache/sam3`（133805 样本，逐短语掩膜）与 **`cache/subject`（56778 样本，逐源单主体掩膜 `.subject.png` + subject.description/area）**，实体在 `/mnt/nfs/bc/data/datasets/cache/subject/shards/*.tar` |
| 键怎么对上 G1 img_id | `sam3_<hex>` / `subject_<hex>` 的 hex **不是** `src_<hex>` 的 hex（直接拼键 0/300 命中）。正确键 = sample meta 的 **`asset_id`**，实测 **300/300 命中**（含 ppr10k_*_a 形式） |
| 掩膜确实对应 reg_a 点名的主体 | 逐条比对 12 例：`meta.subject.description` 与 `g1_region_opp.json` 的 `reg_a` 指令主体串**逐字一致**，`meta.subject.area` 与配置 `subject_area` **数值一致**（如 0.102396 / 0.167145），掩膜与源图**长宽比一致** |
| C_GT 是否可替代 | 用 T4 `source_catalog.jsonl` 的 `semantic-*` 槽 `.cgt.png`（n=79 有覆盖的源）对拍：与 SAM3 主体掩膜 **16×16 IoU 中位 0.913、ρ 中位 0.997**；用 C_GT 当标签算出的 AUC_light 0.666 vs 用 SAM3 的 0.671（同子集）。**两者可互换**，正表统一用 SAM3 subject（覆盖 300/300），C_GT 只作稳健性附注 |

- 对齐口径：掩膜 → 短边 512 bilinear → `ro9_gl_attention.luma_to_grid`（expand2square 黑边
  pad + 面积均值下采样），与 luma16/valid16 **同一函数同一路径**；标签 = 格覆盖 ≥0.5，
  只在 valid 格上算 AUC。AUC 实现 = 归一化 Mann-Whitney U（`scipy.rankdata`，并列平均秩）。
- **同时报 luma16 基线 AUC**：不报基线的话「AUC 0.66」无法排除「主体恰好比背景亮」。

### 9.3 本轮新增代码（analyze_g1.py）

`roc_auc()` / `SubjectMaskBank`（sqlite 索引 + seek 直读 tar，带内存缓存）/
`cross_token_rho`（D-22a）/ `subject_auc`（D-22b，含逐层 AUC 曲线）/
`layer_scan`（逐层，公共源对齐）/ `band_scan`（层区间）/ `by_task_conf`（分层报告）/
`shuffle_control` 的配对读法与逐层曲线。pyright 0 errors。

### 9.4 viz 命名的诚实性修正

原实现把「ρ_region_opp 最低的 6 例」一律叫 `region_success_*`。全量后**无一例 <0.3**
（最低 0.585），文件名会对外声称实验没达到的结论 → 第一轮改为 ρ<0.3 才输出
`region_success_*`、否则 `region_bestcase_*`。主批 viz 的排序分数也从 `ρ_syn − ρ_opp`
（D-17 已废止的旧主判据）改为 `ρ_syn − |ρ_Y|`（现行主批两条判据的裕度）。

**第二轮（终版归档，D-30 落地）**：`bestcase` 仍然是一个价值判断词，继续沿用会在文件列表里
读成「这些是好的」。改为**文件名只陈述实测值与实判结果**：

```
region_<PASS|FAIL>_rho<实测 ρ_region_opp>_<minrho|maxrho>_<img_id>.png
```

`minrho`/`maxrho` 只标「在全批里处于 ρ 最低/最高端」这个事实，不含褒贬；ρ 值同时进文件名与图注。
终版落盘 12 张全部是 `region_FAIL_*`（minrho 端 ρ = 0.58…0.65，maxrho 端 0.98…1.00）。

### 9.5 待主 agent 决策（保守默认已采用）

| # | 决策点 | 保守默认 |
|---|---|---|
| D9 | RO-9 的处置：判据是 gate=FAIL + shuffle 对照证否指令条件性，够不够直接淘汰 RO-9 臂 | 本报告只给**数据与判定**，按 EXPERIMENTS_v3「RO-9 失败判据 → 降级 analysis」写建议，**不擅自改 EXPERIMENTS_v3**（改计划是阶段复盘的事，见 CLAUDE.md 审阅协议 §3） |
| D10 | 层选择：无层满足判别力口径，是否改用「主体显著性 AUC」当选层依据 | `band_scan` 两个口径都报（sep / AUC），推荐区间**明确标注前提**（仅当 RO-9 降级为图像驱动显著性时才成立）；不把 AUC 口径追认为 G1 判据 |
| D11 | style 任务（86 源）fallback 率 17–25%，是否剔除后重报 | **不剔除**（fallback 逐条已记，剔除等于事后选样）；分层表 task_type×conf 里可直接看到 style 段的 fallback 与 AUC 同步走低 |

## 十、终版归档追记（2026-08-03，归档 agent；依据 D-30/D-31）

本轮**不重跑任何前向**，只做分析侧扩展 + 交付形态定稿（`analyze_g1.py` 三处改动，pyright 0 errors）。

### 10.1 metrics.json 新增：AUC 配对检验

`subject_auc.paired_rega_vs_regb_light`——同一张图上 AUC(reg_a) vs AUC(reg_b) 的**配对**读数。
此前只有两个独立中位数（0.648 / 0.642），差 0.006 无法回答「是不是逐图都一样」。新增后：
`paired_n=212`（2 源无掩膜）、`median_delta=+0.0005`、`win_rate(a>b)=0.505`、
`frac(b>a)=0.486`、`wilcoxon_p=0.121`，并按 `region_b_kind` 分解（background 47.3% / spatial 53.2%）。

**这是 gate FAIL 最直接的一个数**：指令说主体还是说背景，s 对主体掩膜的判别力统计上无差别。
重聚合前后**其余所有数字逐位不变**（已用 flatten-diff 核对：新增 13 个键，0 个键值改变，
NaN≠NaN 的 22 处为浮点恒等式伪差异）。

### 10.2 新增 viz：`failure_instrblind_*`（指令无关性主图）

选样规则**单条、写死为代码常量**，避免「挑图」质疑：

| 条件 | 值 | 为什么 |
|---|---|---|
| `region_b_kind == "background"` | — | 指令 B 明说「调背景，主体保持不变」，语义上最不该压主体 |
| `AUC(b) > AUC(a)` | — | 要展示的现象本身 |
| `AUC(b) >= 0.70` | `BLIND_AUC_MIN` | **绝对**意义上压在主体上，不只是相对更高 |
| `subj_area16 >= 0.06` | `BLIND_AREA_MIN` | 16×16 上主体至少 ~15 格，否则肉眼看不出 |

排序 = Δ = AUC(b) − AUC(a) 降序取前 3；**候选池 17 源随日志打印**（`instr-blind viz: 3 cases
(candidate pool=17, region batch=214)`），可核。落盘三例 Δ = +0.090 / +0.074 / +0.068。

面板 = 源图 / SAM3 主体掩膜 / s(A：说主体) / s(B：说背景)，**两张 s 图上叠主体掩膜的红色轮廓**
（第一版没有轮廓、且 suptitle 与子图重叠，肉眼无法判断 s 压在哪，已返工）。

### 10.3 交付定稿

- `REPORT.md` 重写为终版：**按 CLAUDE.md 强制前三行格式开头**（要验证的结论 / 为什么需要验证它 /
  怎么验的），判定表五行给终判（ρ_syn PASS / ρ_Y PASS+D-31 限定 / ρ_opp CONTROL /
  **ρ_region_opp FAIL** / **ρ_shuf 负控制 FAIL**），gate = **FAIL**；
  新增「措辞纪律（D-31）」节与「最直接的一个数」节；D-32 线索关闭写进 §五。
- `REVIEW-result.md` 换为终版结果审阅（独立视角）：判据逐条复核 / D-31 落实核验 /
  **对三条腿的分别影响** / **被证伪与未被证伪的边界（H2 未证伪）** / 论文 analysis 素材价值 /
  七条计划修改建议（其中 **R3「RO-X1 与腿A' B4 的 VLM 侧代表不能再用 RO-9」为 REPORT 未覆盖的新增项**）。
- 旧 viz（24 张，含 `region_bestcase_*` 命名）已整目录重刷，备份在本轮 scratchpad，未回写仓库。

### 10.4 本轮未做（留给主 agent 决策）

- **不改 `EXPERIMENTS_v3` / `PLAN_v2`**：改计划是阶段复盘的事（CLAUDE.md 审阅协议 §3），
  本轮只出数据、判定与建议。R1–R7 待主 agent 裁定后落文档。
- 主批 `success_*` / `failure_*` 的命名未改（它们指的是**主批两条判据的裕度**，与 gate 无关，
  REPORT §六已加注）。审阅建议下一轮统一成判据化命名，本轮不动以免与在途引用冲突。
