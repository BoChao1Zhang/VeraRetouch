# Source QA 评审报告 — `dataset_build/source_qa/`

> 日期：2026-06-15 ｜ 方法：6 视角并行深读真实代码 → 对每条 P0/P1 做对抗式代码验证 → 基于已验证事实出 3 份重设计（25 agents）。
> 所有结论均带 `file:line`，并在“验证”阶段对照实际代码 + 线上 `qa.db`/`source_index.jsonl`/`recipe_index.jsonl` 复核过。
> 关联：`DATASET_QUALITY_AUDIT_2026-06-13.md`（数据集成品质量审计）。本报告审的是**输入清洗子系统本身**。

---

## 执行摘要（TL;DR）

`source_qa` 的**工程骨架是好的**（append-only SQLite 溯源、非破坏性 `auto_verdict` 建议 vs 人工 `final_decision`、问卷单一事实来源、各 runner 幂等可续跑）。但它有 **4 个结构性问题**，按严重度：

| # | 问题 | 严重度 | 一句话 |
|---|---|---|---|
| 1 | **QA 是死端，产物从不回流 build** | **P0** | build 直接读 `source_index.jsonl`/`recipe_index.jsonl`，从不读 `qa.db`；线上 `qa.db` 有 **0 条 final_decision**、仅 9 条 preset_previews。124k 次 35B 调用 + 全部人工决策对 1M 训练集**零影响**。 |
| 2 | **源图无内容/感知去重** | **P1** | `source_id=sha1(path+size)`；线上 17,750 条 exact-path 重复**全部是 ppr10k 的 a/b/c 三胞胎**（同一像素文件被判 3 次）；跨 corpus 近重复**从未测量**。 |
| 3 | **图像 QA 有致命盲区** | **P0** | **无任何分辨率门**（且 `width/height` 在源索引与 qa.db 里 **100% 为 NULL**）；问卷 NULL 坍缩导致 relative gate **静默绕过真实性门**；单评委无校准。 |
| 4 | **预设 render QA 半残 + local-mask 未用** | **P0/P1** | stage2 **只渲染 LUT，跳过全部 6,218 个 param 预设（≈51%）**；`has_local_mask`（线上 290 个，全 xmp）检测了但**全链路零使用**，会被当全局编辑误渲染/误配对。 |

**最重要的一条**：在补上”回流闭环”（让 build 读 QA 结果）之前，上面其它所有改进都是 shelfware。先做闭环，再做其余。

---

## 实施状态（2026-06-15，已落地）

5 个问题全部实现，跑在 **base conda env**。存储 **SQLite→PostgreSQL**（根治死锁）：独立库 `postgresql://vera:vera@127.0.0.1:5432/vera_source_qa`（复用本机 PG16 容器，库级隔离）。`db.py` 用 psycopg3 重写，连接封装做 `?`→`%s` + 兼容 `sqlite3.Row` 的双索引行工厂，旧 SQL 基本不改即用。第三方库（base）：`imagehash` / `pybktree` / `json_repair`。

| 模块 | 状态 | 真实冒烟结果 |
|---|---|---|
| `db.py`(PG) + `config.py` | ✅ | 9 表建好；ingest 预设 12,192 入库 9.7s；ingest 图 + stage1 **并发零死锁** |
| `apply.py`(闭环) | ✅ | 干跑：recipes kept 11,575 / drop 617 / **local_edit 279** / id_coverage 1.0 |
| `dedup.py`(pHash+pybktree+像素 sha256，ppr10k fan-out) | ✅ | 800 图填好 pixel_sha256/phash/megapixels |
| `iqa.py`(分辨率/噪声/人脸) | ✅ | 40 图：musiq + 原生 w/h/megapixels + noise 全落库 |
| `llm_qa.py`(分级问卷+tech-gate) | ✅ | 真实判 3 张：b_quality/comp/subject + n_answered/n_yes 正确；tech-gate 真删 5 低质图 |
| `gate.py`(绝对硬门+NULL fail-closed) | ✅ | keep 2 / review 762 / drop 36 |
| `calibrate.py`(修 min_n+PG) | ✅ | per-corpus 阈值写入 |
| `paired_metrics.py`(ΔE2000/SSIM/EMD) | ✅ | self/self noop=1，b/a ΔE=25.6 |
| `lr_render.py` + `preset_qa.py`(真实 LR) | ✅ | LR :8081 健康、**3 client 在线**；xmp→config.lua **mask 保留**；stage1 3000：pass2323/fail617/**local60** |
| webapp(local 徽标/过滤/render_engine) | ✅ | import 通过 |

**Lightroom 渲染**：直接对接 `/home/bc/retouching/JarvisEvo` 已运行的 LR 任务 server（:8081，反向连接，Win/Mac LrC client）。下发预设存成 **config.lua**（Lua develop 表），故 `lr_render` 复用 JarvisEvo 的 `xmp2lua.parse_xmp`（保留 mask）/`LuaConverter`（lrtemplate）。**未自建监视文件夹协议**。

**⚠ 待调阈值（非 bug）**：`min_megapixels/min_longedge` 初值会把 **tad66k（最大 corpus，~800px/0.4MP）100% 判掉**——已下调到只拦真正缩略图（0.30MP/640px），但这是个需按 corpus / gold-set 标定的策略决定：800px 是否可接受作”退化-还原目标”。`config.GATE` 全部阈值都是起点值，需用人工金标（`judge_kappa`）再调。

---

## 0. 已核实的事实基线（含对初版 finding 的修正）

验证阶段推翻/修正了几条初判，这些修正是后续设计的前提：

1. **`width/height` 不是“没读”，而是根本没有**：源索引与 `qa.db` 中 100% 为 NULL（`registry._emit_source` 只 `stat` 文件大小，从不 decode）。⇒ 分辨率门必须**新增一次 decode**，不能只加阈值。
2. **17,750 条 exact-path 重复 100% 是 ppr10k**：`registry._scan_ppr10k` 对每张源 PNG 发 a/b/c 三行（同 `path`、不同 `source_id`）。非 ppr10k 的 exact-path 重复为 **0**。这三者是 3 个专家**目标**共享一个源像素文件——**不应删 b/c**，而应按像素去重判定再 fan-out（省 17,750 次 IQA + 17,750 次 35B）。
3. **去重无现成依赖**：`imagehash/faiss/hnswlib` 均未安装；CLIP 向量**未持久化**（`tag_cache` 只存标量 `aesthetic*`，`aesthetic.py` 算完即弃 768d 向量）。⇒ 用 **纯 numpy/scipy DCT-pHash + LSH bit-banding**，别假设有 faiss/CLIP-cosine。
4. **param/LUT ≈ 51/49，并非“绝大多数”**：线上 12,192 = **6,218 param**（3,632 xmp + 2,586 lrtemplate）+ **5,974 lut**（5,922 cube + 52 3dl）。但“param 全跳过”仍成立。
5. **local-mask 数量有歧义**：`qa.db` 实测 **290 个**（全 xmp，0 LUT）；`config.yaml:170` 写的是 `local_mask_recipes: 2510`。两者口径不同（2510 应是去重前/全量统计），**需对账**——任一口径下都必须特殊处理。
6. **stage1 已跑、stage2 几乎没跑**：10,571 `preset_meta_pass` / 1,621 `preset_meta_fail`，但 `preset_previews` 仅 9 行；`final_decision` 0 行 ⇒ 人工评审与 render 半条流水线从未真正端到端跑过。**现在重构 stage2 风险极低**（没有大缓存要迁移）。
7. `recipe_index.jsonl` **在盘上**（5.1MB，每行带 `has_local_mask`）。
8. `--luts-only` CLI flag 是**死参数**（`luts_only` 实际由 `not args.all_kinds` 决定）。

---

## 1. 整体流程合理性（问题 a）

**结论：架构合理，但功能上是死端——这是头号问题。**

### FLOW-1 / FLOW-2（P0，已确认）QA 产物从不被 build 消费，且无任何 apply 机制
- `run.load_plan_inputs`（run.py:505-509）直接 `_read_jsonl(source_index.jsonl/recipe_index.jsonl)`，唯一过滤是 `s.corpus in corpora`；recipe 池无条件 `rec_pool = recipes`（run.py:523）。
- 对 `qa.db|source_qa|final_decision|auto_verdict|decisions` 在 `run.py/streams.py/registry.py/pack.py/recipes.py/render.py/config.yaml` grep **零命中**；build 全程不开任何 SQLite。
- 所有决策 sink（`gate.py:124` 的 `--apply-auto-decisions`、webapp 三个 `/api/*decision`）都终止于 `db.add_decision` → 只写 `qa.db`。
- `source_qa/` 内**没有任何导出器**把 QA 结果写回一个 build 能读的索引。
- **影响**：PASS_A 无效图、PASS_B 不合格源、硬 IQA 失败、预设近 no-op/恒等/重复——全部照常流进 S1-S8。约 124k 次 35B 多模态调用 + 全部人工工时，对实际训练数据**零改变**。

### FLOW-3（P1，已确认）无成本分级：35B 评委对每张图都跑
- `llm_qa` 选活只看“没有 A 问卷行”（llm_qa.py:163-164），**不要求 IQA 先跑、也不跳过硬 IQA 失败图**；廉价信号只在 gate 阶段事后使用（gate.py:26-35）。
- 顺序还很脆：允许 llm_qa 先于 iqa 跑，导致“先付费再被 IQA 判死”。

### FLOW-4（P2）单进程 write_lock + 单 SQLite 多写者
- 正确性 OK（WAL + busy_timeout=120s + write_retry 回滚重试），但吞吐被串行化；建议事务批量提交（每 N 条一个 write_retry）。

### FLOW-5（P3，正面）分级顺序与续跑守卫健全
- `ingest→iqa→llm_qa→calibrate→gate→human` 逻辑清晰；各 runner 用 `NOT EXISTS`/upsert 幂等；append-only 溯源 + 单题人工 override（app.py:360-379）设计良好。**唯一缺陷是它什么都喂不到（见 FLOW-1/2）**。

### FLOW-6（P2）gate 覆盖盲区
- gate 只判 `asset_type='image' AND (musiq OR pass_a) 非空`；两项都失败的图被**静默排除**、永远拿不到 verdict；param 预设永远停在 `auto_verdict='review'`。

---

## 2. 源图去重（问题 b）

**结论：不存在任何内容/感知去重。**

- **DEDUP-1（P1，已确认）**：`source_id=_stable_id(path,size)`（registry.py:113-116, 344-353），等于“这条文件路径”，识别不了：换名/跨 corpus 同图、重编码/缩放、近重复连拍。`assets.dup_of` 列存在（db.py:67）但**只对预设写**。
- **DEDUP-2（P1，已确认）**：ppr10k 把每张源 PNG 发成 a/b/c 三行（registry.py:624-645），线上 **26,625 行 / 8,875 文件 = 21.4% 的源**是逐字节同像素，被 IQA + 35B 各判 3 次（净浪费 17,750 + 17,750）。它还把 ppr10k 在 per-corpus 标定里灌水 3×。
- **DEDUP-4（P2）**：fivek/fivek_gold/ppr10k 同源（FiveK 血统）、tad66k web 抓取，跨 corpus 精确/近重复**很可能存在但完全没测**（path+size 抓不到）。需要 pixel-sha256 + pHash 才能量化。
- **DEDUP-5 修正**：原以为可复用缓存 CLIP 向量做近重复——**向量未持久化**，需重算或改 aesthetic 预计算 dump 向量；v1 用纯 numpy pHash 即可。
- **DEDUP-3（P2）**：即便 qa.db 里做了去重，build 也不读 → 同样要靠回流闭环才生效。

---

## 3. QA 设计合理性（问题 c）

### 图像问卷 / 评委
- **QA-1（P0，已确认）无任何分辨率/原生尺寸门**——退化-还原数据集里这是最致命的盲区：300×200 缩略图与 6000×4000 原图被同等对待，低分/已上采样图成为模型要“还原”的干净目标，封死全集还原上限。且 `width/height` 全 NULL（须新增 decode）。IQA 还在 1024 下采样图上算（iqa.py:44-48），把原生分辨率信号进一步抹掉。
- **QA-2（P1，已确认）`all_yes` + 单题 None 坍缩 → pass_a/pass_b 变 NULL → relative gate 静默绕过真实性门**：`_yn` 对意外措辞返回 None（llm_qa.py:109-120），`_allyes` 任一 None 即整问卷 None（llm_qa.py:146-150）；`verdict_relative` 只在 `pass_a==0` 判 drop（gate.py:69），`None==0` 为假 → 跳过，2 票 IQA 坏尾即可判死（gate.py:85）而真实性完全没看；反过来截图/海报若 A 答案丢成 None 还能逃到 `review` 而非 `drop`。**两种模式都会在 NULL 时绕过真实性门**（修正：absolute 模式也不安全）。
- **QA-3（P1，已确认）单一未校准评委是 124k 图唯一 keep/drop 权威**：温度 0.1、thinking 关、无自洽采样、无 ensemble、无 vs 人工金标的一致性测量；pass_a/pass_b 在两种 gate 模式里都是独立硬 drop。
- **QA-4（P2）题目冗余/主观**：A4（致命损坏）已被 PIL decode 确定性捕获，问 VLM 浪费 token；B1（锐利/无缺陷）与 NR-IQA 重叠且无裁决；B2（“专业级”）主观、会误杀好快照；B3（场景一致）把“内容错”和“标签错”混为一谈，标签错就误删好图。
- **QA-5（P2）`max_tokens=700` 对 8 题+8 理由+caption 易截断 → parse 失败**：失败仅记 error、续跑重试；持续失败 ⇒ 无 A/B 行、无 pass 标志，**而 build 不读 qa.db ⇒ 评委失败是 fail-open 进训练集**。
- **QA-6（P2）缺维度**：水印/图库 logo、JPEG 压缩/带状/色噪、portrait 池的人脸存在/质量、AI 生成图——都弱或缺。
- **QA-7（P3）caption 以 answer=NULL 混入答案表**，污染未来“NULL=评委失败”的审计统计。

### 指标 / gate / 标定
- **E-IMG（P2，部分确认 + 修正）**：gate 只用 5 个 NR-IQA；**无分辨率/噪声/人脸门**；`aesthetic`/`aesthetic_vlm` 已入库但 **gate 零权重**；`calibrate.py` 的 `min_n` 是**死代码**（calibrate.py:58-59 比较硬编码为 1 且 body 为 `pass`）；drop=p10 坏尾、keep=p50。**修正**：原 finding 说“no clip”错了（CLIP-IQA+ 确在用，gate.py:42）；原 IMPACT“误删最好的 corpus”被**推翻**——relative 是 per-corpus 各砍自己的底部尾，反而**偏宽松**，不是偏向砍强 corpus。
- **E-PRESET（P2，部分确认 + 修正）**：stage2 只对 after 算 NR-IQA（musiq/clipiqa+），**无 before/after 配对指标**。**修正**：原说“漏 no-op”被推翻——no-op 已在 stage1（`nz==0` / `dev<0.03`）和问卷 C3 拦截。真正缺口是**无法量化编辑幅度/渲染退化**，需全参考配对指标（ΔE/SSIM/EMD/clip%）。

---

## 4. 重设计：图像问卷 + 指标体系（问题 d）

### 4.1 问卷——硬失败用二元，质量用分级，场景用三态
替换 `config.QUESTIONNAIRE_A/B`。三种答案类型：`yn`（二元硬失败）、`yn3`（是/否/不确定，“不确定”→review 永不自动 drop）、`grade`（0-3）。

**门 A — 真实性（二元，任一 NO ⇒ drop）。删 A4**（decode 已确定性捕获，改记事件）：
- `A1` 真实拍摄/扫描的自然照片？（非截图/海报/纯图形/拼图/AI 生成）
- `A2` 无叠加污染？（无水印/平台 logo/文字条/边框/拼贴/二维码）
- `A3` 主体完整、非极端裁切/测试图/碎片？

**门 B — 适用性（二元硬失败 + 分级 + 三态场景）**：
- `B_safe`（yn 硬失败）内容安全合规？（无 NSFW/暴力/敏感）
- `B_scene`（yn3）与场景标签 `{scene}` 一致？——`scene=any/空` 时令评委答“不确定”不扣分；`否`⇒drop、`不确定`⇒review（修复 B3 误删）
- `B_quality`（0-3）退化-还原目标画质等级（已精修成品按画质评分，不因已调色扣分）
- `B_comp`（0-3）压缩/伪影等级
- `B_subject`（0-3）主体/构图清晰度
- `B_face`（0-3，**仅 portrait 池**）人脸清晰、无塑料感/严重模糊？

**通过规则**（替换 `all_yes`/`_allyes`）：
- `pass_a=1` iff A1-3 全部已答且全 yes；`=0` iff 任一答 no；`=NULL` iff 任一未答（**不把 parse-miss 坍缩成 fail**）。
- `pass_b=1` iff `B_safe=yes AND B_scene!=否 AND B_quality>=2 AND B_comp>=2 AND B_subject>=1`（portrait 另需 `B_face>=1`）。
- 分级整数存 `llm_qa.answer`（本就是 INTEGER）；加去规范化列 `b_quality/b_comp/b_subject/b_face`；加 `n_answered/n_yes` 让 gate 区分“缺信号”vs“判失败”。

**鲁棒性**：`max_tokens` 700→1100；理由只在 no/uncertain 时要；若 vLLM 支持 guided-JSON（xgrammar/outlines）则约束 schema；保持 thinking 关（防 CoT 截断）。

### 4.2 指标——补 VLM 弱项的廉价确定性统计（与 IQA 同跑在 GPU0）
保留 musiq/clipiqa+/niqe/brisque/laplacian，**新增**：
| 指标 | 算法 | gate 用法 |
|---|---|---|
| `megapixels`/`longedge` | **新增 decode**（iqa.score_path 本就开图，顺手取 `im.size`），写回 `assets.width/height/megapixels` | **硬门**：min_longedge 1024（portrait 1536）、min_megapixels 1.5 |
| `noise_sigma` | Laplacian 残差 MAD | 佐证 B_quality |
| `jpeg_blockiness` | 8×8 块边界 DCT 能量（`scipy.fftpack.dct`，已装） | 佐证 B_comp |
| `upscale_suspicion` | `sharpness/√megapixels` vs corpus p10 | 软标记→review |
| `max_face_frac`/`face_count` | 轻量检测器（cv2 YuNet/SCRFD），**仅 portrait** | portrait 硬门：人脸须占帧 ≥4% |
| `aesthetic_vlm` | 已入库，给**软 keep 票**（现为零权重） | keep band |

### 4.3 分级顺序——廉价门在 35B 之前
`ingest → dedup → iqa → tech_gate（确定性）→ llm_qa（仅幸存者）→ gate（终判）`。把硬技术门折进 `llm_qa` 选活 WHERE：要求 `musiq IS NOT NULL`（IQA 先跑）+ `NOT(megapixels<MIN_MP OR musiq<TECH_MUSIQ OR sharpness<TECH_SHARP OR jpeg_blockiness>TECH_BLOCK)`，这些直接 `auto_verdict='drop'(tech_floor)`，不进评委。预计省 10-20% 的 35B 调用（叠加 ppr10k fan-out 更多）。

### 4.4 gate 逻辑——NULL fail-closed、绝对硬门 + per-corpus 软尾
在 `verdict_for`/`verdict_relative` 顶部加共享前导：
```python
def _preamble(a):
    if a["megapixels"] is not None and a["megapixels"] < G["min_megapixels"]:
        return "drop", f"megapixels<{G['min_megapixels']}"
    if a["is_portrait_pool"] and (a["max_face_frac"] is None or a["max_face_frac"] < G["min_face_frac"]):
        return "drop", "portrait: no usable face"
    if a["pass_a"] is None or a["pass_b"] is None:
        return "review", "incomplete LLM QA (NULL)"   # fail-CLOSED
    return None
```
**绝对硬门**（分辨率/人脸/真实性）不相对化；**per-corpus relative 只用于 NR-IQA 软尾投票**（calibrate 的前提对 IQA 分布成立）。修 `calibrate.py` 死的 `min_n`：`n<min_n` 时跳过该 corpus 回退 `'*'`。

### 4.5 校准指标（用于信任自动 drop）
- `judge_kappa`：300-500 张人工金标（按 corpus×auto_verdict 分层）逐题 Cohen's κ；κ≥0.6 才允许该题进硬 drop 门，<0.4 仅 review。
- `judge_call_reduction`、`null_pass_rate`（目标 <1%）、`resolution_drop_rate`（任一 corpus 不应 >50% 因分辨率被删）、`portrait_face_coverage`（≥0.9）。

---

## 5. 重设计：源图去重模块

新模块 `source_qa/dedup.py`，跑在 **ingest 之后、iqa/llm_qa 之前**（省下游算力）：

- **Tier-1 精确**：decode→RGB→**pixel_sha256**（不是文件字节，抗重编码/元数据噪声，也顺带收掉 ppr10k 三胞胎）。
- **Tier-2 近重复**：纯 numpy DCT-pHash（灰度→32×32→DCT→左上 8×8 去 DC→中位阈值→64bit），用 **8 段 LSH banding** 聚类（Hamming≤6），O(n)，124k 仅秒级。
- **Tier-2b（可选）**：CLIP-L/14 cosine≥0.95——仅当补了向量持久化。
- **簇头选择**：簇内按 `(megapixels, musiq, clipiqa, aesthetic_vlm)` 字典序取最优为 head；其余 `dup_of=head, auto_verdict='drop'`（**复用预设已有的 dup_of 模式**）。
- **ppr10k 特例（先做、零风险）**：**不删 b/c**，在 iqa/llm_qa 选活时按 `pixel_sha256` 去重、把判定 fan-out 到三个变体——省 35,500 次冗余操作，build 所见不变。
- **新列**：`pixel_sha256 / phash / dup_cluster / megapixels` + 索引；每簇写 `processing_events(stage='dedup')`。
- **防泄漏**：同一感知簇 + ppr10k a/b/c 家族**整体分到同一 train/eval split**（在 apply 的 cleaned 索引与 manifest 里记 `split`/`dup_cluster`）。
- **跨 corpus 确认**：算完 pixel_sha256 出 corpus×corpus 碰撞矩阵，量化 fivek/fivek_gold/ppr10k 隐藏重复（会扭曲 per-corpus 标定）。
- CLI：`python -m dataset_build.source_qa.dedup [--pixel-only]`。

---

## 6. 重设计：真实 Lightroom 渲染 QA + local-mask 标记（问题 e）

**现状（已核实）**：stage2 只用 numpy trilinear 渲 LUT，**硬跳过全部 param 预设**（preset_qa.py:302-304，"param render needs teacher"）；唯一能应用 param 的 `render.py` 是**全局神经近似 teacher，不是 Adobe**；探针在 900px 上再压 JPEG q85（双重有损，污染 C1“塑料/带状”判断）；`has_local_mask`（290，全 xmp）全链路零使用，streams.py 给 region-local 样本换的是 **SAM3 mask**（streams.py:1262），不是预设自己的 mask。

### 6.1 三层渲染服务（诚实可行性：build 主机是 Linux，无法跑进程内 Adobe）
- **Tier-1 真 LrC 节点（判定权威）**：专用 **Win/Mac** 机跑 Lightroom Classic，用 Lua SDK 插件或**监视文件夹协议**（Linux 投 `{job}/probe.tif + preset.xmp` → 插件导入→套用 develop 预设（**含 mask**）→导出 `after.tif/jpg + done.json`）。LrC 仅 GUI/Win/Mac，必须 off-host。
- **Tier-2 Linux 兜底**：`darktable-cli`/`rawtherapee`（确定性栅格引擎，读 ACR 子集）。**local-mask 预设永不走 Tier-2**（不能忠实复现 mask）。
- **Tier-3 降级**：numpy LUT trilinear 仅作 LUT 廉价非恒等预筛；`render.py` 神经 teacher **仅训练用、永不作 QA 判定**。
- 每条预览打 `render_engine` 标签，gate/人工据此加权（lrc > darktable > approx）。
- **引擎选择规则**：`has_local_mask → 仅 lrc（无节点则 needs_local_render 转人工，绝不出全局渲染）`；`lut → lrc 否则 tier3`；`param 无 mask → lrc 否则 darktable`。
- **吞吐（诚实）**：stage1+dedup 后 ≈9k 预设 × 固定 6 探针 ≈ 27–54k 次渲染，单节点 ~数小时一次性离线跑，按 `(content_hash,probe_id,engine)` 缓存持久化，非每次 build 成本。

### 6.2 幂等缓存作业协议 + schema
- **缓存键 = `(preset_content_hash, probe_id, render_engine)`**（不用 recipe_id，否则跨路径同预设重渲）。`preset_content_hash`：param=规范化参数字典 sha256，lut=归一化 grid sha256，stage1 时算一次存 `assets`。
- **固定 6 张探针**写进 `config.PRESET_PROBE_IMAGES`（现为空 → `pick_probe_images` 非确定性 ORDER BY，缓存永不命中）。2 人像/2 风光/1 高纹理/1 低调肤色。
- 新表 `render_jobs(job_id PK, recipe_id, preset_content_hash, probe_id, render_engine, region_local_flag, status, after_tif/jpg_path, engine_version, attempts, node, error, ... , UNIQUE(content_hash,probe_id,engine))`；扩 `preset_previews` 加 `render_engine/job_id/region_local/paired_metrics`。提交前查缓存命中即跳过并 fan-out。

### 6.3 配对 before/after 指标（替换 after-only IQA）
存 `preset_previews.paired_metrics`（CPU、ms/对）：
- `delta_e2000_mean/_p95`（CIEDE2000，编辑幅度/最坏色偏）
- `ssim`（结构保持；过低=破坏性裁剪）
- `hist_emd_L/ab`（色调/调色足迹）
- `clip_pct`（新增高光死白/暗部死黑比例）
- `after_musiq/clipiqa+`（after 绝对质量）
- `noop_score = (ΔE<1.5 AND ssim>0.99)`（确定性 no-op 兜底）
- **编辑幅度门**：所有探针 `ΔE<1.5` ⇒ 近 no-op，**不花评委直接 drop**；`clip_pct>0.10 或 ssim<0.5` ⇒ 标 destructive 给评委。
- **探针以 PNG/16bit TIF 进评委**，不再双重 JPEG。

### 6.4 local-mask 标记 + 特殊路由（用户硬性要求）
- **stage1**：`has_local_mask=1` → `status='preset_meta_local'`, `auto_verdict='needs_local_render'`，绝不落入全局路径。
- **渲染**：`region_local_flag=1` 只走 Tier-1 lrc（含 mask）；无节点则停在 `needs_local_render` 转**强制人工**，结构上禁止全局兜底。
- **webapp**：详情/画廊加醒目徽标 **“LOCAL EDIT — 全局 teacher 无法复现”** + `render_engine` 徽标 + `has_local_mask` 过滤器（现仅 asset.html:38 内联打印 `mask=...`，无警告/过滤/路由）。
- **streams.py（下游契约，修 LENSF-3）**：读 `qa_local_edit`，**把 290 个 mask 预设排除出全局流（S6）**，绝不再用 SAM3 mask 顶替其真 mask；为高价值 mask 预设持久化真 LrC **masked after + 导出 mask 栅格**（keyed by content_hash,probe_id），供未来 region-aware 流当真 `C_GT` 用，替代 `region_composite` 的 SAM3 假 mask。

### 6.5 预设感知去重（替换精确签名）
现 dedup 是 sha1 等值（参数 round 到 0.1，原始滑块 -100..100 ⇒ 几乎不量化；LUT 只沿 axis-0 strided）。改：粗参数量化（5 单位桶）做快筛 → 用**真渲染的 look 嵌入**（中性 ramp 的 ΔLab 传递曲线 + after 的 CLIP/aesthetic）cosine 聚类。问卷 C 增强：≥5 探针、scene 分层、需 ≥80% 探针通过（非 3 选 2 多数）；style/scene 为空时中和 C2。删 `preset_qa.py:155` 死三元、内联 dup→drop。

---

## 7. 实施优先级（高价值优先）

1. **闭环 `apply.py` + config 指向**（P0）：新增 `source_qa/apply.py` 写 `source_index.cleaned.jsonl`/`recipe_index.qa.jsonl`（复用 `registry._AtomicWriter`），把 `config.yaml storage.{source,recipe}_index` 指过去——`load_plan_inputs`/`registry.load_*` 已支持该 indirection，**零热路径改动**。规则：`drop` = `final_decision='drop'` ∪ (`final_decision NULL AND auto_verdict='drop'`) ∪ `dup_of 非空` ∪ 无 QA 行(fail-closed)；`hold/review` 保留并打 `meta.qa_status`。**ID 稳定守卫**：apply 必须对同一索引代跑（断言 ≥99% asset_id 仍能解析，否则中止并报孤儿数；miss 回退 path-join）。**最低成本、最高杠杆**。
2. **`dedup.py` + ppr10k fan-out**（P1）：立省 ~17,750 IQA + ~17,750 35B；纯 numpy，无新依赖。
3. **分辨率 decode 门 + NR-IQA 三级分流**（P0/P1）：补 QA-1 最大洞（注意是**新 decode**，不是改阈值）。
4. **gate NULL fail-closed + 分级问卷**（P1）。
5. **真渲染节点 + local-mask 标记/路由**（P0/P1，基建最重）：离线一次性、单节点数小时；无 Win/Mac 节点则 290 个 mask 转人工、其余走 darktable tier-2。

> ⚠️ 横切风险：`source_id/recipe_id=sha1(path+size)`，任何 rescan（移动/改尺寸）都会变 ID 静默孤儿化全部决策——apply 必须钉同一索引代或回退 path-join。

---

## 附录：16 条已验证 finding

| ID | 严重度(初→终) | 结论 | 一句话 |
|---|---|---|---|
| FLOW-1 | P0 | 确认 | build 不读 qa.db，QA 死端 |
| FLOW-2 | P0 | 确认 | 无任何 apply/导出机制 |
| FLOW-3 | P1 | 确认 | 无成本分级，35B 跑全量 |
| DEDUP-1 | P1 | 确认 | 源图无内容/感知去重 |
| DEDUP-2 | P1 | 确认 | ppr10k 3× 同像素（21.4%） |
| QA-1 | P0 | 确认 | 无分辨率门 + width/height 全 NULL |
| QA-2 | P1 | 确认(+修正) | NULL 坍缩 → 两种模式都绕过真实性门 |
| QA-3 | P1 | 确认 | 单评委无校准是唯一权威 |
| PRESET-1 | P0 | 确认 | stage2 跳过全部 6,218 param 预设 |
| PRESET-2 | P0→P1 | 确认 | has_local_mask 检测但零使用（当前潜伏） |
| E-IMG | P1→P2 | 部分 | gate 弱：无 res/noise/face、aesthetic 零权重、min_n 死代码；“no clip”错、“误删强 corpus”被推翻 |
| E-PRESET | P1→P2 | 部分 | after-only IQA 无配对指标；“漏 no-op”被推翻，真缺口是无法量编辑幅度 |
| LENSF-1 | P1 | 确认(+修正) | param 仅 ~51%（非绝大多数），但全跳过 |
| LENSF-2 | P1 | 确认 | 无真渲染器，teacher 是全局神经近似 |
| LENSF-3 | P1 | 确认 | local-mask 未标记/路由，被当全局误渲 |
| LENSF-5 | P1 | 确认 | 渲染产物同样死端，build 不读 qa.db |
