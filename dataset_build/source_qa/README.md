# Source QA — 数据集源资产清洗与人工评审

清洗 VeraRetouch 构建的**输入资产**（在任何 S1-S8 stream 运行之前），并把结果**回流**到 build：

- **输入图** (`source_index.jsonl`, ~124k) → 内容/感知去重 → NR-IQA + 廉价检测器 → 分级问卷 A(真实性)+B(适用性) → gate 判级
- **预设/LUT** (`recipe_index.jsonl`, ~12k) → 元数据门 → **真实 Lightroom 渲染** + 配对指标 + 问卷 C
- **闭环**：`apply.py` 把判级/去重/local-mask 标记写成 `*.cleaned.jsonl`，build 读它（否则 QA 不影响训练集）

所有处理写入 **PostgreSQL**（不再是 SQLite——MVCC 根治了多写者死锁/`database is locked`）。**不破坏性删除**：自动判级只给 `auto_verdict` 建议，最终 keep/drop/hold 由人工在 Web UI 决策（append-only，可随时覆盖）。

## 设计要点

- **闭环（最关键）**：build 直接读 `source_index.jsonl`/`recipe_index.jsonl`，**从不读 qa 库**。必须跑 `apply.py` 生成 cleaned 索引并在 `config.yaml storage.{source,recipe}_index` 指过去，QA 才会真正过滤训练集。
- **源图去重**：`imagehash` DCT-pHash + `pybktree` 近重复聚类 + 像素 sha256 精确去重；ppr10k 的 a/b/c 三胞胎（同源 3 专家目标）只聚簇不删，judge 一次、apply 时 fan-out。
- **分级问卷**：A 二元硬失败；B = 安全(二元) + 场景(是/否/不确定) + 画质/压缩/主体/人脸(0-3 分级)；NULL **不**坍缩成 fail（缺信号→review，fail-closed）。
- **绝对硬门 + 相对软尾**：分辨率(megapixels/longedge，新 decode)、portrait 人脸为**绝对**硬门；NR-IQA 坏尾用 per-corpus 分位**相对**投票。
- **真实 Lightroom 渲染**：param 预设(xmp/lrtemplate，含 290 个 local-mask)提交到已运行的 **JarvisEvo LR 任务 server(:8081)**，由 Win/Mac LrC client 真渲染（XMP 含 mask → 局部编辑被忠实应用）；LUT 用 numpy trilinear(tier-3)。渲染按 `(content_hash, probe_id, engine)` 缓存到 `render_jobs`，幂等可续。
- **local-mask 重点标记**：`has_local_mask` 预设 stage1 走 `preset_meta_local`、只走真实 LR、UI 醒目徽标、apply 标 `qa_local_edit`、build 据此排除全局流。
- **环境**：base conda env（torch+cuda+cv2+pyiqa+skimage+fastapi+psycopg + imagehash/pybktree/json_repair）；LLM 复用运行中的 vLLM `:8002`（`qwen3_5-35b-a3b` 多模态，thinking 关）；NR-IQA 跑 GPU0。

## PostgreSQL

库级隔离的独立库（复用本机 PG16 容器）：`postgresql://vera:vera@127.0.0.1:5432/vera_source_qa`（可用 `SOURCE_QA_PG_DSN` 覆盖）。表：`assets / iqa_scores / llm_qa / preset_previews / render_jobs / processing_events / decisions / runs / gate_thresholds`。`db.connect()` 返回兼容 `sqlite3.Row` 的薄封装（`?`→`%s`、双索引行）。

## 命令（均从仓库根 `/home/bc/VeraRetouch` 运行，base python）

```bash
PY=/home/bc/miniconda3/bin/python

# 0) 建表（幂等）
$PY -c "from dataset_build.source_qa import db; db.init_db()"

# 1) 入库（图 + 预设 + tag_cache 的 scene/style/aesthetic）
$PY -m dataset_build.source_qa.ingest                 # 全量
$PY -m dataset_build.source_qa.ingest --presets-only  # 只预设(快)

# 2) 源图去重（ingest 后、iqa 前；省下游算力）
$PY -m dataset_build.source_qa.dedup                  # 精确+近重复
$PY -m dataset_build.source_qa.dedup --pixel-only     # 只 tier-1 精确(快)

# 3) NR-IQA（GPU0）：MUSIQ+CLIP-IQA++NIQE+BRISQUE+Laplacian + megapixels/noise/face
$PY -m dataset_build.source_qa.iqa --corpus tad66k
$PY -m dataset_build.source_qa.iqa --limit 50         # 抽样测试

# 4) LLM QA：分级问卷 A+B+caption（含廉价 tech-gate 三级分流，省 35B 调用）
$PY -m dataset_build.source_qa.llm_qa --corpus korean --concurrency 24

# 5) 预设 QA
$PY -m dataset_build.source_qa.preset_qa stage1                 # 元数据门+content_hash+local 路由(快)
$PY -m dataset_build.source_qa.preset_qa stage2 --limit 50      # 真实 LR 渲染+配对指标+问卷C

# 6) 标定 + 自动判级（非破坏，给 auto_verdict 建议）
$PY -m dataset_build.source_qa.calibrate                        # per-corpus 分位阈值
$PY -m dataset_build.source_qa.gate                             # relative 模式
$PY -m dataset_build.source_qa.gate --apply-auto-decisions      # 同时写 auto:gate 决策(仍可人工覆盖)

# 7) 启动评审 Web（默认 :8077）
$PY -m dataset_build.source_qa.webapp.app

# 8) 闭环：把判级/去重/local-mask 写成 cleaned 索引（否则 QA 不影响 build）
$PY -m dataset_build.source_qa.apply --dry-run                  # 先看会删多少
$PY -m dataset_build.source_qa.apply                            # 写 source_index.cleaned.jsonl / recipe_index.qa.jsonl
# 然后在 dataset_build/config.yaml 把 storage.source_index / recipe_index 指向 cleaned 文件
```

推荐流水线顺序：`ingest → dedup → iqa → llm_qa → preset_qa(stage1,stage2) → calibrate → gate → (人工评审) → apply`。

## Lightroom 渲染服务（已运行）

真实渲染走 `/home/bc/retouching/JarvisEvo` 的 **LR 任务 server**（`lrc_scripts/servers/lrc_task_server.py`，:8081，反向连接：Win/Mac 跑 LrC 的 client 来拉任务）。`source_qa/lr_render.py` 用 JarvisEvo 自带转换器把预设转成 `config.lua`（`.xmp`→`xmp2lua.parse_xmp` 保留 mask；`.lrtemplate`→`LuaConverter` 取 `value.settings`），`submit_task_with_files` → 轮询 `task_status` → `download_task_result`。需有 LrC client 在线（`GET :8081/api/stats` 看 active_clients）。

## Web UI

- `/` 看板；`/assets` 画廊（按 type/corpus/状态/判级/决策/PASS_A/PASS_B/**局部编辑**/指标阈值过滤、批量决策）；`/asset/{id}` 详情（原图、IQA 分、问卷逐题、**local-edit 警告徽标**、预设 before/after + render_engine + 配对指标、溯源时间线、决策历史）。
- 快捷键：详情页 `k`/`d`/`h` = keep/drop/hold 并跳下一条。

## 配置

阈值/路径/问卷/端点/渲染都在 `config.py`（`GATE` 绝对硬门+相对软尾；`QUESTIONNAIRE_A/B/C` 为单一事实来源，带 `type`：yn/yn3/grade）。DSN 用 `SOURCE_QA_PG_DSN` 覆盖，LR server 用 `SOURCE_QA_LR_URL`。
