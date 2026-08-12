# DataBuild 实施文档（2026-07-27 合并版）

- 来源：`docs/archive/DATABUILD_CANONICAL_REFACTOR_2026-07-20.md`（契约权威，已归档）
  与 `docs/archive/DATABUILD_IO_REFACTOR_2026-07-27.md`（IO 实测与决策，已归档）。
  两份原文仍是背景权威；本文只保留已完成部分的设计摘要与**待实施清单**。
- 完成度基线（2026-07-27 全面核验）：canonical 契约实现约 95%，删除矩阵 0 残留，
  测试 142 通过（含 57 subtests）。

## 一、已完成设计摘要（不再展开细节）

- **单一生产命令** `python -m construct.agent run --config <toml>`：TOML 严格校验
  （unknown/缺失/枚举/ratio/0600/redaction 全启动期拦截），无环境变量行为覆盖；
  另有 `[legacy_import]` 第二流水线（r5 快照导入 + 重标注），状态机多一个 `import` 阶段。
- **源与选择**：SAM3-ready 源池（发现/门控已穿透归档）、场景分层 + largest-remainder
  配额、(build_id, mode, filter) 命名空间下大类/小类 seeded-bag 等频覆盖、组内 8 preset
  全局去重、失败补位/换大类重开、并发安全可 resume。
- **渲染与 mask**：GPU-only 工厂 fail-closed（无 farm/CPU 边）、1024/q95/pre-encode
  diff、2+2+2+2 mask 协议（semantic 共享 α，7 物理资产）、linear 质量缩放、
  确定性配对、C_GT=effective α。
- **可见性门与 QA**：α 加权 CIEDE2000 双阈值门、OneAlign + 确定性 veto、q≥0.50
  top-2、SFT-only（DPO 全删）。
- **标注**：openai SDK（钉 2.46.0）`/v1/responses` 流式 + 严格 Structured Outputs、
  多 key lane 池（least-inflight、配额摘除、耗尽转本地 vLLM/vGate）、3 轮×4 次
  durable 队列、失败只抑制 SFT 行。
- **状态机与输出**：groups/sft/failures/manifest 四权威 JSONL（fsync journal、
  torn-tail、幂等 resume、flock）；PG 仅为 viewer 投影，宕机不停产。
- **Viewer**：canonical inspector（8 候选条、C_GT/overlay、全维过滤），Playwright
  桌面/移动用例齐全。
- **IO 已完成**：数据全量 WebDataset 化入 NFS（332 组验证 0 失败）、
  `/var/cache/veradata`（模型/索引/preset bank）、`/mnt/ramstage` 24G tmpfs、
  全局 sqlite 索引 + `archive_reader`（immutable=1、范围查询）、打包器
  `MEMBER_ORDER_PLAN` 成员序不变量。

## 二、待实施清单

状态标记：[x] 完成 / [ ] 待做。按阶段顺序执行；A 先行，B/C 可并行，D→E 依赖。

### 阶段 A：止血

- [x] **A1 环境**：iaa437 `pip install -e .` + `openai==2.46.0`（官方 PyPI 源；
  tuna 镜像 403）。验收：全量 pytest 不带 ignore 通过。
- [x] **A2 归档穿透补全**：`archive_reader` 增加 `open_rgb(path)` helper
  （read_bytes + BytesIO + exif_transpose + RGB），替换 5 处直读：
  `rendering.py preprocess_source`、`canonical_masks.load_subject_alpha`、
  `canonical_qa._stats`、`responses.encode_image_data_url`、`iaa._decode_rgb`。
  验收：源图仅在归档时全链路可跑；新增穿透测试。
- [x] **A3 legacy 工具路径**：`render_backend.py:79` `_LUT_PACK_DIR` 默认值改
  `/var/cache/veradata/preset_bank_full`（只影响 bake_luts 等非 canonical 工具；
  canonical 有自己的 `_LutLoader` 按需加载，无此问题）。

### 阶段 B：inventory 预计算 + SQL 化（冷启动 8 min → <1 s）

- [x] `land()` 增加 enrich 钩子：打包 cache/subject 时顺手算 `mask_area` 并把
  eligible/ineligible_reason/scene/subject 等写入 metadata；判据与
  `_inspect_cache_dir` 逐条对齐。
- [x] `build_inventory`：本机 cache 树不存在时改走一条 SQL
  （`samples.meta` 的 `eligible` 过滤）；`_inspect_cache_dir` 保留为参照实现，
  `--legacy-inspect` 可回滚。
- [x] 一次性回填已执行（2026-07-27）：56,778 样本 → eligible 34,534 /
  mask_area_guard 2,234 / not_ready 20,009；catalog 重建 332 组 / 5,538,396 成员。
- 验收达成：新旧路径一致性回归测试在位；真实数据 `build_inventory` 实测 2.43s
  （含 PG 连接超时，SQL 段 ~0.35s；原 8 min，约 200×）。

### 阶段 C：IAA batch=8（吞吐 2.37×）

- [x] `OneAlignRunner` 暴露批量接口（内部 `_score_pils` 已支持 list）。
- [x] `rank_candidates`：先逐候选 veto，再把非 veto 候选 + 源图（≤9 张）一次前向；
  veto 不打分语义不变。
- [x] batch 上限 = TOML 键 `[render] iaa_batch`（默认 8，1=回滚）；不用环境变量。
- 验收：单实例 ≥20 张/s；首轮生产后抽 200 组 top1 变更率 >1% 则退 batch=4。

### 阶段 D：存储根拆分 + land 集成 + 预取

- [x] **D1** IO 事项以 IO 文档 §1.2/§1.5 为准：`output_root` **整体留在 tmpfs**，
  仅把 5 个 toml 的值从 `/dev/shm/veradata/staging/<id>` 改为 `/mnt/ramstage/<id>`
  （专用挂载，nr_inodes=4m）。不新增 config 键、不拆 staging_root——
  candidates/masks/cgt 与四 JSONL + manifest 同根。资产是唯一一套物理文件；
  sft.jsonl 的 I_tar/C_GT 只是对 winner 资产的同路径引用（SFT 无独立 staging
  资产，I_in 指向 img 归档）。
- [x] **D1b 耐久化镜像**：每次 land 检查点把四 JSONL + manifest 原子同步到
  `/mnt/nfs/bc/data/builds/<id>/`（append-only 小文件，整文件覆盖拷贝即可，
  对应 IO 文档「NFS builds 层放 jsonl/manifest」）。resume 语义：进程崩溃
  （tmpfs 尚在）→ 按现状从 tmpfs journal 恢复；重启/掉电（tmpfs 清空）→ 先从
  NFS 镜像恢复 journal 到新 tmpfs 根再 resume。丢失窗口 = 最后一次 land 之后的
  记录与未 land 资产；resume 资产完整性检查把资产缺失的组记 `group_assets_lost`
  终态、补抽替换源，其已写出的 SFT 行随之失效（sft_pack 与 manifest 按该事件
  排除；jsonl 为 append-only 不回改）。阶段 E 的 sft tar 从已 land 的 groups tar
  派生，不在丢失窗口内。
- [x] **D2** land 集成进 agent：检查点内联在每源之后（组间边界，永不与在渲组
  竞争）按水位 16 GiB 触发 + 阶段末强制一次；只 land「资产仍在本地」的已 commit
  组（land 后 unlink，landed 自排除，**零新增 journal**）；同一检查点执行 D1b
  镜像；clean 回收孤儿资产；标注前一次 catalog 刷新使 landed 组可反查。
  groups.jsonl 内路径保持 staging 原值（逻辑键，读取走 read_bytes）。
- [x] **D3-工具** `tools/prefetch.py` 按 (shard, offset) 顺序预取源图到
  `staging_root/prefetch/<sha256(path)>` 双缓冲；`read_bytes` 增加可选 prefetch
  目录查找（启动时传入，非 env）。
- [x] **D3-接线**：`_SourcePrefetch` 双缓冲（PREFETCH_CHUNK=256，块 k 渲染时后台预取 k+1）；
  sft 数据集补 I_in 成员（read_bytes 物化，prefetch 命中零成本）；失败只计数
  （manifest["prefetch"].errors）不入 journal。
- 验收：渲染段 NFS 读趋近 0；50k build 墙钟 ≈ IAA 段；kill/resume 无重复。

### 阶段 E：输出侧双写（IO 文档 §4 第 4 步为准）

- [x] `dataset_plan.write_group` 支持 `preserve_order=True`。
- [x] land 双数据集发布：每个 land 检查点从同一 staging 先后发布两个数据集
  （两次 `land(..., keep_staging=True)`，都成功后再清 staging）：
  `groups/<build_id>`（全部 8 候选 + C_GT + 逐候选 QA）与 `sft/<build_id>`
  （winner 的 I_in + I_tar + C_GT + .vrmeta.json）。**winner 在 QA 时点已确定**
  （winner_ids 随组 commit 进 groups.jsonl），无需等标注；I_in 从 D3 预取缓冲取
  （字节已在 tmpfs，双写不多一次磁盘读，与 IO 文档论证一致）；标注文本活在
  sft.jsonl（D1b 镜像入 NFS builds），不进 tar。
- [x] 标注失败的 winner：tar 成员已写入不回收；标注阶段结束后把逐 winner 的
  标注状态同步进 `sft/<id>` 的 `metadata.jsonl`（归档契约本就允许 land 后外部
  同步），训练/读取侧以 sft.jsonl（或 metadata 标记）过滤。
- [x] `tools/sft_pack.py` 保留为**重建工具**（权威约定：中间产物 tar 权威、
  SFT tar 派生可再生）：从 groups tar + sft.jsonl 顺序回读重打 `sft/<id>`，
  供漂移修复或历史 build 补打。
- 验收：同一 winner 双 tar sha256 一致；`groups/<id>` 同组 8 候选 offset 连续；
  `sft/<id>` 样本数 == winner 总数、sft.jsonl 行数 ≤ 之且 join 后即训练集。

### 阶段 F：冷启动尾项

- [x] OneAlign 权重转 safetensors（2026-07-27 已执行）：`convert_onealign_safetensors`
  逐字节读回校验 + GPU 打分逐位一致（BITWISE_IDENTICAL）；HF from_pretrained 自动
  优先 safetensors，iaa.py 零改动；分片加载实测 **36.5s → 4.3s**。回退 = 删
  safetensors 三个文件。
- [x] preflight 前向可关：`[render] qa_preflight_forward`（false 时 OneAlign 加载
  推迟到首次 QA；CUDA 断言与 SDK preflight 保留）。
- LUT npz→npy+mmap：**可选**，仅 legacy 工具需要时做。

### 阶段 G：canonical 遗留缺陷

- [x] **G1 (P1)** 外部池耗尽边界：`_local_rescue_due` 谓词（从持久 attempt 事件
  重建，resume 安全）在最后一轮为发现耗尽的任务补一次本地 attempt；
  `local_fallback=false` 时补 `external_pool_exhausted_skip` 审计事件。
- [x] **G2 (P2)** `databuild.example.toml` 逐字段注释 + `[legacy_import]` 示例；
  修 `test_canonical_foundation.py` 中 output_root 死替换导致的恒真断言。
- [x] **G4a (P3)** 实况修正：`perceptual_de.jsonl` 7778 行全部只有 `de_med`、
  无 `de_mean`——归档 canonical 文档 §5.2 的 "mean ΔE00" 是笔误，保真判据实为
  中位 ΔE00，`presets.py` 改为诚实只读 `de_med`（行为逐位不变）；viewer README
  已补 `npm run test:e2e`；空 `frontend/src/tabs/` 已删。
- [x] **G4b (P3)** 已并入质量优化轮：`VisibilityError` → 独立错误码
  `visibility_invalid_weights`（agent.py 错误映射，硬失败+换 preset 语义不变）。
  （`VERADATA_CATALOG`/`VERA_ONEALIGN_MODEL` 收 TOML：暂缓，YAGNI。）

### 阶段 H：渲染段 CPU 并行化（2026-07-27 实测新增，评估闭环后立即执行）

背景：eval100 实测渲染段 7.7s/组（466 组/h），构建进程单核 100%、双 H100 采样
0%——串行候选循环里的 CPU 段（512p CIEDE2000 numpy + Lab hints + JPEG q95 编码
+ 解码）把 GPU 饿死。50k 生产按此速率 4.5 天，不可接受。

- [x] **H1 正式剖析表**：`profile_render_segments.py` 对 durable global/local
  1024p LUT 组做 warmup=2、iterations=5 的 per-candidate 分解。global critical
  1168.97ms（QA rank 62.8%、postprocess 19.1%、renderer 8.1%），local critical
  2010.50ms（QA 60.0%、postprocess 12.8%、masked renderer 9.4%、source 14.1%）。
  renderer 热 lookup/operator 仅 0.113/2.701ms，batched D2H+sync 83.282ms；输出见
  `/tmp/h1-profile-formal-final-20260728.json` 与
  `/tmp/h1-profile-local-full-final-20260728.json`。
- [x] **H2 A+B**：8 槽先批量 GPU render，再以 16-worker
  `ThreadPoolExecutor` 做 visibility/hints/JPEG；CIEDE2000 torch backend 保留 numpy
  oracle/fallback，门决策 parity 全绿。renderer 增加同源 `render_many()`，上传一次
  source、按 preset 顺序执行、统一 host transfer；两轮 GPU wave 受 TOML
  `gpu_concurrency=2` 控制。
- [x] **H3 C（有序跨组流水）**：两个 source worker 可并行预处理；selector turn
  严格按 allocation order 串接，在前组 8 个候选 accepted 后、IAA 前交棒，因此
  组 k 的 IAA/QA 可与组 k+1 的渲染重叠，同时 major/minor/preset 语义与串行一致。
  worker failure 会落 `source_worker_failed`，释放 reservation，并放弃 allocation
  缺口之后的 deferred group。
- [x] **H4 语义验收**：全量测试 `261 passed, 6 warnings, 72 subtests`；串行 oracle
  与 speculative slot retry 比较 preset/reservation/failure/selector snapshot；真实
  10-group smoke 的 coverage position 为 global `0,1,2`、local `0..6`。真实
  SIGKILL 命中 2 个 durable group + 3 个 partial JPEG，原 config resume 后 prefix
  SHA-256 保持、10/80 唯一、7/3 目标与 indexed tar verify 全通过。
- [ ] **H5 吞吐红线**：仍未通过并暂停。final production smoke
  `h1-real10-final-window3-20260728` 初始 rendering 25.275s 产出 8 组
  （1139.5 groups/h），warm gap median/p95=0.841/1.957s；GPU0/1 非零采样
  8%/16%。最终 10/80、精确7/3、journal hashes 一致，groups/SFT indexed-tar
  216/52 members 均 verify。相较旧有序 C 已明显提速，且 H1 表明 QA rank 约占
  60%，但仍不满足“多数 GPU 采样非零”与 ~9.9k groups/h 参照，不能宣称完成。

> 标注质量审阅（eval100）与阶段 H 的详细执行手册见
> `docs/DATABUILD_EVAL_AND_PERF_HANDOFF_2026-07-28.md`（自足，供新 agent 接手）。

## 三、总验收

归档版 canonical 文档 §16 全部保持成立，叠加：冷启动 < 1 min；50k build 墙钟 ≈
IAA 段；渲染段 NFS 读趋近 0；重启 resume 给出显式 `group_assets_lost` 账目；
`groups/<id>`、`sft/<id>` 归档 verify 通过。

当前暂停状态：H1/H2/H3/H4 已完成，H5 与冷启动 `<1 min` 尚未完成；不得把上段
目标描述当作已验收终态。
