# 标注质量审阅 + 渲染段 CPU 调优 · 交接文档（2026-07-28）

本文自足：接手 agent 不需要之前会话上下文。两个任务相互独立，可并行。
背景权威：`docs/DATABUILD_IMPLEMENTATION_2026-07-27.md`（执行状态清单）、
`docs/archive/DATABUILD_CANONICAL_REFACTOR_2026-07-20.md`（流水线契约）、
`docs/archive/DATABUILD_IO_REFACTOR_2026-07-27.md`（IO 实测）。

> **终态覆盖说明（2026-07-28 09:45）**：下方 §0 保留 00:30 的接手快照供追溯；
> 当前 eval100、review export、annotation A/B、A+B+C 实现、真实 10-group smoke 与
> SIGKILL/resume 均已结束，详见 §1.6、§2.5–§2.8。H1 正式 global/local 分解表、
> 双 scorer 与三轮 production smoke 已完成；语义/恢复验收继续成立。H5 的 GPU
> 利用率与 ~9.9k groups/h 吞吐红线仍未通过，因此暂停点不得标为性能完成。
>
> **第二轮终态覆盖（2026-07-28 晚）**：§4 记录当日下午的性能战役（WP1 归因 →
> WP2a/2b/2c 修复 → 对抗审查 → 两轮真实 smoke 验证）。~9.9k 红线数字确认作废
> （8+1 双 forward 修正），修正后红线（稳态 ≤ IAA 上限×1.15）已通过；渲染段
> 整体 1449.6 → 3860 组/h，30 组端到端 ~830s → ~150s。剩余瓶颈是 SAM3 relabel
> 同步停顿与 SAM3 环境缺陷（§4.5），需用户决策。§1.6 的标注质量结论经方法学
> 复核不可信，可信重评终态见 §5。

## 0. 当前状态快照（2026-07-28 00:30）

- 全部代码未提交（21 文件修改 + 10 个新文件）；测试基线 **220 passed + 72
  subtests**（`$PY -m pytest dataset_build/tests/ databuild_viewer/backend/test_app.py -q`，
  `PY=/home/bc/.venvs/iaa437/bin/python`，项目已 editable 安装，无需 PYTHONPATH）。
- **eval100 构建**：`databuild.eval100.toml`（0600，勿改权限），build_id
  `eval100-annotqa-20260727`，100 组（70 local + 30 global）已渲染完成并 land 到
  NFS（`/mnt/nfs/bc/data/datasets/{groups,sft}/eval100-annotqa-20260727`），
  winner 151 个。**标注段 resume 正在跑**（PID 2435024；完成判据 =
  `/mnt/ramstage/eval100-annotqa-20260727/manifest.json` 的 `completed.sft` ≈ 151、
  `annotation.terminal_failures` ≈ 0）。若失败，读该目录 `failures.jsonl` 的
  annotation 事件定位（错误码/endpoint/message 齐全）。
- 本次已修的三个真实环境缺陷（都有回归测试，接手者不要回退）：
  1. catalog rebuild 后必须 `archive_reader.invalidate_shared()`（immutable=1
     钉死快照；`agent._refresh_catalog` 已调用）；
  2. SAM3 relabel 会重建本机 `subject_cache` 稀疏树 → `build_inventory` 归档底座
     + 本机覆盖合并（`sources.py`，`local_overlay` 计数）；
  3. `sources.subject_cache` 配置不再要求本机目录存在（逻辑键）。
- 曾做过一次 journal 手术：run1 因缺陷 1 产生的 151 条
  `annotation_image_invalid` 假 terminal 已从 failures.jsonl 滤除（备份在会话
  scratchpad，已随会话失效；如需重做见 §1.4 的手术方法——它就是重标机制）。
- **relay 实况（2026-07-28 实测）**：gpt-5.6-luna 只有 provider-b 两条 lane 真正
  可用；provider-c 报 404 model_not_found（4xx 按契约直接终态）、provider-a 持续
  吐非标准流事件烧传输预算。`databuild.eval100.toml` 已删掉 a/c 三条 lane。
  换 annotation/审阅模型前先用单条请求验证 lane 是否供该模型。注意：改 lane 会
  使 resume 的 effective_config 校验失败——手术方法 = 删 output_root 与 NFS
  mirror 两处的 manifest.json（journal 是权威，manifest 会重写）。

## 1. 任务一：100 组标注质量审阅（用户明确要求的流程，勿删减）

### 1.1 前置：导出审阅包

构建完成后运行：

```bash
$PY dataset_build/tools/export_annot_review.py
```

产出 `/var/cache/veradata/annot_review/eval100-annotqa-20260727/`：每组一个目录
`gNNN_<gid12>/`，含 `before.jpg`、`after_rank1.jpg`（有 rank2 则同名）、
`cgt_rankN.png`（local 组）、`annot.json`（instruction/instruction_short/
reasoning/task_type/qa 分数/style_name/subject/region/slot_mode）。脚本内部走
`archive_reader.read_bytes`，资产在 tmpfs 或已 land 都能取。

### 1.2 审阅执行（用户定案的编排，必须照做）

- **10 个审阅 subagent，每人 10 组，一次并行 5 个（两波），审阅模型
  `gpt-5.6-sol`（reasoning effort `xhigh`）**：经 `databuild.eval100.toml` 里的
  relay lane 调 `/v1/responses`（同标注通道，before/after/C_GT 以 JPEG data URI
  随请求发送），subagent 负责组织请求与落盘 verdict。
- 每个 agent：用 Read 工具看图（before/after/C_GT）+ 读 annot.json，逐样本按
  §1.3 评分；每组从其 SFT 样本（≤2 个）中选出**最差的 1 个**；把该样本的完整
  文件集（before/after/cgt/该样本的标注 json + 一份 verdict.json 评语）复制到
  `/var/cache/veradata/annot_review/eval100-annotqa-20260727/worst/<组目录名>/`；
  返回结构化 JSONL（每组一行：group_id、逐样本各维度分、worst_sft_id、理由）。
- 全部完成后主控在**仓库根**建符号链接目录：`ln -s
  /var/cache/veradata/annot_review/eval100-annotqa-20260727/worst
  /home/bc/VeraRetouch/_annot_worst`（已在 .gitignore 的 `_*` 约定外——记得
  确认 gitignore，不入 git）。
- **主 agent 后续**：汇总 verdict → 聚类失败模式 → 给出优化方案（改
  `responses.py` 的 prompt/hint，见 §1.4）→ 用重标机制对失败样例重跑 → 把优化
  前后对比样本呈给用户审阅。

### 1.3 评分标准（源自 annotate v4/v4.1 定案，见项目 CLAUDE.md）

逐样本 5 个维度，各 1-5 分（5 好），并给一句话理由：

1. **图文一致**：instruction 描述的诉求与 before→after 实际变化匹配（方向、
   区域、幅度）；style 任务 instruction **必须点名风格名**（中文名 verbatim 嵌
   英文句合法）；local 任务**不得点名 preset/风格名**，区域用画面内容指认且与
   C_GT 位置（annot.json 的 region 九宫格）一致。
2. **reasoning 对齐**：六段 token（`<problem_*_start>` ×3 + `<plan_*_start>`
   ×3，顺序 light/global_color/specific_color）结构完整；problems 描述的是
   before 图真实可见的问题；plans 与实际编辑一致（方向不能反：提亮/压暗、
   暖/冷、增/减饱和）。
3. **无内部泄漏**：不得出现 LR 参数键名、Δ数值、IAA/q 分数、mask/geometry/α、
   degrade/auto 字样、候选槽位信息。
4. **语言与自然度**：英文输出（风格名例外）；instruction 像真实用户口吻、
   long/short 一致且 short 是凝练不是截断；persona 不僵硬。
5. **整体可用性**：作为 SFT 训练样本收录是否合格（pass/borderline/fail 落到
   5/3/1 锚点）。

worst = 总分最低者；同分取维度 1（图文一致）更差者。

### 1.4 重标机制（已验证的 resume 语义，不要另造工具）

对要重标的样本：从 `sft.jsonl` 删除对应行（备份原文件），必要时从
`failures.jsonl` 删除该 task 的 terminal 事件 → 重跑
`$PY -m construct.agent run --config /home/bc/VeraRetouch/databuild.eval100.toml`
→ resume 只把缺失的任务重新入队标注（渲染/QA 全跳过）。改 prompt 后重标即
A/B：改动点集中在 `dataset_build/src/construct/responses.py`（system prompt
~84-90 行、task 子句 `_task_clause` ~294-321 行、hints 注入、schema 37-67 行）
与 `visibility.py::objective_edit_hints`（29-81 行）。注意 resume 会校验
manifest 里的 effective_config——**改 prompt 常量不影响 config**，可安全 resume；
若改了 TOML 键则需新 build。

### 1.5 已知边界（审阅时注意，不算标注错误）

- annot.json 的 qa.onealign/q 是 IAA 排序分，仅供参考，不是审阅对象。
- global 组无 C_GT；local 组 semantic 两槽共享同一 C_GT。
- 渲染图短边 1024、JPEG q95；审阅看图以内容为准，不评压缩。

### 1.6 完成记录（2026-07-28 终态）——**本节结论已被 §5 的可信重评取代，仅留档**

- eval100 终态为 100 groups / 800 candidates / 151 SFT；review export 已修为稳定
  `group_id` 排序的 100 个唯一目录，19 个无 winner 组显式标为 `no_sft`，不伪造
  分数。最差样本集链接为仓库根 `_annot_worst`。
- 首轮审阅识别出 44 个 fail，主因是 local 文本把 broad/linear/band 区域误写成
  单一人物或物体，以及编辑幅度/方向描述不足。prompt-v2-local-regional 对全部 44
  个重标后得到 18 pass / 21 borderline / 5 fail；prompt-v3-hint-magnitude 只重跑
  这 5 个，得到 3 pass / 2 borderline / 0 fail。
- 最终 A/B：44 fail → **21 pass / 23 borderline / 0 fail**。均分变化：一致性
  1.409→3.705，reasoning 2.432→3.932，泄漏 5.000→4.977，语言
  4.477→4.364，整体可用性 1.114→3.795；一致性/reasoning/usability 分别有
  42/41/44 个改善，三项均 0 regression。
- 审计产物在
  `/var/cache/veradata/annot_review/eval100-annotqa-20260727/reannot_ab/`，仓库根
  `_annot_ab` 指向该目录；`_annot_ab` 已显式加入 `.gitignore`。

## 2. 任务二：渲染段 CPU 瓶颈调优（阶段 H，用户定案 16-32 并行）

### 2.1 实测基线（eval100，2026-07-27）

| 项 | 值 |
|---|---|
| 出组速率 | 7.7s/组（466 组/h），50k 外推 4.5 天，不可接受 |
| 构建进程 | 单核 100% 长期占满 |
| GPU（H100 ×2） | 采样时刻 0%（IAA 驻卡0 17.5G、渲染卡1），在等 CPU |
| NFS | 无压力（预取双缓冲已工作，`manifest["prefetch"]` errors=0） |
| IAA | batch=8 已上线（2.37×），不是瓶颈 |

串行热路径（`agent.py::_render_source` ~501-744 行，每组 8 候选依次）：
GPU 渲染（快）→ `visibility.compute_visibility`（**512p 全量 CIEDE2000，纯
numpy 含 G/RT 项**，visibility.py:140-225）→ `objective_edit_hints`（Lab 统计）
→ `save_candidate_jpeg`（1024p q95 编码 + fsync，rendering.py:371-382）→
C_GT PNG（组内一次/掩码）。源图解码经 `preprocess_source`（预取命中，非瓶颈）。

### 2.2 H1：先剖析，出表再动刀

- `pip install -i https://pypi.org/simple py-spy`（tuna 镜像 403，必须官方源）
  进 iaa437；对运行中构建 `py-spy top --pid <pid>` / `record -o profile.svg`。
- 或离线微基准：用一组真实 1024p before/after float32，分别计时
  `compute_visibility`、`objective_edit_hints`、JPEG 编码、（对照）GPU 渲染一次
  的墙钟。产出表：每候选各段 ms 与占比。**先有这张表再选方案。**
- 跑一个小构建做对照（`databuild.local10.onealign.toml` 或复制 eval100 改
  target_groups=10 + 新 build_id + 新 output_root；`qa_preflight_forward=false`
  可省预热）。

#### H1 正式结果（2026-07-28 09:20）

可复现工具为 `dataset_build/tools/profile_render_segments.py`。它从 durable journal
选择真实组，经 indexed-shard reader 读取 source/local C_GT，在有界
`/mnt/ramstage` scratch 重放；不改 journal、manifest 或 NFS。两组均为
1024 short-edge、8 个不同 LUT、H100 `cuda:1` render、OneAlign `cuda:0`，
warmup=2、iterations=5。权威输出：

- global：`/tmp/h1-profile-formal-final-20260728.json`；
- local（最终 masked renderer）：`/tmp/h1-profile-local-full-final-20260728.json`；
- batch9 parity：`/tmp/h1-profile-batch9-parity-20260728.json`；
- 双 scorer 初始探针：`/tmp/h1-onealign-concurrency-20260728.json`。

| critical stage | global median / p95 ms | global 占比 | local median / p95 ms | local 占比 |
|---|---:|---:|---:|---:|
| source decode + resize | 51.021 / 53.026 | 4.4% | 283.075 / 283.608 | 14.1% |
| working reference | 64.806 / 73.529 | 5.5% | 74.260 / 79.855 | 3.7% |
| visibility reference | 0.652 / 0.662 | 0.1% | 0.646 / <1 | <0.1% |
| production renderer | 95.188 / 95.524（一波） | 8.1% | 189.281 / 203.830（masked 两波） | 9.4% |
| parallel postprocess（含 QA stats side-channel） | 223.591 / 247.462 | 19.1% | 257.173 / 290.608 | 12.8% |
| cached QA rank | 733.717 / 742.382 | **62.8%** | 1206.065 / 1220.940 | **60.0%** |
| critical sum / 推算 | 1168.975；3079.6 groups/h | 100% | 2010.500；1790.6 groups/h | 100% |
| full group replay（非相加估算） | 1294.004 / 1402.241 | - | 2063.336 / 2094.209 | - |

global renderer 子分解：冷装 8 LUT 为 88.993ms（只在冷路径）；热 cache lookup
0.113ms、source upload+sync 7.026ms、8 LUT operator+sync 2.701ms、batched
host transfer+sync 83.282ms。即 host transfer 是 renderer 内主项，但 renderer 只占
整组 critical path 8.1%，继续优化 LUT kernel 不是总吞吐主抓手。

`iaa_batch=8` 实际收到 **source + 8 candidates = 9 paths**，因此是 `8+1` 两次
forward，旧的 9.9k groups/h 参照漏算了第 9 张。batch9 只作为 profiler override：
固定组 rank order、winner IDs 完全相同，max score delta=0；生产 schema/config 仍按
公开契约保持 1..8。两个独立 scorer 的 fixed-global 探针为 460.777ms/group，
fixed-local 为 648.070ms/group；dual-scorer max score delta=0。两实例 CUDA allocated
33,022,026,752 bytes，reserved 约 34.94GB。aggregate batch18 有 0.097656/0.048828
score drift，故未采用。

### 2.3 H2：方案（按剖析结果取舍，可叠加）

- **A（首选）候选级线程并行**：8 候选的 CPU 后处理（可见性门 + hints + 编码
  落盘）提交 `ThreadPoolExecutor`（起步 16，用户上限 32）；numpy ufunc 与
  PIL 编解码释放 GIL，实测过再定 worker 数。GPU 渲染调用保持既有信号量语义
  （`RENDER_GPU_CONCURRENCY` 语义不变）。**约束**：journal append 顺序与候选
  slot 顺序、attempt 事件序必须与串行一致（收集结果后按 slot 序串行落账）；
  visibility 拒绝→同槽换 preset 的重试循环仍按槽串行（只并行「已渲候选的后
  处理」，不改选择器语义）。
- **B CIEDE2000 迁 GPU**：torch 实现（与渲染同卡同流），金标 = 对
  `test_canonical_foundation.py:986-1037` 的合成场景与随机图，`visible_de`/
  `visible_fraction` 与 numpy 差 ≤1e-4 且**门决策完全一致**；保留 numpy 为
  fallback/test oracle。若剖析显示 CIEDE2000 占比 >60% 则 A+B 都做。
- **C 跨组流水**：渲染组 k+1 时后台做组 k 的 QA/落盘。改动面最大，只有 A+B
  仍不达标才考虑。
- 不做：换 JPEG 库（新依赖）、多进程（序列化成本 + journal 竞争）。

### 2.4 验收（用户红线：CPU 绝不能成为瓶颈）

1. 渲染段 GPU util 持续显著非零（`nvidia-smi` 采样多数时刻 >0）；
2. 出组速率至少与 IAA 吞吐匹配（单 IAA 实例 ~9.9k 组/h 量级为参照；最低要求
   = 墙钟由 GPU/IAA 决定而非 CPU）；
3. 可见性指标与门决策和串行实现一致（新增 parity 测试）；
4. kill/resume 语义不变（现有 `test_land_integration.py` /
   `test_canonical_orchestration.py` 全绿）；
5. 全量测试 ≥220+72 不回归；不加环境变量，新参数一律 TOML 键（参考
   `iaa_batch` 的做法：config.py render_keys + dataclass 尾默认 + example 注释
   + 4 个本机 toml 同步）。

### 2.5 已实现路径与 parity（2026-07-28）

- A：候选后处理使用 16-worker `ThreadPoolExecutor`；GPU render 结果仍按 slot 顺序
  resolve，visibility rejection 的 failure/attempt 顺序与串行一致。
- B：visibility/hints torch backend 上卡，numpy 保留 oracle/fallback；合成与随机图
  的指标容差和门决策 parity 全绿。
- renderer batch：`LocalGpuOnlyRenderer.render_many()` 对同一 source 只上传一次，
  preset 保序执行后批量 host transfer；无 mask LUT 合为一波，masked LUT 保留两波；
  alpha==0 exact endpoint 检查去掉布尔索引大拷贝。`render()` 委托一项 batch，
  capability 在上传前 fail-closed。
- QA：候选 postprocess 在 JPEG fsync 后计算非 durable `_qa_stats`，ranking 复用并在
  journal 前删除；两个独立 OneAlign 实例用有界 pool 驻 `cuda:0`。单实例回退为
  `gpu_concurrency=1`，未新增 TOML/env knob。
- C：两个 source worker 先并行预处理，但 selector turn 由 allocation order 串接；
  前一 source 的 8 个 candidate accepted 后、IAA 前才交棒。因此 QA(k) 与
  render(k+1) 可重叠，而 coverage/preset 不受线程抢锁次序影响。
- production dual-scorer 下 source window 为 3，避免两个 worker 同时卡在 QA 时饿死
  下一 source；其他注入式依赖保留原窗口。小目标 prefetch 预算为 remaining + 1 个
  replacement，50k 仍按 256 块顺序流。两实例在 annotation 前从约 32GB 显存降到
  约 1.26GB，外层 `run()` 不再错误持有 scorer/renderer 强引用。
- parity 证据：serial oracle 与 speculative 路径在 slot0/slot3 首次 reject 场景下，
  preset ID、reservation ID、failure event 顺序/ID、attempt lineage、selector
  snapshot 和 active-counter 清理一致。全量测试最终为
  **261 passed, 6 warnings, 72 subtests**。

### 2.6 真实性能矩阵

各 smoke 使用不同 `build_id`，因此 deterministic source allocation 也不同；下表用于
生产形态诊断，不把小样本差异误当作严格同源 A/B。

| 路径 | rendering 观测 | warm gap | GPU0/1 非零采样 | 终态 |
|---|---:|---:|---:|---|
| A+B threaded/torch | 10组 / 83.600s = 430.6组/h | 2.498s | 10.7% / 9.5% | complete |
| + renderer batch | 初始9组 / 76.257s = 424.9组/h | global 1.453s；local约2.36s | 7.9% / 3.9% | 10组 complete_with_failures |
| + two GPU waves | 初始9组 / 72.875s = 444.6组/h | global 1.606s；local约2.67s | 9.7% / 6.9% | 10组 complete_with_failures |
| C（未有序原型） | 10组 / 75.048s = 479.7组/h | 2.156s | 10.5% / 7.9% | complete；global position `1,0,2`，拒收 |
| C（allocation-ordered） | 初始8组 / 65.442s = 440.1组/h | 1.230s | 12.1% / 6.1% | replacement后10组 / 112.342s = 320.5组/h |
| + dual scorer，旧 256 overfetch | 初始9组 / 69.974s = 463.0组/h | - | 2.9% / 7.1% | prefetch buffered=1131；10组终态 |
| + bounded prefetch | 初始7组 / 26.475s = 951.6组/h | - | 11.5% / 19.2% | prefetch buffered=14；10组终态 |
| final：window3 + masked renderer/endpoint | 初始8组 / 25.275s = 1139.5组/h | median 0.841s；p95 1.957s | 8.0% / 16.0% | prefetch buffered=13；10组终态 |

final build `h1-real10-final-window3-20260728` 的 coverage position 为 global
`0,1,2`、local `0..6`；10 groups / 80 unique candidates / 14 SFT，精确 7/3，
`groups_assets_lost=0`，SAM3/annotation pending=0。tmpfs/NFS 的 groups/SFT/failures
journal SHA-256 分别一致；groups indexed-tar 为 216 members，SFT 为 52 members，
两者 `shard_dataset verify` 通过。`complete_with_failures` 来自 3 个 terminal SAM3
source，由 replacement 补齐目标，不是 shortfall。

bounded prefetch 把 buffered 1131 降到 13（-98.9%）；source window3 消除了之前
成对出现的 2.55s warm 空档。final 初始 warm gap median 0.841s（约 4283 groups/h）、
p95 1.957s。但完整 rendering 仍只有 1139.5 groups/h，GPU0/1 的 1 秒采样非零比例
仅 8%/16%，仍低于多数非零与 ~9.9k 参照。因此 **H1 完成，H5 未通过并在此暂停**。
正式 profile 已证明原 CPU visibility/JPEG 串行热点不再占主导（QA rank 约60%），
但 mode startup、large-source decode/resize 与 local QA 仍使端到端红线不成立。

### 2.7 真实 SIGKILL/resume

- build `resume10-pipeline-ordered-20260728` 在 `rendering` 中直接 SIGKILL Python
  PID 3816878（exit 137）；信号前已有 2 个 fsync group / 16 candidates，且下一组
  有 3 个 candidate JPEG 已落盘但未进 journal。
- kill 前两行 `groups.jsonl` SHA-256 为
  `1a9646162ee76bd2584ad1725876174cd35ee0fa29115e9d3b94d0b04943c601`；用完全相同
  config、无 `--resume` 重启后，最终 journal 前两行 digest 完全相同，prefix 只出现
  一次，3 个 partial 文件全部清理。
- 终态 10 groups / 80 unique candidates / 15 SFT，精确 7 local + 3 global，
  `groups_assets_lost=0`，local/global shortfall 均 0，SAM3/annotation pending 均 0。
  tmpfs/NFS 四 journal digest 全匹配；groups dataset 216 members、SFT dataset
  56 members，两个 uncompressed indexed tar 均 verify 通过。

### 2.8 尚未解决的系统缺陷

- 每次新 build 与 resume 在 output 创建/恢复前仍扫描完整 preset bank，实测约
  5.5–6.5 分钟；phase-aware resume 没有跳过已经完成的 preflight。
- landing 后进入 `annotation` 会同步重建约 6.4 GB
  `/var/cache/veradata/global.sqlite3`，约 6–7 分钟内没有 relay 请求。不能直接跳过，
  landed logical path 依赖 catalog refresh；需要 freshness/digest 协议而非 ad-hoc bypass。
- 只读审计另发现 landing 的 groups/SFT 两次发布与单一 checkpoint event 之间没有
  crash transaction：若恰好在两次 publish 之间死亡，resume 可能重新分配 batch。
  这是独立 P1，当前 mid-render SIGKILL 不覆盖；后续应以可恢复 publish intent/commit
  journal 修复后，再做 landing 临界点 kill injection。
- staged-byte 水位未计 `.land/sft` 的 I_in 临时副本，且 lost group 的幸存资产可能
  被 orphan sweep 继续视为 referenced；两项应在 50k build 前修复并加水位/丢失资产测试。

## 3. 本机踩坑清单（两任务通用，血泪换来的）

- zsh `noclobber`：覆盖已存在文件用 `>|`；
- `pgrep -f`/`pkill -f` 会匹配自己的命令行（等待循环自锁/盯错 PID）；
- 长任务必须用会话托管的后台方式启动，shell `&` 会随工具会话被清理；
- catalog rebuild 后同进程必须 `invalidate_shared()`（已内置，写新代码勿绕过）；
- pip 一律 `-i https://pypi.org/simple`（tuna 403）；
- `/mnt/ramstage` 24G tmpfs 掉电即失；NFS 1GbE 98MB/s 硬顶；
- construct 全链路只用 `/home/bc/.venvs/iaa437/bin/python`。

## 4. 第二轮性能战役终态（2026-07-28 晚，WP1–WP4/WP2c）

### 4.1 归因（WP1，30 组 attrib run + py-spy 20Hz + journal 时间线）

三级落差全部定量解释：1139.5→4032 组/h 全是一次性税（cold fill 9.2s、
global→local fill 串行 10.5s、SAM3 drain 31.7s——30 组 run 里 68.9% 墙钟零产出）；
4032→6082 是 source window 硬编码 3 且 95% 饱和（每组占用 source 线程 2637ms）；
6082 本身是双 OneAlign 上限（8+1 双 forward 修正后 global 460.8 / local 648.1
ms/组），**§2.4 的 ~9.9k 参照就此作废**。QA rank 的 592ms 中 CPU 侧占 59%
（CLIP preprocess 276ms 为大头），GPU forward 仅 284ms。

### 4.2 已落地修复（全部有回归测试；测试 220 基线 → 262 passed + 100 subtests）

| # | 修复 | 前 → 后 |
|---|---|---|
| B4 | prefetch `_locate` JOIN→CROSS JOIN（join-order barrier） | 5442ms → 0.6ms/次 |
| B2 | preflight 改查 luts_meta（保留 strict UTF-8 复检，24 个 GBK LUT 仍拒） | PresetCatalog.load 325s → 4.1s（暖）/19.9s（冷页缓存） |
| B3 | catalog 增量 `upsert()`（全量 rebuild 保留为 CLI 修复工具） | 刷新 390s → 18.8s |
| B6 | SAM3 批次不再卸载/重建 renderer+scorer（只 empty_cache） | 每轮 −11.5s；顺带消除 vGate supervisor 抢卡 OOM 竞态 |
| B7 | C_GT PNG 编码移出 selector-turn 临界区（postprocess 池 + mask_id 去重 + 交棒前屏障） | turn 752.8 → 495.3ms |
| B1 | source window 3→5，新 TOML 键 `render.source_window`（默认 5，范围 1..8） | 稳态 4032 → 5261 组/h |
| WP2c-b | QA 预处理（解码+expand2square+CLIP preprocess）8 线程并行，forward 不变，逐位 parity delta=0 | rank_candidates global 717.5→530.5ms / local 878.6→693.1ms |
| WP2c-c | scorer 实例数 TOML 键 `render.qa_scorer_instances`（默认 2；3 是拐点，4 无增益；1..4 校验；50k 建议 3） | QA 段 8287 → 8903 组/h（3 实例，46.1GiB） |
| schema | annotation schema_failed 改有界重试（累计 3 次，跨 resume 继承；4xx 终态契约不变，无退避） | WP4 丢 3/48 winner → 0 |
| P1 | resume 允许表 `_RESUME_NEUTRAL_DEFAULTS`：旧 manifest 缺语义中性新键时补默认值再判等；错误信息点名差异键 | eval100 重标 resume 解封（只读实测 diff=[]） |

另修（对抗审查 WP3 发现）：P2 upsert wanted 排序 + docstring 改真实不变式
（source_paths last-writer-wins，与 rebuild 可能分歧、字节相同）；P3 C_GT 写失败
短路（不再每 major 白渲 8 候选）；P5 SAM3 批 OOM 后 except 分支释放张量+
traceback；P7 三处 docstring/copyfile 修正；P8 window3 历史 toml 显式写
`source_window = 3`（其旧 manifest resume 会被拒，属预期）。

**否决项**（有实测依据，勿翻案）：QA 吃渲染内存像素（分数依赖 JPEG 后像素，
max delta 0.537，rank 会变）；batch9 合并 forward（第 9 张分数差 0.0488，
WP1 的 delta=0 在别的 fixture 上不成立）；换 JPEG 库/多进程（§2.3 既有决定）。

### 4.3 真实 smoke 验证（WP4 两个 30 组 + WP2c 三个 30 组）

| 指标 | 修复前（WP1） | WP4（B1..P1） | WP2c（+QA 并行+3 scorer） |
|---|---:|---:|---:|
| preflight（进程内实测） | ~340s | 26.3s | — |
| cold fill | 9.18s | 3.02s | — |
| 模式切换空档（B5 未修） | 10.46s | 5.20s | — |
| SAM3 drain 停顿 | 31.7s | 4.3s | — |
| 纯稳态 | 858ms/组 | 684ms/组 | **397–425ms/组** |
| 渲染段整体 | 1449.6 组/h | 3483.5 组/h | 3294–3860 组/h |
| GPU0 均值（渲染窗全窗/去 stall） | 13.3% / — | 17.5% / — | 18.6–19.4% / **43–44%** |
| 30 组端到端 | ~830s | 146.9s | — |

修正后红线（稳态 ≤ 592ms×1.15=680ms）**通过**；GPU0 去 stall 非零采样 71–96%。
SIGKILL/resume 比 §2.7 更严（双 kill 双 resume、零 manifest 手术、25/25 项、
含 B7 eager C_GT 孤儿清理与 P1 兼容路径）全绿。

### 4.4 新观测的既有缺陷（非本轮回归）

- B2/scorer 收益依赖页缓存：冷页缓存下 preflight ~20s、_load_scorer 37.6s，
  50k 首次启动按冷数字预算。
- `_landed_datasets` 每次刷新重登记本 build 全部批次（50k ~16 组/次，几十秒，
  可接受但随规模涨）。upsert 无并发写者保护（单机单构建成立）。
- window5 常驻内存最坏 ~950MB（5 source 在途）；50k 长跑盯一次 RSS。

### 4.5 下一轮性能战役（本轮明确不做，需用户决策）

1. **SAM3 relabel 同步停顿已成第一瓶颈**：渲染段墙钟 56–62% 是 stall
   （`sam3_subject_label_error`/`sam3_relabel_*` 同步调 VLM 打 subject label，
   单次 1.1–7.6s，双卡全闲）。应异步化/批量化/预判，但前置是下一条。
2. **SAM3 环境缺陷（P1）**：iaa437 的 transformers 4.57.1 无 `Sam3Processor`，
   `Sam3Masker._ensure_loaded()` ImportError 被吞成 `relabel_exception`——
   WP1 观测的每轮 11.5s 其实全是空转税。升级 transformers 可能破坏 OneAlign，
   需单独验证或独立 env，用户拍板。
3. B5 global→local fill 串行 ~5.2s（一次性，50k 上可忽略，YAGNI）。
4. §2.8 其余未动：landing 双 publish 无 crash transaction（P1）、staged-byte
   水位缺口（`.land/sft` I_in 副本 + eager C_GT 未入账 ~0.29MB/terminal source
   + lost group 幸存资产）、catalog freshness/digest 协议（upsert 只是最小形态）。

### 4.6 §1.6 标注质量结论为何不可信（WP5 重评的动因）

方法学缺陷六条：只对 44 fail 重审无对照组（回归均值未排除）；单评委单次无
重测方差；评委可见 CONTEXT 元数据（region/style 等 ground truth 泄漏给
consistency 维度）；审阅图降到 512 长边 q85；标注与审阅同族模型
（gpt-5.6-luna/sol）自偏好；IAA/winner 选择质量从未被评估（§1.5 明文排除）。

## 5. WP5 可信重评终态（2026-07-28 晚，取代 §1.6 结论）

方法：A 机械检查 239 单元（零 relay）→ B1 全量 151 盲评（零 CONTEXT、1024px、
盲化审计违规 0）→ B2 重测噪声基线 n=30 → B3 44 对新旧文本盲化配对（A/B 随机）
→ C 25 组×3 对 IAA 成对裁决。344 次 relay 请求零未恢复失败。产物
`/var/cache/veradata/annot_review/eval100-annotqa-20260727/wp5_reeval/`
（统计全量在 `wp5_report.json`）；工具 `dataset_build/tools/reeval_*.py`（5 个）。

### 5.1 核判结果

- **prompt-v2/v3 的改善方向是真的**：配对盲评 44:0 全胜（p=1.14e-13），三效应
  分解后净效应 consistency +1.773 / reasoning +1.386 / usability +1.909，为评委
  噪声的 3–5 倍。回归均值分量存在（usability +1.082）但不是主因——旧文本盲评
  仍 42 fail/2 borderline。
- **但「44→0 fail」不成立**：44 条单条盲评实为 18 pass / 20 borderline /
  **6 fail**；配对格式抬高绝对分。且首轮实为 46 fail，2 条 style fail 从未重标。
- **首轮更大的错误是漏杀**：CONTEXT 泄漏让评委照元数据盖章，首轮 pass 的 68 条
  盲评有 26 borderline + 4 fail；25 个盲评 fail 里 19 个来自从未重审的 107 条。
- **语言维度 v2/v3 变差**（旧版配对 18:2 胜，4.909→4.545）：准确性换了啰嗦。
- **数据集现状（151 全量盲评）**：58 pass / 68 borderline / 25 fail
  （38.4/45.0/16.6%）。local 仅 26% 直接可用；**band 槽是主要失效源**
  （6/11/11，consistency 2.93），semantic 最好（3.56）。
- **IAA（OneAlign）排序有效但有分辨力下限**：winner 总胜率 68%（p=0.0024），
  vs last 88%（p=1.6e-4），剂量-反应干净（分差 >10 时 92%）；但**分差 <2 时
  等于随机**（vs rank2 56%, p=0.69），而全库 45/99 组 rank1–rank2 分差 <1.0。
- **新发现 relay 契约缺陷（P0）**：provider-b-lane-2 约 21% 概率静默把
  gpt-5.6-sol 换成 gpt-5.5（且破坏 strict schema）；首轮审阅两 lane 轮询不校验
  `response.model`——§1.6 结论出自不受控双模型混合评委。非流式请求另会撞
  Cloudflare 120s/524。评审工具必须 `stream=True` + 校验返回模型
  （`reeval_relay.py` 已实现）。
- 机械检查全绿：六段 token/style 点名/泄漏/截断 0 违规（5 条正则命中均为词义
  误判，已进白名单）。

### 5.2 行动建议（待用户定夺）

1. 剔除或重标 25 条盲评 fail（ID 在 `wp5_report.json` 的
   `b1_failure_modes.fail_ids`）；
2. 68 条 borderline（45%）人工抽检定夺，决定能否开训；
3. prompt v4：不推翻 v3，只修语言啰嗦回退 + band/radial 槽几何描述（band fail
   率 11/28 最高），加长度上限；
4. IAA 加弃权门槛而非换 scorer：rank1–rank2 分差 <2.0 的组（约 45%）放弃
   winner 或 VLM 成对复裁；
5. 局限：relay 仅 GPT 族可用（无跨族双评委）、无人工锚点、C 分层各 25 对
   功效偏低。

## 6. v4 标注契约实验终态（2026-07-29，WP6/WP7）

用户 grilling 定案（D1–D12）后实施：七段输出契约（problems×3 →
`<region_scope_start/end>` → plans×3，global/style 退化句 "global adjustment
across the entire frame"）、几何 hints 双变体（v4a 离散描述子 / v4b 原始数值，
`responses.py` 的 `PROMPT_VARIANT` 常量切换，不进 config）、泄漏边界重写
（region_scope 允许几何类名词+方向词；数值/实现词/preset/slot/IAA/LR 键全域
禁止）、长度上限（instruction aim 60/cap 75，short aim 22/cap 30）、标注通道
模型钉死（`response.model` 前缀校验，仅 external 路由，`model_substituted`
retryable 3 次上限 + 换 lane；`annot_model` 审计字段；temperature=1.2/top_p=1.0
模块常量）。对抗审查修掉两个 P0：方向桶按 before 图真实长宽比换算视觉角
（错桶率 13.1%→0%）；v4b 数值回吐探测器补 snake_case/文字数字/percent 形态。

### 6.1 实验结果（预注册六门槛，wp7_v4/wp7_report.json 全量统计）

- **v4a 五条全过胜出**；v4b 门槛 1 两项不达标出局（band fail 4>3、配对 p 未达
  0.01），且 schema 违规率 7.5% vs v4a 1.3%，对撞无收益——数值 hints 路线关闭。
- 全量 151 盲评：pass 38.4%→**60.3%**，fail 16.6%→**5.3%**（91/52/8）；band
  consistency 2.929→3.857、fail 11→3；region_scope 全量均值 4.669、band 槽
  4.643/零 ≤2 分。泄漏 0（v4b 数值回吐也是 0）；长度超限 0。
- **保留意见（勿简化）**：band 配对分层显示 v4a 对 prompt-v1 旧文本 14:0 压倒，
  对已修过的 v2/v3 文本 9:5（p=0.42）未证明增益；language 全量微降 0.112，
  啰嗦被遏制未被修复（旧文本配对 16:8 领先但不显著）。
- eval100 终态：sft.jsonl = v4a 全量 151（tmpfs/NFS/快照三处 sha256 一致），
  manifest 自洽，annotation pending/terminal 均 0。备份齐全（*.pre-wp7）。
- **退役记录（2026-07-29，D11 方案 c）**：eval100 退役为内部 QA 夹具、永不进训练
  （prompt 已在其上调过 v2/v3/v4 三轮）；sft.jsonl 保持 151 行不删（journal 不
  伪造事件、landed metadata.jsonl 不被改写），8 条盲评 fail 以文档级黑名单记录，
  筛后有效 143 行；标记见 `/mnt/nfs/bc/data/datasets/{groups,sft}/eval100-annotqa-20260727/RETIRED.md`。

### 6.2 新发现的 relay 缺陷（P1 待修）

provider-b-lane-1 会用 `400 {"type":"upstream_error"}` 包装上游瞬时故障，
现行「4xx 一律终态」契约误杀（Pass 1 丢 12 任务，journal 手术 + resume 补齐）。
建议对 `type=="upstream_error"` 开有界重试口子。另：provider-b-lane-2 对
gpt-5.6-sol 有 ~21% 静默替换成 gpt-5.5（WP5 实测；luna 生产 175 抽实测 0），
一切评审/标注工具必须 stream=True + 校验 `response.model`。

### 6.3 v4.1 待办（不阻塞 50k）

方位词渗漏残余（4 条 v4a 的 instruction/reasoning 出现 lower portion/upper-right
类词，违反「画面内容指认」）；language 啰嗦的真正修复；机械正则 `masks` 动词/
`feather` 鸟羽两条 benign 白名单待补。

## 7. 出厂验证两轮与 50k 决策点（2026-07-29，WP8-WP12 终态）

### 7.1 已落地（全部有测试，终态 343 passed + 208 subtests）

- **D9 IAA 弃权门槛**：rank1-rank2 OneAlign 分差 <1.0 弃权、1.0-2.0 打
  `winner_confidence:"low"`、≥2.0 显式 "normal"；TOML 键
  `render.qa_winner_margin_{abstain,low}`（默认 1.0/2.0，已进 resume 允许表）。
  实测校准：fresh 批 abstain ~45%、low ~24% 的有 winner 组；**SFT 产出
  ≈0.75-0.8 行/组（较无门槛 −43%）**。注意：`low` 标签尚无下游消费者；
  `winner_top1/2` 监控口径需加上 `winner_abstained` 一起读。
- **传输加固**：`schema_failed`/`model_substituted` 强制换 lane（avoid 集合、
  回合内不清空）；4xx `type=="upstream_error"` 有界重试（3 次）。实效：fresh100
  的 schema terminal 5 → fresh200 **0**（lane-2 仍贡献 100% 的 schema 失败，
  但换 lane 后不再丢行）。
- **v4.1 颜色 hints 修订**（veto 框架 + 死区 1.0/2.0 + 反向锚定）与 WP11 诊断：
  hints 本就是 mask 内 alpha 加权且数值正确，失效是「空间均值≠观者所见」
  （高光截断、chroma 被中性像素稀释、warmth 只看 b*）。**α·max(C) 重加权已被
  实测否决，勿重试。**
- **D11 eval100 退役**：方案 (c) 零改动——sft.jsonl 保持 151 行不删（删行 + resume
  会经 `_sync_annotation_status` 反写已发布 metadata.jsonl），退役靠两个数据集
  目录的 RETIRED.md + 8 条盲评 fail 黑名单 + 本文档记录。eval100 永不进训练。

### 7.2 出厂验证判定（预注册门槛，两轮 NO-GO）

| 门槛 | fresh100（修因前） | fresh200（v4.1 后，n=160） |
|---|---|---|
| fail ≤10% | 10.67% ❌ | 10.62% ❌（超 1 条） |
| pass ≥50% | 49.33% ❌ | 52.50% ✅ |
| band fail ≤15% | 0% ✅ | 15.62% ❌（超 1 条） |
| 泄漏=0 / 七段 100% / 弃权校准 | ✅✅✅ | ✅✅✅ |

组成标准化后两轮**逐点重合**（pass ~49.5/fail ~10.4）——**v4.1 无 out-of-sample
增益**。fail 聚类：颜色/影调方向错误 82%；region_scope 类 29%（semantic 整群
mask ×3、linear 轴向 ×2）。门槛 1/3 的 CI 均横跨阈值（统计上与阈值不可区分），
但按预注册纪律判 NO-GO。**出厂预期以 fresh 批为准：pass ~50%、fail ~10.6%；
eval100 的 60.3% 是调优批含过拟合。**

### 7.3 待用户拍板的岔路（50k 前必须选）

1. **颜色方向失效的下一步**：prompt 措辞路线已证死（v4.0 vs v4.1 统计不可
   区分）。候选：(a) 在 fresh200 n=160 面板上对候选度量做 ROC 判别力筛选
   （14 方向类 fail vs 方向类 pass），选出能预测评委观感的统计量后改 hints
   度量（可能需 render 路径存新统计 → 只对新 build 生效）；(b) 接受现状，
   把 fail ~10% 当作 50k 的已知损耗（配合训练侧按 usability 过滤）。
2. **门槛定义**：两轮各差 1 条样本、CI 横跨阈值。维持字面判定（继续修因），
   还是按「CI 上界 ≤ 阈值×1.5」之类的统计判据重写门槛（需重新预注册）。
3. **SAM3 环境（§4.5 P1，仍未决）**：iaa437 缺 Sam3Processor，relabel 全程
   空转 → 源池损耗 ~19-20%/批 + semantic 整群 mask 疑似其下游代价（fresh200
   3 条 fail）。升级 transformers（可能破坏 OneAlign）vs 独立 env。
4. **linear 轴向桶残余**：fresh200 有 6 条 region_scope ≤2 全在 linear，
   与「方向桶已归零」的 §6 结论冲突，需按对抗审查方法逐条复核。
5. 产能：弃权后 50k 组 ≈ 38k SFT 行；若要保行数需上调 target 或消费
   `winner_confidence` 标签。

### 7.4 本轮新增产物与工具

- 构建：`fresh100-v4a-20260729`（已降级为调优批次，8 行被 v4.1 重标）、
  `fresh200-v4a1-20260729`（干净 out-of-sample，n=160）。
- 审阅根 `/var/cache/veradata/annot_review/{fresh100-v4a-20260729,fresh200-v4a1-20260729}/`
  （d10_report / wp12_report / 失败聚类 / GPU 采样 / 全部盲评原始答复）。
- 工具：`dataset_build/tools/reeval_*.py`（机械检查 --contract auto、盲评五维、
  配对、传输层模型钉死）；诊断脚本副本在各审阅根 tools/ 下。
- 源池观察：`overlay_subject_not_ready` 53→114（07-29），下一个 build 前建议查清。

## 8. hints 度量重设计路线（2026-07-29 晚定案，进行中）

fresh 两轮 NO-GO 的根因收敛到 hints 均值失真（档案 docs/HINTS_MEAN_BIAS_2026-07-29.md
+ 其附录 A 的 WP14 感知标定 + 附录 B 的三路文献调研）。用户定案「度量即标注」
架构。执行序列：WP15ab（Direction List 条目级标签 + 候选度量 ROC 筛选，进行中）
→ WP15c（胜出度量替换 objective_edit_hints，渲染时计算/journal 持久化架构不变，
逐桶沉默阈值锚 sol 实测 JND/ROC 工作点，22 样本重放证伪）→ WP15d（验证门：
渲染重放符号硬门 + 反转对 2AFC Avg@4 仅可判别区 + recaption）→ fresh 批次 #3
出厂判定。preset 重分类（TS-WCL 路线）解耦后置。证据包 `_hints_bias`、
标定 `jnd_calib_20260729`、度量筛选 `wp15_metric_roc`。

## 9. fresh #3 出厂裁决与 v5.2（2026-07-30）

fresh150-v51-20260730（136 SFT 行）+ Claude opus 面板（10 评委、186 样本含 50
对照重判、评委不知批次、逐条数值复核）。**5/6 门过，G3（方向反转 5.1% vs ≤2%）
单门不过 → NO-GO**。产物 wp17_blind_panel* / wp17_panel_scores/。

关键事实：
- **同仪器下 v5.1 全面优于 v4.1**：pass 63.2% vs 52.0%（+11.2pt）、reasoning
  +0.44、after 指涉评委点名率 **1.5% vs 24.0%**（v5.1 禁令跨批直接证明有效）；
  生产 prose 门交付残留 0（触发 2.9% 全被重抽吸收）。
- **仪器漂移 −16pt**：同 50 条对照上轮 68% pass 本轮 52%——本轮评委系统做
  色调曲线级数值复核，捕获旧仪器不可见的反转类；对照批（v4.1）在新仪器下
  反转率也有 4.0%，G3 的 2% 是旧仪器灵敏度下定的数。
- G3 的 7 条反转分解 → **v5.2 三件套**（全部有面板给出的修法）：
  1. contrast 度量结构盲区（std 代理在压平背景下反向；改逐十分位色调曲线斜率）；
  2. surfaces 桶被映射到 mask 外显眼物体（加域内限定 + 位置线索）；
  3. 单色 NOTE 过度触发（fresh 侧 25 条夸大 flag + 1 条灾难；阈值绑定实测
     残余 chroma ≤6.0 双档化）。
- 数据侧发现：源池存在 540×360 低分辨率 before（面板 ~7 例）；同源图多编辑对
  需按源图分组切分 train/eval；SAM3 空转损耗 26% 源池（P1 仍未决）；
  `overlay_subject_not_ready` 53→161 持续爬升待查。

v5.2 实施中（WP18：三件套 + rather-than 语域 + mech benign 五条 + 自我指令
回抄模式 + mini30 冒烟），完成后 fresh #4 同门槛判定。

## 10. WP18 / v5.2 落地终态（2026-07-30）

三件套全部落地，标定数字如下（校准脚本 `/tmp/wp18-*.py`，只读面板产物）。

### 10.1 contrast：色调曲线斜率（`visibility.py`）

度量 = `(色调曲线斜率 − 1) × before-L 加权标准差`，在高 α 核心上按 **固定
10 L\* 宽**的亮度带做最小二乘（评委引用的正是 30-40 / 70-80 这种绝对带，
不是质量分位）。每条带贡献一个点，带权重 = `min(带质量, 核心质量/10)`——
**封顶是修复本身**：p066 的压平背景占 mask 三分之二，按质量加权就会像支配
方差一样支配斜率。

- **量纲等价**：编辑为线性色调映射 `L2 = g·L1 + c` 时，斜率恒等于 g，
  `(斜率−1)·std(L1) ≡ std(L2)−std(L1)`，与 v5.1 数值完全相同；两者只在
  非线性处分歧，也就是 p050/p066 的结构。死区因此仍是 L\* 单位。
- **AUC（fresh200 n=160 Direction-List contrast 标签）**：新度量 **0.9196**
  vs 生产旧度量 `noclip.d_contrast` **0.9117**（`full` 0.9234 / `core`
  0.9248，均在噪声内）。带权方案对照：按质量加权 0.911（p066 回退到 −2.79，
  等于没修）、等权 0.900。
- **死区 1.3 → 2.3**。1.3 本就是错的：它取自 ROC 里 `full.d_contrast` 的
  工作点，而出厂的是 `noclip.d_contrast`（同一次扫描的工作点是 2.37）。
  新值按同一 `operating_point` 程序重算，精度目标从 0.90 提到 **0.95**——
  现在卡住的门是 G3（反转 ≤2%），十次断言错一次的统计量达不到。2.2629→2.3
  时：160 行里断言 76 条、精度 0.934、反转 4——与出厂 v5.1 统计量在它自己
  0.90 工作点上的表现（75 / 0.933 / 4）完全相同，但 p050/p066 是对的。
- **p050 / p066 复核**（归档像素，生产 512 网格，numpy/torch 逐位一致）：
  p050 `+2.27 higher` → **−5.47 lower**（评委：变平、发灰）；
  p066 `−3.25 lower` → **+0.68**（评委：1.67× 扩张；落死区内 → 沉默，
  不再反向断言）。
- **noclip 支撑取消**。p050 的反转正是它造成的：抬黑位的像素全在低轨上，
  被 noclip 删掉了。色调曲线自带封顶，不需要这道滤除。

### 10.2 surfaces 域内锚定（`visibility.py` + `responses.py`）

每桶落盘 `position`：mask 内加权质心 → 三分法粗位置词（`upper left` /
`lower half` / `middle of the frame` …），hint 行渲染成
`the red areas (12% of the region, mostly in the lower half, saturation …)`。
`_HOW_TO_USE` 第 2 条加硬限定「只许点名编辑区域内承载该色的内容」，并写明
位置词是用来**认物**、不得回抄。旧 journal 缺 `position` 键照常渲染。

### 10.3 单色 NOTE 双档（`responses.py`）

落盘补 `chroma["after_mean"]`（mask 内加权 after 图 C 均值）。
`delta ≤ −4.0` 且无相反色面时：`after_mean ≤ 6.0` → 原「转换、颜色没了」；
`> 6.0` → 新「强去饱和，颜色仍可见但被大幅压低」。阈值 6.0 取自面板实测的
兑现案 0.55/3.07/5.73 与虚假案 ≥8.06 之间的空档。旧 journal 无该键时走
**软档**（「颜色没了」是需要证据的那一句）。防灾条款：`delta` 为正或在死区
内一律不注入任何单色措辞。

**fresh150 全量重放（136 行，归档像素）**：v5.1 规则在 **38 行**上注入
「原色已消失」；v5.2 只在 **9 行**保留（after_mean 1.05–5.73，全部落在面板
兑现区间内），**29 行降级**（after_mean 4.65–68.40，含 p138 的 68.40）。
面板点名的「25 条夸大 + 1 条灾难」被完整覆盖。
contrast 侧同一重放：断言 88 → 64 行，唯一一条方向对翻是 p050（修对）。

> 重放读的是归档 JPEG，生产算的是 pre-JPEG；HINTS_MEAN_BIAS §2.2 记录的残差
> 中位数为一阶轴 0.032、contrast 0.093，故重放为指示性数字，不是存档事实。

### 10.4 附带小项

- **task_clause 语域**：删掉 v5.1 结尾的「Do not claim the edit is confined
  to that subject…」——面板的 language 扣分主源正是模型照办的样子（"rather
  than treating the woman alone"、"as one connected local treatment"）。改成
  正面要求「把主体连同一起变化的邻近元素列出来；能走多远只写在 region_scope」。
  按 WP16 教训移除诱因，而不是再加禁词。
- **prose 门第 4 类 `self_instruction_echo`**：把沉默规则回抄成散文
  （"Avoid making a directional contrast or color-balance claim"、"without a
  stated directional change"、"avoiding an asserted contrast shift"）。锚在
  fresh150 实际措辞上：命中 **16/136**，而在该措辞出现之前的 fresh100 +
  fresh200 + eval100 共 **386 行上命中 0**。更宽的 "avoid/without + making"
  被证伪并否决（fresh200 命中 55/160，是正常英语）。
- **mech benign 六条**（各锚一句真实语料）：`feather markings`；表语/并列裸
  `matte`；并列名词裸 `mask`（"the woman, mask, clothing"）；`colour band`
  定中语序；`painted stripes`；`brightness/tone … upward`。另加 mini30 冒烟
  发现的第七条：被摄物体的 `curved band`（手镯）。

### 10.5 mini30 冒烟（`mini30-v52-20260730`，30 组）

`complete_with_failures`，27 SFT 行，relay 27 次成功 + 20 次重试 ≈ **47**
（预算 60）；terminal_failures 1。

- **mech 全量 27/27 clean，零 flag**（对照：同工具跑 v5.1 的 fresh150 是
  120/136 clean，16 条 `self_instruction_echo` + 1 条几何词）。
- 新门在线生效：2 次 `prose_violation` 全部为自我指令回抄，均被有界重抽
  吸收（非 terminal），**交付残留 0**。
- **位置词零回抄**：27 行里没有一行把 `upper/lower half`、`middle of the
  frame` 之类写进散文——这是 §10.2 最大的风险点，已实测排除。
- **6 条人工核对全部相符**：单色 NOTE 两档各一例（after_mean 4.89 的确近乎
  无彩；12.80 的荷叶与荷花颜色明显尚存，v5.1 会在这条上说「颜色没了」）；
  contrast 最正 +5.87（暗部压到 16→4、亮部守住 75→78，画面确为强反差）与
  最负 −3.97（画面发灰发平，且该行 39.6% 像素在轨上，正是 v5.1 会删掉四成
  支撑的情形）；两条位置词与图相符（背景粉色花带 → `upper half`，绿叶 →
  `middle of the frame`）。

### 10.6 遗留

- `after_mean` 分的是「中性 vs 有彩」，不是「单色 vs 多色」。棕褐（sepia）
  转换带着 15 左右的残余 chroma，会落进软档。mini30 该行的实际文本写的是
  "sepia-leaning … while leaving some color visible"，反而比硬档准确，故未
  处理；若日后要区分，需要的是色相集中度而非残余 chroma。
- `profile_render_segments.py` 的 numpy 回退分支调用 `objective_edit_hints_
  from_lab` 时漏传 `support=`（WP15c 起就会 TypeError，该分支从未被触发）。
  本次顺手补上。
- fresh150 §9 列的数据侧问题（540×360 低分辨率源、同源图分组切分、SAM3 空转
  26%、`overlay_subject_not_ready` 爬升）本轮未动。
