# 统一并发模型 v2 — 业务/core 解耦 + 统一镜像 + core-as-server

> 日期：2026-06-16 ｜ 方法：在 v1 全 pipeline DAG 诊断之上，按用户拍板的"业务/core 解耦"方向做 3-owner 深度设计 + 对抗式综合，再与用户"统一镜像 + core 暴露 server"指示调和，作者核验承重事实。
> **诊断（现状 DAG / 空转清单 / 弹性分析）见 v1 `UNIFIED_CONCURRENCY_DESIGN_2026-06-16.md` §1，本文不重复。** 本文是**权威架构**。
> 关联：`lrc_scripts/ROBUST_RENDER_REQUIREMENTS.md`（LR 农场，本架构的 proven 范本）、`SOURCE_QA_REVIEW_2026-06-15.md`。

---

## 0. 执行摘要

**用户决定**：完全重写编排/并发层——**业务**（build streams S1-S8 + source-clean 各阶段）与 **core**（每重资源一条任务队列：vllm / sam3 / iqa / render / lr）**解耦**，业务只做生产者、统一消费 core；**以 vLLM 容器为基底建统一镜像，core 在其中暴露 server 供业务消费**；SPOF = 瘦 broker + 自动重启；不依赖 gflow。

**架构一句话**：每张可用 GPU 跑**一个统一容器**（基于 `vllm/vllm-openai:nightly` 加装 SAM3/pyiqa/renderer 依赖），容器内 = `vllm serve`（不动）+ core worker server（sam3/iqa facade）+ 该卡的 business shard（renderer 进程内）。前置**一个瘦 vLLM broker（:8003）**对 1–2 个容器的 vLLM 做副本发现 + least-outstanding 路由 + 优先级准入。业务一律 `core.vllm/sam3/iqa/render/lr` 同步调用，绝不碰端口/设备/线程池/render_lock。**1↔2 卡塌缩成一个数字**：`全局预算 = Σ 活副本 × 单副本上限`。

**对抗式综合的 4 个承重纠正（决定架构形态，务必遵守）**：
1. **vLLM 客户端全是同步 HTTP、打单一可配置 URL**（`vlm_clean.py:431,504`、`preset_qa.py:320` 裸 `requests.post`、`llm_qa.py:129` requests Session）→ vLLM core = **进程外透明 HTTP broker**，**业务零改写**。仓库里**没有任何 async 客户端**，把 vLLM 包成进程内 AsyncOpenAI 队列是错的。
2. **业务以 N 个独立 shard 进程跑**（`run.py --shard i/N`）→ 进程内 broker 对其他 shard 不可见 → **broker 必须进程外**。这是最关键的接缝。
3. **render 留进程内、每 shard 单 worker**（单 GPU、非线程安全、native-res 数组跨进程传输是浪费，且现有 main-thread-render/pool-clean 重叠循环 `run.py:683-689,863-934` 是已验证的精巧结构）→ **不改写成 asyncio**，业务保持**线程化**，core 客户端是**同步 `submit()`**（阻塞调用线程＝天然背压）。
4. **build 从不调用 IQA**（`run.py` 不 import iqa）→ `core.iqa` 仅 QA 阶段内部用，不在 build 请求路径上。

**为何这条最优**：v1 已论证透明 broker 是成本最低的杠杆；v2 在其上做"业务/core 解耦"且**复用现有线程编排不重写**，把改动收敛为"换传输 + 把 renderer/lock 搬进 core.render + 加一个 broker/supervisor + 一个只读 PG 谓词"。统一镜像把多 conda env 问题一次性消掉。

---

## 1. 架构总览

```
                         ┌────────────────────────── 瘦 vLLM broker :8003 ───────────────────────────┐
                         │  副本发现(轮询 /v1/models, served-name 断言) · least-outstanding 路由         │
                         │  3 级加权公平准入(build-annotate>qa-judge>tag) · 全局预算=Σ活副本×cap · 429   │
                         └───────────▲───────────────────────────────────────────────▲────────────────┘
                                     │ HTTP(OpenAI /v1)                                │
        ┌────────────────────────────┴──────────┐            ┌─────────────────────────┴──────────────┐
        │  统一容器 @GPU0 (FROM vllm/vllm-openai) │            │  统一容器 @GPU1 (1 卡时无此容器)          │
        │  ┌─ vllm serve :8000 (35B 副本, 不动) ─┐│            │  ┌─ vllm serve :8000 ────────────────┐  │
        │  ├─ core worker server :C ────────────┤│            │  ├─ core worker server :C ──────────┤  │
        │  │   /sam3/masks  /iqa/score          ││            │  │  ...                              │  │
        │  ├─ business shard i/N (base 逻辑) ────┤│            │  └────────────────────────────────┘  │
        │  │   ThreadPool 编排 + ShardWriter(主线程)││           └─────────────────────────────────────┘
        │  │   render worker(进程内): gpu-lease→render_lock→renderer.render()
        │  └─ 共卡 gpu-compute lease(render/iqa/sam3 串行)        supervisor: nvidia-smi 轮询→docker 起 1/2 容器
        └────────────────────────────────────────┘
   业务调用:  vllm→broker:8003 │ sam3/iqa→本容器 core worker(localhost) │ render→进程内 │ lr→PG render_jobs + :8081 长轮询
   先清洗后消费:  build planner 只读 PG 谓词(assets.auto_verdict/final_decision/dup_of, 保守=仅已清)
```

**业务（生产者）**：`streams.py` S1-S8 + source-clean 阶段。改写后只剩 `plan → core.* → assemble → ShardWriter`。`core` 句柄在 `run.py` 建一次、串进 `StreamCtx`；`render_lock`+`renderer` 从 `StreamCtx` 搬进 `core.render` worker。

**core（每资源一队列，按传输现实分型）**：见 §3/§4。

---

## 2. 进程拓扑（每进程 · 环境 · 传输）

1. **统一容器 ×1–2**（`FROM vllm/vllm-openai:nightly` + SAM3/pyiqa/renderer 依赖）。每张活 GPU 一个。内含：
   - `vllm serve`（内部 :8000，**保持 stock 不动**——保留持久编译缓存 `/home/bc/data/vllm_cache`、证明过的连续批处理；**不嵌 AsyncLLMEngine**，否则丢掉透明 broker 的零改写优势）。
   - **core worker server**（同容器进程，暴露 `/sam3/masks`、`/iqa/score`）。
   - **business shard 进程**（`python -m dataset_build.run --shard i/N`，与该卡 renderer 进程内共存）。
2. **vLLM broker :8003**（base/纯 Python，无 torch）：发现 1–2 容器的 vLLM :8000、least-outstanding、3 级准入、429+Retry-After。SPOF→systemd 自启；重启后重新发现（副本对 broker 无状态）。
3. **supervisor daemon**（base，不在请求路径）：每 ~15s 轮询 `nvidia-smi`，按空闲卡数起 1/2 个统一容器（复用 `launch_reasoning.sh:60-71` docker run 模板 + 持久缓存挂载），注册/注销到 broker。
4. **LR 农场**（off-host Win/Mac LrC ×~3 反向连接 `lrc_task_server.py:8081`，durable `render_jobs`）。**不动**——它是本架构的范本。
5. **PostgreSQL** `vera_source_qa`：QA 阶段写（`db.py:395` write_retry + `FOR UPDATE SKIP LOCKED`），**build 只读**（清洗谓词）。`render_jobs` 即 LR durable 队列。

**传输小结**：vLLM = 跨进程 HTTP（真正的解耦边界，跨 GPU 均衡靠它）；sam3/iqa = 本容器 localhost（统一镜像后同 env）；render = 进程内（单 GPU 无 IPC 收益）；lr = PG + HTTP 长轮询（已验证）。

> **统一镜像 vs SAM3 隔离（✅ 实测已定，见 §10）**：原 `monetgpt_sam3` 独立 env 是 **pin 冲突**（base tf4.57 vs SAM3 tf5.2），非运行时硬约束。镜像的 **tf5.7 同时兼容 SAM3 与 renderer**，实测均加载通过——**SAM3 与 renderer 同居统一镜像、无需 sidecar**。共卡时 **SAM3 与 renderer 永不同时占卡**（gpu-compute lease 串行；本就一个是 stage-0 预计算、一个是 build 期进程内）。唯一遗留：SAM3 `Sam3Processor` 的 `tokenizer.json` 畸形（tf5.2/5.7 通病，需单独修；stage-0 cache 已在，不阻塞）。

---

## 3. core API（业务唯一接口，同步阻塞）

| 调用 | 签名 | 队列/后端 | 返回 |
|---|---|---|---|
| `core.vllm.submit` | `(messages, images=None, json_mode=False, prio='build-annotate'\|'qa-judge'\|'tag', timeout=120) -> dict` | 进程外 broker :8003，least-outstanding 到活副本；prio→`X-vgate-class` 加权公平；超预算 429（客户端 shim 有界重试） | `{content, usage}` |
| `core.sam3.masks` | `(image_path, concepts) -> {concept: np.float32[H,W]}` | 默认读 `sam3_cache` PNG（`CachedMasker`）；统一镜像后可选 live worker `/sam3/masks` | 已缓存概念的 mask |
| `core.iqa.score` | `(image_path) -> dict` **[QA 内部]** | pyiqa 在租用卡上 batch16；**build 不调** | `{musiq,clipiqa,niqe,brisque,...}` |
| `core.render.submit` | `(image_paths, param_dicts, batch_size=8) -> [np.uint8\|None]` | **进程内**单 worker：gpu-lease→`render_lock`(threading.Lock)→`renderer.render()`→降采样 768。多数样本不触发(`run.py:705-719`) | 降采样 after，逐项 |
| `core.lr.submit` | `(probe, xmp, preset_id, content_hash, region_local=False, ai_mask=False, engine='lrc', timeout=1200) -> dict` | 现有 `render_jobs`(UNIQUE 幂等) + `:8081` 长轮询，**不动** | `{after_jpg_path, engine, metrics}` |

---

## 4. 资源队列（分池/分队列）

| 队列 | 后端 | worker | 准入/优先级 | 批处理 | 缓存 | env |
|---|---|---|---|---|---|---|
| **vllm** | 进程外透明 HTTP broker :8003（泛化 `lrc_task_server:81-92` 的 Condition/pending/active 到 HTTP least-outstanding），前置 1–2 容器 | broker 异步；各 shard 用现有 `ThreadPool(concurrency=32)` 同步发；QA 用 16 | 全局 `cap×活副本`；加权公平 build-annotate(P0~60%)>qa-judge(P1~30%)>tag(P2~10%)；429 | 无（vLLM 引擎自做连续批） | 图 data-URI LRU（`vlm_clean.py:569`）复用；`tag_cache` 短路 tag 类 | broker base；副本 Docker |
| **sam3** | stage-0 PNG 预计算（统一镜像后无需独立 conda env，同镜像跑）+ `CachedMasker` 读 | 预计算 N sharded（每卡一）；build 期 0 worker | n/a（FS 读）；预计算可续 | 每图多概念 | PNG `use_cache:true`；`--from-tags` 提命中率 | 统一镜像（原 monetgpt_sam3 折叠进来） |
| **iqa** | pyiqa（MUSIQ/CLIPIQA+/NIQE/BRISQUE+laplacian/noise/face），**QA 内部** | QA iqa 阶段 threadpool，与 render 共 gpu-lease | 卡 lease 后批 16 | 16 | 结果落 assets 列，已评跳过 | 统一镜像 |
| **render** | **进程内**单 worker = 现有 `run.py:904-925` 主线程批渲染；gpu-lease→`render_lock`→`renderer.render()`→768 | 每 shard 1（teacher 非线程安全）；现有重叠循环喂 | `render_lock` 仅包 `renderer.render()`；`max_outstanding=32` 限 | 8（现 `_render_after_batch`） | 无；ShardWriter done_ids 续跑 | 统一镜像，shard 的卡 |
| **lr** | 现有 `render_jobs` PG + `:8081` 农场，**不动** | ~3 off-host LrC（cap 3：6% vs 62%@6） | 农场背压 + UNIQUE 去重 | 每 job | content_hash+engine UNIQUE | off-host |

---

## 5. 业务改写（S2 为例，before/after）

**BEFORE（耦合）**：主线程 `afters = stream._render_after_batch(...)` → `with ctx.render_lock: ctx.renderer.render(...)`（`streams.py:640-641`，业务**拥有锁+渲染器+设备**）；pool 线程 `ctx.cleaner.gen_instruction(...)` 同步 POST 到写死的 `:8001`（业务**拥有端口+客户端**）；SAM3 已经是 `CachedMasker` 读（已解耦）。

**AFTER（解耦）**：
- 主线程 `afters = core.render.submit(paths, params, batch_size=8)`。render worker（**就是同一个重叠循环，搬到 core.render 背后**）取 gpu-lease→`render_lock`→`renderer.render()`→768→释放。锁与渲染器**只在 core.render**。重叠流水线（渲第 K+1 批 ‖ pool 清第 K 批）**保留**。
- pool 线程 `content = core.vllm.submit(messages, images=[src], prio='build-annotate')` 取代 `cleaner.gen_instruction`。`QwenVLCleaner` 的 prompt 构建/JSON 解析/图 LRU（`vlm_clean.py:312,569`）**在 core.vllm 客户端 shim 内复用**，只把 base_url 从 :8001 翻到 broker :8003。**算法零改**。
- `mask = core.sam3.masks(src, concepts)`（同 `CachedMasker` 读，统一 API 包装）。
- pool 线程**阻塞在 `core.vllm.submit`**＝背压（`run.py:928` max_outstanding 闸保留）。无 asyncio、无事件循环、不改 drain 逻辑。

**source-clean 阶段**：`llm_qa.py`/`preset_qa.py` base_url 翻到 broker :8003（已读 `config.VLLM_BASE_URL`/`SOURCE_QA_VLLM` env），白拿 least-outstanding + `prio='qa-judge'`，**零改写**；iqa 阶段走 `core.iqa`（gpu-lease）；`gate/calibrate/apply` 纯 CPU 不动；清洗谓词加在 `run.py load_plan_inputs`。

---

## 6. 模块树

```
dataset_build/
├── core/                       # 新增：解耦的 core（仅编排/并发）
│   ├── __init__.py             # facade：按 config 建 vllm/sam3/iqa/render/lr 句柄；同步 submit 面
│   ├── client.py               # 统一客户端：VLLMClient(HTTP→broker)/Sam3Client(Cached 读)/IqaClient(gpu-lease)/RenderClient(进程内)/LrClient
│   ├── gpu_compute.py          # 每卡 lease/semaphore；严格序 lease→render_lock；least-busy 卡选择
│   ├── render_worker.py        # 持 render_lock(threading.Lock)+renderer；包 streams._render_after_batch
│   └── broker/                 # 新增：进程外 vLLM broker（独立进程，systemd）
│       ├── app.py              # FastAPI 透明 /v1 代理 + 3 级加权公平 + 429 Retry-After（~250 行）
│       ├── discovery.py        # 轮询 :8001/:8002 /v1/models，served-name 断言，least-outstanding 表
│       └── supervisor.py       # nvidia-smi 轮询，docker 起停(launch_reasoning 模板)，注册副本
├── run.py                      # 小改：建 core、串进 StreamCtx、render/vllm 重指 core.*、load_plan_inputs 加只读 PG 谓词。ThreadPool+重叠+drain+ShardWriter 原样
├── streams.py                  # 改：ctx.render_lock/renderer/cleaner 直用 → ctx.core.*；删 render_lock 字段。plan/build_one 逻辑不动
├── vlm_clean.py                # 复用：QwenVLCleaner prompt/parse/图 LRU 被 core/client.py import；base_url 翻 :8003
├── render.py / masking.py / sam3_precompute.py / recipes.py / contracts.py / pack.py / aesthetic.py / *_cache.py   # 不动
├── docker/                     # 新增统一镜像 Dockerfile（FROM vllm + SAM3/pyiqa/renderer 依赖）；launch_reasoning.sh 被 supervisor 复用
└── source_qa/
    ├── config.py               # 1 行：VLLM_BASE_URL 默认→broker :8003（已 env 可覆盖）
    ├── llm_qa.py / preset_qa.py / iqa.py   # 改：prio 头 + base_url 经 broker + iqa 加 gpu-lease。逻辑不动
    └── db.py / gate.py / calibrate.py / apply.py / dedup.py / lr_render.py   # 不动
lrc_scripts/servers/lrc_task_server.py   # 不动（LR 农场；broker 范本）
```

---

## 7. 必守不变量核验

| 不变量 | 如何保住 | 残余风险 |
|---|---|---|
| render_lock 仅包 `renderer.render()`、非重入；若加 lease 则 lease 先于 lock | `threading.Lock`（**非 RLock**，`streams.py:173`）原样搬入 `core/render_worker.py`；`gpu_compute` 严格 lease→lock | 单 shard 单 render worker，序局部；未来 render+iqa 共卡须 lease 非重入 |
| SAM3 env 隔离 | 默认 stage-0 预计算 + `CachedMasker` 读；统一镜像后同镜像跑，**与 renderer 永不同卡并发**（lease 串行） | cache miss fallback 当前禁用→缺 mask；live SAM3 仅在实测无冲突后启 |
| 先清洗后消费（保守=仅已清） | `run.py load_plan_inputs` 加只读 PG 谓词：`dup_of IS NULL AND 已显式 cleared`，verdict=drop **fail-closed** | source_index 单一混合文件→谓词按**资产 id** 而非 corpus；plan 时 MVCC 快照读 + 保守默认 |
| PG 写语义（build 只读，故障优雅降级） | 写仍在 QA 阶段 `write_retry`+`SKIP LOCKED`；build 仅加只读连接 | **生产构建谓词失败应 HALT（fail-closed），非 fail-open** 放进未清资产（见开放决策） |
| ShardWriter 单线程/原子 | 编排仍线程化但 `write` 只在主线程 drain 步调用；core.* 不写 shard；无 asyncio→无并发写 | 无新增 |
| ppr10k 三专家 fan-out | `apply.py:13-22` 纯 CPU 不动，core 不碰 | 无 |
| served-name 跨副本一致 | `discovery.py` 断言每副本 `/v1/models` == `qwen3_5-35b-a3b`，异构副本入表前隔离 | 误起错镜像→容量静默降 1，需告警 |
| LR cap ~3 | core.lr 薄包，农场背压 + UNIQUE 不动 | 无新增 |
| teacher render 多数样本空转 | `core.render.submit` 仅 `after_needed` 触发；无预热 | 1 卡模式 render 突发与 vLLM KV 争用，靠 mem-util 0.80 + lease 串行；verify 率若全局拉高则单卡假设破 |

---

## 8. 对抗发现（综合阶段查出的常见错误，务必避开）

1. **render_lock 是 `threading.Lock` 不是 RLock**（`streams.py:173`）——用 RLock 会吞掉重入 bug。
2. **仓库无 async vLLM 客户端**——把 vLLM 包成进程内 AsyncOpenAI 队列要重写全部三个客户端、且零解耦收益；透明 HTTP broker 是唯一零改写解。
3. **业务是 N 个独立进程**——进程内 broker 对其他 shard 不可见；**只有进程外 broker** 能服务所有 shard + QA。这是头号接缝。
4. **render 跨生产者合批是伪问题**——render 是每 shard 单 GPU，现有循环已在 shard 内合批 8；跨 shard 合批需共享 GPU IPC，被"单卡非线程安全"禁止。render 队列必须**进程内每 shard**。
5. **SAM3 传输**：stdin/stdout 传二进制 mask 脆弱；若将来上 live SAM3，用 **FS-drop + PG 状态**，不要 stdout。
6. **build 从不调 IQA**——`core.iqa` 是 QA 内部路径，不在 build 请求路径。
7. **broker 重启中的在途请求**：客户端 shim 必须把 broker-down 当**有界退避重试**（非样本失败）；supervisor 快速重启 broker。
8. **背压不要双重计数**：权威背压是 `run.py:928` max_outstanding 闸；broker 429 是安全阀、由客户端重试吸收，不作流控信号回灌业务。

---

## 9. Cutover 计划（strangler-fig，每阶段独立可跑/可回滚）

| Phase | 范围 | 可交付/回滚 | 复用 | 删除 |
|---|---|---|---|---|
| **0 broker drop-in**（最高性价比，**不需统一镜像**） | `core/broker/app.py`(透明 /v1 代理)+`discovery.py`(轮询 :8001/:8002, served-name 断言, least-outstanding)。`config.yaml:75` base_url + `config.py:41` SOURCE_QA_VLLM 重指 :8003。**业务零改** | 今日 build+QA 原样跑通 :8003，**立消 reason_g1 84GB@0%**；回滚=翻回 :8001 | `lrc_task_server:81-92` 模式、三客户端原样 | 无 |
| **1 准入+弹性** | broker 加 `X-vgate-class`+加权公平+per-replica cap+429。`core/broker/supervisor.py`(nvidia-smi+docker 起停)+systemd | 1↔2 卡弹性；P0>P1>P2；broker 自启。各件独立可测 | `launch_reasoning.sh:60-71`、持久缓存 | 无 |
| **2 统一镜像 + core facade** | 建统一镜像 Dockerfile（FROM vllm + SAM3/pyiqa/renderer，**前置 §10 实测**）。建 `core/{__init__,client,gpu_compute,render_worker}.py`；core 串进 StreamCtx；render_lock+renderer 搬入 core.render；`run.py:910` 改 `core.render.submit`、cleaner.* 改 `core.vllm.submit`；`CachedMasker` 包成 `core.sam3.masks`；iqa+render 加 gpu-lease。**重叠循环+drain+ShardWriter 原样** | **单 stream（S2）端到端跑 core，与 cutover 前 shard 输出 diff 证字节一致** | `run.py:863-934`、`streams.py` plan/build_one、render.py、masking.py、vlm_clean.py | 无 |
| **3 清洗谓词 + source-clean 上 core** | `run.py load_plan_inputs` 加只读 PG 谓词；llm_qa/preset_qa 经 broker `prio='qa-judge'`；iqa 走 gpu-lease | 全 clean→build 带判级闸；谓词可在 fixture DB 独测 | `db.py:395`、gate/apply/calibrate | 无 |
| **4 收尾** | 删硬编码 :8001/:8002 与 /2 假设；合并 mk_cfg/launch_dual 重复；文档化 core API。可选：gpu-compute :8004 HTTP 服务（仅当跨进程租约值得）、live SAM3 PG-subprocess（仅当需动态概念） | 单一规范启动路径 | 前序 | `launch_dual.sh` 硬编端口、`config_g8001/g8002` |

> **Phase 0 不依赖统一镜像**（broker 挡在现有 stock 容器前），是即刻可发的高价值低风险第一步。统一镜像是 Phase 2 才需要的事，可与 Phase 0/1 并行做 §10 实测去风险。

---

## 10. 统一镜像前置实测（✅ 已完成 2026-06-16）+ 开放决策

**实测裁定：全统一镜像可行且生产安全。** 镜像基线 `vllm/vllm-openai:nightly` = **torch 2.11.0+cu130 / transformers 5.7.0 / py3.12**。在其上 `uv pip install --system`（清华镜像）`python-box attrdict3 peft==0.15.0 diffusers==0.34.0 timm==1.0.24 pyiqa`，**全程不触碰 torch/transformers/vllm**。Dockerfile：`dataset_build/docker/Dockerfile.unified`；in-image 自测：`dataset_build/docker/smoke_unified.py`。

| 组件 | 结果 | 证据 |
|---|---|---|
| **vLLM** | 基线，保持 stock `vllm serve`（**不嵌 AsyncLLMEngine**，保留透明 broker 零改写） | — |
| **SAM3 模型** | ✅ 镜像内 `Sam3VideoModel.from_pretrained` 加载 OK（13.8s，detector 840M）。`Sam3VideoModel/Processor/Model` 均为 transformers 5.7 内建 | 旧 `monetgpt_sam3`(5.2) 分 env 仅因 pin 冲突；5.7 同样载入 |
| **renderer（曾判最高风险）** | ✅ LLaVA/Qwen2 `VeraRetouchForCausalLLM_Unified` 加载（36.9s）+ 前向 OK；**vs base(4.57) 数值 parity：max\|Δ\|=2/255，PSNR 56.85 dB**（bf16 GPU 非确定性 + 4→5 微差的舍入噪声，无结构分歧） | `requirements.txt` 的 `transformers==4.57.1` 只是测试基线、非硬约束 |
| **pyiqa** | ✅ 镜像内安装+评分 OK（musiq 70.4 / clipiqa+ 0.77）；其 `≥` 依赖被镜像满足，未改 torch/torchvision | — |

**=> 原计划的"SAM3/renderer 退 base-env sidecar 容器"退路用不上**：vLLM + SAM3 + renderer + pyiqa + IQA 同居一个镜像、一个 env、一张卡（共卡仍按 gpu-compute lease 串行 render/iqa/sam3，与 vLLM 共存按 mem-util）。business shard（含进程内 renderer）跑在统一容器内。

**遗留待办**：
- ✅ **SAM3 tokenizer 已修复（2026-06-16）**。根因不是"损坏"而是**错配/污染**：`/home/bc/data/models` 根的 tokenizer + preprocessor 被 **Qwen2.5-VL 整套文件覆盖**（tokenizer_config=Qwen2Tokenizer、词表 ~151k、含 `<|object_ref_start|>` 等 grounding token；preprocessor=Qwen2VLImageProcessorFast），而 SAM3 `text_config.vocab_size=**49408**` 要的是 **CLIP tokenizer**。规范源 `facebook/sam3-base` gated（主机无 HF token），故用本地 `clip-vit-large-patch14` 的标准 CLIP tokenizer（vocab 49408 精确匹配）替换，Qwen 污染件备份在 `/home/bc/data/models/_qwen_pollution_backup_2026-06-16/`（可回退）。**金标验证**：修复后对已 cache 的图重生成 mask，与缓存 PNG 比对 **IoU 0.88–0.99 / corr 0.92–0.99**（错 tokenizer 会近 0）→ 复现原行为，`Sam3Processor` 正常加载，live SAM3 / 重建 cache 能力恢复。（移走 Qwen preprocessor_config 后 mask 仍正确 → SAM3 用 `processor_config.json` 内嵌 image processor 即可。）
- `torch_dtype=` 已在 transformers 5.x 改 `dtype=`（render.py:193），仅 deprecation warning，可顺手改。
- cutover 上生产前对 teacher 渲染做更大样本的 parity 抽检（当前单图 max\|Δ\|=2/255 已是噪声底，预计无虞）。

**开放决策（带推荐默认）**：
1. **单副本 cap 标定**：28/32/64？对 fp8 35B-A3B@0.85 mem-util 的 KV 压力**需 load test**，起 32 二分。（Phase 1）
2. **1 卡 render 与 vLLM 共存**：(a) mem-util 降 0.80 让 render 突发 ｜ (b) 硬串行 render vs vLLM。**默认 (a)**（verify 当前多数关，render 少）。
3. **清洗谓词 PG 故障**：fail-open（降级放未清）vs fail-closed/HALT。**默认 fail-closed**（合不变量；fail-open 仅在响亮告警下可接受）。
4. **SAM3 stage-0 vs live**：**默认 stage-0**（合今日仓库、最简）；统一镜像后 live 才低成本可选，仅当需动态 tag 驱动概念扩展时上。
5. **shard 数 N**：**保持固定**（=2 或按需），只弹性化副本数；已写 shard 不可重切，N 别绑副本数。
6. **业务并发模型**：**同步线程化**（已定）；若将来单 shard 需上千并发在途 vLLM，async shim 藏在同一 `core.vllm.submit` 签名后即可迁移。
7. **gpu-compute lease**：**进程内 semaphore**（Phase 2）；`:8004` HTTP 服务仅当 render/iqa 跨进程租约值得（Phase 4 可选）。
