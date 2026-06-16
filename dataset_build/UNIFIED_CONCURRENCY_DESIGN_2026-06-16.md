# 统一并发模型设计 — VeraRetouch 数据 pipeline

> 日期：2026-06-16 ｜ 方法：6 视角并行深读真实代码 → 合成全 pipeline DAG（资源需求向量）→ 3 套独立设计 → 对抗式裁决 → 作者核验 3 个承重事实。
> 所有非显然结论带 `file:line`。关联：`SOURCE_QA_REVIEW_2026-06-15.md`、`DATASET_QUALITY_AUDIT_2026-06-13.md`、`lrc_scripts/ROBUST_RENDER_REQUIREMENTS.md`。
> **范围**：本文只设计**统一并发/调度模型**（分池、分队列、阶段、弹性 GPU），不动 QA 判级逻辑与 SFT 目标本身。
>
> ⚠️ **本文（§0–§7）是 v1：增量 vGate 透明代理路线。** 用户 2026-06-16 拍板**完全重写、业务/core 解耦 + 统一镜像 + core-as-server**——见 **`UNIFIED_CONCURRENCY_DESIGN_v2_2026-06-16.md`（v2，权威）**。v1 的 §1 现状诊断（DAG / 空转清单 / 弹性分析）仍 100% 有效，v2 直接引用之；v1 的设计路线（§3 起）被 v2 取代。

---

## 0. 执行摘要（TL;DR）

**病灶**：一个逻辑负载被当两套框架各跑各的。build 的 `QwenVLCleaner`（gen_instruction/reason_params/verify/tag）与 source_qa 的 `llm_qa` judge + `preset_qa` 问卷 C，**是同一个模型 `qwen3_5-35b-a3b`、同一类多模态 chat 请求**（KEY UNIFYING FACT），却被三个互不协调的客户端、各自限流（32 / 16 / 无上限）、各打各的端口消费，**无共享准入、无副本感知路由**。现场实证：GPU1 上 35B 占 84GB 显存、利用率 **0%**——副本满载却收不到流量。叠加 source_qa 是**顺序 CLI**（一个阶段跑时其余资源全闲），整条 pipeline 任一时刻大部分硬件空转。

**方向**（用户硬约束）：以**弹性 vLLM 推理池**为中心，把"几张卡 → 几个副本 + 非 vLLM GPU 活如何共卡"吸收进调度层，让上层 pipeline 对卡数无感；**不依赖 gflow**（它是粗粒度作业调度，我们的是百万级毫秒请求）；分池 / 分队列 / 阶段。

**设计 = vGate**：一个**透明 OpenAI 兼容反向代理 broker** 挡在 1–2 个 vLLM 副本前（副本发现 + 全局准入预算 + 优先级分类队列 + least-outstanding 路由 + 背压），加一个**轻量 supervisor** 管副本生命周期（按可用卡数起 1 或 2 个副本），加一个**逐资产 PG 闸门**让 QA 与 build 在保持"先清洗后消费"顺序的前提下窗口重叠。**1↔2 卡这件事被压缩成一个数字**：`全局预算 = Σ 活副本 × 单副本上限`。

**为什么是它**（对抗式裁决，86 分，三选一最高）：代码里**每个 vLLM 消费者都已经走单一可配置 URL**（`vlm_clean.py:472` 从 config、`config.py:41` `SOURCE_QA_VLLM` 环境变量、`preset_qa.py:320` 裸 `requests.post(config.VLLM_BASE_URL)`），所以"透明代理"是近乎零改动的 drop-in。第一步（~250 行 FastAPI 透传 + 一处 base_url 重指）就消灭了 GPU1 空转和"shard 0/2 打到没起的 :8001"两个故障，且完全可回滚。提案 3 的 GPU 计算租约作为后续增量、提案 1 的优先级分类从第一天就并入。

---

## 1. 现状 pipeline 的 DAG 与资源画像

### 1.1 节点（按资源池分组，带 file:line）

**vLLM 推理池（DOMINANT）**——全部打同一个 `qwen3_5-35b-a3b`：
| 节点 | 框架 | 每单位成本 | 并发原语 | 端点 |
|---|---|---|---|---|
| `QwenVLCleaner` gen_instruction/reason_params/verify/tag | build | 1 img+chat ~0.5–2s/调用，region-local 样本 3–4 调用 | `vlm_clean.py:429-482,504-628`；image encode sem=4 (`config.yaml:90`) | `:8001`(shard0) / `:8002`(shard1) |
| `llm_qa` 问卷 A/B/caption | source_qa | 1 多模态 chat/图 ~1.5s | `ThreadPoolExecutor(16)` `llm_qa.py:370`；cap=`VLLM_CONCURRENCY` `config.py:45` | `:8002` 硬编码 `config.py:41` |
| `preset_qa` 问卷 C | source_qa | 1 chat/预设 **无上限** | `preset_qa.py:320` 裸 `requests.post` | `:8002` |
| `tag_precompute`（stage-0 离线） | build | 1 tag/源 | `ThreadPoolExecutor(64)` `tag_precompute.py:206` | `:8001/:8002` |
| 副本 `reason_g0` / `reason_g1` | shared | 服务 N 并发 chat | TP=1, gpu-mem-util 0.85 (~84GB) `launch_reasoning.sh:68` | `:8001` / `:8002` |

**非 vLLM GPU 计算池**——与 vLLM 副本抢同一张 H100：
| 节点 | 设备 | 线程安全 | 备注 |
|---|---|---|---|
| 神经 teacher renderer | shard 的 `CUDA_VISIBLE_DEVICES` 卡，`render.py:112` | **否**，靠 `render_lock` 串行 `streams.py:640-641` | 仅 `after_needed` 时渲染（见 §1.4） |
| `iqa` NR-IQA (MUSIQ/CLIPIQA+/NIQE/BRISQUE) | **`cuda:0` 硬编码** `config.py:54`,`iqa.py:45,71` | 单模型循环 | 与 GPU0 的 `reason_g0`+renderer 同卡 |
| `preset_qa` stage2 NR-IQA | `cuda:0`，`gpu_lock` `preset_qa.py:444` | 锁串行 | 与上同卡 |
| live SAM3 masker | shard 卡 | 否（但缓存命中时不上 GPU） | `use_cache:true` → `CachedMasker` 读 PNG `config.yaml:42` |
| CLIP 美学 | GPU | `aesthetic._lock` `aesthetic.py:85` | tag_precompute 内 |
| `sam3_precompute`（stage-0） | **独立 conda env `monetgpt_sam3`** `sam3_precompute.py:3` | 独立进程，sharded | **不能与 renderer 共存**，绝不可建模为同进程/同卡租约成员 |

**CPU 池**：`dedup`(ThreadPool 16, `dedup.py:66`)、`preset_qa` stage1、`calibrate`、`gate`、build 的 S1/S4/S7/S8 `_build_cgt`、`tag_cache` 合并。
**LR 农场池**（off-host）：`preset_qa` stage2 → `lrc_task_server :8081` 反向连接，~3 在线 client，幂等队列 `render_jobs`（`db.py:154-176`）。容量天花板 ~3 并发（6% 失败 vs 6 并发 62%，见 ROBUST_RENDER_REQUIREMENTS）。
**PG 池**：`vera_source_qa`，每 runner 单连接，MVCC，`write_retry` 8× 退避（`db.py:355-423`）。**build 当前完全不碰 PG**。

### 1.2 关键依赖边（DAG）

```
source_qa（顺序 CLI）:
  ingest ─┬─ dedup ─ iqa ─┬─ llm_qa ──┐
          │               └─ calibrate ┤
          └─ preset_qa_s1 ─ preset_qa_s2 ─┐
                                          │
            {iqa, llm_qa, calibrate} ─ gate ─ apply ──┐  ← 唯一跨框架接缝
                                                       │
build stage-0: tag_precompute(vllm) / sam3_precompute(独立 env, gpu) ─ 喂缓存 ─┤
                                                                              ▼
build streams S1..S8: plan ─(after_needed 才 render)─ annotate(vllm) ─ verify(vllm) ─ ShardWriter
```

**跨框架接缝（唯一一处硬序）**：`apply.py` 写 `source_index.cleaned.jsonl` / `recipe_index.qa.jsonl`；build `load_plan_inputs` 读 `storage.source_index`（`run.py:505-506`）。**核验结论**：当前 `config.yaml:301-302` 指向**原始**文件、cleaned 指向被注释（`config.yaml:308-309`）——**QA 闭环现在根本没接上**（印证 SOURCE_QA_REVIEW FLOW-1）。且 `run.py:478,505,513` 证实 **source_index 是单一共享混合文件**，不是按 corpus 分文件。

### 1.3 跨框架资源争用边（无任何协调）

- `vllm_cleaner` ⟷ `sqa_llm_qa` ⟷ `preset_qa问卷C`：同模型 `:8002`，三个私有 cap（32/16/无），无共享准入、无副本感知路由 → **副本要么过载要么空转**。
- `sqa_iqa(cuda:0)` ⟷ `reason_g0(GPU0)` ⟷ `teacher_renderer(GPU0)`：三者挤同一张 H100，无仲裁。
- `teacher_renderer` ⟷ `reason_g{0,1}`：mirror 下 renderer 与副本同卡（`MEMUTIL 0.55/0.85` 留头寸，`launch_dual.sh:19`）。
- 两个 sharded orchestrator 进程之间无跨 shard work-stealing（`launch_dual.sh:81-85`）。

### 1.4 空转清单（idle inventory，本设计要消灭的）

1. **`reason_g1` 84GB@0%**——副本满载零请求（只有 `reason_g0` 是 build 活靶 + source_qa 没跑）。**典型浪费**。
2. **`dedup`（纯 CPU）跑时两张 GPU + 两个副本 100% 闲**。
3. **`iqa`（GPU）跑时 vLLM 副本 0%**；反过来 **`llm_qa`（vLLM）跑时 `cuda:0` 的 IQA 模型在显存里干等**。
4. **所有 sqa CPU 阶段（ingest/stage1/calibrate/gate/apply）期间 2×35B 全程 0%**。
5. **teacher renderer 对大多数样本空转**：S1/S4/S7/S8 `after=None`、S5 有 real_jpg 时旁路、S5/S6 verify-off（`run.py:705-719`）——renderer 这个 `render_lock` 保护的资源在 1M 预算里多数时间无活。
6. **build clean 线程卡在 `max_outstanding=32` 背压**：main 线程 `drain(block=True)` 等最老 future（`run.py:928-929`），GPU 在 render 批次之间饿着，而 vLLM 才是长杆。
7. **LR 农场 + ~3 client 除 stage2 外全程空转**（long-poll 返回 `{}`）。

### 1.5 弹性分析：哪些硬编码到 2 卡

1. `launch_reasoning.sh:91-94` 写死起**两个**容器，任一失败 exit 1，**无单卡模式**。
2. `launch_dual.sh:76-88` 起 `qwen_g0`+`qwen_g1` 再跑两个 orchestrator，shard 分母写死 `/2`。
3. 每 shard 的 vLLM 靶在 config 期固定：默认 `:8001`（`config.yaml:75`），`mk_cfg` sed 出 `:8002`（`launch_dual.sh:43-46`）——**无 fallback URL 列表、无路由**。
4. source_qa 写死 `:8002`（`config.py:41`），不知道有第二个副本。
5. `iqa` 写死 `cuda:0`（`config.py:54`）——即便 GPU1 空也用不上。

**今日恶果**：只有 `reason_g1`(:8002) 起、`reason_g0`(:8001) 没起时，build `--shard 0/2`（默认 config → :8001）**找不到副本**，而 `reason_g1` 84GB 干等。系统无法 rebalance：每个客户端静态绑死一个端口，shard 数静态 `/2`。

**但核验发现弹性其实唾手可得**：`run.py:740` `if pos % shard_n != shard_i` + `_parse_shard_flag`（`run.py:580-590`）接受**任意 `i/n`**。`/2` 纯属 launcher 约定，非代码约束。`config.VLLM_BASE_URL` 与 `SOURCE_QA_IQA_DEVICE` 都已是 env 可覆盖。**改动面比看起来小得多。**

---

## 2. 设计目标与约束

| 目标 | 判据 |
|---|---|
| 消灭 vLLM 空转 | least-outstanding 路由，副本不再 0%/过载并存 |
| 吸收 1↔2 卡 | 上层只认一个 broker URL，卡数变化只改一个预算数字 |
| 统一两套框架 | cleaner 与 judge 共享一个池 + 一个全局预算 + 优先级公平 |
| 分池 / 分队列 / 阶段 | 5 池、每池队列+背压、阶段图按资源需求向量调度 |
| 不依赖 gflow | broker 用进程内 semaphore/队列仲裁毫秒级请求 |
| 保持正确性 | render_lock / 先清洗后消费 / PG 写语义 / ppr10k fan-out / SAM3 env 隔离 全部不破 |
| 增量可回滚 | 每步独立 shippable，先小后大 |

---

## 3. 统一并发模型：vGate

### 3.1 分池（5 池）

1. **vllm-inference（逻辑池，单一 broker URL，如 `:8003`，drop-in OpenAI `/v1`）**——DOMINANT。成员：build cleaner、`llm_qa`、`preset_qa` 问卷 C、`tag_precompute` live miss。后端 1–2 个活副本。
2. **gpu-compute**——起步用 supervisor 设容量的**每卡计数 semaphore**，后续升级为独立 `:8004` 租约服务。成员：teacher renderer（仍受 `render_lock`）、`iqa`、stage2 NR-IQA、CLIP 美学。**排除 `sam3_precompute`**（独立 env，单独 sharded stage-0 pass，绝不作同卡租约成员）。
3. **cpu**——不托管，OS 调度，照旧：`dedup(16)`、stage1、calibrate、gate、build `_build_cgt`。
4. **lr-farm**——照旧，`render_jobs` PG 表即持久队列，并发提交 cap 3。
5. **pg**——照旧，单连接/runner，MVCC，`write_retry` / `FOR UPDATE SKIP LOCKED`。

### 3.2 vLLM broker 机制（vGate 核心）

- **副本发现**：对候选端口列表 `[8001,8002,...]` 每 ~10s 探 `GET /v1/models`（即 `launch_reasoning.sh:81` / `launch_dual.sh:38` 已用的就绪检查）；副本 LIVE ⟺ 服务的 model name 匹配。**无静态 0/2//1/2 绑定**。必须断言**所有活副本 served-name 一致**（`config.yaml:77 == launch_reasoning.sh:46`），否则拒绝跨异构模型路由。
- **准入控制**：**一个全局预算 = Σ 活副本 × `单副本上限`**（起步 ~28–32，按 fp8 35B 在 0.85 mem-util 下的 KV 头寸调）。2 副本 → ~64；1 副本 → ~32。**这一个数字就是 1↔2 卡问题的全部落点**。三个私有 cap（32/16/无）就此塌缩成一个。
- **优先级分类（分队列）**：3 条加权公平队列，靠一行 `X-vgate-class` 头打标：
  - **P0 build-annotate**（gen_instruction/reason_params/verify/tag）——最高权重（SFT 交付物 + verify 与 render 重叠延迟耦合）。
  - **P1 qa-judge**（`llm_qa` A/B + 问卷 C）——门控 pass，须先于 build 消费某资产完成，但单调用不在关键延迟路径。
  - **P2 tag/aesthetic backfill**——最低。
  - 加权公平（如 4:2:1）+ **最大排队等待提升**（防饿死），任一方都不会把另一方饿死。
- **路由**：活副本中 **least-outstanding**（argmin inflight），tie-break round-robin。**这就是 `reason_g1 84GB@0%` 的直接解**。
- **背压**：每类有界队列 → HTTP 429 + `Retry-After`。build cleaner（`vlm_clean.py:149-180`）与 llm_qa（自带 3× 重试）已能吸收；**⚠ `preset_qa` 问卷 C 无重试**（`preset_qa.py:294` 异常即返回 None）——见开放决策。
- **载荷**：broker 保持**瘦反向代理**，admission+routing only，流式转发不缓冲整个多模态 body（base64 data-URI 可达 MB 级）；先压测纯透传再加逻辑。

> 与 LR 农场同构：vGate 结构上就是 `lrc_task_server.py:91-92` 那套 `asyncio.Lock + Condition + pending/active 映射` 反向 broker，只是后端从"反向轮询的 LrC client"换成"正向 HTTP 转发到 vLLM 副本"。**仓库内已有验证过的范本可抄。**

### 3.3 弹性 GPU supervisor（1 vs 2 卡）

一个轻量 daemon（~80 行）替掉 `launch_reasoning.sh` 的 all-or-nothing start。每 ~15s 探 `nvidia-smi` 空闲显存/利用率，调谐期望副本数，**复用一模一样的 `docker run`（`launch_reasoning.sh:60-71`，含持久编译缓存挂载 `/home/bc/data/vllm_cache`，故重启跳过 ~30min 重编译）**，把活副本集注册进 broker、把活副本数写给 launcher。

- **2 卡**：两副本都起，least-outstanding 喂满两卡；renderer + IQA 取每卡 ~13GB 头寸经 gpu-compute semaphore，调度到 vLLM 头寸更足的卡（业务逻辑里去掉 per-shard `CUDA_VISIBLE_DEVICES` 钉死，`render.py:112` 设备改为 leased index）。**重型独立 IQA pass** 时 supervisor 可把一卡降级为 `gpu_compute` 角色（停该副本→预算减半→存活副本经 least-outstanding 吸收全部 vLLM）再升回——**带 hysteresis/min-dwell 防抖**。
- **1 卡**：只起一个副本（即今日"只有 reason_g1"的现实），预算自动减半。renderer + 单副本**时分共卡**——现有 `MEMUTIL 0.55/0.85` 共存切分已为此设计，且**多数 stream 不渲染**（`run.py:705-719`），renderer 只在 `render_lock` 短爆发里占卡，其余时间 vLLM 独享。共卡须**下调 vLLM mem-util（如 0.80）保 renderer 头寸**或经 semaphore 硬串行 render vs vLLM，**不可乐观超订 97GB**（见开放决策）。
- **弹性切换 2→1 / 1→2**：supervisor 增删副本并更新 broker 活副本集；垂死副本上的在途请求被 broker 重排（无状态 chat，幂等；verify/judge 需写时去重，靠 `run.py` done_ids 与 `llm_qa.py:341` `NOT EXISTS`）。上层只见 429/延迟抖动，**永不见连接到死端口**。
- **弹性 shard 数**：`launch_dual.sh` 写死的 `--shard 0/2 & 1/2` 改成 `for i in 0..N-1: --shard i/N`，N 取自 supervisor；`run.py:740` 已接受任意 `i/n`。ShardWriter 的文件分片 `/N` 可与副本数解耦独立保留。

### 3.4 分队列 + 背压（每池）

| 池 | 队列 | 背压 |
|---|---|---|
| vllm | broker 内 3 条优先级队列 | 有界 → 429/Retry-After |
| gpu-compute | 每卡计数 semaphore（或 `:8004` 租约），按 kind FIFO 等待 | lease-block；**严格获取序：先 lease 后 render_lock，按 kind 不可重入**（否则死锁） |
| lr-farm | 现有 `render_jobs` PG 表 | cap 3 并发 |
| cpu / pg | 现有 ThreadPool 尺寸 + SKIP LOCKED | 同上 |
| build RAM 守卫 | —— | 保留 `max_outstanding_samples=32`（`config.yaml:88`）作 held-after-array 内存帽，独立于 broker 准入 |

### 3.5 阶段图（阶段）+ 逐资产闭环闸门

```
sqa:   ingest ─┬ dedup ─ iqa ─┬ llm_qa(vllm P1) ┐
               │              └ calibrate ───────┤
               └ preset_s1 ─ preset_s2(lr ‖ gpu-iqa ‖ vllm P1) ┐
                  {iqa,llm_qa,calibrate} ─ gate ─ apply ────────┤
接缝(逐资产): apply[asset] ─ 解锁 build 对该资产的 plan 资格 ─────┤
build s0: tag_precompute(vllm P2) / sam3_precompute(独立 env, gpu, sharded) ─喂缓存─┤
                                                                                   ▼
build S1-S8: plan ─(after_needed 才 render: gpu-compute)─ annotate(vllm P0) ─ verify(vllm P0) ─ ShardWriter(该 shard 进程主线程)
```

**先清洗后消费——改成逐资产数据依赖，而非整框架壁垒**：`apply` 在 PG 写 `auto_verdict`；build planner `_planned_items`（`run.py:734`）加只读谓词"该资产 `auto_verdict ∈ {keep, needs_local_render}`"。于是 QA 把判级逐资产流入 PG，build 持续抽走已清资产，**两框架真正并发**——source_qa 的 vLLM-空闲阶段（dedup CPU / iqa GPU）经共享 broker 与 build-annotate / tag_precompute 重叠，填平 §1.4 空转清单，而"绝不消费未清资产"的序不变。

> **核验澄清**：source_index 是单一共享混合文件（非按 corpus 分文件），故重叠**只能走 PG 行级谓词**，按 corpus 分文件那条路不存在。谓词必须保守（仅 keep/needs_local_render），使 MVCC 快照时序竞态最坏只是漏掉刚清的资产（无害，下轮重试），绝不放进未清的。`apply` 仍照写 JSONL 快照供复现。
> **前置依赖**：本闸门预设 `apply` 已接通（当前 `config.yaml:308-309` cleaned 指向被注释，闭环未启）。Step 4 之前须先按 SOURCE_QA_REVIEW §7.1 把 apply 闭环接上。

---

## 4. 必须保持的不变量（任何实现都不可破）

1. **render_lock teacher 串行**：renderer 非线程安全；`render_lock` 须**恰好且仅**包住 `renderer.render()`（`streams.py:598-601,640-641`），保持 per-process `threading.Lock`。broker 不加任何 GPU 活，故无新争用。若加 gpu-compute 租约：**先 lease 后 render_lock，按 kind 不可重入**，否则死锁。
2. **先清洗后消费**：build 读 `storage.source_index` 无自动 fallback；逐资产闸门须保守（仅 keep/needs_local_render）；`apply` 仍出 JSONL 快照。
3. **ppr10k 三专家 fan-out**：a/b/c 同文件兄弟，`apply` 把簇头判级继承给兄弟而非当跨文件重复删（`apply.py:13-22` rule 3）。broker/supervisor 一概不碰。
4. **SAM3 env 隔离**：`sam3_precompute` 跑 `monetgpt_sam3` env、**不能与 renderer 共存**（`sam3_precompute.py:3`）；它是独立进程/独立 env 的 sharded stage-0，输出供 `CachedMasker` 读。**绝不可建模为同卡租约成员或同进程单例**（提案 1/3 原样均违反此点，已在裁决中纠正）。
5. **PG 写语义**：单连接/runner，MVCC 读不阻塞写，写经 `write_lock`/`db_lock` + `write_retry` 8×（`db.py:355-423`）。新队列认领用 `FOR UPDATE SKIP LOCKED`。build 今日不碰 PG；Step 4 只加只读谓词连接，PG 故障须优雅降级 build planning，不得损坏。
6. **ShardWriter 单线程**：`write()` per shard 进程主线程（`run.py:759-774,950`，原子 tmp+fsync+rename `masking.py:552-576`）。不可跨线程扇出写。
7. **served-name 跨副本一致**：least-outstanding 假设所有活副本同 model name；broker 发现时须断言并拒绝跨异构名路由。
8. **重排请求 at-least-once + 写时去重**：垂死副本上重排在途请求可能重复执行；annotation 幂等，verify/judge 靠现有 done_ids（`run.py`）/ `NOT EXISTS`（`llm_qa.py:341`）去重。
9. **LR 农场 catalog 膨胀天花板**：并发提交 ~3（6% 失败）而非 6（62%），见 ROBUST_RENDER_REQUIREMENTS。
10. **render-only-when-needed**：teacher after 仅 `after_needed` 时渲染（`run.py:705-719`）；GPU 调度设计须依赖"renderer 多数时间空闲"这一事实，而非假设它常热。

---

## 5. 迁移路线（增量 5 步，每步独立 shippable + 可回滚）

| Step | 内容 | 改动面 | 收益 |
|---|---|---|---|
| **1** | vGate 作**透明 1:1 透传代理**：`config.yaml:75` base_url + `SOURCE_QA_VLLM`(`config.py:41`) + `preset_qa` 的 `VLLM_BASE_URL` 全指 `:8003`；加副本发现 + least-outstanding。**QwenVLCleaner/llm_qa/preset_qa 传输零改动** | ~250 行 FastAPI + 一处配置重指 | **立消** `reason_g1` 空转 + "shard 打到死端口"故障 |
| **2** | 加 `X-vgate-class` 头（`vlm_clean.py:619`=build-annotate；llm_qa + 问卷 C=qa-judge）+ 全局预算 + 加权公平队列；三私有 cap 降级为无害上界。给问卷 C 加有界重试或为该类改 broker-block | 3 处一行 + broker 逻辑 | 三池合一，公平准入 |
| **3** | supervisor 接管 docker 生命周期（复用 `launch_reasoning.sh:60-71`）；`launch_dual.sh` 改单 orchestrator 弹性 N；`IQA_DEVICE` 走 supervisor 账本（已 env） | 替 launcher + ~80 行 daemon | 真正 1↔2 卡弹性 |
| **4** | 逐资产 cleared 闸门：`run.py:734` `_planned_items` 加只读 PG 谓词（`auto_verdict ∈ keep/needs_local_render`）；`apply` 仍写 JSONL 快照。**前置：先接通 apply 闭环** | build 加只读 PG 连接 | QA+build 窗口重叠 |
| **5**（可选 fast-follow，来自提案 3） | gpu-compute semaphore 升级为 `:8004` 租约服务，render/IQA/aesthetic 显式分时共卡（严格锁序）；可选把 sqa CLI 阶段改成 SKIP-LOCKED 常驻 pool worker | 2 小服务 | 攻非 vLLM 的 GPU 空转 |

**复用不重写**：`render.py`+`render_lock`、`masking.py`、`aesthetic.py`、`db.py`(conn+write_retry+render_jobs)、gate/calibrate/apply、整个 LR 农场、所有 prompt/parser、`QwenVLCleaner` image LRU/encode-sem、`CachedTagger`/`CachedMasker`、持久编译缓存。
**新增**：vGate(~250 行) + supervisor(~80 行) + PG 谓词。**删除**：`mk_cfg` sed、all-or-nothing start、per-shard 端口绑定。

> **Step 1–2 已交付两个 must-have（消空转 + 1↔2 卡弹性）**。Step 3–5 各自再加一个 daemon/耦合，可按需推进。

---

## 6. 开放决策（需用户拍板）

1. **单副本上限定值**：broker 外部 per-replica cap 是 fp8 35B-A3B MoE 在 0.85 mem-util 下 KV 压力的**代理**，非测量。太高 → OOM/抢占风暴（正是 `config.py:45` "cap to avoid engine overload" 要防的）；太低 → 丢吞吐。建议起 32 对真实 KV 头寸二分。
2. **问卷 C 无重试**（`preset_qa.py:294`）：429 背压下会静默丢 QA。(a) 给该调用点加有界重试，还是 (b) 让 broker 对 qa-judge 类改阻塞 long-poll 而非 429？
3. **1 卡共存策略**：(a) 降 vLLM mem-util 保 renderer 头寸 / (b) semaphore 硬串行 render vs vLLM（接受 render 延迟尖峰）/ (c) 给 renderer 保留时间片或抢占预算？renderer native-res 可达 ~2s，在 97GB 卡上对 84GB 副本有 OOM/抖动风险。
4. **整卡降级/升级**（提案 3）：把一卡整给重型 IQA pass 的吞吐收益，是否值 docker stop/start 抖动 + vLLM 预算减半？hysteresis/min-dwell 取值？
5. **范围/紧迫度**：Step 1–2 已交付两个 must-have。是先到此为止，还是本轮直接推到 Step 3–5？
6. **SPOF 容忍**：broker 成为全部推理的单一前置依赖（今日端口故障只伤一个 shard）。瘦+自动重启的 localhost broker 可接受，还是要保留"client 在 broker 不可达时直连已知端口"的兜底路径？

---

## 附：与裁决的偏差与作者修正

- 裁决把 `iqa` 的 `cuda:0` 称 "hardcode 需 de-hardcode"，但**它已是 env 可覆盖**（`SOURCE_QA_IQA_DEVICE`，`config.py:54`），Step 3 仅改取值来源，trivial。
- 裁决的"overlap 按 corpus 分文件"被作者核验否定：source_index 单一混合文件 → 只能逐资产 PG 谓词（已并入 §3.5）。
- 提案 1（单进程 in-process gateway）的优先级分类法（P0/P1/P2）被采纳并入骨干；其"把两框架并进一个 OS 进程"因丢失进程隔离/与同步 OpenAI client+独立 conda env 现实冲突而**不采纳**打包方式。
- 提案 3 的 GPU 计算租约**思路正确但 as-written 有缺陷**（把 SAM3 列为同卡租约成员违反 env 隔离；lease/render_lock 双锁死锁；共卡 mem-util 算术脆弱），故作为 Step 5 **修正后**的 fast-follow，而非第一天上。
