# TOOL-MaskBackfill-1 · 实施前核实记录 / 逐块 diff 理由 / 待决策项

日期：2026-08-12　执行环境：`/home/bc/VeraRetouch`（branch `lens-exp`）

---

## 一、实施前核实（任务卡「已核实事实」逐条复核）

| # | 任务卡陈述 | 复核结果 |
|---|---|---|
| 1 | catalog `samples` / `"group"='cache/subject'` / `ineligible_reason='subject_not_ready'` = 20,009 | **成立**。`cache/subject` 共 56,778 行，reason 直方图 `{None: 34534, subject_not_ready: 20009, subject_mask_area_guard: 2234, not_a_cache_entry: 1}` |
| 1 | 源路径要从 PG 反查，`sample_id = 'subject_' + sha1(path)[:16]` | **成立且必需**。这 20,009 行的 `meta.source_path` 与 `meta.asset_id` 都是 `null`，catalog 自己给不出源路径；目录名是唯一线索 |
| 1 | PG `vera_source_qa` 56,777 行、20,009 条全部能反查 | **成立**。56,777 行 → 56,777 个唯一 path_key（**0 碰撞**），20,009 全部命中，0 未命中 |
| 1 | PG DSN | **不成立（改正）**。`config.PG_DSN` 默认 `postgresql:///vera_source_qa` 走 unix socket，本机没有；库实际在 docker 容器 `research_pg` 里，走 `postgresql://research:research@127.0.0.1:5432/vera_source_qa` |
| 2 | 310 张在本地、19,699 张只在归档 | **成立且是干净划分**：310 张全部是 `/home/bc/datasets/MMArt-PPR10k/global/*/before.jpg`，且**全部不在归档**；19,699 张全部在归档，且**全部不在本地** |
| 2 | 归档组 | 8 组：unsplash_work 11,046 / awards 3,212 / raise6k 2,827 / fivek_gold 1,315 / quandian 629 / ppr10k 361 / korean 283 / greysky 26；149 个 shard；31.45 GB（均值 1.60 MB） |
| 2 | root 要改写到 `/mnt/nfs-ro` | **成立且是硬约束**。catalog `groups.root` 全部是 `/mnt/nfs/bc/data/datasets/...`，即 hard 挂载。任何未改写 root 的 `archive_reader` 调用都会碰硬挂载 |
| 3 | 续跑断点 = subject_cache 里 subject.json 存在性 | 成立。本地 subject_cache 现有 **6,376** 个目录，与这 20,009 个 key **交集为 0**，所以坏条目不会被碰、也不会挡住补跑 |
| 3 | `_data_uri(row["source_path"])` 是要修的断点 | 成立，但**不止这一处**：还有 `_call_selector` 的 `_data_uri`、`_numbered_overlay` 的 `_preview_source`、`_sam_proposals` 与 `_sam_forward_batch` 的 `masker._as_pil`，一共 5 处像素读 |
| 3 | gpu0 当前完全空闲 | **不成立**。gpu0 上有 `q3vl.whereb.scripts.run_amort_arm --arm P3prime`（31 GB / pid 3743391）。冒烟仍用 cuda:0，SAM3 bf16 只占几 GB，没有干扰；全量跑批前主 agent 应确认该实验状态 |
| 5 | 外部 relay / 模型 gpt-5.6-terra / effort low | **成立**。`api.tokenskingdom.com/v1` + provider-c-lane-1 可用，`response.model` 精确回显 `'gpt-5.6-terra'` |
| 5 | 用 L6 TOML | 可用，但**L6 的 `external_model` 是 `gpt-5.6-luna`**；`databuild.prod-l7-local400k-20260811.toml` 里同一个 endpoint（base_url / api_key sha256 前 8 位 `3b1ad0d3` / concurrency 16 完全一致）声明的才是 `gpt-5.6-terra` + effort low + max_output_tokens 6000。全量建议改用 L7 TOML，语义自洽 |
| 6 | 6,375 个 `subject_label_error` 坏条目不要动 | 成立，且结构上碰不到（key 无交集 + 续跑跳过） |

### 核实过程中发现的、任务卡没有的事实

1. **`SUBJECT_SELECTION_SCHEMA` 在外部 relay 上 100% 失败**。第一次单张冒烟拿到
   `transport_skipped`，二分定位到：schema 里 `instance_ids.uniqueItems` 这一个关键字
   会让 relay 回 **Cloudflare 502 origin_bad_gateway**（不是 400，不报字段名）。
   对照实验：`as-is` ERR / `去掉 uniqueItems` OK / `去掉 maxItems` ERR / `去掉 items.minimum` ERR。
   `SUBJECT_LABEL_SCHEMA` 没有 `uniqueItems`，所以 label 调用一直是好的——只有 selector 挂，
   而 `finalize` 把 selector 的 transport_error 记成 `transport_skipped`，表面上看不出是 schema 问题。
2. **`request_text` 的重试没有任何 backoff**，三次连打，而 502 的 `retry_after` 是 60 s。
   schema 修好之后冒烟 0 失败，所以没有加 backoff（见「待决策」4）。
3. **26 张 `.dng`（全部 greysky）PIL 只能解出 192×256 的内嵌缩略图**。SAM3 会在缩略图分辨率上
   出 mask。这是既有行为（原管线也用 PIL），不是本次改动引入的。
4. **消费侧确实能看见新写的本地条目**：`build_inventory` 对本地 subject_cache 目录做
   overlay 逐条实检，并覆盖 catalog 预计算里同 `cache_dir` 的行（`sources.py:365-380`）。
   所以不必等 catalog 重建，补出来的条目就能进池。
5. **L7 build 正在用同一条 relay lane 的同一个 key 和同一个模型**。全量跑批会与它抢
   provider-c-lane-1 的 concurrency 16。

---

## 二、管线改动逐块理由

三个文件，135 插入 / 18 删除。原则：`source_path` 全程保持原始逻辑路径，只让像素读走别名；
外部 relay 的差异全部收在 transport 层，本地 vGate 路径逐字节不变。

### A. `dataset_build/tools/eval_subject_instance_selector.py`（+42/-13）

| 块 | 改动 | 为什么必须改 |
|---|---|---|
| A1 | 新增 `_read_path(row)`：`row.get("read_path") or row["source_path"]` | 19,699 张源图本地不存在。**不能**把 `source_path` 改成 scratch 路径：它是 `path_key` 的输入，也是 `subject.json` 要落的值，消费侧拿它过 `path_exists`。所以引入一个只服务于像素读的第二字段，两者语义分离 |
| A2 | `_sam_proposals` / `_call_subject_label` / `_call_selector` / `_numbered_overlay` / 三处报告渲染共 7 个读点改走 `_read_path` | 这就是 A1 的落地。漏一处就是一次 `source_unreadable` |
| A3 | `_proposals_from_result` 两个返回 dict 里加 `"read_path": row.get("read_path")` | `record` 是 SAM 之后 selector/overlay 唯一拿得到的载体，不透传就断链 |
| A4 | `_call_subject_label` / `_call_selector` 增加 `reasoning_effort` / `expect_model` / `max_output_tokens` 三个 keyword，默认值等于旧行为 | 外部 relay 需要 `reasoning.effort`；reasoning token 计入 output，320 的旧默认必然截断；模型回显要校验。三个参数默认 `None/None/320`，本地 vGate 调用完全不变 |

### B. `dataset_build/core/responses_vlm.py`（+65/-11）

| 块 | 改动 | 为什么必须改 |
|---|---|---|
| B1 | `consume_text` 拆成 `consume_response(stream) -> (text, model)`，`consume_text` 变成取第一项的薄包装 | 要拿 `response.model` 才能做偷换校验。保留 `consume_text` 原签名：`tests/test_canonical_responses.py:598` 直接调它 |
| B2 | `ResponsesText` 加 `model: str \| None = None`（带默认值，位置参数兼容） | 把回显模型交回调用方 |
| B3 | `request_text` 加 `reasoning_effort`：非 None 时 payload 用 `{"reasoning": {"effort": ...}}`，None 时才放 `extra_body.chat_template_kwargs` | `chat_template_kwargs` 是 vLLM 专有；hosted reasoning 模型两者互斥。**默认 None ⇒ 旧 payload 逐键不变** |
| B4 | `request_text` 加 `expect_model`：回显 ≠ 期望即抛 `ModelSubstituted`，落进原有的 `attempts=3` 重试循环 | 任务卡要求「精确等值校验 + 失败重试 ≤3 次」。复用既有循环，不新增重试机制 |
| B5 | `_hosted_schema()`：只在 `reasoning_effort is not None`（即外部路由）时从 schema 里剥掉 `uniqueItems` | 上面发现 1。**唯一性没有丢**：`_parse_selection` 本来就在本地校验 `len(ids) != len(set(ids))`，重复 id 会被判 `parse_error`。做成路由内可见而不是直接改那个冻结常量，是为了让本地 vGate 的 schema 一个字节都不动 |

### C. `dataset_build/source_qa/sam3_subject_instances.py`（+46/-6）

| 块 | 改动 | 为什么必须改 |
|---|---|---|
| C1 | `_sam_forward_batch` 的 `masker._as_pil` 改走 `r.get("read_path") or r["source_path"]` | 批前向是这个文件里唯一的像素读点（逐张回退那条在 A2 已覆盖） |
| C2 | `run()` 里组一个 `vlm_opts` dict，两个调用点 `**vlm_opts` | 把 A4 的三个新参数 + 原有 api_key 一起透传。全部用 `getattr(args, ..., 默认)`，所以 `relabel_sources()` 里那个手搓 Namespace 不用改也不会 AttributeError |
| C3 | `main()` 新增 `--vlm-config / --vlm-endpoint / --vlm-effort / --vlm-max-output-tokens / --vlm-verify-model` | key 只能从 0600 TOML 读，不能进 argv |
| C4 | `_external_endpoint()` 读 TOML 取 base_url + api_key | 同上；只在启动时读一次进进程，不打印、不落盘、不写进 subject.json |

**没有改的**：`--min-score 0.30 / --dedupe-iou 0.92 / --max-proposals 16 / --focus-radius 0.025 /
--max-mask-area 0.85 / --max-group-envelope 0.67 / seed 20260711 / MASK_LONG_EDGE 1536 /
`_data_uri` long_edge 1024 / 两段 prompt / `clean_mask` / 续跑与分片逻辑 / 落盘 schema。
`dataset_build/src/construct/` 一行未动（只读参考）。

### 回归

`pytest dataset_build/tests/{test_canonical_responses,test_sam3_batch_fallback,test_selector_speculative_parity,test_sources_archive_fallback,test_archive_reader,test_prefetch_and_sft_pack,test_inventory_sql}.py`
→ **95 passed, 1 failed**。唯一失败是
`RequestShapeTests::test_pinned_sdk_and_strict_request_shapes`：`construct` 硬性要求
`openai==2.46.0`，而跑测试的 `monetgpt_sam3` env 是 2.21.0——纯环境 pin，与本次改动无关
（`databuild` env 是 2.46.0 但没装 pytest）。

---

## 三、假设清单（已自行核实的部分不再列）

1. **假设**：`subject.json` 的 `source_path` 必须是原始逻辑路径。
   **核实**：`construct/sources.py:183-201` —— 拿它过 `path_exists` 再 `read_bytes` 解码。已用
   `verify_consumer.py` 在 4 个「源图只在归档里」的 ready 条目上实证通过。
2. **假设**：预取缓冲能让消费侧闸门不碰硬挂载。
   **核实**：`archive_reader._local_or_prefetch` 的读梯是 本地 → prefetch → archive；缓冲命中就
   永远不会走到 `read()`。`verify_consumer.py` 在调闸门前先确认缓冲文件在，不在就跳过并报错。
3. **假设**：冒烟里 32 次 relay 调用的 `response.model` 全是 `gpt-5.6-terra`。
   **核实**：`--vlm-verify-model` 打开时，回显不等值会抛 `ModelSubstituted` → 三次重试耗尽 →
   `transport_error` → 该行记 `transport_skipped`。冒烟 16/16 全部落到正常 status、
   **transport_skipped = 0**，所以 32 次调用无一被判偷换。另有 3 次独立探针直接打印回显，
   均为 `'gpt-5.6-terra'`（label 4 次 + selector 4 次的用量探针也是 `models={'gpt-5.6-terra'}`）。
4. **未验证**：全量 20,009 张里是否存在会让 SAM3 OOM 的超大图。已有逐张回退兜底，未专门压测。
5. **未验证**：relay 在 16 并发下的稳定性。冒烟只到 4 并发。

---

## 四、待主 agent 决策

1. **全量跑批写哪个 cache root。** 冒烟写的是
   `/home/bc/data/scratch/mask_backfill/smoke_cache`（与生产树隔离，可随时删）。
   全量默认应写生产 `…/vera_directionA_1M/subject_cache`——但 L7 build 正在跑，而
   `build_inventory` 会 overlay 扫这个目录。L7 的 inventory 是 13 h 前建的、进程内不重扫，
   风险应该为零，但这个判断需要你确认。**保守默认：先写 staging 目录，L7 结束后再 rsync 进生产。**
2. **用哪个 TOML。** 任务卡指定 L6；但 L6 的 `external_model` 是 `gpt-5.6-luna`，
   L7 才是 `gpt-5.6-terra`（endpoint / key / concurrency 三者完全相同）。
   **保守默认：命令行显式 `--vlm-model gpt-5.6-terra`（已如此），TOML 只用来取 base_url+key；
   建议全量改指 L7 TOML。**
3. **`--vlm-workers` 取多少。** endpoint 声明 concurrency=16，而 L7 build 正在用同一条 lane
   的同一个 key。两边加起来超过 16 会互相挤。**保守默认：8**（L7 在跑时），L7 结束后再提到 16。
4. **要不要给 `request_text` 加重试 backoff。** 现在三次重试连打，而 relay 502 带
   `retry_after: 60`。schema 修好后冒烟 0 失败，所以按 YAGNI 没加。若全量出现成片
   `transport_skipped`，加一个 5–15 s 的退避比提高 attempts 更有用（transport_skipped 不落盘，
   续跑会自动重跑，所以这只是吞吐问题，不是正确性问题）。
5. **`uniqueItems` 的处理位置。** 现在是路由内剥离（本地 vGate 的 schema 不变）。
   另一个选择是直接从 `SUBJECT_SELECTION_SCHEMA` 常量里删掉——更简单，但会同时改变本地路由
   在 v6 定稿后的 schema。**保守默认：保持路由内剥离。**
6. **模型校验用精确等值还是前缀。** 现在是精确等值（任务卡要求），实测回显就是
   `gpt-5.6-terra`。`construct/responses.py:1872` 用的是前缀匹配，理由是允许带日期的变体。
   若 relay 哪天开始回 `gpt-5.6-terra-2026-xx-xx`，精确校验会把整批打成 transport_skipped。
7. **31.5 GB scratch 的去留。** 跑完之后：删掉 → 后续消费侧 `_inspect_cache_dir` 会去读
   `/mnt/nfs` 上的 shard（生产本来就这么干）；留着 → 硬链接缓冲同时让消费侧免于碰硬挂载。
   **保守默认：留到 L7 之后的下一次全量 build 结束。**
8. **26 张 DNG。** PIL 只解得出 192×256 缩略图。是接受（几乎必然 `no_subject`／低质 mask）、
   还是从 pool 里剔除、还是补一个 rawpy 解码路径。**保守默认：接受，跑完看这 26 条的 status。**

---

## 五、全量成本外推（依据上表实测）

| 阶段 | 依据 | 20,009 张外推 |
|---|---|---|
| 预取 | 200 张实测 40.7 img/s / 58.2 MB/s（8 线程 + sha256） | 19,699 张、31.45 GB → **约 8 分钟**，磁盘 **31.5 GB**（硬链接缓冲 0 额外字节） |
| SAM3 | 16 张 ×2 轮实测 0.25–0.30 s/img（batch 8，长边 1536） | **1.4–1.7 h** GPU（与 VLM 重叠，非瓶颈） |
| VLM 调用数 | 冒烟 16 张里 8 张走到 selector（no_subject 6 + no_center_candidate 2 不走） | 20,009 label + **约 10,000** selector ≈ **30,000 次** |
| VLM token | label in 5,912 / out 98；selector in 6,250 / out 81（各 4 张实测均值） | **约 181M input tok**、**约 2.8M output tok**（output 含 reasoning） |
| 端到端墙钟 | 16 张 / 133 s @ workers=4（含 35 s 模型加载）；单调用 median label 19.0 s、selector 15.1 s | workers=16 且 lane 不被占：**约 8.5–12 h**；workers=8（与 L7 分 lane）：**约 17–24 h** |
| 产出预期 | 冒烟 ready 5/16 = 31% | **约 6,200 条**新的 eligible 源（其余落 no_subject / 守卫拒绝） |
