# EXEC-2 · generated-context 两卡生产作业 —— 准备记录（**未启动**）

任务：用选定的 Base SFT checkpoint 对全量样本自回归生成两段 reasoning
（`q3vl/whereb/scripts/make_generated_context.py`，schema `genwhere/2`），两张 H100 分片并行。
本目录只准备，**一行 GPU 都没碰**（全部 dry-run 都 `CUDA_VISIBLE_DEVICES=""`）。

引用文档节：`where_b/PREFLIGHT_WHERE_B_PENDING.md` §S2、`what/PREFLIGHT_WHAT_PENDING.md` WT-J10
与 §三-bis、`q3vl/whereb/{config,gencontext,data,stores,hiddens}.py`、
`q3vl/what/{config,stores}.py`、`q3vl/whereb/scripts/{make_generated_context.py,run_where_b.sh}`、
`q3vl/what/scripts/{run_what.py,evaluate_what.py}`、`s0_base_sft/chain/{joblib.sh,verify_checkpoints.py}`。

---

## 1 · 核实记录（全部本仓库/本机实测，未用外部检索）

| # | 事实 | 怎么核实的 |
|---|---|---|
| V1 | 生产脚本**没有**任何分片参数：只有 `--limit`（前 N 条），无 `--shard`/`--offset` | 读 `make_generated_context.py` 全文 + `--help` 实跑（`dryrun.log`） |
| V2 | `--report-dir` 下的文件名是 `genctx_<leaf>.json`，**leaf 不含分片号** | 同上，L165 |
| V3 | 发布是原子的且拒绝已存在目录（`_prepare_output` 用 `_lexists`） | `q3vl/data/shardio.py` L86 + `dataset_build/tools/indexed_tar.py` L655 |
| V4 | 消费方按 `manifest.json` + `indexes/shard-*.idx.jsonl` + `shards/*.tar` 定位，目录里多出的条目无害 | `q3vl/whereb/stores.py` `PublishedStore.__init__/read` |
| V5 | 样本量：train 159,215、V_where 896、V_what 897（`open_dataset` 无 build 剔除，等于索引行数） | 逐行解析三个 `*.index.jsonl`；`open_dataset` 实跑 n=896 |
| V6 | prompt 长度 p50 = **464** tok（visual p50 384），`total_tokens` p50 653、max 1142 | 索引自带 `n_visual_tokens`/`total_tokens`；`Sft2SegCollator.encode_one` 实跑 12 条取中位数 |
| V7 | CPU 侧每样本 **~31 ms** 且**不与 GPU 重叠**：`dataset[i]` 冷 24 ms（热 4.3 ms）+ `encode_one` 1.7 ms + `image_processor` 5.5 ms/图 | CPU 实测 48 条（V_where），见 §4 |
| V8 | 两卡是 H100 94 GB（97871 MiB），当前被 Base SFT 占 54 GB×2 | `nvidia-smi` |
| V9 | 模型 KV：36 层 × 8 kv-head × 128 head_dim × 2(KV) × 2 B = **144 KiB/token** | `/home/bc/data/models/Qwen3-VL-4B-Instruct/config.json` |
| V10 | GT 两段长度 p50 210 / p90 251 / p99 310 / **max 326** tok（64 条 V_where 实测） | `chain/verify_checkpoints.py` L703-705 注释；与 `whereb/config.py` 的 204/332 口径一致 |
| V11 | `run_where_b.py` 把 genctx 根**写死**成 `GENCTX_DIR/<split>`，无 CLI 覆盖 | 读 L128 与 argparse 段 |
| V12 | `run_what.py` / `evaluate_what.py` 用 `<root>/<split>/<mode>`，有 `--color-genctx` | 读 L223、L247、`evaluate_what.py` L179 |
| V13 | 生成作业把全部记录**攒在内存里最后一次性发布**（`records = list(...)` 后 `publish_generated`） | `make_generated_context.py` L143-150 |
| V14 | `generate_where` 是左 padding + KV cache + greedy，`prefix_ids` 用于 forced_color | `q3vl/whereb/hiddens.py` L227-268 |
| V15 | 消费侧**不校验** genctx 记录里的 `checkpoint` 与训练用的 SFT checkpoint 一致 | `assert_genctx_coverage` 只查覆盖；`ColorGenContextStore` 只查 schema/mode/覆盖 |
| V16 | **单步解码 ~20–27 ms**（B=1、`max_new=448`、FA2、bf16、H100）：单样本生成实测 8.1–12.5 s；checkpoint 加载 134.5 s | 链上 GPU0 验证作业的实时日志 `chain/logs/gpu0_ckpt_verify.log`（11:52） |
| V17 | 生成记录里的 `generated_ids` 含**整批统一宽度**的尾部 pad/eos；下游 `extract_segment` 先切 eos 再切 `</where>`，所以**上下文本身没被污染**，但 `n_generated_tokens` / summary 的 `generated_tokens` 是"该批的解码步数"而非样本自身长度 | 读 `hiddens.generate_where` L267-268 与 `context.extract_segment` L186-187 |

---

## 2 · 分片方案

生产脚本一行未改。`genctx_shard.py` 在**包自己的工厂** `open_dataset()` 建好数据集之后切
`ds.refs`，然后把控制权交回脚本自己的 `main()`——切片之后的一切（prompt、生成、记录、发布）
都是原作者的代码。

* **切法**：`interleave`（`refs[i::n]`，默认）。索引本身按 sample_id 哈希序，build 已充分打散
  （train 159,215 条里有 142,365 次相邻 build 变化），所以连续切也行；选交错是因为它对
  "两卡同时收工"更稳，且分片大小最多差 1。`contiguous` 保留作为可复现的备选。
* **强制隔离**：wrapper 自己给 `--out-root` 和 `--report-dir` 追加 `shard<i>of<n>`。
  - 不隔离 out-root：第二个分片在**跑完几小时之后**才因目录已存在而死（发布是原子的）；
  - 不隔离 report-dir：**8 个分片静默互相覆盖同一个 `genctx_train.json`**，最后留下一个
    看起来完整、其实只描述 1/8 工作量的报告——正是 Stage-What R5 事故那一类。
* **`--limit` 语义**：先对整个 split 生效，再切分片（`--limit 64 --num-shards 8` = 全 split 取 64 条、
  每片 8 条），标定用。
* **合并**：`merge_genctx.py` 把各分片的记录流回**原作者的** `publish_generated()` 重新发布成
  一个根。不手工拼 tar/index：那等于重写 manifest + 每分片摘要 + SQLite catalog。
  合并前逐项断言（见 §3），合并后用消费方自己的类读回。
* **分片数**：train 默认 8（每卡 4 段、每段 ~1 h），小 split 2。多于卡数是有意的——
  见 V13，作业中途被杀会**丢掉全部产出**，8 片把最坏损失压到 1/8，代价是多 6 次模型加载（~3%）。
  已完成的分片重跑时自动跳过（manifest `status: complete`）。

---

## 3 · 合并的断言（全部在发布任何字节之前）

1. 每个分片根存在、`manifest.status == complete`、索引可读；
2. 每个分片的 sample_id 集合 **恰好等于**它应该分到的那一片——不是"互不相交且并集覆盖"。
   两个分片各自用错 `--shard` 但恰好互补时，只有前者能抓住；
3. 并集 == `open_dataset(split)` 的样本集（同一个 build 过滤口径），不多不少；
4. 每条记录 `schema_version == genwhere/2`、`mode` == 本次要的、`checkpoint` **全 split 唯一**
   （半个 split 由 2488 生成、半个由 4976 生成，在 loss 里完全看不出来）。

合并后再用 `GenContextStore`（Where-B）与 `ColorGenContextStore`（Stage-What）读回，
写 `genctx_<leaf>.json`（已存在则写 `.merged.json`，`--force-report` 才覆盖并留 `.superseded`）。

---

## 4 · 墙钟预估（**模型推算，不是实测**）

单批耗时 = CPU 装配 + prefill + 解码步数 × 单步时延：

| 项 | 数值 | 来源 |
|---|---|---|
| CPU 装配 | B × 31 ms（**不与 GPU 重叠**，循环里是先取样本再 generate） | V7 实测 |
| prefill | B=64 时 ~0.8 s（文本 64×464 tok + 视觉 64×1536 patch） | 由 V6 推算 |
| 解码步数 S | HF `generate` 等**整批都结束**才停：正常终止时 S ≈ 批内最大长度 ≈ 330（V10 max 326+4 标签）；只要批里有一条不吐 EOS，S 就是 `max_new_tokens` = 512。B=64、不终止率 1% → 47% 的批会跑满，2% → 73% | V10 + `generate_where` 语义 |
| 单步时延 | **B=1 实测 20–27 ms**（V16）；B=64 估 25–35 ms（权重 8.8 GB + KV 8.6 GB 的带宽下限约 7 ms，其余是 HF 的 python/launch 开销，随 B 增长很慢） | V16 实测 + V8/V9 推算 |

| 场景 | 每样本 | train 分两卡 | + V_where/V_what |
|---|---|---|---|
| 乐观（S=420、单步 25 ms） | 0.20 s | **约 4.5 h** | +6 min |
| 悲观（S=512、单步 35 ms） | 0.31 s | **约 7 h** | +9 min |

* `two_segment`（train + V_where + V_what，161,008 条）：**两卡 ~4–7 h**；
* `forced_color`（train + V_what，160,112 条）：**再来一遍 ~4–7 h**；
* 单卡跑就是翻倍（7–13 h）。

**这不是"空档填充"级别的作业**，请主 agent 按此排期（§6 决策 D1）。

**怎么把估算换成实测（两个现成的读数）**：
1. GPU0 的 checkpoint 验证作业跑完后，`s0_base_sft/checkpoint_verification.json` 里
   `summary.generated_tokens` 分位数（S 的真值）与 `generation_seconds`（batch 8、max_new 448）；
   日志里还有每批 `generated (X.Xs this batch)`。
2. 本作业起来约 2 分钟后，分片日志的第一行 `{"done": 200, ..., "samples_per_s": x, "eta_s": y}`
   就是该卡的真实吞吐；`ETA_小时 = 每卡样本数 / samples_per_s / 3600`。
   两卡各自的 shard00 日志都要看。

**显存**：B=64 时估 ~25 GiB（权重 8.8 + KV 8.6 + 视觉激活/pixel_values 若干），94 GB 卡上很宽裕；
B=128 估 ~40 GiB 也进得去。默认取 **64**（保守），标定后可上调；作业报告里有 `peak_memory_gib`。

---

## 5 · 发现的问题

### CX-1（阻塞级，路径口径冲突）——两个消费方对同一份产物的路径约定不一致

* Where-B：`GENCTX_DIR/<split>`，写死在 `run_where_b.py` L128，无 CLI 覆盖（V11）；
* Stage-What：`<root>/<split>/<mode>`，`run_what.py` L223 / `evaluate_what.py` L179（V12）。

即 `genwhere/train` 必须**同时**是"一个已发布数据集根"和"一个按 mode 分的目录"。
`PREFLIGHT_WHERE_B_PENDING.md` §S2 写的产物路径是前者，WT-J10 写的是"每 split × 每 mode 一套"，
双方都没发现对不上。

**本目录采用的解法**（零代码改动、零重复 GPU 开销）：真实发布放 Where-B 的位置，
再在里面放两个兼容软链——

```
genwhere/train/                     ← 真发布（two_segment），Where-B 直接读
genwhere/train/two_segment  -> ../train
genwhere/train/forced_color -> ../train-forced_color
genwhere/train-forced_color/        ← 真发布（forced_color）
```

消费方只看 `manifest.json`/`indexes/`/`shards/`（V4），多出来的软链是惰性的。
已在 dry-run 里用**真的** `ColorGenContextStore(link, mode=...)` + `assert_covers` 验证读得通，
且 `mode` 不匹配照样硬停。**备选**是给 `run_where_b.py` 加一个 `--genctx` 参数（一行），
但那是改 q3vl 既有文件，本任务卡禁止 → 见 §6 决策 D3。

### CX-2（静默级）——没有人校验 genctx 的 checkpoint 与训练用的 checkpoint 一致（V15）

`assert_genctx_coverage` 只查覆盖，`ColorGenContextStore` 只查 schema/mode/覆盖。
若 genctx 用 2488 生成、而 `run_where_b.py`/`run_what.py` 走默认的
`SFT_CHECKPOINT = .../checkpoint-4976`，则"生成上下文"的 ids 来自 A 模型、重放它的冻结 VLM 是
B 模型——**loss 里完全看不出来**，只会表现为"generated 档不如 gt 档"，和 s 缓存契约里
"第二种失败模式是静默的"同构。

本目录的对策：合并断言 checkpoint 全 split 唯一，并把它写进 `genctx_<leaf>.json` 的顶层字段，
作为可审计的凭据。**主 agent 裁定 2488 时，必须同时给 Where-B / What 的启动命令显式传同一个
`--checkpoint` / `--sft-checkpoint`。**

### CX-3（效率）——`max_new_tokens=512` 与批生成相乘

HF `generate` 等整批结束。不终止率 1–2% 时，B=64 下大多数批都会跑满 512 步，
而"正常"只需要 ~330 步。可选缓解见 §6 决策 D2。

### CX-5（**新出现，需主 agent 注意**）——左 padding 的批/单不一致

链上 GPU0 验证作业 11:52 的实时日志：

```
[verify] checkpoint-2488: LEFT-PADDING MISMATCH 1/4 -- falling back to batch_size=1
```

即同样 4 个样本，batch=8 左 padding 生成的文本与逐条生成的文本有 1/4 不同。贪心解码是
前缀确定性的，一旦某个位置两个 logit 接近、bf16/FA2 在不同 batch 形状下的规约顺序变了，
后面整段就分叉——这类差异本身不算 bug，但它意味着：

* **生成作业的产物依赖 batch 组成**。注意这与分片**无关**：生产脚本本来就按 `--batch-size`
  批量生成，任何 batch 边界的变化都会改变一部分记录。分片只是多了几条批边界。
* 验证作业为安全起见回退到 `batch_size=1`；**生产作业不能这么做**——B=1 实测每样本 8–12 s，
  160k 条要 400+ 小时。
* 因此本作业的立场必须**写明**：`genwhere/2` 是"该模型在给定 batch 配置下的贪心输出"，
  不是"逐条生成的唯一真值"。`job.marker` 与分片报告都记了 `batch_size`，驱动对一个 split
  的全部分片用同一个值。若主 agent 认为这条对论文口径有影响，请在 D5 里一并裁定。

**验证作业跑完后必看**（它是本作业唯一的真实先验）：
`s0_base_sft/checkpoint_verification.json` 的 `summary.generated_tokens` 分位数与
`truncated` / `terminated` 比例——**若被选中的 checkpoint 大量不吐 EOS，S 就恒等于
`max_new_tokens`，墙钟直接落到上表的悲观档**，D2 的取值也随之变得值钱。

### CX-6（口径 nit，不影响正确性）

`generate_where` 返回的是整批统一宽度的行，提前结束的样本尾部是 pad/eos（V17）。
下游 `extract_segment` 先切 eos，所以**上下文没被污染**；但记录里的 `n_generated_tokens`
与 summary 的 `generated_tokens` 分位数实际是"该批的解码步数"。读报告时不要把
`generated_tokens p50 ≈ 500` 误读成"模型在胡说"——真正的段长看 `where_tokens` / `color_tokens`。

### CX-4（已在本目录修复的坑）

* 用绝对路径跑脚本时 `import q3vl` 失败（`python /abs/x.py` 把**脚本目录**放 `sys.path`，
  `cd $REPO` 不管用，只有 `-m` 才加 cwd）。dry-run 抓到，两个脚本都显式插了仓库根；
* `publish_generated` 自己会调 `genwhere_payload`，传 payload 进去会 `TypeError`。dry-run 抓到。

---

## 6 · 待主 agent 决策

| id | 事项 | 保守默认（已采用） | 备选 |
|---|---|---|---|
| **D1** | 排期：这不是空档作业，`two_segment` 两卡 4–7 h，`forced_color` 再 4–7 h | 先只跑 `two_segment`（解锁全部 8 个 Where-B 臂 + 12 个 What 臂里的 10 个）；`forced_color` 只服务 C01/C02，等这两臂真的排上再跑 | 两个模式连着跑，占卡 8–14 h |
| **D2** | `MAX_NEW_TOKENS` | **512**（协议 `GEN_MAX_NEW_TOKENS`，`whereb/config.py` 有出处） | 416（仍 > GT max 330，不会截断合法输出，最坏情况省 ~19%）。属于改协议常量，不静默拍板 |
| **D3** | CX-1 的解法 | 兼容软链（零改动，已验证） | 给 `run_where_b.py` 加 `--genctx`（改 q3vl 既有文件，需实现审阅） |
| **D4** | checkpoint | 脚本参数化，等裁定 | 一旦选 2488，Where-B/What 启动命令必须显式同步（CX-2） |
| **D5** | `BATCH_SIZE` | 64 | 标定后按实测上调（96/128 显存都够）；改动只需重跑未完成的分片 |
| **D6** | 是否给 T_final / T_lut_unseen 也生成 | **不生成**。`evaluate_what.py` 默认 `--split V_what`，且 `--allow-split` 明确写着"打开 T_final 必须是有意为之（协议 12.4）" | 最终测试板需要时再补，届时是 ~4,700 条的小作业 |

---

## 7 · 交付物

| 文件 | 作用 |
|---|---|
| `genctx_shard.py` | 单分片 runner（切 `ds.refs` + 强制隔离 out-root/report-dir，其余透传） |
| `merge_genctx.py` | 分片合并 + 四类断言 + 兼容软链 + 交付报告 |
| `genctx_dual.sh` | 两卡驱动：每卡一个分片队列，**逐作业** D-20 四步，可续跑，收尾自动合并 |
| `genctx_forced_color_dual.sh` | 同一驱动，`MODE=forced_color`、`SPLITS="V_what train"` |
| `dryrun/run_dryrun.sh` `dryrun/test_shard_and_merge.py` | 79 条断言，CPU-only |
| `dryrun/dryrun.log` `dryrun/DRYRUN_RESULT.md` | dry-run 记录 |
| `dryrun/cpu_smoke_real_producer.log` | **真生产脚本**在 CPU 上跑通两个分片的实跑日志（`--limit 4`、base 权重、`--max-new-tokens 8`） |
| `README.md` | 启动命令卡片 + 监控 |
| `logs/` `reports/` | 运行时产出（现在是空的） |
