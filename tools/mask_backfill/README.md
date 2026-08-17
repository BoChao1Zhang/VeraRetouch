# TOOL-MaskBackfill-1 — 为 20,009 张 `subject_not_ready` 源图补 subject 掩膜

现有 SAM3 主体掩膜管线（`dataset_build/source_qa/sam3_subject_instances.py`）本来假设
源图都在本地。迁移之后 20,009 张待补源图里有 19,699 张只在 NFS 归档里，而且 VLM 要从
本地 vGate 改走外部 relay。这个目录是**让那条管线能跑这批数据**所需的三个外挂步骤，
管线本身只做了必要的最小改动（见 `NOTES.md`）。

```
enumerate.py        catalog + PG 反查 → backfill_pool.jsonl（20,009 行，run --pool 可直接吃）
prefetch.py         19,699 张归档源图 → /home/bc/data/scratch/mask_backfill/src（按 shard 顺序）
verify_consumer.py  用真正的消费侧闸门 construct.sources._inspect_cache_dir 验收产物
```

## 铁律（写在最前面）

* **禁止访问 `/mnt/nfs`**（hard 挂载，一次 stall 不可恢复）。catalog 里记的 archive
  root 全是 `/mnt/nfs/...`，`prefetch.py` 在打开任何描述符之前把它改写成
  `/mnt/nfs-ro/...`；`verify_consumer.py` 在调闸门之前先确认源图能从本地或预取缓冲
  读到，否则直接跳过并报 `skipped_would_touch_nfs`。
* **`subject.json` 里的 `source_path` 必须是原始逻辑路径**，不能是预取的 scratch 路径。
  消费侧 `_inspect_cache_dir` 会拿它过 archive-aware 的 `path_exists`；写成 scratch
  路径的话，scratch 一清理这些条目就全部失效，而且失效是静默的。管线的写法保证了这点
  （`source_path` 全程不动，只有像素读走 `read_path`），`verify_consumer.py` 是这条的实证。
* API key 只从 0600 的 databuild TOML 读进进程，不落盘、不打印、不进 argv。

## 全量跑批（主 agent 经 GPU 队列启动）

```bash
# 1. 枚举（databuild env：要 psycopg）
/home/bc/envs/databuild/bin/python tools/mask_backfill/enumerate.py \
    --out /home/bc/data/scratch/mask_backfill/backfill_pool.jsonl \
    --report /home/bc/data/scratch/mask_backfill/enumerate_report.json

# 2. 预取（约 31.5 GB / 8 分钟 / 只读 /mnt/nfs-ro，可断点续跑）
/home/bc/envs/databuild/bin/python tools/mask_backfill/prefetch.py \
    --pool /home/bc/data/scratch/mask_backfill/backfill_pool.jsonl --workers 8

# 3. 跑批（monetgpt_sam3 env，gpu0；--limit 0 = 全量，可续跑）
CUDA_VISIBLE_DEVICES=0 /home/bc/miniconda3/envs/monetgpt_sam3/bin/python \
  -m dataset_build.source_qa.sam3_subject_instances run \
    --pool /home/bc/data/scratch/mask_backfill/backfill_pool.jsonl \
    --device cuda:0 --sam-batch 8 --chunk 256 --vlm-workers 16 \
    --vlm-config /home/bc/VeraRetouch/databuild.prod-l7-local400k-20260811.toml \
    --vlm-endpoint provider-c-lane-1 --vlm-model gpt-5.6-terra \
    --vlm-effort low --vlm-max-output-tokens 6000 --vlm-verify-model

# 4. 验收
/home/bc/envs/databuild/bin/python tools/mask_backfill/verify_consumer.py \
    --cache-root /home/bc/data/datasets/vera_directionA_1M/subject_cache
```

`run` 不加 `--cache-root` 就写生产 subject_cache（默认值）；续跑判据是
`<cache-root>/<path_key>/subject.json` 是否存在，所以现存的 6,375 个
`subject_label_error` 条目会被自动跳过、不会被覆盖。冒烟用的是
`--cache-root /home/bc/data/scratch/mask_backfill/smoke_cache`，与生产树隔离。

## 实测数字（2026-08-12，16 张冒烟 + 200 张预取采样）

| 项 | 实测 |
|---|---|
| 枚举 | catalog 20,009 not_ready ↔ PG 56,777 源，sha1 反查命中 **20,009 / 0 未命中 / 0 碰撞** |
| 分布 | 本地 310（全部 MMArt-PPR10k，且都不在归档里）+ 归档 19,699（8 组 / 149 shard / 31.45 GB） |
| 预取 | 8 线程 + sha256 校验：**40.7 img/s、58.2 MB/s**；19,699 张外推 **约 8 分钟** |
| SAM3 | batch 8、长边 1536：**0.25–0.30 s/img**（20,009 张 ≈ 1.4–1.7 h GPU） |
| VLM label | median 19.0 s，in 5,912 tok / out 98 tok |
| VLM selector | median 15.1 s，in 6,250 tok / out 81 tok；16 张里 8 张（50%）走到 selector |
| 端到端 | 16 张 / 133 s @ `--vlm-workers 4`（含 35 s 模型加载）→ 16 workers 外推 **8.5–12 h** |
| relay 总量外推 | 约 30,000 次调用、**约 181M input tok / 2.8M output tok** |
| 冒烟 status | ready 5 / no_subject 6 / group_envelope_too_large 3 / no_center_candidate 2；**source_unreadable 0、transport 失败 0** |
| 消费侧 | 5 个 ready 条目全部 `eligible`（其中 4 个源图只在归档里，靠预取缓冲读到） |

## 预取缓冲的两个名字

每张预取的图有两个硬链接（同一个 inode，不多占字节）：

* `src/<path_key>.<ext>` —— pool 行里的 `read_path`，SAM3 管线直接打开的路径；
* `buffer/<sha256(source_path)>` —— `archive_reader.set_prefetch_dir()` 认的布局。

第二个名字是给**归档感知的消费者**用的：`_inspect_cache_dir` 会用 `read_bytes(source_path)`
重新解码一次源图，没有这个缓冲它就会去读 `/mnt/nfs` 上的 shard。`verify_consumer.py`
先 `set_prefetch_dir(buffer)` 再调闸门，所以整个验收过程一次都没碰硬挂载。
