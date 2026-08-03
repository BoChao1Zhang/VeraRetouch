# VeraRetouch 数据资产说明（截至 2026-08-02 晚）

本文回答三个问题：现有 source 资产有哪些；最近生成了哪些数据；每条数据的格式。
数字来自 journal 归档与在途 build 的实时统计（`journal-archive/` + `/mnt/ramstage/`）。

## 1. Source 资产（输入侧）

源图不以散文件存放，物理上在 **NFS indexed-tar 归档**里，通过 18G 的全局目录
`/var/cache/veradata/global.sqlite3` 按「逻辑路径」检索（`archive_reader` 三级读
梯：本地 → prefetch 缓冲 → 归档）。journal 里的 `source_path` 是逻辑键，不是可
直接打开的文件。

两波战役实际使用了 **33,662 个不同源图**（采样门 = SAM3 主体就绪，合格池约
34,350，基本采穿）。按池分布（跨 build 组数占用，可重复采样）：

| 源池（逻辑目录） | 组占用 | 内容 |
|---|---|---|
| `_scratch/unsplash` | 62,507 | Unsplash 摄影 |
| `presets_sources/awards` | 39,038 | 获奖摄影集 |
| `ppr10k/source` | 24,080 | PPR10K 人像 |
| `RAISE-6k/jpg_preview` | 14,909 | RAISE 原片预览 |
| `presets_sources/quandian` | 9,765 | 圈点风格集 |
| `presets_sources/korean` | 9,079 | 韩系人像 |
| `presets_sources/greysky/raw` 等长尾 | ~600 | 灰天/fivek_gold/MMArt 少量 |

**LUT 资产**：生产只采 LUT preset（param 路线未用）。两波共动用 **3,522 个不同
preset**（taxonomy major/minor 分层，组内 8 槽硬去重，build 内按最少使用均衡
复用——单 LUT 单 build 最高被渲染 1,360 次，尾部仅 5 次）。

**SAM3 主体缓存**：local 线的 region 任务依赖每源的 subject mask 缓存
（`sources.subject_cache`），无主体源在 sam3_relabel 阶段转终态并由替换源补员。

## 2. 最近生成的数据（输出侧）

两波共 10 个生产 build（v5.2 标注契约 + gpt-5.6-luna 转写 + OneAlign IAA 排序 +
「度量即标注」hints）。采样保证：组内 8 preset 互异；跨 build (source,preset)
对实测零碰撞——**训练集里不存在“同图同 LUT”重复样本**。

| Build | 组数 | SFT 行 | 状态 |
|---|---|---|---|
| prod-g1-global25k-20260731 | 25,000 | 22,803 | ✅ |
| prod-g2-global25k-20260731 | 25,000 | 22,370 | ✅ |
| prod-l1-local17k-20260731 | 17,000 | 13,978 | ✅ |
| prod-l2-local17k-20260731 | 17,000 | 12,814 | ✅ |
| prod-l3-local17k-20260731 | 16,914 | 13,690 | ✅（-86 孤儿源手术） |
| prod-g3-global25k-20260801 | 25,000 | 22,912 | ✅ |
| prod-g4-global25k-20260801 | 25,000 | 标注中（目标 ~22.5k） | 🔄 |
| prod-l4-local17k-20260801 | 17,000 | 10,687+ | 🔄 标注中 |
| prod-l5-local17k-20260801 | 17,000 | 排队 | ⏳ |
| prod-l6-local17k-20260801 | 17,000 | 渲染中 | ⏳ |

**已落地 119,254+ 行**，全部完成后预计 ~17 万行。产出率 ≈0.85 行/组（弃权政策
`winner margin <1.0` + local 线 SAM3 损耗）。约 33% 行带 `winner_confidence=low`
（margin 1.0-2.0），训练侧可按此过滤。

完成 build 的 journal 统一归档在
`/var/cache/veradata/annot_review/journal-archive/<build_id>/`。
另有 20+ 个战前小 build（eval100/fresh*/wp* 系列）留在 NFS，为 QA 夹具与实验
记录；`eval100-annotqa-20260727` 已退役为纯 QA 夹具（**永不进训练**）。

## 3. 数据格式

### 3.1 落盘归档（训练消费入口）

```
/mnt/nfs/bc/data/datasets/sft/<build_id>/
  manifest.json                # build 终态、有效配置、产出统计
  batch-NNNN/
    shards/shard-*.tar         # 样本成员，组内连续、槽序排列
    indexes/shard-*.idx.jsonl  # tar 成员偏移索引
    indexes/catalog.sqlite3    # 逻辑路径 → (shard, offset) 点查
    metadata.jsonl             # 每成员一行的落盘元数据
```

每个样本三类成员：`I_tar`（赢家渲染图 .jpg）、`C_GT`（配色真值 .cgt.png，
两槽可共享同 inode）、`I_in`（原图拷贝，扩展名随源）。

`metadata.jsonl` 字段：`sample_id / sft_id / group / member / role / logical_path
/ source_path / bytes_size / annotated / annotation_failure_code / schema_version`。

### 3.2 SFT 行（`sft.jsonl`，每行一条训练样本）

| 字段 | 类型 | 说明 |
|---|---|---|
| `sft_id` | str | 样本主键（`sft_<hash>`） |
| `build_id` / `group_id` / `candidate_id` | str | 溯源三级键 |
| `I_in` | str | 原图逻辑路径 |
| `I_tar` | str | 赢家渲染图路径（tmpfs 期路径，落盘后以归档为准） |
| `task_type` | str | `style`（global 线）/ `local`（region 线） |
| `local` | null/obj | local 任务的 region 描述（global 为 null） |
| `reasoning` | str | **v5.2 七段契约**：`<problem_light/color/texture>` 三问题段 → `<region_scope>` 专段 → 三 plan 段 → 收束（专用 start/end token 包裹） |
| `instruction` | str | 完整编辑指令（风格名 + 方向词，veto 死区过滤） |
| `instruction_short` | str | 短指令变体 |
| `recipe` | obj | 渲染配方：`format=lut / preset_id / preset_path / major / minor / slot…`（不进 prompt，仅溯源） |
| `qa` | obj | 质量链：OneAlign IAA 分数与 rank、annotation 尝试记录（endpoint/attempt）、visibility 度量（v5.2 hints：warmth/hue_gm/tone-curve contrast/brightness/chroma/surfaces） |
| `winner_rank` | int | 赢家在 8 候选中的 IAA 名次 |
| `winner_confidence` | str | `normal` / `low`（margin 1.0-2.0；<1.0 已弃权不出行） |
| `annot_model` / `annot_src` | str | 标注模型与通道（如 `responses:external:provider-c-lane-1`，MODEL_SUBSTITUTED 校验过） |
| `annotation_task_id` | str | 标注任务键（journal 重放去重用） |

### 3.3 组行（`groups.jsonl`，每行一个渲染组）

`group_id / source_id / source_path / render_mode(global|local) / scene /
preset_filter / major / candidates[8]`；每个 candidate 含
`candidate_id / preset_id / major / minor / slot_id / format / qa(IAA 分数+rank)
/ after_path / cgt_path / mask_id / region`。附 `failures.jsonl`（全失败事件流，
含终态标记，是 resume 与统计的权威来源）。
