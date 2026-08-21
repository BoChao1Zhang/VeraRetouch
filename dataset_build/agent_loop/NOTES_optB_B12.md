# NOTES · B12 直方图证据（source 侧 + LUT 侧）落地

revision：`lut-intent-v7-optB` → **`lut-intent-v7.1-optB`**

---

## 1. 待决策 / 假设（保守默认继续，未静默拍板）

### 1.1 `Δh` 分组切片写法有歧义（已按分区实现）

任务卡写 `d_shadow=Δh[0:2]和`、`d_mid=Δh[3:5]和`、`d_high=Δh[6:8]和`。

- 按 Python 半开切片读：bin {0,1} / {3,4} / {6,7}，bin 2 与 bin 5 落空，8 个箱不被覆盖。
- 按闭区间读：bin {0,1,2} / {3,4,5} / {6,7}（`8` 越界，截到 7），恰好是 8 箱的完整分区。

采用**闭区间分区**（`HISTOGRAM_GROUPS = {"d_shadow": (0,1,2), "d_mid": (3,4,5),
"d_high": (6,7)}`），因为只有它满足 `d_shadow + d_mid + d_high == 0`（分布差分之和为 0），
也只有它不丢箱。若需改成半开读法，改 `segment_fingerprints.HISTOGRAM_GROUPS` 一处，
该常量已进 prompt registry，改动会自动换 revision，但**必须重建 v2 指纹文件**（SHA 会变）。

### 1.2 `d_shadow>0` 的符号语义与「提阴影」不自洽（已按字面实现，符号做成常量）

`Δh[i] = l_bins_out[i] - l_bins_in[i]`（输出减输入的箱占比差）。在这个定义下：

- LUT 把阴影提亮 → 阴影箱里的像素**变少** → `d_shadow < 0`；
- LUT 把高光压下来 → 高光箱里的像素**变少** → `d_high < 0`。

任务卡的打分规则写的是：source `clip_low` 高 ↔ LUT `d_shadow>0`（括号注「提阴影」）加分；
`clip_high` 高 ↔ `d_high<0` 加分。其中 `d_high<0` 与上面自洽，`d_shadow>0` 与括号注不自洽
（`d_shadow>0` 在上述定义下是「更多像素被压进阴影」）。

**实现按任务卡字面**：`clip_low` 过阈且 `d_shadow > 0` 加分；`clip_high` 过阈且 `d_high < 0`
加分。同时把两个符号提成预注册常量，翻转是一行改动且自动进 revision：

```
HISTOGRAM_MATCH_GATE = {
    "clip_low_min": 0.02, "clip_high_min": 0.02, "delta_scale": 0.05,
    "shadow_weight": 0.5, "highlight_weight": 0.5,
    "shadow_sign": 1.0,        # 阴影项在 shadow_sign * d_shadow > 0 时触发
    "highlight_sign": -1.0,    # 高光项在 highlight_sign * d_high > 0 时触发
}
```

若确认想奖励「提阴影」的 LUT，把 `shadow_sign` 改成 `-1.0` 即可。**请用户裁决。**

**2026-08-21 用户已裁决**：翻转，`shadow_sign = -1.0`（clipped-shadow 源奖励 `d_shadow < 0`）。
`highlight_sign` 保持 `-1.0` 不变；`HISTOGRAM_GROUPS` 与 v2 指纹文件不动，`prompt_revision_fingerprint()` 随之改变。

### 1.3 探针集选了 6 个中的哪 4 个

任务卡写「4 探针 × 4096px 或其缓存」。仓库里的冻结探针缓存
（`/home/bc/data/scratch/lut_reannotate/probes/before_*.png`，长边 768）共 **6** 个
（red / yellow / green / blue / skin / neutral），`lut_render_distance` 的默认探针只有 **1** 个。

采用：`HISTOGRAM_PROBE_NAMES = ("neutral", "skin", "red", "blue")`，每探针 4096 px，
共 16384 px（`probe_pixels(probes, 4*4096, seed=20260819)` 均分，复用
`lut_render_distance.probe_pixels`）。选取理由记录在 tool docstring：两个决定色调分布的
（neutral / skin）+ 一暖一冷各一个。**若应改成 6 探针或另一组 4 个，请指明。**

### 1.4 离线 `annotate_source` / `reach_candidates` 未接直方图项

`_score` 已带 `source_histogram` 可选参数并在 global / local 在线检索接线；
但 `source_annotations.annotate_source` 里的 `catalog.reach_candidates(...)`（离线 top-300
reach 子集）**没有**接。理由：接上会改变已冻结的 `preset_reach` 产物，现有 200 源的
`sources200.annotated.jsonl` 需要重标。任务卡未要求重标。**待决策**。

### 1.5 新增 run config 未启动

新建 `configs/agent_loop.local-v2-b12.toml`（由 iter5b 复制，只改 campaign_id、artifacts
root、`segment_fingerprints` 指向 v2）。**未启动任何 run**。
`configs/agent_loop.local-v2-iter5b.toml`（run-v4b 正在读）**未改动**，`git diff` 为空。

---

## 2. 实现清单

### 2.1 新模块 `dataset_build/agent_loop/source_histogram.py`

叶子模块，不 import 任何 agent_loop 模块（有测试守卫）。

| 项 | 值 |
|---|---|
| `SOURCE_HISTOGRAM_CONTRACT` | `source-histogram-lab-8bin-v1` |
| 采样 | `np.linspace(0, N-1, 4096)` 定步长、无种子、无 RNG |
| `l_bins` | L\* 在 [0,100] 上 8 等宽箱占比，和为 1 |
| `clip_low` / `clip_high` | `L*<2` / `L*>98` 占比（是首/末箱的子份额，不是额外箱） |
| `c_bins` | C\*ab 4 箱，切点 10/25/50，和为 1 |
| `hue_sectors` | Lab hue 6 个 60° 扇区，只统计 `C*ab >= 10` 的像素，和为 `1 - c_bins[0]` |
| 序列化格式 | L `{:.3f}`、CLIP `{:.4f}`、C `{:.3f}`、H `{:.3f}` |

### 2.2 v2 指纹（`segment_fingerprints.py` + 新 tool）

- `SEGMENT_FINGERPRINT_SCHEMA_V2 = "lut-segment-fingerprint-v2"`，
  `HISTOGRAM_DERIVATION_REVISION = "probe-l8-binshare-v1"`。
- v2 行 = **v1 全部字段逐字保留** + `histogram` 组（`l_bins_in` / `l_bins_out` /
  `delta` 各 8 数 + `d_shadow` / `d_mid` / `d_high`）+ `histogram_revision` + `probe_sha256`。
  `source_sha256`（annotations SHA）继承自 v1，输入 SHA 双份齐全。
- `validate_segment_fingerprint_row` 同时接受 v1/v2；v1 行带 `histogram` 会报错，
  v2 行缺 `probe_sha256` / `histogram_revision` 会报错。
- `load_segment_fingerprints` 对 v2 文件返回与 v1 完全相同的 segments 映射（有测试）；
  `load_segment_histograms` 对 v1 文件返回 `{}`，混 schema 报错。
- 构建工具：`dataset_build/tools/build_lut_histogram_fingerprints.py`，复用
  `lut_render_distance.probe_pixels` / `render_all`（packed-LUT CPU oracle，已返回 Lab）。

### 2.3 挂载与打分

- `LutRecord.histogram_response`（`None` = 未挂载）、`histogram_aggregates()`；
  `prompt_view()` 仅在挂载时多出 `histogram` 键。
- `LutCatalog.histogram_responses_mounted` 属性。
- `LutCatalog._score(..., source_histogram=None)` 末尾加
  `histogram_match_bonus(source_histogram, row.histogram_response)`，任一侧缺失时**恰好为 0.0**，
  v1 挂载逐位复现 B12 前的分数。
- `source_histogram` 透传：`global_shortlist` / `reach_candidates` / `intent_rows` /
  `build_local_packets`，全部是带默认值的可选关键字参数（旧调用点零改动）。

### 2.4 prompt 接线

- global 稳定前缀顺序：rules → source 图 → `{"diagnosis": ...}` → **`SOURCE_HISTOGRAM_HEADER`
  + 一行紧凑序列化** → shortlist 表。仍在 stable prefix 内，prefix cache 不破。
- shortlist 行末尾追加 `d_shadow d_mid d_high`（`{:.3f}`，负零去号）。局部行顺序为
  `... | caption | mask_reach_de | d_shadow d_mid d_high`。
- `SHORTLIST_HEADER` 追加 `HISTOGRAM_SHORTLIST_NOTE`。

### 2.5 运行时断言（预注册判据必须接线）

- `graph._prepare_node`：算完立刻 `assert_histogram_columns(histogram)`。
- `graph._shortlist_node` / `_global_propose_node`：再次 `assert_histogram_columns`，
  state 里没有就直接抛。
- `build_local_packets` 返回 `source_histogram_applied`，`_intent_packets` 未拿到就抛
  `RuntimeError("source_histogram_not_wired")`（与既有 `mask_reach_gate_not_wired` /
  `direction_prefilter_not_wired` 同一档）。

### 2.6 registry

41 → **44** 键，新增：

| key | 内容 |
|---|---|
| `source_histogram` | contract / sample_pixels / 箱几何 / 四个数字格式 / header 全文 |
| `segment_fingerprint_histogram` | derivation_revision / v2 表 SHA / 分组 / 聚合名 / 行格式 / note |
| `histogram_match_gate` | `HISTOGRAM_MATCH_GATE` 全量（含两个符号常量） |

`PROMPT_REGISTRY_KEYS` 显式清单同步更新，`test_prompt_registry_key_set_is_the_explicit_frozen_list`
的硬编码计数 41 → 44。

### 2.7 v1 仍可挂载

`runtime.require_frozen_segment_fingerprints` 从「等于单个 SHA」放宽为
「属于 `REGISTERED_SEGMENT_FINGERPRINT_TABLES`（v1, v2）」。iter5b（v1）若重启仍能通过启动断言。

---

## 3. 数字

- v2 文件：`/home/bc/data/scratch/lut_reannotate/out/segment_fingerprints.v2.jsonl`
  - SHA256 `62aae05eef857cc3557f7121e21b6d0528b7822e6b5917a5ffe45b1a02b43681`
  - 4,051 行，4,249,968 bytes；重建一次得到同一 SHA（确定性已实测）
  - `probe_sha256` `f2ff40ef3721ca93d5d118d43db63b5247a5a90468c47ed1a55ad6fd97002eb6`
- 报告：`segment_fingerprints.v2.report.json`
- `prompt_revision_fingerprint()` = `e56fbe2473c7d78a32f26ed21365cc697aa0417560b8158c1d48c8f51c669624`

新列分位（4,051 条）：

| 列 | min | p05 | p25 | p50 | p75 | p95 | max |
|---|---|---|---|---|---|---|---|
| `d_shadow` | -0.465819 | -0.212004 | -0.059264 | 0.004151 | 0.083589 | 0.155732 | 0.331850 |
| `d_mid` | -0.237488 | -0.149048 | -0.071350 | -0.024292 | 0.018280 | 0.124938 | 0.358521 |
| `d_high` | -0.102661 | -0.102661 | -0.027069 | 0.013245 | 0.059112 | 0.167054 | 0.602845 |

输入分布 `l_bins_in` = `[0.109863, 0.273193, 0.155212, 0.153809, 0.116943, 0.088318,
0.072632, 0.030029]`

token 增量（tiktoken `o200k_base` 实测）：

| 位置 | 前 | 后 | Δ |
|---|---|---|---|
| shortlist 单行（真实 catalog 首条，中文 caption） | 70 | 83 | **+13** |
| `SHORTLIST_HEADER`（每 source 一次） | 164 | 278 | **+114** |
| source histogram 块（header + 一行，每 source 一次） | 0 | 243 | **+243**（其中数据行 108） |

source 直方图行样例（`/home/bc/datasets/MMArt-PPR10k/global/230_7/before.jpg`）：

```
source_histogram source-histogram-lab-8bin-v1 | L8 0.480 0.080 0.033 0.063 0.084 0.085 0.062 0.112 | CLIP 0.1260 0.0000 | C4 0.785 0.194 0.012 0.010 | H6 0.149 0.023 0.039 0.000 0.000 0.004
```

shortlist 行样例（v1 挂载 → v2 挂载）：

```
0 | natural,medium | 5.2 1.108 -9.1 -1.4 -110 6.9 -8.0 13.9 | 以暖黄阴影对照青绿冷中高光，显著提亮红黄暖色，同时压低绿蓝饱和度。
0 | natural,medium | 5.2 1.108 -9.1 -1.4 -110 6.9 -8.0 13.9 | 以暖黄阴影对照青绿冷中高光，显著提亮红黄暖色，同时压低绿蓝饱和度。 | -0.023 -0.020 0.043
```

## 4. 测试

- 新增 `dataset_build/tests/test_source_histogram.py`：**33 passed**。
- `test_agent_loop.py` 150 passed（条数不变，只改了 6 处既有断言：revision 字符串 ×2、
  registry 计数 41→44、shortlist artifact 键集、行序列化、`require_frozen_*` 错误串）。
- `test_lut_annotations.py` 7、`test_direction_match.py` 19，全绿。
- 全量 `dataset_build/tests`（`PYTHONPATH=dataset_build/src`）：**796 passed / 30 failed**，
  30 条全部是既有环境问题（`databuild.example.toml` 缺文件，分布在
  `test_canonical_foundation` / `test_iaa_batch` / `test_source_window_and_cgt` /
  `test_winner_margin`），与 B12 无关，改动前后同一批。B12 之前基线为 763 passed。
- `ruff check` + `py_compile`：改动文件全通过（仓库其余 29 条 ruff 报错为既有，未触碰）。
- **未跑任何 run。**
