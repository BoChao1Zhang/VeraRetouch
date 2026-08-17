# tools/lut_reannotate — LUT 重标注管线 (TOOL-LutReannot-1)

为 preset bank 里的 **4,051 个 LUT**（cube 4,000 + 3dl 51）重做 VLM 标注。旧标注
（`preset_bank_full/vlm_names.jsonl`）产于红蓝互换 bug 修复之前，描述的是从未被渲染出来的画面，
**全部作废**。本管线重建证据（6 组探针 before/after 真实渲染）与标签（外部 relay VLM 标注）。

> **本卡只做到冒烟。** 全量 `render` / `annotate` / `pack` 由主 agent 另行启动。

## 快速开始

```bash
cd /home/bc/VeraRetouch/tools/lut_reannotate
P=/home/bc/miniconda3/bin/python3      # 必须用 conda 的 3.13；/usr/bin/python3 是 3.10，没有 tomllib

$P pipeline.py probes                                   # 1. 拉 6 张探针源图（一次性，秒级）
$P pipeline.py render  --workers 32                     # 2. 全量渲染 24,306 张（约 9 min，纯 CPU）
$P pipeline.py annotate --concurrency 32                # 3. 全量标注 4,051 条（约 46 min）
$P pipeline.py pack --out /home/bc/VeraRetouch/lut_bank_reannot_20260811.zip
```

三步都**幂等续跑**：重跑只补没做完的。`render` 靠产物文件是否存在判断，`annotate` 靠
`annotations.jsonl` 里已有的 `ok` 行判断（append-only，永不重复计费）。

## 子命令

| 命令 | 作用 | 关键参数 |
|---|---|---|
| `probes` | 取 6 张固定探针源图，缩到长边 768，写 `probes/before_<slot>.{png,jpg}` + `probes.json` | — |
| `render` | 每个 LUT × 每张探针 → JPEG q90 4:4:4，CPU 多进程 | `--workers 32 --limit N --ids ...`，末尾默认跑 `--selfcheck`（`--no-selfcheck` 关） |
| `annotate` | 12 张图 + HSL 响应表 → relay VLM → `annotations.jsonl` | `--concurrency 32 --limit N --ids ... --model --effort --config --attempts` |
| `failures` | 列出还没有 `ok` 行的 preset_id（直接重跑 `annotate` 即自动重试） | `--out` |
| `pack` | LUT 本体 + `annotations.jsonl` + `MANIFEST.json` 打 zip 并自检 | `--out`（必填）`--limit --ids --store` |

全局 `--work <dir>` 改工作目录（默认 `/home/bc/data/scratch/lut_reannotate`）。

## 工作目录布局

```
/home/bc/data/scratch/lut_reannotate/
  probes/before_<slot>.png     # 渲染输入（无损，长边 768）
  probes/before_<slot>.jpg     # 送模型的 before（q90 4:4:4）
  probes/probes.json           # 6 探针 provenance：asset_id / 源路径 / sha256 / 走没走 NFS 归档
  renders/<xx>/<preset_id>/<slot>.jpg
  out/annotations.jsonl        # append-only，key=preset_id
  out/render_progress.json     out/annotate_progress.json
```

## 输出契约

`annotations.jsonl` 每行：

```json
{"key": "rcp_…", "preset_id": "rcp_…", "ok": true,
 "name": "…", "per_probe": {"红":"…","黄":"…","绿":"…","蓝":"…","肤色":"…","中性":"…"},
 "caption": "…", "style_major": "…", "style_minor": "…",
 "scene_affinity": "portrait|landscape|general", "strength": "subtle|moderate|strong",
 "hsl_features": {"spec_rev":"hsl8-v1","bands":{…},"neutral_ramp":[…],"summary":{…}},
 "provenance": {"model":"gpt-5.6-terra","response_model":"gpt-5.6-terra","reasoning_effort":"low",
   "lane":"provider-c-lane-1","attempt":0,"timestamp":"…","prompt_rev":"lutreannot-v1",
   "render_rev":"…","hsl_spec_rev":"hsl8-v1","probe_order":["红",…],
   "input_tokens":12077,"output_tokens":381,"latency_s":15.5,"retried_errors":[],
   "rubric_sha256":"…","schema_sha256":"…"}}
```

失败行 `ok:false`，带 `error` 与 `retried_errors`，不占用 `done` 集合，下次重跑自动重试。

zip 内：`luts/<fmt>/<原文件名>`（4,051 个本体；重名会加 `<preset_id>_` 前缀消歧——实测本
bank 内**无重名**）+ 顶层 `annotations.jsonl` + 顶层 `MANIFEST.json`（逐 LUT 的
`preset_content_hash` 与打包时实算的 `file_sha256`、数量、渲染与标注 provenance）。
`pack` 结束后校验成员数 = LUT 数 + 2 且 `annotations.jsonl` 行数 = 匹配到的标注数，不一致返回 2。

## 两条被硬检查的东西

1. **轴序**。`luts.npz` 的网格是 `[b][g][r]`、值通道 RGB、0..1。`pipeline.py:apply_lut`
   是独立重写的实现，`render --selfcheck` 抽 3 个 preset × 6 探针与
   `dataset_build.src.construct.rendering.apply_lut_cpu_oracle` 对拍，`max|diff| ≥ 1e-5` 直接失败。
   *这正是本次重标注要消除的那个 bug，不允许再犯一次。*
2. **模型偷换**。relay 有过 1/3 概率把请求的模型换掉的实测记录（provider-b，2026-07-28）。
   每次调用都校验 `response.model`，本管线在 `reeval_relay` 的前缀校验之上再做**精确等值**
   校验（钉死 `gpt-5.6-terra`），不等即该条失败重试并计入 `substituted`。

## 依赖与环境

- **解释器**：`/home/bc/miniconda3/bin/python3`（3.13）。`timeout ... python3` 会解析到
  `/usr/bin/python3`（3.10，无 `tomllib`），务必写全路径。
- 复用 `dataset_build/tools/reeval_relay.py`（`load_lanes` / `call_once` / `JsonlStore` /
  `prompt_digest`）与 `dataset_build/tools/archive_reader.py`；两者都**只读不改**。
  `reeval_relay.CONFIG` 指向一个已不存在的文件，所以本管线总是显式传
  `--config databuild.prod-l6-local17k-20260801.toml`。
- API key 只从该 TOML 读取，不落盘、不打印。
- **不碰 GPU**（两张卡都在跑训练）；渲染是纯 numpy gather + multiprocessing。
- NFS 只读且只走 `/mnt/nfs-ro`：`archive_reader` 的 catalog 里 shard root 是 `/mnt/nfs/…`，
  本管线用一个子类在 `locate()` 之后把 root 改写到 `/mnt/nfs-ro/…`。
