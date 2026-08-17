# NOTES — TOOL-LutReannot-1 实施前核实 / 假设 / 待决策

日期：2026-08-11。实施者：编码 subagent。任务卡：TOOL-LutReannot-1（建工具 + 8 条冒烟）。

---

## 0. 任务卡「已核实事实」逐条复核

| # | 卡上说法 | 实查结果 |
|---|---|---|
| 1 | `features.jsonl` 里 `kind=lut` 共 4,051（cube 4000 + 3dl 51） | ✅ 实测 4,051 = cube 4,000 + 3dl 51（总行 7,778，其余 3,727 为 `kind=param`） |
| 1 | `luts.npz` key=preset_id，`[b][g][r]`、RGB、0..1 | ✅ 4,051 key，与 `luts_meta.json` 完全一致；网格 16/17/32/33/64/65 混合；轴序见 §3 实证 |
| 1 | `luts_meta.json` 有 path/dmin/dmax | ✅；**全部 4,051 条 dmin=(0,0,0) dmax=(1,1,1)**，无非单位域 |
| 2 | CPU 参考实现 `rendering.py:77 apply_lut_cpu_oracle` | ✅ 行号一致 |
| 2 | GPU 两卡满载 → 一律 CPU | ✅ 实测 gpu0 65% / gpu1 100%，本管线全程未碰 GPU（渲染 2208% CPU = ~22 核） |
| 3 | 6 个 probe asset_id，各 12,078 行 `preset_previews` | ✅ 六个 id 各 12,078 行 |
| 3 | 「先查 PG `assets` 拿各自 path」 | ❌ **只有 2/6 在 `assets` 里**。恢复过程见 §2 |
| 3 | PG 只读连接用 conninfo `options=…` | ✅ 照做，`current_setting('transaction_read_only')='on'` |
| 4 | `[[annotation.external_endpoints]]` id=`provider-c-lane-1`，tokenskingdom | ✅ |
| 4 | 模型 = `[annotation].external_model` | ⚠️ **已被主 agent 中途改写**：改用 `gpt-5.6-terra`（TOML 里是 `gpt-5.6-luna`）。按指令以常量/命令行参数显式传入，**未改 TOML**；`response.model` 精确钉死 terra |
| 4 | `reeval_relay.py:38 CONFIG` 指向已不存在的文件 | ✅ 指向 `databuild.eval100.toml`（不存在）；本管线始终显式传 l6 TOML，未改该文件 |
| 5 | 旧管线在 git 历史里 | ✅ `git show ee3652c^:dataset_build/source_qa/pilot_preset.py`、`git show 3bdae81^:dataset_build/source_qa/config.py` 都取到了 |
| 6 | 现有 taxonomy 10 个大类名 | ✅ 作为 prompt 里的「参考词，不强制沿用」 |

其它自查发现：

- **`preset_content_hash` 不是 LUT 文件的 sha256**（实测两者不等，是上游对解析后内容的规范化哈希，
  产出脚本未进版本控制）。因此 `MANIFEST.json` 里**两个都记**：从 `features.jsonl` 带下来的
  `preset_content_hash`（对账 bank 身份）+ 打包时实算的 `file_sha256`（对账 zip 成员字节）。
- **4,051 个 LUT 源文件全部本地存在**，合计 **5.72 GB**（卡上估 5.3 GB）；**同 fmt 内无重名**，
  所以消歧分支实际不会触发（代码保留）。冒烟实测 deflate 压缩比 ≈ 33%，全量 zip 估 **~1.9 GB**。
- `/usr/bin/python3` 是 3.10（无 `tomllib`），`timeout ... python3` 会走到它。本管线 shebang 与
  文档都钉死 `/home/bc/miniconda3/bin/python3`。

---

## 2. 探针槽位对应关系（卡上说「别猜」，这里是证据链）

### 2.1 卡上给的查法走不通

`vera_source_qa.assets` 里**只有 2 个** probe asset_id：`src_308bb19e31eeb228`（PPR10k）、
`src_8e1419bab64e14ad`（quandian）。另外 4 个在全库任何一张表里都只出现在
`preset_previews.probe_image` 这一列（逐表扫过 `assets / ds_source_registry / construct_renders /
construct_groups / iqa_scores / llm_qa / processing_events / sam3_masks / source_captions /
ds_sft_samples`）。`preset_previews.before_path` 指向的
`/home/bc/data/datasets/vera_directionA_1M/source_qa/preset_previews/…` 整棵树已被删除，且**不在
NFS 归档 catalog 里**（前缀区间查询命中 0 条）。历史 `pilot/manifest.json` 也已随 `source_qa/pilot/`
一起消失，git 里没有。

### 2.2 恢复来源

从写出 `probe_before.json` 的那次 axisfix 会话记录里，把当时 `resolve_probes_lab` 的返回值原样取回：

```
/home/bc/.codex/sessions/2026/07/17/rollout-2026-07-17T21-51-17-019f7058-86e5-78d2-bfbe-79ce80056f6f.jsonl
```

（同一天 `runs` 表里 `lut_axfix_rerender_1784299601_d2737e`，`{"n_luts": 4051}`，`ok=4051`，
`preset_previews` 最后一批 24,306 行就是它写的。）

### 2.3 两道独立交叉验证（都过）

**(a) 与 bank 自己的 `probe_before.json` 对拍**——把恢复出来的 6 张源图缩到长边 768、算全图平均
L\*a\*b\* 与平均 chroma，逐槽与 `preset_bank_full/probe_before.json` 比：

| 槽 | asset_id | 源路径 | L (ours/bank) | a\* | b\* | C |
|---|---|---|---|---|---|---|
| 1 红 | `src_308bb19e31eeb228` | `/home/bc/datasets/MMArt-PPR10k/global/790_2/before.jpg` | 50.39 / 50.39 | 45.42 / 45.53 | 36.67 / 36.66 | 58.89 / 58.95 |
| 2 黄 | `src_dbafb5a380332b8a` | `…/_scratch/TAD66K/38974475@N0530442571530.jpg` | 48.61 / 48.66 | 12.32 / 12.31 | 49.90 / 49.91 | 52.18 / 52.20 |
| 3 绿 | `src_fe5abed17e617006` | `…/_scratch/TAD66K/emanuelezallocco34195304370.jpg` | 49.70 / 49.75 | −38.40 / −38.41 | 52.08 / 52.11 | 64.83 / 64.86 |
| 4 蓝 | `src_9bb8c3b8ec140e3a` | `…/_scratch/TAD66K/michelblanchette11450465554.jpg` | 24.33 / 24.35 | 19.07 / 19.12 | −32.86 / −32.93 | 40.63 / 40.72 |
| 5 肤色 | `src_8e1419bab64e14ad` | `/home/bc/data/datasets/presets_sources/quandian/quandian_000870.jpg` | 39.08 / 39.17 | 7.45 / 7.47 | 11.33 / 11.32 | 18.92 / 18.91 |
| 6 中性 | `src_48fda67912a38548` | `…/_scratch/TAD66K/oscarplaza16256301813.jpg` | 40.57 / 40.64 | 0.04 / 0.04 | 3.04 / 3.03 | 3.17 / 3.18 |

六槽全部吻合到 0.1 个 LAB 单位以内（残差来自分辨率不同）。

**(b) 复现 `features.jsonl:lab_vec`**——`lab_vec` 是 24 维 = 6 探针 ×(ΔL, Δa\*, Δb\*, ΔC)，槽序
red/yellow/green/blue/skin/neutral（依据 `dataset_build/tools/build_taxonomy.py:31 PROBES` 与
`Tax.hue_row` 的 `v[4*i:4*i+4]` 切法）。用恢复出来的 6 张图 + 本管线渲染器重算
`rcp_b4568f4084c9eec8`：**max|Δ| = 0.236，corr = 0.99979**（残差同样是分辨率）。
红蓝互换会让 Δa\*/Δb\* 面目全非，这条同时锁死了槽位映射**和**轴序。

**(c)** 另外把 6 组 before/after 拼图肉眼看过一遍：红=红裙人像、黄=黄底蜥蜴、绿=绿色麦田、
蓝=夜景教堂圣诞树、肤色=人像、中性=黑白街拍，与槽名一致。

### 2.4 取图耗时

`probes` 子命令实测：`red` 走本地（0.01 s），其余 **5 张走 `/mnt/nfs-ro` 归档**
（yellow 0.15 s / green 0.07 s / blue 0.13 s / skin 0.13 s / neutral 0.11 s），全命令 4.3 s
（含 numpy/PIL 起解释器）。归档 group 分别是 `img/still_life|landscape|night|street/tad66k`、
`img/unknown/quandian`、`sft/prod-g4-global25k-20260801/batch-0000`。
`archive_reader` 的 catalog 里 root 写的是 `/mnt/nfs/…`（hard 挂载），本管线用
`ReadOnlyRootReader` 在 `locate()` 之后把 root 改写成 `/mnt/nfs-ro/…`，全程只读。

---

## 3. HSL 8 色相带采样规格（可复核）

实现在 `hslfeat.py`，规格全部是模块顶部的五个常量，改常量即改规格；`spec_rev` 随行落进
`annotations.jsonl` 与 `MANIFEST.json`。

- **八带中心（HSL 色相角，度）**：红 0 / 橙 30 / 黄 60 / 绿 120 / 浅绿 180 / 蓝 240 / 紫 270 / 洋红 300。
  **这是本工具的约定**：Lightroom HSL 面板只给出八个带的*名字*，上面是这些名字在 sRGB 色轮上的
  规范位置。没有去「核实」某个 Adobe 内部数值，因为这条特征是本卡要求「新写、确定性」的自定义量，
  不是引用的外部事实。
- **每带 3×3×3 = 27 个代表色**：色相偏移 `{−12°, 0°, +12°}` × 饱和 `{0.45, 0.70, 0.95}` ×
  明度 `{0.35, 0.50, 0.65}`。共 8×27 = **216 个采样点**。
  选中等明度、中高饱和：色带响应只在该色带真有颜色的地方才有意义；接近黑/白的样本量到的
  基本是明度曲线，会与中性灰段重复。
- **中性灰阶 5 级**：`{0.15, 0.30, 0.50, 0.70, 0.85}`（S=0）。**色温/色罩只在这一段读**，
  这是旧 prompt 那条硬约束的确定性版本。
- **采样色 → LUT**：HSL→sRGB（`colorsys`）→ 与渲染探针**完全同一个** `apply_lut`（注入而非重写）
  → sRGB→HSL 读回；中性段另走 sRGB→CIE L\*a\*b\*（D65，与 bank 同口径）。
- **每带聚合**：27 个样本的**中位数**（不是均值）——一个在网格角上被 clip 的样本不该拽动整行；
  均值与 IQR 一并写进机器可读 dict。
  - `d_hue_deg`：色相旋转，度，绕回 [−180,180]
  - `d_sat_pct`：**相对**饱和变化 `(S_out−S_in)/S_in×100`
  - `d_lum_pct`：HSL 明度**点数**变化 `(L_out−L_in)×100`
- **聚合列**：`contrast_ratio`（L\* 跨度比 out/in）、`shadow_dL` / `highlight_dL`、
  `mid_gray_a` / `mid_gray_b`（中灰色罩）、`sat_pct_mean`、`hue_rot_abs_max`。
- 两种形态：`compute()` 出 dict（进 `annotations.jsonl`），`render_table()` 出紧凑中文表（进 prompt）。
  进 prompt 的数字与进 jsonl 的数字是同一批，未做任何二次归一化。

---

## 4. 渲染口径

- 探针原图 → EXIF 校正 → RGB → **先缩到长边 768（LANCZOS）→ 再过 LUT**（顺序按任务卡；
  与「先过 LUT 再缩」结果不同，此处遵卡）。渲染输入是无损 PNG 缓存，不是 JPEG，避免把
  before 的压缩噪声送进 LUT。
- 输出 JPEG **q90，`subsampling=0`（4:4:4）**。相对生产 `responses.py:_encode_image`（q90 默认
  4:2:0）是**有意偏离**：本任务判的就是色相偏移，色度抽样会把要判的东西糊掉。像素数据不受
  `optimize` 影响，故渲染侧未开 `optimize`（省 CPU）。
- `apply_lut` 是独立重写的实现（扁平索引），与 `apply_lut_cpu_oracle`（fancy indexing）不同写法，
  这样 `--selfcheck` 才是真的第二意见而不是自己抄自己。冒烟实测 3 preset × 6 探针
  **worst max|diff| = 1.788e−07**（容差 1e−5）。

---

## 5. 假设清单（已自行核实的部分）

1. **`lab_vec` 槽序 = red/yellow/green/blue/skin/neutral** —— 依据 `build_taxonomy.py:31` 并由
   §2.3(b) 的数值复现坐实。
2. **`preset_previews` 没有槽位列** —— 实查表结构确认，故槽位映射只能走 §2.2 的恢复路径。
3. **归档 root 改写到 `/mnt/nfs-ro` 是安全的** —— 归档是不可变的 indexed tar，`nfs-ro` 与 `nfs`
   是同一份数据的两个挂载；读到的字节与 catalog 里的 size / tar header 逐条校验（`ArchiveReader.read`
   自带），不匹配会抛错。
4. **`reeval_relay.call_once` 的 `startswith` 前缀校验不够** —— 它会放行
   `gpt-5.6-terra-<snapshot>`。本卡要求「钉死」，故在其之上补了一层精确等值校验。
   代价：如果 provider 哪天开始返回带日期后缀的模型名，全量会全部失败重试——**这是有意的**，
   宁可停下来问，不要静默换标注器。
5. **并发 32 覆盖 lane 配置的 16** —— 按用户明示。lane 配置未改。
6. **`np.load` 的 `NpzFile` 不是线程安全的** —— 它包着一个 `ZipFile` 句柄。所以 HSL 特征在起
   worker 线程**之前**单线程算完（4,051 条估 < 1 min），不让 32 个线程去抢同一个 zip 句柄。
   渲染侧是多**进程**，每个进程各开各的 npz，无此问题。

---

## 6. 冒烟实测数字（全量外推的依据）

| 项 | 实测 |
|---|---|
| `probes` | 6/6 就位，5 张走 NFS 归档，单张 0.07–0.15 s，全命令 4.3 s |
| `render --limit 8` | 8/8，selfcheck worst max\|diff\| **1.788e−07** PASS |
| 渲染吞吐（`--workers 32`，128 preset 实测） | 120 preset / 17.6 s ≈ **7.6 preset/s** → 全量 **~9 min** |
| 渲染产物大小 | 137 KB/张 × 24,306 ≈ **3.3 GB** |
| `annotate --limit 8 --concurrency 4`（终版代码，真打 relay） | **8/8 成功，0 次模型偷换，0 次重试**，`response.model` 全部 `gpt-5.6-terra`，总耗时 43.4 s |
| 延迟 | 均值 **19.8 s**，中位 18.2 s，min 13.0 s，max 36.5 s |
| token | input **12,077**/条（六张 before 固定、六张 after 尺寸相同，几乎是常数）；output 均值 **432**（314–659） |
| 重试（终版之前那轮 10 条的观察） | 有 2 条 attempt 0 传输错误、attempt 1 成功；因此加了 `provenance.retried_errors` 把错误串留痕 |
| **全量 token 外推** | input **≈ 48.9 M**，output **≈ 1.75 M** |
| **全量时长外推** | concurrency 32 → 1.62 条/s → **≈ 42 min**（若 relay 只吃得下 16，则 ≈ 84 min） |
| `pack --limit 8` | 10 成员 = 8 luts + `annotations.jsonl` + `MANIFEST.json`，自检 PASS；raw 12.0 MB → zip 4.0 MB |
| **全量 pack 外推** | 源 **5.72 GB** → deflate 后 **≈ 1.9 GB**，单线程约 8–12 min |

标注质量抽查（8 条）：name 8/8 互不相同（琥珀增艳 / 青绿交叉 / 琥珀暗金 / 黄绿复古 / 暖亮褪彩 /
褪彩压高光 / 暖金褪影 / 暖灰褪彩），caption 8/8 互不相同，`per_probe` 每条内 6 行全部互不相同
（旧 prompt 那条「同大类 per_probe 必须不同」的约束看起来是生效的），且逐行引用了响应表里的实测数字。

---

## 7. 待主 agent 决策

1. **旧标注怎么处置。** 本管线只往工作目录写，**没有动**
   `preset_bank_full/vlm_names.jsonl` / `taxonomy.jsonl`（它们是红蓝互换前的产物）。
   新标注要不要回灌 bank、要不要据新 `style_major/minor` 重跑 `build_taxonomy.py`——留给主 agent。
   保守默认：**不回灌**，新结果只活在 `out/annotations.jsonl` 与 zip 里。
2. **`taxonomy` 大类要不要闭集。** 现在 prompt 把 10 个旧大类当「参考词，不强制沿用」，
   `style_major` 是自由文本。好处是能长出新类，坏处是 4,051 条跑完可能出现几百个同义大类名，
   需要后处理归并。若主 agent 希望直接可用，可以把 `style_major` 改成 enum 闭集（改
   `pipeline.py:SCHEMA` 一行）。保守默认：**开集**（不可逆的信息损失优先避免）。
3. **`3dl` 那 51 个要不要单独看。** 它们和 cube 一样走 `luts.npz`，网格已解析，管线不区分；
   但 3dl 的 domain 语义历史上更容易出岔。已确认这 51 条 dmin/dmax 也都是单位域。
   保守默认：**同等对待**，不特判。
4. **失败条目的收口策略。** 现在 `annotate` 跑完返回 1（有失败即非零），失败清单靠
   `pipeline.py failures` 列出、重跑 `annotate` 自动重试。若主 agent 要求「全量必须 4051/4051」，
   需要约定重试轮数上限与人工介入点。保守默认：**跑三轮 `annotate`，剩下的报给主 agent**。
5. **精确模型钉死的副作用**（见 §5.4）。若主 agent 认为带日期后缀的 terra 快照可接受，
   需要显式放宽；现在是宁失败不静默。
6. **zip 落地位置。** 卡上写 `--out /home/bc/VeraRetouch/<名字>.zip`，即**落在 git 仓库里**
   （~1.9 GB）。已确认 `.gitignore` 不覆盖仓库根的 `*.zip`，误提交会很难看。
   保守默认：**本卡不执行全量 pack**；建议主 agent 落到 `/home/bc/data/scratch/lut_reannotate/`
   或先加 ignore 规则。
