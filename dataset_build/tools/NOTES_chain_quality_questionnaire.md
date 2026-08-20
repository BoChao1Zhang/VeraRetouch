# NOTES · chain_quality_questionnaire.py

假设与保守默认，供人类复核（不含结论）。

## A9（2026-08-19，整链质量 300 条盲标）

- **快照**：campaign `local-v1-iter1`，读取时刻 sources=500（accepted 408 / source_rejected 56 /
  error 36）、global branches=1383、local branches=4189、committed leaves=1515。
  后台 resume 仍在跑，重跑会拿到不同快照；快照计数写进 `item_key.json.snapshot`。
- **总体口径**：一条链 = 一个 global branch × 一个 `commit_status=committed` 的 local leaf；
  链的 id 取 leaf `branch_id`，`global_branch_id` 一并落进 item_key。
- **人口剔除（两处，均非静默）**：
  1. `source_manifest_null(error)` 69 条。source 图 blob 只能从 tree manifest 的
     `source_artifact` 拿（DB 的 `source_sha256` ≠ source 图 sha256，见 NOTES_agent_loop_review），
     error source 的 `manifest_json` 为 NULL → 三联首格无法出图，故整条排除。
  2. `global_after_blob_missing` 3 条（blob 在任何 artifact root 与 landed-tar 目录里都取不到）。
  剩余可抽样人口 = 1443 条 / 409 个 source。**待决策**：是否要为这 69 条 error source 另找
  source 图来源（若有）并纳入人口。
- **local bin 词表**：本 campaign 的 local leaf 实际只出现 `subtle` / `natural` / `strong`
  三个取值，没有 `moderate`；`natural` 本是 global 词表的取值。九宫格因此实际只有 8 个非空格
  （`natural × strong` 人口为 0）。格的迭代顺序 = global(natural, medium, bold) ×
  local(LOCAL_BIN_ORDER 命中项 subtle, strong 在前，词表外的 natural 追加在后)，
  该顺序只影响展示与配额修正的先后，抽样本身由 sha1 定序。
- **配额修正**：quota = floor(300 × 占比 + 0.5) 后 clamp 到格内人口，差额按「人口最大的格优先」
  逐 1 补/减到总数恰 300。本次修正前后各格均能满额，`quota_shortfall_before_backfill` 与
  `cap_backfill` 均为空；每源上限 2 条实际未触发缺口（选中 300 条来自 238 个 source，单源最多 2）。
- **版式**：选了**横排单行三联**（`grid-template-columns: repeat(3,1fr)`），窄屏（<1100px）自动
  回落成「上 source、下 global|final」两行。缩略图按任务卡取短边 512，按 sha256 去重共 828 张。
- **分辨率不对称（待决策）**：source blob 本身就是长边 ~512 的小图（如 512×342），而
  global_after / final_after 渲染产物是长边 768（768×512）。三格 CSS 等宽显示时 source 会被
  浏览器放大约 1.2×，比两张渲染图略软。当前**保守保持任务卡口径（三张统一短边 512、复用 A8 的
  Thumbnails 不改）**；若担心「更软的原图」影响劣化/提升判断，需要把渲染图按该 source 的
  原始像素尺寸降采样后再出图，这会脱离 A8 的按 digest 去重逻辑，未擅自改。
- **盲标**：页面正文与图片文件名里不出现 bin / strength / ΔE / preset / mask / branch_id /
  scene；三格 caption 用「原图 / 第一步处理后 / 最终结果」。item_id 为展示序 `c001…c300`，
  展示序 = 全局 sha1(branch_id) 升序（打散九宫格，避免同格连排）。参数全部只在 `item_key.json`。
- **键盘**：全部操作都有按钮（无键盘依赖）；额外保留 1–5 / ←→ 快捷键，输入框内不拦截。
- **analyze 的 ΔE 分箱**：6 个等宽箱的边界取自 **item_key 里 300 条的取值范围**（不是全人口），
  因此边界只依赖抽样结果、与回填 CSV 无关；空值单列 `n_null_value`，未评 / 非法评分 /
  item_key 里没有的 item_id 分三列计数。
- `mask_family` 与 `direction_cosine` 已写进 item_key，但任务卡指定的 analyze 输出里没有这两维，
  故未出表。

## A9c（2026-08-19）· 档位 2 文案改写 + analyze 拆列

- **改动**：`RATING_LABELS` 档位 2 `略有劣化` → `global 改善但 local 劣化`；其余四档不动。
- **不跑 build**：无 DSN 且 build 会重抽样。已发布的 `chainq.html` 里的 `ITEMS` / `IMAGES` /
  `STORE_KEY` 用正则从页面回读，先以**旧文案**调 `write_page()` 复现，字节与线上页一致
  （md5 `da2474fdeb2b999808a91ab2abf2a5d4`），据此证明 re-emit 等价；再以新文案写出
  （md5 `d75db73c522372db47e530c721d9dfa0`）。`chainq.csv` / `item_key.json` 未触碰。
- **localStorage key 不变**（`chain-quality-questionnaire-v1:local-v1-iter1`），已标记的评分不丢。
- **analyze**：`_stats` 增 `rate_1` / `rate_2` 两列（原 `deteriorated_rate`（≤2）保留不变），
  grid / ΔE 曲线两张表的表头与数据行同步加 `=1` `=2` 两列。
