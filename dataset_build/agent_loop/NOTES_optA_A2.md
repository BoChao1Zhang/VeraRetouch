# A2 实施记录（A 类决策 §2.1–2.4 / §2.6 / §2.7）

规格来源：`docs/DECISIONS_agent_loop_annotation_optA_20260819.md`。本文件只记录假设、待决策与
保守默认，不含结论。

## 1. revision 与 preflight

- `CANDIDATE_SERIALIZATION_REVISION`: `lut-objective-v4-local-reach` → `lut-fingerprint-v5-optA`
  （**A2b 再 bump 到 `lut-fingerprint-v5.1-optA`**，见文末 A2b 节）。
- `prompt_revision_fingerprint()` 的 registry 追加 `shortlist_header`、`reason_codes`，并覆盖新
  schema/rules，故 `thread_revision` 变化。
- **旧 provider preflight 对新 revision 失效**（preflight key 绑 prompt fingerprint）：生产前需按
  新 prompt fingerprint 重跑 preflight，旧结果不得沿用。
- `catalog_contract` 追加 `cluster_artifact`：接线聚类 artifact 会再次改变 `thread_revision`。

## 2. 规格里未写死、由本卡选定的实现（保守默认，未静默拍板）

1. **行首 `id` 列 = 该 shortlist 内的 `row_index`（0 起）**，`preset_id` 不再进 prompt。理由：§2.4
   已改为 pick-by-index，preset_id 对模型冗余且占 token。如需回退为送 preset_id，只需改
   `prompts.shortlist_row_text` 一列。
2. **`enum(offered)` 用运行时校验实现，未做 per-request 动态 JSON schema**（`major`、`mask_id` 在
   schema 里是 string，非法值在 `graph` 校验器里拒绝）。理由：schema 进 `prefix_cache_key`，把
   per-source 的 mask_id / major 塞进 enum 会按源打散 provider 前缀缓存（现有测试
   `test_repair_and_sibling_prompt_prefixes_are_stable` 即锁这条）。拒绝行为等价。
3. **`reason_codes` 取 1–3 个**（`minItems=1, maxItems=3`，元素须互异）。词表仍是闭集 8 词。
   §2.4 只写"多选、≤8 词表"，上限 3 是为压 output token，需要放宽改 `REASON_CODES_SCHEMA` 即可。
4. **自由文本删除后的替代来源**（**A2b W1 已改**：leaf 的 `visible_region` 改为
   `region_descriptor()` 区域键，仅 validator prompt 仍用 center_hint）：
   `visible_region` 改取所指派 mask 的 `center_hint`，validator 的
   `assignment.direction` 改取 mask 的 `direction`（Mask v2 元数据，非模型生成）。
   `assignment` 另加 `repair_count`（整数），用于审计/测试区分首轮与修复轮。
5. **local 阶段看到的 global 决策**裁剪为 `{bin, reason_codes}`（`_public_global`），不再下发
   `preset_id`/自由文本 direction。
6. **shortlist 返回结构变化**：
   - `global_shortlist()` → `{"by_major": {...}, "quota_deficits": [...], "offered_majors": [...]}`，
     整体写入 `global_shortlist_artifact`（缺额落在 artifact 里，§2.3 要求）。
   - `local_shortlist()` → `{"rows": [...], "quota_deficits": [...]}`。local 侧没有独立 artifact，
     缺额随分支落库：`agent_branch.result_json.local_shortlist_deficits`（也进 tree manifest）。
7. **配额算法：稀缺档位优先 + 覆盖计数**，而非"每档独占取 N 条"。理由：固定 natural→medium→bold
   顺序的贪心会先把同时可达 bold 的高分 LUT 吃进 natural 配额，导致 bold 档在明明有可达候选时
   仍缺额。现实现按各档可达候选数升序（同数按档位顺序）处理，并统计"已选行中可达该档的条数"。
   配额不满足只记录 `{bin, required, covered, reachable_candidates}`，不抛异常。
8. **纠偏硬过滤字段选择**：annotation 没有独立的 neutral-ramp 中灰响应字段，用
   `hsl_features.summary.mid_gray_a/mid_gray_b` 作中灰响应向量，source 色偏取
   `palette.lab_a_mean/lab_b_mean`；点积 < 0 记为反向；某 major 内反向 LUT 占多数（严格过半）才
   保留该 major。**若过滤后为空则保留全部 major**，并在 `quota_deficits` 记
   `correction_filter_empty`（否则整源无候选，属于比"未纠偏"更差的失败模式）。过滤生效时记
   `correction_filter_applied` + 被丢弃的 major 列表。
9. **簇去重**：cluster key = `style_major \x1f cluster_id`（跨 major 不串簇）；同簇最多 1 条；代表用
   `sha256(source_sha256 + "\x1f" + cluster_key)` 的前 64 bit 对簇成员数取模轮换，成员按
   `(-score, preset_id)` 排序。未配置 `catalog.cluster_artifact` 时每 preset 自成一簇（等价现状）。
   聚类 artifact 是 A1 的产物，本卡只留接口，**未接线**（等 pilot 阈值验收）。
10. **`scorer_rank`** 定义为"簇去重前、该 major 内按打分排序的名次（0 起）"，审计
    `scorer_top1 = (rank == 0)`、`scorer_top3 = (rank < 3)`。
    （**A2b W2 已改为双口径 `scorer_rank_raw` / `scorer_rank_offered`**，见文末 A2b 节。）
11. **方向余弦**只在 local leaf 记录（1 条 local 链 1 个值），向量取
    `(mid_gray_dL, mid_gray_a, mid_gray_b, sat_pct_mean)`；任一向量为零向量时记 NULL（不拦截）。
12. **proposal_id 改为确定性生成**（schema 里已无该字段）：global `g{row_index}-{bin}`，
    local `l{repair_count}-{row_index}-{mask_id}`。
13. 指纹 8 数全部来自 `hsl_features.summary` 的现有字段（closed-v1 实测 8 个键齐全：
    `contrast_ratio / highlight_dL / hue_rot_abs_max / mid_gray_a / mid_gray_b / mid_gray_dL /
    sat_pct_mean / shadow_dL`）；缺字段时按 0.0 计入（不报错）。

## 3. 审计新列落库位置（§2.6）

新表 `proposal_audit`（sqlite + postgres 同构），主键 `branch_id`：

```
branch_id | campaign_id | source_sha256 | level | preset_id | scorer_top1 | scorer_top3 |
scorer_top1_raw | direction_cosine | created_at

```

- `level='global'` 行：`scorer_top1/scorer_top3` 有值，`direction_cosine` 为 NULL。
- `level='local'` 行：`direction_cosine` 有值，scorer 两列为 NULL。
- 只记录不拦截；`export_tables()` 已包含该表，可直接 SQL 查询。

## 3b. 顺带对齐的配置文件（如不同意请改回）

`configs/agent_loop.local-v1.toml`、`configs/agent_loop.terra-smoke.toml` 原显式写死
`global_major_limit=8 / global_per_major_limit=4 / local_limit=12`（覆盖默认值），已改为
`3 / 7 / 9` 与 §2.3 对齐；否则 smoke 跑的仍是旧配额。

## 4. 未做 / 明确留给后续卡

- 聚类 artifact 的实际接线与阈值（A1 + 用户验收）。
- smoke 重跑与 §3 token 对比表（收尾卡）。
- Mask v2 gate、sibling 多样性、调用次数/fan-out、commit 合同、canonical pipeline、渲染与校准：本卡未动。
- `local↔global` 方向余弦的拦截阈值仍是 B 类挂起项，本卡只落列。

## A2b（审阅 warning 修复）

规格 §2.1–2.7 语义未改；本节只记录处置方式、由本卡选定的实现与被改变的行为。

### W2 scorer_rank 双口径（必修）

- `shortlist` 行不再有 `scorer_rank`，改为两列：
  - `scorer_rank_raw`：簇去重 + 配额选择**之前**、该 major（或 local 候选池）内按分数的名次（0 起）。
  - `scorer_rank_offered`：**最终 shortlist 行内**按分数的名次（0 起，即 prompt 里 row_index 的顺序）。
- `proposal_audit.scorer_top1/scorer_top3` 改由 **offered** 口径派生；raw 口径另存新列
  `scorer_top1_raw`。sqlite/postgres DDL 与 INSERT（9→10 列）同构更新。
- 行为变化：接线聚类 artifact 后 `scorer_top1` 不再结构性恒 0；旧库的 `proposal_audit`
  没有 `scorer_top1_raw` 列，**需重建表或 ALTER TABLE ADD COLUMN**（本卡未写迁移脚本）。
- `graph._enrich_global_proposals` / `_enrich_local_proposals` 同步带两列进 proposal 记录。

### W1 visible_region 低粒度区域键

- 新增 `candidates.region_descriptor(mask)`，返回 `f"{family}:{direction_or_position}"`：
  direction 非空且不等于 `elliptical` 时取 direction（`band:diagonal`/`linear:top`/
  `semantic:subject`）；否则（radial）取 **alpha_projection 8×8 质心的 3×3 粗位置**
  （`radial:center`/`radial:top-left`…）。位置桶阈值为 0–7 坐标的三等分（7/3、14/3）。
- 用途拆分：leaf 记录里的 `visible_region`（进 `select_committed_leaves` 的 regions 多样性项、
  进 `agent_branch.result_json` 与 tree manifest）= 区域键；**validator prompt 的
  `assignment.visible_region` 维持人读版 `center_hint` 不变**，prompt 字节未因 W1 变化。
- **commit 挑选行为因此变化**：旧口径下 radial 与 band 共用同一句 center_hint，
  regions 多样性项对二者恒为 0（family 项已覆盖）；新口径下同 family 内不同方向/位置也计入
  多样性，`select_committed_leaves` 第 3 条起的挑选顺序会与 A2 不同。
- 保守默认：位置桶只到 3×3、不引入角度；`alpha_projection` 缺失/全零时返回 `center`。
  若需更细粒度（例如 radial 按主轴方向分桶），改 `_projection_position` 即可。

### W3 配额下限校验

`config._validate` 新增：`global_per_major_limit >= 2*3`、`local_limit >= 3*3`、
`global_major_limit >= 1`，不满足抛 `ConfigError`（原先静默截断到配额之下）。常量按
§2.3 硬编码在 config.py（不从 candidates.py import，避免 config↔candidates 循环导入）。

### W4 / W6 / revision

- W4：`prompt_revision_fingerprint()` registry 追加 `fingerprint_fields` 与
  `fingerprint_formats`，字段顺序/精度改动现在会改 `thread_revision`。
- W6：新增 `prompts._fingerprint_number`，格式化后若 `float(text) == 0.0` 则去掉负号
  （`-0.04 → "0.0"`、`-0.4 → "0"`、`-0.0004 → "0.000"`），测试钉死。
- **`CANDIDATE_SERIALIZATION_REVISION` 已 bump：`lut-fingerprint-v5-optA` →
  `lut-fingerprint-v5.1-optA`**（W6 改变了 prompt 字节）。因此
  **旧 provider preflight 再次失效**，生产前需按新 prompt fingerprint 重跑 preflight；
  W4 单独也会改 fingerprint（不改字节）。

### W7 / W8

- W7：`global_local_preset_equal` **选择"保留 + 注明"（断言级防御）**，未删除。理由：该分支是
  `local_shortlist(exclude=(global preset,))` 接线的兜底，删掉后接线回归会静默放行父子同 preset；
  改为 raise 会把可恢复的校验失败升级为整链崩溃，运维上更差。已在源码加注释说明其只在
  exclude 接线回归时触发，并补一条直接构造 validator 的测试把它从死代码变成有测试的防御。
- W8：删除无调用的 `LutRecord.public()`（及其 bands/summary 拼装）；全仓已无引用。

### W5 / W9 不修的理由

- W5（`hsl_features.summary` 缺字段按 0.0 计入、不报错）：当前 closed-v1 标注 **4051/4051 条
  8 个键齐全**，缺字段路径不可达；改为报错会在数据未变的前提下引入新的失败模式。
- W9（bin 词表未在 catalog 侧再校验）：`GLOBAL_BATCH_SCHEMA` / `LOCAL_BATCH_SCHEMA` 的
  `enum` 已兜底非法 bin，`_quota_select` 只按 `bin_order` 遍历，越界词表不会进 shortlist。

### 测试补强（W10）

- `direction_cosine`：已知向量精确值（(3,4,0,0)·(4,-3,0,0)=0、反向 -1.0、
  (1,1,0,0)·(3,4,0,0)=0.989949、零向量与未知 preset → None）。
- `correction_filter_empty`：单 major 且全部 LUT 同向加剧色偏 → 保留全部 major +
  `quota_deficits[0]` 精确等于 `correction_filter_empty` 记录。
- `local_shortlist_deficits` 非空：3 preset 目录下跑整图，断言缺额同时出现在
  `agent_branch.result_json.local_shortlist_deficits` 与 tree manifest 的 branches 上。
- 另补：`region_descriptor` 六种几何得到六个不同键；`-0.0` 归一 + `FINGERPRINT_FORMATS`
  改动会改 fingerprint；配额下限三个非法配置各抛 `ConfigError`；`global_local_preset_equal`
  防御分支可触发。
