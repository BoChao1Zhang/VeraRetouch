# NOTES · B1 背景角色 Mask（Mask v2 role 化）

规格依据：`docs/DECISIONS_agent_loop_local_intent_optB_20260819.md` §2-C1/C3、§4-B1。
代码：`dataset_build/agent_loop/candidates.py`、`dataset_build/agent_loop/prompts.py`、
`dataset_build/tools/sample_mask_role_pilot.py`、`dataset_build/tests/test_agent_loop.py`。

## 1. 预注册镜像门（代码常量 `BACKGROUND_ROLE_GATE`）

| 列 | 方向 | 预注册初值 | pilot 后是否调整 |
| --- | --- | --- | --- |
| `subject_alpha_mean`（主体像素 alpha 均值） | ≤ | 0.15 | 未调整 |
| `subject_high_coverage`（主体像素中 alpha≥0.5 占比） | ≤ | 0.02 | 未调整 |
| `background_alpha_mean`（主体 mask 之外像素的 alpha 均值） | ≥ | 0.35 | 未调整 |
| `half_area`（alpha≥0.5 像素占全图比例） | ≥ | 0.12 | 未调整 |

100 源 pilot 共 446 个背景 mask，四列分位（`docs/assets/mask_role_pilot_20260819/summary.json`
+ `samples.jsonl`）：

```
subject_alpha_mean       min=0.0014 p50=0.0365 p90=0.1124 max=0.1500
subject_high_coverage    min=0.0000 p50=0.0125 p90=0.0196 max=0.0200
background_alpha_mean    min=0.3504 p50=0.6376 p90=0.9364 max=0.9991
half_area                min=0.1807 p50=0.4984 p90=0.8782 max=0.9934
```

门违规 0/446（门是生成期硬约束，越界几何直接丢弃计数）。两个「≤」列的 max 恰好压在门上：
搜索是「满足避让门的最小排除区域」，二分收敛到门边界。

## 2. 假设与保守默认（未静默拍板项）

- **A1 「mask 有效面积占比 ≥ 0.12」口径**：取 `half_area`（alpha≥0.5 像素占全图比例）。
  另一种读法是 `effective_alpha_mean`（全图 alpha 均值，现行 subject 门用的就是这个名字）。
  446 个背景 mask 在两种读法下都过门：`effective_alpha_mean` min=0.1871。口径待用户确认；
  改口径不改本次 pilot 结论。
- **A2 旧调用路径不变**：`build_mask_bank(..., include_background=False)` 为默认，行为与改动前
  一致；`graph.py` 未接线（背景角色进候选包/schema/prompt 属 B1.5）。`allocate_mask_packets`
  一字未改，角色分配是新函数 `allocate_role_packets`。
- **A3 family 命名**：背景 mask 复用 `radial`/`band`/`linear` 三个 family 名（不新建
  `bg_*`），角色只放在 `role` 字段。`region_descriptor` 仅在 `role == "background"` 时加
  `background:` 前缀，subject 记录的区域键与 A2 交付完全一致（回归测试仍绿）。
- **A4 `stable_index` 会出现空档**：背景几何在全分辨率复核失败时 `continue`，该槽位的索引被跳过。
  subject 记录的 `stable_index` 取值不变。
- **A5 pilot 分辨率**：pilot 用长边 640 的显示尺寸当 `render_size`，不是生产 render artifact 尺寸；
  几何搜索仍走 256 短边 search core + 全分辨率复核（与生产同一条路径）。
- **A6 revision**：`MASK_SUMMARY_REVISION` 由 `mask-summary-v2` → `mask-summary-v3-role`，
  已进 `prompt_revision_fingerprint()`；即所有 stage 的 prompt revision 指纹改变，
  新老 campaign 靠 revision 隔离。
- **A7 sibling 第三位自由**：合同只要求 subject/background 各 ≥1；第三位由「family 多样性 →
  digest」决定。实测分布见 §4。

## 3. 实现要点

- **角色**：`MASK_ROLES = ("subject", "background")`，每条 mask 记录带 `role`，
  `mask_summary()` 新增 `role` 字段；`validate_mask_bank` 对 background 记录跑
  `_validate_background_mask`（四列运行时断言），family 组成校验只数 subject 角色。
- **几何池**：background 用与 subject 相同的参数族与 falloff（`_smoothstep`、radial 的
  `1.25/0.50` 边缘、band 的 `1.25 - d/half`），搜索目标反转：
  - `radial`：主体 PCA 椭圆的**补集**，二分「能避开主体的最小椭圆」→ 背景覆盖最大。
  - `band`：包住主体的最窄条带的**补集**（两侧半平面）。
  - `linear`：从余量最大的两条边打进来的梯度，二分「不碰主体的最大 offset」，
    falloff 常数 `BACKGROUND_LINEAR_FALLOFF = 0.35`。
  每族 2 个候选（`BACKGROUND_FAMILY_COUNTS`），共 ≤6。
- **量化容差重试复用**：`_expanded_geometry()` 把原来写死在 `_fit_full_resolution_geometry`
  里的 `1.01**expansion` 抽出来共用；background 方向相反地更安全（椭圆/条带放大、linear
  offset 缩小都降低主体 alpha）。`_fit_background_geometry` 最多 4 次扩张；因为两半门对扩张
  单调（主体 alpha ↓、背景覆盖 ↓），第一个过避让门的扩张就是背景覆盖最大的，覆盖门再失败即判不可行。
- **失败不抛错**：几何搜索失败/全分辨率复核失败写进 `diagnostics` 列表（`build_mask_bank`
  的可选出参），带 `family` + `reason`（`subject_avoidance_unreachable` /
  `background_coverage_search` / `full_resolution_gate`）。
- **sibling 分配**：`allocate_role_packets` 用角色覆盖（subject ≥1 且 background ≥1）替换
  「≥2 family」硬检查，family 多样性降级为 tie-break（排序键仍是 `(-len(families), digest)`）；
  semantic ≤1、8×8 投影距离 ≥0.02 两个旧检查保留。背景不可行时回退到「只从 subject 记录里选、
  ≥2 family」并在返回的 note 里记 `fallback="background_infeasible"`。
- **center_hint**：背景角色统一 `"the background around the main subject"`。

## 4. 100 源 pilot（`docs/assets/mask_role_pilot_20260819/index.html`）

抽样：`/home/bc/data/agent_loop/local-v1/sources5k.jsonl` 按 `sha1(source_id)` 升序，
跳过 `subject.mask_area > 0.60`（跳过计数 9），取前 100 条构建成功的源；attempts=100，
build_failures=0。主体面积范围 0.0057–0.5942。

- 包内含背景角色：98/100；回退 `background_infeasible`：2/100（`src_df173b8096396920`
  area=0.239、`src_6e99de087dfd232a` area=0.301，两源背景 bank 为 0，六个槽全部
  `background_coverage_search` 失败）。
- 角色组合：`subject=2,background=1` 63 条；`subject=1,background=2` 35 条；
  `subject=3,background=0` 2 条。
- 按 subject_area 分层（bank 背景 mask 数 = 6 槽中过门的个数）：

| 区间 | n | 包内含背景 | 回退 | bank 背景数均值 | bank 背景数=0 |
| --- | --- | --- | --- | --- | --- |
| 0.00–0.15 | 41 | 41 | 0 | 5.29 | 0 |
| 0.15–0.30 | 30 | 29 | 1 | 4.07 | 1 |
| 0.30–0.45 | 18 | 17 | 1 | 3.72 | 1 |
| 0.45–0.60 | 11 | 11 | 0 | 3.64 | 0 |

- 槽位丢弃计数：`band:background_coverage_search` 77、`linear:background_coverage_search` 70、
  `radial:background_coverage_search` 4、`linear:full_resolution_gate` 3。
- 过门背景 mask 的 family 分布：radial 196 / band 123 / linear 127（共 446）。

## 5. 回归与测试

- `pytest dataset_build/tests/test_agent_loop.py dataset_build/tests/test_lut_annotations.py`：
  90 passed（改动前 86 passed，新增 4：角色门+角色分配 small/large 两参数、四列门断言+role 断言、
  mask summary revision 进指纹链）。
- 原 `test_mask_v2_bank_and_sibling_contract` 未改动且仍绿（subject 门与几何未动）。
