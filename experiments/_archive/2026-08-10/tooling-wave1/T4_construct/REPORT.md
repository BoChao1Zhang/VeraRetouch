# T4 · INF-2 构造数据生成器（容量阶梯 L0–L7）· 交付报告（wave-1.5 B3 修复重生成版）

日期：2026-08-03 ｜ 编码 subagent ｜ git commit：见 `config/run_config.json` / manifest 逐行 `git_commit`

## 目标

按 PLAN §3·第二级「合成数据」与 EXPERIMENTS_v3 INF-2 行，实现容量阶梯 L0–L7 合成数据生成器：
几何四族 / 变换五类四档 + 困难对照 / 羽化 σ∈{0,2,8,24}px / 面积 α∈{5,15,40,70}%；
源图与语义掩膜取自家 l 系生产 build（DATA_ASSIGNMENT D-CONSTRUCT）。

**本版为审阅 B3（split 纪律违规）修复后的重生成交付**：S-split 一律读 T1 冻结旁表
`tools/data_splits/splits.sqlite3`（内联 sha1 规则已 DEPRECATED 并加拒启守卫）；
正式格式 PNG（D-08）；幅度档对齐 PLAN §3 权威表（D-06）；manifest 增 `feather_kind`（D-05）。
wave-1 内联 split 的 JPEG 旧批已整体移入 `_deprecated/`。

## 设置

- 代码：`tools/construct/{splits,masks,transforms,generate,selftest}.py`。
- 源池：`prod-l{1..6}-local17k` 六 build × 前 4 batch，39,861 候选 / 14,374 源；
  经 T1 旁表过滤后 **train 12,865 / val 747 / test 760 / unknown 2**（unknown 自动排除）。
- 语义掩膜：只取 `slot_id=semantic-*` 候选的 `.cgt.png`（l 系混有几何槽掩膜，semantic 仅 ~13.5%）。
- 合成公式 `O=(1−m)·I+m·T1(I)`（T0=恒等）；长边 1024；RGB Lanczos / mask bilinear。
- 执行方式（长任务纪律）：train 逐级前台生成（每级一条 `--levels Lk` 命令，144–212s/级，
  共 1438s），**逐级过 0 容忍 split 验证后**按 L0→L7 拼装 manifest；样本确定性按
  `(master_seed, level_id, index)` 逐样本独立、采样器无状态，与单命令全量跑逐 bit 等价
  （selftest `replay.bitwise` 断言）。val 单命令前台跑完（215s）。

## 数据（split 代号）

D-CONSTRUCT：train 套 = **S-train**（T1 旁表，seed 20260802，8 级 ×200 = 1600），
val 套 = **S-val**（T1 旁表，seed 20260803，8 级 ×24 = 192）；
每行 manifest 记 `split_rule: t1_side_table:splits.sqlite3(rows=33652)`；两套文件、manifest、统计完全独立。

## 预注册判据 vs 实测

| 判据 | 实测 | 结论 |
|---|---|---|
| train 8 级 ×200 + val 8 级 ×24，PNG，seed 20260802/20260803 | 1600 + 192 全数生成（`sanity/{train,val}/`，in/out/mask 全 PNG） | 过 |
| **split 0 容忍**：train 全 ∈ S-train、val 全 ∈ S-val（T1 旁表逐条） | 逐级独立核验 8+8 级全 0 错配；selftest `split.manifest_vs_table` **train 1600/1600、val 192/192** | 过 |
| 每级一张 3×3 拼图（输入/掩膜/GT）落 viz/ | `viz/L{0..7}_{train,val}_grid.png` 共 16 张（本次重出） | 过 |
| manifest 可复现（seed 记录） | 每行含 `seed_ints=[master_seed, level_id, index]` + 全参数 + 溯源 + `git_commit`；`replay.bitwise` PASS | 过 |
| 面积档命中（±1%） | L4 200/200、L5 渲染掩膜 200/200 全中（求解器 worst 误差 0.0029，selftest 64 组合 0 失败） | 过 |
| 自检 | **15/15 PASS**（含旁表背书、内联 fallback 拒启守卫、manifest×旁表 0 容忍硬门；`config/selftest.json`） | 过 |

关键分布数字（`metrics.json`）：

- **L5 错配**（负控制）：IoU(渲染 radial, 语义 GT) 中位 0.126 / p90 0.282，约束 <0.30 达成
  train 188/200（val 24/24）；超限样本取 6 次重采最小值并记录（max 0.561），可按 `iou_constraint_met` 过滤。
- **L6 双重叠**：二值化 IoU ∈ [0.10, 0.55] 达成 train 200/200（中位 0.288，min 0.100 / max 0.540）；双掩膜 PNG 全保留。
- **L7 同色区横切**：跨界 sRGB 色距中位 **0.0023**（selftest 同池普通 linear 掩膜抽样对照中位 0.076，差 ~33×），
  「边界不可由 RGB 推断」的构造性质成立；homog 窗口 std 中位 0.0061（门限 0.06，max 0.0545）。
- 源多样性：train 1600 样本用 1488 个不同源（最大复用 3；src/ppr10k/raise = 1354/194/52）；
  val 192 样本用 168 个不同源（最大复用 4）。

## 交付物

```
experiments/tooling-wave1/T4_construct/
  REPORT.md  metrics.json
  cache/source_catalog.jsonl          # 源目录（含 slot_id 与成员偏移）
  sanity/{train,val}/manifest.jsonl   # 1600 + 192 行，全参数可复现
  sanity/{train,val}/L{0..7}/         # *_in.png *_out.png *_mask.png（L5 另 *_maskrender.png；L6 另 *_maska/_maskb.png）
  sanity/{train,val}/gen_stats.json
  viz/L{0..7}_{train,val}_grid.png    # 16 张 3 行 ×（input|mask|target(GT)）拼图
  config/{run_config.json, selftest.json, gen_train.log, gen_val.log}
  _deprecated/                        # wave-1 内联 split JPEG 旧批（B3 污染，勿用）
```

工具使用说明见 `tools/construct/README.md`；设计与待决策见 `tools/construct/NOTES.md`。

## 结论与建议下一步

1. **B3 数据侧关账达成**：train+val 两套按 T1 冻结旁表全量重生成（PNG），0 容忍 split
   验证全过（1600/1600 + 192/192），15/15 自检全绿——D-CONSTRUCT 可供 W1 容量阶梯实验使用。
2. 正式 ≥2000/级 生产只需 `--per-level 2000`（建议先扩 catalog：`--batches-per-build 16`
   以上，semantic 源余量更足）；属长任务，按纪律走 gflow/nohup。
3. 待主 agent 拍板（`tools/construct/NOTES.md` §七）：D8 变换域实现与 PLAN 表述的口径差
  （档值已对齐，域实现维持现状）；⚑U2 catalog 是否收缩在途 build（旁表 unknown 已自动排除，实害为零）。
4. L5 的 IoU<0.3 硬约束在大面积档偶然不可达（train 12/200 记录在案，val 0/24）；
   若实验侧要求硬约束，可把 L5 面积档限制到 {5,15,40}%（一行改动）。
