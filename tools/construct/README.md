# tools/construct — INF-2 构造数据生成器（容量阶梯 L0–L7）

任务卡 T4（tooling wave-1）。规格出处：PLAN §3·第二级「合成数据」、EXPERIMENTS_v3 INF-2 行、
DATA_ASSIGNMENT D-CONSTRUCT。设计决策与待拍板项见同目录 `NOTES.md`。

## 模块

| 文件 | 内容 |
|---|---|
| `splits.py` | S-split：**默认强制读 T1 冻结旁表 `tools/data_splits/splits.sqlite3`**（sqlite/csv/jsonl 均可外挂）；旧内联 sha1 规则已 DEPRECATED，仅当旁表文件不存在时 fallback，且 fallback 启动时若旁表存在会抽 1000 源做一致性校验、不一致即拒绝启动（wave-1 审阅 B3 修复） |
| `masks.py` | 几何掩膜采样器（radial/linear/elliptical/vignette，smoothstep 羽化，面积二分命中 ±1%）+ 语义掩膜库（l 系 shards 按 idx.jsonl 偏移量随机读 `.cgt.png` / `.in.*` / `.vrmeta.json`） |
| `transforms.py` | 五类变换（Exposure ΔEV / WB 对角 / Sat / Hue / Tone γ）×4 幅度档 + 困难对照（3 结点非单调分段线性）；档值以 **PLAN §3 2026-08-03 内联权威表**为准（D-06）；sRGB [0,1]；`O=(1−m)·I+m·T1(I)` |
| `generate.py` | L0–L7 生成 + manifest.jsonl（含 `feather_kind` 字段，D-05）+ 3×3 拼图；CLI 见下 |
| `selftest.py` | 15 项自检（真实 shard 上跑；含旁表口径 split 检查与 manifest×旁表 0 容忍硬门） |

## CLI

```bash
cd /home/bc/VeraRetouch

# 1) 扫 l 系 build 建源目录（含 source_id / 成员偏移量）
python3 -m tools.construct.generate build-catalog \
    --out experiments/tooling-wave1/T4_construct/cache/source_catalog.jsonl \
    --batches-per-build 4        # 每 build 扫前 N 个 batch；6 build 共 ~14k 源

# 2) 生成（split ∈ train/val/test，源按 T1 冻结旁表过滤；两套独立；正式格式 PNG，D-08）
python3 -m tools.construct.generate generate \
    --catalog experiments/tooling-wave1/T4_construct/cache/source_catalog.jsonl \
    --out-dir experiments/tooling-wave1/T4_construct/sanity \
    --split train --per-level 200 --seed 20260802 \
    --img-format png \
    --viz-dir experiments/tooling-wave1/T4_construct/viz \
    [--levels L0 L4 ...]         # 默认全部 L0–L7
    [--split-table 路径]         # 覆盖旁表文件（sqlite3/csv/jsonl）；默认 tools/data_splits/splits.sqlite3

# 3) 自检（--manifests 触发 manifest×T1 旁表逐条一致的 0 容忍硬门）
python3 -m tools.construct.selftest --catalog <catalog.jsonl> \
    --manifests <out-dir>/train/manifest.jsonl <out-dir>/val/manifest.jsonl \
    [--out-json selftest.json]
```

## 输出布局

```
<out-dir>/<split>/
  manifest.jsonl          # 每样本一行：全部参数 + seed_ints + split 规则 + 溯源 + 文件名
  gen_stats.json          # 每级：面积命中率 / IoU 约束达成 / 源重试 / 耗时
  L0..L7/{uid}_in.png  {uid}_out.png  {uid}_mask.png   # in/out 扩展名随 --img-format
        L5 另有 {uid}_maskrender.png（实际渲染掩膜，训练侧不可见）
        L6 另有 {uid}_maska.png {uid}_maskb.png（GT 全保留；_mask.png=max(a,b) 便携版）
<viz-dir>/L{k}_{split}_grid.png   # 3 行样本 ×（input | mask | target(GT)）
```

## 阶梯语义（详见 NOTES.md §5）

L0 全局（m≡1）｜L1 二值语义（cgt≥0.5）｜L2 与 L1 同构造、独立种子（差异在实验期 s 来源）｜
L3 软 matte（cgt 原软掩膜 + 高斯羽化档）｜L4 几何四族 ×羽化 4 档 ×面积 4 档｜
L5 错配负控制（radial 渲染、语义 GT，IoU<0.3）｜L6 双重叠软掩膜顺序双变换（秩上限）｜
L7 线性边界横切同色区（跨界色距 ~0.003 vs 普通 linear ~0.33，边界不可由 RGB 推断）。

## 复现

manifest 每行含 `seed_ints=[master_seed, level_id, index]`；同 catalog + 同 seed 重跑
逐 bit 一致（selftest `replay.bitwise` 断言）。幅度/羽化/面积三轴按 index 确定性轮转，
200 张内全档覆盖。
