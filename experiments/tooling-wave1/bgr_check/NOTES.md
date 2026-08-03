# NOTES · F5 bgr 风险对拍（tools/bgr_check）

## 一、实施前核实记录（全部本机原始来源，无外部 URL）

1. **生产 LUT 装载链**：`dataset_build/lut_io.py:10-36`（`.cube` R 最快行序 → C-order reshape → `grid[b,g,r]`，值通道 RGB）；`dataset_build/src/construct/rendering.py:390-405`（`permute(3,0,1,2)` + `grid_sample` points=(R,G,B)→(W,H,D)）；`dataset_build/tools/pack_lut_npz.py`（luts.npz 用同一 `load_lut` 预解析，`/var/cache/veradata/preset_bank_full/luts_meta.json` 实查 dmin/dmax 全 [0,1]）。静态推导：语义 RGB 正确，`bgr` 仅轴序标注。
2. **preprocess 语义**：`rendering.py:49-59`（short_edge=1024、LANCZOS、float32/255）+ `archive_reader.open_rgb`（EXIF transpose → RGB）；prod toml 实查 `short_edge=1024`、`jpeg_quality=95`（config.py:506 还有硬校验）。
3. **after 落盘语义**：`rendering.py:417-428` `save_candidate_jpeg`（round→uint8→PIL JPEG q95 缺省参数）。
4. **l 线复合语义**：`rendering.py:311-314`（alpha==0/1 端点吸附）+ `save_cgt_png`（alpha 8-bit 量化归档——对拍时的已知噪声源，实测 p50 超额 ≤0.09 ΔE00）。
5. **colour 侧约定**：沿用 T2 已审证据（colour 0.4.7 order='F'，table[r,g,b]），并在 100 个 preset 上做逐一数值互验（max 2.98e-8）兜底。
6. **数据定位**：journal `groups.jsonl`（recipe.preset_path/candidate_id/render_mode）；after/cgt 在 `/mnt/nfs/bc/data/datasets/groups/<build>` 的 idx.jsonl + tar ranged-read（T5 oracle 同约定）；I_in 在 img bank（metadata.jsonl + catalog.sqlite3）。路径常量对齐 `tools/data_splits/vr_common.py`。

## 二、假设与当场核实

| 假设 | 核实结果 |
|---|---|
| bank 归档字节 == 生产渲染输入 | **unsplash 池不成立需修正**：生产输入是 `_scratch/unsplash/` 工作副本，与 unsplash-lite 原图（basename 相同）是不同编码。改为 source_path 全路径优先匹配后命中 `unsplash_work` bank，首轮 8 对 ±1px 尺寸不匹配与 13 对 ΔE 超额全部消失。其余池（awards/korean/quandian=primary、mmart=本机路径）全路径/精确匹配成立 |
| 本机 PIL 解码与生产一致 | 间接核实：g 线 after 与 CPU 复刻+q95 再编码逐字节一致率 99.95%+，说明解码/编码栈行为一致 |
| .cube 全部 DOMAIN 缺省 | 100/100 实测缺省 [0,1]（与 luts_meta.json 全库一致） |
| journal 的 lut 候选覆盖 .3dl | 抽到的 100 对全为 `.cube`（sampler 显式限 .cube，.3dl 生产占比小且 T2 已单独核 OCIO 约定，不影响本判定范围） |

## 三、待主 agent 决策（无阻塞项，均为建议）

1. REPORT §六-3：I_in 回取全路径优先匹配是否回灌 T1 `action_g_render_audit.py` 的 bank 命中口径（现按 basename stem；对 unsplash 池可能把「可用」判给另一个编码的文件）。建议采纳，属口径修正非本 gate 阻塞。
2. REPORT §六-4：ppr10k/raise6k/fivek_gold 三池 20 例回取失败是否在 D-RENDER 全量供数前单独复核。

## 四、wave-1.5 修复记录

- **F5（本项，REVIEW「五工具间接口一致性」风险移交 / DECISIONS §三）**：新建 `tools/bgr_check/`（common/sample/run_check/selfcheck），100 对（100 preset、g/l 各 50、7 个 prod build、lut_size 6 档）对拍完成，**判定 (a)**：`axis_order:"bgr"` 仅内部轴序标注，输出 RGB 语义严格一致（g 线 tri−floor p50 超额 max=0.0000；置换测验 100/100 RGB；反事实 swap 中位 7.91 ΔE00 证明检测灵敏）。无需修正函数，RD-G Stage-1 gate **放行**。
- 开发中修正 2 处 selfcheck 规格错误（非检测器错误）：① BGR-*apply* bug 不是纯输出通道置换，perm 测验对其不必然报警——其检测器是 ΔE00 量级门（合成注入 p50=9.32 被侦测）；perm 测验改为核「输出侧 BGR 置换」（假设 b 的直接检测器），PASS。② corr(self) 非对角是随机图的天然跨通道相关（~1/√N），阈值从 1e-9 改 0.05。
- 修正 BankResolver：source_path 全路径匹配优先于 basename stem 匹配（见 §二 unsplash 教训）。
- selfcheck 最终 **13/13 PASS**；全量 100/100 无错误。
