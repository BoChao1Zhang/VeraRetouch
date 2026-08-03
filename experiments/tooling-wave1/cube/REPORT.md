# T2 cube 语料工具链 · REPORT

- 任务：tooling wave-1 T2（DATA_ASSIGNMENT §2 D-CUBE/D-HALD + §4 行动项 E；规格书 IMPL_DOSSIER §4.3）
- 代码：`tools/cube/`（inventory.py / parse.py / hald.py / selfcheck.py + cubelib.py、im_crosscheck.py、propose_supplement.py、viz_report.py）
- 环境快照：`config/env.json`（git 0e5d04a7，seed=0，colour-science 0.4.7 / torch 2.6.0 / numpy 2.4.6）
- 实施前核实与待决策项：`tools/cube/NOTES.md`

## 1. 目标

为 D-CUBE（preset .cube 全量）与 D-HALD（Hald 训测对，颜色空间 split）提供盘点、解析、生成与自检工具链；完成行动项 E 的 3,522 差额对账并交付 D-CUBE 清单。

## 2. 设置与数据

- 生产入口：journal 归档 6 个完成 build（prod-g1/g2/g3 + prod-l1/l2/l3）的 `groups.jsonl`，1,008,000 候选行，收集 `candidates[].preset_path` 与 `recipe.preset_path`。
- 磁盘底账：`/home/bc/data/datasets/recipes/{quandian,e18}`。
- 统一化：全部 LUT 重采样为 canonical npy `(33,33,33,3) float32`、索引 `[r,g,b]`、RGB、domain [0,1]（DOMAIN_MIN/MAX 与 .3dl shaper 轴烘焙进表）→ `/var/cache/veradata/dcube/npy33/`（3.0 GB，不进 git）。
- D-HALD：训 128³（每通道 8-bit 偶数值）→ 1024×2048×3；测 256³−128³ 留出色 → 3584×4096×3；R 最快、左上黑（GLUT/Hald 协议，DOSSIER §4.3 条 5/6/9）→ `/var/cache/veradata/dcube/hald/`。
- 应用器两路：torch `grid_sample`（三线性，`align_corners=True`，训练路径）与 colour 四面体插值（GT 路径）。
- split 纪律：本工具链不自造 split；清单带 preset_id/major/minor/bucket 字段，T1 P-split 旁表就绪后直接 join。

## 3. 预注册判据 vs 实测

| 判据（任务卡预注册） | 实测 | 判定 |
|---|---|---|
| 行动项 E：distinct preset_path 与 3,522 对账 | distinct = **3,522**（candidates 与 recipe 两口径对称差 0；磁盘全部可读，missing=0） | ✅ 精确对账 |
| identity LUT 对拍 max ΔE=0 量级（阈值 1e-4） | 四面体路径 max ΔE00 = **0.0**（逐位恒等）；grid_sample 路径 max ΔE00 = **4.2e-5**（float32 精度） | ✅ |
| 全库解析成功率 | **7,083/7,098 = 99.79%**（恢复通道后）；生产 3,522 个 **零失败**；15 个失败均为 AppleDouble 资源叉（非 LUT，显式拒收，见 `parse/parse_failures.txt`） | ✅ |
| 近恒等剔除报告（全域 ΔE00<0.2） | **101 个**进剔除清单（`selfcheck/near_identity_cull.txt`），**全部不在生产 3,522 内**；全库 ΔE00_max 分位 p1=0.13 / p50=33.9 / p99=77.1（`near_identity_stats.jsonl`） | ✅ |
| D-CUBE 清单落盘 | `inventory/dcube_manifest.jsonl`（7,098 行：id/path/format/bucket/md5/used_in_prod/builds/preset_ids/majors/minors/dup_of_used）+ `used_presets.txt`(3,522) | ✅ |

### 随机 20 preset 两路互拍（grid_sample vs 四面体，非门槛、注册报告项）

留出色集 2M 像素子采样：ΔE00 均值平均 **0.033**、20 个中位 max **1.08**、最差 **6.38**（quandian_009126，路径间 PSNR 仍 ≥53.5 dB；逐 preset 见 `selfcheck/pairpath_presets.jsonl`，最差样例可视化 `viz/panel_quandian__quandian_009126.png`）。差异集中在高曲率格元的三线性/四面体固有分歧，均值量级正常；**含义**：训练走 grid_sample、GT 走四面体时，评测存在 ~0.03 ΔE00 的口径底噪，个别色点可到数个 ΔE00（见 §5 建议）。

### 附加交叉验证（DOSSIER §4.3 条 8）

ImageMagick 独立对拍（`hald:8` 编码确认 R 最快、左上黑；真实 preset e18_000001 端到端 IM `-hald-clut` vs 自家四面体）：mean ΔE00 0.027 / max 0.75——行序或 domain 错误会呈几十 ΔE 量级，**约定验证通过**（`selfcheck/im_crosscheck.json`）。colour vs skimage 的 sRGB→Lab(D65) 口径互验通过（ΔE00 口径差 ≤0.007，白点圆整差异已归因，NOTES §5.3）。

## 4. 行动项 E：差额对账与补齐提案

| 项 | 数字 |
|---|---|
| 生产用 distinct preset | 3,522（quandian 1,935 + e18 1,587；.cube 3,479 + .3dl 43） |
| 磁盘 LUT 总量 | 7,098（quandian 4,084 + e18 3,014） |
| 未进生产 | 3,576，其中内容 md5 与已用重复 965、解析失败 15、近恒等 101 |
| 去重去废后候选池 | **1,981**（`inventory/supplement_pool.txt`） |
| 补齐至 4,000 差额 | **478**；按 bucket 均衡 + major/minor 未覆盖优先的建议清单 `inventory/supplement_proposal.txt`（e18 239 + quandian 239；与已用/剔除零重叠） |

差额结论：磁盘池充足，478 个建议清单已交付，**最终取舍待主 agent 拍板**（NOTES 待决策 #2）。

## 5. 结论与建议下一步

1. 工具链四件全部落地并过自检；D-CUBE/D-HALD 可直接供 E1/E1b/E22/RD-G 使用。E1 的 400 个分层子集可从 `dcube_manifest.jsonl` 的 major/minor 字段直接抽。
2. **建议（评测口径）**：E1/E22 报 Hald 指标时注明应用路径（grid_sample vs tetrahedral）；对比外部数字（GLUT 45.5 dB 锚点）时用四面体 GT 口径，训练内部 checkpoint 比较用同路径自洽口径，避免 ~0.03 ΔE00 底噪混入结论。
3. **风险移交（不在 T2 范围）**：生产渲染器 `render_diagnostics.axis_order:"bgr"`——拿 D-RENDER after 图当 L_cube 监督时，必须先做一次生产 after 图 vs colour 四面体渲染对拍（NOTES 待决策 #4）。
4. **文档修订建议**：DATA_ASSIGNMENT §2 D-CUBE 行的「行动项 B」应为「行动项 E」（NOTES 待决策 #3）。
5. 43 个生产 .3dl 为 Lustre 3DMESH 格式，parse.py 按 OCIO 官方约定内置解析（B 最快 + 位深推断 + shaper 轴），52/52 全解析成功；处置为保守默认，待主 agent 追认（NOTES §5.1）。

## 6. 交付物索引

```
experiments/tooling-wave1/cube/
  REPORT.md                          本文
  config/env.json                    git commit / seed / 环境 / 命令
  inventory/  inventory_summary.json dcube_manifest.jsonl used_presets.txt
              unused_presets.txt missing_files.txt dup_content.json
              supplement_{pool,proposal}.txt supplement_summary.json
  parse/      parse_summary.json parse_report.jsonl parse_failures.txt
  selfcheck/  selfcheck_summary.json near_identity_cull.txt
              near_identity_stats.jsonl pairpath_presets.jsonl im_crosscheck.json
  viz/        hald_train_1024x2048.png hald_eval_preview_1024x896.png
              panel_quandian__quandian_009126.png   ← 最差互拍案例（失败样例）
              panel_quandian__quandian_011765.png panel_e18__e18_000001.png
```
