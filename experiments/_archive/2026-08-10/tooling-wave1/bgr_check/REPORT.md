# F5 · bgr 风险对拍：生产渲染器 `axis_order:"bgr"` vs colour 四面体路径

- 日期：2026-08-03　执行：wave-1.5 编码 subagent　代码：`tools/bgr_check/`
- 性质：REVIEW-impl-wave1「五工具间接口一致性」表的风险移交项（T2 NOTES 待决策 #4，DECISIONS_2026-08-03 §三 F5）
- **本项为 RD-G Stage-1 开工前置 gate**

## 一、目标

生产渲染器在 `render_diagnostics` 里写 `axis_order:"bgr"`，而 colour 四面体路径（RD-G Stage-1 拿 D-RENDER after 图当 L_cube 监督时的 GT 渲染路径）按 RGB 约定。若两者语义不一致，after 图当监督会整体错色。判定三选一：

- (a) 生产 bgr 只是内部存储约定，输出与 RGB 语义一致（对拍 ΔE≈0）；
- (b) 存在确定性的通道置换，给出修正函数；
- (c) 不一致且不可简单修正。

## 二、结论（先行）

**判定 (a) 成立：`axis_order:"bgr"` 仅是 LUT 网格的内存轴序标注（`grid[b][g][r]`，值通道始终 RGB），生产 after 图的输出语义与 RGB 完全一致，不存在任何通道置换，无需修正函数。RD-G Stage-1 可以直接用 D-RENDER after 图做 L_cube 监督。**

判定依据链（详见 §五）：

1. **静态链路核验**：`.cube` 行序 R 最快 → `lut_io.load_lut` C-order reshape 得 `grid[b][g][r]`（值通道 RGB）→ 渲染器 `permute(3,0,1,2)` 得 (C,D,H,W)=(RGB, b 轴, g 轴, r 轴) → `grid_sample` 的 (x,y,z) 采样 (W,H,D)，points 通道序 (R,G,B) → **x=R 采 r 轴、y=G 采 g 轴、z=B 采 b 轴，语义正确**（dataset_build/src/construct/rendering.py:390-405）。打包快取 `luts.npz`（pack_lut_npz.py）用同一 `load_lut`，无第二套解析。
2. **解析器互验**：100 个 preset 逐一验证 `colour.read_LUT_IridasCube.table[r,g,b] == load_lut.grid[b,g,r].transpose(2,1,0,3)`，max|diff| = **2.98e-8**（float32 精度尘埃）；100 个全部 DOMAIN 缺省 [0,1]。
3. **逐像素对拍**：after vs 生产数学 CPU 复刻（trilinear），残差与**逐对实测的 JPEG q95 编解码底噪**几乎完全重合（g 线 p50 超额 max=0.0000）；字节级抽查 g 线 99.95–99.99% 像素逐字节一致。
4. **通道置换测验**：6 种输出通道置换的 MAE，100/100 对 identity(RGB) 最优。
5. **反事实灵敏度**：模拟 BGR-swap bug 的重渲染 ΔE00 中位数 7.91（g 线），检测器无盲区（selfcheck 合成注入 p50=9.32 亦被侦测）。

## 三、设置与数据

| 项 | 值 |
|---|---|
| 抽样 | 100 对，seed=20260803；g 线 50（g1/g2/g3 = 17/17/16）+ l 线 50（l1–l4 = 13/13/12/12） |
| preset 覆盖 | **100 个互不相同**（预注册要求 ≥50）；桶 quandian 79 / e18 21；lut_size ∈ {17,25,32,33,64,65} |
| 源覆盖 | 100 个互不相同 source；池 awards/korean/mmart_ppr10k/quandian/unsplash |
| I_in | img bank 原始字节（83 对**全路径精确匹配** bank 条目、17 对本机路径）→ `preprocess_source` 复刻（EXIF 转正 → RGB → 短边 1024 LANCZOS → float32/255） |
| after | groups shards 归档 `.jpg`（ranged-read，JPEG q95） |
| l 线复合 | 归档 `.cgt.png`/255 作 alpha，端点吸附混合（复刻 `composite_srgb`/GPU 分支语义） |
| 重渲路径① tri | 生产数学复刻：trilinear on `grid[b,g,r]`（`apply_lut_cpu_oracle` 逐式复刻）——隔离「插值口径差」用 |
| 重渲路径② tet | **colour 四面体**（`read_LUT_IridasCube` + `table_interpolation_tetrahedral`）——RD-G 实际监督路径 |
| 反事实 swap | `apply(img[...,::-1])[...,::-1]`：模拟 BGR 装载 bug 的样子 |
| ΔE00 | colour CIE2000（D65），与 T2/T3 工具链同口径 |

## 四、预注册判据 vs 实测

**阈值依据**：after 是 JPEG q95 有损归档，故不用 max、不用固定绝对阈值，而是**逐对实测编解码底噪做参照**——对生产数学复刻图 tri 施加与 `save_candidate_jpeg` 完全相同的编码（round→uint8→PIL JPEG q95 缺省参数）再解码，`floor = ΔE00(jpeg(tri), tri)`。若 after 与 tri 的差不超过这个底噪，则 after 的全部有损性都由 JPEG 解释，渲染语义严格一致。判据在 run_check 实现时预注册（先于全量跑）。

| 判据（预注册） | 门 | 实测 | 判定 |
|---|---|---|---|
| 置换测验 identity(RGB) 最优 | 100/100 | **100/100** | ✅ |
| after-vs-tri 的 p99 超额 ≤ floor_p99 + 1.0 | 全数 | **100/100**（g 线超额 max=0.0008，l 线 max=0.0123） | ✅ |
| after-vs-tri 的 p50 超额中位数 ≈ 0 | ≤0.1 | **0.0000**（g/l 两线均 0.0000；单对最大 0.0925，l 线） | ✅ |
| 解析器互验 max abs diff | <1e-6 | **2.98e-8** | ✅ |
| 反事实 swap 显著大于底噪（灵敏度佐证） | 中位数 ≫ floor | **g 线 swap p50 中位 7.91 vs floor 0.79** | ✅ |

## 五、实测数字

### 5.1 ΔE00 分位（100 对汇总，逐对数字见 metrics.json）

| 对拍 | p50 中位 | p99 中位 | 说明 |
|---|---|---|---|
| after vs **tri**（生产数学复刻） | 0.755 | 4.074 | 与 floor 几乎逐位重合 |
| jpeg **floor**（纯编解码底噪） | 0.755 | 4.073 | 阈值参照 |
| after vs **tet**（colour 四面体，RD-G 路径） | 0.764 | 4.074 | 四面体 vs 三线性口径差被 JPEG 噪声完全淹没（p50 中位差 0.009） |
| after vs **swap**（反事实 bug） | g 线 7.91 / l 线 1.33 | — | l 线被 mask 面积稀释；个别通道对称 LUT 天然不敏感（最低 g039 p50=0.08），判定不依赖 swap 下限 |

分线超额（tri − floor，逐对）：

| 线 | p50 超额 med/max | p99 超额 med/max | 归因 |
|---|---|---|---|
| g（50 对） | 0.0000 / **0.0000** | 0.0000 / 0.0008 | 严格一致 |
| l（50 对） | 0.0000 / 0.0925 | 0.0000 / 0.0123 | 归档 `.cgt.png` 是 8-bit 量化，而渲染时用 float alpha——带内混合微差，非通道问题 |

### 5.2 字节级抽查（after vs jpeg(tri) 的 uint8 逐像素）

| 对 | 线 | 逐字节一致像素 | max abs diff |
|---|---|---|---|
| g000 | g | 99.998% | 2/255 |
| g021 | g | 99.95% | 4/255 |
| l050 | l | 83.7% | 7/255 |
| l087 | l | 83.3% | 7/255 |

g 线达到「再编码即复现」的程度（残差 = GPU fp32 vs CPU 舍入在量化边界的翻转经 JPEG 块传播）；l 线差异由 alpha 量化涟漪解释，ΔE00 层面 p50 超额 ≤0.09。

### 5.3 通道相关矩阵（中位，行=重渲 tet、列=after）

```
raw                          delta（相对 I_in 的变化量）
[0.999 0.971 0.919]          [0.991 0.803 0.487]
[0.971 1.000 0.973]          [0.805 0.994 0.749]
[0.917 0.972 0.999]          [0.489 0.783 0.986]
```

对角显著占优（delta 对角中位 0.986–0.994）。非对角偏高是调色变换的亮度共动所致（全通道同向增减），不是通道混叠信号——判别力更强的是置换 MAE 测验（100/100 RGB 最优）与反事实 ΔE00 量级。

### 5.4 可视化（viz/，6 张；全部一致故按预案取随机 6 对：seed=20260803，g/l 各 3）

`consistent_g009 / g015 / g033 / l054 / l072 / l090.png`，每张 8 联：I_in、归档 after、tet 重渲、swap 反事实、mask（l 线）、ΔE00(after,tet) 热图、ΔE00(after,swap) 热图、|after−tet|×20。肉眼可核：after 与 tet 不可区分，ΔE00(after,tet) 仅剩 JPEG 块状/边缘噪声，而 swap 反事实成片饱和（l 线严格限于 mask 内——复合语义亦得到可视核验）。

## 六、对 RD-G Stage-1 的操作性结论

1. **直接可用**：D-RENDER after 图做 L_cube 监督无通道语义障碍，无需任何修正函数。
2. **监督噪声预算**：after 相对「理想 float 渲染」的噪声 = JPEG q95 底噪（p50 中位 ≈0.76、p99 中位 ≈4.1 ΔE00）+ 四面体/三线性口径差（p50 量级 0.01，被前者淹没）+ l 线 alpha 量化（p50 超额 ≤0.09）。训练损失设计按此定权重/鲁棒项即可。
3. **⚠ I_in 回取纪律（本实验发现的实际风险）**：unsplash 池的 I_in **必须从 `unsplash_work` bank 按 source_path 全路径匹配**回取（生产输入是 `_scratch/unsplash/` 工作副本）；按 basename 落到 `unsplash` bank 的 unsplash-lite 原图是**另一个编码**（±1px 尺寸差、像素不同，对拍超额可达 0.5 ΔE00，并曾造成本实验首轮 8 对尺寸不匹配）。任何用 D-RENDER 的 I_in 重建（含 T1 行动项 G 的「bank 命中」口径）都应升级为全路径优先匹配。
4. 次要：本抽样中 ppr10k/raise6k/fivek_gold 池共 20 例 source 未能按现行 bank 口径回取（已换样补齐，不影响判定）；D-RENDER 全量供数前建议对这三池的 I_in 可回取率单独复核。

## 七、复现

```bash
/home/bc/miniconda3/bin/python3 tools/bgr_check/selfcheck.py     # 13/13 PASS
/home/bc/miniconda3/bin/python3 tools/bgr_check/sample.py        # manifest.jsonl（seed 20260803）
/home/bc/miniconda3/bin/python3 tools/bgr_check/run_check.py --workers 8   # metrics.json（约 3 min）
/home/bc/miniconda3/bin/python3 tools/bgr_check/run_check.py --viz g009 g015 g033 l054 l072 l090
```
