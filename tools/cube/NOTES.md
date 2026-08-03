# T2 cube 语料工具链 · NOTES

日期：2026-08-02（实施前核实 → 实施中追加）
任务卡：T2（DATA_ASSIGNMENT 行动项 E + IMPL_DOSSIER §4.3）

## 一、实施前核实记录（外部事实，均已打开原始来源）

| # | 事实 | 来源 | 结论 |
|---|---|---|---|
| 1 | `read_LUT_IridasCube` 返回 `LUT3x1D \| LUT3D`；3D 表 reshape `(size,size,size,3), order="F"` → 文件行序 **R 变最快、B 最慢**，表索引 `table[r_idx, g_idx, b_idx] -> (R,G,B)` | context7 colour readthedocs + 本地安装源码 `colour/io/luts/iridas_cube.py:182`（colour-science 0.4.7） | 与 DOSSIER §4.3 条目 2 一致 ✅ |
| 2 | `LUT3D.apply` 对 DOMAIN_MIN/MAX 非 0–1 的 LUT：`linear_conversion(x, (domain_min, domain_max), (0,1))` 归一后查表；可传 `interpolator=table_interpolation_tetrahedral` | 本地源码 `colour/io/luts/lut.py` LUT3D.apply（line 2220 起） | 非 0–1 domain 走 colour 处理 ✅ |
| 3 | `LUT3D.linear_table(33)` 生成 identity 表，索引语义 `table[i,j,k]=(r_i,g_j,b_k)` | 同上 + readthedocs 示例 | 用作 identity LUT 与重采样网格 ✅ |
| 4 | `torch.nn.functional.grid_sample` 5D 约定：input `(N,C,D,H,W)`，grid 最后维 `(x,y,z)`，x→W、y→H、z→D，`align_corners=True` 时格点在 [-1,1] 端点 | torch 已装（本机 2.x），约定为 torch 稳定文档语义；identity 对拍在 selfcheck 中实测验证（非仅凭记忆） | LUT 表按 `permute(3,2,1,0)` 排成 `(3,B,G,R)`，grid=(2R-1,2G-1,2B-1) |
| 5 | groups.jsonl 全量扫描：6 build × 21,000 组 × 8 候选 = 1,008,000 候选；distinct `candidates[].preset_path` = **3,522**，与 `recipe.preset_path` 完全一致（对称差 0）；全部文件可读 | 本地 journal-archive 实扫 | 与 DATA_ASSIGNMENT「3,522」精确对账 ✅ |
| 6 | 磁盘 .cube 底账：`/home/bc/data/datasets/recipes/quandian` 4,032 个 + `/home/bc/data/datasets/recipes/e18` 3,014 个 = **7,046**；生产用到 quandian 1,935 + e18 1,587 | fd 实扫 | 差额池 3,524 个，足以补齐至 ~4000 |
| 7 | ΔE00 口径：sRGB(D65)→Lab 用 colour `sRGB_to_XYZ`+`XYZ_to_Lab`（默认 D65），与 skimage rgb2lab 一致性在 selfcheck 中实测（CLAUDE.md 要求 D65 sRGB 口径） | 本地实测（selfcheck 附带） | 差异 < 1e-4 量级则视为同口径 |

## 二、依赖变更

- `pip install colour-science` → **0.4.7** 新装（环境原缺）。numpy 2.4.6 / torch / scipy / skimage / tqdm 均已有。

## 三、假设清单（已自行核实或采保守默认）

1. **Hald 训练采样 128³**：按任务卡「np.mgrid 均匀 128³」取每通道 8-bit 偶数值 {0,2,…,254}（`np.mgrid` 步长 2 的自然语义，均匀 stride）；评测集 = 其余 256³−128³ 个 8-bit 颜色（至少一通道为奇数），行序 R 最快、确定性排列。1024×2048=128³、3584×4096=256³−128³ 均已核对。
2. **重采样插值器**：任意尺寸 → 33³ 用 `LUT.apply(identity_grid_33, interpolator=table_interpolation_tetrahedral)`（与 GT 应用路径同口径）。33³ 输入在格点上取值，插值器选择无影响；32³/64³ 输入存在插值/降采样损耗，原始尺寸记录进元数据。
3. **npy 语义**：shape (33,33,33,3) float32，索引 `[r,g,b]`，通道 RGB，隐式 domain [0,1]（domain 已在重采样时烘焙）。
4. **identity 对拍「=0 量级」**：判读阈值 max ΔE00 < 1e-4（float32 机器精度量级）。
5. **近恒等判据**：33³ 全格点（35,937 点，即全域格点覆盖）ΔE00 全部 < 0.2 → 进剔除清单。
6. **LUT3x1D**（若遇到）：同样经 apply→33³ 统一化，元数据标 `type=3x1D`。

## 四、待主 agent 决策（保守默认已注明，未静默拍板）

1. **33³ npy 落盘位置**：约 7,046 × 431KB ≈ 3.0GB，不宜进 git 仓库。**保守默认**：`/var/cache/veradata/dcube/npy33/`（本战役数据区，已验证可写）；仓库只存清单与报告。若需进 NFS 数据区请指示。
2. **差额补齐到 ~4000 的选法**：默认交付**全部候选池清单**（未进生产、解析成功、非近恒等、内容 md5 与已用 preset 不重复），并按 major/minor 未覆盖优先给出一个 478 个的**建议补齐清单**；最终取舍由主 agent 拍板。
3. **文档小矛盾**：DATA_ASSIGNMENT §2 D-CUBE 行写「差额补齐……行动项 B」，但 §4 行动项表中 cube 差额是 **E**（B 是 C_GT）。按 E 执行，建议修订文档。
4. **风险记录（不在 T2 范围）**：groups.jsonl `render_diagnostics.axis_order:"bgr"` 表明生产渲染器按 BGR 轴序装表（对应 DOSSIER §4.3 条目 3 对 AceTone `lut[..., ::-1]` 的警示）。本工具链两路应用器自洽且过 identity 对拍；但日后拿 **D-RENDER after 图**当 L_cube 监督 GT 与本工具链渲染对拍时，必须先核对生产渲染器轴序（建议单独安排一次生产 after 图 vs colour 四面体渲染的对拍）。
5. **T1 split 旁表未就绪**：D-CUBE 清单已带 preset_id/major/minor/bucket 字段，P-split 到位后可直接 join；本清单不自造 split（守 DATA_ASSIGNMENT 纪律）。

## 五、实施中追加记录

### 5.1 盘点发现：3,522 个生产 preset 中含 43 个 .3dl（非 .cube）

- 路径形如 `quandian/quandian_0044xx.3dl`，Picture Instruments Look Creator 生成的 Lustre `3DMESH` 格式（`Mesh 4 10`，17 点 shaper 轴 `0 64 … 960 1023`，17³=4913 行数据）。磁盘共 52 个 .3dl（quandian），43 个进了生产。
- colour-science **无 .3dl 读取器**（io/luts 仅 csp/cube/spi1d/spi3d/spimtx）。
- **外部事实核实（OpenColorIO 官方源码 `src/OpenColorIO/fileformats/FileFormat3DL.cpp`，WebFetch 原文）**：
  1. .3dl 数据行序 **B 变最快**（“The 3dl format stores the LUT entries in blue-fastest order”）——与 .cube 相反；
  2. 输出位深按数据最大值推断：max≤511→8bit(255)、≤2047→10bit(1023)、≤8191→12bit(4095)、更大→16bit(65535)；
  3. shaper 行（>3 个整数的行）是输入轴采样位置，近恒等（容差内）则跳过；`Mesh N M` 行本身仅元数据，解析时忽略。
- **处置（保守默认，待主 agent 追认）**：parse.py 内置最小 3DMESH 解析器（按上述 OCIO 约定），shaper 轴按实际位置用 scipy `RegularGridInterpolator`（支持非均匀 rectilinear）重采样到均匀 33³；shaper 严重非均匀（相对均匀轴最大偏差 >2%）记 `shaper_nonuniform` 并仍按实际位置处理。.3dl 重采样用三线性（scipy），.cube 用四面体（colour）——两者均在格点邻域内一致，差异远小于量化步长。

### 5.2 全库解析首轮 46 个失败的归因与恢复（均不在生产 3,522 内）

- **15 个 AppleDouble 资源叉**（魔数 `0x00051607`，176–542 字节，macOS 元数据非 LUT）→ 显式拒收，保留在失败清单（原因注明），不可恢复。
- **30 个 GBK 标题的 IWLTBAP cube**（如 `TITLE "小武拉莫 傍晚蓝"`，非 UTF-8 字节仅在注释/标题）→ 恢复通道：utf-8 失败后按 gbk→latin-1 链再编码到临时文件重解析，元数据记 `reencoded_from`。
- **1 个 DaVinci Resolve 方言**（`LUT_3D_INPUT_RANGE` 关键字）→ 转 `colour.io.read_LUT_ResolveCube`（本地已验证返回 LUT3D 33³ domain [0,1]），元数据记 `dialect=resolve`。
- 恢复后成功率 7,083/7,098 = **99.79%**；生产 3,522 个 preset **解析零失败**。

### 5.3 Lab 口径互验的白点差异（已归因，非缺陷）

selfcheck 的 colour vs skimage `rgb2lab` 互验首轮报 max 偏差 0.023：根因是 **D65 白点取值精度**——skimage 用经典圆整 (0.95047, 1, 1.08883)，colour 从色度 (0.3127, 0.3290) 推导 (0.950456, 1, 1.089058)。两工具链对**同一对图像**算 ΔE00 的口径差 ≤0.004（白点偏移在差值中基本抵消），比近恒等判据 0.2 低两个量级。互验门修订为：Lab 绝对偏差 <0.05 且 ΔE00 口径差 <0.01。

### 5.4 ImageMagick 独立交叉验证（DOSSIER §4.3 条目 8）

`convert hald:8` 的编码实测 R 最快、左上黑（与我方 Hald 约定一致）；用真实 preset（e18_000001）走「IM 生成 identity hald → 我方四面体烘焙 → IM `-hald-clut` 应用于测试图 vs 我方直接应用」端到端对拍：mean ΔE00 0.027 / max 0.75 / p99<1（残差 = IM 64³ 三线性 vs 我方 33³ 四面体 + 16-bit 量化）。行序/domain 若有错会呈几十 ΔE 量级 → **约定验证通过**。工具：im_crosscheck.py。

### 5.5 parse.py selftest 中 gamma17 断言的放宽说明

17³ LUT 对 x^(1/2.2) 曲线在暗部（节点 0 与 1/16 之间）线性插值误差解析值 ≈0.065（x=1/32 处 0.207 vs 0.142）——是 17 点网格的固有插值误差而非解析器缺陷；断言改为暗部 <0.08 + 中高亮区 <5e-3 双条件。

## 六、wave-1.5 修复记录（2026-08-03，F4 Pyright 扫尾）

仅类型/导入层面，零行为变更（DECISIONS_2026-08-03 §三 F4；cube 模块不在 F1–F3 覆盖内，归 F4 扫尾）：

- `im_crosscheck.py`：`cv2.imread` 返回 `Optional`——加 `img is None → raise IOError` 守卫（成功路径不变；
  失败路径原为 AttributeError 崩溃，现为带路径的明确报错）。
- `parse.py`：`RegularGridInterpolator(..., fill_value=None)` 为 scipy 文档语义（None=线性外推），stub 标注过窄为 float，
  加 `pyright: ignore[reportArgumentType]` 注释保留原行为；`os.cpu_count()` 可空 → `(os.cpu_count() or 2) // 2`（本机恒非空，值不变）。
- `selfcheck.py`：`_NODES_LAB/_NPY_DIR` 模块全局补 `| None` 类型标注 + `_ni_worker` 内 assert 收窄
  （`_ni_init` 作为 Pool initializer 在每个 worker 进程先行执行，assert 恒真）；`os.cpu_count()` 同上守卫。

复验（均输出至 scratchpad，不覆盖冻结产物）：

- `pyright tools/cube/` → **0 errors**（修复前 12 条）。
- `parse.py --selftest` → OK。
- `im_crosscheck.py`（同 preset e18_000001）→ mean ΔE00 0.0268571682 / max 0.7523918510，与冻结 im_crosscheck.json **逐位一致**。
- `selfcheck.py` 全量重跑（同参 --n-presets 20 --seed 0）→ 见下方追记。

### 六.1 追记（2026-08-03，F7）：F4 复验重跑中断的死因与 4/4 补跑结果

**中断事实修正**：F4 当时的重跑并非死在 nearident——日志（scratchpad `cube_selfcheck.log`）末行停在
`[3/4] pairpath` 第 3/20 个 preset（e18_000532），`[4/4] near-identity` 根本未开始。

**死因判定：外部 SIGKILL（Bash 工具默认 120s 超时），非代码缺陷**。证据链：

1. 进程存活窗口 ≈70–120s：日志 birth 00:51:24.5 → 末笔写入 00:52:35.1（+70.6s），此后无任何输出、无 Python traceback——
   与前台 Bash 默认 120000ms 超时被 SIGKILL 吻合（该次重跑在回合内前台启动，未传更长 timeout）。
2. 死锁排除：[3/4] pairpath 段是纯单进程循环（mp.Pool 只在 [4/4] 用）；且 F7 冒烟（200 preset 子集全 4 段）与全量重跑均正常走完 [4/4] 的 Pool。
3. OOM 排除：全量重跑峰值 RSS 1.6GB（/usr/bin/time -v），机器 125GB；systemd-oomd/earlyoom 均 inactive，journal 无 oom-kill 记录。

**4/4 补跑（F7，后台任务）**：同参（--n-presets 20 --seed 0，parse_report 7,083 条），wall 3m22s、exit 0。
`selfcheck_summary.json`、`near_identity_cull.txt`（101 个）、`pairpath_presets.jsonl` 与冻结产物
`experiments/tooling-wave1/cube/selfcheck/` **逐字节一致**；`near_identity_stats.jsonl` 内容一致。
即：F4 的类型层修复零行为变更这一结论最终确认。输出仅进 scratchpad（`f7_full/`），未覆盖冻结产物。

**流程教训**（长任务纪律）：selfcheck 全量 ≈3–4 分钟本属短任务，但回合内前台跑必须显式给足 timeout 或走后台——
默认 120s 会静默 SIGKILL，日志无任何报错痕迹，极易误判为死锁/OOM。

## 七、wave-1.5 追记（2026-08-03，F5 结案）：待决策 #4「生产渲染器 bgr 轴序」已核销

本文件「待主 agent 决策 #4 / D4」提出的风险（生产 `render_diagnostics.axis_order:"bgr"` vs 本工具链 RGB 约定，拿 D-RENDER after 当 L_cube 监督前须对拍）已由 wave-1.5 F5 完成：100 对（100 preset、g/l 各 50、7 个 prod build）对拍，**判定 (a)——`bgr` 仅是 `load_lut grid[b][g][r]` 的内存轴序标注，值通道恒 RGB，生产 after 输出语义与本工具链（colour table[r,g,b] 四面体）严格一致**：解析器互验 max|diff|=2.98e-8，g 线 after 与生产数学复刻+q95 再编码逐字节一致率 99.95%+，通道置换测验 100/100 identity。RD-G Stage-1 gate 放行，无需修正函数。证据与工具：`experiments/tooling-wave1/bgr_check/REPORT.md`、`tools/bgr_check/`。
