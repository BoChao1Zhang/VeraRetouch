# tools/cube — T2 cube 语料工具链（D-CUBE / D-HALD）

DATA_ASSIGNMENT §2 的 D-CUBE/D-HALD 与行动项 E 的配套工具。规格书：IMPL_DOSSIER §4.3。
Python 解释器一律 `/home/bc/miniconda3/bin/python3`（依赖：colour-science 0.4.7、numpy、torch、scipy、skimage、PIL）。

## 统一约定（全链共享，见 cubelib.py）

- **canonical npy**：`(33,33,33,3) float32`，索引 `[r,g,b]`，通道 RGB，隐式 domain [0,1]（DOMAIN_MIN/MAX 与 .3dl shaper 轴在重采样时已烘焙）。
- **.cube 行序**：R 变最快（colour reshape `order='F'`，已对照 colour 0.4.7 源码核实）。
- **.3dl（Lustre 3DMESH）行序**：B 变最快；输出位深按最大值推断（OCIO FileFormat3DL 约定）。
- **Hald split（GLUT 协议，颜色空间 split）**：训 128³（每通道 8-bit 偶数值 {0,2,…,254}）→ 1024×2048×3；测 256³−128³ 留出色 → 3584×4096×3。R 最快、左上黑。存 npy（PNG 仅作可视化，写盘 PNG-only 无 JPEG）。

## 工具

### inventory.py — 盘点与差额对账（行动项 E）

```bash
python3 inventory.py --out-dir <OUT>   # 默认扫 journal-archive 全 build + recipes 根目录
```

产出：`inventory_summary.json`（对账数字）、`dcube_manifest.jsonl`（全部磁盘 LUT 一行一个：id/path/format/bucket/md5/used_in_prod/builds/preset_ids/majors/minors/dup_of_used）、`used_presets.txt`、`unused_presets.txt`、`missing_files.txt`、`dup_content.json`。

### parse.py — 全库解析 + 统一 33³

```bash
python3 parse.py --selftest                                   # 合成 LUT 自检
python3 parse.py --manifest <dcube_manifest.jsonl> \
    --npy-dir /var/cache/veradata/dcube/npy33 --out-dir <OUT> [--workers N]
```

- `.cube` 走 `colour.io.read_LUT_IridasCube`（LUT3D 用四面体插值重采样、LUT3x1D 走默认路径）；恢复通道：GBK/latin-1 再编码重试（本库 30 个 GBK 标题文件）、Resolve 方言转 `read_LUT_ResolveCube`；AppleDouble 资源叉显式拒收。
- `.3dl` 走内置 3DMESH 解析器（OCIO 约定），shaper 轴按实际位置用 scipy `RegularGridInterpolator` 重采样。
- 产出：`<npy-dir>/<id>.npy`、`parse_report.jsonl`、`parse_failures.txt`、`parse_summary.json`。

### hald.py — Hald 生成器 + 两路应用器

```bash
python3 hald.py selftest                      # 生成器不变量 + identity + 两路互拍
python3 hald.py gen --out-dir <DIR> [--png]   # hald_train_1024x2048.npy / hald_eval_3584x4096.npy
python3 hald.py apply --npy <LUT.npy> --hald <H.npy> --method grid_sample|tetrahedral --out <OUT.npy>
```

- `grid_sample`：torch 三线性、`align_corners=True`（训练路径，可微，LUT 张量按 `[r,g,b,c]→[c,b,g,r]` permute，grid=(2R−1,2G−1,2B−1)）。
- `tetrahedral`：colour `table_interpolation_tetrahedral`（GT 路径，分行 tile 控内存）。

### selfcheck.py — 三项注册自检

```bash
python3 selfcheck.py --npy-dir <NPY> --parse-report <parse_report.jsonl> \
    --hald-dir <HALD> --out-dir <OUT> [--n-presets 20 --seed 0]
```

1. **identity 恒等对拍**：identity LUT 过两路应用器打真实 hald 训/测图，max ΔE00 阈值 1e-4（"=0 量级"）。
2. **pairpath**：随机 N 个解析成功 preset，grid_sample vs 四面体在训图 + 测图 2M 像素确定性子采样上的差异报告（`pairpath_presets.jsonl`）。
3. **near-identity 剔除**：全库逐 preset 33³ 全格点 vs identity 的 ΔE00，max < 0.2 进 `near_identity_cull.txt`；分布统计 `near_identity_stats.jsonl`。

附带：colour vs skimage 的 sRGB→Lab（D65）口径互验。汇总 `selfcheck_summary.json`。

## 数据落盘位置（非 git）

- 33³ npy：`/var/cache/veradata/dcube/npy33/`（约 7,083 × 431KB ≈ 3.0GB）
- Hald 图：`/var/cache/veradata/dcube/hald/`
- 报告（git 内）：`experiments/tooling-wave1/cube/{inventory,parse,selfcheck}/` + `REPORT.md`
