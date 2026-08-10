# MetaCanvas basis where 全量实验可视化报告

生成时间：2026-08-04  
Checkpoint：两个 arm 均为 `step=6000` 的 `best.pt`  
评估集：完整 `select`，共 4,225 条；可视化集为固定、L1-L6 分层的 20 条 `test` 样本。

## 结果摘要

| arm | selection score ↓ | ΔE00 p50 ↓ | AUC ↑ | soft IoU ↑ | outside leakage ↓ | PSNR in ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Geo8 | **42.427** | **3.743** | **0.915** | **0.528** | 0.002500 | **22.029** |
| VLM14 | 46.319 | 3.775 | 0.866 | 0.503 | **0.002454** | 21.829 |

在完整 select 上，Geo8 是当前更好的方案。它的 AUC、soft IoU、输入区域 PSNR 和综合 selection score 都优于 VLM14。VLM14 的语义 basis 确实让预测 mask 出现更多图像相关细节，但细节较噪，尚未稳定转化为更好的 where 指标。

## 直接对比

- [20 张最终渲染并排对比](viz/compare/render_compare20.png)
- [20 张 where 并排对比](viz/compare/where_compare20.png)
- [20 张 basis 系数并排对比](viz/compare/coefficients_compare20.png)

渲染总览中，每个样本依次为输入、目标、预测。where 总览中，每个样本依次为输入、GT mask、预测 mask、`latent s`；`latent s` 使用蓝-白-红发散色图，蓝色为负值，红色为正值。

## Geo8

- [20 张渲染总览](viz/basis_geo_range8_full/gallery20_overview.png)
- [20 张 where 总览](viz/basis_geo_range8_full/where20_overview.png)
- [where 细节第 1 页](viz/basis_geo_range8_full/where20_p01.png)
- [8 维系数热图](viz/basis_geo_range8_full/where_coefficients20.png)
- [全部 basis 加权贡献第 1 页](viz/basis_geo_range8_full/where_basis_contrib20_p01.png)
- [最终 8-query memory 特征](viz/basis_geo_range8_full/feature_metacanvas.png)
- [49 个 GLUT 参数 query 特征](viz/basis_geo_range8_full/feature_metaquery.png)
- [空间 basis](viz/basis_geo_range8_full/feature_spatial_basis.png)

Geo8 主要依赖低频几何项。20 条样本的系数图显示，`P2(x)`、常数项、`y` 和 `P2(y)` 通常占主导，`L/S` 系数很小。这解释了它的 mask 边界平滑、稳定，但难以贴合人物轮廓或复杂语义区域。

## VLM14

- [20 张渲染总览](viz/basis_vlm14_full/gallery20_overview.png)
- [20 张 where 总览](viz/basis_vlm14_full/where20_overview.png)
- [where 细节第 1 页](viz/basis_vlm14_full/where20_p01.png)
- [14 维系数热图](viz/basis_vlm14_full/where_coefficients20.png)
- [全部 basis 加权贡献第 1 页](viz/basis_vlm14_full/where_basis_contrib20_p01.png)
- [最终 8-query memory 特征](viz/basis_vlm14_full/feature_metacanvas.png)
- [49 个 GLUT 参数 query 特征](viz/basis_vlm14_full/feature_metaquery.png)
- [14 张空间 basis](viz/basis_vlm14_full/feature_spatial_basis.png)

VLM14 的 `e1...e6` 能产生明显的图像相关响应，人物、前景和纹理结构在部分样本中可见；但其空间响应存在块状噪声，并且全局系数幅度通常不大。当前结果说明语义 basis 提供了更强的形状表达能力，但其投影和系数学习还没有被监督充分利用。

## 张量说明

两个 basis arm 并不生成 16×16 的可学习 MetaCanvas feature。兼容文件名 `feature_metacanvas.png` 中实际保存的是 VLM 后、connector 后的最终 8-token query memory，数组形状为 `20×8×384`。`feature_metaquery.png` 是 renderer 的 49 个 GLUT 参数 decoder query，形状为 `20×49×384`。

真正的 where 空间张量是：

- Geo8：`spatial_basis = 20×256×8`，`spatial_coeff = 20×8`
- VLM14：`spatial_basis = 20×256×14`，`spatial_coeff = 20×14`
- 两臂均保存 `pred_mask`、`latent_s` 和 `renderer_s`，形状为 `20×128×128`

全部原始数组位于各 arm 的 `features20.npz`。每个 arm 的 `where_basis_contrib20_p01...p05.png` 覆盖全部 20 条样本，每张 basis contribution 都是 `basis_i × w_i`，同一样本内共享色标，因而可以直接比较各 basis 对最终 where 场的贡献大小。
