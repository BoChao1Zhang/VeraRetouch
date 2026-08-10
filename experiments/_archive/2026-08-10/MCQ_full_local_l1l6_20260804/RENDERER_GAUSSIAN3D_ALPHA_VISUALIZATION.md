# Gaussian 3D + Alpha Renderer 效果

## Renderer 定义

该 arm 保持 VeraRetouch、MetaCanvas、mask head、MetaQuery、训练数据和预算不变，
只把 Config A 的 4D Gaussian renderer 换成：

```text
C = GLUT3D(I_in; theta)
I_pred = I_in + s(x,y) * (C - I_in)
```

因此颜色 LUT 是整图统一的 3D 变换，预测空间场只作为 alpha 混合系数。Config A
则把 `s` 放入 Gaussian kernel 的第四维，能够随空间条件改变局部颜色响应。

## 完整 select 对比

| 指标 | Config A 4D | 3D + alpha | 结论 |
|---|---:|---:|---|
| selection score ↓ | **37.287** | 41.620 | 4D 更好 |
| DeltaE00 p50/p90 ↓ | **3.560 / 7.086** | 3.624 / 7.320 | 3D 略差 |
| PSNR in ↑ | **22.588** | 22.569 | 基本相同 |
| PSNR boundary ↑ | 25.536 | **25.988** | 3D 边界数值略高 |
| PSNR out ↑ | **47.702** | 41.201 | 3D 明显外泄 |
| outside leakage ↓ | **0.00192** | 0.00533 | 3D 约为 2.8 倍 |
| mask AUC ↑ | **0.947** | 0.946 | 几乎相同 |
| soft-IoU ↑ | 0.605 | **0.610** | mask 不是主要差异 |
| Delta const ↑ | **0.780 dB** | 0.428 dB | 3D 空间收益较弱 |
| Delta shuffle ↑ | **1.901 dB** | 1.857 dB | 均未过 3 dB 门槛 |
| color var-ratio ↑ | **0.317** | 0.205 | 3D 颜色变化更保守 |

结论：3D + alpha 能达到接近的区域内颜色和 mask 指标，但 16×16 软空间场的尾部会
将同一全局 3D LUT 混入区域外，导致 out-PSNR 下降约 6.5 dB。当前实现下它不优于
4D renderer，不应替换 Config A。

## 20 张固定 test 可视化

![输入、目标、最终渲染总览](viz/renderer_gaussian3d_alpha/gallery20_overview.png)

[详细第 1 页](viz/renderer_gaussian3d_alpha/gallery20_p01.png) ·
[第 2 页](viz/renderer_gaussian3d_alpha/gallery20_p02.png) ·
[第 3 页](viz/renderer_gaussian3d_alpha/gallery20_p03.png) ·
[第 4 页](viz/renderer_gaussian3d_alpha/gallery20_p04.png)。

详细页列顺序为：`输入 | 目标 | 最终渲染预测 | GT 区域 | 预测区域`。

## Feature

![MetaCanvas 最终 feature](viz/renderer_gaussian3d_alpha/feature_metacanvas.png)

![Renderer MetaQuery 最终 feature](viz/renderer_gaussian3d_alpha/feature_metaquery.png)

![Instruction feature](viz/renderer_gaussian3d_alpha/feature_instruction.png)

20 条样本的原始 feature 保存在 `viz/renderer_gaussian3d_alpha/features20.npz`。
