# Config A/B 可视化与结构说明

## 结论

Config A 与 B 都不是直接用 RGB decoder 预测一张最终图片。两者从同一个
MetaCanvas memory 分出两个可学习分支：

```text
I_in + instruction -> VLM + MetaCanvas
                         |-> dense spatial head -> s(x,y)
                         `-> renderer MetaQuery -> 4D Gaussian GLUT 参数 theta

I_pred = render4d(I_in, s, theta)
```

因此，“空间场”和“颜色变换参数”在输出头层面是分开预测的，但它们不是分开训练：
最终 `I_pred` 经可导 renderer 与 `I_tar` 计算重建损失，梯度同时回传到两个分支和
共享 MetaCanvas；`.cgt` 另外只对空间场施加 BCE。这里的 `s` 虽然按 mask 监督，
在 Config A/B 的 4D renderer 中实际充当第四维条件坐标，不是简单的 alpha 混合。

## 完整 select 指标

| config | best step | score ↓ | DeltaE00 p50/p90 ↓ | PSNR in/out ↑ | AUC ↑ | soft-IoU ↑ | Delta shuffle ↑ | var-ratio ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 5000 | **37.287** | 3.560 / **7.086** | **22.588** / 47.702 | **0.947** | **0.605** | **1.901 dB** | **0.317** |
| B | 5500 | 37.967 | **3.557** / 7.216 | 22.528 / **48.212** | 0.938 | 0.582 | 1.798 dB | 0.275 |

A 的综合选择分数更好，但两者 `Delta shuffle` 都低于预注册的 3 dB 条件因果门槛，
所以当前只能说 A 的 instruction 响应更强，不能据此宣称已经充分听懂指令颜色。

## 同图 A/B 对照

![20 张输入、目标、A/B 最终渲染总览](viz/config_ab_compare/compare20_overview.png)

[详细第 1 页](viz/config_ab_compare/compare20_p01.png) ·
[第 2 页](viz/config_ab_compare/compare20_p02.png) ·
[第 3 页](viz/config_ab_compare/compare20_p03.png) ·
[第 4 页](viz/config_ab_compare/compare20_p04.png)。

详细联图列顺序为：`输入 | 目标 | A 最终渲染 | B 最终渲染 | GT 区域 | A 预测区域 | B 预测区域`。

## Config A feature

![Config A MetaCanvas 最终 feature](viz/config_a/feature_metacanvas.png)

![Config A renderer MetaQuery 最终 feature](viz/config_a/feature_metaquery.png)

[A 的 20 张独立总览](viz/config_a/gallery20_overview.png)；原始 feature 保存在
`viz/config_a/features20.npz`。

## Config B feature

![Config B MetaCanvas 最终 feature](viz/config_b/feature_metacanvas.png)

![Config B renderer MetaQuery 最终 feature](viz/config_b/feature_metaquery.png)

[B 的 20 张独立总览](viz/config_b/gallery20_overview.png)；原始 feature 保存在
`viz/config_b/features20.npz`。
