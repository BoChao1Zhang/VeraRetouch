# E1b NOTES：实施前核实 + 假设清单 + 待决策

## 实施前核实

1. 「3,522 生产 preset」与盘面对账：`experiments/tooling-wave1/cube/inventory/inventory_summary.json`
   expected_used=3522、distinct=3522、missing=0 ✅；used_presets.txt 3,522 行，逐条 `preset_slug` 映射到
   `/var/cache/veradata/dcube/npy33/<slug>.npy`，**零缺失**（脚本内 assert）。
2. npy canonical 布局：`tools/cube/parse.py` 文档头（(33,33,33,3) float32、index [r,g,b]、RGB 通道、domain [0,1]）。
   本实验只做逐格点重建，布局约定不影响结果（展平/还原一致即可）；viz 的 hald 预览走 `cubelib.apply_lut_grid_sample`（协议内工具）。
3. ΔE00 实现：不引外部事实，torch 移植的 sRGB 矩阵/白点常数**运行时从本地 colour-science 提取**，
   与 `cubelib.delta_e00`（= 项目 GT 路径）20 万随机对对拍 max|diff|=1.97e-13，门 5e-3 通过后才允许跑主实验（脚本内 assert）。
4. 无新增外部 URL / 超参引用。

## 假设与口径（当场核实/自行定标）

- 「重建 ΔE00 分位」口径 = 逐 LUT 在 33³ 全格点的 mean ΔE00，再跨 LUT 取 p50/p90/p99（对齐 E1「跨 LUT 报分位」）。
  备选口径（pooled 逐点分位）已同时算出存 metrics.json（结论不变：pooled p90@r=32 为 9.5，更不支持 32–64）。
- 评估色域 = 33³ 均匀格点（LUT 自身定义域），非自然图像色分布。若换 hald_eval 自然分布加权，绝对数会变、
  「无拐点」的形状结论不会变（幂律形状由谱决定）。
- 重建后 clip [0,1]（物理可实现）；float64 全程。
- SVD 变体：plain 为主（任务卡字面），centered 同跑（两者差 <0.05 ΔE00，metrics.json 有双份）。

## 待主 agent 决策

1. **condition 宽度语义**：E1b 回答的是「共享线性基重建整库」的秩需求（≈384 才到 E1 门）。若 4D 高斯升维的
   condition 只需判别 look（近邻/家族级），低秩（32–64）可能仍够——需要一个判别探针实验才能定，建议按 REPORT
   「建议下一步」加小实验；在此之前**保守默认不据 E1b 单方面改 N/condition 宽度**。
2. 是否按 taxonomy 桶分层重跑尾部（预计结论：极端 look 桶抬高 p99）；本轮未做。

## 复现

```
CUDA_VISIBLE_DEVICES=0 python3 experiments/E1b_svd_20260803/run_e1b.py [--smoke]
python3 experiments/E1b_svd_20260803/make_viz.py
```
17 s（全量）/ 4 s（冒烟）。无长任务。
