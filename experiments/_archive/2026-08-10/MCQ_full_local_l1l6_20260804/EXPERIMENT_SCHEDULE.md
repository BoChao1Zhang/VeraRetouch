# MCQ-L 全量实验排期

## 统一协议

所有 arm 使用同一个 durable v2 indexed-tar manifest、同一个 source-disjoint
`fit/select/val/test`、同一个 seed 和样本顺序。结构比较阶段统一训练 6000 step、
batch 8、每样本 1024 个分层像素；每个 arm 都完整训练，不做短跑、子集筛选或
中途淘汰。每 500 step 的 384 条 select-core 只用于 checkpoint 选择和健康检查，
arm 的最终比较一律使用完整 4225 条 select。

硬停止仅限 OOM、非有限值、manifest digest 不一致或评估崩溃。普通 AUC、IoU
或中途 `delta_shuffle` 不触发停训。

## 波次

| 波次 | GPU 0 | GPU 1 | 唯一改变因素 |
|---|---|---|---|
| 当前 anchor | MCQ-STD，lr 2e-4 | MCQ-STD，lr 1e-4 | 优化尺度 |
| renderer | 3D Gaussian + alpha | 17^3 x 5 4D LUT | renderer |
| condition ablation | image-only | instruction-only | 条件输入 |
| instruction form | fixed wrong instruction | `instruction_short` | instruction 形态 |
| MetaCanvas input | instruction bilinear | 8 MetaQuery global | query 读入/空间载体 |
| MetaCanvas output | rank-14 field | pooled parameter MLP | WHERE/WHAT 读出 |
| final | 结构候选第 1 名 | 结构候选第 2 名 | 完整 S-train 重训 |

anchor 两个学习率以完整 select 的 selection score 决定后，后续 arm 全部固定使用
胜出的学习率。image-only、instruction-only 和 fixed wrong instruction 是因果控制，
即使像素指标高也不进入最终候选；其余结构加 anchor 按完整 select 排名前两名。

## Instruction 到颜色的判据

主结论不依赖 preset 分类准确率。报告至少包含 `delta_shuffle`、mask 内/边界/外
PSNR、DeltaE00 p50/p90、outside leakage、颜色变化方差比以及按 L1-L6 的宏平均。
完整输入只有同时优于 image-only 与 fixed wrong instruction，并在固定图像交换
instruction 时改变 WHAT，才能称为 instruction-conditioned color；普通 mask AUC 高
但同图反事实不变，只能解释为图像主体/位置先验。

现有 batch 内循环 shuffle 不是严格同图反事实。训练矩阵完成后，固定 source_id
组装 paired evaluation，补充同图 instruction swap、cross-source shuffle、方向相反
instruction 和 synonym 控制；这些是对完整 checkpoint 的评估，不替代任何 full run。

## 终局纪律

最终两个结构用完整 S-train（fit + select）各重训 6000 step，S-val 选 checkpoint，
S-test 只在训练结束后评一次。中文报告展示这两个 checkpoint 的固定 20 样本联图，
并分别展示最终 MetaCanvas memory、renderer MetaQuery、instruction feature；rank-14
若入选，还展示 basis14 与 w14。解析 ROW w14 需要离线标签，本矩阵的 rank-14 arm
明确是端到端空间因子化，不混写为解析 ROW 基底。

SA-LUT、HDRNet/bilateral grid 和 L6 双空间轴目前没有与 MCQ endpoint 对齐且通过
CI 的实现，因此不进入本轮可执行主榜。它们需要先补实现与 bake/identity CI，再按
同一 full-run 协议追加，不能用设计名替代实验结果。
