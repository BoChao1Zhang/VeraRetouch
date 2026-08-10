# Basis-WHERE 全量 L1-L6 实验

## 目标

停止让 MetaCanvas 直接预测 16x16 空间 logits。MetaCanvas 只读取
`I_in + instruction` 并预测一组全局 basis 系数；空间场由解析/图像 basis 组合产生。

## 统一协议

- 数据：`metacanvas-local-l1l6-v2-20260804` durable indexed-tar manifest。
- 训练池：完整 `fit` 37,370 条；周期评估：固定 384 条 select-core。
- 每个 arm：6000 step、batch 8、每样本 1024 个分层像素。
- checkpoint/评估每 500 step；训练结束后在完整 4,225 条 select 上评估。
- 学习率固定为 Config A 胜出的 `2e-4`，LoRA 学习率 `2e-5`。
- 不做短跑筛选或中途淘汰；仅 OOM、非有限值、manifest/checkpoint 不一致或评估崩溃停止。

## 两个实验臂

| arm | basis | MetaCanvas 输出 |
|---|---|---|
| `basis_geo_range8_full` | `[1,x,y,P2(x),P2(y),xy,L,S]` | 8 个全局系数 |
| `basis_vlm14_full` | 上述 8D + frozen VeraRetouch vision token 的 6D 共享语义投影 | 14 个全局系数 |

两臂均使用 8 个无空间坐标的全局 MetaQuery。MetaCanvas 不输出逐位置值。

```text
q(p) = Phi(I,p) @ w(I,instruction)
s_latent(p) = 3 tanh(q(p)/3)
mask_logit(p) = 6 s_latent(p)
s_renderer(p) = (s_latent(p)+3)/6
```

mask BCE/AUC/soft-IoU 使用 `sigmoid(mask_logit)`；4D renderer 使用连续
`s_renderer`。二者不再共用同一张概率图。

## Smoke

两个 arm 均已完成真实 VeraRetouch 前向、2 step 反向、4 条 select 评估和原子
`best.pt/latest.pt` 写入。smoke 只验证实现，不参与模型选择。
