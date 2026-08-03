# 行动项 G：弃权组渲染产物落盘可用性核查

| build | 组数 | journal 候选 | 落盘 after | 弃权组候选 | 弃权组可用对 | null 组候选 | null 组可用对 | I_in 可用源 |
|---|---|---|---|---|---|---|---|---|
| prod-l1-local17k-20260731 | 300 | 2400 | 136000 | 704 | 680 | 608 | 448 | 270/300 |

- **可用渲染对总数（D-RENDER 规模数）：2,160**（journal 候选 2,400）
- 其中弃权组（abstain）可用对：**680**；按置信度：{"normal": 560, "low": 472, "abstain": 680, "unannotated": 448}
- 口径：usable = after .jpg landed in groups dataset AND source retrievable (img bank / local path)
