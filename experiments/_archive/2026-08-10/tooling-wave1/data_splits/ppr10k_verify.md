# PPR10K 无污染复核报告

- 完成 build：prod-g1-global25k-20260731, prod-g2-global25k-20260731, prod-g3-global25k-20260801, prod-l1-local17k-20260731, prod-l2-local17k-20260731, prod-l3-local17k-20260731, prod-l4-local17k-20260801
- `ppr10k/source` 不同源：**4311**，组占用 **17706**
- 文件索引范围：**1–8871**（官方 train = 前 8,875 个文件）
- 索引 ≥8875（官方 val 段）命中：**0**
- 结论：**干净，无官方 val 污染**

## 附：mmart_ppr10k 池（原 §4-A 结论未覆盖，提请裁决）

- 不同源 3674，组占用 15782，group id 范围 1–1680
- group id ≥1356（论文口径 val 组段，advisory）：**326** 源
- MMArt-PPR10k is built on PPR10K raw photos; official val = last 2,286 files (files ordered by group), so high group ids overlap official val. Boundary 1356 is paper-derived (1,356 train groups), not re-verified verbatim - advisory.
