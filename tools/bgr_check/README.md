# tools/bgr_check — F5 生产 bgr 轴序 vs colour 四面体对拍

一次性 gate 实验工具（wave-1.5 F5，RD-G Stage-1 前置）。结论与数字见
`experiments/tooling-wave1/bgr_check/REPORT.md`：判定 (a)，`axis_order:"bgr"`
仅是 `grid[b][g][r]` 内存轴序标注，生产 after 图输出为正确 RGB 语义。

解释器一律 `/home/bc/miniconda3/bin/python3`（colour-science 0.4.7、numpy、PIL、matplotlib，与 tools/cube 相同）。

```bash
python3 selfcheck.py                 # 13 项：解析互验/恒等/灵敏度/复合端点/度量健全
python3 sample.py [--seed 20260803]  # 抽 100 对 -> manifest.jsonl + sampling_log.json
python3 run_check.py --workers 8     # 全量对拍 -> metrics.json
python3 run_check.py --viz <pair_id ...>   # 8 联对拍图 -> viz/
```

约定：I_in 从 img bank 按 source_path **全路径优先**匹配（unsplash 必须命中
unsplash_work，见 REPORT §六-3）；after/cgt 从 groups shards ranged-read；
生产数学复刻 = trilinear on `load_lut grid[b,g,r]`（rendering.apply_lut_cpu_oracle
逐式）；RD-G 路径 = colour read_LUT_IridasCube + 四面体。JPEG 底噪 =
save_candidate_jpeg 同参数往返。
