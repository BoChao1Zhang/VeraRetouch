# EPR-072-LOCAL 记录

- 用户要求：local续训时穿插global/style批次防止退化；主表先更新R final@6250。
- EPR-072_mmart-ppr10k为既有数据任务，本方案独立保存，任务标签EPR072LOCAL。
- 当前beta来源核查：旧chain训练记录自参考图；`mixed_train.py`使用`alpha=beta/s`；`artedit_eval.execute_six`默认input支持，可通过callable/current模式重算。当前单段EPR-071全部使用执行权重1。
- 新规格：current-state光度支持、独立Subject支持、新支持下重新求码；global/style和local在optimizer-update层面1:1交替。
- 主表来源：`/mnt/nfs-ro/bc/data/runs/epr071_mmart_20260918/train/R/run/evaluations/artedit_full_final.json`，400行。
- full400原始均值：L1×100=8.409841492041474；L2×1000=14.911429624842514；PSNR=20.421280477917257；ΔE00=9.87889014005661。
- 表格待测项：SSIM、SC/PQ/O、ArtiMuse/Q-Align/DeQA。禁止与旧step1800分数拼成一行。
- 状态：任务卡已写，训练未启动；实际六段入口/current支持重解缓存仍需按PROPOSAL实现与验收。
