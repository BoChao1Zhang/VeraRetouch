# EPR-072-LOCAL 记录

## 双臂已启动

用户最终要求覆盖初版建议：PIX纯像素 vs NCE像素+InfoNCE；style30k按tar组抽15k，MMArt全部、local链全部入池；其他优化设置保持Stage1。
启动入口：`veraretouch_sprf/readout/epr072_local_run.sh`；trainer：`epr072_local_train.py`；I/O：`epr072_local_io.py`。
q任务1158/1159，对应GPU0/1。smoke每臂2个update已经通过，随后脚本自动从同一原始R@800重新开始full，smoke权重不作初始化。

| smoke | PIX | NCE |
|---|---:|---:|
| global pixel MAE | 0.06290555745 | 0.06290555745 |
| local pixel MAE | 0.02814047085 | 0.02813688247 |
| global InfoNCE | 0 | 6.938610435 |
| local InfoNCE | 0 | 7.780763745 |
| GPU峰值GiB | 14.1427 | 14.1427 |
| RSS GiB | 5.3696 | 5.3794 |

global keys_sha两臂均为`e1f1dd538a2949e396469294e3bd63f4d48c9d326e8e55b47a9544d503cb2953`；local keys_sha均为`b7bf7e5988976c5f010be358c314e7334c09327f89e34061a97c34dba18b2506`。
首个global前尚未更新，pixel严格一致；经过不同损失更新后local像素损失可不同，这是正常的两臂差异。
I/O预算、mask/Subject约定、6250步的local访问数量边界见PROPOSAL的实际运行配置。

## 初始记录（历史）

- 用户要求：local续训时穿插global/style批次防止退化；主表先更新R final@6250。
- EPR-072_mmart-ppr10k为既有数据任务，本方案独立保存，任务标签EPR072LOCAL。
- 当前beta来源核查：旧chain训练记录自参考图；`mixed_train.py`使用`alpha=beta/s`；`artedit_eval.execute_six`默认input支持，可通过callable/current模式重算。当前单段EPR-071全部使用执行权重1。
- 新规格：current-state光度支持、独立Subject支持、新支持下重新求码；global/style和local在optimizer-update层面1:1交替。
- 主表来源：`/mnt/nfs-ro/bc/data/runs/epr071_mmart_20260918/train/R/run/evaluations/artedit_full_final.json`，400行。
- full400原始均值：L1×100=8.409841492041474；L2×1000=14.911429624842514；PSNR=20.421280477917257；ΔE00=9.87889014005661。
- 表格待测项：SSIM、SC/PQ/O、ArtiMuse/Q-Align/DeQA。禁止与旧step1800分数拼成一行。
- 状态：任务卡已写，训练未启动；实际六段入口/current支持重解缓存仍需按PROPOSAL实现与验收。
