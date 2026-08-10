### 表 1 · 基线锚点与本臂结果并排

| 读出 | AUC | AUC_target | 说明 |
|---|---|---|---|
| **RO-9 原样（G1 落盘，单样本生成）** | 0.6462 | 0.5038 | 已发表 0.648 / 0.504，同批源复算 |
| **RO-9 原样（本臂复刻，批量生成）** | 0.6607 | 0.5083 | 瀑布起点 |
| RO-9 最好单层（head-mean，样本内） | 0.8393 | 0.5300 | L22 / L20 |
| **RO-9b 修复后（主判据 AUC_target 选优）** | 0.7278 | 0.5719 | pre/top3/wsum/a1/z=True |
| **RO-9b 修复后（AUC 选优，定位质量上限）** | 0.9349 | 0.3846 | pre/mean/linfit/a1/z=True |
| RO-1 ClearCLIP（16×16） | 0.941 | 0.736 | 零训练 CLIP |
| `X_deictic` 零指令固定短语 | 0.907 | 0.523 | CLIP，对全体图同一条短语 |
| vanilla CLIP（未改造） | 0.214 | — | RO-1 对照 |
| 亮度基线 | 0.4650 | — | |

### 表 2a · 四处修复的增量瀑布 — 主判据（AUC_target）选优

| 段 | 选中配置 | AUC | CI95 | ΔAUC | 配对 p | AUC_target | ΔAUC_t | 配对 p | Δ_shuffle | Δ_luma | Δ_const |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S0 RO-9 原样 | pre/mean/canon/a1/z=None | 0.6607 | [0.629, 0.681] | — | — | 0.5083 | — | — | -0.0004 (p=0.442) | 0.1957 | 0.1607 |
| S1_layer | pre/mean/wsum/a1/z=True | 0.7241 | [0.707, 0.749] | **0.0633** | 0.000 | 0.5246 | **0.0163** | 0.000 | 0.0052 (p=0.190) | 0.2591 | 0.2241 |
| S2_head | pre/top3/wsum/a1/z=True | 0.7278 | [0.695, 0.764] | **0.0038** | 0.718 | 0.5719 | **0.0474** | 0.000 | 0.0220 (p=0.001) | 0.2629 | 0.2278 |
| S3_selfself | pre/top3/wsum/a1/z=True | 0.7278 | [0.695, 0.764] | **0.0000** | nan | 0.5719 | **0.0000** | nan | 0.0220 (p=0.001) | 0.2629 | 0.2278 |
| S4_postproc | pre/top3/wsum/a1/z=True | 0.7278 | [0.695, 0.764] | **0.0000** | nan | 0.5719 | **0.0000** | nan | 0.0220 (p=0.001) | 0.2629 | 0.2278 |

### 表 2b · 四处修复的增量瀑布 — 定位质量（AUC）选优

| 段 | 选中配置 | AUC | CI95 | ΔAUC | 配对 p | AUC_target | ΔAUC_t | 配对 p | Δ_shuffle | Δ_luma | Δ_const |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S0 RO-9 原样 | pre/mean/canon/a1/z=None | 0.6607 | [0.629, 0.681] | — | — | 0.5083 | — | — | -0.0004 (p=0.442) | 0.1957 | 0.1607 |
| S1_layer | pre/mean/linfit/a1/z=True | 0.9349 | [0.929, 0.943] | **0.2742** | 0.000 | 0.3846 | **-0.1237** | 0.215 | -0.0006 (p=0.365) | 0.4699 | 0.4349 |
| S2_head | pre/mean/linfit/a1/z=True | 0.9349 | [0.929, 0.943] | **0.0000** | nan | 0.3846 | **0.0000** | nan | -0.0006 (p=0.365) | 0.4699 | 0.4349 |
| S3_selfself | pre/mean/linfit/a1/z=True | 0.9349 | [0.929, 0.943] | **0.0000** | nan | 0.3846 | **0.0000** | nan | -0.0006 (p=0.365) | 0.4699 | 0.4349 |
| S4_postproc | pre/mean/linfit/a1/z=True | 0.9349 | [0.929, 0.943] | **0.0000** | nan | 0.3846 | **0.0000** | nan | -0.0006 (p=0.365) | 0.4699 | 0.4349 |

### 表 3 · 每段候选全表（out-of-fold）

| 段 | 候选配置 | AUC | AUC_target |
|---|---|---|---|
| S1_layer | pre/mean/canon/a1/z=False | 0.6607 | 0.5083 |
| S1_layer | pre/mean/best1/a1/z=False | 0.7670 | 0.5230 |
| S1_layer | pre/mean/canon/a1/z=True | 0.6727 | 0.5052 |
| S1_layer | pre/mean/best1/a1/z=True | 0.7670 | 0.5230 |
| S1_layer | pre/mean/band/a1/z=True | 0.7658 | 0.5093 |
| S1_layer | pre/mean/wsum/a1/z=True | 0.7241 | 0.5246 |
| S1_layer | pre/mean/linfit/a1/z=True | 0.6528 | 0.5099 |
| S2_head | pre/mean/wsum/a1/z=True | 0.7241 | 0.5246 |
| S2_head | pre/zmean/wsum/a1/z=True | 0.7647 | 0.5248 |
| S2_head | pre/aucw/wsum/a1/z=True | 0.6835 | 0.5590 |
| S2_head | pre/top3/wsum/a1/z=True | 0.7278 | 0.5719 |
| S2_head | pre/top5/wsum/a1/z=True | 0.6828 | 0.5566 |
| S2_head | post/mean/wsum/a1/z=True | 0.6967 | 0.5071 |
| S2_head | post/zmean/wsum/a1/z=True | 0.7000 | 0.5116 |
| S3_selfself | pre/top3/wsum/a1/z=True | 0.7278 | 0.5719 |
| S3_selfself | kk/top3/wsum/a1/z=True | 0.4835 | 0.5280 |
| S3_selfself | qq/top3/wsum/a1/z=True | 0.7358 | 0.5589 |
| S4_postproc | pre/top3/wsum/none/z=True | 0.7270 | 0.5714 |
| S4_postproc | pre/top3/wsum/a1/z=True | 0.7278 | 0.5719 |
| S4_postproc | pre/top3/wsum/sink/z=True | 0.7025 | 0.5660 |
| S4_postproc | pre/top3/wsum/aff/z=True | 0.7446 | 0.5517 |
| S2_head | pre/mean/linfit/a1/z=True | 0.9349 | 0.3846 |
| S2_head | pre/zmean/linfit/a1/z=True | 0.9345 | 0.3885 |
| S2_head | pre/aucw/linfit/a1/z=True | 0.9192 | 0.4668 |
| S2_head | pre/top3/linfit/a1/z=True | 0.9208 | 0.4869 |
| S2_head | pre/top5/linfit/a1/z=True | 0.9265 | 0.4678 |
| S2_head | post/mean/linfit/a1/z=True | 0.8582 | 0.4890 |
| S2_head | post/zmean/linfit/a1/z=True | 0.8378 | 0.4724 |
| S3_selfself | pre/mean/linfit/a1/z=True | 0.9349 | 0.3846 |
| S3_selfself | kk/mean/linfit/a1/z=True | 0.8814 | 0.4867 |
| S3_selfself | qq/mean/linfit/a1/z=True | 0.8831 | 0.4279 |
| S4_postproc | pre/mean/linfit/none/z=True | 0.9324 | 0.3906 |
| S4_postproc | pre/mean/linfit/a1/z=True | 0.9349 | 0.3846 |
| S4_postproc | pre/mean/linfit/sink/z=True | 0.9175 | 0.3908 |
| S4_postproc | pre/mean/linfit/aff/z=True | 0.9256 | 0.4725 |

### 表 4 · 组 A · 视觉塔 FastViTHD self-self 手术（emb logit lens 读出）

| 臂 | attn / arch / blocks | AUC(目标词) | CI95 | AUC(`calculator`) | 词特异性 Δ | p | AUC(补集词) | AUC(错位词) | AUC_target | Δ vs vanilla | p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kk | kk / vanilla / last | 0.7451 | [0.704, 0.764] | 0.5709 | 0.1378 | 0.000 | 0.4130 | 0.6303 | 0.6615 | 0.0084 | 0.016 |
| naclip_std5 | naclip / vanilla / last | 0.7433 | [0.706, 0.768] | 0.5699 | 0.1362 | 0.000 | 0.4056 | 0.6238 | 0.6619 | 0.0103 | 0.039 |
| naclip_std2 | naclip / vanilla / last | 0.7391 | [0.712, 0.770] | 0.5763 | 0.1266 | 0.000 | 0.4194 | 0.6202 | 0.6616 | 0.0094 | 0.058 |
| csa_stage4 | csa / vanilla / stage4 | 0.7320 | [0.688, 0.758] | 0.6603 | 0.0438 | 0.000 | 0.3147 | 0.5993 | 0.7115 | 0.0008 | 0.260 |
| vanilla | vanilla / vanilla / last | 0.7297 | [0.700, 0.767] | 0.5767 | 0.1204 | 0.000 | 0.4224 | 0.6102 | 0.6576 | 0.0000 | nan |
| qq | qq / vanilla / last | 0.7215 | [0.691, 0.757] | 0.5543 | 0.1338 | 0.000 | 0.4121 | 0.6012 | 0.6468 | -0.0037 | 0.186 |
| clearclip_stage4 | clearclip / reduced / stage4 | 0.6189 | [0.578, 0.674] | 0.6392 | 0.0001 | 0.674 | 0.5913 | 0.6253 | 0.5081 | -0.0722 | 0.000 |
| csa | csa / vanilla / last | 0.6148 | [0.592, 0.642] | 0.5271 | 0.0676 | 0.000 | 0.5119 | 0.5267 | 0.5351 | -0.0834 | 0.000 |
| csa_reduced | csa / reduced / last | 0.5440 | [0.515, 0.584] | 0.4972 | 0.0363 | 0.000 | 0.5268 | 0.5686 | 0.5107 | -0.1597 | 0.000 |
| kk_reduced | kk / reduced / last | 0.4416 | [0.391, 0.477] | 0.4323 | 0.0036 | 0.017 | 0.3470 | 0.4486 | 0.5682 | -0.2552 | 0.000 |
| clearclip | clearclip / reduced / last | 0.4402 | [0.420, 0.502] | 0.4357 | 0.0049 | 0.026 | 0.3492 | 0.4507 | 0.5476 | -0.2793 | 0.000 |
| clearclip_stage34 | clearclip / reduced / stage34 | 0.4326 | [0.419, 0.448] | 0.4298 | -0.0005 | 0.632 | 0.5562 | 0.4392 | 0.4280 | -0.2910 | 0.000 |
| dropres | vanilla / reduced / last | 0.3904 | [0.354, 0.432] | 0.3947 | 0.0010 | 0.613 | 0.4837 | 0.4047 | 0.4467 | -0.3197 | 0.000 |

### 表 5 · 逐层曲线（head-mean，RO-9 口径）

| L | AUC(s_a,M) | AUC(s_b,M) | AUC_target |
|---|---|---|---|
| 0 | 0.5071 | 0.5383 | 0.4863 |
| 1 | 0.5492 | 0.5621 | 0.5088 |
| 2 | 0.7047 | 0.6893 | 0.4972 |
| 3 | 0.4873 | 0.4838 | 0.5022 |
| 4 | 0.5397 | 0.5527 | 0.4976 |
| 5 | 0.4946 | 0.5041 | 0.4958 |
| 6 | 0.7259 | 0.7164 | 0.5159 |
| 7 | 0.6775 | 0.6885 | 0.4948 |
| 8 ⬅ canonical | 0.6016 | 0.6160 | 0.5101 |
| 9 ⬅ canonical | 0.7198 | 0.7185 | 0.4976 |
| 10 ⬅ canonical | 0.5265 | 0.5190 | 0.5003 |
| 11 ⬅ canonical | 0.8162 | 0.8105 | 0.4939 |
| 12 ⬅ canonical | 0.6839 | 0.6759 | 0.4984 |
| 13 ⬅ canonical | 0.5821 | 0.5655 | 0.5078 |
| 14 ⬅ canonical | 0.6550 | 0.6242 | 0.5234 |
| 15 ⬅ canonical | 0.5699 | 0.5772 | 0.4931 |
| 16 | 0.5351 | 0.5134 | 0.5162 |
| 17 | 0.5578 | 0.5178 | 0.5238 |
| 18 | 0.6488 | 0.6063 | 0.5127 |
| 19 | 0.6571 | 0.6712 | 0.5086 |
| 20 | 0.7979 | 0.7799 | 0.5300 |
| 21 | 0.8055 | 0.7778 | 0.5106 |
| 22 | 0.8393 | 0.8199 | 0.5096 |
| 23 | 0.7351 | 0.7283 | 0.5045 |

### 表 6 · 桥接臂（视觉塔手术 + LM 读出，teacher-force 同一段 plan）

| 量 | 值 |
|---|---|
| auc | 0.6572 |
| auc_target_bg | 0.5411 |
| auc_shuffle | 0.6420 |

### 表 7 · D-0 是否空转

| 档 | AUC | AUC_target |
|---|---|---|
| L8–15 head-mean, **D-0 关** | 0.6571 | 0.5109 |
| L8–15 head-mean, **D-0 开** | 0.6607 | 0.5083 |
| Δ | 0.0036 | -0.0026 |

outlier 平均占比 = 0.0343
