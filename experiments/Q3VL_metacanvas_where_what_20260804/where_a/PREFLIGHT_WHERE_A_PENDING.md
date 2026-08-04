# Where-A · 待执行清单（GPU / 重 IO）

生成 2026-08-05，**审阅后修订**（`docs/reviews/REVIEW-impl-WhereA.md` 判 6 个 BLOCKER，已全部修复）。
**以下每一项都未执行。** 阻塞原因：两张 H100 被 Base SFT 占用（rank PID 3395226 / 3395227），
全量 mask 作业与训练读写同一套 NFS build 树与本地盘。

环境前置（每个 shell 都要）：

```bash
export LD_LIBRARY_PATH=/home/bc/miniconda3/envs/llm_factory/lib
PY=/home/bc/envs/q3vl_sft/bin/python           # 与 Base SFT 同一环境
cd /home/bc/VeraRetouch
```

> 没有这一行，`import sqlite3` 直接 `CXXABI_1.3.15 not found`，而 mask 定位要读 build 的
> `catalog.sqlite3` —— 整条 Where-A 数据路径起不来。

执行顺序不能换：**S0 → S1 → S2 → S2b → S4 → S5**。全部入口在
`q3vl/where/scripts/run_where_a.sh`（内置 D-20 四步：`rm -f` 日志 → `ps -p $PID` 判活 →
`tail` 验实质输出 → 写 `job.marker`；**禁用 `pgrep`**）。

---

## S0 · 三项裁定已落地（无需算力，已写进 `config.py`）

| 决策 | 裁定 | 代码位置 |
|---|---|---|
| **D1** | `low` **进校准训练**（local train 75,544 而非 42,752）；**headline 指标只用 `normal`**，`low` 单独分层。放宽只对 Where-A basis 校准生效，不得流进 Where-B / What 的评测 GT | `EXCLUDE_WINNER_CONFIDENCE_LOW = False`、`HEADLINE_WINNER_CONFIDENCE = ("normal",)` |
| **D2** | 内外层目标统一为 `soft_iou_minmax`（红线属旧战役语境；且这恢复了 D3 的包络定理论证） | `FIT_OBJECTIVE = CALIB_OBJECTIVE = "soft_iou_minmax"` |
| **D3** | 维持"每 batch 用当前 B 现拟 latent"，理由改为**包络定理现在成立**（内外层同一个 `f`） | `calibrate.py` 模块 docstring |

---

## S1 · 全量 mask 视图打包（无 GPU，重 IO；**必须等 Base SFT 结束**）

```bash
bash q3vl/where/scripts/run_where_a.sh maskviews
```

产物：`/mnt/nfs/bc/data/datasets/where_a-20260805/maskviews/<split>/`（indexed tar shards，原子发布）；
报告在 `experiments/.../where_a/maskviews/<split>.report.json` + `*.rejections.jsonl` + `*.mask_stats.jsonl`。

**这些 shard 现在有消费方**（审阅 N-3）：`MaskViewStore` 被 `WhereADataSource(maskview_root=...)`
读取，`run_calibration.py` / `make_oracle_latents.py` 默认走它。四个臂 + latent 作业因此只解码一次
`.cgt.png`，而不是每臂各解一遍。

验收（脚本内自动跑 `verify_published(n_random=64)`，退出码非 0 即失败）：

| 判据 | 门槛 | 450 样本抽样实测 |
|---|---|---|
| 定位成功率 | ≥ 99.5% | 100.0%（V_where 200 / train 250） |
| 读取 + sha256 | 100% | 100.0% |
| 宽高比一致率 | ≥ 99.5% | 100.0%（相对误差中位 2.5e-5 / 1.5e-4，最大 3.9e-4） |
| 退化 mask | 报告即可 | 0.0% |
| shard 随机读校验 | 0 失败 | 打包器单测已验 |
| `sample_count` | = eligible 数，与 rejection 文件对得上账 | — |

D1 已裁定，样本量为 **local train 75,544**（含 low）。宽高比不符的样本在**两条链路**上都被丢弃
（审阅 N-4：以前打包作业丢、在线 pipeline 只记 diag，population 会对不上）。

⚠ 目标目录已存在时脚本硬报错（发布是原子的）。重跑必须先**挪走**旧目录，**不要先删**。

---

## S2 · 协议 §14 项 4/5/6 的 GPU 级 preflight（1 GPU，约 5–10 分钟）

```bash
bash q3vl/where/scripts/run_where_a.sh preflight
```

产物 `experiments/.../where_a/preflight_where_a.json`；任一项 `fail` → 退出码 1。
**驱动器现在 fail-closed**（审阅 N-5b）：即使某个检查抛异常，JSON 也一定落盘，并多出一行
`WA-P0-preflight-driver = fail`；"没有报告"不再等价于"没跑过"。

| 检查 id | 协议项 | 断言 | CPU 预跑（4 个真实 V_where 样本，base 权重，float32） |
|---|---|---|---|
| `WA-P4a-fpre-geometry` | 14.4 | 形状 `(H/16·W/16, 1024)`；网格宽高比 == 图像宽高比；正确 unshuffle 的空间相干性 > 朴素 reshape | PASS，相干性 **0.01733 vs 0.00745**，grid 32×48 ← 512×768 |
| `WA-P4d-position-encoding` | 14.4「位置编码」 | **直接**核对：独立复算 `pos_embed.weight` 的双线性插值，与 `fast_pos_embed_interpolate` + unshuffle 逐元素比 | PASS，最大相对误差 **1.85e-7**（审阅 N-6：原来只有相干性这个间接代理） |
| `WA-P4c-upsample-order` | 14.4 | 多通道抛 `ChannelOrderError`；**batch 轴折叠也抛**；常数场被保持 | PASS |
| `WA-P4e-highres-path` | 14.4 + 4.2 | **真实**高分辨率通路：一次 guided upsample → 交付分辨率 soft-IoU、low→hi 落差、s 域越界比例 | PASS（详见下方 B-4 实测） |
| `WA-P4b-fpre-sft-invariance` | 本仓库推论 | SFT checkpoint 的 `patch_embed`/`pos_embed`/24 blocks 与 base **逐位相同** | **SKIP —— checkpoint 还不存在；S2 时必须变成 PASS** |
| `WA-P5-basis-conditioning` | 14.5 | 残差化后最大相关 < 1e-4；phi Gram cond < 1e10；oracle 拟合成功率 ≥ 90% | PASS：相关 **7.6e-7**；design Gram cond 中位 27.9；phi Gram cond 中位 **8.6e4**；成功率 100%/100% |
| `WA-P6a-readout-bounds` | 14.6 | raw=±1e6 下 `h>0`、`k∈[1,40]`、`σ∈[0.025,0.30]`、`o,c∈(0,1)`；μ 网格固定对称；`m(z)∈[0,1]` | PASS |
| `WA-P6b-latent-invariants` | 14.6 | 每个 latent：符号规则、`‖w_dir‖=1`、`α>0`、readout 在界内 | PASS：8/8 规范，范数误差 **1.7e-13**，min α **0.187** |

口径说明（审阅 N-7 / N-8，已写进 JSON 字段）：
- `projector_stage = "seeded_orthogonal_start_not_calibrated"` —— WA-P5 测的是 **BA-0 起点**，
  是开跑前的门，不是校准后的数；**校准结束后必须复测一次**。
- `condition_number_basis = "gram_matrix (= matrix cond ^ 2)"` —— 报的是 Gram 的条件数，
  1e10 阈值相当于 `phi` 本身的 1e5。

### B-4 实测：高分辨率通路的两个静默失效已复现并封堵

审阅指出的两条，在**真实数据**上复现如下（`WA-P4e` detail）：

| 项 | band | cband12 |
|---|---|---|
| 上采样后 s 的**原始**取值域（clamp 前） | **[−4.577, +2.569]** | [−0.455, +2.306] |
| 越出 `S_DOMAIN = (−3, 3)` 的像素比例 | 最大 **0.542%**，均值 0.140% | 0 |
| low → hi 的 soft-IoU 落差（中位） | **0.0177** | **0.0738** |

处置（CLAUDE.md《s 缓存消费契约》：消费方必须声明期望域并断言生数据住在里面）：
1. **生产方声明**：`S_DOMAIN = (−3, 3)`，因为 `s_low = 3·tanh(q/3)` 结构上就落在开区间内；
2. **消费方断言**：`guided_upsample(..., return_domain_report=True)` 永远返回 **clamp 前**的
   `raw_min/raw_max/frac_out_of_domain`，`evaluate_latent` / `Calibrator.evaluate` /
   preflight 逐层上报，越界不可能静默；
3. **处理**：`UpsampleConfig.clamp_domain=True`，把越界值 clamp 回 `[−3, 3]`。选 clamp 而不是
   `3·tanh(s/3)` 重压：后者会把域内的值也整体压缩（s=2.9 → 2.24），而越界纯粹是 guided filter
   的仿射外插产物，clamp 恰好是"回到生产方自己的不变量"。**是整臂常量，不是逐图归一化**（红线）。
4. **CBand12 分母塌陷**：`Σ c_i g_i/(Σ g_i + eps)` 在 σ 取下界 0.025 时，两中心正中与端点外
   `Σ g_i ≈ 6e-27 < eps=1e-9`，`m` 恒等于 **0**。已改为 `logsumexp`（同一公式的 `eps→0` 极限，
   用 `softmax(log g_i)` 稳定求值）：良态区与原式差 <1e-9（单测断言），端点外返回最近基元的 `c_i`
   而不是塌成 0。原式保留为 `normalization="eps"` 供对照，单测**同时钉住旧式的塌陷数字**，
   防止有人改回去而不被发现。

### S2 的两项硬性输出（不是"建议"）

**(a) D5 定档 —— `sweep-upsample`（审阅 B-4 要求 3 / N-16）**

```bash
bash q3vl/where/scripts/run_where_a.sh sweep-upsample
```

`radius_low ∈ {1,2,4}` × `eps ∈ {1e-4,1e-3,1e-2}`，在 24 个真实 V_where 样本上报交付分辨率
soft-IoU、low→hi 落差、越域比例。定档后写回 `config.py` 并把 `GUIDED_PARAMS_PROVISIONAL` 置 `False`。
**当前 `radius_low=2, eps=1e-3` 是临时值**：它们来自 E2 的**全分辨率逐通道** `r=32` 用法
（`experiments/E2_basis_fit_20260803/prep_data.py:149`），正是 §4.2 现在禁止的顺序，先例不可迁移。

**(b) 内层 L-BFGS 吞吐 —— 已内建进 `WA-P5` 的 `throughput` 字段（审阅 N-15）**

CPU 实测（1 核，float64，`n_random=3, max_iter=80`，1536 点）：**1.697 s / 拟合**，
外推 **40.3 CPU-小时/臂**（42,752 × 2 readout）或 **71.2 CPU-小时/臂**（75,544，即 D1 裁定后的口径）。
**GPU 数字必须在 S2 现测**（同一字段会自动填 `device: cuda`）：它是 §11 排期唯一还没有实测支撑的量，
也直接决定 D3 是否需要改用离线 latent 表。

---

## S4 · 四个校准臂（每臂 1 GPU；本机可并行 2 臂）

```bash
bash q3vl/where/scripts/run_where_a.sh calibrate BA-0-Fixed
bash q3vl/where/scripts/run_where_a.sh calibrate BA-1-Band
bash q3vl/where/scripts/run_where_a.sh calibrate BA-2-CBand12
bash q3vl/where/scripts/run_where_a.sh calibrate BA-3-Joint     # 预注册主方案
```

审阅后新增的三项运行期保障：

- **调度基于 eligibility 过滤后的实际样本数**（B-3）。`run_calibration.py` 先跑一遍
  `count_eligible("train")`（只读 record，无 GPU，结果缓存进 `eligible_count.json`），
  `total_steps = ceil(n_eligible / batch)`，跑完**断言 `step_count == total_steps`**，
  不等就 `SystemExit` 并要求重新计数。旧代码用未过滤的 75,544 建表，在 D1 旧默认下
  warmup 会变成实际步数的 5.3%（协议要 3%）、cosine 只走到 55% → 结尾 LR 停在峰值的 41.8%，
  四个臂在**互不相同且从未退火完成**的调度下停机，归因比较直接失效。
  `schedule.json` 落盘 `planned/actual/warmup/final_lr`。
- **rejection report 逐样本落盘**（B-2）：`fit_rejections.jsonl`，每个非 `ok` 拟合一行
  （`sample_id / readout / status / reject_reason / flags / loss / build / winner_confidence`），
  外加每 200 步一行健康样本作基线；`projector_final.pt` 里带整轮 `rejection_summary()`。
- **被拒拟合不反传、不进天花板**（B-1）：`all_starts_failed` 现在返回 `latent=None`
  （**删掉了字面零向量**），`fit.usable` 是唯一闸门；`evaluate` 的聚合只在 `ok` 子集上算，
  被拒集合单独出 `n_rejected` + `reject_reasons`。

必须在报告里出现的分层：
- **`image.upscaled`**：抽样实测 V_where 本地 **15.0%**、train 本地 **10.4%** 被上采样到短边 512。
  这些样本的 GT 边缘是插值出来的，边缘类指标**虚高**，两层必须分开报。
- **`winner_confidence`**：`normal`（headline）与 `low` 分开报——D1 放开的是训练，不是 headline。
- build（l1–l6）：`run_calibration.py` 已产出 `strata.build`；每层都给 low/hi 两个分辨率档。

`BA-3-Joint` 结束后按 §4.4 最后一段**固定 B** 再用 `n_random=6, max_iter=120` 全档重拟
Band/CBand 的最终 `w*, ρ*, s*, r*(z)` —— 已内建在 evaluate 阶段（`attach_hi=True`，天花板在
**交付分辨率**上测），无需额外命令。

---

## S5 · train split 的 oracle latent（1 GPU）—— **Where-B 的开工前置**

```bash
bash q3vl/where/scripts/run_where_a.sh oracle-latents
```

协议 §5.5 的 `L_s = Huber(s_pred/3, s*/3)`、`L_curve = mean_z|R(z;ρ_pred) − r*(z)|`、
`L_dir = 1 − cos(w_dir_pred, w_dir*)` 都是**训练时**监督（前 30% 步权重 1.00），
需要 **train split** 上的逐图 `s*, r*(z), w_dir*`。校准跑只在 `V_where` 上产 latent，
Where-B 一开工就会卡住，而补做等于把 Where-A 最贵的一步（全量内层 L-BFGS）推到发现最晚的时候
（审阅 B-5）。

- 位置：**S4 冻结 `BA-3-Joint` 的 B 之后、Where-B 训练之前**；`B` 从发布的 `B.npy` 读入并冻结，
  basis 文件不存在时脚本直接拒绝运行（不允许对着未校准的 basis 拟 latent）。
- 产物：`/mnt/nfs/bc/data/datasets/where_a-20260805/oracle/BA-3-Joint/train/`（§2.3 indexed shards），
  每样本 `<sample_id>.oracle.json` 含两个 readout 的规范化 `w*, ρ*`、fit report、low/hi 指标，
  以及 `r*(z)` 在固定 121 点 z 网格上的采样（Where-B 的 `L_curve` 直接做向量差，不必再引 readout 代码）。
- 墙钟：**待 S2 的 GPU 吞吐实测**。CPU 外推 71.2 小时（75,544 × 2 readout × 1.697 s）——
  这个数如果在 GPU 上没有大幅下降，就必须回头重裁 D3。

---

## S6 · 收尾（进 REPORT.md 之前）

- [ ] 四个臂的 `eval_V_where.json` 汇总成预注册判据 vs 实测数字并排表，**low / hi 两个分辨率档并列**
- [ ] `viz/success_*` 与 `viz/failure_*`：输入 / 预测 mask / GT / s 场 并排（**失败案例必须有**；
      `fit_rejections.jsonl` 现在直接给出失败样本 id，不必再去猜）
- [ ] `config/`：配置快照 + seed + git commit + 环境（`preflight_where_a.json` 的 `env` 段可直接用）
- [ ] `metrics.json`：机器可读汇总
- [ ] REPORT.md「设置」节写清：D1/D2/D3 裁定、D5 定档值、s 域 clamp 的显式声明与越界比例、
      CBand12 归一化方式，以及 §2.3 对 basis 元数据走裸文件的**显式豁免**（审阅 N-14）
- [ ] 任何"Where-A 天花板 = 0.9x soft-IoU"的说法必须标注是 low 还是 hi 分辨率档
      （审阅 §十：B-4 关闭前不能只报低分辨率数）

---

## 附：已经跑完、不需要重跑的

| 项 | 命令 | 结果 |
|---|---|---|
| CPU 单测 | `$PY -m pytest q3vl/where/tests -q` | **147 passed / 44 s** |
| 数据源抽样验证（450 样本） | `$PY -m q3vl.where.scripts.validate_masks --split V_where --limit 200 --pixel-check 80`；`--split train --limit 250 --pixel-check 60 --include-low` | 定位/读取/宽高比 100%；`corr(|I_tar−I_in|, mask)` 中位 0.787 / 0.783 |
| 数据无关 preflight | `$PY -m q3vl.where.preflight --skip-model` | 2 项 PASS，其余 SKIP |
| CPU 真实数据 preflight | `$PY -m q3vl.where.preflight --device cpu --dtype float32 --limit 4` | **7 项 PASS**，1 项 SKIP（checkpoint 未生成）；归档于 `preflight_where_a_cpu_dryrun.json` |
