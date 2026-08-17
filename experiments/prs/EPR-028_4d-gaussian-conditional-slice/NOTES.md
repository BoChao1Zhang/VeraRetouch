# EPR-028 / EPR-029 · NOTES (2026-08-15, field_pred 补齐 + 冒烟失败修复)

只记假设、裁定与待决策。数字在产物里，不在这里下结论。

## 1. `field_pred` 的两个来源（主 agent 裁定，已执行）

`/home/bc/data/runs/whatb/predfield_stlang_v2seg/` 原本只有 408 个 `.npy`：where 分支
按注册裁定只预测 `render_mode == "local"`（`q3vl/whereb/scripts/run_amort_arm.py:573`），
V_what 的 489 个 `style` 行没有文件。两臂的 required 表都含 `field_pred`
（`q3vl/whatb/arms/g4d.py:ARM_EXTRA_CRITERIA` / `qdual.REQUIRED_QDUAL`），
`criteria.assert_criteria_ran` 因此拒绝出板。

裁定：**style 行的 `field_pred` 取全 1 场**。依据（写进产物 manifest 的 `authority` 字段）：

* 生成律 `dataset_build/src/construct/rendering.py:301-313`：`mask is None -> out = edited`，
  即 style 行 GT α ≡ 1；
* 两个消费方在 GT α 一侧本来就是这么写的：`run_g4d_arm.py:331-332`、
  `run_qdual_arm.py:314-317` 对 style 样本返回 `torch.ones(...)`；
* 冻结的形成式 `Î = (1−α)⊙I + α⊙f̂(I)`。

它**不是补零，也不是 where 臂的预测**。执行方式：

* 产物：`_src/style_ones_fields.py`（脚本与 `manifest.json` 里的 sha256 一起落盘），
  每行写 `(grid_h, grid_w)` 的 float32 全 1，2-D；grid 取自记录的
  `image.grid_h/grid_w`（`q3vl/train/imageproc.py:plan_geometry`），已核对与 where 臂
  跑的 `x.grid_h/x.grid_w` 在 408 个 local 行上 **408/408 一致**；
* 标注：`per_sample.jsonl` 每行带 `source`（`stlang_m_low` / `style_definition_ones`），
  `manifest.json` 增 `sources` / `n_by_source` / `v_what_coverage` / `files_digest_sha256`，
  并用 `composite_note` 明写「旧的顶层键只描述 stlang_m_low 那 408 行」；
* 板上可分辨：两个臂都新增 `field_pred__stlang_m_low` 与
  `field_pred__style_definition_ones` 两列（**预注册的 `field_pred` 仍是合并列，未动**），
  tag 从产物自己的 `per_sample.jsonl` 读，不在臂里按 `task_type` 二次推断。

## 2. G4D 冒烟 rc=2 的真因与本次改法（**待主 agent 追认**）

真因不是崩溃，是退化解守卫按设计杀进程：`quick_eval@step5` 的
`cross_std = 2.32e-05 <= 1e-4`（"one transform for every sample"），
`guards.assert_transform_not_degenerate` 抛 `SystemExit(2)`。

它在 10 步冒烟里**结构上不可能通过**：§3.5 把生成器每个头的末层权重零初始化，
step 0 恒等（proposition 2），于是 `cross_std ≡ 0` 是构造出来的，不是学出来的。
本次在 CPU 上用本臂自己的优化器/损失、16 条真实 V_what 条件量了三条轨迹：

* 条件与目标无关（LUT 随机配）：cross_std 在 4e-5–8e-5 之间摆，300 步仍反复穿越 1e-4；
* 条件与目标绑定（每条 z 固定自己的 LUT）：step 3 越过 1e-4，step 10 到 3.4e-4，step 50 到 1e-1；
* 真冒烟（每步 32 条新样本、总共 160 条）：step 5 = 2.3e-5。

改法（只动 `q3vl/whatb/scripts/run_g4d_arm.py`）：新增 `DEGENERACY_BINDING_MIN_STEPS
= FROZEN["steps_per_epoch"] = 2936` 与 `degeneracy_binding(args, total_steps)`。
判据本身**一个字没改**（三条阈值、调用点、witness 记录全部照旧），改的只是
"verdict 是否杀进程"：

* `total_steps >= 2936` ⇒ binding（全量档：首次 quick eval 就在 2936 步，与判据当初
  写死的视野一致）；
* `total_steps < 2936` 且**没有** `--smoke` ⇒ 仍然 binding（这种 run 能出 published 板）；
* `total_steps < 2936` 且 `--smoke` ⇒ non-binding：三个数照算、照打、照写
  `degeneracy_first_quick_eval.json` 与 `board["degeneracy"]`，`record_degeneracy_check`
  照样落 witness（所以 `publish.assert_publishable` 的「守卫跑过没有」仍然测得到），
  只是不退出。`--smoke` 板永远 `published=False`。

同时补了一条 `publish.assert_publishable` 没有的断言：**published 板上守卫必须是
`ok=True`**（`assert_publishable` 只检查「跑过」不检查「过了」）；豁免只在 `--smoke`
可达，这条断言就是把它钉死的。

**待决策**：是否接受「冒烟档 non-binding」。备选是把冒烟拉到 ≥1 epoch（冒烟就不再是
冒烟）或调 `--degeneracy-cross-std`（那是真的放宽判据，没做）。

**风险（必须盯）**：G4D 全量档首次 quick eval 在 step 2936 仍然 binding。若那时
`cross_std` 还在 1e-4 附近，全量会在约 1 GPU-小时后 rc=2。这是判据在做它该做的事，
不是本次改动引入的。

## 3. 同一失败模式的其他臂（本次**未**改，超出授权范围）

同一条 `cross_std <= 1e-4` 在冒烟里打死了三个臂，且 CARRIER 也贴着地板：

| 臂 | where | point_std | iddev | cross_std | 结果 |
|---|---|---|---|---|---|
| G4D (EPR-028) | quick_eval@step5 | 3.20e-01 | 1.22e-02 | **2.32e-05** | rc=2 |
| INTERPC (EPR-026) | quick_eval@step9 | 7.85e-03 | 4.94e-01 | **6.67e-05** | rc=2 |
| IDGATE (EPR-027) | quick_eval@step4 | 4.97e-03 | 4.97e-01 | **8.58e-05** | rc=2 |
| CARRIER (EPR-024) | step5 / step10 | — | — | 2.11e-04 / 1.48e-04 | 通过（在往地板走） |
| QDUAL (EPR-029) | quick_eval@step5 | 3.05e-01 | 9.42e-02 | 1.41e-02 | 通过 |

**AFFONLY (EPR-025) 的全量档已经死了**：`/home/bc/data/logs/whatb-affonly-20260815.log`
末尾 `DEGENERATE TRANSFORM at quick_eval@step2935`，`std over samples = 0.000000e+00`
（**精确的 0**，不是"小"）。2935 步之后 32 个样本逐位相同，这和 G4D 的"训练太早"
不是一回事，指向 `run_affonly_arm.py:894` 的 `z_probe = torch.stack([s.z for s in
quick_samples])` 拿到的是同一个向量，或 `head.transform` 那条路上 z 被丢了。
gpu1 因此已空。**本任务无权改 AFFONLY**，只报告。

## 4. 载荷脚本的一个误报（`/home/bc/agent-gpu-queue/waves/whatb_epr024_029_arm.sh`，未改）

G4D 冒烟日志里的 `whatb_no_step_row ... payload exited before the first steps.jsonl row`
是**假的**：run 目录里 `steps.jsonl` 有 5 行完整的 step 行。waiter 每 5 秒轮询一次，
第一次轮询时 setup 还没写出文件，等下一次轮询时 payload 已经退出，循环条件
`ps -p $PYPID` 先假，于是走了"死在首步之前"的分支。任何"启动到退出 < 一个 poll"
的失败都会这样误报。该文件正被 CARRIER/AFFONLY 全量占用，本次**不动**。

## 5. 重排命令草案（**未提交**，交主 agent 裁）

队列现状（17:48）：gpu0 跑 `EPR024_CARRIER`（40 分钟）、队列里 `EPR028_G4D`(157) 全量
gate 在旧冒烟板上；gpu1 被 `EPR029_QDUAL`(159) 占着 **gate-wait 33 分钟**，等一块永远
不会出现的 `whatb_QDUAL_smoke_20260815/metrics.json`，卡空转；其后是
`EPR027_IDGATE`(160)（eval bundle 仍无生产者）。

**这两个旧作业必须撤**：它们的载荷不带 `--pred-field-dir`，即使 gate 开了也会在
最后一步出板断言上死（G4D 全量要赔约 2.5 GPU-小时）。

`q submit` 走 `env -i` + 白名单（`/home/bc/agent-gpu-queue/q:69-75`），`WHATB_PREDF`
**传不进去**，所以写在 payload 命令行里。

```bash
CKPT=/home/bc/data/runs/q3vl_base_sft_v2seg_20260814/checkpoint-4976
BANK=/var/cache/veradata/preset_bank_full
RUNS=/home/bc/data/runs/what_b
LOGS=/home/bc/data/logs
ZROOT=/home/bc/data/runs/whatb/zcache_v2seg
PREDF=/home/bc/data/runs/whatb/predfield_stlang_v2seg
ARM=/home/bc/agent-gpu-queue/waves/whatb_epr024_029_arm.sh

# 0) 前置自检（期望 897 / {"stlang_m_low":408,"style_definition_ones":489}）
ls "$PREDF"/*.npy | wc -l
jq -c '.n_by_source, .v_what_coverage' "$PREDF/manifest.json"

# 1) 失败产物备份（红线：清空重跑前先备份）
ts=$(date +%Y%m%d%H%M%S)
mv "$RUNS/whatb_G4D_smoke_20260815"   "$RUNS/whatb_G4D_smoke_20260815.failed-$ts"
mv "$RUNS/whatb_QDUAL_smoke_20260815" "$RUNS/whatb_QDUAL_smoke_20260815.failed-$ts"

# 2) 撤旧 + 立刻补（cancel 必须绑 backfill）；IDGATE 先 hold，别让它占住 gpu1 一小时
cd /home/bc/agent-gpu-queue
bash q cancel EPR029_QDUAL EPR028_G4D
bash q hold   EPR027_IDGATE

# 3) 四条。冒烟都放 gpu1（现在是空的）⇒ 约 20 分钟内两个判决都出来；
#    G4D 全量放 gpu0 排在 CARRIER 后面，gate 在 gpu1 产出的冒烟板上（gate 是文件，跨卡可以）。
GATES=(--gate "$CKPT" --gate "$CKPT/protected_checkpoint.json"
       --gate "$BANK/luts.npz" --gate "$BANK/luts_meta.json"
       --gate "$ZROOT/train.generated.none.zcache.pt"
       --gate "$ZROOT/V_what.generated.none.zcache.pt"
       --gate "$ZROOT/V_what.generated.shuffle.zcache.pt"
       --gate "$ZROOT/V_what.generated.irrelevant.zcache.pt"
       --gate "$ZROOT/V_what.generated.const.zcache.pt"
       --gate "$PREDF/manifest.json")

bash q submit EPR029_QDUAL_SMOKE 1 "$LOGS/whatb-qdual-smoke-20260815-retry2.log" \
  --desc 'EPR-029 QDUAL 10 步冒烟(retry2: field_pred 补齐 489 style 全 1 场)' \
  --ready 'whatb_first_step|step 1/' --ready-timeout 7200 --mem-peak 12 \
  "${GATES[@]}" --gate-timeout-hours 48 --poll 30 \
  -- env WHATB_PREDF="$PREDF" bash "$ARM" QDUAL smoke

bash q submit EPR028_G4D_SMOKE 1 "$LOGS/whatb-g4d-smoke-20260815-retry2.log" \
  --desc 'EPR-028 G4D 10 步冒烟(retry2: field_pred 补齐 + 退化解 verdict 冒烟档非阻断)' \
  --ready 'whatb_first_step' --ready-timeout 7200 --mem-peak 12 \
  "${GATES[@]}" --gate-timeout-hours 48 --poll 30 \
  -- env WHATB_PREDF="$PREDF" bash "$ARM" G4D smoke

bash q submit EPR029_QDUAL 1 "$LOGS/whatb-qdual-20260815-retry2.log" \
  --desc 'EPR-029 QDUAL 主臂 117440 步(40 epoch × 2936)' \
  --ready 'whatb_first_step|step 1/' --ready-timeout 7200 --mem-peak 12 \
  "${GATES[@]}" --gate "$RUNS/whatb_QDUAL_smoke_20260815/metrics.json" \
  --gate-timeout-hours 1 --poll 30 \
  -- env WHATB_PREDF="$PREDF" bash "$ARM" QDUAL full

bash q submit EPR028_G4D 0 "$LOGS/whatb-g4d-20260815-retry2.log" \
  --desc 'EPR-028 G4D 主臂 117440 步(40 epoch × 2936)' \
  --ready 'whatb_first_step' --ready-timeout 7200 --mem-peak 12 \
  "${GATES[@]}" --gate "$RUNS/whatb_G4D_smoke_20260815/metrics.json" \
  --gate-timeout-hours 1 --poll 30 \
  -- env WHATB_PREDF="$PREDF" bash "$ARM" G4D full

# 4) IDGATE 放回队尾（它的 gate 仍缺，1 小时后 rc=78 让卡）
bash q release EPR027_IDGATE
bash q status
```

替代排法（保持原 wave 的卡分配）：把两个 QDUAL 放 gpu1、两个 G4D 放 gpu0；代价是
G4D 冒烟要等 CARRIER 全量跑完（约 2 小时）才开始。

`--mem-peak 12` 沿用 wave 的申报（仍是申报值不是实测；现在 gpu0 used 1.5 GiB、
gpu1 used 0，准入线 used<65 有余量）。

**gpu1 现在空着**：`EPR025_AFFONLY` 全量已 rc=2 死（见第 3 节），`EPR029_QDUAL`(159)
只是在 gate-wait。第 2 步一撤，第 3 步就把卡填上，不留空转。

## 6. 顺带记录（未改）

`run_qdual_arm.py` 的 `field_pred` / 四个 `query_match_*` 走 `extra_columns`，不经
`build_board` 的 normal-only 过滤；本臂 eval 集在 `run_qdual_arm.py:1170` 已经
`normal_only(...)` 过一遍，所以当前没有 low 混入。若将来把 low 放进 eval 集，这几列
会变成 pooled 口径。

## R1 载体层实施（2026-08-16）

范围：只改 `q3vl/whatb/arms/g4d.py`，新建 `q3vl/whatb/tests/test_g4d_r1.py`。
`glut.py` / `generator.py` / `caliber.py` / `carrier.py` / `qdecoder.py` /
`losses_l0.py` / `guards.py` / `criteria.py` / `run_g4d_arm.py` 一字未动。

### 单测计数（`CUDA_VISIBLE_DEVICES=""`，CPU）

- `pytest q3vl/whatb/tests/test_g4d_r1.py -q` → **43 passed**。
- `pytest q3vl/whatb/tests -q` → **719 passed / 44 failed**。44 条全部落在两个文件：
  - `test_g4d.py` 40 条：全部是引用 R1 已删符号或已改名模式的 import/API 层失败
    （`mu4` / `build_rotation_4d` / `l_m4d` / `ROT_SIGN_CHOICES` / `SIGMA44_FLOOR` /
    模式名 `joint`·`blockdiag`·`anchor2`·`maskblend`·`affine_s` / `A4` 不可构造 /
    `loss_preregistration` 的 total 串 / 首行列集含 `R_line`）。**本步未改该文件**。
  - `test_caliber.py` 4 条：同一根因，`run_g4d_arm.py:130` 引用
    `g4d.ROT_SIGN_CHOICES`，argparse 构建即 AttributeError。**runner 归第 2 步**。
- 其中一条值得单独记：`test_g4d.py::test_the_loss_reaches_the_output_layer_of_every_head[A2]`
  报「A2: no gradient reaches ['head_beta']」。这是 R1 §3 规定的行为（A2 生成 β 但不
  接线，梯度恒为 0），不是缺陷；A2 因此有 3N 个不更新的参数。

### 假设（保守默认继续，未静默拍板）

1. **A1 的 s 门用 `lutdata.mix_alpha` 而非裸 `x + s(T-x)`**。两者数学同式；用
   `mix_alpha` 是为了拿到端点吸附（`rendering.py:311-313` 的 `where(a==0/1)`），
   使 `f(x,0) == x`、`f(x,1) == T(x)` 逐位成立，并让 `R_line` 在 A1 上恰为 0。
   若判定端点吸附属于口径改动，请裁定。
2. **「强制 FP32」实现为「下限 FP32」**：`_math_dtype` 把 fp16/bf16 一律抬到
   float32，float64 输入保持 float64（G0 的 gradcheck / 对拍需要 float64）。不存在
   降精度路径。
3. **`GUARD_COLUMNS` 的取值**。任务卡只说删掉四个计数列，`STEP_EXTRA_COLUMNS` 的
   新列表里没有 `weight_underflow_frac`。落地为
   `GUARD_COLUMNS = ("cholesky_info_nonzero", "weight_underflow_frac")`，
   `STEP_EXTRA_COLUMNS` 严格按 R1 §10 列；`weight_underflow_frac` 仍由 aux 产出、
   写进 steps 行，但不进首行必需集。
4. **`STEP_EXTRA_COLUMNS` 按 mode 收窄**。卡片只标了 `beta_absmean` 仅 A3。A0/A1
   没有 τ，`tau_p05/p50/p95` 在这两臂也必然缺席，故新增
   `step_extra_columns(mode)`：A0/A1 去掉 `tau_*`，非 A3 去掉 `beta_absmean`；
   `STEP_EXTRA_COLUMNS` 保留为 A3 的极大集。
5. **`gnorm` 不由载体产出**。它是训练循环的量（梯度范数），载体没有梯度。它列在
   `step_extra_columns` 里（首行断言才能让缺列变响），但 `G4DAux.columns()` 不产
   它，需第 2 步的 runner 填。
6. **`total_loss` 仍无条件计算 `L_hc` / `L_sparse`**（`--loss-level 1` 下权重为 0）。
   这是 EPR-030 冻结七列的既有行为，未改；「关了但仍在算」的硬断言只对
   `L_s4d` / `L_img` / `R_line` 生效。若要连 L_hc 一起省掉，需要动 publish 的
   `FIRST_STEP_COLUMNS`（跨臂冻结），本步不动。
7. **A0/A1 也跑 `L_C` 对角检查**。共享 `glut_forward` 内部保留了 demo 的
   `|det| < eps -> I` 回退（那是 `glut.py`，禁改）。为了让 R1 §7「禁 fallback」在
   A0/A1 上同样成立，`_explicit_gate_forward` 在调用共享载体前先跑
   `lower_from_raw(..., strict=True)`，退化对角在进入回退分支之前就抛。
8. **`gradcheck` 覆盖的是生成侧原始量**（`chol_diag` / `chol_off` / `beta` /
   `tau_raw` / `mu_x` / `mu_s` / `opacity_logit`），τ 通过 `tau_from_raw` 的
   参数化被覆盖，未单独对 τ 本身做 gradcheck。
9. **`torch.quantile` 上限**。输入超 `2**24` 会硬报错。遥测列不该杀掉一次 run，故
   `_quantiles` 在超限时按固定 stride 抽稀（确定性），常量 `_QUANTILE_MAX` 写在
   模块里。当前口径 2,097,152 色/步不触发。

### 待决策

1. **G0 判据 2b 的字面不可满足**。任务卡要求「构造使普通域下溢到 0 的极端参数，
   断言对数域 w 仍非零且 `null_mass < 1`」。因为 `log Z = logaddexp(lse, log ε)`
   精确保留了 GLUT Eq.2 的 `+ε`，只要 `Σ_i a_i ≪ ε`，`w_i = a_i/(Σa+ε)` 与
   `null_mass` 就分别趋 0 和趋 1 —— 与 dtype 无关，对数域也一样。落地拆成两条：
   - `test_log_domain_survives_where_the_ordinary_domain_underflows`：混合构型，
     3 个基元的 `exp(log p)` 在 float32 与 float64 下都恰为 0（普通域已丢失信息），
     而 `log_a` 仍有限、`w` 非零、`null_mass < 1` —— 字面满足卡片的三条断言；
   - `test_log_domain_survives_where_the_ordinary_domain_overflows`：`L_C` 对角
     取 1e-15，普通域 `exp(log p)` 在 float32 溢出、`det` 下溢，naive 实现返回
     `nan/inf`；对数域返回有限 `w`、`null_mass < 1`。这条才是对数域与普通域**真正
     产生差异**的构型。
   若要求判据 2b 只保留字面那一条，请裁定。
2. **R1 §11 的三项原样未决**：warmup / wd（StatLUT 原配方 vs 本战役冻结的
   Adam/无 warmup/wd=0）、A4 是否要跑、`λ_line` 是否先用 G1 扫 {0, 0.1, 1.0}。
   本步 `LAMBDA_LINE = 0.1` 按预注册值落盘，`A4` 在 `G4D_MODES` 里留名、
   `G4DConfig(mode="A4")` 抛 `NotImplementedError`。
3. **`compose_headline` 签名已改为 `(img, field, f_img, *, mode)`**，`mode` 无默认值。
   `run_g4d_arm.py` 现有三处调用会 `TypeError`，按卡片属第 2 步范围。
4. **fp32 下分块不再逐位一致**：`solve_triangular` 在不同 P 分块下有 ~2.4e-7 的
   float32 舍入差（float64 下 4.4e-16）。旧 `test_g4d.py` 的
   `test_apply_to_image_chunking_is_numerically_inert` 若用的是逐位判据，第 2 步需
   改成容差判据。

## R1 runner 层实施（2026-08-16）

范围：只改 `q3vl/whatb/scripts/run_g4d_arm.py`、重写 `q3vl/whatb/tests/test_g4d.py`、
改 `q3vl/whatb/tests/test_caliber.py` 里 2 处 g4d 断言。
`arms/g4d.py` / `tests/test_g4d_r1.py` / `glut.py` / `generator.py` / `caliber.py` /
`carrier.py` / `qdecoder.py` / `losses_l0.py` / `guards.py` / `criteria.py` /
`zcache.py` / `splits.py` / `queries.py` 一字未动（mtime 已核）。

### 实测计数（`CUDA_VISIBLE_DEVICES=""`，CPU）

- `pytest q3vl/whatb/tests -q` → **751 passed / 0 failed**（本步之前：719 passed / 44 failed）。
  唯一 warning 在 `test_qdecoder.py:426`，本步之前即存在。
- `test_g4d.py` 重写为 45 条 runner 层集成测试（旧 40 条全部作废）。
- `test_caliber.py` 4 条失败：2 条随 `--glut4d-rot-sign` 删除自动修复；
  `test_frozen_pair_is_unchanged_on_every_arm[32x256]/[64x128]` 2 条改了断言值
  （`resolve_loss_level` 3→1、`effective_lambdas` (10.0, 0.001)→(0.0, 0.0)），
  并补了一行 `--loss-level 3` 的对照断言，判据未放宽。
- 三档 gate-stage dry-run 实测（`--dry-run`，三条命令都在 scratchpad 落盘）：

  | 档 | mode | n_lut_pool | batch_structure | colours/step = n_pairs_s | steps_per_epoch | total_steps | eval_board | loss_level |
  |---|---|---|---|---|---|---|---|---|
  | G1 | A1 | 1 | 1 × 2048 × 4 | 8,192 | 93,934 | 2,000 | false | 1 |
  | G2 | A2 | 32 | 32 × 512 × 4 | 65,536 | 2,936 | 4,000 | false | 1 |
  | G2 | A3 | 32 | 32 × 512 × 4 | 65,536 | 2,936 | 4,000 | false | 1 |
  | G3（另测，`--data v2seg+l8`） | A3 | 3,166 | 256 × 2048 × 4 | 2,097,152 | 469 | 18,760 | true | 1 |

- LUT 桶实测：`--data v2seg` 的 train normal-only 桶 = **3,081** 条 lut_id；
  `--data v2seg+l8` = **3,166** 条。R1 §1 写的是 3,149，两者都对不上。
- 训练环跑通实测（CPU、小 n_gauss）：G1/A1、G2/A3、G1/A3+`--alpha-s`、
  `--cond oracle` 全量板路径（`--eval-limit 3`）、`--cond vlm --smoke` 路径。
  注入 NaN 的 `L_rec` 实测：`quick_eval@step2` 以 **rc=3** 终止，
  `metrics.json` 落 `void_reason`，`best.pt` 未生成。

### 假设（保守默认继续，未静默拍板）

1. **LUT 池按实测取，不写死 3,149**。G3 的 `n_lut_pool` = 训练桶实测 distinct
   lut_id 数；`run_setup.json` 同时写 `train_lut_bucket_n` 与
   `train_lut_bucket_matches_r1_3149`（当前均为 false）。若 3,149 另有出处
   （例如 `luts_meta.json` 的某个子集，而非 train 桶的 distinct），请裁定。
2. **`--loss-level` 默认落在 caliber 的 1**（R1 §4.4 pure L1）。`--w-img > 0` 时
   仍抬到 4（`L_img` 要靠 level≥4 才进首行列集）。这改了 `test_caliber.py` 里
   跨臂并列的 2 条断言（其余四臂的 config 对象默认仍是 3）。
3. **oracle embedding 初始化 = N(0, 1)**，私有 generator（seed = `--seed`）。
   规格未给初始化；选 1.0 是因为它替代的是 `pi(z_color)` 的输出量级（LayerNorm
   后再线性），生成器各头末层零初始化使 step 0 与初始化尺度无关。
   参数量、初始化串写进 `run_setup.json`。
4. **`--carrier glut3d` 映射到 A0**（R0 的 `maskblend` 已不存在）。A1 也是 3D 载体，
   但它带 s 门、是 4D 目标臂，所以「纯 3D」取 A0。
5. **oracle 档的三负控制与 `field_pred` 判据豁免**。依据 R1 §9.1（G1–G3 不出
   headline+五基线+三负控制的完整板，那是 G4）。豁免列表 `ORACLE_WAIVED_CRITERIA`
   写在 runner 里，板上落 `criteria_waived` + `oracle_reference=true` +
   `published=false`。其余预注册键（含 `loc_in/loc_band/loc_out`、五个 `grid_s*`）
   在 oracle 板上仍然必需。
6. **oracle 档跳过 colorspan 启动断言**，并在 `run_setup.json` 落
   `{"ran": false, "skipped_reason": "cond=oracle: no colour text is read out"}`。
   该断言钉的是颜色文本编码器，只有 Experiment Z 消费。
7. **G1 的退化解 verdict 非阻断**。单 LUT 池下 `cross_std` 恒为 0 是 G1 的设计
   （单 LUT overfit），不是退化。三个数照算照打照落盘，只是不杀进程；
   `binding_reason` 明写原因。判据阈值一个字没改。
8. **epoch 轴仍用记录数**（`ceil(n_train / B)`），不用 LUT 池长度。理由：
   `mining_ratio` 的 5→20 epoch 斜坡是按记录数标定的；若改用池长度，G3 的
   epoch 20 会在 260 步就到达。G1 因此 `steps_per_epoch = 93934`（epoch 恒 ~0），
   但 G1 本来就 `mining=False`。
9. **`n_colors` 改为记 (x,s) 对数**（= `n_pairs_s` = B×Q×4），另加
   `n_colors_distinct`（= B×Q）。依据 R1 §8.1「保持 2,097,152 色/步 …
   256 × 2048 × 4 = 2,097,152」——规格把「色/步」就是按对数在数的。
   两个数并排落盘，任何一个都不能被当成另一个读。
10. **`L_s4d` 的有限差分步长 = `1/(--reg-grid - 1)`**（默认 1/16，与 17⁴ 格距一致），
    每样本 `--reg-points`（默认 256）个随机 4D base point，四个轴各一次前向。
    `g4d.grid_4d` 在训练路径已完全不出现（AST 断言钉死）。
11. **`L_img` 的采样口径**：每步至多 `--img-batch`（默认 4）个样本、每样本
    `--img-pixels`（默认 768）个像素，像素上的 GT 用 `mix_alpha(x, L_l(x), α(px))`
    ——与整图 `I*` 在同一像素上逐位相同。oracle 档没有记录，`--w-img > 0` 时直接
    `SystemExit` 要求 `--cond vlm`。
12. **`compose_headline` 在 field 行用该行自己的场，不再用 GT α**。A0 的
    `Î = I + S⊙[T(I)−I]` 里的 S 就是被扫的那个场；A1–A3 恒等。这同时是 R1 §5
    第二个后果（`E_out` 不再被构造性压到 0）落地的地方。
13. **`gnorm` 取 `clip_grad_norm_` 的返回值**（clip 前）。`--max-grad-norm` 为 0 时
    传 `inf`，即只测不裁。

### 待决策

1. 3,149 的出处（假设 1）。
2. oracle embedding 初始化尺度（假设 3）。
3. R1 §11 三项原样未决：warmup / wd、A4 是否跑、`λ_line` 是否先用 G1 扫
   {0, 0.1, 1.0}（`--lam-line` 旗标已就绪，默认 0.1）。
4. G2 的首次 quick eval 落在 step 2000（`--quick-eval-every` 默认 2000），
   总步数 4000 ⇒ 只有 2 次 quick eval；是否要把 G1/G2 的 `--quick-eval-every`
   调小以拿到更密的曲线（未自行改）。
5. `--cond oracle` 且不带 `--gate-stage` 时，`needs_eval_board` 默认取 true
   （G3 语义）。若希望「无 gate-stage 的 oracle 也只出函数值」，请裁定。
