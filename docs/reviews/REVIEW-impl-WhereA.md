# REVIEW-impl-WhereA：Stage-Where-A basis calibration 实现审阅

> **本文档有两节判决。初审（2026-08-05 早）判 6 BLOCKER；聚焦复审（2026-08-05 晚，见文末
> 《复审》节）确认 6 项全部关闭，新发现 1 项 BLOCKER（B-7）。当前有效判决以《复审》节为准。**

> 审阅人：独立实现审阅 subagent（与实现者无共享上下文）
> 日期：2026-08-05
> 规格权威：`docs/METACANVAS_WHERE_WHAT_FINAL_EXPERIMENT_PROTOCOL_2026-08-04.md` §2.3 / §3 / §4 / §5.5 / §10.2 / §14
> 审阅对象：`q3vl/where/`（13 模块 + 4 脚本 + `tests/` 97 例）、
> `experiments/Q3VL_metacanvas_where_what_20260804/where_a/`（NOTES.md、PREFLIGHT_WHERE_A_PENDING.md、
> `mask_validation_{V_where,train}.json`、`preflight_where_a_cpu_dryrun.json`）
> 本次审阅**未修改任何代码或数据、未使用 GPU**（两卡由 Base SFT 占用，PID 3395226/3395227，实测仍在跑，
> 进度 1199/4976）。只跑只读检查、现有单测（CPU）与不导入 `q3vl.where.*` 的独立复算脚本。

---

## 判决

**BLOCKER 数量 = 6；不准许进入正式校准（S4），也不准许按现状跑 GPU preflight（S2）。**

理由分两层：

1. **S2 本身缺项**：B-4 指出 §14 项 4 要求的"一次 guided upsample 校验"在 preflight 里只有合成数据版本，
   而**真实高分辨率通路在整个生产代码里是死代码**（`evaluate_latent` / `combine_then_upsample` /
   `PreparedSample.guide()` / `CalibConfig.upsample` 全部无生产调用方）。这条通路我实测存在一个
   **静默失效模式**（见 B-4 的实测数字），必须先补进 preflight 再跑 S2，否则 S2 通过也不构成放行依据。
2. **S4 前必须清的**：B-1/B-2/B-3/B-5/B-6。其中 B-3 是会直接毁掉 4 个臂可比性的调度错误，
   B-1/B-2 是 §10.2 rejection 纪律的实现缺口，B-5 是 Where-B 无法开工的产物缺口，
   B-6 是主 agent 裁定 D3 所依据的理由在 D2 下不成立。

**好消息**：本次审阅的核心公式部分（Phi-64、`s_low`、符号规范化、两个 readout 的参数域、
merge-order unshuffle、mask 数据回连链路）**逐符号核对全部通过**，且我用独立脚本另抽 48 个样本
复算 mask 链路，与交付物数字一致。B-1…B-6 全部是"周边工程"缺陷，**没有一条需要重写公式**，
预计清完在半天量级。

---

## 一、必审重点 1：公式逐符号对照（§4.2 / §4.3）

| 符号 | 协议原文 | 实现 | 判定 |
|---|---|---|---|
| `geo5(p)` | `[x, y, P2(x), P2(y), x*y]` | `phi.py:87` `[X, Y, P2(X), P2(Y), X*Y]`，`legendre_p2 = (3t²−1)/2` | **pass** |
| 坐标口径 | §4.1"按每张图真实宽高比生成，不将图像压成 512x512" | `phi.py:70-74` `short_side_unit`：短边跨 [−1,1]、长边跨 [−AR,AR]、格子在特征空间里是方的 | **pass** |
| `range(p)=[L,S]` | "按逐图固定统计口径标准化" | `phi.py:102-106` L=Rec.709 luma、S=HSV `(max−min)/max`；`phi.py:243-244` 逐图零均值单位方差 | **pass**（L/S 定义本身协议未钉，E2 先例，NOTES V7 已核实） |
| 残差化 | "64 个 semantic 通道逐图对 `[1, geo5, L, S]` 做最小二乘残差化" | `phi.py:137-141` 8 列 design block；`phi.py:177-189` 正规方程 + 1e-10 trace-相对 ridge（`lstsq` 无 autograd，B 的唯一梯度通路必须可导，理由正确） | **pass**。注：用标准化后的 L,S 建 block 与用原始 L,S 张成同一子空间（含常数列），等价 |
| 标准化 | "再做零均值/单位方差标准化" | `phi.py:119-134` `standardize_live`：被残差化完全湮灭的通道（std < 1e-8·std_before）**置零并计数**，而不是把 1e-15 的舍入噪声放大成"单位方差特征" | **pass**（超出规格的正确处理） |
| `phi_dir ∈ R^71` | `[geo5, L, S, sem_1..64]` | `config.py:44` `PHI_DIR_DIM=71` + `phi.py:260-262` 运行期断言 | **pass** |
| `w_dir = normalize(w_raw)` | — | `basis.py:31-32` | **pass** |
| `alpha = softplus(alpha_raw)` | — | `basis.py:35-36` | **pass** |
| `s_low = 3*tanh((w0 + alpha*<phi_dir,w_dir>)/3)` | — | `basis.py:118-119` `q = w0 + alpha*(phi@w_dir); 3*tanh(q/3)`，`S_SCALE=3.0` | **pass**（代码正确；**docstring 写错了**，见 N-1） |
| 符号规则 | "绝对值最大系数为正" | `basis.py:130-160`：翻 `(w_raw, w0)` **并** `mirror(rho)`，`s→−s` 且掩膜逐点不变；`w_raw=0` 时不翻（方向未定义） | **pass**（这是最容易只翻 `w_dir` 而改掉掩膜的地方，实现是对的；`test_basis` 12 组随机 latent × 2 readout 实测差 <1e-12） |
| R-Band `b(z)` | `sigmoid(k(z−mu+h)) − sigmoid(k(z−mu−h))` | `readout.py:104` 逐符号一致 | **pass** |
| R-Band `m(z)` | `pi*b + (1−pi)*(1−b)` | `readout.py:105` | **pass** |
| `h>0` | 协议只要求 `h>0` | `config.py:61` 有界 sigmoid `(0.02, 2.50)`；D6 已列为决策 | **pass** |
| `k ∈ [1,40]` | — | `config.py:56` + `bounded_sigmoid` | **pass**（raw=±1e6 实测仍在界内） |
| `pi = sigmoid(pi_raw)` | — | `readout.py:94` | **pass** |
| `mu_i = linspace(−3,3,12)` 固定 | — | `readout.py:73-76` 每次现算的常量，**不是 Parameter**；`param_shapes("cband12")` 里没有 `mu` | **pass** |
| `sigma ∈ [0.025,0.30]` | 禁裸 exp | `readout.py:112` `bounded_sigmoid`；全模块唯一的有界映射入口 | **pass** |
| `g_i / m(z)` | `sum c_i g_i / (sum g_i + eps)` | `readout.py:128-129` 逐符号一致；`o_i, c_i = sigmoid(raw) ∈ (0,1)` | **pass**（与 `experiments/_archive/2026-08-10/E2_basis_fit_20260803/e2lib.py:212` 的 `(N·o·c).sum/((N·o).sum+1e-9)` 数学等价，eps 同为 1e-9） |
| mirror 恒等式 | 协议未要求，实现自加 | Band: `mu→−mu`；CBand12: 索引翻转（`mu` 网格对称）。我手推验证：`b(−z;−mu)=b(z;mu)` 成立 | **pass** |

**结论：公式层无 blocker、无 nit（除 N-1 的注释错字）。**

---

## 二、必审重点 2：guided upsample 顺序（§4.2）

协议原文："组合完成后**只对标量 `s_low`** 做一次 edge-aware guided upsample……**不能先上采样 64 个通道再组合**。"

- `upsample.py:84-89`：`s_low_map.shape[1] != 1` → `ChannelOrderError`。**防护真实存在**，
  `test_upsample.py:24-32` 用 64 通道和 2 通道两个用例实测抛错，preflight `WA-P4c` 也复测。**pass**
- `combine_then_upsample`（`upsample.py:109-127`）先 `s_low(phi, latent)` 得到 `(P,)` 标量再 reshape 上采样，
  顺序正确。**pass**

**但防护可绕过（N-2）**：守卫只看**通道轴**。把 64 个语义通道折进 **batch 轴**
（`sem.reshape(64,1,h,w)`）会顺利通过 `shape[1]==1` 的检查，得到的正是被禁止的"先逐通道上采样"。
当前代码库没有这样调用，属于纵深防御缺口而非现行违规。建议加一条 `B == n_images` 的断言或在
docstring 里写明 batch 轴语义。

---

## 三、必审重点 3：Oracle fit（§4.4 / §10.2）

| 要求 | 实现 | 判定 |
|---|---|---|
| 多起点 L-BFGS | `oracle.py:157-199`：lsq 起点 + 质心-径向起点 + `n_random` 随机 + 1 个近退化起点；band 再 ×2 极性 | **pass** |
| float64 | `oracle.py:262-264` `phi/target → cfg.dtype='float64'`；`calibrate.py:125` 内层显式 `.double()` | **pass**（协议 §10.2 只要求 fp32 oracle fit，float64 更严） |
| 固定容差 | `config.py:88-89` `tol_grad=1e-9 / tol_change=1e-11`，与 E2 `e2lib.py:418` 一致 | **pass** |
| 失败进 rejection report，不静默换零向量 | `fit_latent` 本身**正确**：`status/reject_reason/flags/start_losses/n_failed_starts` 齐全，`test_oracle.py::test_unfittable_target_is_rejected_not_zeroed` 断言返回的仍是最优解而非零向量 | **函数层 pass，消费层 fail → B-1 / B-2** |
| canonicalization 不动像素 | `oracle.py:340-341`：规范化后重算 loss，偏差 >1e-6 直接判 `canonicalisation_changed_loss` 拒绝——这是一条**自检式断言**，写得很好 | **pass** |

---

## 四、必审重点 4：四臂定义（§4.4）

| 臂 | 协议 | 实现 | 判定 |
|---|---|---|---|
| `BA-0-Fixed` | seeded orthogonal 1024→64，不训练 | `config.py:108-114` `ARM_READOUTS["BA-0-Fixed"]=()` → `trains_projector=False` → `requires_grad_(False)`、**不建 optimizer**（`calibrate.py:96-107`）；`projector.py:35-45` 固定 seed 的 QR（符号钉死，可复现）；`test_calibrate.py::test_ba0_never_trains_the_projector` 断言权重逐位不变 | **pass**。实测正交误差 4.3e-7、cond 1.0000004 |
| `BA-1-Band` / `BA-2-CBand12` | 只由单 readout oracle 反传 | `ARM_READOUTS` 单元素，`step` 只对该 readout 累加 | **pass** |
| `BA-3-Joint` | "Band/CBand **独立** oracle latent，共享 `B` 联合校准" | `calibrate.py:140-153`：同一个 `parts.phi_dir` 上对两个 readout **各自独立**跑 `fit_sample`，两个 loss 按 0.5/0.5 加权反传到唯一的 `self.projector` | **pass**（权重 0.5/0.5 是 D7 决策，协议未钉） |
| 校准后固定 B 重拟合 | "分别用多起点 L-BFGS 重拟合最终 `w*,rho*,s*,r*(z)`" | `run_calibration.py:129-132`：`FitConfig(n_random=6, max_iter=120)`（全档），`cal.evaluate(...)` 在 `B` 已固定的情况下重拟 | **部分 pass → B-5**（只在 `V_where` 上重拟并发布，train 侧完全没有） |
| 不使用 instruction | `calibrate.py` / `pipeline.py` 全链路不读 `record["instruction"]` / `record["color"]`（已 grep 确认） | **pass** |

---

## 五、必审重点 5：mask 数据回连链路 —— 独立复算

我写了 `indep_mask_check.py`（**不 import `q3vl.where.*`**，只用 stdlib + numpy + PIL），
在 `V_where` 抽 24 个、`train` 抽 24 个 local 样本（stride 5 / 4001，跨 l1–l6，normal/low 各半），
逐样本做了比实现方**更严**的检查：

| 检查 | 我的做法 | V_where 24 | train 24 |
|---|---|---:|---:|
| 定位 | `record.image.origin.root` → `catalog.sqlite3` `members(sample_id, suffix='.cgt.png')`，要求**恰好 1 行** | 24/24 | 24/24 |
| **偏移量真伪** | 直接读 `offset_data − 512` 处的 512 字节 USTAR 头，比对 `name` 与八进制 `size` | 24/24 名字+大小+`ustar` magic 全对 | 24/24 |
| sha256 | 逐字节复算 | 24/24 | 24/24 |
| PIL 模式 | 必须 `L` | 24/24 | 24/24 |
| 宽高比 | 对 `image.oriented_w/oriented_h` | 最大相对误差 3.0e-4 | 3.7e-4 |
| **对齐（正向）** | `corr(|I_tar − I_in|, mask)`，其中 `I_in` 取**模型真正看到的 baked 512 短边图**（实现方用的是源 `.in.jpg`，我换了一个更贴近训练的口径） | 中位 **0.793**，最小 0.359 | 中位 **0.818**，最小 0.388 |
| 编辑能量落在 mask 内 | `Σ(d·m)/Σd` | 中位 **0.712** | 中位 **0.687** |
| **对齐（负控制）** | 与左右翻转 / 上下翻转的 mask 比相关，正确朝向必须赢 | 23/24 | 20/24 |

**5 个"翻转负控制"未赢的样本全部是 `mask_mean ≈ 0.4989…0.5009` 的左右半幅掩膜**，
对**上下**翻转天然近似不变（`corr_flipud` 与 `corr_d_mask` 差 0.002–0.006），而具判别力的
`corr_fliplr` 全部显著为负（−0.18…−0.67）。因此**不是错位，是我这条启发式检查在半幅掩膜上退化**。
朝向结论：**确认正确**。

另外我独立验证了一件交付物没提、但风险最高的事：**`members.image`（模型真正吃进去的图）是 I_in 不是 I_tar**。
5 个样本上 `MAE(baked, I_in) ∈ [0.0032, 0.0070]`、`MAE(baked, I_tar) ∈ [0.024, 0.131]`，
量级差 5–40 倍，**无 I_tar 泄漏**。

**结论：mask 回连链路 pass，交付物 `mask_validation_*.json` 的核心数字复算一致**
（实现方报 corr 中位 0.787/0.783，我报 0.793/0.818，口径略不同，量级一致）。
唯一问题是 NOTES 里若干数字与归档 JSON 对不上（N-9）。

---

## 六、必审重点 6：§14 项 4/5/6 的 preflight 脚本审查

### 项 6（`WA-P6a` / `WA-P6b`）—— **pass**
`preflight.py:102-139` 在 raw = ±1e6/±50/0 五个极端点实测 `h>0`、`k∈[1,40]`、`pi∈[0,1]`、
`sigma∈[0.025,0.30]`，另加 `mu` 网格 = `linspace(−3,3,12)` 且对称、50 组随机参数下 `m(z)∈[0,1]`。
`preflight.py:284-313` 对每个真实拟合出的 latent 检 `w_dir` 符号规则、单位范数、`alpha>0`、readout 在界内。
四项要求全覆盖，且 `canonical` 标志与实际符号做了**交叉一致性**检查（`preflight.py:295`），是好实践。

### 项 5（`WA-P5`）—— **pass，一条 nit**
残差化后最大相关（阈 1e-4）、design Gram cond、phi Gram cond（阈 1e10）、每 readout 拟合成功率（阈 90%）四项齐全。
N-7：`check_calibration_health` 用的是**新建的 seeded-orthogonal projector**，测的是 BA-3 的**起点**而非校准后的 B。
作为"开跑前的门"这是对的，但报告必须写清，且校准后应重跑一次（残差化条件数会随 B 训练变化）。
N-8：报的是 **Gram** 的条件数（= 矩阵条件数的平方），阈值 1e10 相当于 `phi` 的 1e5，需在报告里标注口径。

### 项 4（`WA-P4a` / `WA-P4c`）—— **形状/宽高比 pass，"位置编码"与"一次 guided upsample"未达标**
- `WA-P4a`（`preflight.py:165-218`）：真实形状 `(H/16·W/16, 1024)`、grid 宽高比 == 图像宽高比、
  以及一条设计得很好的**空间相干性对照**（正确 unshuffle 的邻域余弦增益必须高于朴素 reshape）。**pass**。
- N-6：§14 项 4 明写"**位置编码**"，脚本用相干性做间接代理，没有直接检 `fast_pos_embed_interpolate`。建议补一条平移等变探针。
- **B-4**：`WA-P4c`（`preflight.py:142-160`）只在**合成**张量上验证顺序与常数保持，
  真实数据上的"组合→一次上采样→原分辨率掩膜"从未跑过（详见下一节）。

### `WA-P4b`（冻结泄漏探测器）—— **逻辑正确，两条 nit**
`preflight.py:221-242` 的做法是**比对权重**而不是比对激活。我认为这是**更强**的检查：
`load_vision_tower` 加载的是同一个 `Qwen3VLVisionModel`，若除 `merger.` / `deepstack_merger_list.` 之外
所有张量逐位相同，则 `F_pre` 在任何输入上都必然相同——比抽样比对激活的覆盖面严格更大。

我核实了它依赖的三个事实：
1. **跳过的两个前缀名正确**：`model.safetensors.index.json` 里 `model.visual.*` 的一级前缀恰为
   `blocks(288) / deepstack_merger_list(18) / merger(6) / patch_embed(2) / pos_embed(1)`，共 315 个张量；
   `q3vl/train/constants.py:68` 的 `VISUAL_TRAINABLE_PREFIXES` 也恰是这两个。**跳过集合 = 可训练集合，不多不少**。
2. **`SFT_CHECKPOINTS` 路径真实**：`/home/bc/data/runs/q3vl_base_sft_20260804/job.marker` 的
   `protected_steps = [2488, 4976]`，与 `config.py:151-154` 完全一致（当前只落了 500/1000，属正常进度）。
3. 检查在 `device="cpu"` 上跑（`preflight.py:356`），**不会碰 GPU**。

N-5（两条）：(a) `n_tokens=512` 形参完全未使用，像是被砍掉的激活比对残留；
(b) 若 checkpoint 的权重布局与 base 不同（`load_vision_tower` 对 missing/unexpected 都 `raise RuntimeError`），
异常会**穿透 `run_where_a_preflight` 并让整个 preflight 崩掉、连 JSON 都不落盘**。必须 try/except 成 `fail`。

---

## 七、必审重点 7：shard 契约（§2.3）—— pass + 2 nit

- 不压缩 tar、目标 1 GiB（协议 1–4 GiB）、staging + `os.replace` + `fsync` 原子发布、
  index 行含 `shard/member/offset_data/length/size/sha256/schema_version`、manifest 有
  `sample_count/member_count/shard_count/status`、`verify_published` 随机读 + checksum
  ——全部由 `q3vl/data/shardio.build_from_memory` 提供，`test_packing.py` 5 例覆盖。**pass**
- durable 写 `/mnt/nfs/bc/data/datasets/where_a-20260805/`，本地只放可重建的 run 产物。**pass**
- 同一样本三个成员连续产出（`packing.py:75-90`），尽量同 shard。**pass**
- N-14：§2.3 点名"**basis 元数据**"也在 shard 契约里，而 `write_basis` 写的是裸 `B.npy` + `basis.json`。
  单件 256 KB 产物走 shard 意义不大，但需在 REPORT 里写成**显式豁免**而不是默认省略。
- N-3 / N-4：见下。

---

## 八、BLOCKER 清单

### B-1 · 被拒绝的 oracle fit 仍然驱动 B 的梯度，也仍然进 oracle 天花板统计
**文件行号**：`q3vl/where/calibrate.py:141-155`（训练 step）、`q3vl/where/calibrate.py:201-207`（evaluate）；
零向量来源 `q3vl/where/oracle.py:309-318`。
**规格条文**：§10.2 "失败样本进入显式 rejection/fit report，**不静默换成零向量**"。

`fit_latent` 本身是对的，但**没有任何消费方看 `fit.status`**：

```python
fit = self.fit_sample(...)
if fit.status != "ok":
    rejected += 1          # 只计数
...                        # 没有 continue
latent = fit.latent...     # 照样用
loss_sum = loss_sum + w * loss     # 照样反传
```

两个后果：
1. `all_starts_failed` 分支返回的 latent **字面上就是零向量**（`oracle.py:310-312`），
   它会被 `mask_from_latent` 算出一个常数掩膜并把梯度打进 `B`。这正是协议禁止的"换成零向量"，
   区别只是"有记录"而不是"静默"——但记录并没有阻止它被使用。
2. `evaluate` 把被拒样本的 metrics 也塞进 `per[r]`，于是**报出去的 oracle 天花板中位数被污染**。
   这会直接放宽 §5.6 的 `相对逐图 oracle 的 soft-IoU ≥ 85%` 这道门（分母被压低），
   属于对 Where-B **有利方向**的偏差，不可接受。

**要求**：训练 step 对 `status != "ok"` 的 (样本×readout) 跳过外层 loss；
`evaluate` 的聚合只在 `ok` 子集上算，被拒集合单独出一行（n、reason 分布、若聚合则单列）。

### B-2 · 校准 epoch 的 fit/rejection report 根本没有落盘
**文件行号**：`q3vl/where/scripts/run_calibration.py:118` —— `cal.step(batch, record_fits=False)`。
**规格条文**：§10.2 "失败样本进入**显式** rejection/fit report"。

真正做拟合的那一遍（42,752 或 75,544 个样本 × 2 readout）只在 `steps.jsonl` 里留下每步一个
`n_rejected_fits` 整数；被拒的是哪些 `sample_id`、什么 `reject_reason`、`alpha_collapsed` /
`constant_mask` 打了多少 flag，全部丢弃。只有 `V_where` 的 evaluate 阶段有逐样本行。
一旦某个 build / 某类掩膜系统性拟合失败，事后无法归因，也无法满足交付规范里
"失败案例必须有"的要求。

**要求**：训练阶段至少落一份 `fit_rejections.jsonl`（`sample_id / readout / status / reason / flags / loss`）。
全量 fit 行太大可以只落非 `ok` 的行 + 每 N 步一次的采样行。

### B-3 · LR 调度按**未过滤**的样本数建，warmup 与 cosine 双双错位
**文件行号**：`q3vl/where/scripts/run_calibration.py:91-93`。
**规格条文**：§10.2 `warmup_ratio: 0.03`、`scheduler: cosine`、`epochs: 1.0 over all **eligible** local train samples`。

```python
n_local = sum(1 for r in ShardIndex.load(...).samples if r.meta.get("build") in LOCAL_BUILDS)
total_steps = max(1, (args.train_limit or n_local) // args.batch_size)
```

`n_local` 只按 `build ∈ l1..l6` 数，**没有过 `eligibility()`**；而实际训练循环走的是
`source.iter_split("train")`，会被 `winner_confidence=low`（默认 D1）等条件筛掉。
我独立数了一遍 `train.index.jsonl`：**local 75,544，其中 low 32,792，normal 42,752**（与 NOTES V8 一致）。

于是在当前 D1 默认下：
- `total_steps = 75544//8 = 9443`，实际步数 `42752//8 = 5344`；
- warmup = `round(9443×0.03) = 283` 步 = **实际总步数的 5.3%**（协议要 3%）；
- cosine 只走到 `p = (5344−283)/(9443−283) = 0.552`，**结尾 LR 停在峰值的 ≈ 41.8%，从不退火到 0**。

这不是"稍微不准"：三个训练臂（BA-1/2/3）都会在一个**从未完成退火**的 schedule 下停机，
而 §4.4 要求四臂用于**归因比较**——它们必须共享同一个已完成的调度。
代码注释还写着"eligible-sample count comes from the frozen split index, not from a guess"，
但拿到的恰恰不是 eligible 数。

**要求**：先用 `eligibility()` 数一遍（或先跑一遍 `iter_split` 计数并缓存），再建 schedule；
把最终 `total_steps` / `warmup_steps` / 实际步数写进 `run_setup.json` 并在结束时断言二者相等。

### B-4 · 生产链路里**没有任何东西**走过高分辨率通路，而该通路有实测到的静默失效模式
**文件行号**：`q3vl/where/oracle.py:351-372`（`evaluate_latent`，**无生产调用方**）、
`q3vl/where/upsample.py:109-127`（`combine_then_upsample`，仅被 `evaluate_latent` 与单测调用）、
`q3vl/where/pipeline.py:41-42`（`PreparedSample.guide()`，**无调用方**）、
`q3vl/where/scripts/run_calibration.py:85`（`UpsampleConfig()` 传进 `CalibConfig` 后 `Calibrator` 从不读取）、
`q3vl/where/preflight.py:142-160`（`WA-P4c` 只用合成张量）。
**规格条文**：§14 项 4 "`F_pre` 的真实形状、宽高比、位置编码和**一次 guided upsample 校验**"；
§4.2 "得到原图比例下的 `s(p)`"。

我在 CPU 上实测了这条死通路的两个数值性质（float64，`q3vl.where` 自身的函数）：

1. **guided filter 会把 `s` 推出 (−3,3)**：对一个已饱和到 ±3 的 `s_low`（32×48 → 512×768），
   - 与 guide 不相关时 `s_hi ∈ [−6.51, +6.63]`；
   - **guide 与 s 的阶跃完全对齐**（最有利情况）时仍有 `s_hi ∈ [−3.58, +3.57]`。
2. **CBand12 在网格外与网格间会塌成恒 0**：`sigma` 取下界 0.025、固定中心间距 `6/11 = 0.545` 时，
   即便令**所有** `c_i = 0.982`：

   | z | 0.0（两中心正中） | −0.273（中心上） | 3.0（端点中心） | 3.3 | 4.0 |
   |---|---:|---:|---:|---:|---:|
   | m(z) | **0.000000** | 0.982014 | 0.982014 | **0.000000** | **0.000000** |

   原因是分母 `sum g_i` 掉到 `CBAND_EPS=1e-9` 以下（`g ≈ 6e-27`），`m → 0/eps = 0`。

两条合起来就是一个**教科书级的静默失效**：低分辨率拟合只在 `z ∈ (−3,3)` 上被评价，
所以拟合完全可以挑 `sigma = 0.025` 而不付任何代价；上采样后恰恰在**强边缘处**把 `s` 顶出 ±3.3，
于是最终掩膜在边缘上被打出一排 0。而**孤儿检查/边界检查全程沉默**——这正是 `CLAUDE.md`
《s 缓存消费契约》里"第二种失败模式：场被 clamp / 落到域外，值仍看不出异常，但 s 轴已经没了"的翻版。
按同一条纪律，**消费方必须声明期望域并断言生数据住在里面**，当前实现一条都没有。

另外，D5（`radius_low=2, eps=1e-3`）被标注为"E2 先例"，但 E2 的 `guided_filter` 是
**在全分辨率上逐个 basis 通道**做的（`experiments/_archive/2026-08-10/E2_basis_fit_20260803/prep_data.py:149`，`r=32`）
——**恰好是 §4.2 现在明令禁止的顺序**。先例不可迁移，参数必须重定。

**要求**（三条，缺一不可）：
1. preflight 增加真实高分辨率检查：对 `WA-P4a` 拿到的样本调用 `evaluate_latent(..., guide_hi, target_hi)`，
   报 `s_hi_range`、`hi` 档 soft-IoU、以及 `hi` 与 `low` 的差；
2. 显式声明并断言消费域：`s_hi` 超出 `[−3,3]` 的比例必须被统计；若采用 clamp/rescale，必须写进 config 并说明理由（**禁止逐图归一化**）；
3. D5 的 `radius/eps` 扫描从 PREFLIGHT_WHERE_A_PENDING.md 的"建议"升级为 **S2 的硬性输出**，定档后写回 `config.py`。

### B-5 · train split 的 oracle latent 从未生成，Where-B 的 `L_s/L_curve/L_dir` 无米下锅
**文件行号**：`q3vl/where/scripts/run_calibration.py:128-162`（只对 `V_where` 重拟并 `pack_oracle`）；
`q3vl/where/config.py:158` `ORACLE_DIR / arm / "V_where"`。
**规格条文**：§4.4 "逐图 oracle 参数只作为**监督**和 ceiling"；§5.5 `L_s = Huber(s_pred/3, s*/3)`、
`L_curve = mean_z |R(z;rho_pred) − r*(z)|`、`L_dir = 1 − cos(w_dir_pred, w_dir*)`，
且 §5.5 的两段 schedule 前 30% 步以 1.00 权重使用它们。

这三项都是**训练时**的监督，需要 Where-B 训练样本上的逐图 `s*, r*(z), w_dir*`。
当前交付只在 `V_where`（400 个 local，D1 默认下 224 个）上产 latent，`train` 侧一个都没有。
Where-B 一开工就会卡住；且补做需要**再跑一遍全量内层 L-BFGS**（Where-A 最贵的一步），
等于把最大的一笔算力开销推到发现得最晚的时候。

**要求**：在 `BA-3-Joint` 定档 `B` 之后，把 train split 也走一遍固定-B 重拟合并按 §2.3 发布
（`ORACLE_DIR/BA-3-Joint/train`）。若主 agent 认为这属于 Where-B 的任务卡，请**明文裁定并写进排期**，
不能默认它会出现。

### B-6 · 包络定理（D3 的裁定依据）在 D2 下不成立
**文件行号**：`q3vl/where/calibrate.py:10-17`（docstring 明写 "envelope theorem: at the inner optimum
the latent contributes no first-order term, so the fixed-latent gradient is the right one"）；
相关常量 `q3vl/where/config.py:103-104` `FIT_OBJECTIVE="soft_iou_minmax"` / `CALIB_OBJECTIVE="mse"`。

包络定理成立的前提是**内层与外层是同一个目标函数** `f`：此时在 `λ*(θ) = argmin_λ f(θ,λ)` 处
`∂f/∂λ = 0`，故 `d/dθ f(θ,λ*(θ)) = ∂f/∂θ`。
但这里内层最小化 `f = 1 − softIoU`，外层却对 `g = MSE` 求梯度。
在 `f` 的最优点 `∂g/∂λ ≠ 0`，被丢掉的隐式项 `(∂g/∂λ)·(dλ*/dθ)` 是 **O(1)** 而不是 O(内层残差)。
换言之，当前梯度是"固定 latent 的代理梯度"，**不是**任何良定义的双层目标的梯度。

NOTES §四 D2 已诚实写出"内外层目标不同，双层优化不是严格一致的"，但 `calibrate.py` 的
docstring 仍以包络定理为正确性依据，而主 agent 对 D3 的裁定也写着"（envelope theorem）"。
裁定所依据的理由不成立，需要重新裁一次。

**要求**（二选一，都很便宜）：
- (a) `CALIB_OBJECTIVE = "soft_iou_minmax"`，内外层统一 —— 恢复包络论证，且 §5.5 已明文用 softIoU，
  D2 的红线顾虑按主 agent 裁定已不适用（我同意这个裁定，见 §十）；
- (b) 保留 MSE，但把 docstring 里的包络定理论证删掉，改写成"固定-latent 代理梯度"，
  并在 REPORT 里把它列为方法学限制。
**不允许同时保留"目标不同"和"包络定理保证正确"这两句话。**

---

## 九、NIT 清单

| # | 位置 | 问题 |
|---|---|---|
| N-1 | `q3vl/where/basis.py:5` | docstring 写 `3*tanh((w0 + <phi,w_dir>) * alpha / 3)`，与协议/实现的 `3*tanh((w0 + alpha*<phi,w_dir>)/3)` 不等价（前者把 `alpha` 也乘到了 `w0` 上）。**代码是对的，注释错**——但这类注释正是下一个人照抄的来源 |
| N-2 | `q3vl/where/upsample.py:84-89` | `ChannelOrderError` 只守通道轴；`sem.reshape(64,1,h,w)` 折进 batch 轴可绕过，得到的正是被禁的逐通道上采样。建议加 batch 语义断言 |
| N-3 | `q3vl/where/scripts/extract_maskviews.py` vs `q3vl/where/pipeline.py` | 发布出来的 maskview shard **没有任何消费方**：`run_calibration` 走 `WhereADataSource`，每次都用 `MaskResolver` 重新解码 `.cgt.png`。`extract_maskviews` docstring 自称的理由（"避免每 epoch 重解码 4 次"）与实际链路矛盾。要么让 pipeline 读 shard，要么删掉这个作业 |
| N-4 | `extract_maskviews.py:89-92` vs `pipeline.py:88,114` | 宽高比不符时，打包作业**丢弃**样本，在线 pipeline 只记进 `diag` 不丢。两条链路population 不同 |
| N-5 | `q3vl/where/preflight.py:222,227-228` | (a) `n_tokens=512` 形参未使用；(b) `load_vision_tower` 抛 `RuntimeError` 会穿透 `run_where_a_preflight`，**整个 preflight 崩掉且不落 JSON**。应 try/except 成 `fail` |
| N-6 | `q3vl/where/preflight.py:165-218` | §14 项 4 明写"位置编码"，脚本用空间相干性做间接代理。建议补一条对 `fast_pos_embed_interpolate` 的直接/等变性检查 |
| N-7 | `q3vl/where/preflight.py:251` | `check_calibration_health` 新建 seeded-orthogonal projector，测的是 **BA-3 起点**的条件数与拟合成功率，不是校准后的。作为开跑前的门正确，但报告需标注，且校准后应复测 |
| N-8 | `q3vl/where/phi.py:160,267` + `preflight.py:277` | 报的是 **Gram** 的条件数（矩阵条件数的平方）；1e10 阈值 = `phi` 的 1e5。需标口径 |
| N-9 | `experiments/.../where_a/NOTES.md` §二 / §3.3 | 多处数字与归档 JSON 对不上：NOTES "4 个样本 / 8 个 latent" vs `preflight_where_a_cpu_dryrun.json` `limit=3, n_samples=3, n_latents=6`；"design cond 中位 17.8" vs 20.1；"phi cond 中位 8.6e4 / 最大 1.5e5" vs 8.4e4 / 8.7e4；"相干性 0.0209 vs 0.0089" vs 0.0187 / 0.0085；"V_where upscaled 16.7%" vs `mask_validation_V_where.json` `upscaled_pct = 15.0`；"软边中位 0.204" vs 0.2334。归档证据被一次更小的重跑覆盖了。**结论方向不受影响，但"报告数字必须能在交付文件里查到"是硬要求** |
| N-10 | `q3vl/where/config.py:90` | 注释 "FIT_N_RANDOM = 6 → 8 starts"；实际 `build_starts` 出 `n_random + 3 = 9` 个种子，band 再 ×2 极性 = 18 个起点 |
| N-11 | `q3vl/where/calibrate.py:216-226` | `p10 = xs[int(0.10k)−1]`、`p90 = xs[int(0.90k)]` 不对称（k=100 时取第 10 与第 91 顺序统计量）。§5.6 有 p10 门，需钉死口径 |
| N-12 | `q3vl/where/scripts/run_calibration.py:159` | `if not oracle_root.exists(): pack_oracle(...)` —— 目录已存在时**静默跳过发布**；同一仓库的 `extract_maskviews.py:55` 对同样情况是硬报错。按 `CLAUDE.md`"清空重跑前先备份"，应统一为硬报错 |
| N-13 | `q3vl/where/scripts/run_calibration.py:130` | 一次性物化全部 `V_where` 样本；`WhereASample.fpre` 停留在视觉塔所在的 **GPU** 上（`pipeline.py:133` 的 `.float()` 不搬家），224 样本约 1.4 GB 显存（放开 D1 则 2.5 GB）。建议流式 |
| N-14 | `q3vl/where/packing.py:129-145` | §2.3 点名"basis 元数据"属 shard 契约，实现写裸文件 + digest。可接受，但要在 REPORT 里写成显式豁免 |
| N-15 | 全局 | **内层 L-BFGS 的 GPU 吞吐没有任何实测**。NOTES 只给了 CPU 的 ~2 s/图/readout；按 42,752 样本 × 2 readout 推是 ~47 CPU-小时/臂 × 4 臂。§11 排期无法核。建议把"1 GPU 上的内层拟合吞吐 + 单 batch 显存"列为 S2 的**必交输出**，而不是 PREFLIGHT_WHERE_A_PENDING.md 里的"补充建议" |
| N-16 | `q3vl/where/config.py:69-74`（D5） | `radius_low=2 / eps=1e-3` 标注"E2 先例"，但 E2 是**全分辨率逐通道** `r=32` 的用法（`prep_data.py:149`），正是 §4.2 现在禁止的顺序。先例不可迁移，见 B-4 要求 3 |

---

## 十、对主 agent 已有裁定的意见

**D1（`winner_confidence=low` 是否进 Where-A）—— 建议改判为"训练放开、headline 仍只用 normal"。**
我的独立抽样里有 23 个 `low` 样本（`V_where` 12 / `train` 11），在定位、USTAR 头比对、sha256、
PIL 模式、宽高比、以及 `corr(|I_tar−I_in|, mask)` 上与 `normal` **无法区分**
（train 那 24 个 normal/low 混合样本的 corr 中位 0.818）。
机理上也支持：`winner_confidence` 排的是"**哪个候选胜出**"，而 Where-A 的标签是"**候选区域在哪**"，
两者正交；`CLAUDE.md` 的纪律原文是"low 不进 SFT 主训与评测 GT"，Where-A 的 mask 既不是 SFT 主训
也不是评测 GT。代价很实：保守默认要扔掉 43% 的 local train（75,544 → 42,752）。
**建议**：校准训练用 `--include-low`，`V_where` headline 只用 normal，low 单独一层报。
**约束**：这个放开**只对 Where-A 的 basis 校准生效**，不得顺势流进 Where-B / What 的评测 GT。

**D2（fit=1−softIoU、B 训练=MSE）—— 同意"红线属旧战役语境"的判断，但请顺手把 B-6 一起裁了。**
§5.5 明文用 softIoU 作 Where 主 loss，这条已经把"IoU 禁当优化目标"在本轮的适用面讲清楚了。
问题不在红线，在**内外层不一致**导致 D3 的理由失效（B-6）。既然红线顾虑已解除，
我倾向 **`CALIB_OBJECTIVE = "soft_iou_minmax"`**：一行改动，恢复包络论证，并让四个臂的
外层目标与它们被比较的指标口径一致。若坚持 MSE，请删掉包络定理那段话。

**D3（内层 L-BFGS）—— 方案本身同意，理由需要换，成本需要实测。**
选 (b)"每 batch 用当前 B 现拟 latent"在方法上是对的：(a) 的离线 latent 表在 `B` 走第一步之后就失配，
1 epoch 内每个 latent 只更新一次等于没更新。但 (b) 的正确性依据不是包络定理（B-6），
而且它是 Where-A 的**唯一墙钟瓶颈**却没有 GPU 实测（N-15）。建议：保留 (b)，S2 必须交吞吐数。

**D4–D9 —— 无异议**，但 D4（在 `F_pre` 网格上拟合）必须与 B-4 绑定裁决：
天花板在低分辨率上测，交付掩膜在高分辨率上出，中间那一次 guided upsample 目前是完全未验证的黑箱。
在 B-4 关闭之前，任何"Where-A 天花板 = 0.96 soft-IoU"的说法都只对低分辨率成立，不能写进 REPORT 的结论行。

---

## 十一、明确判 pass 的部分（不需要改）

- **Phi-64 / `s_low` / 符号规范化 / 两个 readout 的全部参数域**：逐符号对照通过（§一）。
  `bounded_sigmoid` 是唯一的有界映射入口，全代码无裸 `exp` 参数化；`mu` 网格是常量不是 Parameter。
- **merge-order unshuffle**：`fpre.py:51-66` 与 `tests/test_fpre.py::test_unshuffle_matches_the_real_processor`
  ——用真实 `Qwen2VLImageProcessorFast` 喂自编码图反查布局，**并断言朴素 reshape 必须失败**
  （防止测试自身退化成同义反复）。这是本次审阅里质量最高的一个测试。
- **`inv_freq` 非持久 buffer 的加载陷阱**：`fpre.py:148-152` 明确不用 `meta + to_empty()`，
  并在返回前检查所有 buffer 有限。这是一个真实的静默错误源，实现方主动堵住了。
- **`BA-0-Fixed` 永不训练**：无 optimizer、`requires_grad_(False)`、单测断言权重逐位不变。
- **mask 数据回连链路**：48 个独立抽样（含 USTAR 头逐字节比对）100% 通过（§五）。
- **模型输入无 I_tar 泄漏**：独立实测 `MAE(baked, I_in)` 比 `MAE(baked, I_tar)` 小 5–40 倍。
- **§14 项 6 的四项**（`w_dir` 符号 / 单位范数 / `alpha>0` / readout 边界）全覆盖，含 `canonical` 标志的交叉校验。
- **`WA-P4b` 的检查逻辑**：比对权重而非激活，是**更强**的冻结泄漏探测器；跳过的两个前缀与
  `VISUAL_TRAINABLE_PREFIXES` 精确一致；`SFT_CHECKPOINTS` 与 `job.marker` 的 `protected_steps` 对得上；跑在 CPU 上。
- **97 个 CPU 单测**：我在 `CUDA_VISIBLE_DEVICES=""` 下复跑，**97 passed / 35.5 s**，与 NOTES 一致。
- **交付纪律**：NOTES 的"实施前核实记录 V1–V9"、"待主 agent 决策 D1–D9"、
  PREFLIGHT_WHERE_A_PENDING.md 的 S1→S4 顺序与 D-20 四步提交，全部按 `CLAUDE.md` 要求写到位，
  没有静默拍板。`run_where_a.sh` 的 `submit()` 正确实现了 `rm -f` → `ps -p $PID` → `tail` → `job.marker`，
  且注释明确禁用 `pgrep`。**这一块是本次交付的亮点**。

---

## 十二、放行条件

| 阶段 | 条件 |
|---|---|
| **S1**（maskviews 全量打包） | 先裁 D1（决定样本量），并处理 N-3/N-4（否则打出来的 shard 无人消费，且与在线 pipeline 的 population 不一致）。Base SFT 结束后可跑 |
| **S2**（GPU preflight） | 必须先补 **B-4 要求 1+2**（真实高分辨率检查 + `s_hi` 域断言）与 **N-5b**（异常不吞掉 JSON）。补完可跑；同时交 N-15 的吞吐数与 B-4 要求 3 的 D5 扫描 |
| **S4**（四臂正式校准） | **B-1 / B-2 / B-3 / B-5 / B-6 全清**，且 S2 的 `WA-P4b` 必须是 **PASS**（不是 SKIP） |

---

> 复现本审阅的独立脚本：`indep_mask_check.py`（scratchpad，未入库；只用 stdlib + numpy + PIL，
> 不 import `q3vl.where.*`）。核心命令：
> `python indep_mask_check.py V_where 24 5` / `python indep_mask_check.py train 24 4001`。
> B-4 的两组数字由 `q3vl.where.upsample.guided_upsample` 与 `q3vl.where.readout.apply_readout`
> 在 float64 下直接算出，未改动任何代码。

---
---

# 复审（2026-08-05 晚）：针对 6 个 BLOCKER 修复的聚焦审阅

> 范围：只审修复 diff 与其单测，不重审全量。仍然只读、未使用 GPU
> （Base SFT 仍在跑，PID 3395226/3395227，各占 54 GB，checkpoint 已到 1500）。
> 被审快照：commit `a2c2427` + `basis.py` / `tests/test_fpre.py` 两处未提交改动。
> 交付物侧：`NOTES.md` 新增《四之二 逐项回应》与 D10、`PREFLIGHT_WHERE_A_PENDING.md` 增 S0/S5/S6、
> `preflight_where_a_cpu_dryrun.json` 已重跑归档。

## 复审判决

**原 6 个 BLOCKER 全部关闭（6/6 pass）。新增 BLOCKER 1 个：B-7（`r*(z)` 网格 121 ≠ 协议 §5.5 的 257）。**

**准许进入 S2（GPU preflight）与 S4（四臂正式校准）。**
B-7 落在 `make_oracle_latents.py`，该作业在 **S5**（S4 之后）才跑，因此不阻塞 S2/S4，
但**必须在 S5 之前清掉**，否则 Where-B 的 `L_curve` 接口与协议不符。
另有 3 项"S4 前应做"的廉价整改（N-19 / N-18 / N-20），见下。

我独立复跑的证据：
- `pytest q3vl/where/tests -q` → **134 passed / 52.6 s**（原 97）；
- `python -m q3vl.where.preflight --skip-model` → 2 PASS；
- `python -m q3vl.where.preflight --device cpu --dtype float32 --limit 3`（输出写我的 scratchpad，
  未覆盖交付物）→ **WA-P6a / P4c / P4a / P4d / P5 / P4e / P6b 全 PASS，P4b SKIP**（checkpoint-2488 尚未产出）。

---

## 一、逐 BLOCKER 复核

### B-1 · 被拒拟合不再驱动梯度、不再进天花板 —— **PASS**

- `oracle.py:236` `latent: Latent | None`；`oracle.py:355-361` `all_starts_failed` 分支**返回 `latent=None`**，
  零向量构造已删除。`oracle.py:242-245` `usable = (status == "ok" and latent is not None)` 是唯一闸门。
- **我 grep 了全部生产消费方，没有一个绕过 `usable`**：
  `calibrate.py:245`（step）、`calibrate.py:339`（evaluate）、
  `scripts/make_oracle_latents.py:127`、`scripts/sweep_upsample.py:88`
  ——四处都是 `if not fit.usable: ... continue` 之后才第一次触碰 `.latent`。
  `preflight.py:409` 另有 `lat = row.get("latent"); if lat is None: continue`。
  `latent=None` 这个设计本身是对的：**任何忘记检查的消费方会在第一次使用时炸掉，而不是安静地拿常数掩膜训 B**。
- 天花板侧：`calibrate.py:339-343` 被拒样本 `continue`，不进 `per[r]`；
  改为分别报 `n_ok / n_rejected / reject_reasons / fit_success_rate`。
  §5.6 的"相对逐图 oracle ≥85%"分母不再被污染。
- 单测覆盖：全拒时 `grad_norm is None` 且 B 逐位不变、被拒不进 ceiling、`usable` 语义。

### B-2 · rejection report 流式落盘 —— **PASS**

- `calibrate.py:229-266`：`step()` 返回 `fit_rows`，含**全部非 `ok` 行** + `sample_every` 每 N 步一条
  健康样本行（有对照基线，是比我要求的更好的做法）；行里带 `sample_id / step / build / winner_confidence /
  reject_reason / flags / n_starts / n_failed_starts`。
- `run_calibration.py:140-155`：`fit_rejections.jsonl` 与 `steps.jsonl` 同时打开，逐 batch 写、按 `log_every` flush
  ——是流式而非结尾一次性 dump，长任务中途被杀也留得住证据。
- `calibrate.py:426-437` `rejection_summary()` 进 `state()`（即每个 `projector_step*.pt`）与 `schedule.json`。

### B-3 · 调度按 eligible 步数建 + 终局断言 —— **PASS（一处残留风险，见 N-19）**

- `pipeline.py:187-211` `count_eligible()` 跑的是**真正的 `eligibility()`**（只读 record，不碰图像/模型），
  并返回 `skipped_reasons` 直方图与 `by_winner_confidence`；结果缓存在 `eligible_count.json`。
- `run_calibration.py:111-113` `total_steps = ceil(min(n_eligible, train_limit)/batch)`；
  `_batched` 会吐最后一个不满批，与 `ceil` 一致。
- `run_calibration.py:171-177` 跑完断言 `cal.step_count == total_steps`，不等即 `SystemExit` 并提示删缓存重数；
  `schedule.json` 落 `planned/actual/warmup/n_seen/final_lr`。
- 单测同时钉住反面：用未过滤数会复现"结尾 LR 停在峰值 30%+"。

### B-4 · 高分辨率通路进生产 + 两个静默失效封堵 —— **PASS（一处门槛缺失，见 N-20）**

四件事都做了，且我逐条独立复算：

1. **通路不再是死代码**：`WhereASample.mask_hi/guide_hi`（`calibrate.py:96-118`，半供即报错）、
   `WhereADataSource(attach_hi=...)`、`evaluate()` 调 `evaluate_latent(...)` 走真实 guided upsample，
   `run_calibration.py:182` 在评测前置 `source.attach_hi = True`。
2. **域声明 + clamp 前统计**：`config.py:54-62` 明写 producer/consumer 两侧的域；
   `upsample.py:126-138` 先算 `raw_min/raw_max/frac_below/frac_above/frac_out_of_domain`，**再** clamp。
   clamp 是 `UpsampleConfig` 上的**整臂常量**（`clamp_domain` + `domain`），不是逐图 —— 未触红线。
3. **CBand12 改 logsumexp**。我用 float64 独立扫了 σ：

   | σ | `max abs(eps 形 − logsumexp 形)`，z∈[−3,3] |
   |---:|---:|
   | 0.025（下界） | **9.78e-1** |
   | 0.050 | 4.33e-3 |
   | 0.100 | 8.98e-8 |
   | 0.150 / 0.200 / 0.300 | ≤ 5.9e-9 |

   即"良态区数值等价"的说法**在 σ ≳ 0.10 成立**，而在 σ ≤ 0.05（拟合完全可以走到的区间）
   两者是不同的函数 —— 这正是修复的着力点。塌陷探针复算：
   `eps` 形在 z ∈ {0.0, 3.3, 4.0, ±6.6} 处恒为 0（尽管所有 `c_i = 0.982`），
   `logsumexp` 形全部返回 0.982014。mirror 恒等式在 logsumexp 下仍精确（最大误差 **4.44e-16**）。
   `softmax(log_o − 0.5((z−μ)/σ)²)` 确实是 `g_i/Σg_j` 的稳定求值，`logsigmoid` 避免了 `o` 下溢产生 `−inf`；
   `m = Σ c_i·resp_i` 是 `c` 的凸组合，因此恒在 (min c, max c) ⊂ (0,1)。**数学等价性成立。**
4. **防 revert 的 pin 测试是真的双向**：`test_eps_form_collapses_to_zero_at_the_sigma_lower_bound`
   钉死旧式的塌陷数字（删掉 `"eps"` 模式即失败），`test_logsumexp_is_the_default`
   钉死默认值（切回 `"eps"` 即失败）。两个方向都拦得住。

**真实数据上第一次有了交付分辨率的数字**（我复跑的 CPU preflight，n=3；归档的 n=4 版本数字一致）：

| readout | headline_low 中位 | headline_hi 中位 | low→hi 中位落差 | clamp 前 s raw | 越域像素 |
|---|---:|---:|---:|---|---:|
| band | 0.8399 | 0.8222 | **0.0177** | **[−4.577, +2.569]** | 最大 **0.542%** |
| cband12 | 0.8612 | 0.7874 | **0.0738** | [−0.455, +2.306] | 0% |

这组数直接印证了初审 B-4 的判断：**低分辨率天花板比交付分辨率高 1.8–7.4 个 IoU 点**，
band 的 `s` 确实被 guided filter 推到 −4.58。现在 `headline_hi` 与 `headline_low` 并排出报表，
`s_domain` 进 `per_readout`，`WA-P4e` 进 preflight，`sweep_upsample.py` 成为 D5 的定档作业。

### B-5 · train split oracle latent —— **接口两点 PASS、z 网格 FAIL（→ B-7）**

- `make_oracle_latents.py:89-94`：`B.npy` 不存在即 `SystemExit`，错误信息写明"必须在 S4 冻结 B 之后跑，
  绝不能对着未校准 basis 拟 latent"。**拒绝逻辑正确**，且默认路径指向 `BASIS_DIR/<arm>/B.npy`，
  报告里另存 `basis_sha256_of_npy`，可事后核对用的是哪一版 B。
- 输出走 §2.3 shards（`pack_oracle` + `verify_published(n_random=64)`），目标已存在即硬报错。
- `run_where_a.sh` 增 `oracle-latents` 步，顺序写成 `... -> calibrate x4 -> oracle-latents` ✓。
- **但 z 网格不对**，见 B-7。

### B-6 · 包络定理论证 —— **PASS（缺运行期护栏，见 N-18）**

`config.py:135-136` `FIT_OBJECTIVE = CALIB_OBJECTIVE = "soft_iou_minmax"`；
`calibrate.py:17-26` 的 docstring 重写得准确：明确写出"该论证**只**在 `CalibConfig.objective` 同时驱动内外层时成立"，
并把"若两层用不同 `f`/`g`，被丢掉的 `(dg/dλ)(dλ*/dB)` 是 O(1)"作为反面写进注释。
`config.py:127-134` 记录了这次改判的理由。表述正确，不再有"目标不同 + 包络定理保证正确"并存的矛盾。

### 附带项 · `build_starts` 的 LinAlgError 降级 —— **PASS，但有两处安静边缘（→ N-21）**

我实测了这条新路径：

- `np.linalg.lstsq` 遇 NaN **确实抛 `LinAlgError`**（`** On entry to DLASCL ...` 只是 LAPACK 在抛之前打到 stderr 的噪声，
  不是"静默返回 NaN"）—— 所以 try/except 是真起作用的，不是装饰。
- 端到端：NaN 污染的 `phi` 走 `fit_latent` → `n_starts` 由 18 掉到 16、
  最终 `status=rejected / reason=all_starts_failed / usable=False`。**降级路径通向一条被记录的拒绝，不是一个坏 latent。**
- 秩亏 `phi` 不抛（lstsq 取最小范数解），行为正常。

两处安静边缘见 N-21。

---

## 二、新增 BLOCKER

### B-7 · `r*(z)` 的 z 网格是 121 点，协议 §5.5 写的是 257 点

**文件行号**：`q3vl/where/scripts/make_oracle_latents.py:58`
```python
CURVE_Z = np.linspace(CBAND_MU_LO, CBAND_MU_HI, 121)
```
**规格条文**：协议 L305 —— `L_curve = mean_z |R(z;rho_pred) - r*(z)|, z=linspace(-3,3,257)`。

`NOTES.md` 的回应表（B-5 行）也明写"固定 **121** 点 z 网格"，说明这是有意选的数，不是笔误 —— 但协议把
257 写死了。脚本自己的 docstring 说这份采样是"让 Where-B 的 `L_curve` 不需要 readout 代码，直接做向量差"，
也就是**要被直接消费**的；121 点的向量喂不进 257 点的 `L_curve`，Where-B 只能二选一：
插值（把误差引进一个监督目标）或偏离 §5.5。

修法一行：`121 → 257`。载荷里 `rho_raw` 仍在，所以没有信息损失，纯粹是接口口径。

**顺带**：`curve_of` 用的是默认的 `logsumexp` 归一化（D10），但发布载荷里**没有 `cband_normalization` 字段**。
Where-B 必须用同一约定重算 `R(z;rho_pred)`，否则 `L_curve` 两侧不是同一个函数。请把该字段写进
`oracle.json` 与 `report`（见 N-25）。

---

## 三、复审新增 NIT

| # | 位置 | 问题与建议 |
|---|---|---|
| N-18 | `q3vl/where/calibrate.py:161-170` | B-6 目前**只靠两个模块常量恰好相等**加一个"只验默认值"的单测（`test_calibrate.py:311-317`）维持。`CalibConfig(objective=...)` / `fit_cfg=FitConfig(objective=...)` 是公开入参，四个脚本都在显式构造它们。建议在 `Calibrator.__init__` 加运行期断言 `cfg.objective == cfg.inner_fit.objective`，并在 `fit_sample` 里校验传入的 `fit_cfg.objective`。**S4 前做，改动 3 行。** |
| N-19 | `q3vl/where/pipeline.py:104-116` vs `pipeline.py:187-211` | `count_eligible()` 只跑 `eligibility()`，而 `prepare()` 还会因 **aspect_mismatch**（N-4 的修复引入）再丢样本；`resolver.resolve/load` 的异常至今**未被 try/except**（`extract_maskviews.py:83-86` 是包了的）。后果：75,544 个样本里只要有 1 个对不上，B-3 的终局断言就在**整整一个 GPU epoch 跑完之后**才 `SystemExit`（`projector_final.pt` 已存，评测与发布阶段全丢）；一个读不出的 `.cgt.png` 更是直接 traceback 打断 epoch。建议：(a) `prepare()` 把 mask IO 异常也转成 `self.rejections` 记录；(b) 终局断言改成"缺口能被 `source.rejections` 逐条解释就记 warning + 落 `schedule.json`，解释不了才 `SystemExit`"。**S4 前做。** |
| N-20 | `q3vl/where/preflight.py:390-395`、`scripts/sweep_upsample.py:127-130` | `WA-P4e` 只在**关掉 clamp** 时才因越域判 fail；默认配置下 `frac_out_of_domain` 与 `median_drop_low_to_hi` 只报不判，检查在默认配置下**永远不会 fail**。`sweep_upsample` 的 `best = max(..., hi_soft_iou median)` 同样完全忽略越域比例，可能选出"IoU 好但大量像素被 clamp"的配置。建议：S2 的 sweep 里预注册两个阈值（`frac_out_of_domain` 中位数上界、`hi−low` 落差下界），把选择改成字典序（先过阈值再比 IoU），并让 `WA-P4e` 对阈值判 fail。**S2 期间定档。** |
| N-21 | `q3vl/where/oracle.py:167-176, 189-194` | informed 起点被降级时**没有 flag**，只有 `n_starts` 从 18 变 16 这一个整数暴露它；若 `phi` 只是轻度退化（informed 失败但随机起点成功），fit 会以 `status="ok"` 正常返回，`flag_counts` 里什么都没有。另外 `_radial_start` 在 NaN `phi` 上**不抛异常**，会返回 `(nan, nan, u)`，`_s_stats` 的有限性兜底只护住 `mu/sd`，`w0/alpha` 仍是 NaN，最终以"起点失败"计数收场。建议：起点构造后统一做有限性检查，丢弃时追加 `flags.append("informed_start_unavailable")`，这样 `rejection_summary().flag_counts` 能聚合。 |
| N-22 | `NOTES.md` §3.3 vs `preflight_where_a_cpu_dryrun.json` | N-9 已基本修好（相干性 0.01733/0.00745、design cond 27.9/95.7、phi cond 8.57e4/1.18e5、min α 0.187、8/8 latent、band raw [−4.577,+2.569] 0.542% 我逐字段核对**全部对上**）。**但新加的吞吐字段又复发同一问题**：NOTES 写"1.697 s/拟合 → 40.3 / 71.2 CPU-小时每臂"，归档 JSON 写 `s_per_fit: 1.889`、`projected_hours_per_arm_42752: 44.87`、`_75544: 79.29`。请以 JSON 为准改 NOTES。 |
| N-23 | `NOTES.md` §3.3 | headline 那两行的 p10/median/p90 是在 **n=2** 上算的（4 个样本里只有 2 个 `normal`；JSON 的 `summarize` 带 `n` 字段，NOTES 没写）。最近秩约定下 k=2 时 median==p10，容易被误读成"稳定"。请把 n 标出来。 |
| N-24 | `scripts/make_oracle_latents.py:99, 111, 123` | 脚本第 99 行 `projector.requires_grad_(False)`，但第 111 行 `Calibrator.__init__` 对 BA-3 又 `requires_grad_(True)` 并建了一个**永远不会被 step 的 AdamW**；`cal.phi_for(sample)` 在 `no_grad` 之外，每个样本都建一次计算图。B 事实上不会被改（从不调用 `cal.step()`），但脚本 docstring 的"`B` ... never touched"应当被**强制**而不是靠约定：加 `assert cal.optimizer is None or not cal.trains_projector` 之类的护栏，或把 `phi_for` 包进 `torch.no_grad()`。 |
| N-25 | `scripts/make_oracle_latents.py:144-148` | 发布的 `oracle.json` 载荷缺 `cband_normalization` 字段（D10）。`r*(z)` 是在 `logsumexp` 约定下采的，Where-B 必须用同一约定算 `R(z;rho_pred)`。请写进载荷与 `report`。 |
| N-26 | `tests/test_readout.py:178-191` | 等价性单测只在 `sig_raw ≈ 2.0`（σ 接近上界）上验。我实测**分界在 σ ≈ 0.10**（σ=0.05 已差 4.3e-3）。建议把 σ=0.10 这个边界点加进用例，这样将来动 `CBAND_SIG_LO/HI` 会被立刻发现。 |
| N-27 | `scripts/sweep_upsample.py:66-76` | 默认用**未校准的 seeded B** 拟 latent 来定 D5。作为 S2 的前置门可以接受，但 BA-3 校准完之后 `s` 场的形状会变，建议 S4 结束时用 `--basis .../BA-3-Joint/B.npy` 复跑一次确认推荐值仍成立，再把 `GUIDED_PARAMS_PROVISIONAL` 翻成 `False`。 |

## 四、初审 nit 的关闭情况

N-1（docstring 公式）、N-2（batch 轴绕过，已加断言 + 单测）、N-3（`MaskViewStore` 让 maskview shard 有了消费方）、
N-4（两条链路对 aspect 不符统一为丢弃）、N-5a/b（`n_tokens` 删除；驱动器 `try/except BaseException` + `flush()`，
单测 `test_driver_writes_json_even_when_a_check_explodes` 覆盖）、N-6（新增 `WA-P4d`，见下）、N-7/N-8（口径已写进
NOTES 与 JSON）、N-9（数字重新对齐，除 N-22）、N-10、N-11（`percentiles()` 单一实现 + 单测）、N-12（硬报错）、
N-13（流式）、N-14（豁免写进 S6）、N-15（吞吐进 `WA-P5.throughput`）、N-16（D5 先例作废 + `GUIDED_PARAMS_PROVISIONAL`）
—— **全部关闭**。

**N-6 值得单独说**：新增的 `WA-P4d-position-encoding`（`preflight.py:267-308`）不是代理指标，
而是把 `pos_embed.weight` 的双线性插值**独立重算一遍**，再经 `unshuffle_to_grid` 与
`visual.fast_pos_embed_interpolate(grid_thw)` 的真实输出逐元素比。我对照了 transformers 4.57.1 的
`fast_pos_embed_interpolate` 源码：HF 用 `linspace(0, n-1, h)` + `int()` 截断 + `clip(max=n-1)` 的双线性权重，
之后 `.view(t, h//m, m, w//m, m, -1).permute(0,1,3,2,4,5)`。preflight 的重算与前半段是同一套数学（这部分算再推导），
**但后半段的 merge-order 置换是真正独立的**：它拿一个空间结构已知的信号（位置格）把
`unshuffle_to_grid` 是不是 HF 置换的精确逆给钉死了。实测 `max_rel_error = 1.85e-7`。这正是 §14 项 4"位置编码"要的东西。

## 五、D1 落地确认

`config.py:156-164`：
```python
EXCLUDE_WINNER_CONFIDENCE_LOW = False       # training population
HEADLINE_WINNER_CONFIDENCE = ("normal",)    # reporting population
```
- 训练含 low ✓（`run_calibration.py` 的开关由 `--include-low` 反转成 `--exclude-low`，默认即含）；
- headline 只 normal ✓（`Calibrator.evaluate(headline_confidence=...)` → `headline_low` / `headline_hi`
  只取该层，`by_winner_confidence_low_res` / `_hi_res` 另出分层）；
- low 单独分层 ✓（另有 `run_calibration.py` 的 `strata.winner_confidence`）；
- 注释里写明"该放宽只对 Where-A 生效，不得流进 Where-B / What 的评测 GT" ✓ —— 与我在初审第十节提的约束一致。

**确认落地。**

## 六、更新后的放行条件

| 阶段 | 状态 |
|---|---|
| **S1** maskviews 打包 | **放行**（D1 已定；N-3/N-4 已闭环）。仍须等 Base SFT 结束 |
| **S2** GPU preflight + D5 sweep | **放行**。必交：`WA-P4b` 必须 PASS（非 SKIP）；GPU 侧内层拟合吞吐；N-20 的两个阈值预注册 |
| **S4** 四臂正式校准 | **放行**。开跑前建议先做 N-18、N-19 两处（合计十余行，避免一整个 epoch 白跑） |
| **S5** train oracle latents | **不放行，直到 B-7 关闭**（`CURVE_Z` 121 → 257，并补 `cband_normalization` 字段） |

## 七、复审判决行

**原 BLOCKER 6 项全部关闭；新增 BLOCKER 1 项（B-7，仅阻塞 S5）；新增 NIT 10 项。
准许进入 S2 GPU preflight 与 S4 正式校准；S5 须先清 B-7。**

> 复审的独立复算：`pytest q3vl/where/tests -q` → 134 passed；
> `q3vl.where.preflight --skip-model` 与 `--device cpu --dtype float32 --limit 3`
> （`--out` 指向 scratchpad，未覆盖交付物）→ 7 PASS / 2 SKIP；
> σ 扫描与塌陷探针、`build_starts` 的 LinAlgError 端到端探针均在 float64 下直接调用被审函数，未改动任何代码。
