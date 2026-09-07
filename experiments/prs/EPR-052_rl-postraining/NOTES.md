# EPR-052 NOTES

## N1（2026-09-06，ENG-2 三臂）与原任务卡的差异，及理由
- **原卡**：臂 2 = 教师全词表 + lr 5e-6；臂 3 = λ0.75 + sft_alpha 0.1 + lr 5e-6（三臂各只改一个变量，lr 保持 5e-6）。
- **实际执行**（coordinator 2026-09-06 追加裁决）：臂 2/3 的 lr 一并固定为 **1e-6**（= 臂 1 的值）。
- 依据：臂 1（lr 1e-6 + warmup 100）在 46 步时六段有序率 **46/46 步均为 8/8**（基线 opsd50k_gkd 自 step 14 起连续 0/8），
  但散度基本不动（前 10 步均值 0.2849 → 后 10 步 0.2758）。若臂 2/3 仍用 5e-6，结构崩塌会掩盖被测变量的效果。
- 代价：臂 2/3 与臂 1 之间是单变量对照；三臂与基线 opsd50k_gkd 之间差**两个**变量（lr + 被测变量），比较时须并标 lr。
  基线 opsd50k_gkd（lr 5e-6、无 warmup）的 30 步数据留档作 lr 对照行。
- 探针：臂 1 用 v1（`gkd_segment_probe.py`）；臂 2/3 起用 **v2**（`gkd_probe_v2.py`，新文件，v1 不动），补 SURVEY_STRUCTURE_DEGEN §1
  的四个诊断量（top-64 覆盖率 / 策略熵 / finish_reason 与长度分位 / rollout-vs-训练前向 logprob 差与 k3）。臂 1 无这四列。
- 运行纪律：在跑作业引用的脚本/插件一律不改，新变量一律新文件（train_opsd_full_v2.sh / train_opsd_full_localteacher.sh / gkd_probe_v2.py）。

## N2（2026-09-06）臂 2 首次启动失败签名与修复（按无人值守规则：记签名 → 重启一次）
- 签名：`ValueError: Failed to automatically match model_type for /data/runs/epr052_rl/s1f_epoch1_merged. Multiple possible types found: ['qwen3_vl','qwen3_vl_emb','qwen3_vl_reranker']`
  （`swift/model/model_meta.py` L227，经 `pipelines/train/rlhf.py` L73 `_prepare_single_model('teacher', …)` 触发）。
- 成因：`--model_type` 只作用于学生；本地教师走 `TeacherModelArguments.teacher_model_type`（`rlhf_args.py` L73），未给即自动匹配失败。
  合并目录同时匹配 qwen3_vl / _emb / _reranker 三个 model_type（首次跑 prompt 断言脚本时同一签名，当时用 `get_processor(model_type=…)` 解决）。
- 修复：`train_opsd_full_localteacher.sh` 加 `--teacher_model_type qwen3_vl`（该脚本此前从未成功跑过，直接改；未动任何在跑文件）。
- 重启：第 1 次（同签名第 2 次失败即停机报告）。失败运行留档 `arms/arm2_localteacher/run_attempt1_fail.log`。

## N3（2026-09-06）臂 2 第二次失败签名与修复
- 签名：`torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 298.00 MiB. GPU has 95.08 GiB, 94.84 GiB in use`（本地全词表教师 + PB=8）。
- 归因（本 agent 复算）：探针 v2 在全词表分支执行 `_align_vocab(s_act.float(), t_act.full_logits.float())`，一次性生成两份
  `[N, V]` fp32（N≈13,000、V=151,936 ⇒ 每份 ≈7.9 GB，合计 ≈15.8 GB）——**探针自身**引入的显存，非训练必需。
  ms-swift 侧固有增量（相对臂 1 的 API 教师）：教师权重 bf16 ≈8.3 GB + 教师全序列 logits `[8, ~3900, 151936]` bf16 ≈9.5 GB。
- 修复：新写 `gkd_probe_v3.py`（v2 不动），把覆盖率/熵/logp/散度合并为**一次分块扫描**，任意时刻只有 `[256, V]` fp32（≈156 MB）。
- 重启：臂 2 第 2 次重启（签名 #1 = model_type ValueError，签名 #2 = OOM，两次签名不同）。若 v3 下仍 OOM（= 签名 #2 第二次），按规则停机报告并建议 PB≤4。
- 预算（未实测，推导）：臂 1 峰值 61.5 GB + 教师权重 8.3 + 教师 logits 9.5 ≈ 79 GB，处于用户 65–80 GB 带的上沿。

## N4（2026-09-06）臂 2 第三次仍 OOM（同签名第 2 次）→ 按规则停机，归因更正
- 结果：探针换 v3 后峰值几乎不变（94.82 vs 94.84 GiB），OOM 仍发生。**N3 把主因归给探针是错的**，现更正：
  traceback 落在 ms-swift 自身 `gkd_loss.py:270 gkd_loss → :139 jsd_loss → :82 default_kl_div`
  （`(exp(t) * (t - s)).sum(-1)`，全词表 [chunk,V] fp32 中间量 + `_align_vocab` 的 [N,V] 副本），
  **不在** `gkd_probe_v3.py`（探针已跑完并落盘 step-0 行，其时 reserved 69.95 / alloc 63.70 GiB）。
- 规则处置：同签名（CUDA OOM）连续 2 次 ⇒ 臂 2 在 PB=8 下停机，不再自行改配置重试。
- 臂 2 唯一可用数据 = step 0 一行（全词表教师）：`div_mean 0.2248`、**`cov_topk64 0.99953`**、`entropy_mean 0.4318`、
  六段有序 8/8、finish_reason 全 stop、触顶 0、reserved 69.95 GiB。
- 建议（待用户裁决，未实施）：臂 2 改 PB=4（引入第二个变量，需并标）；或 `--gkd_logits_topk 1024` 走 top-k 分支近似全词表
  （教师 API 上限 `max_logprobs 64`，须改 server 启动参数）；或开 `--use_logits_to_keep true` 把两侧前向裁到补全区
  （SURVEY 标注多模态下正确性未实测）。

## N5（2026-09-06）臂 3 两项核实：λ 生效但 CE 锚点空转；训推 logprob 采集缺陷
### (1) λ 传参与判定方向 —— 传进去了，方向如卡所设
- args dump 实测 `lmbda=0.75`、`sft_alpha=0.1`、`seed=20260906`。
- `gkd_trainer.py` L293-298：`if self._get_random_num() <= self.lmbda: STUDENT else: DATASET` ⇒ **λ=0.75 = 75% 学生 / 25% 数据集**（方向与卡一致）。
- `_get_random_num()`（L435-448）= `random.Random(seed + global_step).random()`，逐步确定。
- 实测计数（31 行）：**DATASET 8 / STUDENT 23 = 25.8%**；DATASET 步 = 0,1,4,5,12,15,16,21，
  与本地复算 `random.Random(20260906+step).random() > 0.75` 的预测**逐步 0 处不符**。
  （coordinator 报「全部 STUDENT」与盘上文件不符，疑读到写入中途的截断内容。）
### (2) CE 锚点实际未生效 —— 原因不是 λ，是数据没有 assistant 段
- 8 个 DATASET 步全部：`n_valid=0`、`row_lens=[0]*8`、`completion_len=[0]*8`、`div_mean=None`、`finish_reason=[None]*8`。
- 成因：OPSD 数据行只有 user 段（回复靠 on-policy 生成），`resample_encode_failed_inputs(strip_response=False)` 保留的是空回复 ⇒ 标签全 -100。
  于是 JSD 项 = `total*0`（`_compute_jsd_loss` num_valid==0 分支），`sft_alpha * outputs_student.loss` 作用在全 -100 批次上。
  这 8/31 步是**零监督步**，臂 3 因此不能算作「CE 锚点」的检验。
- 要真正启用：数据行需补 `assistant` = GT 六段 CoT（DATASET 分支即变成对 GT 的 teacher-forced CE）。**未擅自改数据**，等裁决。
- 日志里的 `nan` 仅是 `logging_nan_inf_filter=True` 参数名，非损失 NaN；探针 v4 已加 compute_loss 钩直接记录实际 loss 与有限性。
### (3) 训推 logprob 差为何全是 length mismatch
- 根因：`OnPolicySample.rollout_logprobs` 是**每个 choice 一条的嵌套列表**（`rl_core/data.py` L231-234；n=1 ⇒ 外层长度 1）；
  探针 v3 `list(s.rollout_logprobs)` 取到外层 ⇒ 长度 1，与该行 token 数不等 ⇒ 每步 `rows_matched 0 / rollout_lens [1,1,…]`。
- 修复：`gkd_probe_v4.py`（新文件；v3 不动，臂 3 进程已加载 v3 不可改）展开内层；另加 F2 loss 钩、F3 `no_supervised_tokens` 标记。
  已在容器内 import 测试通过（三个钩子就位）。**从下一臂起生效**。

## N6（2026-09-06）臂 3b 首次启动的静默错配（自查发现，未产生数据即修正）
- 现象：`run_arm.sh` 的 `docker compose exec` 只转发它显式列出的 `-e` 变量，宿主侧 `DATA_FULL=...` **没有进容器**；
  实测容器内命令行为 `--dataset /data/runs/epr052_rl/data/opsd50k/opsd_train.jsonl`（旧数据，无 assistant 段）。
- 若不查，臂 3b 会与臂 3 完全同配置却被记成「CE 生效版」——静默错配。
- 修正：`run_arm.sh` 增加 `-e DATA_FULL -e LR` 转发；启动后立即复核 run.log 的 `--dataset` 实际路径（新增例行检查）。
- 该次启动运行 ~1 分钟、未产出探针行，目录已删除重跑。

## N7（2026-09-07）两处读数订正 + 臂 3b 实际步数
- **步数**：`run_arm.sh` 用 `wc -l` 数探针行判停；探针 v4 每步写 **2 行**（seg + loss），故臂 3b 实跑 **30 优化步**（非 60）。
  已改为只数非 `v4-loss` 行。臂 1/2/3 用 v1/v3（每步 1 行），步数不受影响。
- **「散度不降」是窗口假象**：先前用「前 10 步 vs 后 10 步」跨不同总步数比较。改用同窗口（步 0-9 vs 步 20-29）后：
  baseline −43.0%（六段 87/240，结构崩）、arm1 −17.6%（240/240）、arm3 −12.6%（176/176）、arm3b −8.9%（240/240）——**四臂在该窗口内散度都在降**。
- **10 步块轨迹显示 U 形**（lr 1e-6 组）：arm1 0.2849/0.2726/**0.2349**/0.2785/0.2910/0.2964；arm3 0.2761/0.2627/**0.2412**/0.2603/0.2803/0.2928；
  arm3b 0.2726/0.2665/**0.2482**（仅 30 步）。即步 20-29 达最低后回升到起点附近或以上。baseline（lr 5e-6）则单调降 0.2162/0.1530/0.1232，**但结构同时崩**。
- 口径提醒：此处「散度」是在**学生自己每步新采样**的序列上算的，不是固定分布上的训练损失；散度下降既可能是「学生写出教师认可的文本」，
  也可能是「学生退化成易预测文本」（baseline 即后者）。故单看散度不足以判成功，须与六段有序率并读。

## N8（2026-09-07）臂 4（lr 3e-6）判定 = (a) 真崩，且是**数值崩（权重 NaN）**，非渐进结构漂移
判据（全部来自盘上记录，非推测）：
- **step 0 正常**：loss 0.19782、`entropy_mean` 0.36301、`cov_top64` 0.99984、六段有序 8/8、补全长度 1489–1694、`finish_reason` 全 stop、
  训推 logp 差 8/8 行对齐（mean −0.00047）。
- **step ≥1 学生权重 NaN**：`entropy_mean` = NaN 出现在 **18/19** 条记录。熵由**学生训练前向 logits** 直接算，
  其为 NaN ⇒ 学生 logits 为 NaN ⇒ 学生权重已 NaN（与「文本退化但权重正常」可区分）。
- **trainer 自身返回的 loss = NaN**（v4 loss 钩，18 条 step≥1 记录全部 `finite=False`）——不是探针算出来的量。
- **补全全部顶到上限**：18/18 条记录 `row_lens` 恒为 `{2048}`（= `max_completion_length`），`stage_orders` 全空 ⇒ 生成侧（vLLM server，
  每步同步 trainer 权重）也已拿到 NaN 权重。
- 教师侧未受影响：`cov_top64` 仍 0.978–0.980（教师是 gpu0 上独立冻结 server）。
- **`truncated_rows`/`finish_reason` 为 None 的原因是探针缺陷不是漏截断**：这些字段在 `probe_error`
  （`TypeError: must be real number, not NoneType`，NaN 走到 `_q()` 分位格式化）之后才赋值，异常使其未写入；`row_lens` 已足以判定触顶。
- **与 baseline(lr 5e-6) 的失效模式不同**：baseline 是 loss 有限、六段有序率在 10–14 步间逐步从 8/8 掉到 0/8（渐进结构漂移）；
  臂 4 是**单个优化步后直接 NaN**（此时 warmup 使有效 lr 仅 3e-6×1/100 = 3e-8），故「lr 越大越崩」不足以解释，
  数值路径（bf16 权重 + 8-bit AdamW + 该步梯度）需另测。**未下结论**。
- 处置：按无人值守规则记停点 = **step 19**（人工停机，非 early-stop 判据触发），lr 3e-6 归入崩塌类，跳到臂 5。

## N9（2026-09-07）**推翻 N8 的臂 4 判定：NaN 由探针引入，不是 lr 3e-6**
- 控制实验 `ctrl_probe_v4`：与臂 1 **逐参数相同**（args dump 477 项，仅 `external_plugins` 与输出路径不同；`top_p=1.0`、`learning_rate=1e-06`、`lmbda 1.0`、原数据集），
  只把探针从 v1 换成 v4，跑 5 步 ⇒ **step 0 正常（loss 0.29901、六段 8/8、熵 0.41436、补全 1711），step 1–4 全部 loss=nan / 熵=nan / 六段 0/8 / 补全恒 2048**。
- 由此：**臂 4（lr 3e-6）与臂 5（top_p 0.9）的 NaN 都不能归因于其被测变量**；两臂的结论作废，其数据只能作「v4 探针在场时的现象」留档。
  N8 中「lr 3e-6 归入崩塌类」的判定**撤回**。
- 已核实的对照关系：v1 + λ=1（臂 1，60 步）无 NaN；v3 + λ=0.75（臂 3，60 步）无 NaN；v4 + λ=0.75（臂 3b，30 步）无 NaN；
  v4 + λ=1（臂 4 / 臂 5 / 控制臂）**step 1 即 NaN**。缺的对照是 v3 + λ=1，正在跑（`ctrl_probe_v3`，5 步）。
- 探针 v3→v4 的功能增量只有三处只读改动（rollout logprob 取内层、compute_loss 记录值、n_valid=0 标记），
  机制未明；**在查清前，所有正式臂一律回退用 v1 探针**（v1 已有 60 步 λ=1 无 NaN 记录）。

## N10（2026-09-07）零变量对照结论：**探针 v4 的新钩子导致 NaN，v3 无罪**
两个对照除 `external_plugins` 与输出路径外，与臂 1 的 args dump 逐项相同（477 项；`lmbda 1.0`、`learning_rate 1e-06`、`top_p 1.0`、原数据集）：

| 对照 | 探针 | step0 | step1 | step2 | step3 | step4 |
|---|---|---|---|---|---|---|
| `ctrl_probe_v3` | v3 | six 8/8, div 0.27145, 熵 0.36402, 补全 1691 | 8/8, 0.23744, 0.36558, 1723 | 8/8, 0.23563, 0.39094, 1695 | 8/8, 0.34033, 0.34970, 1665 | 8/8, 0.28644, 0.29168, 1676 |
| `ctrl_probe_v4` | v4 | six 8/8, div 0.29901, 熵 0.41436, 补全 1711 | **0/8, nan, nan, 2048** | 0/8, nan, nan, 2048 | 0/8, nan, nan, 2048 | 0/8, nan, nan, 2048 |

`ctrl_probe_v3` 五步 top-64 覆盖率 0.99922/0.99977/0.99961/0.99922/0.99969，`ctrl_probe_v4` 的 trainer 返回 loss 在 step1-4 全 `finite=False`。
**判定**：NaN 与 lr、top_p 均无关，由探针 **v3→v4 的增量钩子**引入。v4 相对 v3 只有三处：
(b1) 新增 `compute_loss` 包装（**首要嫌疑**）、(b2) `_rollout_samples` 里 rollout logprob 取内层（只读）、(b3) `n_valid==0` 标记（只读）。
**受影响数据**：臂 4（lr 3e-6）、臂 5（top_p 0.9）全部作废，只作「v4 在场时的现象」留档；N8 的「lr 3e-6 崩塌」判定已在 N9 撤回。
臂 3b（v4，λ=0.75，30 步无 NaN）说明该缺陷在**全 STUDENT 批次**下才稳定触发，混合 DATASET 批次时未触发——机制待定。
**处置**：查清前正式臂一律用 v1/v3 探针；v5（建在 v4 之上，臂 4b 要用）在修好前不得使用。

## N11（2026-09-07）二分定位到 b1 = `compute_loss` 包装；修复版 = 探针 v7
- 二分（全部为「臂 1 配置 + 只换探针」的 5 步零变量对照）：

| 对照 | 探针内容 | step1 起 |
|---|---|---|
| `ctrl_probe_v3` | v3（无 b1/b2/b3） | 正常（六段 8/8） |
| `ctrl_probe_v4a` | v4 去掉 b1（= v3+b2+b3） | **正常**（8/8, 7/8, 8/8, 8/8；div 0.226/0.195/0.232/0.377/0.260；训推 logp 差 8/8 行仍采到） |
| `ctrl_probe_v4` | v4（含 b1） | **NaN** |
⇒ 元凶 = **b1：包装 `GKDTrainer.compute_loss`**。注意钩内已按建议只做 `float(loss.detach())`（无二次反传、无 no_grad 外读 logits），仍触发 NaN；
机制未查明（候选：ms-swift 对 `compute_loss` 的 `@profiling_decorator` 属性丢失、或 HF 对该方法的签名/属性内省），**未下结论**。
- 修复路线（不再包装任何损失路径函数）：
  - v6 = v4a + `optimizer.register_step_pre_hook` ⇒ 干净但**梯度行 0 条**：ms-swift 不调用 `create_optimizer`（优化器在 `trainers/mixin.py:1108 create_optimizer_and_scheduler` 内建），钩子未触发。
  - **v7 = v4a + `TrainerCallback.on_log`**（transformers 官方扩展点；本版本无 `on_pre_optimizer_step`）⇒ 干净且每步拿到 HF 自算的 `loss` / `grad_norm` / `lr`。
    5 步零变量对照：六段 8/8×3，div 0.33096/0.29351/0.27980，loss 0.33096/0.29351/0.27980（与 div 一致，β=0 下 loss==散度），
    **grad_norm 10.1875 / 12.0 / 6.09375**（有限），lr 1e-8/2e-8/3e-8（warmup 生效）。
- 连带修：`run_arm.sh` 的步数与 early-stop 判据改为只数含 `six_complete_rows` 的行（v4+ 每步会多写 loss/log 行）。
- 作废数据改名留档：`arm4_lr3e6_VOID_probe_v4/`、`arm5_topp09_VOID_probe_v4/`。

## N12（2026-09-07）臂 4/5 重跑（探针 v7）与臂 4b-A 数值路径
- **臂 4（lr 3e-6 + warmup100）60 步**：六段有序 **480/480**、触顶 0、cov 0.99951、峰值 57.6 GiB；
  div 10 步块 0.2790/0.2681/0.2519/0.2477/0.2688/0.2687；熵 0.348→0.331；grad_norm 均值 8.07 最大 11.06 全有限；
  训推 logp 差 |Δ| 均值 0.0132 最大 3.636（480 行）。⇒ **先前「lr 3e-6 崩塌」确为探针假象，已排除**。
- **臂 5（top_p 0.9）60 步**：六段有序 **480/480**、触顶 0、cov **0.99993**、补全中位 1617、峰值 57.7 GiB；
  div 10 步块 0.2742/0.2884/0.2620/0.2657/0.2985/0.3101；熵 0.312→0.323；grad_norm 均值 8.75 最大 12.50；
  训推 logp 差 |Δ| 均值 **0.0222** 最大 2.154（大于 lr1e-6 组的 0.0132–0.0134）。
- **臂 4b-A（lr 3e-6 + `--optim adamw_torch`）20 步**：**无 NaN**，六段 160/160、触顶 0、熵 0.3446、
  峰值 **62.8 GiB**（nvidia-smi 64.4 GiB，**未超 75，故未降 PB**）、grad_norm 均值 7.71 最大 10.94 全有限、loss 全有限。
  同 lr 同步数对照 8-bit：div 均值 0.2735 vs torch 0.2673；grad_norm 8.84 vs 7.71；峰值 57.1 vs 62.8 GiB。
  **口径更正（重要）**：`torch.optim.AdamW` 的状态用 `torch.zeros_like(p)` 分配 ⇒ bf16 参数下状态也是 **bf16**（本机实测
  `exp_avg/exp_avg_sq dtype = torch.bfloat16`）。故臂 4b-A 实际对比的是「bnb 8-bit 状态 vs torch bf16 状态」，
  **不是** fp32 状态；真要测 fp32 状态须走 4b-B（fp32 主权重）。按裁决「若仍 NaN 才做 B」，A 无 NaN ⇒ **B 未触发、未做**。
- 现状小结：结构崩塌问题在探针修好后**不再出现**（臂1/4/5/4b 全部 8/8 满分），剩下的问题是**散度不持续下降**（三臂均为 U 形）。

## N13（2026-09-07）pilot 首次起跑失败：容器 GPU 访问被吊销（非配置问题）
- 签名：`ValueError: Expected a cuda device, but got: cpu`（`swift/rlhf_trainers/vllm_client.py:257 init_communicator` ←
  `swift/arguments/rlhf_args.py:511 init_communicator(device=get_current_device())`）。
- 根因链（实测）：`get_current_device()`（`utils/torch_utils.py:136-145`）在 `is_torch_cuda_available()` 为假时返回 `'cpu'`；
  容器内实测 `torch.cuda.is_available() False / device_count 0`；容器内 `nvidia-smi` 报 **`Failed to initialize NVML: Unknown Error`**，
  而 `/dev/nvidia0,1,ctl,uvm` 节点仍在、宿主 `nvidia-smi` 正常（driver 570.195.03，两卡 persistence Enabled）。
  典型的容器 device-cgroup 允许列表被吊销：**已运行进程不受影响**（gpu0 上两个 vLLM EngineCore 仍在跑、共 76 GB），
  **新建进程**无法初始化 CUDA。命令行与臂 6 逐项相同（仅 max_steps/save_steps/output_dir 不同），排除配置原因。
- 处置：重启容器恢复 device 规则 → 重启 gpu0 的两个 vLLM 服务 → 重跑 pilot。首次失败，按规则重启一次。

## N14（2026-09-07）pilot 运行期两项裁决改动
- **在线评测搬到 gpu1**（`online_eval.sh` 默认 `GPU` 0→1），改动时刻：pilot 已跑到约 **step 60**（step-50 评测已用 gpu0 完成）。
  受影响：**step 100 及以后**的在线评测在 gpu1 上跑（训练峰值 60.4 GB + 评测约 11 GB ≈ 71 GB，回到 65–80 GB 带内）。
  step-50 那次评测使 gpu0 峰值达 **86,290 MiB**（两个 vLLM 服务 75.2 GB + 评测 ~11 GB），超带但未 OOM，数据有效。
  **不动 rollout server 的 `vllm_gpu_memory_utilization`**（改它会改变生成分布，破坏与前六臂的可比性）。
- **判据收紧**（因 step-50 主判据均值 0.4900 = 基线 −1.8%、中位持平）：
  - step 100 若余弦均值 **≥ 0.5089**（= 0.4989 + 0.01）⇒ 按原规则续到 1 epoch；
  - step 200 若仍 **< 0.5089** ⇒ 判「OPSD 在此配置下 200 步内无效」，**停 pilot、不续 epoch**，改写收尾报告
    （六臂表 + pilot 曲线：余弦/散度/熵/六段有序/训推差 + 下一步候选，每条附预算与预注册判据）。

## N15（2026-09-07）pilot 停机与收尾
- pilot 实跑 90 步（预定 200），无 traceback，为外部停机；目录留档 `opsd_full/pilot200_lr5e6_COLLAPSE_step89/`。
- 崩塌定位：六段有序率首个 <8/8 = **step 56**，首个 <0.5 = **step 68**，首个 0/8 = **step 68**，首个触顶 = **step 75**。
  同期散度 0.2250→0.1341、熵 0.361→0.535、grad_norm 6.56→2.28、触顶行 0→10、top-64 覆盖 0.99961→0.99723。
- 崩塌形态（各 3 条 greedy 样例）：文本仍 6 段、三行齐全、到 EOS；**只有阶段 token 身份退化**：`[1,2,3,4,5,6]` → `[1,4,4,4,5,6]`。
  崩塌前样例取自 `checkpoint-50`，崩塌后取自仍持有 step-89 权重的 rollout server（`/infer/` 端点，greedy）。
- 结论（写入报告 §11.4）：**结构崩塌与 lr/warmup 无关，是步数到量必然出现**；60 步小臂全部守住只因崩塌在 60 步之后；
  baseline(5e-6 无 warmup) step 14 崩是同一现象的更早版本，warmup/低 lr 只推迟不消除。
- **硬停条件**（§11.5，对后续所有 RL/蒸馏臂强制）：六段有序率连续 2 个记录点 <0.5 即停；触顶率并列上报。
  本次若已生效应在 **step 69** 停机。该条件写入驱动脚本为待接线项。
- 两卡已释放（rollout/teacher 两个 vLLM 服务已停，gpu0/gpu1 均 0 MiB），未自动起新臂。

## N15（2026-09-07）pilot 停机与两卡释放
- 停机：pilot 跑到 **step 89/200** 人工停（用户裁决），目录改名 `opsd_full/pilot200_lr5e6_COLLAPSE_step89/`，
  保留 `checkpoint-50`（8.3 GB）、`probe_segments.jsonl`（90 步 seg + 91 条 v7-log）、`online/`（step-50 全量评测产物）、`nvsmi.log`。
- 崩塌定位：首次非满分 **step 56**、首次 <50% 与首次 0/8 **step 68**、step 71 起连续 0/8。
- 崩塌前后样例：崩塌后无 ckpt（save_steps=50，step 100 未到），故直接向 **rollout vLLM server** 取（其权重 = 停机时最后同步的 ≈step 89）；
  崩塌前用 `checkpoint-50` 离线生成同 3 条 prompt。落盘 `collapse_samples_{pre_ckpt50,post}.json`（已复制进 run 目录）。
  形态：阶段 token 身份塌陷（`[1,4,4,4,5,6]` / `[1,4,4,5,5,6]`）或整篇无阶段 token 提前 stop；prose 通顺、无重复、无长度爆炸。
- 两卡已释放：容器内 `swift deploy`(:8100) 与 `swift rollout`(:8000) 及其 `VLLM::EngineCore` 子进程全部 kill，
  `nvidia-smi` 两卡均 **0 MiB**、compute-apps 0 条。**按裁决不自动起新臂，等下一路线。**
- 报告：ENG2_REPORT.md 增 §11（pilot 收尾：全曲线/崩塌定位/生成样例/三条观测/硬停条件）与 §12（5 条候选 A–E，各附预算与预注册判据）。
