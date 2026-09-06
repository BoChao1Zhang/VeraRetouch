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
