# veraretouch_sprf.rl —— 后训练（RL / 蒸馏）接线说明（EPR-052 占位，**等调研确认后再实施**）

现状（REPORT_STATUS_20260906 §3.5 / §4）：S1F-FULL + S2F-B 的自回归读出余弦 0.392（喂 GT 文本 0.99），
训推分布不一致；候选闭环信号有二：读出向量对 e*（BK-FULL edit_enc 预建目标）的余弦、执行器 linf8。
本目录只放**接口骨架与占位配置**，不含任何训练循环；两条路线的算法细节（OPD/OPSD 的教师上下文构造、
GRPO 的奖励归一化/KL 形式）标「待调研确认」，实施前须打开原始来源核实。

## 路线 A：OPD / OPSD（on-policy self-distillation，逐 token 蒸馏）

- 学生 π_θ(· | y, 指令)：只看退化图 y + 逐样本编辑指令（`prompts.build.student_messages`），自采样 CoT。
- 教师 = **同一权重** + 特权上下文（GT CoT 作为前缀上下文，`prompts.build.teacher_messages`），
  对学生采样出的同一条序列逐 token 打分 log π_θ(x_t | 特权上下文, x_<t)。
- 目标：min_θ Σ_t KL( π_teacher(·|priv, x_<t) ‖ π_θ(·|x_<t) )（或反向 KL / JSD，**待调研确认**），
  样本来自学生 on-policy 采样（G 条/prompt，T=1.0，max_new_tokens 2048）。
- 数据：S2 快照 train 73,854（含 GT CoT）；val 3,786 只做监控；held-out d6 1,464 只做评测。
- 教师 prompt 格式（占位，待确认）：`[user] <image y> + 指令 + "\n\nReference grade:\n" + GT CoT 六段` →
  `[assistant]` 学生序列。是否把 GT CoT 放 system 段 / 是否加 stop-gradient 到教师前缀：待调研。
- 配置：`configs/opd.yaml`。

## 路线 B：GRPO（组相对策略优化，序列级奖励）

- 采样：每个 prompt G 条（`num_generations`），T=1.0，max_new_tokens 2048；组内优势 = (r − mean_G) / std_G。
- 奖励（二选一或加权，`reward/`）：
  - `latent_reward.cosine_reward`：读出向量（6×128，slot 序）对预建目标 e* 的逐阶段余弦均值；
  - `executor_reward.executor_reward`：读出 → 注入 BK-FULL（edit_enc := Identity，slot→chain `flip(0)`）→
    rollout → linf8 对真值 x0；奖励取 −linf8 或 (identity − linf8)/identity（**待调研确认**）。
  - 缺阶段 token / 未到 EOS 的样本：奖励置组内最小或固定惩罚（待确认；`missing_penalty`）。
- KL：对参考策略（S1F-FULL ckpt_epoch1）的 per-token KL，系数 β（`kl_coef`），KL 估计式 k3（待确认）。
- 配置：`configs/grpo.yaml`。ms-swift `swift rlhf --rlhf_type grpo` 的自定义奖励函数入口以
  `reward_funcs` 指向本包函数（接口签名见 `reward/*.py`；docker 环境见 `docker/ms-swift/`）。

## 守卫（沿用 eval 线，缺一不出数）

A-inj（注入路径 == oracle_lut 路径逐位）、A-lat（评测端重算余弦 == 读出端记录）、G4/G4b（eval-only 键不进训练）、
槽序→链序 flip、`winner_confidence=low` 不进训练与 GT；headline 只报 held-out d6 normal-only。

## 文件

- `reward/executor_reward.py`：执行器奖励骨架（上下文装载 / 读出 / 注入 rollout / linf8）。
- `reward/latent_reward.py`：读出向量对 e* 的余弦奖励（可直接用）。
- `prompts/build.py`：学生 / 特权教师 prompt 构造（复用 `data.cot_text` 指令抽样与 `data.q3vl_text.prompt_text`）。
- `configs/opd.yaml`、`configs/grpo.yaml`：字段占位，全部标「待调研确认」。
