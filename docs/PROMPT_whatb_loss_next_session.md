# 交接 PROMPT（直接粘贴给新会话使用）

---

你接手 VeraRetouch 的 **whatb 分支（what 侧色彩变换）**。仓库 `/home/bc/VeraRetouch`。

## 一句话处境

whatb 六臂已实现并上线，**跑完的三个臂全部无效**：两个 loss 收敛了但 headline 打不过平凡基线，
一个训练直接塌缩。根因已定位到 **loss 配方**，你的首要任务是跑两套 loss 对照把它定下来。

## 立即要做的事（按优先级）

### 1. 【等待，不要动】L8 缓存正在生成

`L8CACHE_A`(gpu0) / `L8CACHE_B`(gpu1) / `L8CACHE_MERGE`(CPU, `--after`) 三个作业在跑，
ETA **2026-08-16 约 02:00**。产物 46,129 条（normal 25,894 + low 20,235）。
**两卡被它占满，在它跑完之前不要提交任何 GPU 作业，也不要 cancel 它。**
用 `q status` / `q events --since 2h` 看进度。

### 2. 【主任务】跑两套 loss 对照

用户 2026-08-15 指令原话：「跑两套 loss，测试用纯 L1 和 修正后的 whatb 添加一个消融项」。

| 臂 | loss | 依据 |
|---|---|---|
| A · 纯 L1 | `L_rec` 单项（去掉 L_hc 与 R_sparse） | 归档 transformer backend 的口径，见下方 §关键事实-4 |
| B · 修正 whatb | `L_rec + 10·L_hc(**C 归一化**) + 0.001·R_sparse` | 唯一被验证有效的档，见 §关键事实-3 |
| 消融行 | 在 B 之上加/去一项（由你设计） | 用户要的「添加一个消融项」 |

**两套都要跑**，因为它们回答不同的问题：纯 L1 回答「L_hc 是不是纯干扰项」，
C 归一化回答「L_hc 归一化后还有没有价值」。只跑一套会留下另一半疑问。

改动量很小：纯 L1 基本是删项；C 归一化是 `q3vl/whatb/colorimetry.py` 里
`c = c / c.detach().mean()` 一行（`valid` 掩码与 `h` 的分母仍用原始 C）。

**但这动的是跨臂冻结口径**（六份提案共用块 sha256 逐字节相同），改了要六臂同步、
之前数字作废。开工前向用户复述一遍你的实验设计。

### 3. 【暂缓】L8 的消费侧接入

缓存生成完之后训练代码**不会自动用上**。接入点已探明（见文档 §3.3），但**建议等 loss 定案、
六臂重跑出可比数字之后再做**——否则「换了 loss」和「加了数据」两个变量叠在一起，归因会糊。

---

## 关键事实（不看文档也要知道的最小集）

### 1. 跑完的三个臂的判据数字（ΔE00，越小越好）

| 列 | CARRIER | IDGATE | AFFONLY |
|---|---|---|---|
| **headline_normal_only** | **7.895** | **10.149** | 训练塌缩，无板 |
| B0_identity（什么都不做） | 8.293 | 8.293 | — |
| B1_libmean（不看指令用平均 LUT） | 7.632 | 7.715 | — |
| B2_librandom | 10.099 | 9.971 | — |
| B3_bucket_retrieval（按类别桶检索） | **6.155** | **6.064** | — |
| B4_oracle（库内最优） | 0.825 | 2.827 | — |

CARRIER 比 B0 只好 0.4、**比 B1 还差**；IDGATE **连 B0 都不如**且三个负控制 delta 全为 0
（换任何指令输出不变）。**B3（6.06–6.16）是真正要打败的线。**

### 2. loss 轨迹说明「收敛 ≠ 学到东西」

| | L_rec 首 → 末 | L_hc 首 → 末 |
|---|---|---|
| CARRIER | 0.5596 → 0.1770 | 25.42 → 2.106（−92%） |
| IDGATE | 0.3044 → 0.1001 | 6.582 → 1.072 |
| AFFONLY | 0.1756 → 0.185（**上升**） | — |

AFFONLY 零初始化时 `f ≡ identity`，其起点 0.1756 **就是恒等变换的水平**；CARRIER 训练
117,399 步后落在 0.1770，同量级。降得最狠的是 L_hc，L_rec 只是搭便车。

### 3. 根因与已验证的解（CPU 实验，真 z 缓存 + 真 bank，seed 20260810，B=32×Q=256）

`L_hc = mean(C·(1−cos Δhue))` 里 **C 是未归一化的 CIELab chroma**（实测 mean C = 32.5），
所以 `10·L_hc` 的等效权重约 **325× L_rec**。AFFONLY 的机制链：
全局分支被 clamp 掐死（零梯度）→ `head_color` 末层隐层 ReLU 全局死亡率
0.469→0.938(s500)→**1.0000(s1300)** → 只剩 bias → 所有样本输出逐位相同 → 不可逆。

| 档 | cross_std@1300 | ReLU 死亡率 | L_rec@1300 |
|---|---|---|---|
| baseline (grad_clip=0) | **0.0** | 1.0000 | 0.18383 |
| grad_clip=1.0 / 0.5 / 0.1 | **0.0 / 0.0 / 0.0** | 1.0000 | 0.1834 / 0.1832 / 0.1833 |
| **C 归一化** | **9.311e-4**（3000 步 3.2e-2） | 0.773 | **0.15675**（3000 步 0.1548） |

**梯度裁剪已被证伪**：三档全塌，两档比不裁更早塌。原因是 pre-clip 梯度范数 ~13，
阈值 1.0/0.5/0.1 ⇒ **100% 的步都触发**，等于每步均匀缩放整个梯度向量，而
**Adam 对均匀缩放基本免疫**。别再试这条路。

**残留风险**：C 归一化档早期（到 ~1200 步）cross_std 在 4e-5–4e-4 徘徊，若守卫在早期
quick-eval 查会误杀。全量档首次 quick eval 在 step 2935，那时已在 1e-3 量级，安全。

### 4. 归档里的对照（`model/glut_repro/`，只取代码事实）

上一战役把生成器换成 transformer backend 时用的 loss：
- `train_rdg.py`（3D）：`loss_main = (render(p,x) − y).abs().mean()` —— **纯 L1，无 L_hc**；
  `--clip` 默认 1.0（`:211`，`:344`）。
- `train_rdg2.py`（4D）：同样纯 L1 + aux + hinge，无 L_hc。
- `losses.py:80-83` 有 `glut_full_loss = l_rec + 10·l_hc + 0.001·r_sparse`，
  但**只有 per-LUT 直接拟合那条线（A0/E1）调用它**，两个 transformer backend 脚本都没用。

即「10·L_hc + 深网络条件生成器」这个组合此前从未被采用过。注意：那次的**实验结论**
按用户裁定不可信，这里只引用代码写了什么；它的 clip=1.0 是配那次的优化器语境，
不能类比到我们的 Adam。

---

## 纪律红线（违反即返工）

- **判据不动**：headline 定义、12 个预注册键、三负控制（N1 shuffle / N2 无关词 / N3 固定短语）、
  五条平凡基线一律不改。AUC 禁用；checkpoint 选择禁用 val loss；禁逐图 min-max 归一化；
  预注册判据必须有运行时断言；跨臂比较必须步数匹配。
- **冻结口径八项**（六份提案共用块 sha256 逐字节相同）：train normal-only n=93934 ／
  B=32×Q=256=8192 色每步 ／ 2936 步每 epoch ／ 总 117440 步 ／ clamp `two` ／
  headline 形成式 `Î=(1−α)⊙I+α⊙f̂(I)` ／ 12 个预注册键 ／ colorspan 自带实现 + 断言。
  **改任何一项都要六臂同步、之前数字作废，属用户级裁定，不要自行决定。**
- **污染源禁读**：`q3vl/what/`、`gpu_render/`、旧 what 实验记录、`trash/`。
  `model/glut_repro/` 可查**代码事实**（loss 定义、超参），其**实验结论**不可信。
- **GPU 一律走队列**（`q submit`），禁裸跑（禁 nohup/setsid/`&`）。同卡禁双训练臂。
  `q` 用 `env -i` + 白名单，**环境变量必须挂在 payload 命令行上**（`-- env KEY=VAL bash ...`），
  写在提交 shell 里无效。
- **先短后长**：每臂先跑冒烟（带完整出板断言 + 退化守卫），冒烟产物作全量的 gate。
  今天这道门拦下了 7 次问题，每次成本几分钟而不是几小时。
- **结果只列数字，禁下结论、禁揣测性表述**（「说明/暗示/可能因为/更好/优于」一律不许出现）。

## 环境与入口

- python `/home/bc/envs/q3vl_sft/bin/python`，`PYTHONPATH=/home/bc/VeraRetouch`
- 六臂 runner：`q3vl/whatb/scripts/run_{carrier,affonly,interpc,idgate,g4d,qdual}_arm.py`
- wave：`/home/bc/agent-gpu-queue/waves/whatb_epr024_029_{wave,arm}.sh`
- 队列：`q status` / `q events --since 2h` / `q submit ...`（skill: gpu-queue）
- CPU 测试：`CUDA_VISIBLE_DEVICES="" pytest q3vl/whatb/tests -q`（当前 638 passed）
- z 缓存：`/home/bc/data/runs/whatb/zcache_v2seg/`（现有 7 份）、
  `/home/bc/data/runs/whatb/zcache_l8/`（L8，生成中）
- 基座：`/home/bc/data/runs/q3vl_base_sft_v2seg_20260814/checkpoint-4976`（冻结，eager attn）

## 详细文档

1. `docs/HANDOFF_whatb_loss_2026-08-15_evening.md` —— 本 prompt 的完整版，含机制链、
   六臂状态、L8 接入方案与三个硬障碍、今天踩过的八个坑
2. `docs/HANDOFF_whatb_2026-08-15.md`（1168 行）—— 战役全貌、三个形式化问题、
   三个数学结果、GLUT 权威参数化、判据体系全文
3. `docs/RESEARCH_what-cglut-supervision_2026-08-14.md`（555 行）—— CGLUT 监督方式调研
4. `experiments/prs/EPR-024~029/PROPOSAL.md` —— 六份提案（冻结口径的共用块在里面）

---

**开工第一步建议**：先 `q status` 确认 L8 缓存还在跑、别动卡；然后读文档 1 的 §一与 §三·五；
然后把你的两套 loss 实验设计（含消融项选什么、怎么保证与已有数字的可比性）复述给用户确认。
