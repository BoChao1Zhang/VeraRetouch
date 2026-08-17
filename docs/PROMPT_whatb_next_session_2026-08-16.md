# 交接 PROMPT（直接粘贴给新会话）

---

你接手 VeraRetouch 的 **whatb 分支（what 侧色彩变换）**。仓库 `/home/bc/VeraRetouch`。
本文写于 **2026-08-16 16:30**，前一会话上下文将满。

## 一句话处境

**EPR-030 是新建臂**（共用 query 主干 + 纯 L1 损失），它把交接文档里那个坏 loss
（`L_rec + 10·L_hc`，未归一化 chroma，等效 325×）换掉了。旧口径下已拿到一个**明显优于所有
平凡基线**的板；今天又连续改了三轮口径（+L8 数据 → 加大色批 → 调 lr），**当前 8 行全量在跑，
剩约 8 小时**（预计 2026-08-17 凌晨 00:30 前后出齐）。

---

## 一、立即要做的事（按优先级）

### 1.【等待】8 行全量在跑，别动卡

pueue id 243–252，`q status` 可查。**卡满载，不要提交新作业，也不要 cancel。**
每 30 分钟查一次晚期 NaN（见 §四的采信纪律）。

### 2.【卡一空就做】改 wave 脚本，否则五臂提交即报错

`/home/bc/agent-gpu-queue/waves/whatb_epr024_029_arm.sh` 的 **QDUAL 档写的 `--data zcache`
已被改名成 `--z-source zcache`**，直接提交会 argparse 报 `invalid choice: 'zcache'`。
**运行中不能改这个 bash 脚本**（7 个在跑作业正在执行它），必须等卡空。

同时要改的：五臂的 `--eval-every` 从 2936 换算成 **469**（一个 epoch @ B=256）。

### 3.【卡一空就做】五臂冒烟测显存，再定并发

EPR-025~029 的口径接入**已完成**（见 §三），但两个臂的算力被自己的变量放大：

| 臂 | 放大 |
|---|---|
| IDGATE | `--gate-ku 4` → **8,388,608 对/步**，是其余四臂的 4× |
| INTERPC | 插值流每步额外约 2,097,152 色前向 + 两次 LUT 求值 |

**先各跑一条冒烟实测 `torch.cuda.max_memory_reserved`，再定 `--mem-peak` 与并发数，禁估算。**
G4D / QDUAL 出板需要 `field_pred`，产物在
`/home/bc/data/runs/whatb/predfield_stlang_v2seg/`（897 条，08-15 17:31 落盘），
提交时把 **`WHATB_PREDF` 挂在 payload 命令行上**（wave `:94` 读它，`:339/:370` 转成
`--pred-field-dir`）。**写在提交 shell 里无效**——这个坑已赔过一轮。

### 4.【等结果】E031 出板后定主干

---

## 二、关键数字（不看文档也要知道的最小集）

### 2.1 五条平凡基线（不随训练变，是要打的线）

| 列 | 值 |
|---|---|
| `B0_identity`（什么都不做） | **8.2926** |
| `B1_libmean`（不看指令用平均 LUT） | **7.6323** |
| `B2_librandom` | **10.0989** |
| **`B3_bucket_retrieval`（真正要打败的线）** | **6.1553** |
| `B4_oracle` | **0.8253** |

每块板都自算一次，取值在所有已出板的行之间逐位一致。

### 2.2 旧口径已拿到的最好结果（`--batch-split 32x256`，8192 色/步，149,800 步，lr 1e-3）

`E030_P4_MLP` = `--backbone mlp`（**447,212** 参数）+ 纯 L1 + `--data v2seg+l8`：

| 列 | mean | 配对 Δ（臂−基线） | p_wilcoxon |
|---|---|---|---|
| **headline_normal_only** | **4.7338** | — | — |
| B0 | 8.2926 | −3.5587 | 1.2e-82 |
| B1 | 7.6323 | −2.8985 | 1.3e-70 |
| **B3** | 6.1553 | **−1.4215** | 3.5e-28 |
| B4 | 0.8253 | +3.9085 | 1.3e-82 |

三负控制 Δ：shuffle **4.4937** / irrelevant **3.4245** / const **3.4903**（p 均 < 1e-76）。
seed 复现 `E030_P4_MLP_SEED2` = **4.6254** ⇒ **run-to-run 方差 0.1084**。
产物：`/home/bc/data/runs/what_b/whatb_QDEC_P4_MLP{,_SEED2}/`。

**对照旧战役**：CARRIER 7.895（打不过 B1）、IDGATE 10.149（连 B0 都不如）。

PSNR（额外诊断，不在 12 个预注册键内，`psnr_diagnostic.json`）：
arm **28.0184** / B0 22.6101 / B1 23.1583 / B3 23.7488（n 均 567）。
`B4_oracle` 30.5835 但 **n 只有 122**（445 张 mse=0 被剔除），**不可与其它行并排读**。

### 2.3 当前口径（E031 批，在跑）

| 项 | 值 |
|---|---|
| 色批 | **`--batch-split 256x8192`** = 2,097,152 色/步（旧口径的 256×） |
| 数据 | `--data v2seg+l8`，train normal-only **n = 119,828**（sft2seg 93,934 + L8 25,894） |
| 步数 | **469 步/epoch**，40 epoch = **18,760 步** |
| lr | **`--base-lr 1e-3`** |
| 损失 | 纯 L1 单项（λ_hc = λ_sparse = λ_mono = 0） |

**这批与 §2.2 的数字不可比**（色批变了）。

---

## 三、今天确立的三件事（都有实证，别再重复踩）

### 3.1 lr 按 sqrt(batch) 缩放：**已被否**

用户裁定过「色批 ×256 → lr ×16 = 1.6e-2」。实测 **7 行 7 死**：

| 死法 | 行 |
|---|---|
| 50 步内 NaN | `E031_MLP_CD512` / `E031_QDEC` / `E031_QDEC_MEM4` / `E031_QDEC_MEM8` / `E031_QDEC_D512` |
| step 469 条件性塌到 **2.02e-09**（地板 1e-4） | `E031_MLP` / `E031_MLP_CD256` |
| step ≈5949 NaN | `E031_MLP_NOL8` |

而 lr **1e-3 / 4e-3 / 3e-4** 至今零死，首次守卫 `std_over_samples` 全在 **0.058–0.115**。
**结论：这个 batch 上 lr 保持 1e-3。**

注意那两行「塌缩」不是 NaN：`std_over_queries` 0.233 ✓、`mean|f(x)−x|` 0.0987 ✓，
但 **32 个样本给出同一个变换** —— 大 batch 稀释了指令条件性的梯度，配大 lr 就收敛到无条件解。
这正是 IDGATE 当初「三负控制全为 0」的同一个病。

### 3.2 `--qdec-mem-rows 1` 时 cross-attention 是**退化**的

softmax 在单 key 上恒等于 1 ⇒ `cross_attn(h,mem,mem) = W_o·W_v·z`，**与 query 无关**。
数值实证（fp32，5 个不同 query 的输出彼此最大偏差）：

| memory 行数 | 偏差 |
|---|---|
| M=1 | **2.98e-08**（float32 噪声） |
| M=4 | 2.58e-01 |

⇒ **默认档跑的不是 attention 解码器**，而是 49 条共享权重 FFN + 每层一个相同的加性条件向量。
已写进 `PROPOSAL.md` **§2.1.1**。
**「MLP 打赢 transformer」这个说法目前站不住** —— 赢的那个 qdec 没用上 attention。

memory 行数的稳定性（全部 lr=1e-3，色批 256x8192）：

| M | 结果 |
|---|---|
| 1（退化） | 稳 |
| 2 | **在跑，3399 步仍稳** |
| 4 | NaN @ ≈2199；**换 lr 3e-4 后已越过该点（2649 步仍稳）** |
| 8 | NaN @ ≈749（另加旧口径三次） |

### 3.3 守卫只绑**首次** quick eval ⇒ NaN 的全量会发布 **headline = 0.0** 的假板

`E030_P4_MLP_NOL8` 在 step≈131,049 NaN，但板上 `headline_normal_only` = **0.0**
（std 0、p10/p50/p90 全 0、n=567），三负控制 Δ 全 0，B0/B1/B3 的 Δ 恰等于 −基线值，
且 `published=true`、rc=0。成因：守卫不复查 + NaN 预测的 ΔE00 算出来是 0.0 而非 NaN。

**采信纪律（硬）**：读任何全量板的 headline 之前，先跑
```bash
jq -c 'select(.step!=null)|{step,L_rec}' <run_dir>/steps.jsonl | tail -3
```
**看到 `null` 就作废这块板。** 今天这个洞咬了两次（一次假板、一次白占 14 小时）。
修法（未做）：每次 quick eval 都查，或至少收尾前查一次。

---

## 四、纪律红线（违反即返工）

- **判据不动**：headline 取 `.contexts.all.headline_normal_only`（**禁用顶层 pooled `.baselines`**，
  混用少算约 0.031）、五条平凡基线、三负控制、12 个预注册键 + 每键运行时断言、
  退化守卫的阈值/判据/witness。**AUC 禁用**；**checkpoint 选择禁 val loss**；
  禁逐图 min-max；跨行比较必须步数匹配（U4）。
- **主 agent 只编排不编码**：实施派零上下文 subagent，改完派独立实现审阅，blocker 未清不得进实验。
- **GPU 一律走 `q submit`**，禁 nohup / setsid / `&`。
- **`q` 用 `env -i` + 白名单** ⇒ 环境变量必须挂在 **payload 命令行**上（`-- env KEY=VAL bash ...`）。
- **提交后必须回读 pueue 存的命令串**确认参数被正确切分：
  ```bash
  pueue status --json | jq -r '.tasks|to_entries[]|.value|select(.label=="<NAME>")|.command'
  ```
  本会话**三次**栽在这 —— zsh 对未加引号的变量展开**不做单词切分**（bash 才做）。
  `WHATB_ARM_PRINT_ARGV=1` 只验 wave 脚本那一层，**验不到 `q submit` 收到的形态，两层都要验**。
- **一律无依赖提交**（不用 `--after`）：pueue 组并行度本来就是 1（我改成了 5，用完记得改回），
  `--after` 只会把一次 NaN 连坐成四次，本会话发生两轮。
- **判活用 `ps -p <pid>`，绝对禁 pgrep**。`rm -f` 日志后再重定向（zsh noclobber）。
- **gate 一律指本地盘产物**，绝不指 `/mnt/nfs`（硬挂载 `test -e` 会进 D 态杀不掉）。
- **进程启动后禁改源码**（提交时 sha256 冻结）。改共用文件前先确认没有在跑作业依赖它。
- **`q pause <group>` 会把在跑任务 SIGSTOP 挂起**，不是只停新作业入队；pause 完记得两个组都 resume
  （本会话漏 resume gpu1 空转 17 分钟）。
- 污染源禁读：`q3vl/what/`、`gpu_render/`、`trash/`、旧 what 实验记录。
  `model/glut_repro/` 可查代码事实，其实验结论不可信。
- **结果只列数字，禁下结论、禁揣测性表述**（「说明/暗示/可能因为/更好/优于」一律不许出现）。

---

## 五、EPR-025~029 五臂重跑（用户已确认，待卡空）

**它们至今没被修好的 loss 检验过，一次都没有。** 交接文档判定「当前 loss 配方下三个臂的数字
全部不可用」，指 CARRIER 7.895 / IDGATE 10.149 / AFFONLY(塌缩)——现在只有 CARRIER 那一路
有等效替代（EPR-030 的 `--backbone mlp` 档）。**AFFONLY 与 IDGATE 尤其值得重跑**，
它们当初的失败正是那个坏 loss 造成的。

口径接入**已完成**（新建 `q3vl/whatb/caliber.py` 作为唯一共用口径层，色批表与 `BASE_LR`
从 `arms/carrier.py` import，L8 并集走 `run_carrier_arm.open_z_caches(extra_sources=)`）：

- 五臂 dry-run 全部对上：n=119,828 / 2,097,152 色/步 / 469 步/epoch / 18,760 步
- pytest **720 passed**
- **`32x256` / `64x128` 逐位不变：五臂 × 两档 = 10/10 sha256 相同**
- NOTES.md 新增 **§18**

提交命令形如（`--base-lr 1e-3` 是我按 §3.1 定的）：
```
bash waves/whatb_epr024_029_arm.sh <ARM> full \
  --batch-split 256x8192 --data v2seg+l8 --base-lr 1e-3 --loss-level 1 --eval-every 469
```

---

## 六、当前在跑的 8 行（pueue id / 变量 / 用途）

| id | 作业 | 变量 | 回答什么 |
|---|---|---|---|
| 243 | `E031_MLP_LR1E3` | mlp，lr 1e-3 | **新口径基准** |
| 244 | `E031_QDEC_LR1E3` | qdec d=256/zero/head-lr 0.01 | 主干对比 |
| 245 | `E031_MLP_CD256_L3` | `--cond-dim 256` | 读出带宽是不是瓶颈 |
| 246 | `E031_MLP_CD512_L3` | `--cond-dim 512` | 同上 |
| 249 | `E031_QDEC_LR4E3` | lr 4e-3 | lr 上界 |
| 250 | `E031_QDEC_MEM2_L3` | `--qdec-mem-rows 2` | **唯一活着的非退化 attention** |
| 251 | `E031_QDEC_MEM4_LR3E4` | mem-rows 4 + lr 3e-4 | 分离「结构不稳」与「lr 过大」 |
| 252 | `E031_MLP_NOL8_L3` | `--data v2seg`，步数对齐 18,760 | **L8 A/B**（配对 243） |

产物根：`/home/bc/data/runs/what_b/whatb_QDEC_<TAG>/`（`metrics.json` / `steps.jsonl` /
`run_setup.json` / `best.pt`）。

---

## 七、待用户裁定（未静默拍板）

1. **优化器口径要不要加 warmup / wd**。StatLUT 原配方是 AdamW wd=0.05 + 5-epoch 线性 warmup；
   本战役冻结的 Adam / lr 1e-3 cosine / 无 warmup / wd=0 是给 0.45M 的 MLP 写的。
   本会话共 **10+ 条 NaN**，全部是 gnorm 尖峰后炸。**未自行改。**
2. **memory 该放什么**（用户方向：针对 Qwen3-VL 设计，不照抄 StatLUT）。
   已探明基础设施支持 `(n, K, 2560)` 三维缓存（`zcache.py` 的 `k_rows`）。
   我提过的候选：`<seg_color>` 1 行 ⊕ 色彩推理六段各自池化 6 行 ⊕ 视觉 token 2×2 池化 4 行，
   带类型嵌入。**代价：缓存要重生成（约 13.5 GB，两卡数小时）。**
   建议先看 245/246 的 cond-dim 结果 —— 如果加宽单向量条件无用，多行 memory 装同一个 z 也无用。
3. **pueue 组并行度我改成了 5，用完要改回 1**（`pueue parallel -g gpu0 1`）。
4. §3.3 的守卫洞要不要修。

---

## 八、环境与入口

- python `/home/bc/envs/q3vl_sft/bin/python`，`PYTHONPATH=/home/bc/VeraRetouch`
- pueue 真实路径 `/home/bc/.local/bin/pueue`（`q` 是它的皮）
- 队列：`q status` / `q events --since 2h` / `q submit NAME GPU LOG [opts] -- cmd`（skill: gpu-queue）
- wave：`/home/bc/agent-gpu-queue/waves/whatb_epr024_029_arm.sh <ARM> <smoke|full> [追加参数...]`
  自检：`WHATB_ARM_PRINT_ARGV=1 ... `（只打印 argv，不起进程）
  run 名由 `WHATB_DATE` 环境变量决定；out-root 由 `WHATB_RUNS` 决定
- CPU 测试：`CUDA_VISIBLE_DEVICES="" /home/bc/envs/q3vl_sft/bin/python -m pytest q3vl/whatb/tests -q`
  （当前 **720 passed**）
- z 缓存：`/home/bc/data/runs/whatb/zcache_v2seg/` + `/home/bc/data/runs/whatb/zcache_l8/`
- 基座：`/home/bc/data/runs/q3vl_base_sft_v2seg_20260814/checkpoint-4976`（冻结，eager attn）

## 九、文档

1. `experiments/prs/EPR-030_shared-query-backbone/PROPOSAL.md`（999 行）
   —— §2.1.1 M=1 退化、§4 全部实验结果、§4.5 两条方法论事实
2. 同目录 `NOTES.md` —— §1/§2/§3 三条必须裁定项、§17 生产档、§18 五臂接入
3. `docs/RESEARCH_whatb_loss_arch_2026-08-15.md`（692 行）—— loss 与结构的调研，
   19 个候选 loss / 13 个候选结构 / 30 条开放问题；含 11 条检索引擎编造记录
4. `docs/HANDOFF_whatb_loss_2026-08-15_evening.md` —— 上一轮交接（坏 loss 的机制链）

---

**开工第一步**：`q status` 确认 8 行还在跑；然后读本文 §三（今天确立的三件事）与 §四（纪律）；
然后按 §一的优先级做。**别在卡满载时改任何共用源码或 wave 脚本。**
