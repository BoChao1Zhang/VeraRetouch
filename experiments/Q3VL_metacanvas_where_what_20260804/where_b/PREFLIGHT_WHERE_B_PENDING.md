# Where-B · 待执行清单（GPU / 重 IO）

生成于 2026-08-05。**以下每一项都未执行。** 阻塞原因：两张 H100 被 Base SFT 正式训练占用
（launcher PID 3395099 / rank 3395226、3395227；本文件生成时进度 583/4976，ETA 11:29），
且 Where-B 的全部前置作业都要读同一套 NFS build 树。

环境前置（每个 shell 都要）：

```bash
export LD_LIBRARY_PATH=/home/bc/miniconda3/envs/llm_factory/lib
PY=/home/bc/envs/q3vl_sft/bin/python           # 与 Base SFT 同一环境
cd /home/bc/VeraRetouch
```

> 没有这一行，`import sqlite3` 直接 `CXXABI_1.3.15 not found`；Where-B 在 Where-A 的
> maskview shard 尚未发布时会回退到实时 mask 定位，那条路径要读 build 的 `catalog.sqlite3`。
>
> **战役 bug R6（2026-08-05 修复）**：三个入口脚本现在自带 `import sqlite3`-before-torch
> guard（且 `q3vl/whereb/__init__.py` 改成惰性再导出，否则包链会先把 torch 装进来、guard
> 形同虚设），所以**它们已经不再依赖这个环境变量**——`logs/r6_no_ld_library_path_smoke.log`
> 是在 `LD_LIBRARY_PATH` 未设置的情况下跑通的。**但 pytest 仍然需要它**（测试模块级 import
> shardio，而 pytest 进程会先经包链装进 torch），所以上面这一行对跑测试依然必要。

**依赖顺序不能换**：Base SFT 完成 → Where-A（S1 maskviews、S4 `BA-3-Joint` 校准）→ **S1 → S2 → S3 → S4 → S5**。

---

## S0 · 已经跑完、不需要重跑（记录在案）

| 项 | 命令 | 结果 |
|---|---|---|
| 213 个 CPU 单测 | `$PY -m pytest q3vl/whereb/tests -q` | 213 passed / 29 s |
| §14 项 7/8/9 的结构半 | `$PY -m q3vl.whereb.preflight` | 5 项 PASS，2 项 **显式 SKIP**；`ok=true, complete=false`；产物 `preflight_where_b_cpu.json` |
| 全尺寸 mock 闭环 + 参数量清单 | 见 `mock_closed_loop.json` | W01/W08 各 60 步，loss 1.107→0.537 / 0.897→0.450，全程有限 |
| 既有套件回归 | `$PY -m pytest q3vl/where/tests q3vl/tests -q` | 120 passed（Where-A 只读不改） |
| **B2 修复的实跑证据** | `make_oracle_latents --split V_where --limit 4 --device cpu`；`make_generated_context --split V_where --limit 2 --device cpu` | 均 exit 0；日志见 `logs/b2_*_smoke.log`。**这两条只证明路径通**（用的是 base 权重 + `--max-new-tokens 8`，所以 `format_failure_rate=1.0` 是预期的），不是质量数字 |

> **`ok` 与 `complete` 的区别（审阅 B1 逼出来的）**：`ok` = 没有 fail 且七项必需检查一个不缺；
> `complete` = 七项**全部 pass**（无 skip）。**S5 的放行条件绑定 `complete`，不是 `ok`。**
> 修复前，`--with-model` 只是把两个 skip 行删掉、两个检查函数零调用者，屏幕照样打印
> `preflight PASS` 且 `n_skip=0` —— 跑了等于没跑，而且看不出来。

---

## S1 · train 段 oracle latent（1 GPU；**Where-A 的 `BA-3-Joint` 必须先定档**）

Where-A 目前只发布 `V_where` 的 oracle latent（`run_calibration.py`），而 §5.5 的
`L_s / L_curve / L_dir` 需要**每个训练样本**都有 `w*, ρ*`。见 NOTES 的 **D-B11**。

```bash
bash q3vl/whereb/scripts/run_where_b.sh oracle train   0
bash q3vl/whereb/scripts/run_where_b.sh oracle V_where 1   # 若 Where-A 的 fit 配置不同，重跑对齐
```

第三个参数是 GPU 号：`submit` 会 `export CUDA_VISIBLE_DEVICES`，两个作业不会挤在同一张卡上
（审阅 nit N7；同时 `ps -p $PID -o pid,cmd` 打印命令行、日志用 `grep -q` 校验实质内容而非只 `tail`）。

产物：`/mnt/nfs/bc/data/datasets/where_a-20260805/oracle/BA-3-Joint/<split>/`（indexed shards，原子发布）
+ `experiments/.../where_b/oracle_BA-3-Joint_<split>.json`。

| 判据 | 门槛 | 依据 |
|---|---|---|
| 拟合成功率（band / cband12） | ≥ 90% | Where-A CPU 预跑 4 个 V_where 样本时是 100%/100% |
| shard 随机读 + sha256 | 0 失败 | `verify_published(n_random=64)` 已内置，非 0 退出即失败 |
| `sample_count` | = eligible local 数 | rejection 计数要能对上账 |

⚠ **墙钟未知，必须先小规模实测**：先 `--limit 64` 跑一次，用它外推全量（local train 75,544 或
42,752，取决于 Where-A 的 D1 裁决），再决定是否要降 `--n-random` / `--max-iter`。
Where-A NOTES 记录 CPU 上 1536 点约 2 s/图/readout。

⚠ 目标目录已存在时脚本直接拒绝运行（发布是原子的）。要重跑必须先把旧目录**挪走，不要删**。

---

## S2 · generated `<where>` 上下文（1 GPU）

§5.4 的一半训练 batch 与主评测板都依赖它。**没有它就不能开训**（代码里没有回退 GT 的路径）。

```bash
bash q3vl/whereb/scripts/run_where_b.sh genctx train   0
bash q3vl/whereb/scripts/run_where_b.sh genctx V_where 1

# Stage-What 控制臂 C01/C02 的 forced-prefix 档（amendment A-4），另发布一套 shard
$PY -m q3vl.whereb.scripts.make_generated_context --split V_what --forced-color-prefix
```

**amendment A-4：一次生成同时留下 `<where>` 与 `<color>` 两段。** What 阶段与 Where-B 对齐
采用 50/50 teacher/generated color context，而 Base SFT 本来就一次吐两段，此前只是把后半段丢了。
schema `/1 → /2` **只增不改**：v1 的每个字段名与含义（= `<where>` 段）原样保留，
Where-B 的消费路径一行未动。`<color>` 缺闭合标签与 `<where>` **同构**处理（固定边界截取 +
记 `color_format_failure`，**不回退 GT**）。生成预算相应从 128 提到 **512**
（实测 `tokens.where + tokens.color` max 332、p99 296，加四个标签）。

产物：`/mnt/nfs/bc/data/datasets/where_b-20260805/genwhere/<split>/`（indexed shards）
+ `experiments/.../where_b/genctx_<split>.json`。

方案与口径（NOTES **D-B3**）：只缓存 **token ids**，训练时用与 teacher 完全同一个
`FrozenVLM.encode` 重放 → "同层、同位置、同归一化"是结构性成立。
贪心（`do_sample=False`）、左 padding、KV cache、bf16、`max_new_tokens=128`。

| 项 | 预期 | 依据 |
|---|---|---|
| `<where>` 段长度 | p50 ~41 tok，max ≤ 96（固定边界） | GT 实测 2711 条：local max 79 + 2 标签 |
| `<color>` 段长度 | p50 ~178 tok，max ≤ 384（固定边界） | GT 实测 2711 条：max 324 + 2 标签 = 326 |
| prompt 长度 | ~448 tok（含 384 visual） | record 的 `tokens.prompt` |
| 吞吐 | **待实测**（先 `--limit 256` 标定 samples/s，再外推） | 单卡 H100、bf16、batch 8 |
| `starts_with_where_open_rate` | 应 ≈ 1.0 | assistant 的第一个 token 就该是 `<where>` |
| `starts_with_color_open_rate` | 应 ≈ 1.0 | `</where>` 之后就该是 `<color>` |
| `format_failure_rate` / `color_format_failure_rate` | **报告即可，不设门**；它们本身是 §5.4 要记录的量 | — |
| `both_segments_well_formed_rate` | 报告 | 两段都闭合的比例，What 消费侧关心 |
| `segments_overlap_rate` | 应 ≈ 0 | 只有 `<where>` 不闭合、96 token 边界切过了 `<color>` 标签时才为真 |

**吞吐不够时的唯一备选**：conda env `vllm`（0.16.0，registry 确认支持
`Qwen3VLForConditionalGeneration`）。但它在另一套 torch/transformers 上、且给不出 hidden，
切换前必须先做一致性抽检（同 256 个样本两条栈的 ids 逐条比对）。

---

## S3 · §14 项 7b / 8b（1 GPU，约 5 分钟）

```bash
bash q3vl/whereb/scripts/run_where_b.sh preflight 0     # 前台，退出码即门
```

| 检查 id | 协议项 | 断言 |
|---|---|---|
| `WB-P7b-hidden-contract` | 本仓库推论 | 把 GT 的 `<where>` ids 走 **generated 路径**再抽 hidden，必须与 teacher 路径**逐位相同**。**FAIL 则不许继续**：那意味着两种上下文的数字不可比，任何"generated 不如 GT"的结论都是废的 |
| `WB-P8b-h-where-causal-independence` | 14.8 | 同图同指令同 `<where>`，只换 `<color>` 正文 → `H_where` **逐位相同**。这是"`Q_where` 读不到 `H_color`"的数值证明（结构证明已在 CPU 半通过） |

**验收标准：JSON 里 `complete == true`。** 只看屏幕上的 `preflight PASS` 不够——
`ok` 在两项被 skip 时同样为 true。模型加载失败会产出 **fail 行**（带 `error` 字段），
不会静默变成缺失。

同时必须记录（不在协议 14 里，但决定 S4 的配置）：

- **micro-batch 显存探测**：`probe_micro_batch` 在 `{2,4,8}` 上试，取最大可行值，GAS 自动补到 effective batch 32（§10.3）。
- 单步墙钟（含冻结 VLM 前向 + guided upsample + hi-res loss），用来估 8 个臂的总排期。
- `MASK_LOSS_SPACE="hi"`（NOTES **D-B4**）在 512×768 上的显存代价；若不可接受，改 `"low"` 前必须先向主 agent 说明它会让 3px 边界项失去意义。

---

## S4 · 裁决 NOTES §四的 D-B1 / D-B2 / D-B5 / D-B11（无需算力，但必须在 S5 之前）

| 决策 | 影响 | 不裁决的后果 |
|---|---|---|
| **D-B1** `L_mask` 是否保留 `1 − softIoU`（红线冲突） | 改的是主损失 | 8 个臂跑完才改 = 全部重跑 |
| **D-B2** `H_where` 取 final-norm 前/后 | 改的是模型输入 | 跨臂不可比 |
| **D-B5** global 是否进训练 | 决定 §5.6 "global soft-IoU ≥ 0.98" 这条 gate 是否可达 | 该 gate 必然失败 |
| **D-B11** train 段 oracle latent 的排期与 fit 配置 | S1 的墙钟 | S5 直接起不来 |

其余（D-B3/4/6/7/8/9/10/12/13）已采保守默认，可在结果审阅后再回看。

**已裁定、无需再等**（2026-08-05 随审阅下达，已落到代码 + NOTES §四）：
**D-B15** shuffled 同时交换 instruction 与 `<where>` 正文（成对取自同一 partner）；
**D-B16** oracle 辅助项按"有 oracle 的样本数"归一，stage-1 的 1.00 名义权重现在真实生效，
`steps.jsonl` 每步记录 `aux_effective_scale` 与 `effective_aux_weights`。

---

## S5 · 8 个主臂（每臂 1 GPU；本机可并行 2 臂 = §11 的 W1–W4 四个 wave）

```bash
bash q3vl/whereb/scripts/run_where_b.sh train W01 0 &   # wave W1, GPU 0
bash q3vl/whereb/scripts/run_where_b.sh train W02 1 &   # wave W1, GPU 1
...
bash q3vl/whereb/scripts/run_where_b.sh train W08 1 &   # wave W4
```

启动时会先断言 **genctx 覆盖率 100%**（nit N2）：`BalancedContextSampler` 是先按索引分池、
后取 genwhere 记录的，缺一条就会在训练数小时后炸 `KeyError`，而按 §5.4 又不允许回退 GT，
所以唯一正确的反应是**拒绝启动**。

每臂产物：

- `/home/bc/data/runs/where_b/<arm>/`：`run_setup.json`、`steps.jsonl`、`eval.jsonl`、
  `where_b_step*.pt`、`where_b_final.pt`、`eval_step*/{per_sample.jsonl,metrics.json}`、`job.marker`
- `experiments/.../where_b/arm_<arm>.json`：setup + 最终四上下文板 + gate + 格式失败统计

**报告必须出现的分层**（任务卡第 7 项 + 审阅要求）：`image.upscaled` / `winner_confidence` /
`build`(g1–g4,l1–l6) / `render_mode`(local,global)。`evaluate_arm` 已对四种上下文各出一份
`strata`，不需要额外命令。

`image.upscaled` 层必须单列：Where-A 抽样实测 V_where 本地样本 **16.7%** 被上采样到短边 512，
它们的 GT 边缘是插值出来的，**边缘类指标会虚高**。

不允许的操作（§10.3 末段）：低成本短跑筛选、基于早期指标中止、看到结果后为某个臂单独改 loss。
只允许因 OOM / NaN / manifest-checkpoint digest 不一致 / 数据读取错误终止并从同一 authority 恢复。

---

## S5.5 · A5-B1 的一条待裁定项（不阻断 S5 训练，**阻断 S6 选型**）

三条负控制与同图配对差分**已全部实现并接线**（`CONTEXT_MODES` 六种模式、`evaluate_arm`
各出一块板、`instruction_paired_delta` 出 Δ/p/CI）。剩下的是**一条口径裁定**：

红线写「同图两条**相反**指令」的配对差分。实测 `V_where` 本地 400 条：同图不同指令 712 对、
语义相反 147 对，但「同主体 + 颜色方向相反 + **GT mask 相同**」只有 **2 对**——
每条指令都绑定自己的候选区域，字面对照在现有语料里不存在。

- **本轮采用（保守可行版）**：同图**不同指令**的配对差分。对 Where 更贴题
  （Where 的输出由主体决定），且图像固定 ⇒ 显著性与中心先验成对抵消。
- **若要字面对照**：反义指令文本变换（翻转 darker↔brighter 等方向词，GT mask 不变，
  **不需要新数据产物**），但它测的是**不变性**（mask 不应随颜色方向改变），方向与「Δ 应为正」相反。

**S6 之前必须由主 agent 二选一并写进 REPORT 的设置节。** 详见 NOTES §12.1 的完整数据表。

---

## S6 · 选型与收尾

- [ ] 8 臂的 `metrics.json` 汇总成一张 **§5.6 九项 gate 的预注册判据 vs 实测数字并排表**
- [ ] `metrics.lexicographic_best` 出唯一冻结 checkpoint；若无臂全过门，**必须打 `WHERE-GATE-FAILED`**，
      且后续 Stage-What 的结论不得宣称完整方法成立（§5.6 末段）
- [ ] `viz/success_*` 与 `viz/failure_*`：`I_in | instruction | GT mask | pred s | pred mask` 并排，
      **失败案例必须有**（按 Where 错误 / 格式错误 / 上下文敏感性缺失分类）
- [ ] §13 的 feature 可视化（`Q_where` 最后一个 block 的 token PCA / norm / attention entropy，
      还原成真实 2D canvas；`w`-canvas 与 `rho`-canvas 差异图）——**attention 导出必须用
      `attn_implementation="eager"` 单独重跑**（红线：FA2/SDPA 返回 None 不许回退）
- [ ] `config/`：resolved config + seed + git commit + 环境；`metrics.json`
- [ ] 把 D-B1/D-B2/D-B5/D-B11/**D-B15/D-B16** 的最终裁决写进 `REPORT.md` 的"设置"节
- [ ] **把每臂 `eval_final/ATTRIBUTION.md` 原样并入 `REPORT.md`**：它由
      `metrics.attribution_section()` 用该臂自己的数字渲染，写明
      `local_soft_iou_median` 同时是主损失支配项与选择第一顺位、因而**不是独立检验**，
      归因重量必须落在 `boundary_f1` / `local_soft_iou_p10` / `auc_target` 与
      null、shuffled 两个 Δ 上（裁定 D-B1 + 审阅 §四）

---

## 附 · 每次提交长任务的 D-20 四步（`run_where_b.sh:submit` 已内置）

1. `rm -f` 目标日志（zsh `noclobber` 会让 `> 已存在的日志` 整条重定向失败，进程压根不起）
2. `ps -p $PID -o pid,etime,cmd` 实证存活**并打印命令行** —— **绝对禁止 `pgrep`**
   （它会匹配到正在 grep 的那条 shell 自己）。PID 取自 `setsid` 的直接子进程，不是包装 shell
3. **`grep -q <实质内容> 日志`**：光 `tail` 出来给人看不算校验，空日志照样返回 0（审阅 nit N7）。
   每个子命令有自己的哨兵串：`oracle` 找 `"basis"`、`genctx` 找 `"vlm"`、`train` 找
   `"total_optimizer_steps"`；600 s 内没等到就报错返回，绝不谎报成功
4. 三步都过才写 `job.marker`（含 PID / GPU / 完整命令 / 日志路径 / 哨兵串与等待秒数），
   才向主 agent 上报
