# WB-IMPL · Stage-Where-B MetaCanvas 实现记录

日期：2026-08-05 ｜ 任务卡：WB-IMPL ｜ 代码：`q3vl/whereb/` ｜ 测试：`q3vl/whereb/tests/`
引用：`docs/METACANVAS_WHERE_WHAT_FINAL_EXPERIMENT_PROTOCOL_2026-08-04.md` §0/§1.1/§2.3/§3/§4/§5/§10.3/§14

状态：**实现完成 + 独立审阅判定的 5 个 BLOCKER 已全部修复 + CPU 级验证通过（213 个单测全绿）；GPU 级 preflight、generated-context 全量产出与 8 个主臂训练全部未启动**。
全程未占用 GPU：两张 H100 被 Base SFT 正式训练占用（launcher PID 3395099，rank 3395226/3395227）。

> 修复轮次二（2026-08-05，依据 `docs/reviews/REVIEW-impl-WhereB.md`）见 **§八**。

---

## 〇、审阅 BLOCKER 修复摘要（第二轮）

| # | 审阅判定 | 修复 | 证据 |
|---|---|---|---|
| **B1** | `preflight --with-model` 只是把两个 skip 行去掉，`check_hidden_contract` / `check_h_where_causal_independence` **零调用者**，却打印 `preflight PASS`、`n_skip=0` | 新增 `run_model_checks()`：真加载 checkpoint + processor + 一条真实样本，构造 `FrozenVLM` 并调用两个函数；加载或检查本身抛异常一律记成 **fail 行**（不吞）。`REQUIRED_CHECKS` 七项 id 齐全性纳入 `ok`（缺失即 fail），新增 `complete`（七项全 pass 才为真）；skip 显式出现在 `skipped` 字段里 | `test_contracts_and_preflight_wiring.py` 9 条；CLI 现打印 `complete=` 与 `skipped=[...]` |
| **B2** | `make_generated_context.py` / `make_oracle_latents.py` 构造 `WhereBDataset` 时既没传 `maskviews` 也没传 `mask_resolver`，**第一个 local 样本即崩** | 新增唯一工厂 `data.open_dataset()`：优先 Where-A 已发布 maskviews，未发布则回退实时 `MaskResolver`；`need_mask=False` 供生成作业跳过整条 mask 路径。`WhereBDataset` 在**任何 IO 之前**校验 mask 来源；`need_mask=False` 的 local 样本调 `mask_target_hi()` **抛错**而不是伪造全 1 | **真实 V_where 索引 CPU 实跑**：`logs/b2_make_oracle_latents_smoke.log`（4 样本，band/cband12 各 4/4 拟合成功，shard 发布 + checksum 校验通过）、`logs/b2_make_generated_context_smoke.log`（2 样本，生成 + 发布 + 读回全通）；`test_dataset_wiring.py` 12 条（含 AST 扫描禁止裸构造） |
| **B3** | `shuffled` 只换 `<where>` 正文，prompt 里的指令仍是样本本人的 —— 与 §5.4/§5.6 gate 名字不符，且属静默拍板 | 见 **D-B15**：instruction 与 where 正文成对交换 | `test_context.py` +4 条；`check_context_flows` +3 条断言 |
| **B4** | `compute_batch` 整体包在 bf16 autocast 里，`phi_dir @ w_dir` 是 matmul → **`s_low` 被降成 bf16**；而评测端没有 autocast → 训练与 gate 不是同一个函数，且 CBand12 臂受损远大于 Band 臂（污染 §5.3 受控对比） | 守卫下沉到 `fields.s_from_params` / `predict_fields` 内部（`torch.autocast(enabled=False)`），**任何外层 autocast 都无法再降精度**；`predict_fields(require_dtype=torch.float32)` 在训练端与评测端都传；`compute_batch` 额外断言 `s_low.dtype == float32` 并把 `s_dtype` 写进 `steps.jsonl` | `test_precision.py` 8 条，含"autocast 内外 `predict_fields` **逐位相同**"与"CBand12 对 bf16 的敏感度确实存在"的正向证据 |
| **B5** | 辅助 loss 被 global 样本稀释到名义权重的 ~0.47 倍，未申报 | 见 **D-B16**：分母改为有 oracle 的样本数 | `test_losses.py` +6 条、`test_precision.py` +1 条；`steps.jsonl` 记录实效权重 |

顺带处理的 nit：**N1**（口径常量上提到 `q3vl/whereb/contracts.py`，并有"禁止再声明"的扫描单测）、**N2**（训练启动前断言 genctx 覆盖率）、**N3**（逐样本 fit seed 与 Where-A 对齐）、**N4**（优先读已发布的 `.masklow.npy`）、**N5**（`render_mode` 潜伏 bug：改为显式赋值 + 断言）、**N7**（`submit` 接收 GPU 参数并 `export CUDA_VISIBLE_DEVICES`、`ps -p` 打印命令行、日志实质内容用 `grep -q` 校验而非只 `tail`）、**N8**（每次 eval 重置格式统计）。

---

## 一、实施前核实记录

凡是写进代码的外部事实都在这里给出核实方式。**没有一条来自"命名约定"或记忆。**

| # | 事实 | 核实方式 | 结果 |
|---|---|---|---|
| V-B1 | `text_config.hidden_size = 2560`、`num_hidden_layers = 36`；`vision_config.hidden_size = 1024` | 直接读 `/home/bc/data/models/Qwen3-VL-4B-Instruct/config.json` | `H_where ∈ R^[T×2560]`、`F_pre ∈ R^[P×1024]`，写进 `config.TEXT_HIDDEN` |
| **V-B2** | **`hidden_states[-1]` 是最后一层 decoder 的输出，在 final RMSNorm 之前** | 用**真实** `Qwen3VLForConditionalGeneration`（缩小配置）跑前向：`lm_head(norm(hs[-1])) == logits` 成立，`lm_head(hs[-1]) == logits` **不成立** | 这决定 `H_where` 到底取哪个张量。默认取 **post-norm**（见 D-B2），并写成 `tests/test_hiddens.py::test_hidden_states_last_entry_is_pre_final_norm` 钉死 |
| V-B3 | 模块路径 `model.model.visual.blocks`（24）、`model.model.language_model.{layers,norm}` | `torch.device("meta")` 上实例化真实类后 `named_children()` | `hiddens.resolve_visual/resolve_language_model` 按此实现，并有单测 |
| V-B4 | 因果注意力 ⇒ 前缀 hidden 与后缀无关 | 真实类前向：改后 6 个 token，前 8 个位置的 `hidden_states[-1]` **逐位相同** | 这就是 §14 项 8 的**数值证明**：`<where>` 段 hidden 不可能读到 `<color>` |
| V-B5 | `<where>` 段 token 长度分布 | 数 V_where+V_what+T_final 共 **2711** 条 record 的 `tokens.where` | local: min 14 / p50 43 / p95 57 / p99 63 / **max 79**；global: p50 8 / max 30。⇒ 固定边界 `WHERE_CONTEXT_MAX_TOKENS=96`（GT 永不被截断：79+2 个标签=81），生成预算 `GEN_MAX_NEW_TOKENS=128` |
| **V-B6** | boundary-F1 可微代理的出处与公式 | 打开 https://arxiv.org/abs/1905.07852 核实标题/作者：*Boundary Loss for Remote Sensing Imagery Semantic Segmentation*, Alexey Bokhovkin, Evgeny Burnaev；再打开 ar5iv 全文取公式：`y_b = pool(1−y, θ0) − (1−y)`；`y_b_ext = pool(y_b, θ)`；`P = Σ(p_b∘gt_b_ext)/Σp_b`；`R = Σ(gt_b∘p_b_ext)/Σgt_b`；`L = 1 − 2PR/(P+R)` | 编号、标题、作者、公式全部对得上，可引用。实现逐符号照抄（`losses.boundary_f1_loss`） |
| V-B7 | vLLM 在本机是否可用、是否支持 Qwen3-VL | 训练 env `import vllm` → `ModuleNotFoundError`；conda env `vllm` → `vllm 0.16.0 / torch 2.9.1 / transformers 4.57.6`；`site-packages/vllm/model_executor/models/registry.py:464` 写着 `"Qwen3VLForConditionalGeneration": ("qwen3_vl", ...)` | **可用但不采用**，理由见 D-B3 |
| V-B8 | split 组成与 shuffle 分组可行性 | 读 `V_where.index.jsonl`（896 行）+ 全部 896 条 record | local 400 / global 496；按 `source_image_id` 分组的 local 组大小：18 个单例（**4.5% 无配对**），其余成组；按 `(source_image_id, build)` 分组则 **92 个单例（23%）**。⇒ D-B6 |
| V-B9 | global 样本长什么样 | 抽 g1–g4 各一条 record | `render_mode="global"`、`region=None`、`where` 文本恒为 "global adjustment across the entire frame"、**无 `.cgt.png`** ⇒ GT mask 恒为全 1（D-B5） |
| V-B10 | `I_tar` 的定位符在哪 | 读 record 结构 | `record["image"]["baked"]` 就是 `.jpg` 渲染件（I_tar）的 shard 定位符。⇒ `data.META_KEYS` **不保留 `image` 块**，样本对象里根本没有这个路径（§14 项 9） |
| V-B11 | Where-A 已发布哪些 oracle latent | 读 `q3vl/where/scripts/run_calibration.py:158` | **只发布 `ORACLE_DIR/<arm>/V_where`**。Where-B 的 `L_s/L_curve/L_dir` 需要 train 段 ⇒ 补了 `scripts/make_oracle_latents.py`（见 D-B11，**待主 agent 排期**） |
| V-B12 | Where-A 的 `build_phi_dir` 每次调用会做两次 71×71 SVD（cond / rank 诊断） | 读 `q3vl/where/phi.py:264-269` | 训练循环里不需要诊断，于是写了 `fields.phi_dir_fast`（只走数值路径），并用 `tests/test_fields.py::test_phi_fast_matches_where_a_bit_for_bit` 断言两者**逐位相同**（3 种网格），杜绝漂移 |
| V-B13 | `q3vl.where.readout.apply_readout` 是否支持 batched 参数 | 读源码 `_band_apply` / `_cband_apply` 的 broadcast 分支 + 实测 | 支持；因此 Where-B 直接复用它，不重写 readout |

---

## 二、消费 Where-A 的接口结论（任务卡"oracle latent 消费"项）

| 项 | 结论 |
|---|---|
| basis `B` | `WHERE_A_BASIS_DIR/BA-3-Joint/{B.npy,basis.json}`；`fields.load_basis` 读盘 + **校验 `basis.json` 里的 sha256**，装成 `FrozenBasis`（`register_buffer`，**不是 Parameter**，进不了 optimizer） |
| oracle latent | `ORACLE_DIR/<arm>/<split>/` indexed shards，成员 `<sid>.oracle.json`；`stores.OracleStore` 按 `(sample_id, suffix)` 随机读，用 **`q3vl.where.basis.Latent.from_dict` 反序列化**（不自己解析字段，Where-A 一改就报错） |
| 拒绝的拟合 | `status != "ok"` → `latent()` 返回 `None`，该样本的三个辅助 loss 被**屏蔽**（不是填零）。§10.2 明文禁止"静默换成零向量" |
| `s*` / `r*(z)` | **不落盘、现算**：`fields.oracle_fields` 用同一张 `phi_dir` 与 latent 重算 `s*`，用 `apply_readout` 在 `linspace(-3,3,257)` 上重算 `r*(z)`。这样 `s_pred` 与 `s*` 永远出自同一个 `B`，不存在"latent 是在旧 B 下拟的"这种静默错配 |
| mask 视图 | 优先读 Where-A 发布的 `MASKVIEW_DIR/<split>` (`.maskhi.png`)；没有时回退到 `q3vl.where.maskdata.MaskResolver` 实时定位 |
| interop 单测 | `tests/test_stores.py` 用 **Where-A 自己的 packer**（`pack_oracle` / `pack_maskviews`）真打一套 shard，再用 Where-B 的 store 读回来，含 float64 精确往返、rejected 状态、覆盖率报告、非原子发布拒绝 |

---

## 三、CPU 级验证结果

### 3.1 单元测试：**170 个全部通过**（`q3vl/whereb/tests/`，29 s）

| 文件 | 数量 | 覆盖 |
|---|---:|---|
| `test_config.py` | 9 | 8 臂表 = §5.3 逐行；4 结构表 = §5.2 逐行（且确认是 4×2 笛卡尔积，无增删）；connector 512/6/8/2048；loss 权重与两段 schedule = §5.5；优化配置 = §10.3；**9 项 gate 与阈值 = §5.6 逐行**；GAS 恒使 effective batch = 32 |
| `test_qwhere.py` | 9 | 初始 canvas 位置是 `[-1,1]²` 均匀格；`tokens`/`positions` 都可学且梯度非零；Fourier 特征在原点是 (sin=0, cos=1) 且有界；**F_pre 位置按真实宽高比**（短边跨 [-1,1]、格子是方的、方图仍方）；**与 Where-A `geo5_grid` 的 (x,y) 逐位一致**；canvas 2D 还原 |
| `test_connector.py` | 9 | **两个 cross-attn gate 初值恰为 0**；初始输出与 `H_where`/`F_pre` **完全无关**（rtol=atol=0）；gate 打开后才有依赖；**全掩码 key 返回精确 0 而不是 NaN**（null 上下文）；掩码与截断等价；**逐分支验证 §5.1 的四步顺序**；gate 有有限梯度 |
| `test_heads.py`(并入 `test_model.py`) | — | 见下 |
| `test_model.py` | 25 | 8 臂各自能前向、只出 `w0/w_dir/alpha/rho` **无 dense logits**、`‖w_dir‖=1`、`alpha>0`；canvas 布局 = 结构表；**DualCanvas 两条 stream 参数张量互不共享**；**SplitHead 与 Joint 的参数名 diff**（`pool.`→`pool_axis.`+`pool_rho.`+双 trunk）；SplitHead 的 `rho` 只依赖 rho canvas、`w` 只依赖 axis canvas；rho 向量拆分形状；**head bias 初值全部落在 readout 界内**；**axis bias 是单位向量不是零**（否则 `w_dir` 在原点梯度 ~1e12）；forward 签名恰为白名单且拒绝 `h_color`；`Q_axis/Q_readout` 暴露给 §6；全尺寸参数量单调 |
| `test_losses.py` | 23 | soft-IoU/balanced-BCE **对照手算**；BCE 对 2% 面积区域不被"全零"打败；boundary map 是内边界；**boundary-F1 loss：完全重合=0、位移超过容差单调上升、3px 容差确实宽恕 2px 位移、两边都无边界=0、在全 1 GT 上凭空造边界被罚>0.9**；`L_mask` = §5.5 加权和；`L_s` = Huber(s/3)；**`L_curve` 对照 `apply_readout` 手算**；`L_dir` = 1−cos（同向 0、反向 2、正交 1）；**schedule 切换点 299/300 精确**、`round` 边界、两段权重值；无 oracle 时只留 `L_mask`；有 oracle 时四项之和与手算一致；`aggregate` 分 GT/generated 两个子批次报告 |
| `test_context.py` | 22 | GT 编码带标签、**过长直接抛错而不是截断**；generated 在第一个 `</where>` 处切断；**未闭合 → 固定边界截断 + `format_failure` + `stop_reason="no_close_tag"`**；**`generated_context` 的签名里根本没有 GT 文本参数（结构性禁回退）**；未闭合结果 ≠ GT span；空生成算失败；EOS 优先；闭合标签越界仍算失败；null 恒 0 token 且构造非法 token 会抛错；shuffled 携带 partner 文本；**ShuffleIndex 是组内 derangement、单例不跨图配对、global 与 local 不互串、对 seed 确定**；**每个 micro-batch 精确 50/50**（micro=2/4/8）、一个 epoch 内每样本只出现一次、两个池不相交、奇数 micro-batch 被拒 |
| `test_fields.py` | 14 | **`phi_dir_fast` 与 `build_phi_dir` 逐位相同**（3 种网格）且可微、形状校验；`FrozenBasis` 是 buffer 不是 Parameter；**`load_basis` 校验 sha256、未校准时直接抛 `FileNotFoundError`**；`s = 3tanh(q/3)` 对照手算且 `|s|<3`；`predict_fields` 与 Where-A latent 路径一致；**多通道上采样抛 `ChannelOrderError`**；梯度到达全部 7 个预测参数；`oracle_fields` 重算的 `s*`/`r*` 与 Where-A 一致 |
| `test_metrics.py` | 14 | AUC 完美排序=1/反排=0/常数=0.5、全局 mask 上返回 `None`；boundary-F1 指标；strata 拆 local/global；**两个跨上下文 gate（GT-gen 差、shuffle 降幅）算术**；9 项 gate 全过/单项失败即整体失败并打 `WHERE-GATE-FAILED`；**缺指标算失败不算通过**；阈值取等号算过；**lexicographic 选择按 §5.6 顺序**（IoU 打平看 boundary F1，再看 p10，再看参数量）；全臂不过门仍选出最优并打 tag |
| `test_stores.py` | 9 | 见第二节 interop |
| `test_hiddens.py` | 7 | **真实 Qwen3VL 类**上：模块路径解析；`hidden_states` 有 L+1 条且 `hidden_states[-1]` 在 final norm 之前；**LastLayerHook 与 `output_hidden_states` 逐位相同**；hook 正确摘除；**改后缀不改前缀（逐位）**；加 `<color>` 后缀不动 `<where>` 段；**右 padding 不污染真实 token** |
| `test_e2e_mock.py` | 10 | W01/W08 mock 闭环（见 3.2）；**zero-init gate ⇒ 首步预测与输入无关且 batch 内恒等**；训练后恢复输入依赖且 gate 已离零；**null 上下文端到端不 NaN**；**global 样本无 oracle 仍贡献 `L_mask`**；optimizer 分组把 gate/LayerNorm/bias/positions 排除出 weight decay |
| `test_evaluate_and_trainer.py` | 10 | 每样本一行、local/global 分开；**无 shuffle partner 的样本被跳过并计入 coverage**；`evaluate_arm` 出四上下文 + 9 项 gate + `per_sample.jsonl` + `metrics.json`；strata 按四个 key 分组；**梯度累积到 effective batch**；**每个 micro-batch 都是 50/50**；**schedule 在 `round(0.3·total)` 精确切换**；checkpoint 落盘；**checkpoint 选择用指标不用 eval_loss**（构造了"eval_loss 单调下降但指标峰值在中间"的场景）；warmup+cosine 端点；奇数 micro-batch 被拒 |
| `test_preflight.py` | 9 | §14 项 7/8/9 的检查函数本身 |

### 3.2 mock 端到端闭环（**全尺寸 connector 512/6/8/2048，协议 LR 2e-4 + warmup 3% + cosine + clip 1.0**）

4 个合成样本（2 个 GT 上下文 + 2 个 generated 上下文），GT mask 由**一个已知 latent 经同一条解析链生成**，所以"loss 下降"等价于"预测的 `(w0,w_dir,alpha,rho)` 确实携带信号"。原始记录：`mock_closed_loop.json`。

```
W01  MC8-Joint / R-Band       34,838,745 params   60 步 15.5 s
step  stage  loss     grad_norm  lr        soft_iou  L_s      L_curve  L_dir    gt_loss  gen_loss
  0     1    1.10702   4.9022    2.00e-4    0.6704   0.08504  0.26277  0.99247  1.18319  1.03085
 10     1    0.81858   1.7949    1.884e-4   0.8097   0.04829  0.19691  0.73861  0.97666  0.66050
 20     2    0.57230   0.8129    1.516e-4   0.8129   0.04185  0.15456  0.60426  0.69174  0.45287
 40     2    0.54050   3.3427    4.84e-5    0.8252   0.05034  0.14904  0.60452  0.62810  0.45290
 59     2    0.53708   0.5796    0          0.8283   0.05187  0.15095  0.60689  0.61523  0.45893

W08  MC16-DualCanvas / R-CBand12   69,851,781 params   60 步 45.0 s
  0     1    0.89676   1.5980    2.00e-4    0.7401   0.10190  0.15648  1.06712  0.96520  0.82831
 10     1    0.67363   0.9713    1.884e-4   0.8284   0.03624  0.12958  0.58022  0.71658  0.63069
 20     2    0.48570   0.7125    1.516e-4   0.8443   0.03842  0.13679  0.54997  0.49066  0.48073
 40     2    0.45465   0.7159    4.84e-5    0.8676   0.04866  0.15038  0.60485  0.45474  0.45456
 59     2    0.45033   0.1982    0          0.8726   0.04849  0.15066  0.61174  0.45073  0.44994
```

- 全程有限，无 NaN/Inf；参数全部有限。
- soft-IoU：W01 0.670→0.828，W08 0.740→0.873；`L_dir` 0.99→0.61 / 1.07→0.61。
- **两段 schedule 在第 18 步（`round(0.3×60)`）切换**，日志里 `stage` 字段可见。
- **zero-init gate 断言**：`preflight` 与单测都实测过——初始时把 `F_pre` 换成 `3F+1`、`H_where` 整个换掉，`w_raw/w0/rho` 的最大差为 **0.0（rtol=atol=0）**；训练后 gate 已离零、预测恢复输入依赖。

### 3.3 四结构 trainable 参数量（全尺寸，`mock_closed_loop.json:parameter_table`）

| Arm | 结构 | readout | queries | streams | pools | **总参数** | banks | streams | heads |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| W01 | MC8-Joint | R-Band | 64 | 1 | 1 | **34,838,745** | 32,896 | 33,449,484 | 1,356,365 |
| W02 | MC8-Joint | R-CBand12 | 64 | 1 | 1 | **34,855,161** | 32,896 | 33,449,484 | 1,372,781 |
| W03 | MC16-Joint | R-Band | 256 | 1 | 1 | **34,937,433** | 131,584 | 33,449,484 | 1,356,365 |
| W04 | MC16-Joint | R-CBand12 | 256 | 1 | 1 | **34,953,849** | 131,584 | 33,449,484 | 1,372,781 |
| W05 | MC16-SplitHead | R-Band | 256 | 1 | 2 | **36,254,297** | 131,584 | 33,449,484 | 2,673,229 |
| W06 | MC16-SplitHead | R-CBand12 | 256 | 1 | 2 | **36,270,713** | 131,584 | 33,449,484 | 2,689,645 |
| W07 | MC16-DualCanvas | R-Band | 256 | 2 | 2 | **69,835,365** | 263,168 | 66,898,968 | 2,673,229 |
| W08 | MC16-DualCanvas | R-CBand12 | 256 | 2 | 2 | **69,851,781** | 263,168 | 66,898,968 | 2,689,645 |

读法（这三档差异正是 §5.2 要归因的东西）：

- **8×8 → 16×16 只加 98,688 参数**（query bank 从 64×512 到 256×512）——容量差异几乎全在 canvas token 数本身，不在网络规模。
- **Joint → SplitHead 加 1,316,864 参数**：多一个 attention pool（LN+4×512²+LN）与一个 trunk（LN+512²）。
- **SplitHead → DualCanvas 加 33,581,068 参数**：整条 stream（含 `H_where`/`F_pre` 投影、位置编码、6 个 block）翻倍。
- readout 差异恒为 **16,416**（band 4 个输出 vs cband12 36 个输出，经 512 维 head）。
- 实证 diff（`test_model.py`）：SplitHead 相对 Joint 的参数**名**差异是 `pool.` → `{pool_axis., pool_rho.}` 与 `trunk.` → `{trunk_axis., trunk_rho.}`；DualCanvas 的两条 stream 参数名集合完全相同但**张量对象互不相同**（`p0[k] is not p1[k]` 全部成立）且初值不同。

### 3.4 真实数据路径 CPU 冒烟（无权重、无 GPU）

用真实 processor + 真实 `V_where.index.jsonl` + 真实 shard，跑通 `WhereBDataset → _PromptShim →
collator.encode_one → gt_context → phi_dir_fast → guided upsample` 整条链（`F_pre` 用随机张量顶替，
因为那一段要 GPU）。

| 样本 | build | out / grid | n_visual | prompt tok | where tok | mask 形状 | mask 均值 / 软边占比 |
|---|---|---|---|---:|---:|---|---|
| `sft_006ce4…` | g2 | 640×512 / 40×32 | 320 | 390 | 8 | 640×512 | **恒 1**（global） |
| `sft_006ef4…` | g1 | 512×768 / 32×48 | 384 | 461 | 8 | 512×768 | **恒 1**（global） |
| `sft_00680d…` | l6 | 512×768 / 32×48 | 384 | — | — | 512×768 | 0.500 / 0.863 |
| `sft_0116bd…` | l5 | 512×768 / 32×48 | 384 | — | — | 512×768 | 0.172 / 0.066 |
| `sft_01f6c3…` | l5 (low) | 768×512 / 48×32 | — | — | — | 768×512 | 0.700 / 0.350 |

同时实证：`phi_dir` 形状 `(grid_h·grid_w, 71)` 且全部有限；guide 形状 `(1,1,out_h,out_w)`；
**样本对象的 `meta` 里既没有 `image` 块也没有任何含 `baked` 的值**（§14 项 9 的数据侧证据）；
local 样本经 `MaskResolver` 实时定位 + sha256 校验读到的 `.cgt.png` 与 spec-5 几何一致、值域在 [0,1]。

### 3.5 §14 项 7/8/9 的 CPU 半（`preflight_where_b_cpu.json`）

```
[PASS] WB-P7-context-flows          四条上下文流 + 50/50 采样 + derangement + 禁回退
[PASS] WB-P8-no-h-color             签名扫描 + 代码标识符扫描 + 关键字被拒
[PASS] WB-P9-no-target-leak         输入白名单 + META_KEYS 不含 image(→image.baked=I_tar) + forward 签名
[PASS] WB-P-zero-init-gates         W01/W08 全尺寸：gate 全 0、初始输出与输入差 0.0
[PASS] WB-P-param-table             8 臂参数量清单与单调性
[SKIP] WB-P7b-hidden-contract              需要真实前向（GPU）
[SKIP] WB-P8b-h-where-causal-independence  需要真实前向（GPU）
```

---

## 四、假设与待确认清单

能自行核实的已当场核实（V-B1 ~ V-B13）。以下是**两种做法都合理、且影响后续**的，一律采**保守默认**继续，**未静默拍板**。

### 待主 agent 决策

**D-B1（最重要，红线冲突）· `L_mask` 的第一项是 `1 − softIoU`。**
协议 §5.5 逐符号写明 `L_mask = (1 − softIoU) + 0.25·balanced_BCE + 0.10·boundary_F1_3px`，任务卡也逐符号复述；但 `CLAUDE.md` 红线写 "IoU 禁当优化目标"，而 §5.6 的主排序指标又正好是 soft-IoU——等于"对着选择指标训练"。
- 保守默认（**当前代码**）：**照协议字面实现**，因为任务卡把公式逐符号列为交付要求；权重常量集中在 `config.MASK_IOU_W/MASK_BCE_W/MASK_BF1_W`，`losses.mask_loss` 的 `iou_kind` 也可切换。
- 若主 agent 判红线优先：把 `MASK_IOU_W` 项换成 Dice 或 MSE 是**一处**改动，`soft_iou` 仍作为**报告**指标保留。
- 与 Where-A 的 D2 是同一件事的两半：Where-A 拿 MSE 训 B、拿 soft-IoU 拟 oracle；Where-B 目前拿 soft-IoU 训 connector。**建议两阶段统一裁决。**

**D-B2 · `H_where` 取 final-norm 前还是后。**
V-B2 已实测：`hidden_states[-1]` 是 **final RMSNorm 之前**的残差流。协议只说"`<where>...</where>` 全部 token hidden"。
- 保守默认（**当前代码**）：`WHERE_HIDDEN_FINAL_NORM = True`，即取 `norm(hidden_states[-1])`——lm_head 看到的那个状态，尺度已归一化，也正是 SFT 损失塑造的那个空间。
- 代价：若主 agent 想要"原始残差流"，改一个布尔常量即可，但**必须在 8 个臂开跑前定**，否则跨臂不可比。

**D-B3 · generated context 缓存 token ids 还是 hidden states。**
- 保守默认（**当前代码**）：缓存 **token ids**（~200 B/样本），训练时用**与 teacher 完全同一个函数** `FrozenVLM.encode` 重放。这让"两种上下文的 hidden 抽取口径一致"变成**结构性成立**而不是纪律性成立；代价是训练时多一次文本前向（视觉前向本来就要跑）。
- 备选：缓存 hidden（T×2560×2 B ≈ 200 KB/样本，全量 8–15 GB）。省一次前向，但 hidden 由**另一次调用**产生，teacher/generated 的可比性就只能靠人守。
- **vLLM**：V-B7 证实本机 conda env `vllm` 有 0.16.0 且支持 `Qwen3VLForConditionalGeneration`。**默认不用**——它在另一套 torch/transformers 上，产出的 ids 与训练栈可能不一致，且它给不出 hidden。若墙钟成为瓶颈可切换，属于纯吞吐优化。

**D-B4 · `L_mask` 在哪个分辨率上算。**
- 保守默认（**当前代码**）：`MASK_LOSS_SPACE = "hi"`，即经**唯一一次** guided upsample 到 spec-5 图像网格（512×768 量级）再算三项。理由：3px 边界容差在 32×48 的 `F_pre` 网格上没有意义（3 格 = 48 图像像素），且 §5.6 的 gate 是对着 `.cgt` 说的。低分辨率指标仍然一并记录。
- 代价：每样本多一次 guided filter（若干 box 滤波），以及 512×768 的 loss 计算。显存/吞吐**待 GPU 实测**。

**D-B5 · global（g1–g4）是否进 Where-B 训练。**
- 保守默认（**当前代码**）：**进**（`INCLUDE_GLOBAL = True`）。§5.6 的 gate 有一项 "global mask soft-IoU ≥ 0.98"，没见过 global 样本的模型不可能达到；§2.1 也写"所有实验使用相同的数据范围"。
- 实现后果：global 样本 GT mask 恒为全 1（V-B9），**没有** Where-A oracle latent，因此 `L_s/L_curve/L_dir` 对它们**屏蔽**，只有 `L_mask` 生效。`boundary_F1` 在"两边都无边界"时定义为 0（不是常数 1），否则每个 global 样本会白白扛一个无梯度的常数罚项。
- 若主 agent 决定 Where-B 只训 local，`INCLUDE_GLOBAL=False` 一行，但那条 gate 就必须从 §5.6 删掉或改判据。

**D-B6 · shuffled 的分组口径。**
协议写"同一图像、同一局部层级内交换"。两种读法实测差别很大（V-B8）：
- 保守默认（**当前代码**）：`(source_image_id, render_mode)`，V_where 上 local **4.5% 无配对**；
- 更严的 `(source_image_id, build)`：**23% 无配对**。
无配对样本**不跨图凑对**，而是记为 uncovered 并在报告里给 coverage（跨图配对会把"是否听指令"弱化成"是否对随机指令有反应"）。`config.SHUFFLE_GROUP_KEYS` 可切换。

**D-B7 · shuffled 用 partner 的 GT 文本还是 partner 的 generated 文本。**
默认用 **GT**：这样 shuffled 与 GT 板之间**只差"是不是自己那条"**，不额外引入"生成质量"这个混杂变量。协议未指明。
（原行文写作"只换指令内容"，与当时的代码相反，已由审阅 B3 指出并在 2026-08-05 更正；换什么由 **D-B15** 裁定。）

**D-B8 · 残差 gate 的形状与作用范围。**
协议只说"各 cross-attention residual gate 采用 zero initialization"。默认：**只有两个 cross-attn 带 gate**（self-attn 与 FFN 是普通残差），gate 是**每 block 每分支一个标量**。备选是 per-channel（LayerScale 式），更灵活但偏离"gate"的字面。

**D-B9 · weight decay 的排除集。**
§10.4 明写 "geometry、bias、LayerNorm 和 ModLN 不做 weight decay"，§10.3 没写。默认按同一惯例把 **bias / LayerNorm / gate / 可学习位置 / pool probe** 排除（`NO_DECAY_ON_BIAS_NORM=True`）。

**D-B10 · soft-IoU 用 min/max 还是 product 形式。**
默认 **min/max**（PLAN v2 L205 / E2 / Where-A `soft_iou_minmax`），这样 loss、gate 与"相对 oracle 的比值"三处是同一个数。

**D-B11（排期项）· train 段的 oracle latent 谁来产。**
V-B11：Where-A 目前只发布 `V_where` 的 latent，而 §5.5 的三个辅助 loss 需要**每个训练样本**都有 `w*,ρ*`。已在本包内补 `scripts/make_oracle_latents.py`（复用 Where-A 的 `fit_latent`/`pack_oracle`，用冻结的 `BA-3-Joint` B，不改 `q3vl/where/` 任何文件）。
**这是 Where-B 的硬前置，需要 GPU 时间，请主 agent 排期**：每图一次视觉前向 + 每 readout 一次多起点 L-BFGS。Where-A 的 NOTES 记录 CPU 上 1536 点约 2 s/图/readout；GPU 上更快但仍是主要开销。**墙钟待实测（见 PENDING 的 S2）。**

**D-B12 · boundary 容差 3px 的实现口径。**
原文的 `θ` 是 pooling 窗口（论文取 5–7）。"3px 容差"默认解释为**半径 3**（窗口 7）；另一种解释是窗口 3（半径 1）。`config.BOUNDARY_TOL_PX` 可切。

**D-B13 · "1 epoch" 与 "50/50" 的相容读法。**
默认：数据集一次性切成两个**不相交**的一半，teacher 半与 generated 半交错成 batch。于是"每个样本一个 epoch 只见一次"（§10.3）与"每个 batch 恒 50/50"（§5.4）同时成立。备选是每个样本两种上下文各见一次（等于 2 个 epoch 的样本量），与 §10.3 冲突。

**D-B14 · `winner_confidence=low`。** 按任务卡第 7 项的主 agent 裁定**保留**（`EXCLUDE_WINNER_CONFIDENCE_LOW=False`），并在每份报告里单独分层。

### 已裁定（2026-08-05，随 `REVIEW-impl-WhereB.md` 一并下达）

**D-B15 · shuffled 同时交换 instruction 与 `<where>` 正文，成对取自同一 partner。**（审阅 B3）
`Q_where` 只读 `H_where` + `F_pre`，而指令**唯一**的入口就是 `H_where`（因果注意力下 `<where>` 位置会 attend 到 prompt 里的指令）。
只换 `<where>` 正文、留着本人的指令，等于让正确答案仍然可达：一个"完全靠正确指令、不看 `<where>` 推理"的模型降幅接近 0 会被 §5.6 第 6 行 gate **误杀**，一个靠图像显著性作弊的模型降幅也接近 0 —— 这条控制两头都抓不住，与 §5.4 写的目的正好相反。
落地：`WhereContext.instruction`（**只有** shuffled 模式允许非空，构造时强制）；`_PromptShim(s, instruction=...)`；`ShuffleIndex` 要求每条 record 同时带 `where` 与 `instruction`，缺一即拒；`shuffled_context(tokenizer, partner_id, partner_where, partner_instruction)` 两半必须来自**同一个** partner（错配的 (A 的指令, B 的 where) 是模型训练时从未见过的第三种东西，测不出指令依赖）。
单测：`test_context.py` 4 条 + `preflight.check_context_flows` 3 条断言。

**D-B16 · oracle 辅助项的分母 = 有 oracle 的样本数，不是 batch 大小。**（审阅 B5）
`AUX_DENOMINATOR = "with_oracle"`。原实现对全 batch 取平均，而 global 样本没有 oracle，于是 §5.5 的名义权重被**实测 47.45% 的 local 占比**打折：stage-1 的 `1.00 L_s + 1.00 L_curve` 实际跑成 `≈0.47`，比 §5.5 给 stage-2 定的 0.25 更接近 stage-2，两段 schedule 的对比被压扁——而 stage-1 存在的全部理由就是"用大权重的 oracle 监督压住早期坍缩"。
落地：`SampleLoss` 拆出 `mask_term` / `aux_term`；`aggregate` 用 `mean(mask) + sum(aux)/n_with_oracle`；`"batch"` 档保留仅为复现该效应。
`steps.jsonl` 每步记录 `n`、`n_with_oracle`、`oracle_fraction`、`aux_denominator`、`aux_effective_scale` 与 `effective_aux_weights`——**任何未来的稀释都会写在日志里，不会再是隐形的**。
单测：`test_losses.py` 6 条 + `test_precision.py` 1 条。

---

## 五、代码地图

```
q3vl/whereb/
  config.py     所有冻结常量（§5.1/5.2/5.3/5.4/5.5/5.6/10.3）+ 3 个 dataclass 配置
  qwhere.py     §5.1  MetaCanvas query bank（可学习 2D 位置）+ 真实宽高比位置编码
  connector.py  §5.1  6 个 pre-norm block、512/8/2048、四步顺序、**zero-init cross gate**、
                      全掩码 key 返回 0 而非 NaN
  heads.py      §5.1/5.2  attention pool、joint/split head、只出全局 w0/w_dir/alpha/rho
  model.py      §5.2  四种结构装配 + 参数清单 + 输入白名单 MODEL_INPUT_KEYS
  fields.py     §4.2  冻结 basis（buffer）、phi_dir_fast（与 Where-A 逐位一致）、
                      s=3tanh(q/3)、一次 guided upsample、oracle s*/r*(z) 重算
  losses.py     §5.5  softIoU / balanced BCE / boundary-F1(arXiv:1905.07852) / L_s / L_curve /
                      L_dir / 两段 schedule / 分上下文聚合
  context.py    §5.4  四种上下文、固定边界截断（禁回退 GT）、ShuffleIndex、50/50 采样器
  hiddens.py    §5.1/14.8  **唯一的前向契约**：一次前向同时给 F_pre 与 H_where；
                      LastLayerHook（省 0.6 GiB）；批量贪心生成
  stores.py     §2.3  oracle / maskview / genwhere 三种已发布 shard 的随机读
  data.py       §14.9  Dataset（META_KEYS 白名单，image.baked 进不来）+ BatchBuilder
  metrics.py    §5.6  soft-IoU / boundary-F1 / AUC / 9 项 gate / lexicographic 选择
  trainer.py    §10.3  AdamW 2e-4 + warmup 3% + cosine + clip 1.0 + bf16 + GAS + eval/save 500
  evaluate.py   §5.4/5.6  四上下文分开评测 + strata + per_sample.jsonl + gate 板
  preflight.py  §14 项 7/8/9（+7b 抽取契约、8b 因果独立性）
  scripts/
    make_generated_context.py  ⏸ 生成 <where> 段并发布 indexed shards（未跑）
    make_oracle_latents.py     ⏸ train 段 oracle latent（未跑，D-B11）
    run_where_b.py             ⏸ 单臂全量训练（未跑）
    run_where_b.sh             ⏸ 提交入口，内置 D-20 四步
  tests/                       170 个用例，全绿
```

---

## 六、红线自查

| 红线 | 本实现 |
|---|---|
| 全局仿射 G 初始化 = 0 | 不适用（Where-B 不含渲染器）。对应位置是 **cross-attn gate 初始化为 0**，且实测初始输出与输入无关 |
| σ 参数化禁裸 exp（有界 sigmoid） | readout 参数全部走 Where-A 的 `bounded_sigmoid`；head 只产 raw，边界在 `q3vl/where/readout.py` 里，Where-B 没有第二条通路。单测实测 bias 初值在界内 |
| s 轴禁平滑正则 | 全代码无任何 s 轴正则项；`L_s` 是对 oracle `s*` 的 Huber 监督，不是平滑项 |
| 逐像素算子禁 (x,y)/邻域/MLP/排序 | 约束的是渲染器。Where-B 的 `phi_dir` 含 geo5 是 §4.2 的 basis 定义本身；guided upsample 是 §4.2 明文允许的**唯一一次**邻域操作，且只作用于标量 |
| checkpoint 选择禁用 val loss | `WhereBTrainer.best()` 按 `local_soft_iou_median`（generated 上下文板）选；`eval_loss` 只记录。单测构造了"eval_loss 单调下降但指标峰值在中间"的场景验证 |
| s 禁逐图 min-max/softmax 归一化 | 没有。`s = 3tanh(q/3)` 是全局有界映射；逐图标准化只作用在 §4.2 明文要求的输入特征 L/S 与语义块上 |
| 每个消融行必带 Δ_const/Δ_shuffle | `evaluate_arm` 强制出四上下文（null = Δ_const 的角色，shuffled = Δ_shuffle），并派生 `instruction_shuffle_iou_drop` 与 `gt_generated_iou_gap` 两个 gate |
| IoU 禁当优化目标 | **见 D-B1，与协议 §5.5 直接冲突，已上报未拍板** |
| VLM 干预对象 = 整段 image tokens 非 last token | `F_pre` 是整张 `H/16×W/16` 网格全部 token；`H_where` 是 `<where>...</where>` **全部** token（含两个标签），不是 last token |
| 烘焙一致性一等指标 | 属于 Stage-What |
| attention 导出必须 eager | 本阶段不导出 attention。`FrozenVLM` 用生产 `attn_implementation`（FA2）；§13 的 attention 可视化届时必须单独用 eager 重跑，已写进 PENDING |

---

## 七、给实现审阅的提醒（最容易出错的四处）

1. **`hidden_states[-1]` 是 pre-final-norm**（V-B2）。取错不会报错，只会让 `H_where` 的尺度差一个 RMSNorm，connector 的输入投影会被迫补偿——这类错误只会表现为"效果差一点"。已写成断言。
2. **null 上下文的全掩码 softmax**。默认写法会得到 NaN 并污染整个 batch 的梯度；`MultiheadAttention` 把无有效 key 的行改成全有效再把输出置零，等价于"这条 cross-attn 分支不贡献"。有专门单测。
3. **禁回退 GT 是结构性的**：`generated_context` 的签名里没有 GT 文本参数（preflight 会检查签名）。任何"找不到生成结果就用 GT 顶上"的补丁都会让 GT/generated 的 gap 指标失去意义。
4. **`phi_dir_fast` 与 Where-A 的 `build_phi_dir` 必须逐位相同**。我为省两次 71×71 SVD 复制了数值路径，用逐位相等的单测钉住；Where-A 若改 `phi.py` 的数值部分，这个测试会立刻红。

---

## 八、第二轮（BLOCKER 修复）的验证记录

### 8.1 单元测试：**213 个全部通过**（原 170 + 新增 43），23 s

新增/改动的测试文件：

| 文件 | 数量 | 覆盖 |
|---|---:|---|
| `test_precision.py`（新） | 15 | **先断言"没有守卫时 matmul 确实会被降成 bf16"**（防止这条测试退化成恒真）；`s_from_params` 在 autocast 内外**逐位相同**；`predict_fields` 的 `s_low/m_low/s_hi/m_hi` 在 autocast 内外逐位相同；**CBand12 在 σ 下界附近对 bf16 的敏感度实测存在（Band 不敏感）**，而实装路径不产生该差异；`require_dtype` 拒绝错误精度；`compute_batch` 在 autocast 下 `s_dtype == torch.float32`；`aux_effective_scale` 上报 |
| `test_contracts_and_preflight_wiring.py`（新） | 12 | `contracts.py` 是唯一定义点，**全包扫描禁止再声明**；`FrozenVLM` 默认值来自 contracts；`REQUIRED_CHECKS` 含两项 model check；**缺失必需检查时 `ok=False`**（修复前是 `True`）；skip 显式可见且 `complete=False`；model check 加载失败时产出 **fail 行而不是消失**；CPU driver 的 skip 落盘 |
| `test_dataset_wiring.py`（新） | 12 | `WhereBDataset` 无 mask 来源时**在任何 IO 之前**抛错；`need_mask=False` 的 local 样本 `mask_target_hi()` 抛错而非伪造全 1；global 仍返回全 1；**AST 扫描三个脚本：必须走 `open_dataset()`，不得裸构造 `WhereBDataset`**；生成作业必须 `need_mask=False`；genctx 覆盖率断言；`shuffle_records` 同时带两半 |
| `test_context.py`（+4） | 26 | shuffled 同时交换指令；**只有 shuffled 允许覆盖指令**；空/空白 partner 指令被拒；`ShuffleIndex` 缺任一半即拒 |
| `test_losses.py`（+7） | 30 | 辅助项按有 oracle 的样本数归一（`n=4, n_oracle=1, aux=4` → `total = 1+4` 而不是 `1+1`）；`"batch"` 档复现 0.25 稀释并把它写进 `aux_effective_scale`；mask 项仍按全 batch 平均；全覆盖时两档等价；未知分母被拒；`SampleLoss` 两项可分 |
| `test_evaluate_and_trainer.py`（改） | 10 | 归因说明块随每份 board 落盘 |

### 8.2 B2 复现证据（真实 `V_where.index.jsonl` + 真实 shard，CPU，不占 GPU）

`logs/b2_make_oracle_latents_smoke.log`（exit 0，11.7 s）：

```
dataset.mask_source = "live_mask_resolver"
  maskview_unavailable = FileNotFoundError: .../where_a-20260805/maskviews/V_where 未发布
n_ok = {band: 4, cband12: 4}     n_rejected = {}     ok_rate = 1.0 / 1.0
manifest.status = complete, sample_count = 4, shard_count = 1
verify = {n_random_checked: 4, checksum_failures: [], ok: true}
```

`logs/b2_make_generated_context_smoke.log`（exit 0，模型加载 + 生成 19.4 s）：

```
dataset = {need_mask: false, mask_source: null, n_samples: 2}   # 不再解 .cgt.png
manifest.status = complete, sample_count = 2, shard_count = 1
store = {n: 2, ..., index_rows: 2}                              # 发布后能读回
```

⚠ 该 smoke 用的是 **base（未 SFT）权重 + `--max-new-tokens 8`**，所以
`format_failure_rate = 1.0`、`starts_with_where_open_rate = 0.0` 是**预期**的——base 模型从没见过那四个 special token
（与 `s0_preflight/joint` 的生成管线检查同一结论：结构率全 0）。**这两个数只证明路径通，不是质量信号**；
真实数字要等 S2 用 `checkpoint-4976` 全量跑完。

两个脚本在修复前分别是"第一个 local 样本即 `RuntimeError`"和"train 段撞上第一个 local 样本即崩"。

### 8.3 preflight（CPU 半）

```
[PASS] WB-P7-context-flows      （含 shuffled 交换指令的 3 条新断言）
[PASS] WB-P8-no-h-color
[PASS] WB-P9-no-target-leak
[PASS] WB-P-zero-init-gates
[PASS] WB-P-param-table
[SKIP] WB-P7b-hidden-contract               --skip-model (needs a real VLM forward)
[SKIP] WB-P8b-h-where-causal-independence   --skip-model (needs a real VLM forward)

preflight PASS  (complete=False; skipped=['WB-P7b-hidden-contract', 'WB-P8b-h-where-causal-independence'])
```

**`complete=False` 是这一轮最重要的一行**：修复前，同样这条命令打印 `PASS` 且 `n_skip=0`，
读的人会以为 §14 项 7b/8b 已经过了。现在 skip 必须显式出现，`complete` 只有七项全 pass 才为真，
而 `PREFLIGHT_WHERE_B_PENDING.md` 的 S5 放行条件绑定的是 `complete`，不是 `ok`。

### 8.4 既有套件回归

`pytest q3vl/where/tests q3vl/tests -q` → **120 passed**（Where-A 正被 WA-IMPL 修复中，本次只读不改）。
