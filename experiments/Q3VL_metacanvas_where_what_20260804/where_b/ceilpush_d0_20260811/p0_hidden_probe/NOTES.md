# P0 / P0a — NOTES（实施前核实记录 + 假设清单 + 待主 agent 决策）

实验：`<where>` hidden state 的逐层线性 probe（几何码读出）
脚本：`q3vl/whereb/scripts/run_p0_hidden_probe.py`
执行方式：**直接前台跑，未进队列**（两张卡都在训练，本作业须与之共存；实测峰值显存
9.0 GiB，micro-batch 4，`--device cuda:0 --attn eager --dtype bfloat16`）。

---

## 1. 读过的文档/代码节（实施前）

| 来源 | 读的是什么 | 用途 |
|---|---|---|
| `docs/RESEARCH_geometry-extraction-arch_2026-08-11.md` §4「P0a」、§3 读表结论 | P0a 的判读口径（probe ≥ 82% ⇒ C/B1 前提成立；< 70% ⇒ 降优先级）与它在三臂比较里的位置 | 确认本实验的验收语义 |
| `q3vl/whereb/amort/geomparse.py` | `GEOM_SLOTS` / `GEOM_DIM=20` / `geom_features` / `geom_features_from_vrmeta` | 标签空间与文本基线解析器 |
| `q3vl/whereb/hiddens.py` | `FrozenVLM.encode` 的 `use_hook=False` 分支、`LastLayerHook`、`resolve_language_model`；模块头两条已核实事实（`hidden_states` 有 `L+1` 项、`hidden_states[-1]` 是**最终 RMSNorm 之前**；`<where>` 位置的 hidden 与后续是否有 `<color>` 段**逐位相同**，因为因果注意力） | 逐层 hook + 池化的实现契约；`H_where` 到底指哪一个状态 |
| `q3vl/whereb/context.py` | `gt_context` / `generated_context` / `encode_where_span` / `extract_segment` | 两个 context 臂的 token 段构造，且 generated 段**不允许回落 GT** |
| `q3vl/whereb/amort/data.py::AmortBatchBuilder.build` / `_vrmeta_code` | prompt 构造范式（`Sft2SegCollator(proc, max_length=2048, system_prompt=None)` + `_PromptShim` + `enc["input_ids"][:n_prompt_tokens]`）、vrmeta 解析范式 | 保证 prompt 与训练同构 |
| `q3vl/where/maskdata.py::MaskResolver` | `resolve()` / `read_bytes()`；每 root 缓存一个 sqlite 连接 | 标签来源与线程安全边界 |
| `CLAUDE.md`：AUC 红线、s 缓存契约、长任务提交纪律 | — | 判据选择与运行纪律 |

## 2. 在线核实

本实验**未引用任何外部 URL / 论文数字 / 第三方超参**，全部输入都是仓库内已落盘的代码与
数据，因此没有需要打开原始来源核实的外部事实。所有仓库内事实均**当场读代码核实**，
未依赖检索：

- `GEOM_DIM == 20`、`GEOM_SLOTS` 顺序 —— 直接读 `geomparse.py`；
- `WHERE_HIDDEN_LAYER == -1`、`WHERE_HIDDEN_FINAL_NORM == True` —— 读
  `q3vl/whereb/contracts.py:30,32`（`config.py` 只是重导出）。这解释了
  `LastLayerHook` 为何能同时满足「hook `lm.layers[layer]`」与「等于
  `out.hidden_states[layer]`」：只在 `layer == -1` 时两种索引约定重合；
- `V_where` local = 400、家族分布 radial 100 / linear 112 / band 108 / semantic 80
  —— 与同目录 `../metrics.json`（D0-3 by_family）逐个吻合，本次实测再次一致；
- `sklearn` **确实未安装**（`ModuleNotFoundError`），probe 因此用 torch 自己实现。

## 3. 关键实现决定（及理由）

1. **标签只来自构造侧 `.vrmeta.json`**，不解析任何文本。`slot_id` → shape 家族，
   `region` → 方向槽（按空白切分，"lower right" 同时点亮 bottom 与 right）。
2. **extent 槽全部排除**。vrmeta 不含 extent，留 0 且**不进任何准确率/F1/组指标**；
   同样排除的还有 5 个纯文本槽（`shape_oval`、`dir_edge`、`dir_horizontal`、
   `dir_vertical`、`dir_diagonal`）。20 槽中只有 9 个被 vrmeta 填充，全部数字只针对
   这 9 个。已写进 `metrics.json.excluded_slots_note`。
3. **每线程一个 `MaskResolver`**（`threading.local()`）。它持 sqlite 句柄，共享一个会
   在负载下静默丢失大比例查询；本次实测 400/400 全部解析成功、skipped 为空。
4. **`RLIMIT_NOFILE` soft→hard 在 `main()` 开头抬起**（1048576）。
5. **逐层 hook + 立即池化**，不用 `output_hidden_states=True`：后者会分配
   `(L+1, B, T, 2560)` 再扔掉 37/38。每层 hook 内就 mask-mean-pool 到 `(B, 2560)`，
   峰值只留一层激活 —— 这正是它能和两个训练作业共存的原因（实测峰值 9.0 GiB）。
6. **调用 `model.model(...)` 而非 `model(...)`**，跳过 lm_head：B=4 / T≈2000 时 logits
   张量约 1.2 GiB，而这里没有任何东西读它。
7. **layer key 约定**：`0` = token embedding（`embed_tokens` 输出），`1..36` = 第 i 层
   decoder 输出（**最终 RMSNorm 之前**），`36_norm` = 最后一层过完最终 RMSNorm ——
   即 `q3vl.whereb.hiddens` 口径下的 `H_where`（生产读出真正拿到的那个状态）。
   RMSNorm 是逐位置非线性，**先 norm 再池化 ≠ 先池化再 norm**，所以 `36_norm` 是在
   hook 里对 token 级 hidden 做 norm 后再池化的。
8. **λ 由 train 半边内部的 3 折 CV 选**，test 半边全程不参与。n≈200 / d=2560 下固定
   一个 λ 等于抛硬币（欠正则与过正则的差别就是全部结论），而在 test 上调 λ 是另一种
   错误；CV-on-train 是唯一无泄漏选项，选中的 λ 逐层落盘。
   烟测（n_train=29）在 1e-3..1e2 的网格上**每层都选到网格上界**，说明网格被截断，
   正式跑改成 `1e-2 … 1e4`，使选择落在内点。
9. **必带的两条零信息对照**（AUC 红线的同源纪律：任何"我们读到了几何"的主张都必须
   出示零参数对照）：
   - **常数预测器**列（按 train 多数票 / train 最频家族）。方向组在本 split 上
     76.25% 是 `center`（305/400），没有这一列的 `dir_any` 完全不可读；
   - **打乱标签控制**列（train 半边内置换标签后重训，同 λ，同 test）。

## 4. 待主 agent 决策 ← **请看这一节**

### D-1（已采用保守默认，但结论口径必须由主 agent 裁定）：GT 教师强制臂天然近饱和

任务卡指定用 **GT（teacher-forced）** `<where>` 段，理由是"这测的是 hidden *能*
携带什么，即问题所求的上界"。这个定义没错，但它有一个必须写在结论旁边的性质：

> **GT 段的文本本身就在念答案**。probe 因此可以只做"把刚喂进去的 token 读回来"，
> 而不必证明模型"知道"几何。

这不是推测，烟测已经直接量到：**layer 0（纯 token embedding，零层 transformer 计算）
在 GT 臂上 shape_acc 就已经 0.90**。一个零计算的词袋能到 0.90，说明 GT 臂的高分里
有多大一块来自"输入即答案"。

因此我**在任务卡的 GT 臂之外，追加了一个 `generated` 臂**（模型自己那 82% 准确率的
`<where>` 段，来自已发布的 `GenContextStore(V_where)`，走 `generated_context`，
无 GT 回落）。两臂同样的 prompt、同样的样本、同样的 sha1 半分、同样的 9 个槽。

- **`generated` 臂才是与"文本 0.820"直接可比的那个**：它问的是"当模型只写出 82% 正确
  的文本时，它同一时刻的 hidden 里是否还留着比文本更多的几何"。这正是任务卡问题句
  「Does the hidden carry the geometry *more losslessly* than the sampled text does?」
  的字面含义。
- **`gt` 臂保留为上界**，但引用时必须与 layer-0 行一起引用，否则会把"读回输入"误报成
  "hidden 携带几何"。

两臂都已跑、都已落盘、都在 `metrics.json.verdict` 里各自给了 verdict。
**待裁定**：三臂比较的立项依据应以哪一臂为准。我的保守默认是
**以 `generated` 臂为准、`gt` 臂只作上界注脚**，但这属于影响后续排期的决策，不自行拍板。

### D-2（已采用保守默认）：probe 容量固定为线性

任务卡写"linear probe"，`RESEARCH_geometry-extraction-arch` §4 P0a 写"线性/2 层 MLP"。
本次**只跑线性**（多标签 one-vs-rest 逻辑回归）。理由：线性 probe 的读数才对应
"信息是否线性可读"这一可迁移到读出头的性质；MLP probe 抬高的分数不保证一个
attention readout / query bridge 能拿到。若主 agent 要 MLP 档，可在同一份特征上
加跑（特征提取是全部成本的大头，已可复用）。

### D-3（已采用保守默认）：交付物里没有 `viz/success_*` / `viz/failure_*`

CLAUDE.md 的交付物规范要求成功/失败案例并排可视化，但本实验**没有空间场输出**
（输入是 hidden 向量，输出是 9 个离散槽的概率），没有可并排的"输入/输出/GT/s 场"。
代偿：`per_sample_{gt,generated}.jsonl` 给出 test 半边**每个样本**的真值槽、预测概率、
预测/真值家族与 `<where>` 段原文，失败案例可直接按 `pred_shape != true_shape` 过滤；
`metrics.json` 里另有 4×4 的 shape 混淆矩阵。若主 agent 要求补图，请指定形式。

## 4bis. 跑完之后新增的两条发现（**会影响判据引用，必须看**）

### F-1：任务卡给的「direction any-overlap 0.582」**在本实验的标签口径下复现不出来**，不可用

实测（同样本、同 test 半边、同 9 槽）：

| 解析方式 | generated 段 dir_any | GT 段 dir_any |
|---|---|---|
| `geom_features` 原样（只看 `edit scope:` 从句） | **0.1717** | 0.1061 |
| 放开限制、整段都解析 | 0.3030 | 0.2424 |
| 任务卡给定值 | **0.582** | — |

**根因是语义错配，不是解析器 bug**（已逐样本核对）：vrmeta 的 `region` 说的是
**掩膜在画幅里的位置**，而 `<where>` 模板的 `edit scope:` 从句说的是**衰减几何**，
两者根本不是同一个量。实例：

```
region = center      scope: "a vertical band spanning the whole frame,
                             strongest toward the top edge and fading
                             continuously toward the bottom edge"
         → 解析出 dir_top + dir_bottom，而真值是 dir_center。两句都对，量不同。

region = lower left  span:  "subject: the lizard perched on the branch in the
                             lower-left; edit scope: stays within the lizard"
         → 方位词只出现在 subject 从句里，而 geomparse._scope() 按设计把 subject
           从句剥掉（避免 "the large dog" 的 large 误触 extent 槽）。
```

**结论**：`0.582` 与本实验任何 direction 数字**不可比**，我没有拿它做判据。
方向组的判据改用**常数基线 0.7576**（永远预测 center），这条本来就是必带列。

顺带：`shape any-overlap 0.820` **复现良好**（本次实测 generated 段全 400 = 0.795、
test 半边 = 0.7778，n=198 的抽样误差约 ±0.04），所以 shape 一列是可比的，
标签管线与解析器本身没问题。`shape exact-set 0.386` 也复现不出（实测 0.62），
预期原因是给定值算在全部 20 槽上、包含 vrmeta 从不点亮的 `shape_oval`。

### F-2：GT 臂的高分几乎全部来自「把喂进去的 token 读回来」

| | layer 0（纯 token embedding，零层 transformer） | 最佳层 | 整个 36 层的净增益 |
|---|---|---|---|
| GT 臂 | **0.9646** | 0.9848 (L9) | **+0.0202** |
| generated 臂 | 0.7273 | 0.7374 (L2) | +0.0101 |

这坐实了 §4 D-1 里事前提出的担心：GT 教师强制段的文本本身在念答案，probe 只要做词袋
就能到 0.96。**引用 GT 臂 0.9848 时必须同时引用 layer-0 的 0.9646**，否则会把
「读回输入」误报成「hidden 携带几何」。

### 配对检验（McNemar 精确二项，198 对）

| 比较 | Δ | 只有前者对 / 只有后者对 | p |
|---|---|---|---|
| **generated：probe(最佳层) vs 同段文本解析** | **−0.0404** | 6 / 14 | **0.115（不显著）** |
| generated：probe 方向 vs 永远 center | +0.0354 | 7 / 0 | 0.016 |
| GT：probe(最佳层) vs 同段文本解析 | +0.1111 | 22 / 0 | 4.8e-7 |

**读法**：在模型自己那 82% 的段上，hidden 与文本**打平**（差异不显著，且点估计还偏负）；
方向组比常数基线只多对了 **7/198** 个样本 —— 统计上可测、实质上可忽略，即 hidden 里
基本没有超出类别先验的方向信息。

## 5. 未做的事（明确声明）

- **没有做任何 AUC**（红线）。判据是 accuracy / F1 / exact-set / any-overlap，
  每一列都配了常数基线与打乱标签控制。
- **没有调参去够 0.82**：λ 由 train 内 CV 决定，网格与折数在跑之前固定，跑完未改。
- **没有动 VLM 任何权重**（全程 `requires_grad_(False)` + `torch.no_grad()`）。
- 本实验**不写 `REPORT.md`**（任务卡明确要求不写）。
