# PR-ATT1-E1 · 实施前核实记录 / 假设 / 待主 agent 决策

## 一、读过的文档节

- `docs/WHERE_HEAD_REDESIGN_PROPOSAL_2026-08-10.md` §2.2(信息通路)、§4 P-W1 / P-W3
- `docs/WHERE_HEAD_REDESIGN_DELTA_2026-08-10.md` 全文(含运行期新增的 §五「模型侧调整清单」)
- `docs/WHERE_HEAD_REQUIREMENTS_2026-08-10.md` §1.4、§2.1–2.5
- `CLAUDE.md`:红线速查、空间场可视化纪律、AUC 全实验禁用、长任务提交纪律、s 缓存消费契约

DELTA 优先于 PROPOSAL(派工令)。本卡只做 **raw 臂**(仅 sink 排除,无差分、无共模减法),
差分臂不在本卡范围。

## 二、在线/实测核实记录

**一律实测,零检索引擎引用。** 本项目检索引擎有编造前科,所以下面每条都写明「怎么测的」。

| # | 事实 | 核实方式 | 结果 |
|---|---|---|---|
| V1 | 文本栈 36 层 / 32 query 头 / 8 KV 头 / hidden 2560 / head_dim 128 | 读 checkpoint-4976 `config.json` + 运行期 `assert_model_facts` 断言 | 确认,与 PROPOSAL 一致 |
| V2 | `image_token_id=151655`,`<where>`=151669,`</where>`=151670 | `config.json` + `added_tokens.json` + `tokenizer_config.json` 的 `added_tokens_decoder`(`special:true`) | 确认 |
| V3 | image token 网格 = (out_h/32, out_w/32) | `vision_config.patch_size=16` × `spatial_merge_size=2`;并与索引的 `n_visual_tokens` 逐样本断言相等 | 确认(384 = 16×24) |
| V4 | eager 才有 attention;FA2/SDPA 返回 None | 读 transformers 4.57.1 `modeling_qwen3_vl.py:440` —— `attention_interface = eager_attention_forward` 仅当 `config._attn_implementation == "eager"`;`eager_attention_forward` 恒返回 `attn_weights` | 确认。`AttentionTap` 每层断言非 None,**不回退**(红线) |
| V5 | GQA 是否需要手工展开 | 读 `eager_attention_forward:152` —— `repeat_kv` 在 matmul **之前**调用 | `attn_weights` 已是 32 个 query 头,无需展开 |
| V6 | `output_attentions=True` 会不会更省 | 读 `_can_record_outputs`(line 558)与 `check_model_inputs` | 它会保留全部 36 层;改用逐层 forward hook 当场切片,实测峰值显存 **8.9 GiB** |
| V7 | transformers / torch 版本 | `q3vl_sft` env 实测 | 4.57.1 / 2.10.0+cu128,与 `pyproject.toml` 钉版一致 |

### V8 —— **PROPOSAL 的一处事实错误(deepstack 注入层)**

PROPOSAL §2.2 与「问 3」写:*「deepstack 在 LLM 第 5/11/17 层二次注入视觉特征,层扫描必须全 36
层做、可预期注入层之后出现读出峰」*。

实测 `Qwen3VLTextModel.forward`(modeling_qwen3_vl.py:862):

```python
if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
    hidden_states = self._deepstack_process(...)
```

`len(deepstack_visual_embeds) == 3` → 注入发生在 **language 层 0 / 1 / 2**。
`config.vision_config.deepstack_visual_indexes = [5, 11, 17]` 是**视觉塔 block 下标**,即特征
**取自**哪三层,不是注入到 LLM 的哪三层。

**影响**:①「注入层之后出现读出峰」的预期落在层 3 以后(几乎全部层),不再是一个有区分度的预测;
② 预注册的「排除前 2 层」过滤恰好落在注入区内(删掉层 0/1,留下层 2)。我**按预注册执行不改**,
把层 2 单独标注,避免事后改规则。本次选中的头全部在层 8–20,与该边界无关。

## 三、已采用的决策(保守默认,未静默拍板的都列在第四节)

| # | 决策 | 取值 | 理由 |
|---|---|---|---|
| D1 | grid soft-IoU 的口径 | `soft_iou(topk_mask(field, k_GT), gt_soft)`,仅 valid 格 | 任务卡明写「面积匹配 top-k」。对**原始 attention 场**直接算 soft-IoU 无意义:场的量级 ~1e-3、GT ~1,`sum(min)/sum(max)` 实际在报告「有多少注意力质量落在图上」而不是「落在哪」。top-k 二值化同时是红线指定的阈值化方式,且对任何单调变换不变 |
| D2 | GT 降采到 merged grid | `.masklow.npy` (out/16) → 精确 2×2 面积均值 → (out/32) | 复用 Where-A 已发布的低分辨率视图,一次精确面积均值,无插值、无重采样相位误差 |
| D3 | sink 判据 | 层/头均值列 profile 上 `median + 3·MAD`,**且** gt 与 shuffled 两档同时超阈(合取) | k=3.0 直接沿用 RO9b D-0 的 `MAD_K`(`tools/readout/ro9_gl_attention.py:53`),**不调参**——按结果挑 k 就是事后旋钮。合取式落实任务卡「∧ shuffle 上下文秩不变」:只在真指令下亮的格是信号,不该被剔除 |
| D4 | 列 profile 的 query 行 | 图像块**之后**的全部文本行 | attention 是因果的:把图像行算进来,靠前的图像列天然被更多 query 看到,会在 profile 上压出一条单调位置斜坡,让前几格假装成 sink |
| D5 | 逐头合成前的归一化 | 每 (layer, head) 一个 **fit 折整臂常量** (μ, σ),先乘 `n_img` 再标准化 | 各头「投在图上的总质量」差一个数量级,直接凸组合等于按质量加权。整臂常量是 s 缓存契约唯一许可的归一化形式;先乘 `n_img` 是确定性几何因子(非逐图统计量),否则同一个头在 16×16 与 16×64 上不同尺度,一个常量服务不了两者。**判据数字**:单头数字完全免疫(top-k 对单调变换不变);**但 8 头凸组合不免疫** —— 融合场的值依赖每个头的 (μ,σ),换一组常量就是**另一个场**,不是同一个场的单调变换。此措辞已按结果审阅 B4 更正(原文的免疫论证被过度推广) |
| D6 | `where_special` 池 | 只取 `<where>` 开标记行 | `</where>` 另作第 4 列**探索性**报告,不进 3 池 FWER 家族(DELTA 裁定 4 禁事后枚举复活) |
| D7 | 内容词过滤 | 保留含字母数字的 token,纯标点/空白剔除;过滤后为空则回退全部内段 token | 刻意做得很笨。精选停用词表是可调旋钮,而 query 池上的可调旋钮正是「枚举到出阳性」的路径 |
| D8 | fit/OOF 切分 | 按 `source_image_id` 分组,用旁表自己的 `sha1("verasplit-v1:"+source_id)` 规则定序,再贪心均衡 **local** 计数 | 见第四节 A —— 旁表切不出 448/448 |
| D9 | 只前向 local 400 | global 496 不跑 | global 无 GT mask(`mask_target_hi` 按构造返回全 1),进不了任何空间判据;跑它只是把墙钟翻倍去生产没人能打分的场。fit/OOF 仍在全部 896 行上定义 |
| D10 | 代码落点 | 新建 `q3vl/whereb/attnread.py` + `attnprobe.py`,未改 `hiddens.py` 的 `FrozenVLM` | DELTA §五-2 建议扩 `FrozenVLM`。但 `FrozenVLM.encode()` 的契约是「批量 → F_pre + H_where」,有既有测试;attention tap 需要 batch=1 + 逐样本行/列选择,塞进去会污染一个已被审阅过的契约。同级新模块复用 `load_model` / `Sft2SegCollator` / `context.py` / `metrics.py` / `viz.py`,风险更低。**若主 agent 要求合并进 FrozenVLM,是一次纯搬运** |
| D11 | 只取 post-softmax | 不导 pre-softmax | 本卡未要求;PROPOSAL §2.2⑤ 自述两版仅差 ±0.03。见第四节 C |

## 四、待主 agent 决策

### A.(已发生,采用保守默认)fit/OOF 切分:S/P 旁表切不出 448/448

任务卡要求「fit/OOF = 448/448 按 `source_image_id` 用既有 S/P 旁表切,禁 ad-hoc」。**实测该旁表不
含这个划分**:

- 旁表本体 `tools/data_splits/splits.sqlite3`,schema 为 `sources(source_id, split, pool)`,
  `split ∈ {train, val, test}`;
- V_where 896 条只有 **162 个** unique `source_image_id`,去 join 命中 160 个,分布
  **train 146 / val 10 / test 4**;
- 另有 **2 个 source 完全 join 不上**(`src_a96fe7d9865a6f08`、`src_d9091d3f6c248d9a`),因为旁表
  `meta.builds` 只记到 g1–g3 / l1–l4,而 V_where 含 g4/l5/l6 —— **旁表落后 3 个 build**。

即旁表里根本没有一个 448/448 的二分可读。采用的保守默认:**沿用旁表自己发布的规则族**
(`sha1(seed:source_id)`,`meta.split_seed = "verasplit-v1"`)给组定序,再按 local 计数贪心均衡,
组绝不拆分。实得 **local 200 / 200**(总计 fit 435 / oof 461 —— 组完整性优先于总数配平)。

组完整性在这里是硬要求:896 条样本只出自 162 张源图,逐样本切会把同一张照片的近重复放到两侧,
把每个 OOF 数字都抬高。

> **请裁定**:接受此规则,还是要求旁表增量重跑到 g4/l5/l6 后重切?本实验结论不依赖于此
> (P2/P3 都是**同图配对**检验,配对检验对切分方式不敏感),但后续卡应统一。

### B. shuffle partner 覆盖不全 → 18 条 local 样本被排除

`ShuffleIndex` 的分组键是 `(source_image_id, render_mode)`,组内只有 1 条的样本拿不到 partner
(全 split 覆盖率 0.955)。local 400 里 **18 条**因此没有 shuffled 档,已整条排除(P3 是配对检验,
只有一臂的样本进不去)。最终 **fit 193 / OOF 189**。

> **请裁定**:是否接受「无 partner 即排除」?替代做法是跨图取 partner,但那会同时改变图像内容与
> 指令,污染 P3 的语义(它要测的是「同图换指令」)。已采用排除法。

### C. pre-softmax 档未导

PROPOSAL §2.2⑤ 与 DELTA §五-6 把「pre/post-softmax」列为待探针裁决的分叉。本卡的判据表没有要求
它,且 PROPOSAL 自述两版差 ±0.03,而本次实测的缺口是 **−0.03 ~ −0.05(方向为负)** 且
Δ_shuffle ≈ 0 —— 一个 ±0.03 量级的变体改不动「指令无关」这个结论。

> **请裁定**:是否需要补 pre-softmax 档?建议**否**(理由见上;需要的话改一行 monkeypatch
> `eager_attention_forward` 即可,再跑 6 分钟)。

### D. GPU 编排偏离(已发生,附理由)

任务卡写「GPU 任务经 gpu-queue(q 命令)提交,2×H100 当前全空」。**实测两卡都不空**:W01(pid
43355)与 W02(pid 66290)仍在跑,各占 27.9 GiB,已跑约 7 小时;且 `q status` 显示 `gpu0` 组处于
**PAUSED**。

向一个 paused 组提交,作业会静默不启动 —— 正是 D-20 纪律要防的失败模式。故本次 6 分钟、8.9 GiB
的推理作业按 D-20 四步**直接提交**(`rm -f` 日志 → `ps -p` 实证存活,未用 `pgrep` → `tail` 确认
实质输出 → 写 `job.marker`),未动 W01/W02,未改队列状态。marker 在 `logs/export_full.job.marker`。

> **请裁定**:后续卡是否要求先 `q resume gpu0`。

### E. 任务卡对 genwhere 的描述与磁盘实际不符(已按实际执行)

任务卡写 genwhere shards 是「teacher-forced `<where>` 文本 token ids:gt 档 + shuffled partner 档
(负控制)」。实测:

- `where_b-20260805/genwhere/` 存的是 **generated** 档(`gen.do_sample=false`、
  `forced_prefix_ids=[]`),不是 teacher-forced;
- 磁盘上**没有任何** shuffled / fixed_phrase / irrelevant_words 落盘档,七个 context 档位全是
  `q3vl/whereb/context.py` 的**运行期**构造。

故 gt 档走 sft2seg `.rec.json` 的 `where` 字段 → `context.gt_context()`,shuffled 走
`ShuffleIndex` + `context.shuffled_context()`(指令与 where 成对交换,protocol 5.4)。这**不是替代
方案**,而是训练臂用的同一条代码路径,探针因此不会与被探对象漂移。

## 四之补 · E1b 补件后新增/更正的待决策与事实(2026-08-10)

- **C(pre-softmax)已由审阅确认可不跑**,记为「未跑 + 理由」而非静默缺席(N7)。
  `</where>` 第 4 列(D6 承诺的探索列)同样确认未产出,一并记为未跑。
- **N1 路径不符**:本交付路径 `probe_e1_whereattn_20260810/` 与登记的
  `probe_pw3_headscan_20260810/` + `probe_pw1_sink_20260810/` 不一致。把 P-W1 并进同一次 prefill
  省了一次前向、且审阅认可该做法,但**计划文档的登记路径需同步更新**,否则后人按登记路径找不到。
  → **请主 agent 在 PROPOSAL §4 内改登记路径**(我不改权威文档)。
- **新增待裁定(来自 E1b/B2)**:shuffle partner 的分组键 `(source_image_id, render_mode)` 产生的
  partner 目标**重叠中位达 0.480**,质心距中位仅 0.071。这使 P3 的灵敏度被结构性稀释。
  → **后续任何用 P3 的卡,建议改用「pair-GT IoU < 0.3」的分离子集,或改分组键**。本卡已用分离
  子集复核(结论不变),但下一张卡不该继续用被稀释的全集口径当主读数。
- **新增判据学结论(来自审阅 R5,已进 REPORT §8-5)**:面积匹配 top-k 的随机地板 `a/(2−a)` 必须
  强制并排;本 eval 集有一半样本的地板(0.527)高于任何被测场,P2 的 +0.10 门在该档接近算术不可达。

## 五、其它可核查事实

- **merged grid 定向已实测,不是推断**:构造「顶部横带」与「左侧竖带」两张合成图,测其相对全黑图
  的 per-token 特征差。横带在 **行** profile 上聚集(峰在行 0–1,列方向平坦),竖带在 **列** profile
  上聚集(峰在列 0–1,行方向平坦)→ 确认 image token 展平顺序为 **(gh', gw') 行优先**。
  (第一版用原始特征范数做,不可判 —— 被 massive-activation 格支配;记此以免后人重踩。)
- **sink 规则的退化分支修正**:`MAD == 0` 时 RO9b 原版返回「无离群」,即**失败开放**——一条除两个
  巨峰外全平的 profile 会把巨峰留下。改为「严格大于 median 即离群」,失败关闭。真实 profile 上
  MAD>0,不会误触发。有单测。
- **valid mask 的代价已定量**:被排除格占 12.0%,同时带走 12.0% 的 GT 质量 —— 即 sink 相对 GT
  **近似均匀分布**,不偏向也不回避目标。oracle 天花板因此从 0.79(不排除)降到 **0.729**。
  P1 的 0.45 参考线 ≈ 天花板的 62%。
- 505 条既有 `q3vl/whereb` 测试全过,无回归;新增 **30** 条单测(`tests/test_attnread.py`,
  含 B4 的域断言三条:越界必抛、跨零可报、clamp 可检出)。
- **E1b 新增脚本**:`analyze_attn_diff.py`(差分臂 + P4 + P5 守卫 + 随机地板)、
  `analyze_attn_pairs.py`(B2 partner 分离度)、`analyze_sink_b3.py`(B3 口径统一 + 跨分辨率
  Jaccard)、`write_attn_norm_meta.py`(B4 域声明 + 消费断言记录)。
- **oracle 天花板有两种口径,勿混用**:`valid-only`(与场分数同支撑,可比)与
  `全格`(把 sink 上的 GT 质量算作损失,是「掩膜代价」的正确口径)。发布的 `oracle_ceiling`
  列是后者,这正是 N3 那 3 个「中心先验超天花板」样本的成因。
