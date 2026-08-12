# E3 · 核实记录 / 假设 / 待决策

## 一、实测核实

| # | 事实 | 核实方式 | 结果 |
|---|---|---|---|
| A1 | `eval_final/per_sample.jsonl` 已有 7 个上下文 x 400 local | 直接统计:gt/generated/null/irrelevant_words/fixed_phrase/antonym 各 896 行,shuffled 856 | **已有的是指标,不是场**。(i)(ii) 是关于输出**形状**的问题,per_sample 里没有场,必须重跑前向 |
| A2 | 已有 board 的 grid_soft_iou 中位(W01) | gt 0.550 / generated 0.533 / **antonym 0.547** / shuffled 0.405 / null 0.409 / irrelevant 0.312 / fixed_phrase 0.320;中心先验 0.517 | antonym 与 gt 差 0.003 —— 与本卡场级测到的 0.31% 完全同向,**两条独立证据** |
| A3 | 加载 checkpoint 的正确姿势 | `WhereBModel(arm_config(arm))` + `torch.load(...)['model']`;`strict=False` 并打印 missing/unexpected | 本卡实测 missing=0 unexpected=0(日志可查) |
| A4 | shuffled 换的是「指令 + where 文本」整对 | `context.py:281-300` 明文,且拒绝缺指令的 partner(review blocker B3) | 因此 shuffled 是**表面内容**对照,antonym 才是**语义极性**对照。两者差 150 倍正是本卡的核心读数 |
| A5 | 跨样本比较需要共同栅格 | F_pre 栅格逐图不同(长宽比) | 统一 area 插值到 16x16;逐图相关仍在原生栅格上算,不混用 |

## 二、决策(保守默认)

| # | 决策 | 取值 | 理由 |
|---|---|---|---|
| D1 | 上下文取 gt / antonym / shuffled 三档 | 不跑全部 7 档 | **(勘误)** antonym 是**不变性**控制不是 M4 探针;shuffled 是换主体的条件性探针。其余 4 档已在 board 上有指标,M4 的判读改用 board(见待决策 B) |
| D2 | 「输出是否常数」用**跨样本配对相关**,并**并排 GT 的同一读数** | 阈值 0.9 且需高出 GT 0.2 | 单看「输出相关 0.53」无法判读——GT 自己也有 0.23 的自相似度。不并排 GT 列就会把「目标本来就有共性」误读成「模型塌缩」 |
| D3 | M4 判据用**相对场幅度**而非绝对 L1 | <1% | 绝对 L1 依赖场的量纲;相对值才可跨臂比较 |
| D4 | 未做「去先验输入重训」对照 | 跳过 | 登记的 E3 第三腿之一。W01/W02 **没有显式先验场输入通道**(它们是 MetaCanvas 头吃 VLM hidden),「去掉先验通道」在这两个臂上不存在可操作对象。已在 REPORT 中以「corr(输出, 中心先验)」替代,并说明这是几何先验而非输入通道 |

## 三、待主 agent 决策

### A. M3 的形态改写,是否接受

登记的 M3 判据写的是「输出约常数 **或** corr(先验) >> corr(GT)」。实测是**后者成立、前者不成立**
(输出跨样本相关 0.525,GT 参照 0.233)。本卡的读法是「M3 成立,但形态是**方差压缩 + 几何先验主导**,
不是常数塌缩」,修法因此不同(见 REPORT 3-1)。需确认接受该改写。

### B.(**已勘误,撤回初稿建议**)antonym 不是 M4 探针

**初稿错误**:把 `|gt − antonym|` 约 0 读成「M4 成立」,并建议 P1 把配对反义指令升为一等训练信号。
**方向读反了。**

核实结果:`q3vl/whereb/antonyms.py` 模块头写明这是**不变性控制**——只翻颜色方向词
(darker↔brighter / warmer↔cooler / saturated↔desaturated),**主体短语不动**;
Where 的掩膜按设计是主体的函数,所以「掩膜不动」是 **PASS**。
`q3vl/whereb/metrics.py:282` 预注册 `|Δ| ≤ 0.05`,并明文「negative-control column, **not a gate**」。
W01 已落盘 board:`antonym_invariance.median_abs_delta = 0.0`、`within_threshold = true`
—— **该控制本来就是通过的**。

**撤回的建议**:不要为 antonym 造分离/互补损失项。那会主动逼模型让掩膜依赖颜色词,
**破坏一个当前通过的不变性控制**,并与「主体决定掩膜」的协议定义冲突。

**改正后的读数**:M4 **不成立**。正确探针是换主体档,W01 board 的 grid_soft-IoU 中位:
gt 0.550 / shuffled 0.405(−0.145)/ fixed_phrase 0.320(−0.230)/ irrelevant_words 0.312(−0.238)。
指令确实在条件化输出。**真正的病是 gt 只比零参数中心先验(0.517)高 +0.033** —— 那是 M3。

已同步修正:`REPORT.md` §2(iii)/§3/§4/§5、`metrics.json`(加 `ERRATUM_2026-08-10`,
并把 `M4_no_conditioning` 改名为 `..__RETRACTED`)、`run_amort_e3.py`(判据改为
`antonym_invariance_pass` + `conditioning_responds_to_subject_swap`,防止复跑再产出错误结论)。

### C. 建议给 P1 增补的预注册证伪条款

「训练后若 `corr(输出, 中心先验) > corr(输出, GT)` 仍成立,则该臂与 W01/W02 同病,不得晋级。」
本卡已把这条的测量代码跑通(可直接复用 `run_amort_e3.py::_analyse`)。**是否登记请裁定。**

## 四、其它

- 运行 9.0 分钟,单卡(cuda:0)与在跑的 C01/C02 共存,峰值显存未触及上限,**未动队列**。
- 产物:`metrics.json`、`fields/{W01,W02}_gt.npz`(gt 档全部 400 个预测场,供后续复用)、
  `viz/`(24 张)、`config/run_setup.json`。
- 本卡未产生任何 AUC。
