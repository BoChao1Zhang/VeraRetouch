# objective_edit_hints 的统计均值失真（2026-07-29 立档）

**状态**：已证结论，用户确认。本文自足——读者不需要任何会话上下文，也不需要先读
`DATABUILD_EVAL_AND_PERF_HANDOFF_2026-07-28.md`（那篇 §6/§7 是本文的来源，但结论
在这里被完整重述）。

**配套证据**：`/home/bc/VeraRetouch/_hints_bias` → `/var/cache/veradata/annot_review/hints_mean_bias_2026-07-29/`
（22 个样本目录，图 + 数 + 判词；见 §9）。

---

## 1. 命题

**现行 `objective_edit_hints` 的统计量——mask 支撑上 alpha 加权的 Lab 空间一阶
均值——对「观者感知的颜色方向」是失真的。**

一句话结论：**这不是计算 bug，不是标注模型不听话，也不是 prompt 措辞能修的问题；
是这个统计量的定义本身回答了一个观者不会问的问题。**

三条都被逐一排除，不是推测：

| 假设 | 判定 | 依据 |
|---|---|---|
| 标注模型没照 hints 写 | **排除** | fresh100 八条盲评 fail 的文本 **8/8** 忠实转写了 hints 的方向词 |
| hints 算错了（比如没按 mask 加权、被整帧稀释） | **排除** | 从归档像素按 `mask.effective_alpha` 复算，22/22 逐条吻合；88 个轴里 87 个符号一致，唯一例外是一个 ±0.04 的零值轴 |
| 措辞不够强硬，换个说法就好 | **排除** | 两轮 out-of-sample：v4.0 直陈 vs v4.1 veto 框架，组成标准化后 pass 49.3% vs 49.8%，统计上不可区分 |

剩下的只有定义。**空间均值 ≠ 观者所见。**

### 1.1 这个统计量到底是什么

`dataset_build/src/construct/visibility.py::objective_edit_hints_from_lab`
（第 45-92 行）。给定 before/after 的 Lab 对与权重图 `w`（local 组 = mask 的
effective alpha，style/global 组 = 全 1），四个数各是：

| 轴 | 定义 | 单位 |
|---|---|---|
| `brightness` | `Σw·(L₂−L₁) / Σw` | L\* |
| `warmth` | `Σw·(b₂\*−b₁\*) / Σw` | b\*（**只有 b\* 轴**） |
| `chroma` | `Σw·(C₂−C₁) / Σw`，`C = hypot(a*, b*)` | C\* |
| `contrast` | `√(加权 L\* 方差)` 的前后之差 | L\* |

然后 `responses.py::_objective_hints`（第 477 行）把每个数离散成一句话塞进标注
prompt：

- **v4.0**：|Δ| ≥ 1.0（四轴同一阈值）给「moderately / strongly + 方向词」；
  低于 1.0 给一句含糊的 "near-neutral or visually mixed"。
- **v4.1**：死区按轴分开（tonal 1.0 / colour 2.0），低于死区改为明确的
  "the shift is subtle and may not be visible -- do not state a direction either
  way"（旧文本被模型读成了发挥许可）；高于死区加反向锚定
  "do not describe it as <opposite>"。**度量本身一个字没改。**

**其中三个数是一阶矩（contrast 是二阶矩，同样是全局标量）。它们统统把「哪些像素
在变」这个信息丢掉了——而这恰恰是观者唯一在看的东西。**

---

## 2. 因果链：三个环节，坏在最后一环

### 2.1 环节一 · 模型忠实转写（无过错）

fresh100 的 8 条盲评 fail，逐条比对 instruction/reasoning 与当时注入的 hint 短语：
**8/8 方向词一致**。模型没有自由发挥，它老老实实把「moderately richer」写成了
"richer color"。判官看图后说"明显更淡"。

出处：`dataset_build/src/construct/responses.py` 第 409-411 行的实施注释
（"All eight of its blind fails transcribed these hints *faithfully* and the
reviewer disputed the picture anyway"）。

### 2.2 环节二 · 数值在自身定义下正确（无过错）

WP11 当时的判定是：从归档像素按 `mask.effective_alpha` 复算（`agent.py` 交给
`objective_edit_hints` 的正是这个权重），**每个存档数字都被复现到 0.1 以内**——
没有整帧稀释可修，没有权重接错可修。

本次（WP13）在两批共 22 条上独立重跑同一方法，结论不变：

- **fresh100 的 8 条**：WP11 诊断表引用的每个数字**逐一复现，无一例外**——
  `23dcd3b9` 的 clip 5.3%→8.6%、`5b7972ca` 的 Δa\*=−1.56、`be62412a` 的有色像素
  ΔC=−10.73、`e30c2ded` 的 ΔC=−5.17/ΔL=+6.05、`fc1cb9a2` 的 clip 0%→11.6% 与
  ΔC=−0.64，以及 `1e3cbddf` 的「α≥0.75 区占 80% 面积、ΔL=+3.06」（复算
  area_frac=0.8028、d_L=+3.0564）。
- **22 条整体**：88 个轴（22×4）里 **87 个符号与存档一致**；唯一例外是
  `72ffba32` 的 brightness（存档 −0.0038、复算 +0.044），两侧都是零、都落在 1.0
  死区内，prompt 上本就不发声。
- **幅度**：三个一阶轴（brightness/warmth/chroma）的偏差中位数 **0.032**，
  最大 0.61；最大的两条都是 style 行（`d0b53892` 0.61、`3e6b7528` 0.38），
  这类行权重为全 1、复算的 before 需把原始源图重采样到候选图尺寸，与生产路径的
  工作图分辨率不完全同路。contrast 是二阶矩、对 JPEG 平滑更敏感，偏差中位数
  0.093、最大 0.45。

**这些残差全部远小于失真本身的量级**（§3 里动辄十几个 C\* 单位的符号反转），
不影响任何一条结论。「hints 算错了」这条假设到此出局。

### 2.3 环节三 · 定义与观者不匹配（问题所在）

观者报告的是**显著物体上的颜色变化**；均值报告的是**mask 支撑上每个像素的平均
变化**。这两个量在真实照片上系统性地不相等，且不相等的方式有规律。

---

## 3. 三种失真机制（每条配具体样本数字）

> §3.1–3.3 是命题里说的三种颜色度量失真；§3.4 是与之同源的空间失配，§3.5 是两条
> 反面对照。

### 3.1 高光截断污染

编辑把一部分像素推到接近纯白。白化像素的 b\* 和 C\* 都趋近 0，于是它们把均值往
「更冷、更去饱和」的方向拽，而**存活下来的中间调其实更艳了**——判官看到的是后者。

| 样本 | 存档 hint | 截断 | 判官 |
|---|---|---|---|
| `23dcd3b9`（style） | 亮+10.66 / 冷-8.05 / 去饱-6.79 | clip **5.3% → 8.6%** | "AFTER 明显更暖、橙红更饱和" |
| `fc1cb9a2`（radial） | 去饱-4.51，措辞 "strongly" | clip **0% → 11.6%** | "红发和木头反而更浓" |
| `a9d7dd6d`（band, fresh200） | 艳+1.10（落 2.0 死区→不发声）/ 对比-1.47 | clip **14.4% → 12.9%**（基线本就高） | "强烈的青/橙饱和度提升、对比依然强烈" |

`fc1cb9a2` 是最干净的一例：截断从 0 涨到 11.6%，同时**仅在有色像素上 ΔC 只有
−0.64**——均值报的 −4.51 几乎全部由白化像素贡献。

`a9d7dd6d` 是另一种形态：截断没有变大反而略降，但**前后都有 13-14% 的权重压在
近乎无彩的白化像素上**，这批像素把均值长期钉在 0 附近——于是有色像素上 +6.93 的
增饱和被压成 +1.10，正好落进 2.0 死区，v4.1 反而让模型对它闭了嘴。

### 3.2 中性像素稀释（含符号反转）

mask 支撑里绝大多数像素接近中性色（灰墙、皮肤、天空、沙地）。发生在少数有色物体
上的强烈变化，被这些中性像素在均值里抹平，甚至**被反号**。

| 样本 | 均值 chroma | 仅有色像素（C₁≥20）ΔC | 判官 |
|---|---|---|---|
| `be62412a`（linear） | **−0.51**（<1.0，v4.0 只给了含糊的 "near-neutral"） | **−10.73** | "强烈去饱和，尤其口红" |
| `e30c2ded`（linear） | **+1.19** → v4.0 说 "moderately richer" | **−5.17**（ΔL+6.05 使 C/L 进一步下滑） | "明显更淡、更不饱和" |
| `72ffba32`（band, fresh200） | **−5.08** | **−21.48** | "强烈去饱到近单色" |
| `e424ffd1`（band, fresh200） | **+5.03** → "richer" | **−13.28** | "把考拉/树干/树叶强烈去饱到苍白橄榄近单色" |
| `d0b53892`（style, fresh200） | **+3.34** | **−17.91** | "明显变暗、肤色变冷、细节变差" |

`e30c2ded`、`e424ffd1`、`d0b53892`、`3e6b7528` 四条是**符号反转**：均值说"更艳"，
有色像素说"去饱"，判官站在有色像素这边。这不是幅度误差，是方向错误。

`e30c2ded` 还叠加了第二重：ΔL=+6.05 而 ΔC 仅 +1.19，人眼读的是 C/L 比值，
**提亮本身就在偷饱和度**——+6 的 L\* 配 +1 的 C\*，看上去是变淡而不是变艳。

### 3.3 warmth 只看 b\* 轴（a\* 上的偏移完全不可见）

`warmth` 定义为 Δb\*。CIELAB 的色相由 (a\*, b\*) 共同决定，**青绿↔品红这一整条轴
（a\*）不在 warmth 的视野里**。于是一次明显的偏绿或偏红，hints 一个字都不会说。

| 样本 | Δb\*（= warmth） | Δa\* | 判官 |
|---|---|---|---|
| `5b7972ca`（linear） | **+1.03** → "warmer" | **−1.56**（青绿向） | "偏向青绿而非更暖" |
| `7213ca74`（style, fresh200） | **−4.18** → "cooler" | **+5.87**（红向） | "整体明显更暖更红" |
| `721780ad`（radial, fresh200） | **+4.17** → "warmer" | **−9.05**（绿向） | "椭圆编辑区更暗、明显偏绿偏冷" |

`721780ad` 的 Δa\* 幅度是 Δb\* 的两倍以上，而 hints 报的方向恰好相反。

### 3.4 第四种（相邻，非颜色度量本身）：空间注意力错配

严格说这不是「颜色轴选错」，但它与均值定义同源——**梯度类 slot（linear/band/
radial）里，均值把整条梯度摊平，而判官只看被 instruction 点名的主体**。

- `9d53ccd3`（linear）：hint 冷-4.61/去饱-2.09/对比-1.77（整条梯度的平均），
  判官："狗和叶子更艳更硬"。
- `08d6877b`（radial, fresh200）：hint 暖+6.71/艳+7.14，实际只有废墟周围的天空
  在变，废墟/崖/海几乎没动。
- `f0bbdd20`（radial, fresh200）：hint 亮-2.96 "更暗"，判官："编辑的中心区反而更亮"。

### 3.5 反面对照（同样重要）

两条对照证明这份诊断不是"凡 fail 皆归咎于 hints"：

- **`1e3cbddf`（linear）**：hints 无错。复算 α≥0.75 的区域占 80% 面积、
  ΔL=+3.06，与存档的 +2.98 一致，方向是"更亮"。判官却说"下半部可见更暗更密"。
  这一条是**判官感知误差**，记在册上不修改。
- **`d5f8537e`（semantic）**：四个 hint 全部正确。fail 来自 SAM3 的整群 mask
  缺陷（声称"限于该人"、实际编辑波及多人），与颜色度量无关。

---

## 4. 两轮 out-of-sample：prompt 措辞层修复无效

### 4.1 两批的定位

| 批次 | build_id | n(SFT 行) | 标注契约 | 定位 |
|---|---|---|---|---|
| eval100 | `eval100-annotqa-20260727` | 151 | v4a | **调优批**（prompt v2/v3/v4 三轮都在它上面调过，含过拟合），已退役、永不进训练 |
| fresh100 | `fresh100-v4a-20260729` | 75 | v4a（v4.0 措辞） | 第一轮 out-of-sample |
| fresh200 | `fresh200-v4a1-20260729` | 160 | v4a1（v4.1 措辞：死区 1.0/2.0 + veto 反向锚定） | 第二轮 out-of-sample，**修因之后** |

v4.1 相对 v4.0 的全部改动就在颜色 hints 的措辞层：小于死区不再给"near-neutral or
visually mixed"（旧文本被模型读成了发挥许可），改为明确的"do not state a direction
either way"；超过死区则加反向锚定。度量本身一个字没改。

### 4.2 预注册门槛：两轮都 NO-GO，且几乎逐点重合

| 门槛 | fresh100（修因前，n=75） | fresh200（v4.1 后，n=160） |
|---|---|---|
| fail ≤ 10% | **10.67%** ❌（8 条） | **10.62%** ❌（17 条，超 1 条） |
| pass ≥ 50% | **49.33%** ❌（37 条） | **52.50%** ✅（84 条） |
| band slot fail ≤ 15% | 0%（0/12）✅ | **15.62%** ❌（5/32，超 1 条） |
| 泄漏 = 0 | ✅（1 个原始命中，裁定 benign） | ✅（6 个原始命中，逐条裁定全部 benign） |
| 七段契约 100% | ✅（75/75） | ✅（160/160） |
| 弃权率校准 | ✅（abstain 44.87%、low 24.36%） | ✅（abstain 44.87%、low 20.51%） |

两轮的 Wilson 95% CI 都横跨阈值（fresh200 fail [6.74%, 16.36%]、pass
[44.79%, 60.09%]），即统计上与门槛不可区分；按预注册纪律仍判 NO-GO。

### 4.3 组成标准化后的对照（关键证据）

fresh100 与 fresh200 的 slot 组成不同（linear 占比 30.7% vs 23.1% 等），必须先
标准化再比：

| 口径 | pass | borderline | fail |
|---|---|---|---|
| fresh100 实测（v4.0） | 49.33% | 40.00% | 10.67% |
| fresh200 按 fresh100 组成加权（v4.1） | **49.79%** | 40.09% | **10.12%** |
| fresh200 实测（v4.1） | 52.50% | 36.88% | 10.62% |
| fresh200 按 eval100 组成加权 | 53.77% | 35.73% | 10.50% |
| fresh100 按 eval100 组成加权 | 49.55% | 41.27% | 9.19% |
| eval100 实测（调优批，含过拟合） | 60.26% | 34.44% | 5.30% |

**pass 49.33% vs 49.79%、fail 10.67% vs 10.12%——v4.1 的 out-of-sample 增益为零。**
fresh200 的实测 pass 高出 3 个点，全部由 slot 组成差异解释（radial/style 占比更高，
两者本就是高分槽）。

维度均分同样重合（fresh100 → fresh200）：consistency 3.733 → 3.875、
reasoning_alignment 3.440 → 3.606、usability 3.573 → 3.694、region_scope
4.467 → 4.419——移动量与组成差同量级。

### 4.4 fail 聚类：颜色方向占绝对多数

fresh200 的 17 条 fail（`wp12_addendum.json`）：

- `colour_or_tonal_direction_wrong_or_overstated`：**14 条，占 82.4%**
- `region_scope_declaration_mismatch`：5 条，占 29.4%（3 条 semantic 整群 mask
  过度声称单实例、2 条 linear 轴向/方向声明与 region map 不符）
- 两簇重叠 2 条（`3d50b075`、`5acca9d3`）

fresh100 的 8 条 fail 的主导模式记为"颜色方向反转（声称加暖/加饱和，实际变冷/
去饱和，或反之）"；region_scope ≤2 分仅 1 条——**v4 的区域契约在新 source 上是成立
的，出问题的是颜色轴**。

### 4.5 静默行 vs 断言行：两批一致的 −0.24 / −0.28

用 v4.1 的死区（colour 2.0）把每批的 SFT 行分成两组——**颜色轴被静默的行**
（warmth 或 chroma 的 |Δ| < 2.0）与**颜色轴被断言的行**——比较盲评 consistency：

| 批次 | 断言组 n / 均分 | 静默组 n / 均分 | 差 | 静默行占比 |
|---|---|---|---|---|
| fresh100（v4.0，**反事实分组**） | 42 / **3.8571** | 33 / **3.5758** | **−0.2813** | 44.0% |
| fresh200（v4.1，**实际静默**） | 92 / **3.9783** | 68 / **3.7353** | **−0.2430** | 42.5% |

pass 率同向：fresh100 断言组 57.1% vs 静默组 39.4%；fresh200 断言组 59.8% vs
静默组 42.6%。

**这张表是"措辞不是病灶"的最强证据。** fresh100 跑的是 v4.0，它根本没有 2.0 死区
（v4.0 只在 |Δ|<1.0 时给一句含糊的 "near-neutral or visually mixed"），所以那 33
行**当时是被断言了方向的**；分组只是反事实标注"v4.1 会静默哪些行"。fresh200 则是
真的静默了。两种处理方式截然相反，**落差却几乎一样大（−0.28 vs −0.24）**——说明
这个落差是**样本本身的属性**（小 |Δ| 的行就是均值最不可信的行），而不是 prompt
说了什么或没说什么造成的。让模型闭嘴并不能把这些行救回来。

数据出处：`/var/cache/veradata/annot_review/fresh200-v4a1-20260729/wp12_colouraxis.json`
（fresh200 原始产物）与 `/tmp/wp13-fresh100-colouraxis.json`（本次用同一脚本
`fresh200-v4a1-20260729/tools/wp12-colouraxis.py` 在 fresh100 上重跑）。

---

## 5. WP11 诊断表（fresh100，8 条，权威原文）

| sft_id | slot | hint 方向（存档） | masked 重算 | 盲评判词 | 根因 |
|---|---|---|---|---|---|
| 1e3cbddf | linear | 亮+2.98/暖+0.43/艳+0.41/对比-1.54 | 亮+2.94（吻合） | "下半部可见更暗更密" | 判官感知误差（α≥0.75 区 80% 面积 ΔL=+3.06，此条 hints 无错） |
| 23dcd3b9 | style | 亮+10.66/冷-8.05/去饱-6.79 | 吻合 | "AFTER 明显更暖、橙红更饱和" | 高光截断：clip 5.3%→8.6%，白化像素拉低均值，存活中间调更艳 |
| 5b7972ca | linear | 暖+1.02/艳+1.05 | 吻合 | "偏向青绿而非更暖" | warmth 只看 b\*：Δb\*=+1.03 但 Δa\*=−1.56 |
| 9d53ccd3 | linear | 冷-4.61/去饱-2.09/对比-1.77 | 吻合 | "狗和叶子更艳更硬" | 空间注意力错配：判官读被点名主体，hint 平均整条梯度 |
| be62412a | linear | 艳-0.51→死区不发声 | -0.46 | "强烈去饱和，尤其口红" | 中性像素稀释：有色像素上 ΔC=−10.73 被均值抹平 |
| d5f8537e | semantic | 全部正确 | 正确 | "编辑波及多人，与'限于该人'矛盾" | 非颜色问题（SAM3 mask 整群缺陷，对照项） |
| e30c2ded | linear | 艳+1.19→"moderately richer" | +1.21 | "明显更淡、更不饱和" | chroma 符号反转：有色像素上 ΔC=−5.17；ΔL+6.05 使 C/L 下滑 |
| fc1cb9a2 | radial | 去饱-4.51"strongly" | -4.52 | "红发和木头反而更浓" | 截断+稀释：clip 0%→11.6%；有色像素 ΔC 仅−0.64 |

sft_id 为前 8 位缩写。完整 ID 见
`/var/cache/veradata/annot_review/fresh100-v4a-20260729/blind_wp11/sample_index.jsonl`
与同批 `run/sft.jsonl`；本文 §9 的证据包内每个 `hints.json` 也带完整 ID。

**关于这 8 条的重标结果**：v4.1 措辞下重标 + 重盲后，6 条升到 borderline、2 条仍
fail。**这不能当作 v4.1 有效的证据**——这 8 条正是 v4.1 死区参数（1.0/2.0）的调参
面板，属 in-sample。out-of-sample 判定见 §4。

---

## 6. fresh200 方向类 fail 全集（14 条）

来自 `/var/cache/veradata/annot_review/fresh200-v4a1-20260729/wp12_addendum.json`
的 `colour_or_tonal_direction_wrong_or_overstated` 聚类；判词摘自同批
`blind/b1_fresh200.jsonl`。"机制"列是本次（WP13）依据复算探针 + 判词做的**初判**，
不是既有产物中的既有结论，做 ROC 筛选时当线索用、不要当已证事实引用。

| sft_id | slot | 一致性分 | 盲评判词摘句 | 机制初判 |
|---|---|---|---|---|
| 7213ca74 | style | 2 | "its overall balance is dramatically warmer and redder rather than cooler, making a central directional claim incorrect" | warmth 只看 b\*（Δb\*=−4.18，Δa\*=**+5.87**） |
| 721780ad | radial | 1 | "the annotation reverses the dominant visible change: the edited oval becomes darker and markedly greener/cooler, rather than brighter, warmer, and richer" | warmth 只看 b\*（Δa\*=**−9.05**）+ chroma 反转 + 空间稀释 |
| e424ffd1 | band | 1 | "strongly desaturates the koala, trunk, and foliage into a pale olive-toned near-monochrome rather than deepening greens" | chroma 符号反转（均值 +5.03，有色 **−13.28**） |
| d0b53892 | style | 1 | "substantially darkens rather than brightens the portrait, reduces skin warmth… directly reversing the main claimed adjustment" | chroma 符号反转（均值 +3.34，有色 **−17.91**）+ 空间稀释 |
| 3e6b7528 | style | 2 | "the AFTER is generally darker with deeper shadows rather than substantially brighter as instructed" | chroma 符号反转（均值 +1.22，有色 −9.84）+ 空间稀释 |
| 72ffba32 | band | 1 | "strongly desaturated and darker rather than warmer… loses substantial facial and dark-clothing detail" | chroma 稀释（均值 −5.08，有色 **−21.48**）+ warmth 只看 b\* |
| 5dc02e09 | band | 1 | "deeper blacks alongside brighter branch and metallic highlights, stronger contrast, and visibly warmer or richer color, contradicting the requested darkening with gentler contrast" | chroma 符号反转（有色 +3.31）+ contrast 方向反转 |
| a9d7dd6d | band | 2 | "applies a strong cyan-and-orange color boost and retains pronounced contrast rather than the moderate, natural, less-contrasty treatment described" | chroma 稀释被 2.0 死区静默（均值 +1.10，有色 +6.93）+ 高光截断（clip 14.4%→12.9%）+ contrast 反转 |
| 5b4a26dd | band | 2 | "the main claim that the tree and nearby sky were brightened is not supported and the edited area appears darker" | 空间稀释 + chroma 稀释（有色 −12.48） |
| 08d6877b | radial | 2 | "the ruin, cliff, and sea remain essentially black or unchanged, so the claimed structural brightening and broader warm enhancement are overstated" | 空间稀释（均值把局部变化摊到整个 mask） |
| f0bbdd20 | radial | 1 | "the edited central area appears brighter rather than moderately darker, reversing the main requested tonal change" | 空间稀释（梯度均值 vs 判官读中心） |
| 9f3ea60d | linear | 2 | "the AFTER mainly increases warmth and contrast while deepening many shadows rather than substantially brightening" | contrast 方向反转（均值 −0.60）+ 空间稀释 |
| 3d50b075 | linear | 2 | "generally deeper shadows rather than the requested broad brightening, and the declared spatial extent does not match the right-weighted mask" | 空间稀释；**与 region_scope 簇重叠** |
| 5acca9d3 | linear | 3 | "the cooling is stronger than the claimed restrained shift and the annotation omits the conspicuous tighter crop" | 幅度失准；**与 region_scope 簇重叠** |

按机制归并（一条可归多类，14 条共 24 个归属）：chroma 稀释/符号反转 **8** 条、
空间稀释 **8** 条、warmth 只看 b\* **3** 条、contrast 方向反转 **3** 条、
高光截断 1 条、幅度失准 1 条。其中 **4 条是符号反转**（`e424ffd1`、`d0b53892`、
`3e6b7528`、`5dc02e09`）——均值与有色像素给出相反的方向，判官站在有色像素这边。

---

## 7. 已否决方案：α·max(C_before, C_after) 重加权

**不要重试。** 曾试图不改度量定义、只改权重：把 alpha 权重乘上
`max(C_before, C_after)`，让有色像素在均值里占更大分量。在 fresh100 的 11 样本
面板上实测：

- **修好 2 条**（原本被中性像素稀释的）
- **软化 2 条**（方向仍偏，但幅度接近了）
- **反转 1 条**——把一条判官已经打了 5/5 的样本改坏了

没有任何单一重加权能同时调和这两端。这就是为什么 v4.1 选择改「告诉模型怎么用这个
数」而不是改这个数本身（结果见 §4：也没用）。

否决证据的归档位置：`dataset_build/src/construct/responses.py` **第 429-432 行**
的实施注释（"Reweighting was tried and rejected on evidence: alpha \*
max(C_before, C_after) repairs two of those fails, softens two, and inverts one
the reviewer had already scored 5/5."）。同段第 406-437 行完整记录了 v4.1 的动因
与死区取值理由。

---

## 8. 建议路径：在 fresh200 面板上做 ROC 判别力筛选

**不要再在措辞上试。** 下一步应当是：给现有的 fresh200 n=160 面板打上标签
（14 条方向类 fail = 正类，方向类 pass = 负类），对一组候选度量逐个算 ROC/AUC，
选出真正能预测判官观感的统计量，再拿它替换或补充现行的四个均值。

候选度量（括号内是本次复算里已经实现的字段名，见 §9）：

1. **仅有色像素的 chroma 统计** —— 只在 `C_before ≥ 20` 的像素上做 alpha 加权
   ΔC（`probe.d_chroma_on_vivid`）。针对 §3.2：fresh200 的 14 条里对 8 条有解释力
   （其中 4 条是符号反转），fresh100 另有 `be62412a` / `e30c2ded` / `fc1cb9a2` 三条。
2. **截断感知聚合** —— 编辑前后的高光截断比例（`probe.clip_frac_before/after`，
   阈值 L\*>96），或在排除截断像素后重算均值。针对 §3.1。
3. **a\*/b\* 联合色相方向** —— 至少把 Δa\*（`probe.d_a_star`）与 Δb\* 一起报，
   或直接报色相角变化 Δh 与色度矢量 (Δa\*, Δb\*) 的模长/方向。针对 §3.3。
4. **被点名主体区域统计** —— 在高 alpha 区（`core_half_max_alpha`，α ≥ 半个峰值）
   或 subject mask 内单独统计，而不是整条梯度取平均（`alpha_bands` 提供按
   alpha 四分档的 ΔL/Δb\*/ΔC）。针对 §3.4。
5. **感知饱和度 C/L** —— `probe.d_saturation_x100`（`C/max(L,1)` 的前后差）。
   针对 §3.2 里"提亮偷饱和度"那一重。

**工程成本要先说清楚**：这些量若要进生产 hints，得由 render 路径在算 hints 时一并
落盘。已 build 的批次没有这些字段，**只对新 build 生效**——所以 ROC 筛选必须在动
生产代码之前做完，不能边改边试。

筛选的执行成本很低：`/tmp/wp13-recompute.py` 已经能对任意 sft_id 列表算出上述全部
字段，把输入从 14 条换成 fresh200 的全部 160 条即可（14 条实测 10.5 秒，全量约
2 分钟量级）。注意复算读的是归档 JPEG，一阶轴残差中位 0.03、二阶（contrast）中位
0.09（§2.2）——做 ROC 排序足够，但若某个候选度量的判别边界落在这个量级内，需要
改由 render 路径在 pre-JPEG 上直接落盘再判。

**并列的另一条岔路**（用户尚未拍板，记录在此以免遗忘）：接受现状，把 fail ~10%
当作 50k 的已知损耗，训练侧按 usability 过滤。出厂预期应以 fresh 批为准：
**pass ~50%、fail ~10.6%**；eval100 的 60.3% 是调优批含过拟合，不可作为预期。

---

## 9. 证据样本包

**位置**：`/var/cache/veradata/annot_review/hints_mean_bias_2026-07-29/`
**仓库入口**：`/home/bc/VeraRetouch/_hints_bias`（符号链接，已加入 `.gitignore`；
沿用 `_annot_worst` / `_annot_ab` 的先例）

**内容**：22 个样本 = fresh100 的 WP11 诊断 8 条 + fresh200 的方向类 fail 14 条。
无缺失资产（22/22 的 before/after 齐全；18 条 local 组的 `cgt.png` 齐全，4 条
style 组按契约本就没有 C_GT）。已核对两个 build 的 source 交集为 0，22 条互不重复。

**取数快照（重要）**：fresh100 侧一律取 **WP11 重标之前**的状态——hints 来自
`groups.jsonl.pre-wp11`，标注文本来自审阅根的 `run/sft.jsonl`（重标前快照），
盲评来自 `blind/b1_fresh100.jsonl`。这样档案里的三样东西（hint / 标注 / 判词）
属于同一时刻，与 WP11 诊断表对得上。v4.1 重标后的重盲另存在 `verdict.json` 的
`wp11_reblind_after_v41_relabel` 字段里，标注了 in-sample 警告。

```
_hints_bias/
├── INDEX.md                        # 样本清单表 + 阅读指引 + 复算脚本路径（先读这个）
├── manifest.json                   # 机器可读清单：资产来源路径、盲评分、重盲结果
├── fresh100_linear_1e3cbddf/       # 命名：<批次>_<slot>_<sft_id 前 8 位>
│   ├── before.jpg
│   ├── after.jpg
│   ├── cgt.png                     # local 组的 C_GT（灰度即 alpha）；style 组无
│   ├── hints.json                  # 存档 hint + WP11 表原文 + WP13 复算与探针
│   ├── annotation.json             # instruction / instruction_short / reasoning 七段原文
│   └── verdict.json                # 盲评五维分与判词原文（+ fresh100 的重盲结果）
├── fresh100_linear_5b7972ca/       …（同构）
├── fresh100_linear_9d53ccd3/
├── fresh100_linear_be62412a/
├── fresh100_linear_e30c2ded/
├── fresh100_radial_fc1cb9a2/
├── fresh100_semantic_d5f8537e/
├── fresh100_style_23dcd3b9/
├── fresh200_band_5b4a26dd/
├── fresh200_band_5dc02e09/
├── fresh200_band_72ffba32/
├── fresh200_band_a9d7dd6d/
├── fresh200_band_e424ffd1/
├── fresh200_linear_3d50b075/
├── fresh200_linear_5acca9d3/
├── fresh200_linear_9f3ea60d/
├── fresh200_radial_08d6877b/
├── fresh200_radial_721780ad/
├── fresh200_radial_f0bbdd20/
├── fresh200_style_3e6b7528/
├── fresh200_style_7213ca74/
└── fresh200_style_d0b53892/
```

**浏览方式**：先只看 `before.jpg` / `after.jpg`（有 `cgt.png` 就叠着看编辑落在
哪儿），自己判断颜色往哪走；再读 `hints.json` 的 `archived_objective_hints`；
最后读 `verdict.json` 的 `reasons.consistency`。三者对不上的地方就是本文的全部
内容。`INDEX.md` 里有 22 条的一句话对照表（"hint 说 X / 判官看到 Y"）。

**数字来源分层**（`hints.json` 里每类都标了 source，勿混）：

| 字段 | 来源 | 权威性 |
|---|---|---|
| `archived_objective_hints` | build 的 groups journal，`candidates[<id>].objective_hints` 原值 | 存档事实 |
| `wp11_diagnostic_table_row` | §5 的 WP11 诊断表逐条抄录（仅 fresh100 8 条） | 用户确认的权威结论 |
| `wp13_recompute` | 本次用 WP11 方法在归档像素上重跑（`/tmp/wp13-recompute.py`） | 可复现的复算，已逐一复现 WP11 表的每个数字 |
| `wp13_mechanism_read` | 本次依据探针 + 判词的机制初判（仅 fresh200 14 条） | **初判，非既有结论** |

**复算脚本**（放 `/tmp`，不入库）：

| 路径 | 用途 |
|---|---|
| `/tmp/wp13-recompute.py` | 本次复算与候选度量探针（度量定义抄自 `/tmp/wp11-diag2.py`） |
| `/tmp/wp13-build-pack.py` | 组装证据包（只读既有产物，只写新目录） |
| `/tmp/wp13-fresh{100,200}-recompute.jsonl` | 复算原始输出（已内联进各 `hints.json`） |
| `/tmp/wp13-fresh100-colouraxis.json` | §4.5 fresh100 侧的静默/断言分组统计 |
| `/tmp/wp13-fresh{100,200}-ids.txt` | 两批的目标 sft_id 列表（复算与组包的输入） |
| `/tmp/wp13-fresh100-build/` | 符号链接目录：`groups.jsonl`→`groups.jsonl.pre-wp11`、`sft.jsonl`→审阅根 `run/sft.jsonl`，供上面两个脚本按统一的 BUILD 布局读 fresh100 的重标前快照 |
| `/tmp/wp11-diag.py`、`/tmp/wp11-diag2.py` | WP11 原始诊断脚本（本次未改动，仅作度量定义来源） |

调用方式（两批各一次）：

```bash
$PY /tmp/wp13-recompute.py <BUILD 目录> <审阅根> <blind 名> <ids 文件>
# fresh100: /tmp/wp13-fresh100-build  /var/cache/veradata/annot_review/fresh100-v4a-20260729  b1_fresh100  /tmp/wp13-fresh100-ids.txt
# fresh200: /mnt/nfs/bc/data/builds/fresh200-v4a1-20260729  /var/cache/veradata/annot_review/fresh200-v4a1-20260729  b1_fresh200  /tmp/wp13-fresh200-ids.txt
$PY /tmp/wp13-build-pack.py          # 读上面两份 jsonl，组装整个证据包
```

---

## 10. 相关既有产物索引

| 路径 | 内容 |
|---|---|
| `/var/cache/veradata/annot_review/fresh100-v4a-20260729/` | 第一轮 out-of-sample 审阅根：`d10_report.json`（门槛/失败聚类）、`blind/`（75 行盲评原始答复 + sample_index）、`blind_wp11/`（8 条重标后的重盲）、`run/`（sft.jsonl 快照、failures、manifest、GPU 采样）、`mech/`、`tools/wp9-*.py` |
| `/var/cache/veradata/annot_review/fresh200-v4a1-20260729/` | 第二轮审阅根：`wp12_report.json`（六门槛）、`wp12_comparison.json`（三批组成标准化对照）、`wp12_addendum.json`（fail 聚类 + 泄漏逐条裁定 + relay 预算）、`wp12_colouraxis.json`（静默/断言分组）、`blind/`、`run/`（含 groups.jsonl）、`tools/wp12-*.py` |
| `/mnt/nfs/bc/data/builds/fresh{100-v4a,200-v4a1}-20260729/` | 两批的 groups/sft/failures journal 与 manifest（fresh100 另有 `*.pre-wp11` 备份） |
| `dataset_build/src/construct/visibility.py` L45-121 | `objective_edit_hints{,_from_lab}`：本文所论统计量的定义 |
| `dataset_build/src/construct/responses.py` L406-511 | v4.1 hints 措辞、死区常量与取值理由、**L429-432 的重加权否决记录** |
| `docs/DATABUILD_EVAL_AND_PERF_HANDOFF_2026-07-28.md` §6/§7 | 本文的上游：v4 契约实验终态、两轮出厂验证与 50k 决策点 |

---

## 附录 A：WP14 感知标定实验（2026-07-29，本档案的因果闭环）

三臂心理物理实验（579 次调用，产物 `/var/cache/veradata/annot_review/jnd_calib_20260729/`）：
8 张真实图 × 5 轴（L/b*/a*/C/对比）× 4 档幅度 × 全图/band 局部，纯 numpy 单轴扰动，
曲线用实现幅度拟合。

**结论（把本档案的相关性证据升级为受控因果证明）：**

1. **luna（标注模型）颜色 JND = 3.4-7.4 Lab 单位**（人类的 3-7 倍；a* 轴任何幅度
   都不可靠）。fail 样本的实际幅度 0.5-3 落在其感知盲区——luna 在方向断言上
   只能转写 hints。
2. **锚定效应是灾难级**：同一像素上错误 hint 把 luna 55.8%→5.8%（60 对翻错、
   0 错翻对、弃权归零）；luna 裸眼 94.4% 的可感知区被压到 27.8%，且模型会编造
   相符的视觉证据。**hints 完全支配标注输出；在换成验证过的度量之前，注入现行
   方向 hints 的期望收益为负。**
3. **sol（评委）JND ≈1.8（颜色轴 1.4-1.8，brightness 6.2 偏弱）**；22 条证据
   样本里评委噪声只解释 1 条——盲评门槛有效。
4. 11 条「四轴均值全在 luna 阈下」的行注意：那是**均值意义上的阈下**（口红案
   均值 −0.46 但口红上 −10.73 评委可见）——阈下豁免必须定义在新分桶度量上。

**架构定案（用户确认）**：「度量即标注」——颜色方向事实由度量生成（色名分桶+
截断感知+a*b* 联合+逐桶沉默判据），模型职责收缩为语言润色与区域/语义指认；
方向正确性由渲染重放符号硬门 + 反转对 2AFC 评委（仅可判别区）复核，recaption
定点改写替代自由重标。preset 重分类解耦后置（TS-WCL 对比嵌入赛道）。

## 附录 B：三路文献调研要点（2026-07-29，报告全文见会话产物）

- **A 图像差异描述**：主线把光照/颜色当 distractor（不适用）；RetouchLLM 的
  生成-验证闭环（候选编译成可执行编辑→回渲染→选最优）是最接近的系统；
  Direction List 判官协议（ρ=0.94）；Blind-Faith-in-Text 解释忠实转写。
- **B 编辑指令管线**：CLIP 方向相似度与人评负相关（已死）；recaption 而非
  丢弃/重标（+0.16 消融）；内生自纠正无效（2310.01798）；判官 Avg@4 >
  单次强模型；我们验证:产出 1:1 在业界属保守下限。
- **C 色彩变换表征**：content-free LUT 探针被否（占用率加权有 2.44→1.09
  定量先例）；聚类用 preset-身份对比嵌入（TS-WCL，R@1 93.19，2 万条人类
  排序验证）+ 生成期 DPP/最小间距约束；45% 弃权率是 VLM 分辨力天花板
  （LMM-JND），解法在候选生成侧。
