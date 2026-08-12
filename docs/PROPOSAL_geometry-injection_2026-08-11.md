# 统一实验提案：reasoning 几何码提取与注入（geometry-injection）

日期：2026-08-11 ｜ 状态：待主 agent 批准；批准即冻结 §2.1 接口与 §4 判据
来源：四份方案书（A1-parsing / B1-attn-readout / C-query-bridge / INJ-module；SAM-component 缺件）
＋三份评审（代码保真 / 设定相容 / 可实现）。本提案**取代**各方案书中与之冲突的段落。

---

## 0. 裁定摘要（评审逼出的套件级裁定，逐条编号，实现审阅按此执行）

| 编号 | 裁定 | 依据 |
|---|---|---|
| D-1 | **PCH（INJ-module）为唯一注入模块**。A1 §3.4、B1 注入接口、C 的 C-Inject 三套私设注入器**全部作废**；三臂降级为「只产 GeoCode」。PCH 架构与超参三臂共享，**权重逐臂独立训练**（码分布不同） | 评审「可实现」套件级发现 1 |
| D-2 | **GT 码上界档只有一个 = INJ 的 M1**（A1-GT / B1 A-gt / C 的 A1 臂全部并入）。Gate-0 数字统一：证伪线 **<+0.008**、灰区 **[0.008, 0.015)** 允许一次 tap 结构迭代（判据不改）、晋级 **≥+0.015** | 套件级发现 2 |
| D-3 | 三臂晋级线统一为**相对量** `Δ_arm ≥ max(+0.008, 0.5·Δ_GT)`（Δ_GT = M1−M0 实测值），消除 C 方案的预注册死区 | 评审「可实现」C 条 fatal |
| D-4 | **conf 生产规格三臂全覆盖**（§2.3）：A1 = 掩码后重归一化 softmax 的 span-min；B1/C = 温度标定后槽位概率折算；GT = 1；valid=0 → 全 0 | 套件级发现 3 |
| D-5 | **Dice 红线书面裁定**：B1 阶段 A 粗场辅助支的 Dice **豁免**——该支非主输出场、推理期丢弃、不进任何判据表；「IoU 禁当优化目标」红线的辖域 = 主输出场损失与一切进判据的量。实现审阅按本条放行，不再各自解释 | 套件级发现 4 |
| D-6 | **SAM-component 编号关闭**：并入 PCH 作结构锚（附录 A 记备选形态），不再要求独立方案书 | 套件级发现 5 |
| D-7 | **A1 默认改两遍式（two-pass）**：pass-1 完全自由生成 reasoning；pass-2 复用 KV、forced prefix `<geom>` + JSON schema 约束解码——`<geom>` 尾段引出的存在性 **by construction** 成立。原单遍 structural-tag 版降为消融臂 A1-onepass | 评审「可实现」A1 条 fatal |
| D-8 | **B1 的 shape 头改 BCE 多热**（softmax+CE 与「5 形状 multi-hot」设定冲突）；连续量头三臂统一 **sigmoid → [0,1]**（有界参数化纪律；C 的 tanh 域错位一并修复） | 设定相容 B1/C 条 |
| D-9 | **连续量统一 q=3**（cx, cy, cover ∈ [0,1]）；A1 schema 中 cx/cy/cover 从 required 改 **optional**，缺失 → conf_cont=0 走 null 路径（防止格式压力下编造数值）。若构造侧核实 reasoning 文本确含数值再升 required（见 §0.1 待核实清单） | 设定相容 A1 条 |
| D-10 | **捕获率同度量化**：一切「打过 82%」的主张改为与 **A1 实测码级捕获率 F1_A1**（同 GT 码、同 macro-F1 度量）对照，废除词级准确率与 macro-F1 的异质换算；0.82/0.87/0.85 降为参考值 | 评审「可实现」B1 弱点 3 |
| D-11 | **PCH-Lite 补 tap B-lite**（零初始化 1×1 conv 128→1024，≈0.13M）：G6 若裁定共享 Lite，rank-1 tap A 饱和的逃生口不随降档消失 | 设定相容 INJ 弱点 |
| D-12 | **B1 的「绕过 82%」为条件性论断**写入正文：只对 teacher-forced 档无条件成立；自生成档的 attention 行以已采样词为条件，18% 错词进入证据流。Gate-3 专测，失败预案 = 与 A1 合流（错词位替换） | 设定相容 B1 条 fatal |
| D-13 | 每张消融表**强制 Δ_shuffle 与 Δ_const 双控制列**（红线）；INJ 资格赛补常量码臂 M2b；A1 补 A1-const 臂 | 红线 + 评审多处 |
| D-14 | B1 的 C-slot 负控制与自生成档各需**第二遍全量 attention 导出**（缓存只存 per-slot merge 后张量），成本各 +3–4 GPU·h，入 §5 排期账 | 评审「可实现」B1 弱点 1 |

### 0.1 实施前待核实清单（NOTES.md 必答，属「待主 agent 决策」的按协议上报）

1. 构造侧 GT 码 schema 的连续量维度与语义（本提案按 q=3：cx/cy/cover）；
2. reasoning 文本是否按设计含连续量数值（决定 D-9 是否升 required）；
3. dense 头「倒数第二层」tap 点的实际通道数（≠32 则按 §2.2 加 1×1 适配）；
4. 本项目 VLM 栈的 eager attention 回归测试参考基准（B1 风险 1；F-LMM 的 transformers 4.39.1 钉版不直接迁移）。

---

## 1. 目标与 go/no-go 门

### 1.1 要验证的结论

如果实验成功，我们就能说：**冻结 VLM reasoning 段中的 ~20 维几何码（5 形状 + 9 方向 + 6 范围 multi-hot + 3 连续量）可被提取并注入冻结 dense 头，把 soft-IoU 推过 (image, instruction) 的信息论天花板（重放上界 0.76，现模型 0.7622）**；失败则不能说 reasoning 携带可注入的超天花板几何信息。

### 1.2 为什么需要验证

天花板已被打满，reasoning 文本是设定中**唯一**的额外信息源（词级几何描述准确率 82%，对应 hidden 是采样前的、可能信息更高）。已证伪路径：pooled hidden + FiLM（完美文本仅 +0.008）；broadcast 通道注入为下界形态在测。

### 1.3 Go/No-Go 门（全序列最便宜的死刑排最前）

```
Gate-0（GT 码上界，INJ 资格赛 M1 vs M0，先于一切提取臂）
  Δ_GT = Δ(M1 − M0) soft-IoU，配对 bootstrap 10^4 + Wilcoxon，S-val n≥1000
  ├─ Δ_GT < +0.008（不超过已证伪的 pooled-FiLM 完美文本水平）
  │     → 20 维码路线整体证伪，三臂全停，直接写结论报告
  ├─ +0.008 ≤ Δ_GT < +0.015（灰区）
  │     → 允许一次 tap 结构迭代（tap A/B 单开消融），判据数字不许改；
  │       重测仍 <+0.008 → 停；≥+0.008 → 以实测 Δ_GT 进入相对量晋级线继续
  └─ Δ_GT ≥ +0.015 → 全线开跑
配套硬门（Gate-0 同批）：
  G2 内容特异 Δ(M1 − M2 derangement) ≥ +0.010（p<0.01），M2≈M1 → 先验陷阱同构，证伪
  G4 回退健全 |Δ(M3 全空码 − M0)| ≤ 0.003，违反 = blocker（null 路径有副作用，全部 Δ 无效）
各提取臂晋级（D-3 统一相对量）：
  Δ_arm ≥ max(+0.008, 0.5·Δ_GT)，配对 p<0.05
  且 Δ(arm − arm-shuffle) ≥ +0.008、|Δ(arm-const − M0)| ≤ 0.005
  且 grid 边界 F1 ≥ −0.005（不降）、soft-IoU > 中心先验基线列（配对 Δ + p 必报）
```

---

## 2. 共享注入模块 PCH 最终规格（评审修复后）

### 2.1 GeoCode 接口契约 v1.0（批准即冻结；提取臂只产码，模块只认码）

```
GeoCode:
  c_disc : float32 (20,)  # [0:5]=shape multi-hot, [5:14]=direction, [14:20]=extent
                          # 取值域 [0,1]（提取臂给概率，GT 给 0/1）
  c_cont : float32 (3,)   # (cx, cy, cover) ∈ [0,1]；z-score 用整臂常量（禁逐图，s 缓存契约同源）
                          # meta.norm.domain 必填；消费侧启动时断言生数据住在声明域内
  conf   : float32 (4,)   # 组级 [shape, dir, extent, cont] ∈ [0,1]；生产规格见 §2.3
                          # 禁 NaN、禁缺省；解析失败/无 reasoning 段/多段冲突 → 全 0
  valid  : bool           # false ⇔ conf 全 0（冗余显式化，消费侧断言一致）
```

### 2.2 张量流（PCH-Full，d=256；结构锚均为已读代码，见 §7）

```
输入: F_dense (B, N, 1024)，N = H/16 × W/16；valid_grid (B,N) pad 掩码；GeoCode
(1) 码→原型 token（锚 SurgicalSAM Learnable_Prototypes；SAM not_a_point_embed/no_mask_embed）
    E_shape(5,256) E_dir(9,256) E_ext(6,256) 组别嵌入表
    t_g   = LayerNorm( Σ_i c_i·E_g[i] / max(Σ_i c_i, 1) )        g ∈ {shape,dir,ext}
    t_cont= LayerNorm( W_c · c_cont ),  W_c:(256,3)
    T_g   = conf_g · t_g + (1−conf_g) · null_g                    null_g: Embedding(4,256)
    tokens= [m; T_shape; T_dir; T_ext; T_cont] → (B,5,256)        m: field token, Embedding(1,256)
(2) K = LayerNorm(Conv1×1 1024→256 (F_dense)) → (B,N,256)；key_pe = PositionEmbeddingRandom(128)
(3) TwoWayTransformer(depth=2, dim=256, heads=8, mlp=2048, downsample_rate=2)
    → hs (B,5,256), keys (B,N,256)；两处 cross-attn 均带 valid_grid key_padding_mask（pad 红线）
(4) tap A（主路，rank-1 hypernetwork，锚 SAM mask_decoder）:
    w = MLP(256,256,32,3层)(hs[:,0])；U = dense 头倒数第二层特征（通道≠32 加 1×1 适配到 32）
    logit_cond = (w @ U.view(B,32,h'·w')).view(B,1,h',w')
    logits_final = logits_uncond + tanh(γ)·logit_cond，γ 标量 init=0（有界参数化 + 零初始化红线）
(5) tap B（dense 残差，锚 ControlNet make_zero_conv）:
    F'_dense = F_dense + zero_conv(W_out·keys)，W_out:256→1024，zero_conv 零初始化 1×1
禁忌: 无 broadcast 注入；模块内禁一切逐图 min-max/softmax 场归一化（READ 的该机制明确不移植）；
      进判据数字一律未归一化原始场。
```

**PCH-Lite**（D-11 修订）：d=128 / heads=4 / depth=1 / mlp=256 / cross 内部 64；hypernet MLP(128,128,32,2 层)；**tap B-lite = 零初始化 1×1 conv 128→1024 兼任投影与 zero_conv（≈0.13M）**。参数 ≈ **0.52M**（原 0.39M + tap B-lite）。Full ≈ **3.96M**。

退化双保险（原 INJ §3.4 不变）：结构保证（tanh(0)=0 + zero_conv → 初始 ≡ 0.7622，收益只能向上长）；分布保证（conf=0 → 精确退到可学习 null 常量；条件 dropout 成型 null 路径）。

### 2.3 conf 生产规格（D-4，三臂全覆盖）

| 生产者 | 离散三组 conf_g | cont 组 conf |
|---|---|---|
| GT（M 系列） | 1.0 | 1.0 |
| A1 | pass-2 掩码后重归一化 log-softmax：槽位 conf = value span 内逐 token min；**组 conf = 组内槽位 min**（A1 原文 §3.1/§3.4 的粒度矛盾按组级裁定） | 三个 value span 各取 min 后再取 min；字段缺失（optional）→ 0 |
| B1 / C | 读出头 sigmoid 概率经**温度标定**（S-val 上逐组拟合单标量 T，预注册）后 conf_g = min_i max(p_i, 1−p_i) | valid=1 → 1.0；valid=0 → 0 |
| 任何臂 | reasoning 段缺失 / 解析失败 / 多段冲突 → conf 全 0（禁丢样本、禁崩，锚 READ 的 seg_token_counts==1 硬过滤教训之反面） | 同左 |

### 2.4 训练配方（逐臂各训一份 PCH 实例）

| 项 | 取值 | 锚 |
|---|---|---|
| 可训参数 | 仅 PCH（VLM、dense 头骨干全冻结；dense 特征预缓存） | SurgicalSAM 冻结+缓存范式 |
| 损失 | dense 头现行场损失不变（BCE-with-logits 族）；**dice 不移植**（红线，D-5 辖域内） | 项目红线 |
| 优化器 | AdamW，lr=3e-4，wd=1e-4，cosine，warmup 300 步 | SurgicalSAM lr 1e-4–1e-3、wd 1e-4 取中 |
| batch/步数 | 64 / 10k 步 | 单卡数小时内 |
| 条件 dropout | p=0.15 全组置 null + 逐组独立 p=0.05（UNVERIFIED-CODE，CFG 式通行做法，两值列入 M4 附检扫描） | — |
| checkpoint | S-val soft-IoU（**禁 val loss**） | 红线 |
| 归一化 | c_cont z-score 整臂常量 + domain 断言；Δlogit 域断言随交付落盘（任何「注入无效」结论先出示断言通过记录） | s 缓存契约 |
| 提交纪律 | rm -f 日志 → `ps -p $PID` → tail 实质输出 → job.marker | D-20 |

### 2.5 资格赛 M 系列（先于三臂；D-13 补臂）

| 臂 | 定义 | 成本 |
|---|---|---|
| M0 | 冻结无条件基线（0.7622） | — |
| M1-full / M1-lite | GT 码注入两档 | 2–3h / <1h |
| **M1-prob**（新增） | 概率化 GT 码消融：标签平滑（0.9/0.1）+ conf ~ Beta(5,2) 采样，覆盖部署期概率码分布迁移 | 1–2h |
| M2 | M1 ckpt + eval 集码 derangement（保边缘分布） | eval-only |
| **M2b**（新增） | 全数据均值常量码 → Δ_const 列 | eval-only |
| M3 | 全空码 conf=0 | eval-only |
| M4 | conf ∈ {0,.25,.5,.75,1} 扫描（附 dropout 概率敏感性） | eval-only |
| M5 | 单组注入（前置检查：M3 过 G4 **且**训练日志确认每组 null dropout 覆盖 ≥5% 样本，否则 M5 结果标记不可解释） | eval-only |

判据：G1（Δ_GT ≥ +0.015 晋级）/ G2 / G3（边界 F1 ≥ −0.005）/ G4 / G5（hs[:,0] 线性探针回读 macro-F1 ≥ 0.90；翻方向位→质心顺从 ≥75%）/ G6（|full−lite| ≤ 0.005 → 三臂共享 Lite，含 tap B-lite）。证伪线见 Gate-0。

---

## 3. 三臂对照定义

### 3.0 公共约定

- 解码全臂统一 greedy（T=0），seed + git commit + 环境入 `config/`；
- 三臂只产 GeoCode（D-1），注入一律走 §2 PCH（档位按 G6 裁定）；
- 数据：S/P split 旁表，`eval100-annotqa-20260727` 永不进训练；探针类走 S-val 源；
- 离线产物三臂错峰共享：`codes.jsonl`（A1 产，A1 臂训练 + 捕获率对照用）、attention 缓存（B1）、hidden 缓存（C）。

### 3.1 A1 · 约束解码解析臂（可靠性基线 / 下界锚，82% 语义封顶）

**结构（D-7 两遍式，张量链）**

```
pass-1: frozen VLM.generate(greedy, max_new_tokens=192)，无 logits processor → reasoning 文本（82% 能力不受扰）
pass-2: 上下文 = prompt + reasoning + 固定指令「将上述几何描述复述为 JSON:」+ forced prefix "<geom>"
        复用 pass-1 KV cache；XGrammar compile_json_schema(SCHEMA) 从 content 起点接管；≤96 tok
SCHEMA: shape/dir/extent 三 enum 数组（minItems=1, maxItems=2/3/2）为 required；
        cx/cy/cover 定点两位小数 string+pattern 为 optional（D-9）；
        any_whitespace=True + max_whitespace_cnt=4；strict_mode=True；
        回避 uniqueItems/multipleOf/patternProperties（vLLM 白名单核实不支持）
接入:   子类化 xgr.contrib.hf.LogitsProcessor → RecordingLogitsProcessor
        （裸 assert 换 try/except → ABSTAIN；每 generate 新实例；bitmask 按 config.vocab_size 全词表分配）
解析:   json.loads（语法保证）+ 离线 accept_string + is_completed() 全量复核
三态门: HIT（min_g conf_g ≥ τ_h=0.60 初值）/ LOW（≥ τ_a=0.35，注入且 conf 随码进 PCH）
        / ABSTAIN（<τ_a 或 R_trunc/R_degen 触发 → conf 全 0 → PCH null 路径）
        τ 网格只在 S-val 搜索并预注册，test 单次
试点门: 先跑 500 样本，R_gram=0（>0 = harness bug 停查）且 R_trunc<0.5% 方可跑全集
```

**训练配方**：PCH 按 §2.4，训练码 = 解析码（含语义错与 conf<1，训练分布 = 部署分布）；场 GT 监督；无码空间监督（解析不可导，无梯度回 VLM）。lr ∈ {3e-4, 1e-3} 两点小扫（自定，无上游锚，已标注）。

**参数量**：新增可训 = PCH 一份（Lite 0.52M / Full 3.96M）。原 A1 私设 1.1M 注入器作废（D-1）。

**消融/控制臂**：A1-shuf（样本间打乱码）、**A1-const**（均值码，D-13）、A1-null（全 ABSTAIN）、A1-onepass（单遍 structural-tag 版，测 R_trunc 与语义差）、A1-free（无约束 + 正则解析，只测格式失败率与语义准确率，产 F1_A1 对照线）。

**失败率协议**：R_gram / R_trunc(<0.5%) / R_json(恒 0) / R_degen(>2% → schema 拆 `global` 独立布尔重编译一次) / R_abstain / 覆盖率（<70% 时报覆盖分层 Δ，禁只报全体均值）。

### 3.2 B1 · 冻结 attention 读出臂（F-LMM 式；teacher-forced 档无损，自生成档条件性——D-12）

**结构（张量链；L·H 运行时从 config 推断，下以 36×16=576 例）**

```
[离线缓存 · eager（红线）· torch.no_grad · 逐层 hook 即切即弃（防 L·H·S² 峰值）]
input = prompt + 指令 + reasoning（训练用 GT 文本 teacher-forcing，锚 F-LMM png.py）
slot_ids ∈ {-1,0,1,2,3}^seq：shape 词/direction 词/extent 词/其余（tokenization 期打标）
  自生成档打标（D-14 补规格）：构造侧 schema 词表（5+9+6 词干）+ 预注册同义词表
  （与 A1-free 正则解析同一词表），词形还原匹配；未命中 → slot=3
每层: attn [1,H,S,S] → 切 reasoning 行 × 图像列 [H,T_r,256] → per-slot mean-merge [H,4,256]
跨层 concat → A ∈ [4, 576, 16, 16]（fp16 ≈1.2MB/样本，全集 ≈60GB，uint8 可 15GB）
pad 处理（对 F-LMM 唯一机制性偏离，有据）：有效格重归一化 A←A/Σ_V，pad 清零 + pad_mask
  （F-LMM normalize_input 是全图 sum；本项目实测 pad 吃 53–74% 质量）
[GeoReadout ≈1.7M]
stem: 共享 Conv3×3(576→64) 逐 slot → concat [256,16,16]
trunk: BasicConvBlock(256→128)+MaxPool2 → BasicConvBlock(128→256) → [256,8,8]（锚 mmseg unet.py）
码支: GAP → MLP 256→128→26 = shape 5(**sigmoid/BCE，D-8**) ⊕ dir 9(sigmoid) ⊕ ext 6(sigmoid)
      ⊕ cont 3(**sigmoid → [0,1]，D-8/D-9**)
粗场辅助支(训练期): InterpConv ×2 → conv_seg(64→1) → [1,16,16] logits（GAP 相位丢失保险；推理丢弃）
conf: §2.3 规则（温度标定 + min_i max(p,1−p)）
```

**训练配方**：阶段 A（探针，缓存上）照抄 F-LMM config：AdamW lr=1e-4 betas(0.9,0.999) wd=0.01 clip=1，3% warmup（start_factor 1e-5）→ cosine，bf16，batch=64（缓存后无 F-LMM 的 batch=1 显存约束），8 epochs；损失 = BCE(shape)+BCE(dir)+BCE(ext)+SmoothL1(cont)+1.0·BCE(粗场)+1.0·Dice(粗场，**D-5 豁免件**)。阶段 B（注入）：GeoReadout lr 1e-5（0.1×）+ PCH lr 3e-4，场损失 + 0.1×码辅助损失，~2 epochs；评测双档 = teacher-forced + 自生成（eager 重放前向，与生成期逐 token 一致，无近似）。

**参数量**：GeoReadout ≈1.7M + PCH（0.52M/3.96M）。原私设注入器作废。

**控制臂**：C-shuf（码 batch 内 shuffle）、**B1-const**（D-13）、C-slot（slot_ids 打乱重打标再读出，**需第二遍全量导出 +3–4 GPU·h，D-14 记账**）。

### 3.3 C · learnable query 桥臂（MetaQueries 式；对「采样前 hidden」设定前提最忠实）

**结构（张量链；d_L 以实际骨干为准，示例 2048）**

```
出发张量: reasoning 段每位置末层采样前 hidden H_r ∈ [T_r × d_L]
  （lm_head→Identity 等价物：hook 末层 norm 后输出；span 索引切段；离线落盘 fp16 ≈16GB）
C1 主臂（外桥，Q-Former 式只读 reasoning 段）:
  H_r + PosEmb_mem[256×d_L](N(0,0.02)) → Bridge Block ×2（pre-LN，双向 is_causal=False，
    heads=8=d_b//64，init N(0,0.014)——锚 metaquery transformer_encoder）:
    self-attn [K×512]；cross-attn q=[K×512], k/v = W_k,W_v: d_L→512（异宽投影，锚 BLIP-2
    encoder_width 机制）；FFN 512→2048→512 GELU(tanh)
  Q [K×512]，K ∈ {1,2,4,8,16}，init N(0,0.02)（锚 blip2.py query_tokens）
  mean-pool over K → [512] → LayerNorm → Linear 512→128 → GELU → Linear 128→23
    = 20 multi-hot logits（BCE 组内不 softmax）⊕ 3 连续量（**sigmoid → [0,1]，D-8**）
  conf: §2.3 规则
C2 对照臂（in-context 忠实变体）: K 个 soft token 经 inputs_embeds 外挂（不 resize 词表，
  保共享 lm_head），拼 reasoning 段后增量 forward（复用 KV）；4D attention mask 限域
  query 只见 reasoning 段 + 先序 query；C2-full 消融放开全上下文；参数 ≈0.1M
```

**训练配方**：Stage A（探针逐 K）：AdamW，lr 1e-4 → cosine_with_min_lr → 1e-5（逐字面锚 qwen2p5vl3b_sana.yaml），wd=0（yaml 未覆写），warmup 200 步（原 5000 的同比例缩），batch 64，10 epoch ≈ 6k 步；监督 = GT 码（BCE + SmoothL1）。Stage B（注入）：桥冻结（主设定）或 0.1×lr 联调（消融），PCH 按 §2.4，场损失 + 0.1×码辅助。**24 层 connector 不移植**（316M 为对齐 2304 维扩散条件空间所设，20 维目标必过拟合，2 层为两锚保守内插）。

**参数量**：C1 桥 ≈11.6M + head 0.07M + PCH（0.52M/3.96M）；C2 ≈0.1M。C-Inject 作废（D-1）。

**探针判据（先行）**：P-K 容量曲线（K 五档，读出头恒定，曲线只由 K 驱动）：
- 捕获率（K=8）宏平均 macro-F1 ≥ **F1_A1 + 0.03**（D-10 同度量线）→「hidden 增益成立」；≤ F1_A1 → 关停回退 A1；
- 平台判据 F1(K=4) ≥ F1(K=16) − 0.02 → 目标真低维，**query 高维失败史豁免成立**；至 K=16 仍单调上升 → 豁免失败，路线停止；
- Δ_shuffle ≤ chance+0.05、Δ_const 列必报；层选择附检（末层 vs 0.5/0.75 深度，只报数）。

**控制臂**：A4a shuffle 码 / A4b 常量码（|Δ|<0.005）、A5 打乱指令 reasoning 段（指令条件性，配对差分 + 负控制，禁 AUC 变体）。

---

## 4. 统一判据表（全臂强制；**AUC 一切变体全禁**——红线）

**主表列（每臂每表缺一不可）**：

| 列 | 规则 |
|---|---|
| soft-IoU / hard-IoU | 阈值化 = 匹配 GT 面积 top-k，禁逐场调阈值 |
| grid 级边界 F1 | 禁像素级 3px 版 |
| 中心先验基线列 | 零参数 −中心距离场，同支撑同阈值化；配对 Δ + p 必报 |
| 配对检验 | 图像级配对 bootstrap 10^4 + Wilcoxon |
| Δ_shuffle / Δ_const | 每张消融表双列强制（D-13） |
| 捕获率表 | shape/dir/ext macro-F1（阈 0.5）分组报 + 宏平均；cont MAE（≤0.08 参考线）；全部对 GT 码同度量（D-10） |

**逐臂证伪数字（预注册，test 单次）**：

| 门 | 臂 | 数字 | 触发处置 |
|---|---|---|---|
| Gate-0 | M1（GT 上界） | <+0.008 证伪 / [0.008,0.015) 灰区一次 tap 迭代 / ≥+0.015 晋级 | 证伪 → 三臂全停 |
| G2/G4 | M2/M3 | Δ(M1−M2) ≥ +0.010；\|M3−M0\| ≤ 0.003 | G4 违反 = blocker |
| A1 晋级 | A1 | Δ ≥ max(+0.008, 0.5·Δ_GT·0.82)†；Δ(A1−A1-shuf) ≥ +0.008；R_trunc<0.5%；约束 vs A1-free 语义差 ≥ −2pt（低于 → 两遍式已是默认，改判「约束扭曲」并停臂分析） | 淘汰线 Δ ≤ +0.003 |
| A1 回退 | A1-null | \|Δ\| ≤ 0.002 | 违反 → 全部 Δ 无效 |
| B1 Gate-1 | B1-probe | 宏平均 F1 ≥ F1_A1 + 0.03 成立 / ≤ F1_A1 关停 / 灰区一次改型 | 关停 → 资源转 A1 |
| B1 Gate-2 | B1-inject | Δ ≥ max(+0.008, 0.5·Δ_GT)，p<0.05；C-shuf ≤ +0.002；C-slot 捕获率显著低于 B1-probe（p<0.05） | 不达且 Gate-1 过 → 一次接口消融（tap 单开） |
| B1 Gate-3 | 自生成档 | 增益 < teacher-forced 档 50% → 记「读出成立但需与 A1 合流」 | 预案 = 错词位替换 |
| C 探针 | P-K | 见 §3.3（F1_A1+0.03 线 + 平台判据双杀线） | 无平台 → 路线停 |
| C 晋级 | A2 | Δ ≥ max(+0.008, 0.5·Δ_GT)（D-3，死区已消）；A2 ≤ Δ_GT（违反即查泄漏）；A4a/A4b \|Δ\|<0.005 | A1 过而 A2 不过 → 看 P-K 决定加 K 或转 B1 |

† A1 为 82% 语义封顶的下界臂，晋级线按语义折扣缩放；预期兑现 Δ_GT 的 70–80%×0.82。

**交付物**：每臂按项目规范（REPORT.md 强制三行 / metrics.json / viz success+failure——失败案例至少含「语义错码带偏场」「弃权样本」「多热矛盾」三类 / config / NOTES.md）。

---

## 5. 排期与资源（单卡口径；括号内为第二卡可并行项）

| 阶段 | 内容 | GPU·h | 依赖 |
|---|---|---|---|
| P0 资格赛 | M1-full(2–3h) + M1-lite(<1h) + M1-prob(1–2h) + M2/M2b/M3/M4/M5 eval-only(≈1h) | ≈6–8 | dense 特征缓存（已有） |
| P1 离线产物 | A1 两遍式全集解析 → codes.jsonl(3–6h) + A1-free 诊断遍(1–2h)；B1 teacher-forced attention 缓存(3–4h)；C hidden 导出(≈1h，随生成流程) | ≈8–13（可与 P0 并行跑另一卡） | P0 不依赖 P1 |
| P2 探针 | A1 捕获率表(CPU 级)；B1 阶段 A(<1h)；C P-K 五档(<2h)；C-slot 第二遍导出(3–4h，仅 B1 Gate-1 过后触发) | ≈3–7 | P1；Gate-0 过 |
| P3 注入臂 | A1×2 lr 点(4h)；B1 阶段 B(1–2h) + 自生成档导出(3–4h)；C Stage B(1–3h) + C2(3–4h)；控制臂 eval-only(≈2h) | ≈12–17 | P2 各自探针门过 |
| P4 终评 | 统一评测 + 三份 REPORT + REVIEW-result | ≈2 | P3 |

**合计 ≈ 31–47 GPU·h ≈ 单卡 3–4 卡·日；双卡 ≈ 2–2.5 日。** 每道门失败即截断下游（Gate-0 失败最省：仅 P0 一天出局）。磁盘：attention 缓存 60GB（uint8 15GB）+ hidden 16GB + codes.jsonl 忽略不计。NFS 写一律套 `nfsx`。

---

## 6. 风险清单（评审 fatal / 裁定级发现逐条修复对照）

| # | 评审发现（lens） | 修复 | 状态 |
|---|---|---|---|
| 1 | 四套互不兼容注入器自称「共享」，对照被污染（可实现·套件 1） | D-1：PCH 唯一，三臂注入节作废 | **已修复（本提案 §2/§3）** |
| 2 | GT 上界门三个死刑数字并存（套件 2） | D-2：单一 M1 + 统一 0.008/0.015；D-3 相对量晋级线 | 已修复 |
| 3 | conf 只有 A1 有生产定义（套件 3） | D-4：§2.3 三臂全覆盖 + 温度标定规格 | 已修复 |
| 4 | Dice 与「IoU 禁当目标」红线解释冲突（套件 4） | D-5：书面豁免裁定（辖域=主输出场） | 已修复，**待主 agent 批准生效** |
| 5 | SAM-component 方案书缺件（三 lens 一致 fatal） | D-6：编号关闭并入 PCH，附录 A 记备选 | 已处置 |
| 6 | A1 缺 `<geom>` 引出规格，开工即卡死（可实现·A1 fatal） | D-7：两遍式 + forced prefix，存在性 by construction；500 样本试点门 | 已修复 |
| 7 | B1「绕过 82%」条件性失真（设定相容·B1 fatal） | D-12：正文承认条件性；Gate-3 专测 + 合流预案 | 已修复 |
| 8 | B1 shape softmax+CE 与多热设定冲突（设定相容） | D-8：改 BCE；连续量统一 sigmoid | 已修复 |
| 9 | C 预注册判据死区（A1 存活 [0.010,0.015) 时 A2 构造性不可达）（可实现·C fatal） | D-3：晋级线改相对量 | 已修复 |
| 10 | C tanh 域 [−1,1] 与 GT [0,1] 错位 | D-8 | 已修复 |
| 11 | 捕获率异质度量换算无依据（0.82 词级 vs macro-F1） | D-10：同度量对照线 F1_A1 | 已修复 |
| 12 | C-slot / 自生成档二遍导出成本漏记；自生成 slot 打标无规格 | D-14 记账（各 +3–4h）；§3.2 词表匹配器规格 | 已修复 |
| 13 | 连续量维度四方案四个数（3/~6/~4/q≤4） | D-9：q=3，待构造侧核实（§0.1-1） | 已修复，**留一核实项** |
| 14 | A1 conf 粒度自相矛盾（组级 vs 逐槽位） | §2.3 组级裁定 | 已修复 |
| 15 | A1 required cx/cy/cover 强制编造数值风险 | D-9：optional + conf_cont=0 门 | 已修复 |
| 16 | INJ 用硬 GT 码训练 vs 部署概率码的分布迁移 | M1-prob 消融 + 条件 dropout + M4 扫描 | 已修复 |
| 17 | Lite 砍 tap B 后 rank-1 饱和无逃生口 | D-11：tap B-lite | 已修复 |
| 18 | M5 依赖 dropout 训出的鲁棒性无前置检查 | §2.5 M5 前置条件（G4 + 覆盖日志） | 已修复 |
| 19 | Δ_const 列多处缺失（红线） | D-13：M2b / A1-const / B1-const / A4b 全补 | 已修复 |

**残余开放风险**（无修复、只有监控/预案）：hidden 非线性可及（C 风险 5：中层附检为唯一救济，失败回退 B1）；query 塌缩（监控 cross-attn 熵与两两余弦，观察到才加 NTXentLoss τ=0.07，预注册决策点）；零初始化慢启动（前 ~1k 步平坦为预期，**不许据此提前杀臂**；唯一 rescue = lr×0.5 重启一次）；eager attention 与本 VLM 栈的数值回归（§0.1-4）；掩码后 softmax 在合法延续极少时的置信虚高（A1 校准问题，由三态门 + M4 conf 扫描部分兜底，REPORT 单列 conf 直方图）。

---

## 7. 代码证据索引（全部为方案书作者亲读、评审在线抽查逐字符核实的原始来源）

### 7.1 仓库与关键文件

| 仓库 | 关键文件 | 消费方 |
|---|---|---|
| github.com/mlc-ai/xgrammar | `python/xgrammar/grammar.py`（from_json_schema 签名 + 质量退化 docstring）、`matcher.py`（bitmask/padding 陷阱/is_completed）、`compiler.py`、`structural_tag.py`（组合子全集）、`contrib/hf.py`（LogitsProcessor 裸 assert/单次性） | A1 |
| github.com/vllm-project/vllm | `vllm/v1/structured_output/backend_xgrammar.py`（schema 特性白名单） | A1 |
| github.com/dottxt-ai/outlines（tag 0.1.11） | `outlines/processors/structured.py`（schema→regex→FSM，备胎结论） | A1 |
| github.com/wusize/F-LMM | `flmm/models/frozen_llava.py`（in_channels=heads×layers 运行时覆盖 config、attention 切片、apply_merge）、`flmm/models/mask_head/mask_decoder.py`（normalize_input/conv_seg）、`mask_refiner.py`、`llava/modeling_llava.py`（image_to_overwrite/mask_ids）、`flmm/datasets/png.py`（teacher-forcing 打标）、`configs/llava/frozen_llava_1_5_vicuna_7b_unet_sam_l_refcoco_png.py`（全套训练数字） | B1 |
| github.com/open-mmlab/mmsegmentation | `mmseg/models/backbones/unet.py`（BasicConvBlock/InterpConv） | B1 |
| github.com/facebookresearch/metaquery | `models/model.py`（词表内 soft token 机制 + connector 规格；**注意仓库根级 models/，非 metaquery/models/**）、`models/transformer_encoder.py`（is_causal=False/qk_norm）、`models/metaquery.py`、`configs/qwen2p5vl3b_sana.yaml`（训练配方逐字面） | C |
| github.com/salesforce/LAVIS | `lavis/models/blip2_models/blip2.py`（encoder_width 异宽 k/v） | C |
| github.com/facebookresearch/segment-anything | `modeling/prompt_encoder.py`（not_a_point_embed/no_mask_embed）、`modeling/mask_decoder.py`（hypernetwork MLP + masks 点积）、`modeling/transformer.py`（TwoWayTransformer）、`build_sam.py`（超参出处） | PCH / C |
| github.com/facebookresearch/sam2 | `sam2/modeling/sam/mask_decoder.py`（hypernet 形态 v2 保留 + obj_score_token） | PCH |
| github.com/wenxi-yue/SurgicalSAM | `surgicalSAM/model.py`（Learnable_Prototypes/相似度门控）、`surgicalSAM/train.py`（lr/wd/batch/seed 666/NTXentLoss τ=0.07） | PCH |
| github.com/rui-qian/READ | `model/READ.py`（similarity_as_points；其逐图 min-max 与硬过滤为**反面教材**，明确不移植） | PCH |
| github.com/lllyasviel/ControlNet | `cldm/cldm.py`（make_zero_conv/加性融合） | PCH |

### 7.2 论文（abs 页 / export API 已核实）

arXiv:2411.15100（XGrammar）、2307.09702（Outlines）、2408.02442（格式约束伤 reasoning——A1 自由段设计依据）、2406.05821（F-LMM）、2504.06256（MetaQueries）、2301.12597（BLIP-2）、2304.02643（SAM）、2408.00714（SAM 2）、2412.17741（READ）、2308.08746（SurgicalSAM）、2302.05543（ControlNet）、2205.13147（MRL，仅借嵌套容量曲线思想）。

### 7.3 反诈与 UNVERIFIED-CODE 残余账

- 检索引擎假仓库前科再现并被抓：`xichenpan/MetaQueries` 为假，官方为 facebookresearch/metaquery；
- 残余 UNVERIFIED-CODE（均已绕开或标注，不承载设计决定）：xgrammar `cpp/json_schema_converter.cc` 数值区间细节（定点 string+pattern 回避）、HF `compute_transition_scores`（自研 recorder 替代）、guidance 库（不采用）、CFG 式条件 dropout（无单一代码锚，M4 扫描兜底）、eager 比 SDPA 慢 1.5–2×（经验值）。

---

## 附录 A · SAM 组件档（编号关闭的备选记录，D-6）

前轮已裁定「预训练 SAM 对齐降为组件」。本轮四方案对 SAM 仅作**代码结构锚**引用（TwoWayTransformer / hypernetwork / prompt encoder 回退形态），无运行时权重依赖，故不设独立臂。备选形态留档：若 PCH 资格赛 G5 显示原型嵌入质量差，可评估以 SAM `prompt_encoder` 的 `no_mask_embed` / `not_a_point_embed` **预训练权重初始化 null_g**（SurgicalSAM 对借用嵌入的处置是冻结 + `.detach()`，届时同法），作为一次预注册迭代选项；除此之外 SAM 权重不进入本战役。

## 附录 B · 与现行文档的关系

- 本提案批准后追加进 EXPERIMENTS 活文档 changelog（日期 2026-08-11）；
- 各方案书作为证据附件保留原文，冲突段落以本提案 D-1…D-14 为准；
- 实现审阅 checklist 增补：PCH pad key_padding_mask / bitmask 全词表分配 / Δlogit 域断言 / conf 禁 NaN / D-5 Dice 辖域。
