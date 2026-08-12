# PR-AMORT P1 / P3' · 实施前核实记录 + 假设 + 待决策

> 本文件对 **两个臂共用**（同一份实现，同一份 loss，同一套判据）。
> 交付按臂分开：`amort_p1_20260810/`、`amort_p3prime_20260810/`。
> 依据文件：`amort_p1_20260810/NOTES.md` §7 全节（主 agent 指示「§7.1–7.6 已含全部已定裁定，
> 无需重新裁量」）、`RESEARCH_amortized-oracle-fitting_2026-08-10.md` §4 + 附录 A、
> `WHERE_HEAD_REDESIGN_DELTA_2026-08-10.md` §五/§七、E1/E2/E3/E5 四张 REPORT。

---

## 一、实测核实（全部打开代码/磁盘/权重验证，零转述）

| # | 事实 | 核实方式 | 结果 |
|---|---|---|---|
| V1 | **merger 输出确实没有 hook**，且加 hook 不需要第二次视觉前向 | 读 `transformers 4.57.1` 的 `Qwen3VLVisionModel.forward`：末行 `hidden_states = self.merger(hidden_states)` 后直接 return；`FPreHook` 挂在 `visual.blocks[-1]`，即 merger **之前** | NOTES §7.4 成立。已加 `MergerHook`（`q3vl/where/fpre.py`），挂 `visual.merger`，与 `FPreHook` 在**同一次前向**内并列，protocol 2.3 安全 |
| V2 | **新 hook 数值正确、且行序正确** | 在线前向 vs E1 已落盘缓存（`amort_cache_20260810`，另一条产出路径 `visual(...)[0]`）逐元素比对，6 样本 | **max abs diff 2.98e-08、median diff 0.0、逐行 cosine 最小 0.9999998**。产物：`merger_hook_check.json`、脚本 `q3vl/whereb/scripts/check_merger_hook.py` |
| V3 | **merger 输出对 attention kernel 敏感**（新发现，见 §三-A） | 同一比对在 `sdpa` 下重跑 | **相对 max diff 0.12、corr 0.999**（bf16）。`eager` 下才是 3e-9。⇒ 相似度场的整臂归一化常量**只在拟合它的 kernel 下有效** |
| V4 | Phi-71 的构造与调用签名 | 读 `q3vl/whereb/fields.py:phi_dir_fast` / `predict_fields`；`PHI_DIR_DIM=71 = geo5 + L + S + 64 语义` | P1 直接复用 `predict_fields`（含 `no_autocast` + `require_dtype=float32`，review blocker B4 的护栏原样继承） |
| V5 | guided upsample 的域 | `UpsampleConfig.domain = S_DOMAIN = (-3, 3)`、`clamp_domain=True` | P3' 的场用 `3·tanh(·/3)` **结构性**落在域内 ⇒ s 缓存契约第二种（静默 clamp）失效模式**不可能发生**，不靠断言靠构造 |
| V6 | `.vrmeta.json → slot_id` 取法可用 | `mask_type_stats.py:68` 实测：`MaskResolver(verify="none", suffix=".vrmeta.json")` → `slot_id` → `family_of` | 可用。**注意**：`q3vl/` 里 `MaskResolver` 的默认 suffix 是 `.cgt.png`，`.vrmeta.json` 只经参数传入（一次代码普查曾据此误报「不存在」，此处记录以免复犯） |
| V7 | train split 的 genctx 覆盖 | `GenContextStore(GENCTX_DIR/'train')` → **159,215 行**（= train 全量） | 训练可用 teacher/generated 混合，与 W01/W02 同协议；不需要「缺记录回退 GT」这种静默污染路径 |
| V8 | H/16 栅格**不固定** | 遍历 E1 缓存 400 样本：**13 种不同形状**（32×48 / 48×32 / 40×32 / … / 32×58） | conv 塔改为**逐样本前向**（不做 padded batch），因此本表示里**根本没有 pad 格**；`valid` 全 1 且被断言 |
| V9 | 中心先验列的**公开口径** | 用原始掩膜重算 V_where local 400 | **top-k(匹配面积) IoU = 0.5088**、**grid 边界 F1 = 0.1994**、**随机地板 = 0.2582** —— 与 E5 已发布数字**逐位一致**。见 §三-B |
| V10 | NFS 读写纪律 | `WHERE_A_MASKVIEW_DIR` / `GENCTX_DIR` / `SPLIT_DIR` 全部指向 `/mnt/nfs-ro` | 本实现**零 `/mnt/nfs` 写**，无需 `nfsx` |

---

## 二、发现的三个真问题（都会静默出错，已修）

### A. 中心先验 soft-IoU 列此前会算出**负数**

`q3vl/whereb/metrics.py:center_prior_field` 返回的是 `−到画幅中心距离`，**全域为负**。
直接喂进 `soft_iou_value` 得到的是无意义值（实测 **−5.632**）。

- **为什么它能躲过检查**：`hard_iou` 与 `grid_boundary_f1` 都先做 top-k 阈值化，而阈值化对
  任何**单调映射不变** ⇒ 这两列完全正常。只有 soft 列坏掉，而 soft 列不看图就看不出来。
  本次是**画了五联图、肉眼看到标题写着 `centre prior=-5.632`** 才发现的。
- **修法**：`amort/evaluate.py:center_prior_unit()` 做解析映射到 [0,1]。
  该映射只依赖 `(grid_h, grid_w)`、**不碰任何图像内容**，因此**不是**可视化红线禁止的
  「逐图 min-max」（那条红线针对的是分母会被离群值/pad 格挟持的数据相关归一化，这里两者都不存在）。

### B. 全项目公开数字的口径是 **matched-area top-k IoU，不是裸 soft-IoU**

同一个中心先验场：top-k 口径 **0.5088**（= E5 发布值），裸 soft-IoU **0.4548**。两者差 5.4 个点。
E5 表里 `full_fit 解码 soft-IoU` 与 `full_fit hard-IoU` 一模一样（0.4478/0.4478），正是因为
「阈值化一律匹配 GT 面积的 top-k」之后 soft 与 hard 同值。

⇒ **本实现的主列改为 top-k 口径**（`topk_iou`），裸 soft 以 `soft_iou_raw` 并列但**永不替代**。
否则整块板与 0.5088 / 0.550 / 0.2582 / 0.9737 **不可比**，而这种不可比是不会报错的。
`local_soft_iou_median` 这个 key 保留历史名字（选优代码与旧板对齐），但**承载 top-k 口径**，
已在代码里写明。

---

### C. mask family 普查在队列里失败 83.7%，手跑却 100% 通过 —— **环境差异**

**症状演化**（三次，越查越深，值得完整记录）：

1. 第一版（共享 `MaskResolver`）：队列里 **unknown 54,630 / 75,544 = 72.3%**。
   而 NOTES §1 的已发布普查是 **0 失败**。
2. 最阴的一点：**失败对 family 无偏** ⇒ 幸存直方图看上去完全正常
   （30.9/25.9/25.9/17.3 vs 已发布 30.8/25.5/26.2/17.5），唯一症状是一个大 `unknown` 桶。
   若当初没打印 family 直方图、或把 `unknown` 并进「其它」，**这个 bug 会一路进交付**。
3. 我在**自己的 shell** 里全量复跑：75,544/75,544、44 秒、0 unknown ⇒ 我据此认为
   「线程安全」是根因并已修好。**这个验证是假的。**
4. 加了 `>5% unknown 即 raise` 的硬守卫后，队列里第二次跑**当场炸**，并把真因打了出来：
   `OperationalError: unable to open database file` ×55,512、`OSError: [Errno 24] Too many open files`。

**真因（实测，非推断）**：

| 环境 | `Max open files` 软限 |
|---|---|
| 交互 shell（我验证的地方） | **1,048,576** |
| 队列作业（继承 `pueued`） | **1,024** |

本臂要同时打开 35 个 batch 的 sqlite catalog、maskview/genctx 三个已发布 store，外加一个
已加载的 VLM；在 1024 上直接把描述符耗尽，sqlite 报「打不开数据库文件」。
**线程安全（`InterfaceError`）确实存在，但只是次要成因；队列里的主因自始至终是 fd 耗尽。**

**修法**：入口处把 `RLIMIT_NOFILE` 软限抬到硬限（`family_labels` 内再做一次防御性抬升），
并保留 `>5%` 硬守卫。
**验证方式（关键）**：用 `ulimit -Sn 1024` **复现约束条件**后再跑全量 —— 结果
soft 1024 → 1,048,576，**75,544/75,544、0 unknown、28 秒**。

### 由 C 得到的一条项目纪律（建议进 CLAUDE.md）

> **在错误环境里做的全量验证毫无价值，甚至有害** —— 它给出一个「已修复」的假信号，
> 反而让人停止追查。
> 凡是要验证一个**会在队列/后台环境里出现**的失败，必须先**复现该环境的约束**
> （fd 软限、`CUDA_VISIBLE_DEVICES`、`LD_LIBRARY_PATH`、cwd、noclobber…），再跑验证。
>
> 本轮同一根因的两个变体：
> - `CUDA_VISIBLE_DEVICES=$GPU` ⇒ 卡内逻辑序号恒为 0，`--device cuda:1` 必炸，
>   但**在 0 号卡上会「碰巧」通过**（P1 活、P3' 死，看起来像臂特有 bug）；
> - `pueued` 的 fd 软限 1024 vs shell 的 1,048,576 ⇒ 同一份代码，一个 83.7% 失败、一个 0 失败。
>
> **两者都属于「手跑通过 ⇒ 队列里必炸」类，且都不会自己报错到正确的位置。**
> 配套要求：任何「静默降级」路径（`except: return "unknown"` 之类）必须配一条
> **覆盖率硬守卫**，否则它会把环境问题伪装成数据问题。

### D. 「修好了但没生效」——本轮出现三次，全部只被冒烟测试抓到

同一形态的三个实例（**都不报错，都看起来已修**）：

| # | 修法 | 为什么没生效 | 怎么发现的 |
|---|---|---|---|
| 1 | U4 步数对齐 `max_steps` | 我把 clamp 写在 `total_steps` **赋值之前**，随后被覆盖 | 冒烟：`--max-steps 6` 实跑 **8 步** |
| 2 | 判据表 p 值列 | `metrics.paired_delta` 返回 `p_value`，我读 `p` | 冒烟：整列 p 全是 `n/a`，**Δ 照常显示** |
| 3 | 分离项守卫「没生效」的**误报** | `steps.jsonl` 以 append 打开，`head -3` 读到的是上一轮死跑的行 | 改读 `tail` 后 `frac_with_partner` = 0.20–0.33，与实测 31.4% 吻合 |

**纪律**：
1. **每一处修复都要有一个能证伪它的观测量**，并在冒烟里核对该观测量本身
   （步数对齐 → 看实跑步数；p 值列 → 看 p 值不是 `n/a`；守卫 → 看覆盖率下降）。
   「代码改了 + 不报错」**不是**证据。
2. **日志一律 rotate，禁止跨轮 append**。混跑的 `steps.jsonl` 里 `head` 是死跑、`tail` 是活跑，
   中间没有任何标记；结果审阅只看交付文件夹，必然读错。已在 `AmortTrainer._rotate` 修复
   （**move-aside 而非删除**，遵守「清空重跑前必先备份」）。
3. 第 2 例尤其阴：**丢掉的是整条显著性列，而 Δ 列照常显示**——板子看上去完整。
   凡「有 Δ 必须有 p」的列，缺 p 应当**报错而不是打印 n/a**。

## 二·五、工作纪律：GPU 满载与队列深度（2026-08-11 失守后固化）

### 硬规则

> **任何「取消 / 完成」事件之后，必须立即自查两张卡的待跑队列；某卡为空即刻补一个预备作业。**

失守经过：撤销消融臂时我只执行了「取消」，没有执行「补位」，于是 POOLED 跑完后
gpu0 直接空转，**是用户发现的，不是我发现的**。取消动作天然会掏空队列，所以补位必须与
取消绑定为同一个动作，而不是一个事后想起来的步骤。

**预备作业清单（按优先级维护，用掉一个就补一个）**：
1. `AMORT_P1_CONT` —— P1 续训（与 P3_CONT 对称，收敛态复验「Phi 负债」）
2. `P2STRUCT_A/B` —— P2 族门控结构损失包 500 步短训探针（curv/mono 两档）
3. FAFM 判据修正重跑候选

### 自查命令（每次提交/取消后跑）

```bash
q status | grep -cE "\| gpu0 \| Queued"   # 必须 > 0
q status | grep -cE "\| gpu1 \| Queued"   # 必须 > 0
```

### 同一天踩到的三个队列级陷阱（都不报错）

| # | 现象 | 真因 | 教训 |
|---|---|---|---|
| 1 | 撤销消融后 gpu0 空转 | 取消未绑定补位 | 上面的硬规则 |
| 2 | ~~我自己把 `AMORT_P1_CONT` 杀了~~ **（已更正：不是事故）** | 14:02 的那次 `q cancel` 是**主 agent 执行用户裁定「不做对称续训」**。我从事件日志的时间巧合反推成自己的误杀，**归因错了**。真实存在的问题只是同一时刻我的 shell 循环提交失败（`q: missing GPU`） | ①**禁止用 shell 循环批量提交队列作业**（那条仍成立）；②**更重要**：`q events` 只记录「谁在何时做了什么」，不记录「为什么」——多主体共用队列时，时间相关性**不是**因果证据，别把别人的裁定读成自己的 bug |
| 3 | 「预备作业」提交后立刻开跑、四个训练同时抢卡 | `pueue` 组并行度被改成 **2**（不是 1），`q submit` 于是不排队直接起 | 提交后**看 `q status` 的实际落位**，不要假设它排进了队列；并行度用 `pueue group` 核 |

**推论**：队列状态必须**实测**，不能从「我提交了」推断；**同理，队列事件的归因也必须核对，
不能从时间巧合推断**（第 2 条就是我犯的这个错，方向相反：把别人的正常操作当成自己的事故）。第 2、3 两条都是「命令返回了、
但发生的事和我以为的不一样」，与 fd 软限、`CUDA_VISIBLE_DEVICES` 属同一类
（见 §二-C 的环境差异纪律）。

## 二·五·一、数据纪律新增：重放 GT 时的 stale `.cgt.png` 地雷（2026-08-11，D0-1 审计发现）

构造侧 `mask_id = stable_id("mask", build_id, source_id, physical_key)`
（`canonical_masks.py:176`）**不含 `seed`**；而写入侧 `_write_cgt_once`
（`agent.py:188-201`）见到文件已存在就直接返回：

```python
def _write_cgt_once(mask, path):
    if path.is_file():
        return path      # 采纳上一次的字节
```

**后果**：换 `seed` 重跑但沿用同一 `build_id` / `output_root`，会**静默复用旧的 `.cgt.png` 字节**，
同时把**新的**几何写进 journal —— 掩膜与元数据从此不一致，且没有任何报错。

**纪律**：任何 GT 重放/重生成 **必须换 `build_id` 或换 `output_root`**。
（本轮 D0-7 直接在内存里调 `build_mask_plan`、不写 `.cgt.png`，因此不触雷；
仍按裁定每个 seed 用独立 `build_id`，作为双保险。）

## 二·五·二、P2 结构损失包定案：**不进配方**（2026-08-11）

NaN 数值问题修复后三臂全部复活（`nonfinite_tensors=0`、硬门全过），但**全部低于基线**：

| 臂 | top-k IoU | vs 基线 0.7622 |
|---|---|---|
| P3'_cont 基线 | **0.7622** | — |
| P2 curv/mono 0.05 | 0.7522 | −0.0100 |
| P2 W-A shaped 5× | 0.7486 | −0.0136 |
| P2 W-B shaped 10× | 0.7477 | −0.0145 |

**按预注册判据：结构损失包不进配方**（三个独立数据点同向，不是单次噪声）。
需同步更新 `WHERE_STATE` 的 S8 条目：**结构损失费 IoU，三数据点**。
注：`SHAPE3_B` 仍照跑，但它的角色是**形状残差对照**（层 2 最强形态），**不是 IoU 竞争者**。

## 二·六、已采纳进配方：门控引导上采样（B 档，2026-08-11）

**裁定**：`probe_gated_upsample_20260811` 双判据 PASS，B 档**正式进入 P3' 推理配方**，
`AmortModel.gate_upsample` 默认 `True`；所有后续 P3' 评测默认 B 档，**并保留 A 档一列对照**
（`gate_upsample=False` 复现采纳前行为）。

| 判据 | 预注册 | 实测 | 结论 |
|---|---|---|---|
| 几何族 κ̃ 改善 | 显著改善 | **3.398 → 0.080（−97.6%，配对 p=1e-4）** | PASS |
| 几何族 IoU 不降 | 不降 | **Δ −0.0000（p=0.99）** | PASS（零代价） |
| 语义族负对照 | C 档显著掉 | **bF1 −0.107（p=1e-4）、IoU −0.0118（p=8e-4）** | PASS（引导确实有价值） |
| 路由可用性 | 类型词规则 | **400/400 与 GT family 一致** | 零训练可部署 |

**期望管理（写在这里免得被误读）**：门控修的是**边缘解析质量（κ̃）**，
**不会**提高 top-k IoU——IoU 差分实测就是 0（p=0.99）。把 0.74 推过 0.75 的是别的东西（见下）。

## 二·七、DX-4 惰性损失项（单独立档，进 P2 修复范围）

五项预注册 loss 里有两项在当前权重下**几乎不干活**（反事实伪影审计，占**总加权损失**的比例）：

| 伪影 | TOTAL | BCE | SDF_boundary | area_band |
|---|---|---|---|---|
| contour_warp | 75.1% | 70.5% | **0.44%** | **0.00%** |
| texture_engrave | 7.2% | 7.1% | **0.09%** | **0.00%** |
| global_widen | 166.6% | 157.0% | **1.35%** | 5.58% |

- **全部信号由 BCE 承担**；`SDF_boundary`(w=0.10) 贡献 ≤1.4%，`area_band`(w=0.05) ≈0%；
- 因此「(c) 损失盲区」的正确表述**不是**「损失看不见伪影」（contour_warp 75%、global_widen 167% 都看得见），
  **而是**「两个 shaped 项在当前权重下形同虚设，全部工作压在 BCE 上」；
- **归入 P2 修复范围**：`P2STRUCT` 探针把 **shaped 项权重重标定**作为一个变量组。
  **纪律**：新权重是**新的预注册**（写在开跑前、随 config 落盘），不是看到结果后的调参。

## 三、设计决策（保守默认 + 依据）

| # | 决策 | 取值 | 依据 |
|---|---|---|---|
| D1 | **不给头任何坐标通道 / 位置编码**，卷积用 `reflect` padding | 无坐标、reflect | DELTA §五.6 明令「融合头坐标输入/位置编码禁（防头自学中心先验，结构级免疫）」；E3 已实测两臂 `corr(输出,中心先验)=0.64 > corr(输出,GT)=0.47`。零 padding 是 CNN 学绝对位置的公认后门 |
| D2 | **w 的读出不经任何单向量池化** | `w_raw[j] = Σ_p Φ[p,j]·a_j(p)/P` | 每个系数在**自己基函数的支撑上**读出（任务卡「按 Phi 支撑组织，禁单向量池化」）；恰好是解析解码器的伴随算子，`a_j=const` 时退化为普通投影，起点非退化 |
| D3 | 池化路径**仅**用于 `w0, alpha, rho`（band 共 6 个标量） | 保留 | `R(s;ρ)` 按定义是全局映射，这些量没有空间可住。**这是对「禁池化」的一处明示偏离**，且正是预注册第二轮消融「pooled 单向量对照臂」要判别的对象 |
| D4 | 语义头监督 = **直接回归 `.cgt` 软 alpha**（不先二值化） | 软 alpha | NOTES §5-C 保守默认；`.cgt` 已证为真轮廓（vs 拟合椭圆 IoU 0.698，radial 阳性对照 0.990），其 4.8% 软边带是信号 |
| D5 | 几何三族**不做子路由** | 共用一条 conv 塔 | NOTES §5-B 保守默认；生成文本四分只有 74.2%，且误差全在几何族内部 |
| D6 | 路由只取**语义/几何二分**，且读**条件文本**（部署路径） | `is_semantic_text(ctx.text)` | NOTES §3：二分在 generated 文本上 396/396 = 100.0%。用 GT 文本评会**美化**自己 |
| D7 | `mask_type` 标签**现取不落盘** | 训练时 48 线程现取 | NOTES §5-A 保守默认，避免再造一份会与构造侧漂移的副本 |
| D8 | 训练上下文 = teacher/generated **各 50%** + 15% foreign | `teacher_fraction=0.5` | 与 W01/W02 同协议（可比）；V7 证明 train genctx 全量覆盖，无需静默回退 |
| D9 | **antonym 零损失项** | 只报不训 | NOTES §7.2 明令；E3 勘误后的第一条建议。W01 已 PASS（`median_abs_delta=0.0`），为它造损失会主动破坏一个通过的控制 |
| D10 | **不加任何 w 空间辅助项**（`L_dir` 等） | 不加 | NOTES §7.5：3.92% 训练目标坐在符号翻转边界上；输出空间监督对该翻转天然免疫，加 w 空间项就得先剔样本 |
| D11 | attention kernel 钉 `eager` | `--attn eager` | 与 E1 缓存产出侧一致，且 V3 显示 kernel 会移动 merger 输出 0.12（相对） |

### 换主体配对分离项的形式（我设计的，预注册在此 + `config/loss_preregistration.json`）

```
L_sep = relu( margin − [ d(m, gt_partner) − d(m, gt_own) ] ),  d = 有效格上的平均绝对差
margin = 0.05,  权重 0.30
```

**为什么是这个形式**：E3 勘误后的真实病灶不是「指令没进来」（进来了：换主体 −0.145、
固定短语 −0.230），而是「进来了却只比零参数中心先验高 **+0.033**」。上式的关键性质是
**一个中心先验形状的输出，对 `gt_own` 与 `gt_partner` 等距**，margin 恒为 0，被罚满额 ——
**照抄几何先验拿不到任何分数，只有真正分辨「指令指的是哪个主体」才能降低它**。
这正是值得优化的那个量。

**代价为零的实现**：`ShuffleIndex` 按 `(source_image_id, render_mode)` 分组，partner 是
**同一张图的另一次编辑**，所以只多读一张 `.cgt`，**不多跑一次 VLM 前向**。

### 假指令零掩膜项与换主体项**不合并**（NOTES §7.2 明令）

- `foreign`（跨图）：主体通常**不在**这张图里 ⇒ 目标是 `Σm→0`；
- `shuffled`（同图）：主体**在**图里，只是换了一个 ⇒ 目标是**另一张掩膜**。

`ForeignIndex` 另加**主体名词重叠护栏**：若外来指令的主体名词与本图自己的名词集合相交
（如两图都有 "sky"），该配对**作废**（计数上报），否则「输出空掩膜」这个标签本身就是错的。

---

## 四、待主 agent 决策（**未静默拍板**，已按保守默认继续）

### A.（可能影响 4h 预算）attention kernel 是否维持 `eager`

`eager` 比 `sdpa`/FA2 慢。若实测吞吐不足以在 3.6h 内跑完一个 epoch，两个选项：
1. **保守默认（已采用）**：维持 `eager`，靠 `--max-hours` 硬墙截断，**不足一个 epoch 也照常出板**
   （两臂同样截断 ⇒ A/B 仍然成立）；
2. 两臂**同时**换 `sdpa`/FA2 并**重拟合**相似度场归一化常量（`SimFieldNorm` 会强制断言 kernel 一致，
   不会静默漂移）。

**请裁定**。注：本轮 A/B 的比较是臂间的，任一选择内部自洽；但**跨轮**与 E1/E5 缓存对比时口径不同。

### B. 主列口径变更是否需要回溯标注旧板

§二-B 显示公开数字是 top-k 口径。本卡已对齐。**是否要在 E3/E5 的 REPORT 里补一句口径说明**
（它们本身没错，只是「soft-IoU」这个词承载的是 top-k 含义），由主 agent 定。

### C. `epochs` / 训练量

保守默认 `epochs=1.0` + `max_hours=3.6` 硬墙。local train 75,544 条、effective batch 32
⇒ 名义 2,360 步。若实测每步耗时使 3.6h 跑不完，按 A-1 截断。**未擅自加大 batch 或降采样。**

### D. P3' 的 `gain` 可学（init 2.0）

`sigmoid(3)=0.953` 会给 BCE 垫一个与定位无关的地板，故让 `gain` 可学。
这是 P1 没有的一个自由度（P1 的上限由 `R(s;ρ)` 的 `ρ` 决定）。**两臂并非逐参数对称**，
但都在「同 loss 同判据」下自由达到各自最优 —— 若认为该自由度破坏 A/B 公平性，请裁定固定为常数。

---

## 五、判据与红线自检

- **零 AUC**：全实现无任何 ROC-AUC 代码路径；
- 空间场三列齐备：top-k IoU + grid 边界 F1 + **中心先验列**，另加**随机地板 `a/(2−a)`**；
- **area 分层**（<0.15 / 0.15–0.30 / 0.30–0.45 / ≥0.45）与 **mask_type 分层**、**head 分层**；
- **换主体配对差分**（同图内配对 + 置换检验）与 **antonym 不变性 |Δ|≤0.05**（只报不训）；
- 另加 **E3 证伪列**：`corr(pred, 中心先验)` vs `corr(pred, GT)` 的配对差；
  NOTES §7.1 登记为晋级门 —— 前者仍 > 后者即与 W01/W02 同病，不得晋级；
- 可视化：色标**固定 0..1**、只取有效格、叠图走 `grid_to_img` 严格逆映射；
- checkpoint 选择：硬门（面积比中位 ∈[0.8,1.5]、换主体 Δ>0、antonym |Δ|≤0.05）+
  median top-k IoU 选优，**禁用 val loss**；
- 早警列每步落 `steps.jsonl`：`area_ratio_median`、`std_ratio_median`、`L_*` 分项、
  `frac_semantic_head`，触发即打 `EARLY_WARNING` 标记。

## 六、产物

- 代码：`q3vl/where/fpre.py::MergerHook`、`q3vl/whereb/hiddens.py`（`want_merger`）、
  `q3vl/whereb/amort/{simfield,heads,model,losses,data,trainer,evaluate,viz}.py`、
  `q3vl/whereb/scripts/{run_amort_arm,check_merger_hook}.py`、
  `/home/bc/agent-gpu-queue/waves/amort_arm.sh`
- 核实产物：`amort_p1_20260810/merger_hook_check.json`
