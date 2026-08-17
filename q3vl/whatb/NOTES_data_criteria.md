# NOTES — whatb 数据侧 / 判据侧公共件（假设与待决策）

范围：`colorspan.py` / `readout.py` / `colorimetry.py` / `lutdata.py` / `splits.py` /
`queries.py` / `criteria.py` / `publish.py` / `degeneracy.py` 及其单测。
GLUT 前向、生成器、门、`guards.py` 属另一份任务卡，本文件不涉及。

## 保守默认（已按此实施，未静默拍板；需主 agent 确认）

1. **没有改 `q3vl/whereb/readout.py` —— 主 agent 已裁定：维持现状（2026-08-15）。**
   HANDOFF §八·8.4 要求在该文件新增 `seg_color` / `color_span_pool` 两档。裁定理由：
   where 侧六臂（EPR-018..023）结果**已发布**、`q3vl/whereb/readout.py` 的源码 sha256 已冻结
   在各自 run_setup 里，回改该文件会破坏那批已发布结果的冻结前提。因此该文件**逐字节不动**，
   两档留在 `q3vl/whatb/readout.py:154-190`；其余四档（`color_close` / `im_end` / `seg_where`
   / `qtok`）委托 `whereb.readout.build_reply`，`ReplyPlan` / `verify_plan` / `readout_hidden`
   全部复用 whereb 的 —— 每档仍然只有一份实现。
   **若 where 侧将来要用这两档：做一次统一迁移**（把 whatb 的两个分支整段搬进
   `whereb/readout.py`，whatb 侧改成纯委托），**不许两边各写一份**。迁移窗口 = where 侧下一轮
   开跑之前；迁移后 whatb 的 `_OWN_KINDS` 应变成空集合，`tests/test_readout.py` 的委托断言会
   自动覆盖新路径。
2. **`degeneracy.py` 是别名模块，不是第二份实现**：三条退化解断言由 `guards.py`
   （另一份任务卡的产物）实现，本模块只 re-export，避免阈值出现两份。
3. **`losses.py` 必须 import `colorimetry.py` 的 CIELab / ΔE00**，不得自带一份
   （冻结块「共同依赖只写一份」）。`chroma_hue(lab, eps_c=1e-3)` 已按 `L_hc` 的
   `h=(a,b)/max(C,ε_C)` + 硬 mask `1[C≥ε_C]` 口径给出，`n_hc_masked` 数被 mask 的点。
4. **12 个预注册键取 `REQUIRED_EPR024` 代码块的写法**（含 `B3_bucket_retrieval`）。
   §4.H 的散文句「所有 arm 必含 {headline, B0..B2, B4, N1..N3}」漏了 B3，与同节代码块及
   冻结块的 12 键表冲突；本实现以代码块 / 冻结键表为准。
5. **B4 / B6 的选择度量默认 `de76` + 9³ 网格**（§4.C 实测协议：B0 32.79 / B1 25.33 /
   B2 35.37 / B4 9.93 / B6 10.17 就是这套口径）。选出 `ℓ*` 之后，进板的那一列仍按
   headline 口径（ΔE00 / 短边 512 / GT α）重算 —— 两套数字不可混用。
   `LibraryValues.distance_to(..., metric="de00")` 可切换，切换须写进 RESULT 方法节。
6. **B3 桶池只用 train 记录自带的 `minor`**（不用 `splits_presets.csv`）。桶内无池时
   该样本返回 `None` 并计数，**不退化成全库随机**（实测四个 split 都不会触发）。
7. **查询点主档 = 128³ 均匀（偶数 8-bit 级）**，评测未见色档 = 其补集（奇数级）。
   `image_hist` 档只为 EPR-029:575 的消融行提供，主臂不用。
   RNG 一律私有 `torch.Generator`，不碰全局流（where 侧 N1 教训），CPU 生成后再 `.to(device)`
   以保证 CPU / GPU 逐位一致。
8. **`assert_criteria_ran` 只管列**（每个必需键 n>0）；「published 板必须有
   `.contexts.all.headline_normal_only`、interim 板缺席要记账而不是报错」由
   `publish.assert_publishable` 管（§十·10.2：quick eval 抽到全 low 时该键本来就不存在）。
9. **不实现的东西（§4.I 以「函数不存在」落实）**：任何 AUC、逐图 min-max / softmax 归一化、
   分位裁剪均值、顶层 pooled（混 low）headline、IoU。`test_criteria.py` 用 AST 扫 `def` 名与
   `.cpu()` 调用把这两条钉住。

## 仍需上游拍板 / 不在本交付内

- **N2「无关词」语料来源**（HANDOFF §11.2-A）：本层不决定，`criteria.control_columns`
  只消费调用方算好的 `E_N2_irrelevant` / `M_N2_irrelevant`；实际用的是哪种句子由臂写进
  `run_config`。
- **`interp.py`（§4.F IP-A 完整协议 + `interp_grid` 列）不在本交付内**。§4.F-B 的六个路径量
  已由 `criteria.path_quantities()` 给出；P1 臂必须用 `build_board(extra_columns=...)`
  注册 `interp_grid` / `path_len` / `mono_rate` / `oob_rate`，否则 `assert_criteria_ran` 拒绝出板。
- **GT α / 图像的装载（`where_a-20260805/maskviews`、图 tar）不在本交付内**：
  `criteria` 一律吃调用方给的张量，α 的三分层按冻结口径固定在 GT α / 短边 512。
- **z 缓存 `zcache.py`** 不在本交付内；`readout.WhatReadoutBuilder.facts()` 已把
  colorspan 启动断言的记录带进 `run_setup`，缓存侧的 `checkpoint` / `readout_kind` 断言
  由 zcache 自己补。

## 2026-08-15 审阅整改（B1 / B2 / W2 / W3 / W4）落地记录

- **B1 中性色 NaN 梯度（`colorimetry.py`）**：`chroma_hue` 的 `sqrt(a²+b²)` 与 `xyz_to_lab` 的
  立方根分支在中性色处 backward = `0×inf`。修法选的是**前向逐位不变**的一种：
  `_safe_sqrt`（double-`where`，0 处次梯度取 0）+ 立方根 `clamp_min(d3)`（被选中的分支上
  clamp 是恒等）。因此**不引入任何新的 NOVEL numeric，冻结口径 `h=(a,b)/max(C,ε_C)` 不变，
  已发布数字不动**。EPR-029 原来的 `h=ab/√(ab²+ε_C²)` + `LAB_FLOOR=1e-6` 是**改前向**的另一种
  修法，本轮把它撤掉、qdual 改用共享实现，六臂颜色转换只剩一份。
  **待主 agent 知悉的取舍**：EPR-029 那种写法额外把 hue 梯度**限幅**；本写法在
  「预测恰好中性、目标有彩度」处 `∂h/∂a = 1/ε_C = 1e3`（有限但大）。若 EPR-027（`grad_clip=None`，
  审阅 W6 实测 15 步内梯度范数到 2.7e3）出现梯度尖峰，备选就是把 EPR-029 的限幅写法提到共享层
  （只改预测侧一处，六臂同步）。本轮不擅自改，记录在此。
- **W2 z 缓存合一**：`q3vl/whatb/zcache.py` 是唯一实现（原先五份 + g4d 的 `ConditionStore` 共六份，
  四套断言、五种盘上布局）。盘上布局统一为 `<root>/<split>__<tag>/{z.npy,index.jsonl,meta.json}`；
  **dtype 纪律：读写两侧都只接受 fp32（ruling 11.1-5），非预期 dtype 一律报错**
  （`ZCacheDtypeError`），只有 float64→float32 这种「不无中生有精度」的窄化允许。
  qdual 原来的 `np.asarray(data["z"], dtype=np.float32)` 会把 bf16 静默上采成「看起来像 fp32」，
  已删除。`context_source` 不进目录名：`teacher` / `generated` 各占一个 root，
  用启动断言而不是命名来防混用。
- **图像 / GT α 装载**：为了给 EPR-026 补上与其余五臂同口径的 headline 选优，`SampleStore`
  从 `arms/carrier.py` 提到 `q3vl/whatb/evaldata.py`（内容逐字未改，carrier 改为 re-export）。
  **g4d 的 `EvalData` 与 qdual 的 `AlphaStore` 仍是各自的本地实现**（各自吃自己的 eval bundle），
  没有并进来 —— 这是已知欠账，`tests/test_zcache.py::test_no_new_image_and_alpha_loader_appears`
  用白名单把它钉住：再出现第**三**份会直接失败。
- **N2「无关词」语料**（HANDOFF §11.2-A，仍未拍板）：`scripts/build_zcache.py` 的保守默认是
  `--irrelevant-source where_span`（同 split 他样本的 `<where>` 文本，纯空间描述、无色彩词，
  按词数就近取等长），实际用的是哪一种写进缓存 `meta.json` 与 build_report。备选
  `--irrelevant-source file` 读一行一句的外部语料。**请主 agent 拍板**。
- **`--tag none` 的 span 来源**：train / V_where 直接复用 where 侧 `genwhere_v2seg` 已生成的
  `where_ids`/`color_ids`（该 store 实测 159,215 条、`checkpoint` 字段 == 本次基座），
  只跑一次纯前向；V_what 等 where 侧没跑过的 split 自动回落到重新生成（`--genctx auto`）。
- **z 缓存已在跑（2026-08-15 15:10 实测）**：`ZCACHE_A`(gpu0) / `ZCACHE_B`(gpu1) 两个队列作业，
  producer = `/home/bc/data/runs/whatb/zcache_v2seg/_src/build_z.py`（另一个 subagent 的产物，
  刻意不 import `q3vl.whatb`），写的是 `.pt` 编码
  `<split>.<context>.<tag>.zcache.pt` + 同名 `.report.json` 边车。
  **共享 `ZCache` 已能直接读这种编码**（同一个类、同一套断言、同样拒绝非 fp32），
  `ZCacheDir` 解析顺序 = 规范目录 `<split>__<tag>/` 优先，其次 `.pt`。
  因此**不需要重跑**：六臂 `--z-cache /home/bc/data/runs/whatb/zcache_v2seg` 即可。
  `.pt` 里没有 `reply_token_ids`，1% `verify_plan` 回放做不了，`assert_belongs_to` 会在
  `verify_plan_note` 里明写「回放不了 + producer 是谁」而不是静默通过
  （该 producer 自称在写盘时对**每一条** plan 都跑了 `verify_plan`，比 1% 抽样更强，
  但那是它的自述，不是本层的运行时证据）。
  `q3vl/whatb/scripts/build_zcache.py` 保留为备用 producer（带「复用 where 侧 span、只跑纯前向」
  的省时路径），docstring 里已写明不要与在跑作业并行提交。
- **EPR-027 的 eval bundle 有生产者了**：`q3vl/whatb/scripts/build_eval_bundle.py`（新增，**CPU、
  不跑 VLM**）。它按 `run_idgate_arm.py:257-284` 的 schema 写
  `<out>/index.jsonl` + `<out>/<sample_id>.npz`，料全部来自已有产物：图与 GT α 走共享的
  `evaldata.SampleStore`（不新增第三个装载器），四个 z 走共享的 `zcache.ZCacheDir`
  （HANDOFF §4.H 的三条启动断言在产 bundle 时就跑一遍，另一个基座的缓存烘不进 bundle）。
  行集 = `normal_only(load_index(split))`，与其余五臂评测的行集逐条相同，所以跨臂配对 Δ
  在同一批样本上；`low` 不进 bundle ⇒ 板上 `n_low_excluded = 0`（与 carrier/affonly 一致）。
  **两个字段刻意缺席并写进 `meta.json`**：`z_null`（空提示读出 = 又一次基座前向，本产物不跑；
  `z_lambda` 在 λ=1 时逐位返回 z，`REQUIRED_CRITERIA` 无一列读它）与 `alpha_pred`
  （where 臂在该 split 的 `m_pix` 盘上不存在；`field_pred` 不在 `REQUIRED_P2P3`）。
  **一个待拍板**：`alpha_shuffle`（`field_shuffle` 这一必含列的料）的施主池默认取
  `--shuffle-pool local`，即只从有真实 mask 的行里抽（约束：施主 `source_image_id` 必须不同，
  种子置换、无全局 RNG）。原因是 V_what normal 的 567 行里 321 行是 style（GT α 恒等于 1），
  不加限制会让 57% 的行拿到与 GT 相同的「打乱场」，`field_shuffle` 静默测不到东西。
  `--shuffle-pool all` 是字面意义的「另一个样本的场」。**请主 agent 拍板**；重建一次 bundle
  约 1 分钟（567 行 / 4.4 GB，CPU 实测 43 s）。
- **EPR-026 的 eval 段现在出两块板**：`board_functionspace.json`（原样保留，E^grid / 未见色 /
  IP-A / IP-B，它是 EPR-026 自己的 P1 指标）与新增的 `metrics.json`（12 个预注册键 + 4 个 P1 键，
  用共享 `criteria.build_board` + 共享 `evaldata.SampleStore` + 冻结形成式
  `Î=(1-a)I+a f̂(I)`）。根因：eval 段的 per-sample 行只装了 `grid_error` / `unseen_color_error`
  两个函数值列，从来没装图像列，而 `loss_preregistration()` 早就承诺了 16 个键 ——
  典型的「定义了没接线」。因此 `metrics.json` 写盘后**无论有没有 `--publish` 都跑一次
  `assert_criteria_ran`**，`--publish` 再叠 `assert_publishable_interpc`。
  B1 的库均值烘焙（`library_mean_volume`）是 carrier / affonly / idgate 之外的**第四份**同样六行，
  没有合并：合并要同时改五个 runner，上线前不做，记在这里。烘焙格点默认 33（与 affonly / idgate
  同，carrier 是 65），写在板的 `headline_protocol.libmean_grid` 上。
