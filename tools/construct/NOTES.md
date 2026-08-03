# T4 构造数据生成器（INF-2, L0–L7）· NOTES

日期：2026-08-02 ｜ 编码 subagent ｜ 分支 lens-exp（只新建文件，未动现有代码）

## 一、引用的权威文档节（只读）

- `docs/PLAN_v2_local-retouch_2026-07-31.md` §3「实验梯」·第二级（L218 合成数据配方；L220–233 容量阶梯 L0–L7 定义表）。
- `docs/EXPERIMENTS_v3_2026-08-02.md` L26（INF-2：八级 ×（幅度 4 档 × 羽化 4 档 × 面积 4 档），每级 ≥2000，GT 掩膜全保留）。
- `docs/DATA_ASSIGNMENT_2026-08-02.md` L18（S-split：`hash(source_id) mod 100`，0–89 train / 90–94 val / 95–99 test）、L42（D-CONSTRUCT：train 用 S-train 源、验证用 S-val 源，两套独立）、L61（INF-2 行：变换参数分布抄 PLAN §3）。
- `docs/IMPL_DOSSIER_2026-08-02.md` §4.4（确定性施加 + 掩膜羽化软化的思路佐证；未给数值范围）。

## 二、已核实事实（本地实证，非检索）

1. **shards 布局**（实读 `prod-l1-local17k-20260731/batch-0000`）：每 batch 有
   `manifest.json / metadata.jsonl / indexes/{catalog.sqlite3, shard-*.idx.jsonl} / shards/shard-*.tar`。
   idx.jsonl 每行含 `member / sample_id / suffix / shard / offset_data / length / sha256`，
   可直接 `seek(offset_data); read(length)` 从 tar 抽取成员，无需 tarfile 顺序扫描（自检里做 sha256 抽查）。
2. **每个 sample 四成员**：`.cgt.png`（单通道 L，软边掩膜）、`.in.jpg|.in.png`（输入图）、`.jpg`（after）、`.vrmeta.json`。
   `.vrmeta.json` 含 `source_id`（如 `src_7de2f9dcaf547048`）、`subject`、`winner_confidence` 等。
3. **cgt 与 in 分辨率不同**（实测 in 1667×2500 vs cgt 1024×1536，同宽高比）→ 读取端必须 resize 对齐。
4. **resize 约定**（沿用仓库既有约定，出处 `git show HEAD:docs/LUT_RENDERER_EXPERIMENT_PROTOCOL.md` L303）：
   RGB 用 Lanczos，soft mask/C_GT 用 bilinear。
4b. **cgt 掩膜并非全是语义掩膜**（首轮 sanity 目检发现，随机 400 条 vrmeta 实测）：
   l 系 slot_id 分布 ≈ linear 35% / band 28.5% / radial 23% / **semantic 13.5%**。
   语义级（L1/L2/L3 及 L5 的提供 GT）**只取 `slot_id=semantic-*` 的候选**，
   catalog 已记 slot_id，manifest 溯源含 slot_id 可审计。
5. 环境：python3.13 / numpy 2.4.6 / PIL 12.2.0 / scipy 1.16.3 均已装，未新装任何依赖。
6. sRGB EOTF（textbook，IEC 61966-2-1，自检中含往返恒等测试）：
   `lin = c/12.92 (c≤0.04045) else ((c+0.055)/1.055)^2.4`；逆变换对偶。
7. 自研向量化 RGB↔HSV 在自检中与标准库 `colorsys` 随机 2000 点对拍（atol 1e-6）。

## 三、S-split 内联实现（T1 旁表未就绪时的过渡）

规则字符串（写进每行 manifest，供 T1 对账）：
`s_split_v0: int(sha1(utf8(source_id)).hexdigest,16) % 100 -> [0,89]=train,[90,94]=val,[95,99]=test`
- 无 salt/seed（DATA_ASSIGNMENT 的「固定 hash seed」以 T1 落表为准；见待决策 D1）。
- 例：`src_7de2f9dcaf547048` → 77 → train（已实测）。

## 四、参数范围：文档缺口与本工具采用值

PLAN §3 写「几何族参数范围见 exec 清单 §1.2」，但全仓库（含 git 全历史）**不存在该清单**。
文档已给死的量照抄：长边 1024；`O=(1−m)·T0(I)+m·T1(I)`（T0=恒等，见 D2）；
羽化 σ∈{0,2,8,24}px（几何族用 smoothstep 半宽=σ；语义族用高斯模糊 σ，见 D3）；
面积 α∈{5,15,40,70}%（对可控几何族按 mask 均值二分求解命中，容差 ±1%）；
变换五类（Exposure ΔEV / WB 对角 / Sat / Hue 旋转 / Tone γ）四档 + 困难对照。

**本工具采用的保守默认（待主 agent 追认，见 D4）**：

| 类 | 域 | 档1/2/3/4（幅度，符号随机） |
|---|---|---|
| Exposure ΔEV | 线性光增益 2^ΔEV | 0.3 / 0.6 / 1.2 / 2.0 EV |
| WB 对角 | 线性光 (1±δ, 1, 1∓δ) | δ = 0.03 / 0.06 / 0.12 / 0.20 |
| Sat（亮度插值） | sRGB, out=Y+k(in−Y) | \|k−1\| = 0.15 / 0.3 / 0.5 / 0.8 |
| Hue 旋转 | HSV h 平移 | 5° / 10° / 20° / 35° |
| Tone γ | sRGB 逐通道幂 | γ=e^±τ, τ = 0.10 / 0.20 / 0.35 / 0.55 |
| 困难对照 | sRGB 分段线性 | 结点最大偏移 0.10 / 0.18 / 0.28 / 0.40 |

几何族采样范围（同为保守默认）：
- radial：圆心 ∈ [0.25,0.75]²（α=70% 时 [0.35,0.65]²），半径由面积二分解出。
- linear：方向 θ∈[0,2π) 均匀；阈值由投影分位数二分解出。
- elliptical：轴比 ρ∈[1.5,4]，旋转 φ∈[0,π)，圆心同 radial，尺度二分解出。
- vignette：中心 ∈ [0.45,0.55]²，椭圆度跟随画幅比 ×[0.85,1.2]，掩膜=外圈（随 d 增大），环半径二分解出。

## 五、L5 / L6 / L7 设计（按任务卡要求先写清再实现）

**L5 错配（负控制）**：渲染用 radial 几何掩膜 m_render（面积/羽化档照常），
但交付给训练侧的 GT 掩膜是**同一张图的语义 cgt 软掩膜**（错配监督）。
约束 IoU(m_render 二值化, cgt 二值化) < 0.30（重采圆心至多 6 次，取最小者），实际 IoU 记入 manifest。
落盘：`*_mask.png` = 提供的（错配）语义 GT；`*_maskrender.png` = 实际渲染掩膜（供审计，训练侧不可见）。
预期：正确使用 s 的模型在此级 FAIL。

**L6 双重叠软掩膜**：两张软几何掩膜 m_a（radial）、m_b（elliptical），
羽化强制 σ∈{8,24}，面积各 ∈{15,40}%，二值化 IoU 约束 ∈[0.10,0.55]（重采至多 8 次，不中取最近者并记录）。
两次不同类的变换**顺序合成**：`O1=(1−m_a)I+m_a·T_a(I)`，`O=(1−m_b)O1+m_b·T_b(O1)`
——重叠区经历双重变换，单一 (mask, LUT) 秩不足以表达，触发秩上限。
落盘：`*_maska.png`、`*_maskb.png` 分开保留（GT 全保留），另存 `*_mask.png`=max(m_a,m_b) 作单通道便携版。
预期 FAIL（触发双轴 gate）。

**L7 边界横切同色区**：构造「掩膜边界不可由 RGB 推断」的样本。
步骤：① 在 1/4 分辨率亮度图上用积分图算 16px 窗局部标准差；
② 在画面中央 60% 区域取局部 std 最低的候选点 p（要求窗口 std < 0.06，不满足则换源图）；
③ 过 p 取随机方向的直线为 linear 掩膜边界（阈值恰过 p，不做面积控制，α 记实测值）；
④ 在 p 两侧沿法向各取半径 40px 半盘，计算两侧 sRGB 均值距离，8 个随机方向里取最小者，
   该「跨界同色度」指标记入 manifest（越小=边界越不可见）；
⑤ 羽化仅 σ∈{0,2}（锐边界是本级要义）。
预期 FAIL；若通过 = 模型偷到 (x,y)，是 bug（PLAN 原话）。

**L2 说明**：PLAN 定义 L2=「L1 换 VLM 实际 s」——差异在实验期的 s 来源，不在数据。
本生成器将 L2 生成为与 L1 同构造、独立种子的另一套样本（level 字段=L2），供 L1/L2 两臂各自持有数据。

## 六、其他实现口径（均记录进 manifest，可复现）

- 源池：仅 `prod-l{1..6}-local17k-*` 六个生产 build（eval100/fresh*/wp*/mini* 是 QA build，
  依 CLAUDE.md 数据纪律不入；g 系无 cgt 掩膜，不入源池）。默认每 build 扫 4 个 batch 建源目录
  （约 5k+ 去重源，S-val 按 5% ≈ 250+ 源，足够 sanity 批）。
- `winner_confidence` **不过滤**：D-CONSTRUCT 只取源图与掩膜几何，不消费标注质量（见 D5）。
- 语义掩膜有效性门：cgt 均值 ∈[0.02,0.85]，否则同源换候选/换源。
- 每样本 RNG：`numpy SeedSequence(master_seed, level_id, index)`，manifest 记 master_seed 与 index，
  重放同参数即 bit 级复现（自检含重放一致性断言）。
- 源选取：源列表按 source_id 排序后，**每级用 `SeedSequence(master_seed, level_id, 0xA5A5)` 洗牌**再按
  index 取（排序裸列表会让 ppr10k_* 前缀簇拥在前——首轮 sanity 实测 70% 样本落 ppr10k 且八级共用同源，已修）。
  源不可用（无语义候选/掩膜不合格/L7 无同色区）时以 `index + attempt*7919` 步进重试，attempt 数记 manifest。
- sanity 批落盘：in/out 用 JPEG q95（sanity 供人检与参数核对；正式 ≥2000/级生产时建议 `--img-format png`），
  掩膜一律 PNG（L，8bit）。
- 幅度/羽化/面积三轴用确定性轮转（i mod 覆盖）保证 200 张内全档覆盖，符号项由 RNG 决定。

## 七、待主 agent 决策（采用保守默认继续，未拍板）

- **D1**：S-split 内联 sha1 规则无 salt。T1 落表若带 seed/salt 或不同 hash，本工具须以 `--split-table` 挂载旁表重跑 sanity（接口已预留）。
- **D2**：T0 取恒等（O 掩膜外=原图）。PLAN 公式允许 T0 非平凡（两侧都变换）；默认恒等是最保守、GT 最干净的读法。
- **D3**：语义掩膜（L3）的羽化档用高斯模糊 σ 实现（cgt 本身已软边，σ=0 档=原样）；几何族按任务卡用 smoothstep（半宽=σ px）。两者语义略不同，若需统一口径请拍板。
- **D4**：§四全部数值范围（exec 清单 §1.2 缺失所致）。幅度档参照摄影常用幅度设定，档4 刻意进入「重编辑」区间；如 T1/主线已有正式范围表，改 `configs` 常量即可重生成。
- **D5**：`winner_confidence=low` 样本的源图与 cgt 是否准入 D-CONSTRUCT（现准入；CLAUDE.md 的 low 禁令针对 SFT 主训与评测 GT，构造集不属两者，但语义掩膜质量可能受 low 影响）。
- **D6**：L6 的单通道便携版 mask=max(m_a,m_b) 仅为方便可视化/粗训练，正式训练侧应吃双通道；哪个进 harness 请拍板。
- **D7**：sanity 批 in/out 用 JPEG q95（体积 ~×10 小）。若审阅要求无损，重跑加 `--img-format png` 即可（manifest 已记格式）。

## 七点五、wave-1.5 修复记录（2026-08-03，清 B3 / F3）

依据：`docs/reviews/REVIEW-impl-wave1.md` T4-B3、`docs/DECISIONS_2026-08-03.md` D-04/05/06/08。

1. **split 挂 T1 冻结旁表（B3 主修复）**：`splits.make_splitter` 默认强制读
   `tools/data_splits/splits.sqlite3`（33,652 源；亦支持 csv/jsonl 外挂）。
   旧内联 sha1 规则（`s_split_v0`，与 T1 冻结规则仅 81.2% 一致）标 **DEPRECATED**，
   仅当旁表文件不存在时 fallback；fallback 启动时若旁表存在则抽 1000 源做
   内联 vs 旁表一致性校验，任一不一致即 RuntimeError 拒绝启动（实测即拒——预期行为，
   等效于旁表在位时内联规则永不可用）。不在旁表内的源归 `unknown` 自动排除
   （catalog 14,374 源中 2 个 unknown，来自在途 build）。manifest 行的 `split_rule`
   字段现记 `t1_side_table:splits.sqlite3(rows=N)`；`source_bucket`（内联桶号）字段删除。
2. **旁表重生成**：旧 sanity 两套（内联 split、JPEG）与 16 张拼图整体移
   `experiments/tooling-wave1/T4_construct/_deprecated/`（未删除）；用旁表重生成
   train 8 级 ×200（seed 20260802）+ val 8 级 ×24（seed 20260803），seed 不变，
   `--img-format png`（D-08 正式格式）。重生成后 0 容忍验证：train 全部源 ∈ S-train、
   val 全部源 ∈ S-val（结果见交付 REPORT.md 与 metrics.json `split_verification`）。
3. **D-05 feather_kind**：manifest 每行新增 `feather_kind` 字段——
   L3=`gaussian_sigma`；L4/L5(渲染掩膜)/L6/L7=`smoothstep_halfwidth`；
   L1/L2=`none`（硬边）；L0=null。评测按 kind 分层报（D-05 裁定）。
4. **D-06 幅度档对齐 PLAN §3 内联权威表**（2026-08-03 定稿表，替换本 NOTES §四旧默认）：
   exposure |ΔEV|∈{0.15,0.30,0.60,1.20}（符号随机）；wb δ∈{0.03,0.06,0.12,0.24}（符号随机）；
   sat k∈{0.70,0.85,1.15,1.40}（**表值直取**，sign 字段记 sign(k−1)）；
   hue ∈{5°,10°,20°,40°}（符号随机）；gamma γ∈{0.80,0.90,1.10,1.25}（表值直取）。
   hardpwl 结点位移档 PLAN 表未给数值，维持 §四默认 {0.10,0.18,0.28,0.40}。
   schema 版本 `t4construct_v1` → **`t4construct_v2`**（字段增删 + 档值变更）。
5. **Pyright 清零**：generate.py 的 None 运算（cgt 可空分支加 assert 收窄，6 处）、
   float→int（lv_stats 显式 `dict[str, float]`）；selftest.py `*img.shape[:2]` 解包改显式 h/w。
   相对导入在仓库根（`tools/__init__.py` 在位）下无报错，`pyright tools/construct/` 0 errors。
6. **selftest 13→15 项**：`split.known_vector` 重写为 `split.table_backed`
   （默认 splitter 须旁表背书 + 200 源 vs sqlite 直查抽检）；新增
   `split.inline_guard_refuses`（旁表在位时内联 fallback 必须拒绝启动）与
   `split.manifest_vs_table`（`--manifests` 全行 source_id 的 split 与 T1 旁表
   逐条一致，0 容忍防回归硬门，审阅 B3 修复指引第 3 条）。
7. **未动**（非本卡范围）：catalog 源池仍含 l5/l6 在途 build（旁表 unknown 自动排除，
   实害为零；DEFAULT_BUILDS 收缩待 ⚑U2 拍板）；变换实现域与 PLAN 表的口径差
   （sat 亮度插值 vs YUV chroma、hue HSV 平移 vs YUV 平面旋转、tone 纯 γ vs γ+S-curve、
   hardpwl 单曲线共通道 vs 表述「通道独立」）——D-06 裁定范围是**档值数值**，
   域实现维持现状，列入下方待决策 D8 提请拍板。

### 待主 agent 决策（wave-1.5 新增）

- **D8**：PLAN §3 权威表的变换**域**描述与现实现不一致（见上第 7 条）。
  档值已按表对齐；域是否也按表改（sat/hue 换 YUV、tone 加 S-curve、hardpwl 拆通道）请拍板——
  改域会使 v2 与 v1 数据不可比，且 hardpwl 拆通道需改 manifest schema。

## 八、自检清单（selftest.py 全绿才交付）

1. sha1 split 规则已知向量 + 分布 sanity（train:val:test ≈ 90:5:5）。
2. idx.jsonl 定位抽取 vs sha256 校验（真实 shard 抽查 12 成员）。
3. RGB↔HSV vs colorsys 对拍；sRGB↔linear 往返恒等（atol 1e-5）。
4. 合成恒等式：m=0 处 O==I；硬掩膜 m=1 处 O==T1(I)（bit 级）。
5. 几何面积求解：四族 × α 四档 × σ 四档，|实测−目标| ≤ 1%（clip 不可达时记录并重采）。
6. 困难对照曲线程序化验证非单调（存在负斜率段）。
7. manifest 重放：同 seed 重生成 → 数组逐元素相等。
8. L5 IoU<0.3、L6 IoU∈[0.1,0.55]、L7 跨界色距中位数 < 普通 linear 掩膜跨界色距中位数。

## 九、wave-1.5 B3 数据侧关账追记（2026-08-03）

- 中断的重生成（train 仅 L0 + L1@~52，无存活进程）已清理重跑：train 逐级前台生成
  （每级一条 `--levels Lk`，逐级过 sqlite 直查 0 容忍核验后按 L0→L7 拼装 manifest；
  逐样本确定性 = `(master_seed, level_id, index)`、采样器无状态，与单命令全量等价），
  val 单命令跑完。1600 + 192 全 PNG，seed 20260802/20260803。
- `selftest --manifests` 15/15 PASS：`split.manifest_vs_table` train 1600/1600、
  val 192/192（0 容忍硬门全过）。16 张拼图重出。交付细节见
  `experiments/tooling-wave1/T4_construct/{REPORT.md,metrics.json,config/}`。
