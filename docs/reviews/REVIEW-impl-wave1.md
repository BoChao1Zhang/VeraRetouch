# REVIEW-impl-wave1 — tooling wave-1 五工具实现审阅

- 日期：2026-08-03　审阅者：实现审阅 subagent（独立，Opus/xhigh）
- 审阅对象：`tools/data_splits`（T1）、`tools/cube`（T2）、`tools/harness`（T3）、`tools/construct`（T4）、`tools/scache`（T5）
- 依据：CLAUDE.md（红线速查）、PLAN §3、EXPERIMENTS_v3 §1（INF-1/2/3/5）、DATA_ASSIGNMENT §1–2/§4、IMPL_DOSSIER §4
- 方法：逐工具实读全部源码 + NOTES.md；**自检不信自述**——T1 selfcheck（--quick）与 T3 selfcheck（全量 42 项）由审阅者本机重跑，其余核对落盘产物；T4 的 split 一致性用 T1 旁表独立复算（见 T4-B1 的实测数字）。

## 总判定

| 工具 | 判定 | blocker | 备注 |
|---|---|---|---|
| T1 data_splits | **pass（附 1 个再跑路径 blocker）** | B1 | 当前冻结旁表本身有效可用 |
| T2 cube | **pass** | 0 | 全部判据核实成立 |
| T3 harness | **pass（附 1 个 INF-1 完整性 blocker）** | B2 | 交付代码无缺陷；烘焙一致性缺件系任务卡范围问题 |
| T4 construct | **blocker** | B3 | 生成器本体质量高，但已交付数据违反 split 纪律，须重生成 |
| T5 scache | **pass** | 0 | 判据达成，产物与自述一致 |

**blocker 共 3 个**（B1/B2/B3，修复指引见下）。红线速查逐条过检：PSNR round×255 口径 ✅（T3 metrics.py:42-73）；ΔE00 D65 口径 ✅（T2/T3 双工具链互验差 <0.01）；s 零归一化 ✅（T3 collapse_probes 原样透传）；榜单强制 Δ_const/Δ_shuffle 列 ✅（T3 leaderboard.py MANDATORY + N/A ⚠ + stderr 警告，自检覆盖）；colour 行序（R 最快、order='F'）✅（T2 本地源码核实 + ImageMagick 独立对拍 + identity 双路 ΔE00 < 1e-4）；split 哈希稳定性 ⚠（T1 的 S-split 纯函数稳定，但 T4 内联规则与之**不同构**，见 B3；P-split 再跑不稳定，见 B1）；烘焙一致性一等指标 ⚠（INF-1 缺件，见 B2）。

---

## T1 `tools/data_splits`（S/P split 旁表 + PPR10K 复核 + 行动项 G）

### pass

- **S-split 规则**：`sha1("verasplit-v1:"+source_id)[:8] mod 100`，seed 冻结入 meta 表与 README，纯函数 ⇒ 跨 build 恒同（vr_common.py:47-62）。符合 DATA_ASSIGNMENT §1.1「固定 hash seed、物化旁表」。
- **产物核实**：splits.sqlite3 实查 sources 30,229/1,723/1,700（89.8/5.1/5.1）、presets 3,172/175/175（预估 ≈3,170/176/176），meta.split_seed=verasplit-v1，builds 含 7 个完成 build（l4 已纳入；generated_at 2026-08-02T16:11Z = 本地 08-03 00:11，与「l4 23:55 归档后重生成」自洽）。
- **selfcheck 复跑**（审阅者 --quick 实跑）：7/7 PASS，exit 0。覆盖率、存储 split==独立重算、比例、P 层重推导全过。
- **PPR10K 复核**：verify_ppr10k.json 实查 4,311 源、索引 1–8871、≥8875 命中 0、exit 0——与 DATA_ASSIGNMENT §4-A 已知结论一致。**新发现 mmart_ppr10k 池 326 源疑似落官方 val 组段**如实上报并标 advisory（gid≥1356 边界未逐字核实，诚实标注），处置提交决策，未静默拍板 ✅。
- **行动项 G**：action_g_report.json 实查 totals 1,144,000/1,144,000 可用（normal 315,952 / low 217,512 / abstain 362,616 / null 247,920），7 build 全列；「可用」定义（after 落盘 ∧ I_in 银行可回取）写进报告口径字段。D-RENDER 规模数与自述一致。
- 只用 stdlib、零新依赖；未动仓库既有代码。

### blocker

- **T1-B1（再跑路径）：P-split 层内位置法在 preset 集合变化时不稳定，且 `build_splits.py` 重跑会无声整表重写**。`p_split_layer` 按层内 preset_id 排序取尾部 5% 为 test（vr_common.py:69-94）——若 g4/l5/l6 续入引入新 preset，同层既有 preset 的位置会平移，**train/val/test 归属可能翻转**（P-test 永不进条件化训练的纪律会被静默破坏）；而 build_splits.py:110-112 直接 `os.remove(db_path)` 重建，无任何 diff 门。T1 自己在 REPORT §7.2 已预警但未设防。
  **修复指引**：给 build_splits.py 加**再跑稳定性守卫**——重跑时若旧表存在，先载入旧 assignments：(a) 既有 source_id/preset_id 的 split 与新算不一致 → 硬失败并输出 diff 报告（除非显式 `--allow-reassign`）；(b) 新 preset 推荐改为**只追加**：新 id 按现规则插层但既有 id 归属冻结（或改为对 preset_id 也用带 seed 的纯哈希 + 层内配额修正）。selfcheck.py 增加「与上一版表 diff=0」检查项。**在 g4/l5/l6 续入前必须落地**。

### nit

- selfcheck 的「跨 build 一致性抽查」是同一纯函数重复求值（selfcheck.py:84 对同一 sid 重算集合），构造上必然通过——判据本身由纯函数性质保证，此检查为形式化仪式，可注明或改为「journal 原始行独立重扫比对」。
- `action_g_render_audit.py` 的 `is_local` 判定（`"-l" in b.split("global")[0] or "local" in b`）依赖 build 命名习惯，换命名会失效；建议以 idx 中是否存在 `.cgt.png` 判定。
- `verify_ppr10k.py` 用 basename 末段数字串解析索引，对非常规命名静默取 -1 计入 unparsed——已计数并纳入 clean 判定，可接受。

---

## T2 `tools/cube`（D-CUBE 盘点/解析/Hald/自检，行动项 E + DOSSIER §4.3）

### pass

- **行序约定**：cubelib 注释与 NOTES 均以 colour-science 0.4.7 本地源码为据（reshape order='F'，R 最快）；`apply_lut_grid_sample` 的 permute(3,2,1,0)+grid=(R,G,B) 与 torch 5D grid_sample (x→W,y→H,z→D) 约定推导正确（cubelib.py:97-114），并有三重独立验证：identity 双路 max ΔE00 = 4.2e-5 < 1e-4（selfcheck_summary.json 实查）、格点处双插值器一致 <1e-5、**ImageMagick `-hald-clut` 端到端对拍 mean ΔE00 0.027**（im_crosscheck.json 实查，行序错会呈几十 ΔE）。
- **对账（行动项 E）**：inventory_summary.json 实查 distinct_used=3,522==预期、candidates/recipe 对称差 0、missing=0、盘面 7,098、差额池与 478 建议清单落盘。
- **解析**：7,083/7,098（99.79%），生产 3,522 零失败；15 个失败全为 AppleDouble 资源叉（显式拒收，归因正确）；GBK 再编码与 Resolve 方言恢复通道合理且记录进元数据。
- **.3dl 处置**：OCIO FileFormat3DL.cpp 约定（B 最快、位深按 max 推断、shaper 轴）经 WebFetch 原文核实后内置最小解析器，52 个全部推断 out_scale=1023（parse_report 实查），与 `Mesh 4 10` 头一致；合成 identity .3dl selftest 通过。
- **近恒等剔除**：101 个（全域格点 ΔE00<0.2），实查全不在生产集内。Lab 白点差异（skimage vs colour）归因到位、门修订合理。
- Hald 训/测集互斥且穷尽 8-bit 色空间（gen 内断言）；「颜色空间 split 非图像 split」口径正确。

### blocker

无。

### nit

- `_infer_scale` 按数据 max 推位深：若某 .3dl 输出整体很暗（max<512 而真实 10-bit），会误推 255 导致过亮 1.57×。当前 52 个全落 1023 无实害；更稳做法是解析 `Mesh N M` 头的输出位深字段做主判据、max 推断做回退。
- pairpath 最差单点 ΔE00 6.38（三线性 vs 四面体固有分歧）已在报告注明评测口径建议——正式评测必须**单口径**（GT 一律四面体），提醒进 EXPERIMENTS_v3 固化。
- 33³ npy 落 `/var/cache/veradata/dcube/`（本机盘）——若 GPU 训练机不同机需迁 NFS（已列决策项）。

---

## T3 `tools/harness`（INF-1 统一评测 harness）

### pass

- **PSNR 口径红线**：`quantize255` = round(x*255)，MSE 后 10·log10(255²/mse)，逐图 batch=1（metrics.py:42-73），与 IMPL_DOSSIER §4.1-6 逐字一致。
- **masked 三分**：距二值边界双侧欧氏距离 ≤k 为 band，in/out 为扣带后的核，三区互斥且并集全图（region_partition，自检以独立闭式距离公式逐像素对拍）；软掩膜阈值可配。M7 口径成立。
- **ΔE00**：skimage rgb2lab 默认 D65（本机 inspect 签名核实）→ deltaE_ciede2000；与 T2 的 colour 工具链互验口径差 <0.01（T2 selfcheck 实查）——**两工具 ΔE00 跨工具一致性成立**。
- **Δ_const/Δ_shuffle**：s 原样透传零归一化（红线守住）；置换用 seeded permutation+roll(1) 保证 derangement（推导正确，自检 seed 0–4 逐一验证）；s_∅ 缺省口径以 `delta_const_mode` 字段留痕。σ_s 统计含 M3 >80% 红线自动判定。
- **榜单强制列**：MANDATORY=(delta_const, delta_shuffle)，缺失标 `N/A ⚠` + stderr 警告，排序主键 psnr_in（主指标=掩膜内，符合 PLAN §3 效应量论证）；自检覆盖缺列/警告/排序三情形。
- **自检真实性**：审阅者本机重跑 `selfcheck.py` → **42 passed, 0 failed, exit 0**（非转述）。真实数据冒烟产物齐全（metrics.json、leaderboard_demo.md、viz success×2 + failure×1，failure 案例存在 ✅），sign p=0.03125=2·0.5⁶ 手验一致。
- 冒烟发现「16/24 带 cgt 样本无 .in.jpg」如实上报，对下游是有效警示（与 T5 的「每组仅 1–2 候选带 C_GT 落 sft shards」互证）。

### blocker

- **T3-B2（INF-1 完整性，非交付代码缺陷）：烘焙一致性（四面体插值回读）评测器缺件**。EXPERIMENTS_v3 §1 INF-1 明文含「烘焙一致性（四面体插值回读）」，CLAUDE.md 红线「烘焙一致性从第一天当一等指标」；T3 任务卡 5 文件范围未列，T3 已在 NOTES §四-3 诚实上报。责任在任务卡拆分而非 T3，但 **INF-1 作为基建不完整，RD 系臂与 E22 进实验前必须补齐**。
  **修复指引**：另开小任务卡（预计 ≤1 天）：复用 `tools/cube/cubelib.py`（identity_table/apply_lut_tetrahedral/hald）实现 `tools/harness/bake.py`——render_fn 在 N_s 切片上采样 → 重采样均匀 33³ → 写 .cube → colour 四面体回读 → Hald+自然图上 PSNR/ΔE00，输出并入 metrics.json（榜单可增列）。按 PLAN §3 烘焙预算的判据（ΔE<2 且 <0.3dB）预注册。

### nit

- `band_px=3` 默认系占位（NOTES 已声明）——正式实验前必须在 EXPERIMENTS_v3 定死 k 并全局沿用，否则三分 PSNR 跨实验不可比。
- `evaluate_batch` 的 mean 是逐图 nanmean——三分区空区域（全图掩膜等）被剔除后有效 n 已随 `n_valid_*` 报出，口径清楚；聚合时按掩膜面积分层报告的建议（NOTES §五）应写进正式评测协议。
- 大面积软渐变掩膜下三分区退化问题已被 failure viz 捕获并写建议——处理得当。

---

## T4 `tools/construct`（INF-2 构造数据生成器 L0–L7）

### pass（生成器本体）

- **L0–L7 语义与 PLAN §3 阶梯逐级对齐**：L0 全 1、L1/L2 语义二值（L2=同构独立种子，正确理解「L1/L2 差异在实验期 s 来源」）、L3 软 matte+羽化档、L4 四几何族×羽化×面积轮转全覆盖、L5 错配负控制（渲染掩膜与交付 GT 分开落盘、maskrender 仅审计可见——设计干净）、L6 双软掩膜顺序双变换（双 GT 全保留 ✅ INF-2「GT 掩膜全保留」）、L7 同色区横切（局部 std 选点 + 8 方向最小跨界色距，构造性质量化记录）。L5/L6/L7 先写 NOTES §5 再实现，符合任务卡。
- **变换正确性**：曝光/WB 在线性光域（sRGB EOTF 教科书式、往返自检）、Sat 亮度插值、HSV 与 colorsys 对拍 0 误差、非单调 PWL 程序化保证；幅度档数值缺口（exec 清单 §1.2 仓库不存在）如实上报并给保守默认（D4）。
- **确定性**：SeedSequence(master_seed, level_id, index) 逐样本 RNG，manifest 记 seed_ints/全参数/git commit，重放 bit 级一致（自检项）；每级确定性洗牌修复 ppr10k 簇拥（首轮目检发现并修复，好）。
- **语义 slot 过滤**：实测仅 ~13.5% 候选为 semantic slot，语义级按 `slot_id=semantic-*` 过滤——这是实质性的数据正确性发现（不过滤会把 linear/band/radial 槽掩膜当语义 GT）。
- **自检真实性**：selftest.json 实查 13/13 全 PASS（含真实 shard sha256 抽查 12/12、64 组面积命中 worst 0.0029、重放一致）；gen_stats.json 实查 L4/L5 面积命中 200/200、L5 IoU<0.3 189/200、L6 200/200，与自述一致；16 张拼图落盘。

### blocker

- **T4-B3：内联 S-split 规则与 T1 冻结旁表不同构，已交付的 train/val 两套数据违反 split 纪律**。T4 用 `int(sha1(source_id).hexdigest(),16)%100`（无 seed、全 digest；splits.py:24-25），T1 冻结规则为 `int(sha1("verasplit-v1:"+sid).hexdigest()[:8],16)%100`。**审阅者对 33,652 个真实 source_id 实测：两规则仅 81.2% 一致（6,310 不一致）**。对已交付 sanity 数据实测：
  - train manifest（1,600 样本 / 1,481 源）：按 T1 权威表，其中 **88 源是 S-val、87 源是 S-test**（≈11.8% 污染）；
  - val manifest（192 样本 / 172 源）：其中 **153 源（89%）是 S-train**，仅 9 源真 S-val；
  - 另有 1 源不在 T1 表内（来自在途 build l5/l6——T4 catalog 含 l5/l6，见下一条 nit）。
  DATA_ASSIGNMENT §1.1「所有实验只读旁表，禁止 ad-hoc 切分」为硬规则。T4 在 NOTES D1 已声明此风险且预留了 `--split-table`，属「旁表未就绪时的过渡」，但**该数据在修复前不得进任何实验（含 G2/G3/RD 冒烟）**。
  **修复指引**（机械，半天内）：
  1. 从 `tools/data_splits/splits.sqlite3`（或 splits_sources.csv）导出 jsonl（`{"source_id":...,"split":...}`），或给 `splits.load_split_table` 加 CSV/sqlite 直读；
  2. 用 `--split-table` 重生成 train（seed 20260802）与 val（seed 20260803）两套 + 重跑 selftest + 重出拼图；不在表内的源（l5/l6 在途 build）会自动归 `unknown` 被排除——行为正确；
  3. **在 selftest.py 增加一条硬检查**：manifest 全部 source_id 的 split 与 T1 旁表逐条一致（防回归）；
  4. 删除或明确标记 `SPLIT_RULE`（s_split_v0）为 deprecated，防止后续误用内联规则。

### nit

- **catalog 源池含在途 build l5/l6**（masks.py DEFAULT_BUILDS）：l5/l6 无 journal 归档、源不在 T1 旁表；且 DATA_ASSIGNMENT 行动项 D 拟把 l6 整体划 held-out 打榜集——若采纳，l6 源进 D-CONSTRUCT train 会污染该计划。建议 DEFAULT_BUILDS 收缩到已归档 build（l1–l4），l5/l6 待归档+续入旁表+行动项 D 拍板后再放开。
- 已交付 sanity 批仅 200/级（INF-2 规格 ≥2000/级）——任务卡 sanity 范围内合理，但正式 ≥2000/级生产（`--img-format png`，D7）需在 split 修复与 D4 幅度档追认后另行排产。
- `load_split_table` 仅支持 jsonl，与 T1 产物格式（sqlite/CSV）不直接对接——并入 B3 修复。
- L3 语义羽化用高斯 σ、几何族用 smoothstep 半宽=σ（D3）——口径不一致已如实上报，建议正式生产前统一或在 manifest 中保持现状但于评测分层时分开统计。

---

## T5 `tools/scache`（INF-5 s 缓存服务 + oracle 目录）

### pass

- **API 符合 INF-5**：`<root>/<arm>/<img_id>__<instr_hash>.npy`（float16，默认 32×32，可原生分辨率）+ 同名 meta.json，meta 必含 层号/归一化参数/生成时间/arm 版本 全齐（api.py:144-154）并防 extra_meta 覆盖；原子写（mkstemp+os.replace）；批量接口齐。双下划线分隔相对规格原文单下划线系任务卡细化（NOTES §一-5 记录在案），img_id 含 `__` 时显式报错，键可无歧义反解 ✅。仅依赖 numpy ✅。
- **oracle 构建**：idx.jsonl 定位 + seek/read 直读（短读校验）；C_GT→32×32 用 PIL BOX 'F' 模式（面积加权、支持分数覆盖——比 adaptive_avg_pool 论证正确）；meta 存 (shard, offset_data, length, sha256) 双成员引用可零拷贝回读全分辨率 ✅；instr_hash 取 journal sft.jsonl instruction md5[:12]，非 SFT 行 `noinstr`（--sft-only 已实现）。
- **upsample**：kornia 0.8.2 `guided_blur` 签名本机源码核实；subsample 整除坑发现并 pad/crop 处理（源码级核实，非猜测）；参数改判（k≈scale/4、eps 1e-4、sub=1）基于 16 组真实扫描而非拍脑袋，且大核摊薄小掩膜的归因（k=97 worst 0.356）有数据支撑。
- **判据与产物**：metrics.json 实查 IoU@0.5 mean 0.9695 > 0.95 ✅、median 0.9966、min 0.580、逐条>0.95 占 85.4%；cache 目录实查 185 条 ×2 文件=370 个文件与 n_entries 一致；单条 2,176B≈2KB 合规格；viz best/median/worst 三张齐。小掩膜（area<2% mean 0.645）归因为表示极限并给出 64×64 建议——归因诚实（guided vs bilinear 增益甚微 0.9695 vs 0.9694 也如实报告，未粉饰）。
- meta 同存 source_id/group_id/mask_id，下游可按 T1 旁表过滤 split——与 split 纪律兼容。

### blocker

无。

### nit

- `iter_build_samples`/`read_member` 与 T4 `masks.py` 的 `_scan_one_batch`/`read_member` 是两套独立实现的 shard 定位读取（各 ~40 行，约定一致）——当前无害，第三个消费者出现前建议抽 `tools/shardio.py` 共享（顺带统一 `.in.jpg|.in.png` 回退与短读校验行为）。
- oracle 判据「IoU>0.95」按 mean 判定系 T5 自行选择的口径（逐张分布已并报，D4 提请拍板）——正式验收口径应在 EXPERIMENTS_v3 固化。
- `noinstr` 条目与 SFT 条目同臂共存，下游按 instr_hash 过滤即可；若 RO 臂缓存将全部带指令，建议 oracle 生产跑加 `--sft-only` 与全量各建一档。

---

## 五工具间接口一致性

| 接口 | 判定 | 说明 |
|---|---|---|
| T4 内联 split vs T1 旁表 | ❌ **不同构**（B3） | 实测 81.2% 一致率；修复=挂 T1 表重生成 |
| T4 `--split-table`(jsonl) vs T1 产物(sqlite/CSV) | ⚠ 格式不对接 | 并入 B3：导出 jsonl 或加 CSV 读取 |
| T5 掩膜定位 vs T4 shard 读取器 | ⚠ 重复造轮子（轻） | 两套独立 ranged-read，约定一致、行为正确；建议后续合并 |
| T3 `load_samples` s 命名 vs T5 缓存命名 | ✅ | `{stem}_*.npy` glob 兼容 `__` 双分隔 |
| T3 ΔE00(skimage) vs T2 ΔE00(colour) | ✅ | T2 selfcheck 实测同图对口径差 <0.01（白点差已归因） |
| T3 metrics.json schema vs 各工具报告 | ✅（新定 schema） | leaderboard 依 README schema 解析；T4/T5 的 metrics.json 是工具报告非榜单行，不冲突 |
| T2 npy33 约定 vs 未来 RD 渲染器 | ⚠ 风险移交 | 生产渲染器 `axis_order:"bgr"`——用 D-RENDER after 图当 L_cube 监督前必须对拍（T2 NOTES D4，采纳） |
| T1 完成 build 判据 vs T4 源池 | ⚠ 不一致 | T1=journal 归档存在；T4 catalog 收了在途 l5/l6（见 T4 nit） |

---

## 「待主 agent 决策」汇总（五份 NOTES 归并，均已采保守默认）

**数据治理 / split（优先级高，阻塞面广）**
1. [T1-1] mmart_ppr10k 池 3,674 源中 326 源（gid≥1356，advisory）疑似落 PPR10K 官方 val 组段——E20 报官方口径前须处置（剔除或强制 S-train）。
2. [T1-2] null（未注释终态）组 247,920 对是否计入 D-RENDER（默认计入；剔除则 896,080）。
3. [T1-3] P-split 28 个 n<10 小层无 val/test 名额是否接受（默认接受）。
4. [T4-D1] split 对齐（→ 本审阅升级为 blocker B3，须执行非拍板）。
5. [T4 nit] l5/l6 在途 build 源是否准入 D-CONSTRUCT 源池（建议暂不准入，联动行动项 D「l6 整体 held-out」拍板）。
6. [T5-3] 生产 s_cache 根目录落位（建议 `/mnt/nfs/bc/data/datasets/s_cache/`）。
7. [T2-1] 33³ npy 落盘位置（现 `/var/cache/veradata/dcube/`，是否迁 NFS）。

**语料口径**
8. [T2-2] 补齐至 ~4000 的 478 个建议清单取舍（inventory/supplement_proposal.txt）。
9. [T2-5] 43 个生产 .3dl 按 OCIO 3DMESH 约定内置解析——追认。
10. [T4-D2] T0=恒等（掩膜外原图）——追认。
11. [T4-D3] 语义羽化（高斯 σ）与几何羽化（smoothstep 半宽）口径不统一——拍板。
12. [T4-D4] 变换幅度档数值（exec 清单 §1.2 缺失，T4 保守默认表）——追认或换正式表。
13. [T4-D5] winner_confidence=low 源图/掩膜准入 D-CONSTRUCT（现准入）。
14. [T4-D7] 正式生产 `--img-format png`（sanity 为 JPEG q95）——确认。

**评测口径**
15. [T3-1] 边界带 band_px 定值（现占位 3）——**必须在 EXPERIMENTS_v3 定死**。
16. [T3-2] Δ_const 的 s_∅ 口径（默认=评测集均值常量场；模型自带走 s_null 参数）——确认双轨。
17. [T3-4] metrics.json schema 追认（或提供既定 schema 改 leaderboard 映射）。
18. [T4-D6] L6 单通道便携掩膜 vs 双通道，哪个进 harness。
19. [T5-1] oracle img_id=candidate_id vs source_id（默认前者，meta 全 id 在录改名零成本）。
20. [T5-2] 非 SFT 行候选 `noinstr` 建条目 vs 跳过（默认建）。
21. [T5-4] IoU 判据 mean vs 逐张口径（默认 mean）。

**文档修订**
22. [T2-3] DATA_ASSIGNMENT §2 D-CUBE 行「行动项 B」应为「E」——修订。

---

## 下一 wave 建议（按依赖序）

1. **清 B3（最优先）**：T1 表导出 jsonl → T4 挂表重生成 sanity 两套 + selftest 加旁表一致性硬检查。B3 未清，G2/G3/RD 冒烟、RO 评分一律不得用 D-CONSTRUCT。
2. **清 B1**：build_splits.py 加再跑稳定性守卫 + selfcheck diff 项——必须赶在 g4/l5/l6 归档续入之前。
3. **清 B2**：开烘焙一致性小卡（tools/harness/bake.py，复用 tools/cube），进 metrics.json/榜单；RD 系与 E22 的前置。
4. **band_px 与评测口径冻结**：在 EXPERIMENTS_v3 定死 k、ΔE00 GT 插值口径（四面体单口径）、IoU 判据口径，追加 changelog。
5. **生产渲染器 BGR 轴序一次性对拍**（T2 风险移交）：抽 ≥100 个 (I_in, preset, after) 用 colour 四面体重渲对拍——RD-G Stage-1 用 D-RENDER 做 L_cube 监督的前置。
6. **mmart_ppr10k 裁决**（决策 1）：先从 PPR10K 官方文件序核实 gid→val 精确边界，再定剔除/强制 S-train——E20 前置。
7. **INF-2 正式生产**：D4/D3/D7 拍板后按 ≥2000/级 ×两套 split 排产（PNG）。
8. 低优先：shard ranged-read 合并为 `tools/shardio.py`；T2 差额 478 清单拍板后跑 parse 补入 D-CUBE；小掩膜 s_cache 64×64 档位实验（T5 建议）。

---

## wave-1.5 关账（2026-08-03，复核 subagent 实测，不信自述）

复核方法：全部自检由复核者本机重跑（非转述）；旧归属零变化用备份库独立 diff；分位超额从 metrics.json 逐对重算；Pyright 全 tools/ 重跑。

### B1（T1 再跑稳定性）— **CLOSED**

- 复核者重跑 `tools/data_splits/selfcheck.py` 全量：**8/8 PASS, exit 0**（含新第 8 项稳定性守卫 moved=0, new_assigned=10/10，走真实 `merge_presets` 路径）。
- 备份库（scratchpad/splits.sqlite3.bak）vs 现库独立 diff：sources 33,652/33,652、presets 3,522/3,522，**moved=0, missing=0**；meta.mode_last_run=incremental，gen 全 0。
- 复核者独立性质测试：`p_split_increment({}, ids) == p_split_layer(ids)` n=1..150 全等；两批增量冻结性质 100/100 无翻转、train 恒非空。
- 代码实读：默认增量 append、既有归属只读不删；`os.remove` 仅在显式 `--force` 分支可达且重建前打印新旧 diff；seed 不符 rc=2 硬失败；旧表自动 ALTER 加 gen 列。与修复指引 (a)(b) 逐条对上。

### B2（烘焙一致性评测器）— **CLOSED**

- `tools/harness/bake_consistency.py` 实读：33³ 格点采样 → canonical [r,g,b] LUT → **复用 `tools/cube/cubelib.apply_lut_tetrahedral` 回读（import 复用，非重写）**；评测面 128³ 全奇数留出色（与 GLUT 训练色互斥）+ 可选自然图；ΔE00 D65 与 metrics 同链、PSNR round×255 口径；判据 `DE00_P99_PASS=0.5` 写死。
- 复核者重跑 `tools/harness/selfcheck.py`：**50 passed, 0 failed, exit 0**，§8 两例 8 项全过（identity 逐位 0 / PSNR 触 cap；gamma2 max_abs_err 实测 2.44129e-4 vs 解析 2.44137e-4，差在 LUT float32 舍入量级；natural 路径覆盖）。
- 遗留（不阻塞关账）：结果尚未并入 metrics.json schema v1 / 榜单列（按 D-11 需版本号，另行排卡）；s 条件渲染器的 N_s 切片扫描属调用方职责。

### B3（construct split 纪律）— **REOPEN（代码侧已修好，数据侧未交付）**

代码侧全部核实到位：
- `splits.py` 默认强制读 T1 冻结旁表（sqlite/CSV/jsonl 三格式直读）；内联规则标 DEPRECATED，fallback 仅限旁表不存在，且旁表在场时 1000 源审计不一致即拒绝启动；表外源归 `unknown` 被排除。
- 已生成的部分 train manifest（245 行 / 243 源）复核者按 T1 权威表逐条比对：**0 污染**（245/245 = S-train），manifest 记 `split_rule: t1_side_table:splits.sqlite3(rows=33652)`。
- 幅度档与 PLAN §3「权威数值表（2026-08-03 内联定稿）」逐值一致：exposure {0.15,0.30,0.60,1.20} / wb {0.03,0.06,0.12,0.24} / sat {0.70,0.85,1.15,1.40} / hue {5,10,20,40} / gamma {0.80,0.90,1.10,1.25}；羽化 {0,2,8,24}px、面积 {5,15,40,70}% 同表；manifest 新增 feather_kind 字段。
- selftest.py 已加旁表一致性硬检查（#1 抽 200 行 vs sqlite 直查；#9 --manifests 逐条 manifest source_id==T1 表）；旧污染批已移 `_deprecated/sanity_wave1_inline-split_jpg/`。

REOPEN 理由（交付缺口）：
- **重生成中断且进程已死**：train 仅 L0 完成（200）+ L1 停在 ~52/200（sanity/train/L1 最后写入 01:01:00，此后 16+ 分钟无写入，ps 无 generate.py 进程；gen_train.log 停在「[train/L0] 200 samples in 155.2s」）；**val 整套未生成**；L2–L7 未生成。
- F3-5（重生成后 0 容忍 split 验证 + selftest 重跑 + REPORT/NOTES 更新）未执行（任务 #20 in_progress / #21 pending）。
- 关账条件：重启生成跑完 train+val 两套 → `selftest.py --manifests` 0 容忍全过 → REPORT/NOTES 落档。机制已验证无误，纯执行收尾。

### F4（Pyright 扫尾 + cube 复验）— 部分完成（不设 blocker 编号，附尾巴）

- 复核者重跑 `pyright` 逐模块：**data_splits / harness / construct / cube / scache 全部 0 errors 0 warnings**；`python -m compileall tools/` exit 0。
- cube 三文件改动实读均为类型/守卫层面（None guard、pyright:ignore 定点、cpu_count or 2），NOTES §六记录 im_crosscheck 复验与冻结产物逐位一致。
- 尾巴 1：F4 承诺的 cube selfcheck 全量重跑（第 4 段 nearident，7,083 preset）**未完成**——scratchpad/cube_selfcheck.log 停在 [3/4] pairpath（00:52:35 后无输出，无进程），NOTES §六「见下方追记」的追记缺位。类型层改动风险低，但「零行为变更」的全量复验闭环未合上。
- 尾巴 2：F5 新建的 `tools/bgr_check/` 有 **6 条 Pyright 报错**（PIL.LANCZOS 别名、min(key=dict.get) stub、3 处流程可证的 possibly-unbound——复核者逐条实读均为运行时无害），故「tools/ 整树 Pyright 干净」当前不成立（wave-1 五模块成立）。一次性 gate 工具，低优先清扫即可。

### F5（生产渲染器 bgr 轴序对拍）— 判定有据，**采纳判定 (a)，RD-G Stage-1 gate 放行**

- 复核者重跑 `tools/bgr_check/selfcheck.py`：**13/13 PASS**（含 BGR 注入 p50=9.32 被侦测、输出侧置换被 perm 测验命中、恒等 LUT 无操作、复合端点精确）。
- metrics.json 实查：100 对 / 0 错误 / **100 个互不相同 preset** / g:l = 50:50 / 7 个 prod build 全覆盖；解析器互验 max|diff|=2.98e-8。
- 逐对超额复核者独立重算：g 线 tri−floor p50 超额 **max=0.0000**、l 线 max=0.0925（8-bit alpha 量化归因合理）；p99 ≤ floor+1.0 达 **100/100**；perm 测验 identity(RGB) 最优 **100/100**。
- 反事实灵敏度：swap 中位 7.91 系 **g 线**口径（全体中位 3.73，l 线被 mask 面积稀释至 1.33）——REPORT §五如实分层报告且声明「判定不依赖 swap 下限」，无粉饰。
- 结论：`axis_order:"bgr"` 仅为 `grid[b][g][r]` 内存轴序标注，值通道恒 RGB，生产 after 语义正确，无需修正函数。遗留 2 项建议（unsplash bank 全路径口径回灌 T1 审计；ppr10k/raise6k/fivek_gold 三池 I_in 可回取率复核）转入待决策清单，不阻塞。

### 关账总结

| 项 | 状态 |
|---|---|
| B1 | **CLOSED**（8/8 自检复跑 + 备份库零迁移独立 diff + 独立性质测试） |
| B2 | **CLOSED**（50/50 自检复跑；复用 cube 四面体应用器核实；schema 并入另行排卡） |
| B3 | **REOPEN**（机制修好且部分数据 0 污染实证；但重生成中断进程已死、val 缺、0 容忍验证未跑） |
| F4 | 五模块 Pyright 0 错（复核实测）；cube nearident 全量复验与 bgr_check 6 条 Pyright 为两条尾巴 |
| F5 | **判定 (a) 采纳**，RD-G Stage-1 gate 放行 |

**INF 基建就绪判定：暂不可宣布。** T1/T2/T3/T5 + bgr gate 已就绪（含 B1/B2 关账）；唯 B3 数据交付未完成——D-CONSTRUCT 依赖的 W1 实验（容量阶梯 T1/L0/L1 等）在 train+val 重生成并通过 0 容忍验证前不得开跑；不依赖 D-CONSTRUCT 的准备性工作（harness 冒烟、s_cache、RD-G Stage-1 前置）不受阻。

### B3 最终状态（2026-08-03 追记，B3 收尾执行 agent 实测）

**B3 — CLOSED（数据侧交付完成，关账条件三项全达成）**：清理中断半成品（train L0+L1 部分）后按 T1 旁表全量重生成——train 8 级 ×200 = 1600（seed 20260802，逐级前台生成、逐级 0 容忍核验后拼装，与单命令全量跑逐 bit 等价，`replay.bitwise` 断言）+ val 8 级 ×24 = 192（seed 20260803），全 PNG；`selftest.py --manifests` **15/15 PASS**，其中 `split.manifest_vs_table` 0 容忍硬门 **train 1600/1600、val 192/192 全一致**（另有逐级独立 sqlite 直查核验 8+8 级全 0 错配）；16 张拼图重出（`viz/L{0..7}_{train,val}_grid.png`）；REPORT.md / metrics.json / run_config.json / NOTES.md 已按重生成版落档。**上表 B3 行的 REOPEN 至此清结；「D-CONSTRUCT 依赖的 W1 实验不得开跑」的限制解除。**

---

## 复核（wave-1.6 + W1 batch-1 五路，2026-08-03，独立复核 agent 实测）

### 路 1 · B3 关账 — **判定：CLOSED 成立（复核确认）**

复核者不信任冻结产物，独立重查：

- **产物齐备**：`sanity/train/manifest.jsonl` 1600 行（8 级 ×200）、`sanity/val/manifest.jsonl` 192 行（8 级 ×24），逐行 `level` 分布逐级恰 200/24；manifest 引用文件 **train 5400/5400、val 648/648 全部在盘**（L5 含 maskrender、L6 含 maska/maskb，解释目录 800/1000 PNG 计数）；`img_format` 1792/1792 全 png；seed 恰 {20260802, 20260803}；`split_rule` 1792/1792 全 `t1_side_table:splits.sqlite3(rows=33652)`。
- **0 容忍独立复算**：复核者绕开 selftest，直接 python+sqlite3 对 `tools/data_splits/splits.sqlite3` sources 表（33,652 行）逐条对账：**train 1600/1600 ok、val 192/192 ok，miss=0 wrong=0**。与冻结 `config/selftest.json` 的 `split.manifest_vs_table` PASS 一致。
- **selftest 冻结产物**：15/15 全 PASS 实读确认（含 `split.inline_guard_refuses` 拒启守卫、`replay.bitwise`、`geom.area_within_1pct` worst=0.0029 fails=0/64）。任务卡「13 项」与实际 15 项之差 = wave-1.5 新增 3 项守卫，路 1 自述如实。
- **拼图**：`viz/L{0..7}_{train,val}_grid.png` 16 张全在，mtime 02:01–02:02（重生成后重出）。
- **关键分布**：metrics.json 实读与自述数字逐项吻合（L4/L5 area_hits 200/200；L5 mismatch_iou median 0.126、iou_ok train 188/200 / val 24/24；L6 200/200；L7 cross_boundary median 0.0023；train 1488 distinct source、最大复用 3）。
- **文档**：REPORT.md 为重生成版全文（判据并排表含 0 容忍行）；run_config.json 记录逐级执行方式与等价性论证；`tools/construct/NOTES.md` §九在案；本文件上方「B3 最终状态」块在案。

**结论：B3 关账块所述全部经独立实测属实，B3 CLOSED 维持。逐级拆分 + 逐级验证 + 拼装的执行方式符合长任务纪律，等价性论证（逐样本 seed 确定性 + replay.bitwise）成立。**

### 路 2 · F7 尾巴清扫 — **判定：PASS（两条尾巴均闭环）**

- **Pyright 全树复核者重跑**（pyright 1.1.408，仓库根 `pyrightconfig.json` 生效）：
  `tools/cube` **0 err** ｜ `tools/bgr_check` **0 err**（F4 尾巴 2 的 6 条清零确认）｜ `tools/harness` **0 err** ｜ `tools/scache` **0 err** ｜ `tools/data_splits` **0 err** ｜ `tools/construct` **0 err** ｜ `tools/readout` **4 err**（全部在 `ro9_gl_attention.py`——W1c-2 在编文件，自述如实披露）；`pyright tools/` 整树合计 **4 errors, 0 warnings**，且 4 条全来自该在编文件。`tools/harness/bake_consistency.py` 逐文件 ignore 补丁确认已删（改由配置解析）。
- **cube selfcheck 4/4**（F4 尾巴 1）：scratchpad `f7_full/` 重跑产物实查——run.log 含 [1/4]→[4/4] 全四段、`Exit status: 0`、峰值 RSS 1.6GB；4 个输出（selfcheck_summary.json / near_identity_cull.txt(101 行) / pairpath_presets.jsonl / near_identity_stats.jsonl）与冻结产物 `experiments/tooling-wave1/cube/selfcheck/` **cmp 逐字节一致**（复核者亲测）→ F4「类型层修复零行为变更」闭环最终合上。死因判定（前台 120s 超时 SIGKILL）与日志时间窗证据自洽。`tools/cube/NOTES.md` §六.1 追记在案。
- 建议采纳：readout 4 条 Pyright 并入 W1c-2 完成判据（待决策项 ①）。

### 路 3/4/5 · W1a / W1b / W1c 交付形态（不评实验结论）

**路 3 · W1a（A0+E1）— 形态合规（进行中实验的合规形态）**：
`model/glut_repro/` 10 文件全在（不动既有 model/ 代码确认）。两实验目录均有 NOTES.md / STATUS.md / config/（lut 清单+launch 脚本+run_env.json）/ smoke/（metrics.json+per_lut/per_fit.jsonl 冒烟数字真实落盘）/ logs / runs。STATUS.md 含 PID、日志路径、预计时长、检查命令、gflow 回退理由与排卡记录——复核时 PID 3234243/3234244 均存活（PPID=1 nohup 守护 bash，python 子进程 3234246/3234247 分别在 GPU0/GPU1 上活跃，nvidia-smi 确认），E1 进度 128/3200、A0 已过 GT 缓存进入 chunk 训练，与 ETA 自洽。**缺口（预期内，STATUS 已列为完成后待办）**：顶层 REPORT.md/metrics.json 待全量出；A0 viz/ 仅 2 张 success 无 failure_*（冒烟版）——按交付规范 failure 必须有，**全量收尾时必须补齐，作为 A0/E1 结果验收硬条件**。

**路 4 · W1b（E1b+E2）— 形态合规**：
E1b **全量完成版**齐备：REPORT.md / metrics.json(210K) / per_lut_mean_de00.npz / viz（主曲线+奇异谱+success_r48+failure_r48+cases.json）/ config/run.json / NOTES.md，另保留 _smoke 后缀对照产物，无长任务故无 STATUS 属正常。E2 工具链 5 文件 + 冒烟产物（metrics_smoke.json / results_smoke.jsonl / viz 五族 success+failure 各 5 张 + ring_evidence + ablation）+ REPORT.md（冒烟数字如实标注）+ NOTES.md + STATUS.md（PID 3248254、完成标志 FULL_RUN_DONE、检查命令齐全）；复核时 PID 存活，链路已进入 fit 段（4344 任务，36 workers）。

**路 5 · W1c（RO-9+G1）— 形态合规**：
G1 目录含 REPORT.md（判据表+n=15 中间读数）/ metrics.json（中间版）/ NOTES.md / STATUS.md / config/（采样器脚本+g1_samples.json+env.json+运行脚本）/ viz 9 张（success 5 + failure 4）/ run_smoke30 + run_full + run_shufctrl 三批分目录。STATUS.md 长任务信息完整（两 PID、gbatch job 2 提交-PD-gcancel 全过程留痕、共卡可行性验证、检查命令、完成后分析命令）。复核时 PID 3228455 在 GPU0 活跃（run_full 40/810 条推进中）、PID 3232359 链式等待中；scache `/var/cache/veradata/scache/ro9/` 已有 178 文件真实落盘。冒烟 Gate FAIL（ρ_opp 不分离）如实报告且预注册了失败模式与对照批——**这是合规的负结果披露，非交付缺陷**；结论判读留给结果审阅。

### 「INF 基建就绪」最终宣告 — **是（宣布就绪）**

理由：① B1/B2/B3 三 blocker 全部 CLOSED 且 B3 经复核者独立 0 容忍对账确认；② F4/F7 两条尾巴闭环——cube selfcheck 4/4 逐字节复验 + bgr_check Pyright 清零 + 仓库级 pyrightconfig 使 tools/ 六模块在任意上下文 0 errors（唯一 4 条残留在 W1c-2 在编文件，非基建范畴）；③ split 纪律有三重硬门（T1 旁表强制 + 内联拒启守卫 + manifest×旁表 0 容忍）在 selftest 常驻；④ F5 bgr gate 已放行。**W1 batch-2（依赖 D-CONSTRUCT 的容量阶梯 / RD 臂 / G2 / G3）可以排入下一波。**

排期注意事项（非阻塞）：(a) 两卡当前承载 5 个在途长任务（E1 ~6–9h 最长），batch-2 GPU 任务需等位或共卡评估；(b) 根盘 91%（gtcache ~17GB 可删重建），排新 GPU 任务前建议先清理；(c) 正式 ≥2000/级 D-CONSTRUCT 生产属长任务，届时按纪律走 gflow（unmanaged 进程释放后）；(d) A0/E1/E2/G1 全量收尾时按交付规范补齐 REPORT/metrics/failure viz，作为结果验收硬条件。
