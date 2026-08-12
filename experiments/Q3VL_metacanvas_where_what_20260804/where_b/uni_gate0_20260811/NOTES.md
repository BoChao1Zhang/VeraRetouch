# uni_gate0_20260811 · 实施前核实记录 / 假设 / 待决策

## 1. 实施前核实(在线与在库,逐条)

| # | 待核实 | 方式 | 结论 |
|---|---|---|---|
| 1 | `U_I` 是否对信号线性 | 读 `q3vl/where/upsample.py` 源码 + 数值实测 | **严格线性**。系数 $a,b$ 中含 $p$ 的项全线性、$\mathrm{var}(I)$ 只含引导图;实测叠加残差 4.67e-16(n=256)。`clamp_domain` 是唯一非线性,已在算子内关闭、clip 移到外面 |
| 2 | `D` 是否对 $w$ 线性 | 读 `basis.py::s_low` + 实测 | **否**。`s_low = 3*tanh(q/3)` 再过 readout,实测中心化残差 0.962,非线性起点精确定位在 tanh |
| 3 | 家族标签来源 | 读 `mask_type_stats.py` + 实跑 | `.vrmeta.json::slot_id`,`MaskResolver(suffix='.vrmeta.json')`;四族 radial/linear/band/semantic |
| 4 | **生成参数 ω 是否可得** | 实拉 `.vrmeta.json` 全字段 | **不可得**。只有 `slot_id` 与 `region`,无中心/轴/角度/衰减率。构造侧生成器在另一仓库 ⇒ C 案编译器改为**拟合**(文档签名 $C(z,\omega,y;x)$ 允许消费 $y$),后果已在 C 卡 REPORT §4-1 记账 |
| 5 | merger hook 是否已落地 | 读 `hiddens.py` / `fpre.py` 当前工作树 | **已落地**(另一 agent 本轮加入):`MergerHook` + `EncodeResult.f_merger` + `want_merger` 开关,`ExitStack` 保证单次前向。本 Gate 阶段未用到(用已有缓存),FAFM 预计算走 `get_image_features` 同一路径 |
| 6 | 判据函数是否已存在 | 读 `q3vl/whereb/metrics.py` | soft-IoU(min/max)、hard-IoU、grid 边界 F1、中心先验、面积匹配 top-k 全部现成;`a/(2-a)` 随机地板此前是两处内联,本轮在 `unifield.field_row` 统一 |
| 7 | 训练 env | 读 `agent-gpu-queue/waves/amort_arm.sh` | `/home/bc/envs/q3vl_sft/bin/python` + `LD_LIBRARY_PATH=/home/bc/miniconda3/envs/llm_factory/lib`;**`--attn eager` 必须**(merger 输出对 kernel 敏感:eager vs sdpa 在 bf16 下相对最大差 0.12) |
| 8 | oracle 潜码位置/格式 | 实读 NFS | `where_a-20260805/oracle/BA-3-Joint/s5/{split}`,`OracleStore.latent(sid, readout)`;仅 BA-3-Joint 有发布 |

## 2. 已采用的保守默认(两可决策,均已留档,未静默拍板)

| # | 决策点 | 两个合理选项 | 采用 | 理由 |
|---|---|---|---|---|
| D1 | Gate 用哪个 split | S-train vs V_where | **B 用 S-train 1000;A/C 用 V_where 全体** | B 只需图像+掩膜,用 train 可完全不碰评测集;A/C 需 $F_{pre}$/merger,而**只有 V_where 有缓存**(NOTES §7.4:train 无缓存,重建要 166 GB)。三卡都是**零训练天花板测量**,V_where 用法符合「探针类一律 S-val 源」 |
| D2 | 粗场网格 | 固定 32×32 vs 原生 H/16 | **原生 H/16**(512×768 → 32×48) | `radius_low=1` 是在该尺度标定的;文档说「约 32×32」正是 H/16。真实网格已逐样本落盘 |
| D3 | B 的 $\varepsilon$ | 1e-4 / 1e-3 / 1e-2 | **1e-3 为主,三档全报** | 项目先例 `run_amort_e5`。实测三档重建差 2e-4,不敏感 |
| D4 | A 的潜码定义 | $\mathbb R^{71}$ 字面 vs 完整潜码 | **两个都跑,判据只读 A71** | 文档字面是 $\mathbb R^{71}$;不许用超集偷偷放宽判据。A71p 作为「71 维限制花了多少钱」的诚实并列列 |
| D5 | A 的 E0a 残差口径 | 未中心化 vs 中心化 | **中心化为判据,未中心化并列** | $D_I$ 对 $w$ 仿射;闭式岭码只需仿射,未中心化会把常数项误算成非线性。两者都挂(0.962 / 2.278),结论不受口径影响 |
| D6 | A 的整臂初始化 | 坐标中位 vs medoid | **medoid**,并报 zeros 初始化敏感度 | 坐标中位抵消成 norm 0.518(典型潜码 2.14),解码近常数场、GN 起点曲率≈0。medoid 是真实已发布潜码,仍是单一整臂常量 |
| D7 | C 的 LPOSS 取舍 | 只报文档管线 vs 并报无传播 | **两者都报,判据取较好者** | 羽化槽按 L2 选(红线禁 IoU 作优化目标),而平滑降 L2 却破坏重叠。取较好者保证 FAIL 不是本卡解释器选型的产物 |
| D8 | C 的段特征源 | `semantic_low`(64) vs merger(2560) | **merger** | 实测(n=14):最佳单段 0.424 vs 0.397,oracle OR 0.546 vs 0.544 |
| D9 | FAFM 条件的名词来源 | GT `<where>` vs instruction | **instruction** | 推理期 `<where>` 由模型自生成、可用;但探针里 GT `<where>` 是标签,拿它建条件通道会高估。instruction 零泄漏 |
| D10 | FAFM 的 $c^*$ 求解分辨率 | 全分辨率 vs 1/2 | **1/2**,并在本次数据上复测等价性 | 实测全分辨率−半分辨率 soft-IoU 差:pilot 5e-5,本次运行 n=32 均值 8.6e-5、最大 2.4e-4;提速 3.7× |

## 3. 本轮修掉的三个自身实现 bug(记录,避免复现)

1. **A 的 Gauss-Newton 发散**:`pi_raw` 整臂常量 12.45(sigmoid 完全饱和)⇒ 该方向曲率≈0 ⇒ 无阻尼 GN 步长爆炸,掩膜飞成常数,A71p 的 E0c 读数 **0.0000**。改为逐样本 accept/reject 的 LM 阻尼(确定性、无分支),A71p 恢复到 0.9314。
2. **C 的量化把范围越界误报成量化代价**:拟合未被约束在量化器的箱范围内,越界后被 clamp 到边界,量化代价虚高到 **0.115**;改为投影梯度(拟合与量化同箱)后降到 **8.6e-5**。
3. **C 的 LPOSS 缺归一化因子**:`(I-αW+μI)f=seed` 少了 $(1-\alpha+\mu)$,场被放大约 5 倍并整体饱和到 1,轮廓 soft-IoU 从 0.409 掉到 0.170(≈GT 面积占比,饱和场的指纹)。修正后仍有害(见 D7),但那是度量错配不是 bug。

另:C 的 radial 家族在纯随机重启下只有 0.50(椭圆目标多盆地),改用 GT 一二阶矩匹配起点后 0.9111 —— 与项目自带 oracle 拟合器 `build_starts` 的 `centroid-radial informed start` 同法。**若不修这三处,三案的 Gate 判读会全错。**

## 4. 主 agent 裁定(2026-08-11,四项全部保守默认,已落实)

| # | 议题 | 裁定 | 本 subagent 的落实动作 |
|---|---|---|---|
| 1 | B 的 semantic 边缘通过(0.8949 vs 0.88) | **门不收紧**——预注册门在出数后不改,**该纪律双向适用** | 判据 6 的通过线原样保留;新增**以家族自身天花板为分母**的 `frac_of_ceiling` 列与 **`neck_tax` 单列**,已写死进 `run_fafm_probe.py::GATE0_FAMILY_CEILING`(semantic 天花板 0.8949,颈部税 0.1051) |
| 2 | A/C 后备(空间 latent token / HiMTok) | **不启动**;两案按各自预注册证伪线处置落档 | 两案 REPORT 已按证伪落档、训练臂未开;后备选项转入明早**方案修正议题清单**(原 `docs/plan/PROPOSAL_REVISION_AGENDA_2026-08-11.md` **已作废归置**;议题消化处见 `docs/EXPERIMENT_INDEX.md` 的 EPR-H17/H18/H19 与死亡清单)交用户 |
| 3 | 探针规模 20k(文档 §3.6 写 ~70k) | **批准**(探针级规模);若探针过判据,全量臂再上 70k | 偏离与理由已写进运行产物 `metrics.json::train.scale_deviation`,含「探针数字不得当全量臂结果引用」的约束 |
| 4 | A 的 E0b 正面结论(放大 ≤7.7 vs 8e5) | **独立入账**,与 B 的谱界互为佐证 | 已在 caseA REPORT §4-2 与总表「三条跨案结论」第 2 条并列记述;两条证据机理不同(A 是扰动实测,B 是谱界 4.6e7→2.56e4) |

**裁定 1 的口径要点(实现审阅请重点核)**:semantic 档若用全体天花板 0.9967 当分母,会把冻结算子在模型介入之前就收走的 **0.1051 颈部税**记到模型头上——把天花板误读成失败。故 `frac_of_ceiling` 逐家族用**该家族自己的** Gate 0 中位,`neck_tax` 单列。

## 5. 纪律自检

- **禁 AUC**:三卡全程无任何 AUC 变体,判据一律 soft-IoU / hard-IoU(面积匹配 top-k)+ grid 边界 F1 + 中心先验列 + `a/(2-a)` 随机地板 + area 分层。
- **IoU 禁作优化目标**:A 的 $C_\varepsilon$ 与 C 的编译器一律 **L2**;C 的羽化槽也按 L2 选(代价见 D7,如实报告)。
- **归一化只用整臂常量**:A 的 $w_0,\rho$ 取整臂 medoid/中位;C 的箱范围预注册;B 的 $c^*$ 值域按整臂统计落盘。**无逐图归一化**。
- **可视化**:掩膜固定 0..1、差分固定 ±1、$c^*$ 用整臂 p99.5 对称色标并在标题标注;**无逐图 min-max**;全分辨率图无需网格反映射(不涉及 16×16 叠图)。
- **grid 判据列在三张上界卡上均饱和到 1.0000**,已在各 REPORT 显式声明「上界卡的正常现象,判别列是全分辨率 soft-IoU」,未拿饱和列充当证据。
- **D-20**:三卡与预计算均 `rm -f` 日志 → nohup → `ps -p <PID>` 实证 → tail 见实质输出 → 写 `job.marker`。全程未用 `pgrep` 判活。
- **NFS**:全程只读 `/mnt/nfs-ro`,未写 `/mnt/nfs`,未触发 `nfsx`。

## 6. 实现审阅后的更正(2026-08-11,`docs/reviews/REVIEW-impl-amort-uni.md` B9)

`run_uni_gate0_a.py` 的 E0a 探针幅度变量 `scale` 被 stage 循环内的同名变量遮蔽,
**256 个探针中 255 个**按上一张图的场范数(38.648)而非整臂常量(2.143)驱动 —— 18 倍过驱动。
已改名 `denom` 并加整臂常量回归守卫,**全卡重跑**。

| 量 | 首版(污染) | 重跑(修复后) | 结论 |
|---|---|---|---|
| E0a 完整 $D_I$ 中心化残差中位 | 0.9624 | **0.5697** | **FAIL 不变**(≫0.01),线性分支仍永久退役 |
| E0a $q$(tanh 之前) | 3.20e-16 | **3.21e-16** | 非线性起点仍精确在 tanh |

**G0a(case B,4.67e-16)未受影响,Gate B 判读不重开**。两条独立证据:(a) case B 的 G0a 循环
没有任何外部幅度变量,探针每轮现抽;(b) 幅度从 1.0 扫到 100.0(含污染值 38.648)时相对残差
恒为 ~4.5e-16、极差比 1.16 —— $U_I$ 严格线性时相对叠加残差按构造与幅度无关。

E0b/E0c 不受该 bug 影响(它们不消费 `scale`),重跑后逐位复现。

