# E2 NOTES：实施前核实 + 假设清单 + 待决策项

## 实施前核实（协议三件事之一）

1. **规格出处**：PLAN_v2 §1.3（基底定稿：14 维 = [1]⊕[x,y,P2(x),P2(y),xy]⊕[L,S]⊕[e1..e6]；ψ=3tanh(q/3)；
   w 规范分解 w0/α/w_dir，‖w_dir‖=1，α≥0，α=0 精确退化全局）+ PLAN §第1.5级（判据表、消融、条件数检查）。
2. **语义掩膜读取器**：`tools/construct/masks.py` SemanticBank（idx.jsonl 偏移 ranged read；
   `semantic_rows_of` 按 slot_id 前缀 semantic-\* 过滤；`load_cgt` bilinear、`load_image` Lanczos = 仓库 L303 约定；
   cgt admissibility 0.02–0.85）。实测 catalog 39,861 行，slot_id 只有 semantic-0/1 两种（另有 linear-\*/radial-\*/band-\*）。
3. **S-split**：SemanticBank 默认走 `tools/data_splits/splits.sqlite3`（T1 旁表，加载 33,652 源），
   split_rule 实测 = `t1_side_table:splits.sqlite3(rows=33652)`，非 fallback ✅。S-val 源 747 个。
   探针类实验一律 S-val 源（CLAUDE.md 数据纪律）✅。
4. **CLIP**：本地缓存 `openai/clip-vit-large-patch14-336`（HF snapshot 已有权重；tokenizer 4 个文件缺失，
   已从官方 repo 直连补齐到同一 snapshot 后 local_files_only 加载）。稠密特征 = MaskCLIP 式末层 value-projection
   读出（末块 attention 换恒等，走 v_proj→out_proj→残差→MLP→post_LN→visual_projection），输入 672²（位置编码双三次插值到 48²）。
   合成双色图上原始 cos-sim 空间对比度极小（CLIP cos-sim 天然窄带），真实图上逐图标准化后信号明确
   （人像 C_GT 与背景通道 sky/foliage/water 强负相关 −0.4~−0.6）——按管线预期工作。
5. **无编造外部事实**：所用模型/文件全部本地核实；未引入 DOSSIER 附录 B 之外的 URL 结论。

## 关键实现口径（假设，已当场核实/自行定标）

- **soft-IoU 形式**：min/max 版 Σmin(m̂,m)/Σmax(m̂,m)（软目标完美恢复=1.0，与 ≥0.97 阈值语义一致；
  乘积版对软掩膜在完美恢复时 <1，无法支撑 0.97 判据）。乘积版同存 metrics（iou_prod）。
  注意：IoU 此处是**离线拟合实验的预注册损失/度量**（PLAN §1.5 明文），不违反「IoU 禁当（训练）优化目标」红线——后者针对在线训练。
- **带通读出** = 两阈值 logistic 带 σ(k(s−μ+h))−σ(k(s−μ−h))（k∈(1,40)、h∈(0.02,2.5) 有界 sigmoid，禁裸 exp）。
  依据：渲染器 s 轴是 M=12 高斯**混合**，可合成平顶带；单高斯读出无平顶，环形（梯形剖面）实测中位数只到 0.78
  （已存档于冒烟记录），logistic 带才是「高斯轴能买到什么」的代表性单带读出。单调读出 = σ(g·s+b)。
- **拟合协议**：L-BFGS（strong Wolfe，max_iter=120，float64），逐掩膜；重启 = logit-LSQ 启发式 + 6 随机 + α≈0 起点
  + 质心-径向解析起点（x²+y² 的 Legendre 换算）；tie-break（损失差<1e-4 时取 α 最小）保证常数掩膜的退化解可达。
  拟合在 stride-2 网格，soft-IoU 在全分辨率评估。
- **几何族参数**：linear/radial/elliptical 按 PLAN §2 数值表适配 512²；wedge = 顶点内置的单支角扇（apex U(0.3,0.7)²，
  半角 U(15°,45°)，角羽化 U(2°,8°)）；**ring（专项）**= 薄环：r_mid U(0.24,0.35)·D，hw = r_mid·U(0.06,0.10)，
  feather = hw·U(0.3,0.6)。
- **常数掩膜度量**：全 0 目标下 min/max IoU 恒 0（分子恒 0），判据本就用 α 与 std（另报 MAE 与补集 IoU），
  IoU 聚合不含 constant 族。
- **几何掩膜的 range/语义通道**：与随机配对的 S-val 真实图像（100 张池，round-robin）绑定——14 维全量在场，
  几何判据同时检验「干扰通道在场时优化器仍找到几何解」。
- **语义通道后处理链**：cos-sim 48² → bilinear 上采样 → guided filter（r=32, eps=1e-3，guide=luma；
  PLAN §1.3「guided filter 作用在基通道上」）→ 逐图标准化 → 对 [1,geo5,L,S] 残差正交化 → 再标准化。
- **语义五类**：l 系 C_GT 语义槽只有 semantic-0/1（皆主体类掩膜），无五类标签；按掩膜内主导内容锚
  （sky/skin/foliage/water/architecture 五类）指派 class5 并分组报告。winner_confidence=low 的候选行剔除（保守）。

## 冒烟后的两处修订（透明记录，非事后调参）

1. **环形族几何修正**：初版采样（r_mid U(0.15,0.35)D，hw U(0.03,0.10)D）会产出「胖环/近圆盘」，
   任何单调读出的解析上界 = 覆盖圆盘 IoU = 1−(r_in/r_out)²，胖环时该上界 0.5–0.77，预注册 ≤0.40 **解析不可达**。
   改为 hw ≤ 0.10·r_mid 的薄环后上界 0.21–0.33。这是掩膜族定义修正（让判据检验读出能力而非掩膜肥瘦），
   拟合器与判据未动；修正在看到冒烟数字后做出，特此声明。
2. **带通读出形式**：单高斯 → logistic 带（理由见上；单高斯冒烟中位数 0.78 作为次要观察保留在本 NOTES）。

## 冒烟结果（全链路验证，_smoke 后缀文件）

线性 0.989 ✅ / 径向椭圆 0.987 ✅ / **环 单调 0.31 vs 带通 0.975**（核心对 ✅✅）/
束状楔形 单调 0.92（超出预期窗 0.55–0.70，见下）/ 语义(n=10) 0.83（差 0.02 达 0.85 门，n=200 待定）/
常数 α_max=1e-3、std_max=1.4e-5 ✅ / Legendre Gram cond 9.0<10（单项式 17.3）✅ /
残差化后块外相关 Frobenius ≈9e-15（精确投影，构造性满足 <0.05）✅。
额外发现：lin3 基下 wedge=0.50（=「领结一半」预测正中），二次项把 wedge 抬到 0.92——
「单支楔形表达不了」的 limitation 需改写为「线性基表达不了、二次基可近似到 ~0.9」。

## 待主 agent 决策

1. **wedge 预期窗改写**：预注册 0.55–0.70 基于领结论证，但二次项（P2/xy 的抛物条带）实测 ~0.92；
   建议 EXPERIMENTS_v3 MB-1 行把 wedge 判据改为「lin3 基 ≈0.5（领结确认）+ 全基 ≥0.85」。
2. **带通读出形式定稿**（logistic 带 vs 单高斯）影响渲染器烘焙口径的措辞——本实验按 logistic 带出数。
3. 语义五类的类平衡（class5 分布由数据决定，预计 skin 主导）是否需要按类配额重采样再跑一遍。
4. 若全量语义中位数落在 0.75–0.85：按判据表走 k=8 重跑 or 查文本基共线，需要主 agent 排卡。

## 复现

```
python3 prep_data.py [--smoke]   # GPU（CLIP 特征 + 掩膜生成 + 残差化记录）
python3 run_fit.py [--smoke] --workers 36
python3 analyze.py [--smoke]     # metrics.json + viz
```
全量以 run_full.sh 后台执行中（见 STATUS.md）。
