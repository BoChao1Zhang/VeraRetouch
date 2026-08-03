# NOTES — T5 s 缓存服务（INF-5）+ oracle 目录

日期：2026-08-02 ｜ 实施者：编码 subagent ｜ 规格：EXPERIMENTS_v3 §INF-5

## 一、实施前核实记录（全部为本机直接核实，非检索）

1. **kornia 版本与 `guided_blur` 签名**（打开已装源码
   `/home/bc/miniconda3/lib/python3.13/site-packages/kornia/filters/guided.py` 核实，kornia==0.8.2）：
   `kornia.filters.guided_blur(guidance, input, kernel_size, eps, border_type='reflect', subsample=1)`
   - `guidance`/`input` 均须 `(B,C,H,W)`，batch 与空间尺寸一致；通道数可不同。
   - `guidance` 单通道走灰度路径，多通道走 3×3 协方差求解路径。
   - **坑（源码级）**：`subsample>1` 时内部 `interpolate(scale_factor=1/s)` 向下取整，末端
     `interpolate(scale_factor=s)` 恢复，若 H/W 不能被 subsample 整除会形状不匹配直接报错。
     → `upsample.py` 在 subsample>1 时先 replicate pad 到整除、算完再裁回。
2. **环境**：python3=/home/bc/miniconda3（3.13），torch 2.6.0+cu124，numpy 2.4.6，kornia 0.8.2，PIL 已装。
   **零新增依赖**。CUDA 可用（2 卡；实测 GPU0 空闲、GPU1 有任务 → 自检默认 cuda:0，可 `--device cpu`）。
3. **prod-l1 落盘结构**（实地 ls + 解码核实，与任务卡入口描述略有出入）：
   - `manifest.json` / `metadata.jsonl` 在 **每个 batch 目录内**（`batch-XXXX/{manifest.json,metadata.jsonl,shards/,indexes/}`），顶层没有。
   - shard 索引行含 `member / sample_id / suffix / offset_data / length / sha256 / shard`，
     tar 为无压缩 ustar，可直接 `seek(offset_data); read(length)` 取成员。
   - 每 sample（=候选）成员：`.cgt.png`（L 模式掩膜）、`.in.jpg` 或 `.in.png`（**原生分辨率**输入图，可为 RGBA）、
     `.jpg`（赢家渲染，与掩膜同尺寸）、`.vrmeta.json`（含 group_id/source_id/candidate_id/mask_id）。
   - **C_GT 尺寸不定**（见 1024×1536 与 1024×1368），且与输入图尺寸**不一致**（输入是原生分辨率）
     → 引导上采样时把输入图 resize 到掩膜尺寸作 guide。
   - 掩膜值域 0–255，软边（与 DOSSIER §附录 B 待核清单 B 的已核实结论一致）。
4. **journal 归档**：`/var/cache/veradata/annot_review/journal-archive/prod-l1-local17k-20260731/`
   有 `groups.jsonl / failures.jsonl / sft.jsonl / manifest.json`；`sft.jsonl` 13,978 行 =
   DATA_ASSIGNMENT 表中 l1 行数，`candidate_id` 唯一且带 `instruction` → 用作 instr_hash 来源。
5. **INF-5 规格原文**（EXPERIMENTS_v3 L17、L29）：s 场离线缓存 32×32（约 2KB/图）、
   目录 `s_cache/{arm}/{img_id}_{instr_hash}.npy` + 元数据（层号/归一化参数），含 oracle 目录（GT 掩膜）。
   任务卡细化为 `<img_id>__<instr_hash>.npy`（双下划线）+ float16 + meta 加生成时间/arm 版本，按任务卡执行。

## 二、假设清单（已自行核实/采定，随代码可查）

- IoU 定义：两掩膜各自在 0.5 阈值（归一化域）二值化后 |∩|/|∪|；双空掩膜记 IoU=1（l1 掩膜无空例，防御性定义）。
- 面积加权下采样：PIL `Image.resize(..., Image.BOX)` 于 float32（'F' 模式）图上执行——BOX 滤波即区域
  面积加权平均，支持非整除尺寸的分数覆盖，比 `adaptive_avg_pool2d` 更精确。
- 上采样默认参数：bilinear 到 guide 尺寸 → guided_blur(guide=RGB 原图/255)；kernel_size 自动
  = 2*ceil(scale)+1（scale=短边放大倍率，32→1024 时约 65），eps=1e-3（任务卡域 1e-4~1e-2 的几何中点），
  subsample 默认 8（Fast Guided Filter，He 2015；速度需要，自检含 subsample=1 对照）。
- float16 精度损失对掩膜可忽略（值域 [0,1]，fp16 相对精度 ~1e-3）。
- 「前 100 组」= shard 迭代序（batch-0000 起、索引序）下前 100 个**不同 group_id**；每组全部候选都建 oracle 条目，往返测试对全部条目算 IoU。

## 三、待主 agent 决策（均已用保守默认继续，不阻塞）

1. **oracle 的 img_id 取什么**：默认 `candidate_id`（掩膜是逐候选落盘的，candidate_id 全局唯一、
   可直接回链 groups.jsonl/sft.jsonl）。备选 `source_id`（更贴"图"语义，但同图多候选会互相覆盖，需再并
   mask_id）。**默认：candidate_id**；meta 里同时存 sample_id/group_id/source_id/mask_id，改名零成本。
2. **无 instruction 的候选**（非 SFT 行，l1 中约 8 候选/组只有部分进 SFT）：instr_hash 记 `noinstr`
   仍建条目（oracle 是掩膜真值，与指令无关）；若后续要求 oracle 只覆盖 SFT 行，加 `--sft-only` 即可（已实现）。
3. **s_cache 根目录落位**：工具不硬编码根目录（构造参数必填）；自检写
   `experiments/tooling-wave1/scache/s_cache/`（100 组仅 ~3MB）。生产建议 `/mnt/nfs/bc/data/datasets/s_cache/`，待主 agent 定。
4. **判据口径**：任务卡"IoU>0.95"未写明均值还是逐张。自检两者都报（mean / min / 逐张达标率），
   达标判定按 **mean>0.95** 报告，逐张分布附上。

## 四、实施后记录（2026-08-02）

- 新增文件：`tools/scache/{api.py, oracle.py, upsample.py, selfcheck.py, README.md, NOTES.md}`；
  未改动任何仓库现有代码。
- **上采样默认参数改判**（相对第二节初始假设，依据 prod-l1 实测扫描而非拍脑袋）：
  smoke 10 条 + 16 组参数扫描显示大核把小掩膜质量摊薄（k=97 时 worst IoU 0.356，k=13 时 0.580），
  故 auto kernel 从 2*ceil(scale)+1 改为 ~scale/4 的奇数（32→1536 时 k=13）；
  DEFAULT_EPS 1e-3→1e-4、DEFAULT_SUBSAMPLE 8→1（sub=8 mean 掉 ~0.01）。扫描记录在 REPORT.md。
- 数据事实补充：前 100 组只有 185 个候选带 C_GT 进 shards（每组 1–2 个，非 8 个全落盘）；
  同组两候选可共享同一 mask_id；输入图无 EXIF Orientation（已核，guide 无旋转风险）。
- 自检结果：mean IoU 0.9695 ✅（判据 >0.95）；median 0.9966，min 0.5800；
  小掩膜（area<5%）是表示极限——详见 `experiments/tooling-wave1/scache/REPORT.md` 与 `metrics.json`。

## 五、wave-1.5 修复记录（2026-08-03，F4 Pyright 扫尾）

- 范围裁定：仅类型/导入层面，零行为变更（DECISIONS_2026-08-03 §三 F4）。
- `oracle.py`：`Image.BILINEAR`→`Image.Resampling.BILINEAR`（L94）、`Image.BOX`→`Image.Resampling.BOX`（L102）。
  本机 Pillow 12.2.0 实测：旧模块级常量与 `Resampling` 枚举值相等（2/4），IntEnum 传入 `resize` 行为逐位一致。
- `selfcheck.py`：`Image.NEAREST`→`Image.Resampling.NEAREST`（L52）、`Image.BILINEAR`→`Image.Resampling.BILINEAR`（L112）。
- 任务卡疑点「selfcheck.py 相对导入解析」：pyright（PYRIGHT_PYTHON_FORCE_VERSION=latest，默认配置，仓库无 pyright 配置节）
  对 `tools/scache/` 全量 0 import 报错——`sys.path.insert + from api import` 模式在 pyright 本地目录解析下可解析，无需改动。
- 复验：`pyright tools/scache/` → **0 errors**；重跑 `selfcheck.py --groups 100`（输出至 scratchpad，不覆盖冻结产物）→
  `iou_guided.mean = 0.9694831120741165`，与冻结 metrics.json **逐位一致**，判据 mean>0.95 保持 ✅（IoU@0.5 口径未回退）。
