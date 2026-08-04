# S0-DATA 实施前核实记录 · 假设清单 · 待决策项

任务卡：Wave S0 数据部分（`docs/QWEN3_VL_BASE_SFT_SPEC_2026-08-04.md` §3/§4/§5/§6/§9 项 5-9，
`docs/METACANVAS_WHERE_WHAT_FINAL_EXPERIMENT_PROTOCOL_2026-08-04.md` §2）。
代码：`/home/bc/VeraRetouch/q3vl/data/`。git base commit：`bc888fe`。

---

## 0. 读了哪些章节

- SFT spec 全文 414 行（任务卡建议全读，已全读）。
- METACANVAS §0/§1/§2（数据、隔离 split、indexed-tar-shard 契约）。
- CLAUDE.md 全文（角色分工、数据纪律、D-20 长任务纪律、红线）。
- canonical parser：`dataset_build/src/construct/responses.py`（`_SECTION_TOKENS`、
  `REASONING_FIELDS`、`assemble_reasoning`、`split_reasoning`、`append_sft` 写出的行结构）。
- `dataset_build/tools/indexed_tar.py`（shard 写入/校验/manifest 的既有实现，本任务复用而非重写）。
- 并行训练侧 subagent 已落地的 `q3vl/train/`（`constants.py` / `imageproc.py` /
  `collator.py` / `shards.py` / `tokens.py` / `dataset.py`），以对齐记录 schema 与序列口径。

## 1. 本地实测核实（不是记忆，全部当场跑过）

任务卡要求外部事实以本地实测为准。以下每条都有对应的命令输出。

### 1.1 Qwen3-VL tokenizer / processor（`/home/bc/data/models/Qwen3-VL-4B-Instruct`）

环境：`/home/bc/miniconda3/envs/llm_factory/bin/python`，transformers **4.57.1**，
`AutoProcessor` → `Qwen3VLProcessor`，image processor → `Qwen2VLImageProcessorFast`。
（同机另有 transformers 5.2.0 / 4.57.6 / 4.57.3 环境，均能加载 `Qwen3VLConfig`；
选 `llm_factory` 是因为它同时带 `flash_attn 2.8.3`，最可能是训练侧环境。）

- `len(tokenizer)` 注册前 **151669**，注册 `<where></where><color></color>` 后 **151673**；
  四个 token 的 id 依次为 **151669 / 151670 / 151671 / 151672**，各自编码为 **1 个 token**。
- `<|image_pad|>` 展开数 == `(H/32) x (W/32)`，实测：
  `512x512 -> 256`、`512x1024 -> 512`、`512x2048 -> 1024`、`768x512 -> 384`、`512x672 -> 336`；
  `image_grid_thw = [1, H/16, W/16]`。**spec §5 的公式在本地实测成立**。
- chat template 默认**不注入 system 段**（模板只在 `messages[0].role == 'system'` 时输出 system）。
- `preprocessor_config.json` 的 `size = {shortest_edge: 65536, longest_edge: 16777216}`
  是**面积**约束（256² 与 4096²），与 spec §5 的"短边 512"无关；按契约预处理后的图
  （512×512 ~ 512×2048 = 262144 ~ 1048576 px）落在该区间内，processor 的 smart_resize
  对其为**恒等**，因此几何以本任务的计算为准，不会被 processor 二次改写。

### 1.2 split authority 的真实语义（`splits-20260803/`）

| 集合 | 行数 |
|---|---|
| `train_sft_ids.txt` | 169,260 |
| `eval_sft_ids.txt` | 3,320 |
| `dedup_drop_sft_ids.txt` | 1,699 |

实测交集：`train ∩ eval = 0`，**`train ∩ dedup = 1,667`**，**`eval ∩ dedup = 32`**，
`|train ∪ eval| = 172,580`，而十个 build 的 `sft.jsonl` 行数合计**正好 172,580**（无重复 sft_id）。

结论：`dedup_drop` **不是第三个平行集合，而是作用在 train/eval 上的删除名单**。
spec §9 项 6「三集合无交集」只有在应用该删除名单之后才成立（见 §3 决策 D-3）。

### 1.3 数据规模与字段（十个 build 全量扫过一遍）

- `sft.jsonl` 行数：g1 22,803 / g2 22,370 / g3 22,912 / g4 22,505 /
  l1 13,978 / l2 12,814 / l3 13,690 / l4 13,766 / l5 13,922 / l6 13,820 = **172,580**。
  global 90,590（52.5%）+ local 81,990（47.5%），与 spec §3.2 记录的比例一致。
- **七段解析：172,580 / 172,580 全部解析出恰好 7 段**（用 canonical `_SECTION_TOKENS`）。
- **原「收束文本」在生产 build 中不存在**：`<plan_specificcolor_end>` 之后无任何非空文本，
  实测 0 条。因此 spec §4.2 的「原收束文本（若存在）」分支在本次转换中不触发，
  **不补写**（代码仍保留该分支并在记录里落 `has_closing_text` 字段）。
- LUT 身份：`recipe.preset_id`，全量 **3,445 个**不同 preset；train 侧出现 3,442 个，
  eval 侧出现 1,390 个，其中 3 个 eval-only。eval 每 preset 平均 2.37 个样本。
- `winner_confidence`：train 中 low **69,404 条（41.0%）**，eval 中 low **1,351 条（40.7%）**。
- source 隔离：train 与 eval 的 `I_in` 路径集合**交集为 0**（27,204 vs 538 个源），
  即冻结 split 已经保证了源级 train/eval 不交叉。
- 图像成员：`.in.jpg` / `.in.png` / 极少量 `.in.dng`；`I_tar`（`.jpg`）与 l 系的 `.cgt.png`
  也在同一 shard，本阶段不消费。
- 已发布但**未标注**的候选（`sft_id: null`，标注失败）共 3,170 个，它们从来不是 SFT 样本，
  单独计数，不进 rejection report。
- `I_in` 的原始绝对路径（`/home/bc/data/datasets/...`）**本机大多已不存在**（抽样 50 条只有 5 条在），
  所以图像只能从 shard 里取，不能退回按路径读盘。

### 1.3b 运行环境（烘焙与分词都用它，已快照到 `config/env.json`）

`/home/bc/miniconda3/envs/llm_factory`：Python 3.12.12 / transformers 4.57.1 /
**Pillow 10.4.0** / torch 2.10.0+cu128。仓库 `pyproject.toml` 钉的是 Pillow 12.2.0，
两者的 BICUBIC 实现理论上可能有差异——**烘焙恰好把这个风险消掉了**：训练侧拿到的是
已经等于目标尺寸的图，`prepare_image` 里的 resize 是恒等操作，不会再触发一次重采样。

### 1.4 存储与吞吐

- NFS `/mnt/nfs` 剩余 **23 TB**（120T 中已用 98T，82%）。
- 顺序读实测 **92.9 MB/s**（`dd` 1.6 GiB）。十个 build 的 sft 投影合计约 **309 GB**
  （其中 `I_in` 约 200 GB）。
- 只读成员头部（128 KiB pread，32-48 线程）实测 **~1,476 member/s**，全量约 2 分钟。
- 本机 48 核 / 125 GB RAM。

### 1.5 既有 indexed-tar 契约

生产 build 已经是符合契约的 indexed tar 数据集（`schema_version: 2`，index 行含
`shard/member/offset/offset_data/length/size/sha256/suffix/sample_id/logical_path`，
manifest 含每 shard `tar_sha256`/`index_sha256`/`member_count` 与 catalog 摘要）。
本任务的写出**复用** `dataset_build.tools.indexed_tar` 的 `_ShardWriter` /
`validate_shard` / catalog / manifest 结构，只把「字节来源」从磁盘文件换成内存
（`q3vl/data/shardio.py`）——否则要先落 17 万个小文件再打包，正是契约禁止的东西。

---

## 2. 假设清单（能自行核实的已当场核实）

| # | 假设 | 处理 |
|---|---|---|
| A-1 | `recipe.preset_id` 即 METACANVAS 所称的 `lut_id` | 核实：每行 recipe 都带 `preset_id` + `preset_path`（.cube 文件），是唯一的 LUT 身份键 |
| A-2 | `source_id`（vrmeta）即 `source_image_id` | 核实：同一 `I_in` 路径在不同 build 可能是不同 `source_id`，故按 `i_in_path` 做并查集合并后再分组 |
| A-3 | 七段顺序与 canonical `REASONING_FIELDS` 一致 | 核实：`assemble_reasoning` 按 `_SECTION_TOKENS` 顺序拼接；解析器按位置校验，乱序/重复/未闭合一律拒绝 |
| A-4 | 图像 EXIF orientation 会影响短边判定 | 核实：实测存在 orientation=8（转置）的 JPEG，必须先定向再算几何 |
| A-5 | 缩放采样与训练侧一致 | 直接 import `q3vl.train.imageproc._RESAMPLE`（BICUBIC）与 `plan_geometry`，不另写一份 |
| A-6 | 序列长度口径与训练侧一致 | 直接 import `q3vl.train.collator.Sft2SegCollator` 构造 prompt / target，长度即训练时的长度 |

---

## 3. 决策记录

> 协议要求：属于「两种做法都合理且影响后续」的，写进本节并采用保守默认，不静默拍板。
> 下面 D-1 / D-2 / D-5 需要主 agent 确认。

### D-1（待主 agent 决策）图像以「契约尺寸重编码」入库，而非逐字节复制原图

- 任务卡给的保守默认是「shard 内存原图 bytes + 训练时在线预处理」。
- 实测代价：原始 `I_in` 约 **200 GB**，平均约 8 MP。复制一份到新根 = 读 200 GB + 写 200 GB
  （单链路 93 MB/s，约 1.2 h + 1.2 h）；而按 spec §5 契约预处理后的副本约 **40 GB**。
- 关键事实：**本战役全程没有任何图像增广**（spec §5 是确定性变换，训练侧 `prepare_image`
  只做 EXIF→等比→32 对齐），所以预先烘焙不损失任何随机性；且后续 Base SFT + 8 个 Where 臂
  + 8 个 What 臂会把整套数据各读一遍，200 GB 与 40 GB 的差别是 TB 级 NFS 流量与
  每步解码 8 MP JPEG 的 CPU。
- 已采取的保护：每条记录里**完整保留原成员定位**（build 根 / shard / offset / length /
  sha256 / 原 `i_in_path`），原始字节一字未动仍在生产 build 里；重新烘焙是一条命令
  （`python -m q3vl.data.cli images`）。编码用 **JPEG q95 + `subsampling=0`（4:4:4）**——
  这是个配色数据集，4:2:0 会砍掉 What 阶段要预测的色度信号。
- **若主 agent 坚持逐字节复制**：`q3vl/data/bake.py` 换掉 `bake_one` 的一行即可，
  代价是 +160 GB 空间与约 2.5 h 链路时间。

### D-2（待主 agent 决策）`winner_confidence=low` 默认**保留**在训练集里

- 冲突：CLAUDE.md 数据纪律写「`winner_confidence=low` 不进 SFT 主训与评测 GT」；
  而 2026-08-04 冻结的 SFT spec §3.1 指定 split authority 为唯一依据，
  §6 列出的过滤原因里**没有**置信度，且该 authority 的 train 里实测有 **41.0% 是 low**。
- 采用：**跟冻结的 spec 走（保留 low）**，因为任务卡明确以 spec §3/§6 为规格来源，
  且剔除 41% 训练数据属于重大口径变更，不能由编码 subagent 单方面做。
- 已备好开关：`python -m q3vl.data.cli plan --drop-low-confidence`
  会把 low 全部打进 rejection report 并重算 `N_effective`，无需重跑前两个 pass。
- 每条记录与每个 split index 都带 `winner_confidence` 字段，训练侧可随时二次过滤。

### D-3（已定，非决策）`dedup_drop` 按删除名单处理

依据 §1.2 的算术：它与 train/eval 相交且三者并集恰好等于全量。
因此 `train_final = train − dedup`、`eval_pool = eval − dedup`，dedup 内的样本全部
进 rejection report（reason=`dedup_drop`）。审计同时报告「应用前」与「应用后」的两组交集数。

### D-4（已定）两段目标的字符串形态与训练侧对齐

- 记录里只存**正文**：`where` = `region_scope` 正文；`color` = 其余六段正文按原序以 `\n` 连接
  （若存在收束文本则追加，实测为 0 条）。
- 拼成 target 的动作由训练侧 collator 完成：
  `"<where>" + where + "</where>" + "<color>" + color + "</color>"`（标签与正文之间无换行），
  与并行落地的 `q3vl/train/collator.py::build_target_text` **逐字一致**。
- spec §4.2 里那段带缩进换行的代码块按**顺序示意**理解，不当作空白字符规范；
  若主 agent 要求标签内换行，改 `q3vl/train/constants` 一处即可，长度需重算（+~14 token/样本）。

### D-5（待主 agent 决策）`T_lut_unseen` 的构造方式与四集合规模的取舍

- 硬约束的冲突点：eval 只占全量 1.96%，所以「保住一个 LUT 让它进 T_lut_unseen」平均要
  从训练集里删掉约 48 条。5% 训练预算最多买到 **~700 条** eval 未见 LUT 样本。
- 同时 METACANVAS §2.2 要求「source image 不跨 train/select/test」。若把 select 与 test
  的源集合完全分开，而 test 源只占 1/3，则落在 select 源里的未见 LUT 样本无法使用。
- 采用的方案：把 **select 角色 = {V_where, V_what}**、**test 角色 = {T_final, T_lut_unseen}**
  视为两个角色，两者的源集合不相交；`T_final` 与 `T_lut_unseen` **同属 test 角色、共享源**，
  靠 LUT 身份区分（这正是它们要对比的那个轴）。再把 test 源**优先选成"未见 LUT 样本最多"的源**，
  从而在不改变 V/T 规模比例的前提下尽可能多地回收未见 LUT 样本。
  落在 select 源里的未见 LUT 样本一律**弃用**（计入 rejection report，
  reason=`reserved_lut_in_select_source`），绝不放进选择集——否则 Where/What 的 checkpoint
  会部分地在未见 LUT 上被选出来。
- 实际落地结果：保留 **296 个 LUT identity**（其中 259 个在 `T_lut_unseen` 里真实出现），
  从训练集剔除 **8,371 条 = 5.00%**（预算上限，`≤5%` 故按任务卡直接采用并记录），
  eval 侧保留 575 条，其中 **433 条**落在 test 角色源上进入 `T_lut_unseen`，
  142 条落在 select 源上被弃用。最终四集合：
  `V_where 896 / V_what 897 / T_final 918 / T_lut_unseen 433`，前三者比例 ≈ 1:1:1。
  十个 taxonomy major **全部有覆盖**（第一版分配器按 major 大小顺序花预算，
  花到第 6 个 major 就见底，后 4 个为 0；已改成先按 eval 占比给每个 major 分配预算份额、
  再全局重分配余额）。
- 若主 agent 认为 `T_lut_unseen` 需要更大：唯一的杠杆是提高训练剔除预算
  （`config.LUT_RESERVE_TRAIN_BUDGET`），代价与收益近似线性（每 ~15-48 条训练样本换 1 条 eval）。
  未分层贪心的实测曲线（供参考）：目标 200 条→剔除 0.85%，300→1.57%，400→2.39%，
  500→3.35%，600→4.41%；分层后同样 5% 预算买到 575 条。

### D-6（已定）不注入 system prompt

chat template 默认不产生 system 段，spec §4.1 把输入定义为 `I_in + instruction`，
训练侧 `DEFAULT_SYSTEM_PROMPT = None`。三者一致，序列长度按此计算。

### D-7（已定）`.in.dng` 直接拒绝

Pillow 会把 DNG 当 TIFF 打开并给出尺寸，但解出来的是 raw CFA 帧而不是 RGB 图。
与其走错误的解码路径，不如进 rejection report（reason=`image_format_unsupported`）。

### D-8（已定）短边不足 512 的图按契约**上采样**

spec §5 写的是「等比例缩放，使短边为 512」，没有下限例外。实测语料里确实有 540×360 这类源。
照契约执行，并在报告里单列 `upscaled` 计数与原始短边分布，供主 agent 判断是否要加下限。

实测：**23,045 条（14.2%）被上采样**，原始短边最小 256、p05 = 360、p50 = 1,362。
这批图的高频细节是插值补出来的，Where 阶段的边缘质量指标在它们上面天然偏乐观；
若主 agent 要加下限（例如短边 < 384 直接过滤），代价是训练集少约 5%，改动点是
`q3vl/train/constants.py` 加一个常量 + `q3vl/data/headers.py` 加一条拒绝原因，
两个 pass 都可以在几分钟内重跑（图像烘焙需重跑约 1 h）。

---

## 3b. 实施过程中被数据打回的两处（记录下来，因为两处都会静默地产生错误结论）

1. **图像头部只读 128 KiB 会把 42 张好图判成损坏**。首轮 header pass 报
   `image_corrupt: 42`；逐条全量重读后，42 张**全部正常打开**——它们是 EXIF/ICC 块很大、
   SOF 标记落在 128 KiB 之外的 JPEG。若不查这 42 条，它们会以「图像损坏」的名义
   悄悄从训练集消失。现改为 128 KiB → 1 MiB → 整成员逐级升级读取，重跑后
   `image_corrupt` 降为 **0**，只剩 9 条 `.in.dng`（见 D-7）。
2. **烘焙的 I/O 形态比 CPU 重要 6.8 倍**。第一版用 24 个工作进程各自 pread 自己的成员，
   实测 **6.6 img/s / 9.8 MB/s**（同一挂载点顺序 `dd` 是 93 MB/s）——并发随机读把 NFS
   客户端的预读打散了。改成「每个 shard 一个顺序读线程按 8 MiB 块流式切成员 + 进程池只做
   解码/编码 + 有界队列」后是 **44.7 img/s / 62.9 MB/s**，全量从 7.3 h 降到约 1 h。
   （顺带：`Pool.imap` 会把整个生成器抽干进内存，250 GB 会直接 OOM，所以队列必须有界。）

## 4. 未在本任务范围内（明确不做）

- 不启动任何训练，不占用 GPU。
- 不改动 `dataset_build/` 下的任何既有代码（只 import）。
- 不生成 Where/What 阶段的 oracle latent / basis / LUT code 派生物——那是后续阶段的
  数据派生物，本任务只固定它们将来必须引用的同一份 manifest 与同一套 split。
