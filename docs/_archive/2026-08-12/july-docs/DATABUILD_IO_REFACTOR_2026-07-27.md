# DataBuild IO 重构实施文档（2026-07-27 定案）

本文是**可独立执行**的实施说明：接手者不需要之前的会话上下文。所有性能数字均为本机实测，
测量方法一并给出，改动前请复测确认环境未变。

---

## 0. 一句话背景

`/home/bc/data/datasets`（1317 GiB / 556 万文件）已全部 WebDataset 化迁入 NFS
（332 组 / 995 GiB / 553 万成员），本机只保留 3 个目录。本文档描述**迁移之后**的
IO 重构：消除冷启动、生产期随机读、落盘写入、SFT 读取四处瓶颈，并把 IAA 吞吐提 2.37 倍。

前置事实（迁移已完成、已验证）：

| 项 | 值 |
|---|---|
| 归档组数 | 332（img 25 / preset 272 / cgt 4 / cache 3 / renders 28） |
| 成员 / 样本 | 5,538,396 / 2,609,271 |
| 原路径反查条目 | 4,451,496 |
| 归档体积 | 995 GiB |
| 全量 verify | **332/332 通过，0 失败**（986 GiB 读回重算 sha256 + catalog↔JSONL 交叉比对） |
| 1% 源侧抽样 | **44,514 / 44,514 一致，0 不符** |
| `/home` 空闲 | 304 G → **1.6 T** |

---

## 1. 存储契约（硬规则，任何改动都不得违反）

### 1.1 生产绝不写 `/home`

`/home` 是 LVM LV 横跨 SSD(前 200 GiB) + HDD(3.6 T) 的线性拼接，ext4 不知道设备边界，
**无法指定某个目录落在 SSD 上**。生产数据一律不写 `/home`。

`/home/bc/data/datasets` 现在只剩三个保留项，**只读、不再增长**：

```
/home/bc/data/datasets/
  ppr10k_360p_dl/                 20 GiB  原始 TIF 压缩包（与解压树重命名后的 PNG 名字对不上，无副本）
  recipes/                        10 GiB  preset 本体 + 未入 bank 的孤儿 preset（9,183 个）
  vera_directionA_1M/
    preset_bank_full/              3 GiB  聚合大文件（luts.npz / features.jsonl / taxonomy.jsonl / lab.npz …）
```

`/home/bc/data/models/` 未迁移，原地保留（用户决定：只迁数据，不迁模型）。
仓库内 `gpu_render/fits/baked/`（581 个烘焙 .cube）未归档也未删，保持现状。

### 1.2 四层存储及各自职责

| 层 | 路径 | 用途 | 实测带宽 | 规则 |
|---|---|---|---|---|
| **tmpfs** | `/mnt/ramstage`（待建，24 G）| 生产中间小文件、预取缓冲 | 内存速度 | 掉电即失；只放当前 1–2 轮 |
| **SSD 缓存** | `/var/cache/veradata`（80 G 预算，现用 24 G）| 模型权重、全局索引、preset bank | ~500 MB/s | 全部可重新生成 |
| **NFS 归档** | `/mnt/nfs/bc/data/datasets` | tar 归档（**唯一权威**） | **98 MB/s** | 不可变；只由 `land()` 追加 |
| **NFS builds** | `/mnt/nfs/bc/data/builds` | jsonl / manifest（少量大文件，原样存） | 98 MB/s | 可人读、可 append |

**NFS 是 1 GbE，98 MB/s 是硬顶，读写共用且与其他用户共享。没有软件解法**
（物理网卡 `ens5f0` = 1000 Mb/s；`br-*`/`veth-*` 那些 10000 Mb/s 是 docker 虚拟网桥）。
并行写零收益：1/2/4 流聚合 98.4 / 102.1 / 102.9 MiB/s。

### 1.3 SSD 缓存目录

```
/var/cache/veradata/                        # 属主 bc:bc，需 root 建一次
  models/OneAlign/          15.3 GiB        # IAA 权重（HDD 上加载 ~3–4 min，SSD 实测 46.2 s）
  global.sqlite3             5.8 GiB        # 全局索引（纯派生）
  preset_bank_full/          3.0 GiB        # luts.npz 等聚合大文件，渲染链从此不碰 /home
```

建立方式（已完成，重装机器时重做）：

```bash
sudo mkdir -p /var/cache/veradata/models && sudo chown -R bc:bc /var/cache/veradata
```

**不要做成 80 G 硬上限的 loop 镜像**：内容都是固定尺寸（峰值 = 索引重建时的 2×），
不存在无界增长，多一层 loop 只多一个 fsck 隐患。

注意 `OneAlign/corrupt_downloads/` 是下载失败的稀疏残片（15.29 GiB 名义 / 2.9 GiB 实占），
SSD 副本里已剔除；`/home/bc/data/models/OneAlign/corrupt_downloads/` 那份仍在，可删。
拷贝模型目录时用 `rsync -aS`（带 `--sparse`），否则空洞会被展开成真实零块。

### 1.4 全局 SQLite 索引

**这是纯派生物，不是数据库。权威永远是 tar 旁边的 JSONL 索引。**

| 项 | 值 |
|---|---|
| 默认路径 | `/var/cache/veradata/global.sqlite3`（`archive_reader.DEFAULT_DB`） |
| 环境变量覆盖 | `VERADATA_CATALOG`（`archive_reader.CATALOG_ENV`） |
| 解析时机 | **运行时**（`archive_reader.default_db()`）。不要写进函数默认参数——那会在 import 时冻结，既坑测试也坑运维 |
| 打开方式 | `?immutable=1` + `PRAGMA mmap_size`。**这条不是可选项**：`mode=ro` 实测 24,923 µs/次（40 次/秒），`immutable=1` 实测 15.3 µs/次（66,975 次/秒），**差 1600 倍** |
| immutable 的正当性 | 重建写 `<name>.rebuilding` 再 `os.replace` 原子替换；持旧 fd 的读者只会看到旧快照，永不撕裂。要看新数据重开 reader |
| 重建命令 | `python -m dataset_build.tools.global_catalog`（扫全部 `manifest.json` + `indexes/*.idx.jsonl` + `metadata.jsonl`） |
| 表 | `groups` / `members` / `samples` / `source_paths` |

`source_paths` 是**反查表**：原始绝对路径 → (组, 成员, 偏移)。它让现有代码里遍地的硬编码
绝对路径**语义不变**地继续工作：

```python
from dataset_build.tools.archive_reader import read_bytes, path_exists, iter_source_paths
read_bytes(path)            # 本机文件在 → 读本机；已删 → os.pread 从 tar 取
path_exists(path)           # 存在性门控穿透归档
iter_source_paths(prefix)   # 归档版 scandir，替代 os.listdir 做发现
```

**前缀查询必须用范围比较，不能用 `LIKE`**：`LIKE` 默认大小写不敏感，SQLite 无法用索引，
实测在 445 万行上是全表扫 **963 ms**；范围比较走主键索引 **8.3 ms**（快 116 倍）。

### 1.5 配置里那些"看起来坏了"的路径

`databuild.*.toml` 的 `sources.subject_cache` 仍写着
`/home/bc/data/datasets/vera_directionA_1M/subject_cache`——**这是故意的**。
本机那个目录已删，该值现在是**归档反查表的前缀键（逻辑标识）**，
`construct` 用它调 `iter_source_paths()` 枚举 cache 条目、用 `read_bytes()` 取字节。
**不要"修正"成 NFS 路径，否则源池为空。** 5 个 toml 里都写了这条注释。

已改好的配置：

| 文件 | 键 | 值 |
|---|---|---|
| `databuild.*.toml` ×5 | `output_root` | `/dev/shm/veradata/staging/<build_id>` → **待改为 `/mnt/ramstage/<build_id>`**（见第 3 步） |
| 同上 | `presets.bank_dir` / `taxonomy` | `/var/cache/veradata/preset_bank_full[/taxonomy.jsonl]` |
| 同上 | `sources.subject_cache` | 保持原值（逻辑键） |
| `source_qa/config.py` | `OUT_ROOT` | `/mnt/nfs/bc/data/builds`（`VERA_OUT_ROOT` 可覆盖） |

环境变量清单：

```
VERADATA_CATALOG        全局 sqlite 路径     默认 /var/cache/veradata/global.sqlite3
VERA_ONEALIGN_MODEL     OneAlign 权重目录    默认 /var/cache/veradata/models/OneAlign
VERA_QALIGN_REPO        Q-Align 仓库         默认 /home/bc/code/iaa_models/Q-Align
VERA_OUT_ROOT           source_qa 输出根     默认 /mnt/nfs/bc/data/builds
RENDER_LUT_PACK_DIR     LUT 预解析包目录     默认值是坏的，见第 0 步
```

---

## 2. 实测基线（全部本机测得，含测量方法）

### 2.1 设备与链路

```bash
lsblk -d -o NAME,ROTA,SIZE,MODEL          # sda=SSD 447G(ROTA 0), sdb=HDD 3.6T(ROTA 1)
cat /sys/block/sdb/queue/scheduler        # [mq-deadline] → ionice 完全无效
cat /sys/class/net/ens5f0/speed           # 1000 Mb/s
```

| 通路 | 实测 |
|---|---|
| HDD 顺序读（O_DIRECT 单流） | 74.1 MB/s |
| NFS 顺序读（O_DIRECT） | **98.3 MB/s** |
| NFS 顺序写（1/2/4 流） | 98.4 / 102.1 / 102.9 MiB/s |
| `ionice` | **无效**（mq-deadline 只认 CFQ/BFQ） |

> 陷阱：不加 `iflag=direct` 或读刚写过的文件，会因 page cache（96 GB）命中而虚高。
> 迁移期我曾误得 177 MiB/s，真值是 98。

### 2.2 打包读并发（同 corpus 可比组 A/B）

| read_workers | 实测 |
|---|---|
| 4 | ~21 MiB/s |
| **16** | **37.8 MiB/s** ← `DEFAULT_READ_WORKERS` |
| 32 | 33.1 MiB/s |

小文件负载队列深度越大越好（盘能重排大量在飞的小读）；"机械盘要低并发"只对顺序大文件成立。

### 2.3 冷启动构成

| 项 | 实测 | 测量方法 |
|---|---|---|
| **`build_inventory`** | **8.0 min** | 56,777 条 × 135 ms 中位 ÷ 16 并发；单条抽样 24 个 |
| OneAlign 冷加载 | **46.2 s** | 其中 `Loading checkpoint shards` 36.5 s |
| LUT 预解析包 | 3.22 s / **2677 MiB 常驻** | 4051 个网格，峰值 RSS 2.65 GiB |
| 源池发现（改后） | 77.7 ms | `iter_source_paths` 范围查询，56,777 条 |
| 索引点查 | 14.9 µs | `ArchiveReader.locate()`，5000 次随机 |

`build_inventory` 8 分钟里做的事**大部分冗余**：`_inspect_cache_dir` 对每条读+解码
`subject.json` + `subject.png` + **源图**做完整性门控，而归档每个成员的 sha256 已在
332/332 verify 里逐字节验过、宽高与 size 已在索引列里。真正只能靠解码得到的只有
`mask_area`（subject.png 的 α 面积，须落在 0.005–0.85）。门控结果抽样：
37.5% 在 `subject_not_ready` 短路（7 ms），62.5% 付全额解码。

### 2.4 生产期读取

| 模式 | 实测 |
|---|---|
| 随机读单成员（冷、散落、走 NFS） | **51.2 ms / 20.4 MiB/s** |
| 顺序读（NFS 天花板） | 98 MB/s |

全量一遍 22,444 个源 × 约 2–3 MB ≈ 67 GiB：随机 **56 min**，顺序 **11.4 min**。

### 2.5 IAA 推理（2× H100 96 GB，测时 GPU 空闲）

| 模式 | 耗时 | 吞吐 | 加速 | 与逐张最大偏差 |
|---|---|---|---|---|
| 逐张 ×8 | 856.1 ms | 9.35 张/s | 1.00× | — |
| 逐张复现 | — | — | — | **0.0000**（完全确定性） |
| batch=4 ×2 | 422.8 ms | 18.92 张/s | 2.02× | **0.0000（逐位一致）** |
| **batch=8** | 361.8 ms | 22.11 张/s | **2.37×** | 0.0977，平均偏移 −0.0153 |

**`CLAUDE.md` 里"IAA batch 压分 bug"的记载已被证伪。** 代码层面也读得通：
`QAlignAestheticScorer.forward` 用 `self.input_ids.repeat(B,1)` → 所有序列等长、零 padding，
`[:, -1]` 取的是每行真实末位；`prepare_inputs_labels_for_multimodal` 是逐行拼接。
batch=8 的 ±0.1 分是 fp16 累加噪声，一正两负，非系统性。

换算（8 候选/组）：单实例 4,205 → **9,949 组/h**；4 实例 16.8k → **39.8k 组/h**；
50k build 的 IAA 段 3.0 h → **1.26 h**。

> **副作用**：IAA 提速后 NFS 占空比从 10% 升到 **34%**，所以预取双缓冲（第 3 步）
> 从"锦上添花"变成"必须做"。

### 2.6 资产清点（`global.sqlite3` 查询所得）

**source img — 219,853 样本 / 647,353 成员 / 25 组**，布局 `img/<scene>/<corpus>/`

按 corpus：tad66k 67,126(9.9G)、para 31,220(10.1G)、unsplash 25,000(74.4G)、
unsplash_work 24,997(15.4G)、fivek_gold 20,000(95.7G)、awards 16,849(34.5G)、
ppr10k 8,875(10.6G)、raise6k 5,999(108.2G)、quandian 5,634(18.1G)、fivek_raw 5,000(45.6G)、
korean 4,455(13.8G)、artedit_bench 2,806(1.0G)、artimuse 1,002(0.3G)、
fivek_tiff16 500(29.5G)、greysky 390(9.2G)

按 scene：unknown 152,728(466.3G)、landscape 25,964、still_life 14,938、portrait 6,537、
night 6,011、architecture 5,166、street 3,830、any 2,258、food 1,241、product 1,155、wedding 25

多成员样本形态：`ppr10k` 10 成员/样本（source+target_a/b/c × {png,xmp} + mask_360p/full）、
`fivek_gold` 5–8 成员（before/processed/meta/param_verify/config.lua/prompt×3，
其中只有 16,372/20,000 有 `processed.jpg`，3,654 个是 `param_verify_error.txt`）、
`raise6k` 2 成员（`.raw.nef` + `.preview.jpg`）

**preset — 7,778 样本 / 16,137 成员 / 272 组**，布局 `preset/<fmt>/<major>/<minor>/`
cube 4,000(5.3G) / xmp 1,966(2.4G) / lrtemplate 1,761(1.4G) / 3dl 51(0.02G)。
10 大类 / 120 小类，950 条无 taxonomy 的落 `_unclassified`。每样本 = 本体 +
`.baked.cube`（581 个有）+ `.vrmeta.json`（axes/metrics/coherence/pack_id/major/minor）。

**派生**：renders 1,522,371(456.6G) / cgt 562,037(33.3G) / cache 297,232(10.4G)

---

## 3. 七条决策（2026-07-27 逐题确认）

1. 冷启动**先加速单次加载**（safetensors mmap + LUT .npy mmap + preflight 可关），不做常驻服务化
2. **inventory 落盘时预计算，退化为 SQL 查询**
3. 输入侧**按 (shard, offset) 预取到 tmpfs + 双缓冲**，不重打包源图 tar
4. SFT 归档**只进 NFS**（35 GiB < 96 GB page cache，SSD 副本只省第一个 epoch）
5. 实测证伪"batch 压分"
6. **IAA batch=8**（一次前向一组）
7. **专用 24G tmpfs**（`nr_inodes=4m`）+ 水位 16G 触发 land + 三段流水线

---

## 4. 实施步骤

每步给出：动机 / 改哪里 / 验收 / 回滚。**按顺序执行**，第 0 步是止血不是优化。

### 第 0 步：修两处现在就会失败的路径 ⚠️

**0a. `iaa.py::_decode_rgb` 不走归档**

```python
# dataset_build/source_qa/iaa.py:15
def _decode_rgb(path: str) -> Image.Image:
    with Image.open(path) as image:          # ← 直读文件系统
        return ImageOps.exif_transpose(image).convert("RGB")
```

`agent.py:161 _preflight_scorer` 打分的是**源图路径**，源图已只在归档里 →
**每次 run 的冷启动预检都会崩**。改为：

```python
import io
from dataset_build.tools.archive_reader import read_bytes

def _decode_rgb(path: str) -> Image.Image:
    with Image.open(io.BytesIO(read_bytes(path))) as image:
        return ImageOps.exif_transpose(image).convert("RGB")
```

`read_bytes` 是本机优先的：候选图在 tmpfs 时零额外开销，源图已删时自动走归档。

**0b. `render_backend._LUT_PACK_DIR` 默认值指向已删路径**

```python
# dataset_build/core/render_backend.py:79
_LUT_PACK_DIR = os.environ.get(
    "RENDER_LUT_PACK_DIR", "/home/bc/data/datasets/vera_directionA_1M/preset_bank_full")
```

该目录已删。`_load_lut_pack()` 会打印一行 warning 然后**静默回退到逐次 `load_cube`**——
代码注释自称"GIL 串行 → GPU 饥饿"。默认值改为 `/var/cache/veradata/preset_bank_full`。

**验收**：`RENDER_LUT_PACK_DIR` 不设时，`_load_lut_pack()` 打印
`LUT 预解析包已载入: 4051 个网格`；跑一次 `local10` 构建，preflight 通过。

**回滚**：两处都是单行默认值，git revert 即可。

---

### 第 1 步：inventory 预计算 + SQL 化（最大冷启动收益，8 min → ~0.1 s）

**动机**：见 §2.3。门控里唯一不可从索引获得的是 `mask_area`。

**改动**

1. `dataset_build/tools/land.py`：`land()` 增加可选的 `enrich` 钩子。打包 `cache/subject`
   批次时，字节**已在 tmpfs 内存里**，顺手解码 `subject.png` 算 `mask_area`，并把
   `status` / `asset_id` / `source_path` / `subject.*` 字段一起写进该样本的
   `.vrmeta.json` 与 `metadata.jsonl`。零额外读盘。

   判据字段建议：`eligible`(bool) / `ineligible_reason`(str) / `mask_area`(float) /
   `scene` / `asset_id` / `source_path` / `subject`(dict)。
   判据与 `_inspect_cache_dir` 现有逻辑逐条对齐，包括
   `subject_not_ready` / `invalid_subject_mask` / `subject_mask_area_guard` /
   `missing_source_image` / `decode_or_integrity_failure`。

2. `dataset_build/tools/global_catalog.py`：`samples.meta` 已是完整 JSON，无需改表；
   如需按 `eligible` 过滤加速，可加一个生成列或在 `rebuild()` 里额外建
   `CREATE INDEX samples_eligible ON samples(json_extract(meta,'$.eligible'))`。

3. `dataset_build/src/construct/sources.py::build_inventory`：本机 cache 树不存在时，
   不再逐条 `_inspect_cache_dir`，改为一次 SQL：

   ```sql
   SELECT sample_id, meta FROM samples
   WHERE "group" = 'cache/subject' AND json_extract(meta,'$.eligible') = 1
   ```

   `_inspect_cache_dir` **保留**（本机树还在时仍走它，且是 land 时预计算的参照实现）。

4. **一次性回填**：现有 `cache/subject` 组（150,323 成员 / 约 2 GiB）是迁移时打的，
   没有预计算字段。写一个回填脚本**顺序流式**读该组的 tar（不要逐个随机 pread），
   算 `mask_area` 后重写该组的 `metadata.jsonl`，再 `global_catalog` 重建。
   顺序读 2 GiB @98 MB/s ≈ 21 s + 解码 5.7 万张 PNG。

**验收**
- `build_inventory` 耗时从 8 min 降到 < 1 s（打印 `counts` 与旧实现逐项一致）
- 新旧两条路径对同一批 500 个 cache 条目给出**完全相同**的 eligible 集合与 `mask_area`
  （±1e-6）——这是必须写的回归测试
- `dataset_build/tests/test_sources_archive_fallback.py` 仍绿

**回滚**：`build_inventory` 里保留 `--legacy-inspect` 开关走旧路径。

---

### 第 2 步：IAA batch=8（最大吞吐收益，2.37×）

**动机**：见 §2.5。

**改动**
- `dataset_build/src/construct/canonical_qa.py`：`rank_candidates` 现在逐张调
  `scorer.score(path)`。改为一次收集该组全部候选（正好 8 张）→ 一次
  `_score_pils(images)`。
- `dataset_build/source_qa/iaa.py::OneAlignRunner`：暴露 `score_paths(paths: list[str])`
  批量接口（内部 `_score_pils` 已支持 list），保留 `score_path` 供单张调用者。
- 注意 `self._lock` 是**每实例**串行的，batch 内并行、实例间并行，不要去掉锁。

**验收**
- 单实例吞吐从 9.35 张/s 升到 ≥ 20 张/s（用 §2.5 的探针脚本复测）
- **第一轮真实生产后**抽 200 组，对比 batch 与逐张的**组内 top1 是否变更**的比例。
  这是真正要关心的指标（±0.1 分只在候选分差 < 0.1 时才可能翻转排序）。
  若 top1 变更率 > 1%，退回 batch=4（实测逐位一致，2.02×）。

**回滚**：batch 大小做成配置项 `CONSTRUCT_IAA_BATCH`，设 1 即回到现状。

---

### 第 3 步：预取双缓冲 + 专用 tmpfs + 后台 land 重叠

**3a. 专用 tmpfs（需一条 sudo）**
**已经执行**
```bash
sudo mkdir -p /mnt/ramstage
sudo mount -t tmpfs -o size=24G,nr_inodes=4m,mode=1777 tmpfs /mnt/ramstage
# 持久化：写入 /etc/fstab
# tmpfs /mnt/ramstage tmpfs size=24G,nr_inodes=4m,mode=1777 0 0
```

`nr_inodes=4m` 是重点——tmpfs 默认 inode 数按内存算，几百万小文件会**先撞 inode 上限**
而不是容量上限。用专用挂载点而不是共享的 `/dev/shm`（63 G，别人也在用）。

改 5 个 `databuild.*.toml` 的 `output_root`：`/dev/shm/veradata/staging/<id>` → `/mnt/ramstage/<id>`。

**3b. 预取**

采样顺序在生产前已知（源池来自 DB，`scene_stratified_order` + `allocate_sources` 确定性）。
新增 `dataset_build/tools/prefetch.py`：

```python
def prefetch(source_paths: list[str], dest: Path, db_path=None) -> dict[str, Path]:
    """按归档物理序（shard, offset_data）批量顺序读到 dest，返回 原路径→本地路径。"""
```

- 用一条 SQL 取 `(shard, offset_data)` 并 `ORDER BY shard, offset_data`
- 每个 shard 只开一次 fd，按 offset 递增 `os.pread`（顺序 → 服务端 readahead 生效）
- 消费侧按**采样顺序**从 tmpfs 读，不再碰 NFS
- 领先一轮做双缓冲：轮 N 渲染时后台预取轮 N+1

预算：一轮 2000 组 ≈ 2000 × 2.5 MB = 5 GB，双缓冲 10 GB。

**3c. 后台 land 重叠**

三段流水线：

```
轮 N-1: land → NFS       (后台, ~5 min)
轮 N  : 渲染 + IAA        (GPU, ~9 min)
轮 N+1: 预取 → tmpfs      (后台, ~2 min)
```

tmpfs 水位 16 G 触发 land（封顶 24 G，留 8 G 余量给并发写的瞬时峰值——
写满瞬间是 ENOSPC 而不是优雅停止）。

**验收**
- 生产期 NFS 读流量趋近 0（`nfsstat -c` 或 `/proc/self/io` 对比）
- 单组端到端时延不含 51 ms 的源图随机读
- 一次 50k build 的墙钟时间 ≈ IAA 段时间（1.26 h），NFS 26 min 完全被掩盖

**回滚**：预取关掉即回到直读归档（`read_bytes` 本来就能工作，只是慢）。

---

### 第 4 步：输出侧双写（SFT tar / 中间产物 tar）

**动机**：SFT = QA winner。整批落完再回头索引 winner 很慢（要重读散落成员）；
winner 在产出瞬间双写，训练读顺序、Viewer 查中间产物也快。
因为 staging 在 tmpfs（内存），双写**不多一次磁盘读**，只多一次 NFS 写（约批次的 18–25%）。

**布局**

```
/mnt/nfs/bc/data/datasets/
  groups/<build_id>/     中间产物：全部 8 候选 + C_GT + 逐候选 QA      ~53 GiB/50k
  sft/<build_id>/        winner：I_in + I_tar + C_GT + .vrmeta.json   ~35 GiB/50k
/mnt/nfs/bc/data/builds/<build_id>/   jsonl / manifest（原样，可 append 可人读）
```

**SFT tar 必须也带 `I_in`**，否则训练每样本要回 466 GiB 的 img 归档随机读一次，瓶颈复活。

**成员顺序 = 生产顺序**。使能改动**已完成**（2026-07-27）：打包器的不变量从
"成员名升序"换成"**同 key 连续 + 成员唯一**"，plan 的行序即 tar 的成员序。
相关常量 `MEMBER_ORDER_PLAN = "plan_order"`。测试见
`test_plan_order_is_authoritative_not_lexicographic` /
`test_non_contiguous_sample_is_rejected` / `test_duplicate_member_in_plan_is_rejected`。

这条顺序同时给 Viewer 带来收益：**同一组的 8 个候选物理相邻 → 打开一个组是一次连续读**，
而不是 8 次分散 pread（冷态 8 × 51 ms ≈ 0.4 s）。

**权威约定**：中间产物 tar 是权威，SFT tar 是**派生、可重新生成**（从中间产物 + 记录）。
两者都带逐成员 sha256，漂移可检测。

**改动**
- `dataset_plan.write_group` 增加 `preserve_order=True`（现在无条件按成员名排序）
- `land.py` 支持一次 staging 发布两个数据集：先两次 `land(..., keep_staging=True)`，
  两次都成功后再删 staging（现在是单次 land 成功即删）
- winner 从 staging 里的 `sft*.jsonl` 读取（land 时该文件已在 tmpfs）

**验收**
- 同一 winner 在两个 tar 里 sha256 一致
- `groups/<id>` 里同组 8 候选的 `offset_data` 连续
- 用 wds 语义流式读 `sft/<id>`，样本数 == SFT 记录数

---

### 第 5 步：冷启动其余项（收益最小，46 s → ~10 s，放最后）

- **权重转 safetensors**：`.bin`（torch pickle，不能 mmap）→ safetensors（mmap、零拷贝、
  按需读页）。一次性转换后校验：同一批图的分数与转换前**逐位一致**。
  36.5 s 的 shard 加载应降到 5–10 s。
- **LUT 包 npz → 单 .npy + mmap**：现在 `np.load(npz)` 后逐键 `np.asarray` 解压，
  3.22 s / **2677 MiB 每进程**。改成一个大 `.npy`（4051 × 33³ × 3 float32）+
  `np.load(mmap_mode='r')` + 一份 `luts_meta.json` 记 offset → **0 s 加载，多进程共享 page cache**。
- **preflight 可关**：`agent.py:161 _preflight_scorer` 无条件跑一次真实前向。
  加 `--skip-preflight`（或 `target_groups < N` 时自动跳过），让 local10 那种迭代构建不付全额。

---

## 5. 风险与未决

1. **batch=8 的 ±0.1 分对组内 top1 的影响未测**。只测了 8 张的分数偏差，没测
   "top1 是否变更"的比例。第 2 步验收里必须补这个测量。
2. **eligible 判据固化在 land 时**。以后改判据（如 `mask_area` 阈值）需要回填：
   顺序重读 `cache/subject` 组 2 GiB，约 1 min，可接受。
3. **NFS 1 GbE 是硬顶**。多机训练会抢同一条链路；要突破只能换网卡/交换机。
4. **SSD 那 200 GiB 收不回来**。`/home` 的 SSD 段是 LV 的前 200 GiB
   （`lvdisplay -m` 确认：LE 0–51199 → `/dev/sda3` PE 0–51199），实占 114 GiB。
   两块 PV `PFree` 都是 0，`pvmove` 无处可去；ext4 不支持在线缩容。
   要腾出必须离线：`e2fsck -f` → `resize2fs` 缩容 → `lvreduce` → `pvmove` → `lvcreate`，
   **顺序错了就是截断**。建议不做——`/` 已有 126 G 空闲、`/dev/shm` 63 G 够用。
5. **`sam3_cache` 那些目录是 root 拥有的**（SAM3 服务在容器里以 root 跑），
   `bc` 无法在里面 unlink。以后新产出会重现同样的删不掉问题——若仍走容器，
   需要在容器里设置 uid 映射或产出后 chown。
6. **3 个 canonical 测试文件在本机 import 不了**（`No module named 'construct'`，
   既存问题，与本次改动无关）。跑测试时要 `--ignore` 掉它们：
   `test_canonical_foundation.py` / `test_canonical_orchestration.py` / `test_canonical_responses.py`。

---

## 6. 附录

### 6.1 现有工具速查

```bash
PY=/home/bc/.venvs/iaa437/bin/python      # construct 全链路用这个（主 conda 的 transformers 起不来）
cd /home/bc/VeraRetouch

# 规划（只读源 + 写 plan.jsonl / metadata.jsonl）
$PY -m dataset_build.tools.dataset_plan presets  --bank <dir> --baked <dir> --out <plans> --meta-staging <tmp>
$PY -m dataset_build.tools.dataset_plan images   --snapshot <assets.csv.gz> --out <plans> --meta-staging <tmp> [--corpus X]
$PY -m dataset_build.tools.dataset_plan derived  --out <plans> --meta-staging <tmp>
$PY -m dataset_build.tools.dataset_plan renders  --build <dir> ... --out <plans> --meta-staging <tmp>
$PY -m dataset_build.tools.dataset_plan renders-report --build <dir> ... --report <txt>

# 打包 / 校验
$PY -m dataset_build.tools.shard_dataset pack --plan <plan.jsonl> --output <dst> --read-workers 16
$PY -m dataset_build.tools.shard_dataset pack --source <dir> --output <dst>      # 目录模式
$PY -m dataset_build.tools.shard_dataset verify --dataset <dst>

# 落盘唯一入口（clean → plan → pack → verify → 退役 staging）
$PY -m dataset_build.tools.land --staging <tmpfs dir> --group renders/<build_id> [--keep-staging]

# 全局索引重建（纯派生）
$PY -m dataset_build.tools.global_catalog [--dataset-root ...] [--db ...]

# 测试（必须 ignore 那 3 个既存坏文件）
$PY -m pytest dataset_build/tests/ -q \
  --ignore=dataset_build/tests/test_canonical_foundation.py \
  --ignore=dataset_build/tests/test_canonical_orchestration.py \
  --ignore=dataset_build/tests/test_canonical_responses.py
# 当前：37 passed, 5 subtests passed
```

### 6.2 归档格式契约

```
<组>/
  manifest.json          终态标记（status=complete 才算发布）
  shards/shard-NNNNN.tar 未压缩 USTAR，2 GiB 上限（sample 不跨 shard）
  indexes/shard-NNNNN.idx.jsonl   逐成员 offset/size/sha256 —— 持久权威
  indexes/catalog.sqlite3         该组的派生加速层
  metadata.jsonl         逐成员清洗元信息（land 后由外部同步进来，供全局索引重建）
  .verified              verify 通过的标记（删源脚本会检查它）
```

- `key_policy = webdataset_basename_v1`：成员名 = 扁平 basename，key = 第一个 `.` 之前，
  同 sample 成员共享 key 且**必须连续**
- 成员名限制：ASCII、≤100 字节（USTAR name 字段）、扩展名小写、每段 ≤32 字符
- 元信息成员后缀是 `.vrmeta.json`（**不是** `.json`——ArtEdit-Bench 自带
  `Eval/evalution_info.json` 会与 `.json` 撞名）
- 分片只在 sample 边界轮转，所以"2 GiB"是目标而非硬上限

### 6.3 关键验证命令

```bash
# 全量不变量：0 个样本跨 shard
$PY -c "
import sqlite3
c=sqlite3.connect('file:/var/cache/veradata/global.sqlite3?immutable=1',uri=True)
print(c.execute('''SELECT COUNT(*) FROM (SELECT \"group\",sample_id FROM members
  GROUP BY \"group\",sample_id HAVING COUNT(DISTINCT shard)>1)''').fetchone())"

# 索引点查延迟（应 ~15 µs；若是 ~25 ms 说明没用 immutable=1）
# 见 §2.3 的测量脚本

# NFS 真实带宽（必须 iflag=direct，否则被 page cache 骗）
dd if=<某个 shard.tar> of=/dev/null bs=1M count=3000 iflag=direct
```

### 6.4 本会话踩过的坑（避免重复）

- 用户 zsh 开了 `noclobber`，脚本重定向覆盖已存在文件必须 `>|`
- zsh **不对未加引号的 `$VAR` 做词分割**（bash 会），拼 `--build a --build b` 这类参数
  必须用 bash 数组
- `pgrep -f <pattern>` / `pkill -f <pattern>` 会匹配到**自己的命令行**——包装脚本里
  用它做等待循环会自锁或自杀（退出码 144）
- `du`（块）与 `du --apparent-size`/`du -sb`（名义）在稀疏文件上差很多；
  `rsync -a` 不带 `-S` 会把空洞展开成真实零块
- `dumpe2fs` 可以按块组统计 ext4 的区域占用（前 1600 组 = SSD 那 200 GiB）
