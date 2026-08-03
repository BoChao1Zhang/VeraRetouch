# tools/data_splits — S/P split 旁表 + 数据核查三件套（任务卡 T1）

DATA_ASSIGNMENT §1 行动项 C/G 的物化实现。所有实验**只读**本目录旁表，禁止 ad-hoc 切分。
依赖：仅 Python 标准库（本机 python3 = /home/bc/miniconda3，3.13 实测）。

## 产物

| 文件 | 内容 |
|---|---|
| `splits.sqlite3` | 表 `sources(source_id, split, pool)`、`presets(preset_id, major, minor, split, gen)`、`meta(key, value)` |
| `splits_sources.csv` / `splits_presets.csv` | 同内容 CSV 镜像（presets 含尾列 `gen`；按列名或前 4 列位置读取均兼容） |
| `../../experiments/tooling-wave1/data_splits/` | 统计报告、PPR10K 复核、行动项 G 报告、REPORT.md |

## Split 定义（冻结）

- **seed 常量**：`verasplit-v1`（写死在 `vr_common.py::SPLIT_SEED`，改动即换代，必须整表重生成并重命名版本）。
- **S-split（源级）**：`bucket = int(sha1("verasplit-v1:" + source_id).hexdigest()[:8], 16) % 100`；
  `0–89 = train / 90–94 = val / 95–99 = test`。纯函数 ⇒ 同 source 跨 build 恒同 split。
- **P-split（preset 级，gen 0）**：按 taxonomy `minor` 分层；层内 `preset_id` 升序排序，
  层尾 `floor(n*0.05+0.5)` 个 = test，再前同数 = val，其余 train（train 非空保护）。
  `major` 取 `minor.rsplit('_', 1)[0]`（journal 里 candidate.major 是组请求 major，非 preset 归属，见 NOTES.md §一.4）。
- **重跑 = 增量 append（wave-1.5，清 T1-B1）**：`build_splits.py` 重跑时先读现有
  splits.sqlite3，**已存在的 source_id/preset_id 归属冻结不变**（表内条目也永不删除）；
  只对新增条目分配：新 source 走 S 哈希入桶；新 preset 走层内增量规则
  （`vr_common.p_split_increment`：层内升序，只占「增长后目标配额 − 已占名额」的
  val/test 新增名额，尾部 test、再前 val，其余 train），并带递增代号写入 `presets.gen`
  （初版全 0，每次有新增 preset 的重跑 +1）。旧表 `split_seed` 与代码常量不符时硬失败。
- **整表重建必须显式 `--force`**：重建前打印 sources/presets 两张新旧 diff 摘要
  （各 split 迁移对计数 + ADDED/REMOVED），且 gen 归零。默认路径不再有任何无声重写。
- **n<10 小层无 val/test 名额（既定行为，DECISIONS_2026-08-03 D-01 裁定）**：
  `floor(n*0.05+0.5)` 圆整使 n<10 的 minor 层全部落 train（现表 28 层）；
  对账时这些 minor 不参与未见-LUT 评测。这是裁定后的既定行为，不是缺陷。
- **覆盖范围**：journal 归档（`/var/cache/veradata/annot_review/journal-archive/`）内全部完成 build
  的全部 `source_id` 与 `candidates[].preset_id`。「完成 build」判据 = journal 归档存在 groups.jsonl。

## 工具与用法

```bash
P=/home/bc/miniconda3/bin/python3
$P build_splits.py                 # 生成/增量更新旁表 + 统计（默认增量 append，冻结既有归属）
$P build_splits.py --force         # 整表重建（打印新旧 diff 摘要，gen 归零）——需明确意图
$P build_splits.py --limit N       # 小样本冒烟（写 *.sample.*，不碰 sqlite）
$P verify_ppr10k.py                # PPR10K 无污染复核（退出码 0=干净）；--limit N 冒烟
$P action_g_render_audit.py        # 行动项 G：弃权组渲染产物落盘可用性 + D-RENDER 规模数
$P selfcheck.py                    # 判据自检（覆盖率/跨 build 一致性/比例/gen 重放/增量稳定性）；--quick 冒烟
```

## 读表方式（下游实验）

```python
import sqlite3
con = sqlite3.connect("tools/data_splits/splits.sqlite3")
split, pool = con.execute(
    "SELECT split, pool FROM sources WHERE source_id=?", (sid,)).fetchone()
```

## 行动项 G 口径

「可用渲染对」= 候选 after `.jpg` 已落 `datasets/groups/<build>` shards（以 `indexes/*.idx.jsonl` 为准）
**且** I_in 可回取（源图基名 stem 命中 NFS img 银行 `img/unknown/<pool>` 对应 role，
或 mmart_ppr10k / fivek_gold 的本机路径存在）。l 系 build 另计逐候选 `.cgt.png` 掩膜落盘数。

## 注意

- `winner_confidence` 四值：`normal` / `low` / `abstain`（margin<1.0 弃权，winner_ids 空）/
  `null`（未注释/注释终态失败，报告中记 `unannotated`）。
- MMArt-PPR10k 池存在 PPR10K 官方 val 段疑似重叠（原 §4-A 结论未覆盖），详见
  `ppr10k_verify.md` 附节与 NOTES.md「待主 agent 决策」。
