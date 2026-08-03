# NOTES — tools/data_splits（任务卡 T1）

日期：2026-08-02（初稿）；2026-08-03 续作（l4 归档纳入，旁表全量重生成，判据全过）。

## 〇、2026-08-03 续作记录

- `prod-l4-local17k-20260801` 的 journal 归档出现于 2026-08-02 23:55（初版旁表 23:40 之后）。
  实测归档完整：groups.jsonl 17,000 行、末行 JSON 完整、manifest `status=complete_with_failures`。
- 已按「归档存在 groups.jsonl = 完成 build」判据重跑 `build_splits.py`（7 builds）：
  33,652 源（+3）、3,522 preset（不变）、0 冲突；`selfcheck.py` 全量 7/7 PASS；
  `verify_ppr10k.py` 重跑（7 builds）结论不变（1–8871，≥8875 零命中，exit 0）。
- 行动项 G 报告（23:58 版）已含 7 builds，本次独立抽验证实（见 REPORT.md §6），未重跑。

## 一、实施前已核实的事实（本机实测，非检索）

1. **完成 build 的判定**：journal 归档目录 `/var/cache/veradata/annot_review/journal-archive/` 下现共 7 个 build：
   `prod-g1/g2/g3-global25k`、`prod-l1/l2/l3/l4-local17k`（l4 于 08-02 23:55 归档，08-03 续作时已纳入）。
   `/mnt/nfs/bc/data/datasets/sft/` 下另有 g4、l5、l6，但无 journal 归档（在途），不纳入。以 journal 归档存在为「完成 build」权威判据。
2. **groups.jsonl 结构**（逐行 JSON，g1 实测 25,000 行）：顶层含 `source_id`、`source_path`、`winner_confidence`、`winner_ids`、`winner_margin`、`candidates[8]`；候选含 `candidate_id`、`preset_id`、`major`、`minor`、`qa`、`after_path`；l 系候选另有 `cgt_path`、`mask_id`、`region`、`amount`、`geometry`。每组恒 8 候选（g1 前 8000 行实测无例外）。
3. **winner_confidence 四值**（g1 前 8000 行实测）：`normal`（有 winner）、`low`（有 winner）、`abstain`（winner_ids 空，即 margin<1.0 弃权）、`null`（winner_ids 空，为注释终态失败/未注释组，与 abstain 语义不同，行动项 G 分开计数）。
4. **candidate.major ≠ preset 官方 taxonomy**：g1 抽样 64,000 候选中 90.2% 的 `minor` 前缀 ≠ `candidate.major` → `candidate.major` 是组的请求 major，不是 preset 归属。`minor` 与 `preset_id` 绑定稳定（抽样 0 冲突，3,522 个 preset 全出现）。**preset 的 major 取 `minor.rsplit('_',1)[0]`**；全量扫描仍统计 preset_id→minor 冲突数（应为 0，非 0 则报告）。
5. **落盘布局**：`/mnt/nfs/bc/data/datasets/groups/<build>/batch-*/{shards/*.tar, indexes/*.idx.jsonl, metadata.jsonl}`。groups 数据集含**全部候选**（含 abstain / null 组）的 after `.jpg` + `.vrmeta.json`；l 系另含逐候选 `.cgt.png`（l1 batch-0000 shard-00000 实测 4529×3 后缀成对）。`sft/<build>/` 只含 winner 行（`.in.jpg`+after+meta）。vrmeta 内含 `group_id/source_id/preset_id/winner_confidence` 可回链。
6. **源图可用性**：journal 里 `source_path` 指向 build 机本地路径（`/home/bc/data/datasets/...`），本机已不存在（quandian 实测 False）；但 **NFS img 银行** `/mnt/nfs/bc/data/datasets/img/unknown/<pool>/`（awards/quandian/korean/unsplash/ppr10k/raise6k/greysky/fivek_gold 等）按原始文件名收录源图（quandian_002481 实测在银行内，role=primary）。MMArt-PPR10k 源在 `/home/bc/datasets/MMArt-PPR10k/global/<gid>_<pid>/before.jpg`，本机实测存在。
7. **PPR10K 官方 split（在线核实，2026-08-02）**：
   - GitHub `csjliang/PPR10K` README 原文：*"train with the first 8,875 files and validate with the last 2286 files"*（按文件顺序切，前 8,875 训练）。
   - arXiv 2105.09180 摘要：总量 *1,681 groups / 11,161 photos*。
   - 与任务卡「已知结论：索引 1–8871，≥8875 零命中」自洽（1-based 前 8,875 个文件为 train）。

## 二、设计决定（含依据）

- **S-split 哈希**：`bucket = int(sha1("verasplit-v1:" + source_id).hexdigest()[:8], 16) % 100`；0–89 train / 90–94 val / 95–99 test。seed 常量 `verasplit-v1` 写死在 build_splits.py 与 README。纯函数 ⇒ 跨 build 恒同 split（判据的抽查 100 例在 selfcheck 中用独立重扫验证）。
- **P-split**：按 minor 分层，层内 `preset_id` 升序（preset_id 为 `rcp_<hex>` 哈希形，字典序≈随机序）；`n_test = n_val = floor(n*0.05 + 0.5)`（四舍五入，n=10 → 1），层尾 n_test 个为 test、再前 n_val 个为 val，其余 train；保护条件：n_val+n_test ≥ n 时逐级递减保 train 非空。
- **pool 归类规则**（source_path 前缀）：`presets_sources/<p>` → p；`_scratch/unsplash` → unsplash；`ppr10k/source` → ppr10k；`RAISE-6k` → raise6k；`fivek_gold` → fivek_gold；`MMArt-PPR10k` → mmart_ppr10k；其余 → other（计数报告）。
- **行动项 G 的「可用」定义**：候选 after `.jpg` 已落 groups 数据集（以 idx.jsonl 为准）**且** I_in 可回取（源图基名命中 img 银行 primary 成员，或本机路径存在）。l 系另报 `.cgt.png` 掩膜齐备数。
- 库函数只用 Python 标准库（hashlib/sqlite3/json/csv/tarfile），本机 python3.13 实测可用；未引入新依赖，无 pip 安装。

## 三、待主 agent 决策（保守默认已注明）

1. **MMArt-PPR10k 潜在污染（新发现，原「PPR10K 无污染」结论未覆盖）**：完成 build 的源里有 `mmart_ppr10k` 池（MMArt-PPR10k 基于 PPR10K 原图构建，目录名 `<groupid>_<photoid>`，实测 group id 上至 1679；PPR10K 官方 val 为「最后 2,286 个文件」，文件按 group 排序 ⇒ 高 group id 段几乎必然落官方 val）。verify_ppr10k.py 已把 MMArt group id 分布与疑似 val 段命中数写进报告（paper 口径 train=1,356 groups ⇒ 边界 gid≥1356，该 group 级边界数字来自论文记忆，未从原文逐字核到，报告中已标注）。**保守默认**：split 表照常物化（不因此改动），污染判定与处置（如把 mmart_ppr10k 源强制归入 S-train 或从 E20 口径剔除）留主 agent 拍板。
2. **winner_confidence=null 组的归属**：行动项 G 报告中与 abstain 分开列（默认都计入 D-RENDER 可用池，符合「渲染确定性与置信度无关」）。若主 agent 认为 null 组应排除 D-RENDER，扣除对应行即可（报告给出分项数）。
3. **P-split 圆整规则**：上述 floor(n*0.05+0.5) 使极小层（n<10）无 val/test；若要求「每层至少 1」需改规则并重生成表。默认按现规则。
4. **fivek_gold / mmart_ppr10k 等零星 pool**（g1 中各 ~几十源）同样进 S-split 表（覆盖性判据要求全部 source_id）；是否允许其进训练由各实验的数据分配行决定，split 表不做排除。

## 四、自检清单（selfcheck.py 实测结果见 experiments/tooling-wave1/data_splits/REPORT.md）

- [x] split 表覆盖率：journal 重扫的 source_id/preset_id 100% 在表内（33,652 源 / 3,522 preset，missing=0）
- [x] 跨 build 一致性：抽 100 个多 build 共现 source_id（池 30,768），逐 build 独立重算 split 全一致
- [x] S-split 比例 89.8/5.1/5.1；P-split 层重推导 mismatches=0（3,172/175/175）
- [x] PPR10K：复核已知结论成立（4,311 源，索引 1–8871、≥8875 零命中，exit 0）
- [x] 行动项 G：journal 1,144,000 候选 100% 落盘（物理抽验 tar/JPEG/vrmeta 通过）；
      渲染失败 ~74k 事件均发生在进组前，与落盘数自洽；I_in 银行命中 100%（逐池抽样 27/27）

## 五、wave-1.5 修复记录（2026-08-03，清 REVIEW-impl-wave1 T1-B1 / 任务卡 F1）

- **问题**：① P-split 层内位置法在 preset 集合变化时会平移既有归属（新 preset 插层 ⇒
  同层旧 preset 排位后移，train/val/test 可能翻转）；② `build_splits.py` 重跑
  `os.remove` 无声整表重写，无任何 diff 门。
- **修复**（改动最小化，只动本模块三个文件 + 文档）：
  1. `build_splits.py` 默认改为**增量 append**：存在 splits.sqlite3 时先载入旧表，
     既有 source_id/preset_id 归属**冻结原样保留**（表内条目永不删除）；新增 source 按
     S 哈希入桶；新增 preset 走新函数 `vr_common.p_split_increment`（层内升序、只占
     「增长后目标配额 − 已占名额」的 val/test 新增名额，尾 test / 再前 val / 其余 train；
     `layer_existing={}` 时与 `p_split_layer` 逐 id 等价——性质测试 n=1..200 验证），
     带递增 `gen` 写入新加的 `presets.gen` 列（旧表自动 `ALTER TABLE` 迁移，存量 gen=0）。
     旧表 `split_seed` 与代码常量不符时硬失败（rc=2）。
  2. 整表重建需显式 `--force`：重建前打印 sources/presets 两张新旧 diff 摘要
     （各 split 迁移对计数 + ADDED/REMOVED），gen 归零；默认路径不再删表。
  3. `selfcheck.py`：3b 由「整层重推导」改为 **gen 重放**（gen0 全量规则 + 逐 gen 增量
     重放 == 存储值——该不变式在未来任意次增量 append 后仍恒成立）；新增第 8 检查
     「稳定性守卫」：向真实表模拟加入 10 个新 preset（aaa/zzz 前缀各半、覆盖层内排序
     两端，落在最大的 10 个 minor 层），经 `build_splits.merge_presets` 全路径合并后，
     既有 3,522 条归属移动数必须为 0 且 10 条新 id 全部获得分配。
  4. `splits_presets.csv` 尾部追加 `gen` 列（按列名或前 4 列位置读取均不受影响）。
  5. Pyright：模块修复前后均 0 错误（无需修）。
- **自检实测**（2026-08-03）：
  - 性质测试（scratchpad）：增量=全量等价 n=1..200；300 组随机增长调度下冻结/配额
    上界/train 非空全过；merge 两阶段旧行含 gen 逐字节相同。
  - 沙箱 E2E（假 journal）：fresh → 增量 append（+80 源 +12 preset，340 条旧归属零变化，
    新 preset gen=1）→ `--force` diff 摘要打印 → seed 篡改 rc=2 硬失败，全过。
  - 真实表增量重跑（journal 仍为 7 builds）：`+0 sources +0 presets`，sources 33,652 /
    presets 3,522 首 4 列与重跑前 sqlite **逐行集合相等**（gen 迁移列全 0，meta 增
    `mode_last_run`），CSV/统计数字与 REVIEW 记录一致（30,229/1,723/1,700；3,172/175/175）。
  - `selfcheck.py` 全量 **8/8 PASS**（含 gen 重放与新稳定性守卫；模拟 10 新 preset
    moved=0、new_assigned=10/10，均落 train——最大层现规模下配额已满，符合增量规则）。
- **D-01 落档**：n<10 小层无 val/test 名额（现表 28 层）为 DECISIONS_2026-08-03 D-01
  裁定后的**既定行为**，已写入 README「Split 定义」节。
- **遗留**：g4/l5/l6 归档续入时直接默认重跑即可（守卫已就位）；`--force` 属换代操作，
  须主 agent 拍板并全线换表名/版本。
