# NOTES · export_agent_loop_review.py

假设与保守默认，供人类复核（不含结论）。

## A8（2026-08-19，左目录 + 右详情重排）

- **链的定义**：一条链 = 一个 global branch × 它的一个 local leaf。同一 global 有 N 个 leaf 就
  出 N 个三联块，source 与 global_after 在这 N 块里重复出现（缩略图按 sha256 去重，磁盘不重复）。
  global 无 leaf 时出一条 leaf=None 的链，final_after 位显示 `not rendered`。
- **排序**：committed 链在前（按 global 的 created_at、leaf 的 repair_count/row_index/branch_id），
  其余（cap_excluded / 未 commit / render_rejected）收进 `details.rest` 折叠区。链号 `#k` 按
  committed 段 + 其余段连续编号。
- **badge 口径**：`commit_status` 缺失（source 在 commit 选择前 error/reject）显示 `not committed`；
  只有 leaf 本身不存在才显示 `no leaf`。
- **applied_alpha 图不再上页**。任务卡指定三联只放 source|global_after|final_after，
  applied_alpha 的三个覆盖率数字仍在页尾 `details` 的 per-chain 参数表里（mean / subject_high /
  subject_support）。如需重新上图需改 `_chain_block`。
- **pending source 无 source 图**：`manifest_json` 在 source 跑完前为 NULL，`source_artifact`
  只在 tree manifest 里，DB 的 `source_sha256` 不等于 source 图 blob 的 sha256，
  所以在跑的 source（本次 500 中 127 个）三联首格为 `not rendered`。没有从 source-manifest jsonl
  另开输入去补，因为任务卡要求 CLI 不变。
- **缩略图短边 512**（`--thumb-short-edge` 默认由 384 改 512）。全量导出目录 271MB，未触发
  ≥2GB 降到 448 的条款。文件名改为 `<sha16>-<short_edge>.<ext>`，并在写完 index 后 prune 掉
  本次未被引用的文件——否则改 short edge 后旧 384px 文件同名命中 `is_file()` 会被静默复用。
- **确定性**：同 DB 状态两次导出字节一致；本次两跑的差异仅出现在 summary 的
  campaign 级 status 计数与 token usage 行，来源是 run500 正在写库。
- **页面为单文件**：500 个 source 的 pane 全在一个 index.html（11.3MB），非当前 pane 用 `hidden`
  关闭 + `loading=lazy`，隐藏 pane 的图不发请求。没有拆成每 source 一页，因为任务卡要求页面路径不变。
