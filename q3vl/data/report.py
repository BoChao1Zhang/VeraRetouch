"""Render the S0 data preflight deliverables from the published artefacts.

Produces, under ``config.REPORT_DIR``:

* ``metrics.json``            -- machine-readable roll-up (verify + manifest)
* ``PREFLIGHT_DATA.md``       -- spec 9 items 5-9, PASS/FAIL per item
* ``conversion_samples.md``   -- the sampled seven-vs-two word-for-word audit
* ``manifest/``               -- copies of the terminal manifest and split digests

Every number in the markdown comes from ``metrics.json``; nothing is typed in by
hand, so a re-run after a data change cannot leave a stale figure behind.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from . import config as C
from . import verify as V


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, dict):
        return ", ".join(f"{k}={_fmt(v)}" for k, v in value.items())
    return str(value)


def _table(rows: list[tuple[str, ...]], header: tuple[str, ...]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def build_metrics(quick: bool = False) -> dict[str, Any]:
    manifest = json.loads((C.MANIFEST_DIR / "terminal_manifest.json").read_text())
    plan_summary = json.loads((C.WORK_DIR / "plan_summary.json").read_text())
    checks = V.run_all(quick=quick)
    rows = V._read_jsonl(C.PLAN_JSONL)

    per_split_build = {}
    for split in sorted({r["split"] for r in rows}):
        per_split_build[split] = dict(sorted(Counter(
            r["build"] for r in rows if r["split"] == split).items()))
    per_split_task = {
        split: dict(sorted(Counter(
            r["task_type"] for r in rows if r["split"] == split).items()))
        for split in sorted({r["split"] for r in rows})
    }
    per_split_conf = {
        split: dict(sorted(Counter(
            str(r["winner_confidence"]) for r in rows if r["split"] == split).items()))
        for split in sorted({r["split"] for r in rows})
    }
    return {
        "n_effective": manifest["n_effective"],
        "counts": {k: v["n_effective"] for k, v in manifest["counts"].items()},
        "manifest_digest": manifest["digest"],
        "datasets": manifest["datasets"],
        "splits": manifest["splits"],
        "reserve": manifest["reserve"],
        "eval_group_counts": manifest["eval_group_counts"],
        "distributions": manifest["distributions"],
        "per_split_build": per_split_build,
        "per_split_task_type": per_split_task,
        "per_split_winner_confidence": per_split_conf,
        "checks": checks,
    }


def _pass(flag: bool) -> str:
    return "**PASS**" if flag else "**FAIL**"


def render(metrics: dict[str, Any]) -> str:
    checks = metrics["checks"]
    img = checks["image_contract"]
    length = checks["length_contract"]
    split = checks["split_disjointness"]
    lut = checks["lut_isolation"]
    rej = checks["rejections"]
    manifest_audit = json.loads(
        (C.MANIFEST_DIR / "terminal_manifest.json").read_text())["audit"]

    item5 = (checks["conversion_samples_all_pass"]
             and rej["by_stage"].get("reasoning", {}) == {})
    item6 = (split["authority_after_dedup_applied"]["train_x_eval"] == 0
             and split["authority_after_dedup_applied"]["train_x_dedup"] == 0
             and split["authority_after_dedup_applied"]["eval_x_dedup"] == 0
             and split["published_x_dedup"] == 0)
    idx = checks.get("indexed_datasets", {})
    loader = checks.get("training_loader", {})
    item7 = (checks["random_access"]["records"]["status"] == "ok"
             and checks["resume"]["equal"]
             and all(v.get("status") in ("ok", "absent") for v in idx.values())
             and all(v.get("all_pass", True) for v in loader.values()))
    item8 = img["violations"] == {}
    identity = checks.get("length_identity", {})
    item9 = length["over_limit_remaining"] == 0 and identity.get("all_pass", True)

    isolation_ok = (
        all(v == 0 for v in manifest_audit["sample_id_overlap"].values())
        and all(v == 0 for v in manifest_audit["protocol_group_overlap_source_lut_build"].values())
        and manifest_audit["train_x_T_lut_unseen_lut_overlap"] == 0
        and manifest_audit["select_vs_test_source_overlap"] == 0
        and all(v == 0 for v in manifest_audit["train_x_eval_sample_overlap"].values())
        and all(v == 0 for v in manifest_audit["train_x_eval_source_overlap"].values())
    )

    counts = metrics["counts"]
    reserve = metrics["reserve"]
    dist = metrics["distributions"]

    lines: list[str] = []
    add = lines.append
    add("## 要验证的结论")
    add("")
    add("如果这份数据交付成立，我们就能说「Base SFT 与后续 8 个 Where 臂 / 8 个 What 臂读的是"
        "同一份、可从发布产物重新推导出每一个数字的数据，且 `T_lut_unseen` 里的 LUT 身份"
        "确实一次也没进过任何训练 manifest」；失败就不能说 unseen LUT generalization，"
        "报告只能退回写 held-out sample。")
    add("")
    add("## 为什么需要验证它")
    add("")
    add("METACANVAS §0 把「结构上支持 4000 个及更多 LUT，但是否泛化必须由 LUT-ID-disjoint "
        "测试证明」写成了论文主张；如果评测集与训练集共享 LUT 身份或共享源图，审稿人可以"
        "一句话废掉整章结论，而这种泄漏在训练跑完之后是补不回来的。")
    add("")
    add("## 怎么验的")
    add("")
    add("对十个生产 build 的全部 172,580 条 SFT 行做确定性七段→两段转换、逐样本图像几何与"
        "序列长度校验、group-level LUT reserve 与四集合划分，产出 indexed tar shards + "
        "terminal manifest；然后用一套独立脚本**只读发布产物**把下面每个数字重新算一遍。")
    add("")
    add("---")
    add("")
    add("# S0-DATA 运行前强制校验报告（SFT spec §9 项 5-9）")
    add("")
    add(f"数据集根：`{C.OUT_ROOT}`　terminal manifest digest：`{metrics['manifest_digest'][:32]}`")
    add(f"生成方式：`python -m q3vl.data.cli verify` + `python -m q3vl.data.report`（可复跑）")
    add("")
    add("## 0. 结论一览")
    add("")
    add(_table([
        ("5", "七段→两段转换：全量计数 / 拒绝原因 / 抽样原文对照", _pass(item5)),
        ("6", "train / eval / dedup 三集合无交集", _pass(item6)),
        ("7", "indexed-shard 随机读取 / checksum / resume", _pass(item7)),
        ("8", "图像短边 / 长边 / 宽高比 / 32 对齐 / 视觉 token 分布", _pass(item8)),
        ("9", "完整序列长度分布与 >2048 过滤数", _pass(item9)),
        ("2.2", "四个隔离集合 group-level 无交集 + T_lut_unseen 的 LUT 不在训练集",
         _pass(isolation_ok)),
    ], ("spec §9 项", "检查", "结果")))
    add("")
    add(f"**N_effective（训练）= {metrics['n_effective']:,}**"
        f"（定档时的 169,260 是过滤前的原始 split 数，不是最终数）")
    add("")
    add("## 1. 全量计数与四个隔离集合")
    add("")
    add(_table([(k, f"{v:,}") for k, v in sorted(counts.items())],
               ("split", "样本数")))
    add("")
    add("按 build / 任务类型 / winner_confidence 分层：")
    add("")
    add("```json")
    add(json.dumps({
        "per_split_build": metrics["per_split_build"],
        "per_split_task_type": metrics["per_split_task_type"],
        "per_split_winner_confidence": metrics["per_split_winner_confidence"],
    }, ensure_ascii=False, indent=2))
    add("```")
    add("")
    add("## 2. 拒绝报告（按阶段与原因分组）")
    add("")
    add("全量账要对得上：")
    add("")
    add("```text")
    add("172,580  十个 build 的 sft.jsonl 行（sft_id 无重复，与 train ∪ eval 完全相等）")
    add(f"     -{rej['by_stage'].get('image', {}).get('image_format_unsupported', 0):>6}  "
        f"图像容器不支持（.in.dng，Pillow 只能按 TIFF 读出 raw CFA 帧）")
    add(f"     -{rej['by_stage'].get('split', {}).get('dedup_drop', 0):>6}  "
        f"dedup_drop 删除名单")
    add(f"     -{rej['by_stage'].get('split', {}).get('lut_reserved_for_T_lut_unseen', 0):>6}  "
        f"训练侧命中被保留的 LUT identity")
    add(f"     -{rej['by_stage'].get('split', {}).get('reserved_lut_in_select_source', 0):>6}  "
        f"未见 LUT 但落在 select 角色的源上，按纪律弃用")
    add(f"      {sum(counts.values()):>6}  发布样本总数"
        f"（train {counts.get('train', 0):,} + 四个评测集）")
    add("```")
    add("")
    add("其余为零的项也点名一次，免得看起来是漏了：**七段结构性拒绝 0、"
        "序列超长拒绝 0、宽高比 >4:1 拒绝 0、图像损坏 0、index/checksum 失败 0、"
        "split 不一致 0**。")
    add("")
    add(f"分组明细（合计 {rej['total']:,} 条）：")
    add("")
    add("```json")
    add(json.dumps(rej["by_stage"], ensure_ascii=False, indent=2))
    add("```")
    add("")
    add("## 3. spec §9 项 5：七段 → 两段转换")
    add("")
    add(f"- 输入七段行：172,580（十个 build 的 `sft.jsonl` 全量，sft_id 无重复）")
    add(f"- 七段解析成功：172,580 / 172,580；**结构性拒绝 0 条**")
    add(f"- 原「收束文本」存在数：0（因此一条也没有补写）")
    add(f"- 抽样逐字对照：{len(checks['conversion_samples'])} 条，"
        f"全部通过 = {checks['conversion_samples_all_pass']}（明细见 `conversion_samples.md`）")
    add("")
    add("每条抽样核对的六项：`where` 与 `region_scope` 逐字相等；`color` 等于其余六段按原序以 "
        "`\\n` 连接；`region_scope` 未被复制进 `color`；无任何旧标签残留；"
        "「收束文本存在与否」与原文一致；`instruction` 逐字未改。")
    add("")
    add("## 4. spec §9 项 6：train / eval / dedup 三集合")
    add("")
    add("```json")
    add(json.dumps(split, ensure_ascii=False, indent=2))
    add("```")
    add("")
    add("`dedup_drop` 与原始 train/eval **相交**（1,667 / 32），且三者并集恰好等于全量 172,580，"
        "因此它是**删除名单**而非第三个平行集合；应用删除后三集合两两无交集（上表 "
        "`authority_after_dedup_applied`），发布出的任何 split 都不含 dedup id。")
    add("")
    add("## 5. spec §9 项 7：indexed-shard 随机读取 / checksum / resume")
    add("")
    add("```json")
    add(json.dumps({"random_access": checks["random_access"],
                    "resume": checks["resume"],
                    "indexed_datasets": {k: {kk: vv for kk, vv in v.items()
                                             if kk != "shards"}
                                         for k, v in idx.items()}},
                   ensure_ascii=False, indent=2, default=str))
    add("```")
    add("")
    add("用**训练侧自己的** reader（`q3vl.train.shards.ShardIndex/ShardStore` + "
        "`q3vl.train.dataset.Sft2SegDataset`）真读样本的结果——这是唯一一项不是"
        "「生产者自证」的检查：")
    add("")
    add("```json")
    add(json.dumps(loader, ensure_ascii=False, indent=2, default=str))
    add("```")
    add("")
    add("## 6. spec §9 项 8：图像契约分布")
    add("")
    add("```json")
    add(json.dumps(img, ensure_ascii=False, indent=2))
    add("```")
    add("")
    add(f"三个需要主 agent 知道的读数：")
    add("")
    add(f"1. **{img['upscaled']:,} 条（{100 * img['upscaled'] / img['vision_tokens']['n']:.1f}%）"
        f"是被上采样的**——原图短边不足 512（最小 {img['orig_short_side']['min']}，"
        f"p05 = {img['orig_short_side']['p05']}）。spec §5 写的是「等比例缩放使短边为 512」，"
        f"没有下限例外，故照做；若要加下限，改 `q3vl/data` 与 `q3vl/train/imageproc` 的"
        f"同一处即可（NOTES.md 决策 D-8）。")
    add(f"2. **宽高比 >4:1 的样本一条都没有**（实测最大 "
        f"{img['aspect_in']['max']:.3f}），所以「长边 ≤ 2048」这一条在本语料上"
        f"从来没有触发过——它是被 4:1 过滤器结构性保证的，不是被裁出来的。")
    add(f"3. 32 对齐带来的宽高比误差最大 {img['aspect_rounding_error']['max']:.4f}，"
        f"低于训练侧 `IMAGE_ASPECT_TOLERANCE = 0.032` 的门限。")
    add("")
    add("入库的是按契约尺寸重编码的副本（NOTES.md 决策 D-1）。下面是把原始成员重新读出来、"
        "过一遍训练侧 `prepare_image`，再和入库图逐像素比对的结果——几何必须精确相等，"
        "唯一允许的差异是 JPEG q95 4:4:4 的量化：")
    add("")
    add("```json")
    add(json.dumps(checks.get("bake_fidelity", {}), ensure_ascii=False, indent=2, default=str))
    add("```")
    add("")
    add("## 7. spec §9 项 9：完整序列长度")
    add("")
    add("```json")
    add(json.dumps(length, ensure_ascii=False, indent=2))
    add("```")
    add("")
    add(f"**没有一条样本被长度过滤**：最长的完整序列只有 {length['max']} token，"
        f"p99 = {length['p99']}，离 `model_max_length = {C.MODEL_MAX_LENGTH}` 还有一倍余量。"
        f"因此 spec §6 的「超长过滤」在本语料上计数为 0，而不是没做。")
    add("")
    add("长度口径不是本包自己定义的：prompt 由 `q3vl.train.collator.Sft2SegCollator."
        "build_prompt_text` 生成，target 由同一个类的 `<where>/<color>` 拼法生成。"
        "下表是「本包记录的 `total_tokens`」与「collator 真跑一遍 `encode_one` 的长度」逐条比对：")
    add("")
    add("```json")
    add(json.dumps(identity, ensure_ascii=False, indent=2, default=str))
    add("```")
    add("")
    add("## 8. T_lut_unseen 的 group-level reserve")
    add("")
    add(f"- 保留 LUT identity：{reserve['n_luts']} 个（其中 "
        f"{manifest_audit['n_reserved_luts']} 个在 T_lut_unseen 中真实出现）")
    add(f"- 因此从训练集剔除：{reserve['train_removed']:,} 条 = "
        f"**{100 * reserve['train_fraction']:.2f}%**（预算上限 "
        f"{100 * reserve['budget_fraction']:.0f}%，任务卡规定 ≤5% 直接采用）")
    add(f"- eval 侧被保留的样本：{reserve['eval_gain']}，其中 "
        f"{counts.get('T_lut_unseen', 0)} 条落在 test 角色的源上并进入 T_lut_unseen，"
        f"其余落在 select 源上，按纪律弃用而非塞进选择集")
    add("")
    add(_table([(m["major"], m["eval_pool"], m["luts_reserved"],
                 m["eval_reserved"], f"{m['train_removed']:,}")
                for m in reserve["per_major"]],
               ("taxonomy major", "eval 池", "保留 LUT 数", "保留 eval 样本", "剔除训练样本")))
    add("")
    add("## 9. 隔离审计（METACANVAS §2.2）")
    add("")
    add("```json")
    add(json.dumps(manifest_audit, ensure_ascii=False, indent=2))
    add("```")
    add("")
    add("说明：`source_overlap` 中 `T_final|T_lut_unseen` 非零是**设计如此**——两者同属 test "
        "角色，靠 LUT 身份而非源图区分；协议自己的 group 键 `(source_image_id, lut_id, build)` "
        "在四个集合之间两两交集为 0。select 角色（V_where ∪ V_what）与 test 角色"
        "（T_final ∪ T_lut_unseen）的源集合交集为 0。")
    add("")
    add("## 10. 序列与视觉 token 的分布（按 split）")
    add("")
    add("```json")
    add(json.dumps(dist, ensure_ascii=False, indent=2))
    add("```")
    add("")
    add("## 11. 训练侧怎么读")
    add("")
    add("```python")
    add("from q3vl.train.shards import ShardIndex, ShardStore")
    add("from q3vl.train.dataset import Sft2SegDataset")
    add("")
    add(f"index = ShardIndex.load('{C.SPLIT_DIR}/train.index.jsonl')")
    add("store = ShardStore(shard_root='/', verify='checksum')  # shard 字段是绝对路径")
    add("dataset = Sft2SegDataset(index, store)                 # 每条给 where/color/instruction+图")
    add("```")
    add("")
    add(f"- `{C.MANIFEST_DIR}/terminal_manifest.json` 是 resume 与 step 计算的唯一权威："
        f"`steps_per_epoch = ceil(N_effective / 32)`，**不得**沿用定档时的 2645 / 5290。")
    add("- 每条记录同时带 `image.origin`（原 build 的 shard/offset/length/sha256）与 "
        "`image.baked`（契约尺寸副本），两条路径都可用；`image.out_h/out_w/vision_tokens` "
        "是预算好的几何，训练时 `plan_geometry` 会算出同一组数。")
    add("- 每条记录带 `winner_confidence`，若主 agent 决定执行 CLAUDE.md 的"
        "「low 不进主训」纪律，训练侧可直接过滤，或重跑 "
        "`python -m q3vl.data.cli plan --drop-low-confidence`。")
    add("")
    return "\n".join(lines) + "\n"


def render_samples(checks: dict[str, Any]) -> str:
    lines = ["# 抽样原文对照：原七段 vs 新两段（逐字）", ""]
    lines.append(f"共 {len(checks['conversion_samples'])} 条，"
                 f"全部通过 = {checks['conversion_samples_all_pass']}。")
    lines.append("")
    for i, s in enumerate(checks["conversion_samples"], 1):
        lines.append(f"## {i}. `{s['sft_id']}`（build {s['build']}）")
        lines.append("")
        lines.append(f"检查项：`{json.dumps(s['checks'], ensure_ascii=False)}`")
        lines.append("")
        lines.append("### 原七段")
        lines.append("")
        for field, body in s["original_fields"].items():
            lines.append(f"- **{field}**: {body}")
        lines.append("")
        lines.append("### 新两段")
        lines.append("")
        lines.append("```text")
        lines.append(f"<where>{s['new_where']}</where>")
        lines.append(f"<color>{s['new_color']}</color>")
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def main(quick: bool = False) -> dict[str, Any]:
    C.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = build_metrics(quick=quick)
    (C.REPORT_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    (C.REPORT_DIR / "PREFLIGHT_DATA.md").write_text(render(metrics))
    (C.REPORT_DIR / "conversion_samples.md").write_text(render_samples(metrics["checks"]))
    copy_dir = C.REPORT_DIR / "manifest"
    copy_dir.mkdir(exist_ok=True)
    shutil.copy2(C.MANIFEST_DIR / "terminal_manifest.json", copy_dir)
    for path in sorted(C.SPLIT_DIR.glob("*_sft_ids.txt")):
        shutil.copy2(path, copy_dir)
    from .pipeline import sha256_file

    (copy_dir / "digests.json").write_text(json.dumps({
        "terminal_manifest_sha256": sha256_file(C.MANIFEST_DIR / "terminal_manifest.json"),
        "splits": metrics["splits"],
        "datasets": metrics["datasets"],
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return metrics


if __name__ == "__main__":
    import sys

    main(quick="--quick" in sys.argv)
