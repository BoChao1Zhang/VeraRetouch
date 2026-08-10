"""G1 补充批：区域对立指令对（D-17 修订后的主判据）。

同一张图两条指令，方向词相同、指向**不同区域**；预期 s 场分离（ρ_region_opp<0.3）。

取材（任务卡）：D-SFT-L(S-val) 的 local.subject.description（l 系指令天然带区域）+
local.region（主体粗位置：center/lower/left/...）。主批 300 源中 214 源有 local 行，
但**没有任何源带 ≥2 个不同 subject**（逐源核实），故全部模板构造：
  reg_a = "Please {dir} the {subject.description}, keeping the rest of the image unchanged."
  reg_b = "Please {dir} {complement}, keeping the rest of the image unchanged."
complement 按 local.region 取空间对侧（left→right side...），center→"the background"
（主体 vs 背景，即任务卡四类区域词中的 主体/背景/上下左右方位）。两区域保证不同。
方向词 brighten/darken 逐源交替（对内相同），与 GL=<retouch_light> 的亮度轴对齐。

源 = 主批 g1_samples.json ∩ 有 subject 的源（图已暂存，池近均衡 75/76/63 ≥150 ✓）。
确定性：seed=20260803。输出 g1_region_opp.json + g1_region_opp_report.json。
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
JOURNAL = Path("/var/cache/veradata/annot_review/journal-archive")
SEED = 20260803
CONF_RANK = {"normal": 0, "low": 1}

# local.region -> 对侧区域词（reg_b）；center 用 背景（主体的补集，必然存在）
COMPLEMENT = {
    "center": "the background",
    "lower": "the upper part of the image",
    "upper": "the lower part of the image",
    "left": "the right side of the image",
    "right": "the left side of the image",
    "lower left": "the upper right part of the image",
    "lower right": "the upper left part of the image",
    "upper left": "the lower right part of the image",
    "upper right": "the lower left part of the image",
}
TEMPLATE = "Please {d} {region}, keeping the rest of the image unchanged."


def with_article(desc: str) -> str:
    d = desc.strip().rstrip(".")
    low = d.lower()
    return d if low.startswith(("the ", "a ", "an ")) else "the " + d


def main() -> None:
    samples = json.loads((HERE / "g1_samples.json").read_text())
    main_by_id = {s["img_id"]: s for s in samples}

    # source_id -> local 行（subject 必有 description）
    rows_by_src: dict[str, list[dict]] = defaultdict(list)
    builds = sorted(p.name for p in JOURNAL.iterdir() if p.name.startswith("prod-l")
                    and (p / "sft.jsonl").is_file() and (p / "groups.jsonl").is_file())
    for build in builds:
        g2s = {}
        with open(JOURNAL / build / "groups.jsonl", encoding="utf-8") as f:
            for line in f:
                g = json.loads(line)
                if g.get("source_id") in main_by_id:
                    g2s[g["group_id"]] = g["source_id"]
        with open(JOURNAL / build / "sft.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                sid = g2s.get(r.get("group_id"))
                if sid is None or r.get("task_type") != "local":
                    continue
                loc = r.get("local") or {}
                subj = loc.get("subject") or {}
                if not subj.get("description") or loc.get("region") not in COMPLEMENT:
                    continue
                rows_by_src[sid].append({
                    "build": build, "sft_id": r.get("sft_id"),
                    "conf": r.get("winner_confidence", "low"),
                    "region": loc["region"], "subject_desc": subj["description"],
                    "subject_area": subj.get("area"),
                })

    rng = random.Random(SEED)
    out, report = [], {"builds": builds, "per_pool": defaultdict(int),
                       "region_dist": defaultdict(int), "dir_dist": defaultdict(int),
                       "conf_dist": defaultdict(int), "n_candidate_sources": len(rows_by_src)}
    sids = sorted(rows_by_src)
    rng.shuffle(sids)
    for i, sid in enumerate(sids):
        row = sorted(rows_by_src[sid],
                     key=lambda r: (CONF_RANK.get(r["conf"], 2), r["build"], r["sft_id"]))[0]
        d = "brighten" if i % 2 == 0 else "darken"
        m = main_by_id[sid]
        out.append({
            "img_id": sid, "img_path": m["img_path"], "pool": m["pool"],
            "build": row["build"], "sft_id": row["sft_id"],
            "winner_confidence": row["conf"], "direction": d,
            "subject_region": row["region"], "subject_area": row["subject_area"],
            "region_b_kind": "background" if row["region"] == "center" else "spatial",
            "instructions": {
                "reg_a": TEMPLATE.format(d=d, region=with_article(row["subject_desc"])),
                "reg_b": TEMPLATE.format(d=d, region=COMPLEMENT[row["region"]]),
            },
        })
        report["per_pool"][m["pool"]] += 1
        report["region_dist"][row["region"]] += 1
        report["dir_dist"][d] += 1
        report["conf_dist"][row["conf"]] += 1

    (HERE / "g1_region_opp.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    report["total"] = len(out)
    report = {k: (dict(v) if isinstance(v, defaultdict) else v) for k, v in report.items()}
    (HERE / "g1_region_opp_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
