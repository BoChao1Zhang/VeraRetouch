"""选源规则（**effect-blind**，A / B 两组共用同一批源）。

规则（跑图之前写死，不看任何读出结果）：

1. 候选 = RO-9c 已冻结的 `RO9c_subject_repro_20260805/config/figure_picks.json`
   的 `picked` 列表，**保持冻结顺序**。该列表本身的规则是分层取样
   （pool=awards / unsplash / ppr10k、region_b_kind=spatial / background，
   filters: winner_confidence != low、subject_area ∈ [0.08, 0.35]），
   只用 G1 配置里的源属性，**不看任何读出结果**。
2. **版式过滤**：剔除长宽比 > 1.8 的源。理由是版式而非效果——单行五列 + 每格短边
   ≥ 460 px，长宽比 2:1 的源会让整图宽到 4800 px，PPT 一页内每格短边不足 100 px 无法读。
   本批唯一被剔除的是 `src_08aef2f575be97b1`（1920×960，2.00，而且是左右双联幅）。
3. 取过滤后列表的**前 3 项**。

A 组（图 4）与 B 组（图 5）用**同一批源**，这样两页可以直接对读。
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
G1 = REPO / "experiments" / "G1_s_identifiability_20260803"
RO9C = REPO / "experiments" / "RO9c_subject_repro_20260805"

MAX_ASPECT = 1.8
N_PICK = 3

RULE = ("RO-9c 冻结 figure_picks.json 的 picked 列表，保持冻结顺序；"
        f"剔除长宽比 > {MAX_ASPECT} 的源（版式原因：单行五列在 PPT 一页内不可读；"
        "本批只剔除 src_08aef2f575be97b1 = 1920×960 双联幅）；取前 3 项。"
        "全程 effect-blind——只用 G1 配置里的源属性与图像长宽比，不看任何读出结果。")


def pick_sources(n: int = N_PICK) -> tuple[list[str], dict]:
    from PIL import Image

    region = {r["img_id"]: r for r in
              json.loads((G1 / "config" / "g1_region_opp.json").read_text())}
    frozen = json.loads((RO9C / "config" / "figure_picks.json").read_text())["picked"]
    kept, dropped = [], []
    for iid in frozen:
        w, h = Image.open(region[iid]["img_path"]).size
        asp = max(w, h) / min(w, h)
        (kept if asp <= MAX_ASPECT else dropped).append(
            {"img_id": iid, "size": [w, h], "aspect": round(asp, 3),
             "pool": region[iid]["pool"], "region_b_kind": region[iid]["region_b_kind"]})
    return [k["img_id"] for k in kept[:n]], {
        "rule": RULE, "frozen_list": frozen,
        "dropped_for_layout": dropped, "kept": kept[:n]}
