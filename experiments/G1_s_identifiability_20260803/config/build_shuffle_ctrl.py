"""指令条件性对照（NOTES §六）：前 60 源 × 1 条"错位指令"（shuf）。

shuf(img_i) = 洗牌序下一源的 syn_a 指令（确定性错位，区域名词大概率不同）。
若 corr(s(img,syn_a), s(img,shuf)) 中位数 ≈ ρ_syn，则 s 无指令条件性。
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
N = 60

samples = json.loads((HERE / "g1_samples.json").read_text())
out = []
for i, s in enumerate(samples[:N]):
    donor = samples[(i + 1) % len(samples)]
    out.append({
        "img_id": s["img_id"], "img_path": s["img_path"], "pool": s["pool"],
        "build": s["build"], "donor_img_id": donor["img_id"],
        "instructions": {"shuf": donor["instructions"]["syn_a"]},
    })
(HERE / "g1_shuffle_ctrl.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(f"wrote {len(out)} shuffle-control samples")
