"""Stage 3, row A (campaign env): score AceTone-3B-PST-Preview.

    /home/bc/envs/q3vl_sft/bin/python -m q3vl.whatb.acetone.run_pst_row \\
        --run-dir /home/bc/data/runs/what_b/whatb_ACETONE_pst

Columns this board carries beyond the four baselines and the headline:

``N_ref_shuffle_delta`` / ``N_ref_shuffle_M``
    the reference-shuffle control -- image2 replaced by another row's GT
    after-image.  ``delta`` is the paired ``H(ctrl) - H(true)``; ``M`` is the
    mean 17³ function distance between the two predictions.
``parse_missing_rate``
    the fraction of generations in which fewer than 64 ``<MM..>`` tokens came
    back and ``eval/predict_lut_ddp.py``'s padding branch fired.
``headline_excl_padded``
    the headline recomputed with those samples dropped.
``reference_condition``
    recorded on the artefact: image2 is the row's own GT after-image.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from q3vl.whatb import criteria as C

from . import publish as PUB
from . import scoring as SC


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", default="/home/bc/data/runs/what_b/acetone_inputs")
    ap.add_argument("--run-dir", default="/home/bc/data/runs/what_b/whatb_ACETONE_pst")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    art = run_dir / "artifacts"
    pred_true = np.load(art / "pred_true.npz")
    pred_shuf = np.load(art / "pred_shuffle.npz")
    pst_facts = json.loads((art / "pst_facts.json").read_text(encoding="utf-8"))
    parse = [json.loads(l) for l in
             (art / "parse.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    padded_true = {r["sample_id"] for r in parse
                   if r["condition"] == "true" and r["padded"]}
    padded_any = {r["sample_id"] for r in parse if r["padded"]}
    ref_of = {r["sample_id"]: r["ref_sample_id"] for r in parse
              if r["condition"] == "shuffle"}

    fx = SC.build_fixtures(device=args.device, limit=args.limit or None)
    preds = {r.sample_id: pred_true[r.sample_id] for r in fx.rows}
    ctrl = {"N_ref_shuffle": {r.sample_id: pred_shuf[r.sample_id] for r in fx.rows}}

    rows, finite = SC.score_rows(fx, preds, control_preds=ctrl)
    for r in rows:
        r["parse_padded"] = bool(r["sample_id"] in padded_true)
        r["parse_padded_any_condition"] = bool(r["sample_id"] in padded_any)
        r["shuffle_ref_sample_id"] = ref_of.get(r["sample_id"])

    arm_e = [r["E_arm"] for r in rows]
    keep = [r["E_arm"] for r in rows if not r["parse_padded_any_condition"]]
    cols = {
        **C.control_columns(arm_e, [r["E_N_ref_shuffle"] for r in rows],
                            [r["M_N_ref_shuffle"] for r in rows],
                            name="N_ref_shuffle", seed=SC.SEED),
        "parse_missing_rate": {
            "n": len(rows),
            "mean": float(np.mean([1.0 if r["parse_padded"] else 0.0
                                   for r in rows])),
            "n_padded_true": len(padded_true & {r["sample_id"] for r in rows}),
            "n_padded_any_condition": len(padded_any & {r["sample_id"] for r in rows}),
            "quantity": ("fraction of `true`-condition generations where fewer "
                         "than 64 <MM..> tokens were parsed and the padding "
                         "branch of eval/predict_lut_ddp.py fired")},
        "headline_excl_padded": {
            **C.describe(keep),
            "quantity": ("mean dE00(Î, I*) over the rows whose `true` and "
                         "`shuffle` generations both parsed 64 tokens")},
        "pred_grid_std": PUB.spread_columns(rows),
    }
    facts = {
        "row": "A -- AceTone-3B-PST-Preview, GT after-image as style reference",
        "reference_condition": ("image2 = the row's own GT after-image "
                                "(oracle-reference); the same side as B4"),
        "fixtures": fx.facts(),
        "A_rows": fx.a_rows,
        "A_finite": finite,
        "A_parse": pst_facts.get("A_parse"),
        "A_axis": json.loads(
            (Path("/home/bc/data/runs/what_b/whatb_ACETONE_tokenizer/artifacts/"
                  "vq_facts.json")).read_text(encoding="utf-8")).get("A_axis"),
        "pst": {k: v for k, v in pst_facts.items()
                if k not in ("repo", "A_parse")},
    }
    board = PUB.build_and_publish(rows, arm="EPR-034-A", run_dir=run_dir,
                                  extra_columns=cols, facts=facts)
    print(json.dumps({
        "headline_normal_only": board["contexts"]["all"]["headline_normal_only"],
        **{k: board["criteria_columns"][k] for k in
           ("headline_excl_padded", "parse_missing_rate", "grid_error",
            "N_ref_shuffle_delta", "N_ref_shuffle_M",
            "B0_identity", "B3_bucket_retrieval", "B4_oracle", "B6_libfill")},
    }, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
