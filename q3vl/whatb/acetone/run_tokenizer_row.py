"""Stage 3, row C (campaign env): score the tokenizer round trip.

    /home/bc/envs/q3vl_sft/bin/python -m q3vl.whatb.acetone.run_tokenizer_row \\
        --inputs /home/bc/data/runs/what_b/acetone_inputs \\
        --run-dir /home/bc/data/runs/what_b/whatb_ACETONE_tokenizer

Two LUT sets are scored on the same 567 rows and the same formation:

``tokenizer_recon``  bank LUT -> 32³ (AceTone ``resize_lut``) -> VQ encode ->
                     64 tokens -> VQ decode.  This is the board's headline.
``resample_only``    bank LUT -> 32³ and nothing else.  The resampling step's
                     own error, published as its own column plus the paired
                     ``tokenizer_minus_resample`` delta.

Both are also reported as 17³ function-value ΔE00 against the GT LUT.
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
    ap.add_argument("--run-dir",
                    default="/home/bc/data/runs/what_b/whatb_ACETONE_tokenizer")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    art = run_dir / "artifacts"
    recon = np.load(art / "recon32.npz")
    res32 = np.load(Path(args.inputs) / "luts32.npz")
    vq_facts = json.loads((art / "vq_facts.json").read_text(encoding="utf-8"))

    fx = SC.build_fixtures(device=args.device, limit=args.limit or None)
    preds = {r.sample_id: recon[r.lut_id] for r in fx.rows}
    extra = {"resample_only": {r.sample_id: res32[r.lut_id] for r in fx.rows}}

    rows, finite = SC.score_rows(fx, preds, extra_preds=extra)

    arm_e = [r["E_arm"] for r in rows]
    res_e = [r["resample_only"] for r in rows]
    cols = {
        "resample_only": {**C.describe(res_e),
                          "quantity": ("mean dE00(Î, I*) with f̂ = the GT LUT "
                                       "resampled to 32³ by AceTone's resize_lut")},
        "resample_only_grid_error": {
            **C.describe([r["resample_only_grid_error"] for r in rows]),
            "quantity": "mean dE00(f̂, L_GT) on the 17³ grid, resample only"},
        "tokenizer_minus_resample": {
            **C.paired_stats(arm_e, res_e, seed=SC.SEED),
            "quantity": "paired H(tokenizer_recon) - H(resample_only)"},
        "pred_grid_std": PUB.spread_columns(rows),
        "parse_missing_rate": {"n": len(rows), "mean": 0.0, "std": 0.0,
                               "quantity": ("row C emits no text: the 64 codes "
                                            "come from the encoder, never parsed")},
    }
    facts = {
        "row": "C -- AceTone LUT tokenizer reconstruction of the GT LUTs",
        "n_lut_ids": int(len({r["lut_id"] for r in rows})),
        "fixtures": fx.facts(),
        "A_rows": fx.a_rows,
        "A_axis": vq_facts.get("A_axis"),
        "A_finite": finite,
        "A_parse": {"applicable": False,
                    "reason": "no token text is generated in row C",
                    "n_missing_events": 0, "parse_missing_rate": 0.0},
        "vq": vq_facts.get("vq"),
        "tokenizer_run": vq_facts.get("tokenizer_run"),
        "cube_reader_parity": json.loads(
            (Path(args.inputs) / "cube_parity.json").read_text(encoding="utf-8")),
    }
    board = PUB.build_and_publish(rows, arm="EPR-034-C", run_dir=run_dir,
                                  extra_columns=cols, facts=facts)
    print(json.dumps({
        "headline_normal_only": board["contexts"]["all"]["headline_normal_only"],
        "resample_only": board["criteria_columns"]["resample_only"],
        "tokenizer_minus_resample":
            board["criteria_columns"]["tokenizer_minus_resample"],
        "grid_error": board["criteria_columns"]["grid_error"],
        "resample_only_grid_error":
            board["criteria_columns"]["resample_only_grid_error"],
        **{k: board["criteria_columns"][k] for k in
           ("B0_identity", "B3_bucket_retrieval", "B4_oracle", "B6_libfill")},
    }, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
