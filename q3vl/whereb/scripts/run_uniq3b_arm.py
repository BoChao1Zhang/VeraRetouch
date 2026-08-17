"""RATTN hotfix entry (wave-5): attn readout with the SemanticHead fix.

run_uniq3_arm.py is frozen under the running RCODE arm, so this thin wrapper
re-implements only the attn branch with uniq3b.make_v3_attn_fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--uniq3b-attn-layers", type=int, default=4)
    own, rest = ap.parse_known_args(argv)

    import q3vl.whereb.amort.data as adata
    import q3vl.whereb.amort.model as amodel
    import q3vl.whereb.hiddens as ahid
    from q3vl.whereb.amort import uniq3
    from q3vl.whereb.amort.uniq3b import make_v3_attn_fixed

    uniq3.VARIANT3.update(readout="attn", lora=False)
    layers = uniq3.VARIANT3["attn_layers"][: own.uniq3b_attn_layers]

    def _capture(model, processor, device="cuda", **kw):
        return uniq3.AttnCaptureVLM(model, processor, device,
                                    attn_layers=layers, **kw)

    ahid.FrozenVLM = _capture
    adata.AmortBatchBuilder = uniq3.make_attn_builder(adata.AmortBatchBuilder)
    amodel.AmortModel = make_v3_attn_fixed(len(layers))

    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--out-root", default="/home/bc/data/runs/where_b")
    peek.add_argument("--run-name", default=None)
    loc, _ = peek.parse_known_args(rest)
    if loc.run_name:
        cfg_dir = Path(loc.out_root) / loc.run_name / "config"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        here = Path(__file__).resolve()
        u3b = here.parent.parent / "amort" / "uniq3b.py"
        (cfg_dir / "uniq3b_setup.json").write_text(json.dumps({
            "readout": "attn", "attn_layers": list(layers),
            "fix": "SemanticHead sees sim[:, :1] only (path held fixed)",
            "uniq3b_sha256": hashlib.sha256(u3b.read_bytes()).hexdigest(),
            "wrapper_sha256": hashlib.sha256(here.read_bytes()).hexdigest(),
        }, indent=2), encoding="utf-8")
    print(f"UNIQ3B attn readout, layers={list(layers)} (semantic-path fix on)",
          flush=True)

    from q3vl.whereb.scripts.run_amort_arm import main as base_main

    return base_main(rest)


if __name__ == "__main__":
    raise SystemExit(main())
