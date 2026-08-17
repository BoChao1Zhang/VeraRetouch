"""UNIQ wave-6 entry: in-context query tokens (MQ frozen / ST + LoRA).

Flags: --uniq4-qtok INT (default 8), --uniq4-lora, --uniq4-lora-r INT.
Seams: hiddens.FrozenVLM -> QueryTokVLM; amort.model.AmortModel -> AmortModelV4.
Rest passes verbatim to the frozen run_amort_arm (--arm UNIQ expected).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--uniq4-qtok", type=int, default=8)
    ap.add_argument("--uniq4-lora", action="store_true")
    ap.add_argument("--uniq4-lora-r", type=int, default=16)
    own, rest = ap.parse_known_args(argv)

    import q3vl.whereb.amort.model as amodel
    import q3vl.whereb.hiddens as ahid
    from q3vl.whereb.amort import uniq4

    uniq4.VARIANT4.update(n_qtok=own.uniq4_qtok, lora=own.uniq4_lora,
                          lora_r=own.uniq4_lora_r)

    def _make(model, processor, device="cuda", **kw):
        vlm = uniq4.QueryTokVLM(model, processor, device,
                                n_qtok=own.uniq4_qtok, lora=own.uniq4_lora,
                                lora_r=own.uniq4_lora_r, **kw)
        uniq4.VARIANT4["vlm_ref"] = vlm
        return vlm

    ahid.FrozenVLM = _make
    amodel.AmortModel = uniq4.AmortModelV4

    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--out-root", default="/home/bc/data/runs/where_b")
    peek.add_argument("--run-name", default=None)
    loc, _ = peek.parse_known_args(rest)
    if loc.run_name:
        cfg_dir = Path(loc.out_root) / loc.run_name / "config"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        here = Path(__file__).resolve()
        u4 = here.parent.parent / "amort" / "uniq4.py"
        (cfg_dir / "uniq4_setup.json").write_text(json.dumps({
            "n_qtok": own.uniq4_qtok, "lora": own.uniq4_lora,
            "lora_r": own.uniq4_lora_r,
            "uniq4_sha256": hashlib.sha256(u4.read_bytes()).hexdigest(),
            "wrapper_sha256": hashlib.sha256(here.read_bytes()).hexdigest(),
        }, indent=2), encoding="utf-8")
    print(f"UNIQ4 qtok={own.uniq4_qtok} lora={own.uniq4_lora} (seams installed)",
          flush=True)

    from q3vl.whereb.scripts.run_amort_arm import main as base_main

    return base_main(rest)


if __name__ == "__main__":
    raise SystemExit(main())
