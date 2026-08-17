"""LORA_P3 language-only retry entry (EPR-011).  Seams over the frozen entry:
hiddens.FrozenVLM -> LangOnlyLoRAVLM; amort.model.AmortModel -> AmortModelVLora."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--uniq3c-lora-r", type=int, default=16)
    own, rest = ap.parse_known_args(argv)

    import q3vl.whereb.amort.model as amodel
    import q3vl.whereb.hiddens as ahid
    from q3vl.whereb.amort import uniq3
    from q3vl.whereb.amort.uniq3c import LangOnlyLoRAVLM

    uniq3.VARIANT3.update(readout="hidden", lora=True)

    def _make(model, processor, device="cuda", **kw):
        vlm = LangOnlyLoRAVLM(model, processor, device,
                              lora_r=own.uniq3c_lora_r, **kw)
        uniq3.VARIANT3["vlm_ref"] = vlm
        return vlm

    ahid.FrozenVLM = _make
    amodel.AmortModel = uniq3.AmortModelVLora

    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--out-root", default="/home/bc/data/runs/where_b")
    peek.add_argument("--run-name", default=None)
    loc, _ = peek.parse_known_args(rest)
    if loc.run_name:
        cfg_dir = Path(loc.out_root) / loc.run_name / "config"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        here = Path(__file__).resolve()
        u3c = here.parent.parent / "amort" / "uniq3c.py"
        (cfg_dir / "uniq3c_setup.json").write_text(json.dumps({
            "lora": "language-only", "lora_r": own.uniq3c_lora_r,
            "rationale": "visual tower untouched => sim producer cannot drift",
            "uniq3c_sha256": hashlib.sha256(u3c.read_bytes()).hexdigest(),
            "wrapper_sha256": hashlib.sha256(here.read_bytes()).hexdigest(),
        }, indent=2), encoding="utf-8")
    print(f"UNIQ3C language-only LoRA r={own.uniq3c_lora_r} (seams installed)",
          flush=True)

    from q3vl.whereb.scripts.run_amort_arm import main as base_main

    return base_main(rest)


if __name__ == "__main__":
    raise SystemExit(main())
