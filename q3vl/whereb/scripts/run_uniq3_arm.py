"""UNIQ wave-5 entry: readout comparison + LoRA arms, seams over the frozen entry.

Flags (own):
  --uniq3-readout {hidden,code,attn}   K/V source for the unified query head
  --uniq3-lora                          train LoRA adapters in the VLM
  --uniq3-lora-r INT                    LoRA rank (default 16)
Everything else is passed verbatim to the frozen run_amort_arm (incl. --arm).

Seams installed (all read at call time by the frozen entry):
  amort.model.AmortModel      -> AmortModelV3 (UNIQ) / AmortModelVLora (P3')
  whereb.hiddens.FrozenVLM    -> TrainableVLM (lora) / AttnCaptureVLM (attn)
  amort.data.AmortBatchBuilder-> attn-channel builder (attn readout only)
  amort.losses.uniq_wta_loss  stays the ORIGINAL (no FQ dispatch here)
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--uniq3-readout", default="hidden",
                    choices=["hidden", "code", "attn"])
    ap.add_argument("--uniq3-lora", action="store_true")
    ap.add_argument("--uniq3-lora-r", type=int, default=16)
    own, rest = ap.parse_known_args(argv)

    import q3vl.whereb.amort.data as adata
    import q3vl.whereb.amort.model as amodel
    import q3vl.whereb.hiddens as ahid
    from q3vl.whereb.amort import uniq3

    uniq3.VARIANT3.update(readout=own.uniq3_readout, lora=own.uniq3_lora,
                          lora_r=own.uniq3_lora_r)

    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--arm", default="UNIQ")
    peek.add_argument("--out-root", default="/home/bc/data/runs/where_b")
    peek.add_argument("--run-name", default=None)
    loc, _ = peek.parse_known_args(rest)

    amodel.AmortModel = (uniq3.AmortModelV3 if loc.arm == "UNIQ"
                         else uniq3.AmortModelVLora)
    if own.uniq3_lora:
        def _trainable(model, processor, device="cuda", **kw):
            vlm = uniq3.TrainableVLM(model, processor, device,
                                     lora_r=own.uniq3_lora_r, **kw)
            uniq3.VARIANT3["vlm_ref"] = vlm
            return vlm
        ahid.FrozenVLM = _trainable
    elif own.uniq3_readout == "attn":
        def _capture(model, processor, device="cuda", **kw):
            return uniq3.AttnCaptureVLM(
                model, processor, device,
                attn_layers=uniq3.VARIANT3["attn_layers"], **kw)
        ahid.FrozenVLM = _capture
        adata.AmortBatchBuilder = uniq3.make_attn_builder(
            adata.AmortBatchBuilder)
        n_layers = len(uniq3.VARIANT3["attn_layers"])
        # the model needs the widened stem; passed via VARIANT3-aware ctor
        _orig_v3 = uniq3.AmortModelV3

        class _V3Attn(_orig_v3):
            def __init__(self, arm="P1", **kw):
                kw.setdefault("attn_extra_ch", n_layers)
                super().__init__(arm, **kw)

        amodel.AmortModel = _V3Attn

    if loc.run_name:
        cfg_dir = Path(loc.out_root) / loc.run_name / "config"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        here = Path(__file__).resolve()
        u3 = here.parent.parent / "amort" / "uniq3.py"
        (cfg_dir / "uniq3_setup.json").write_text(json.dumps({
            "variant": {k: v for k, v in uniq3.VARIANT3.items()
                        if k != "vlm_ref"},
            "arm": loc.arm,
            "uniq3_sha256": hashlib.sha256(u3.read_bytes()).hexdigest(),
            "wrapper_sha256": hashlib.sha256(here.read_bytes()).hexdigest(),
        }, indent=2), encoding="utf-8")
    print(f"UNIQ3 readout={own.uniq3_readout} lora={own.uniq3_lora} "
          f"arm={loc.arm} (seams installed)", flush=True)

    from q3vl.whereb.scripts.run_amort_arm import main as base_main

    return base_main(rest)


if __name__ == "__main__":
    raise SystemExit(main())
