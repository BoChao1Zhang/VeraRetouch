"""UNIQ wave-6 entry: in-context query tokens (MQ frozen / ST + LoRA).

Flags: --uniq4-qtok INT (default 8), --uniq4-lora, --uniq4-lora-r INT.

EPR-013 (masked cross-attention refinement) and EPR-016 (training-time
one-to-many auxiliary query groups) hang off the same entry, both default OFF:
with `--uniq4-refine-layers 0` and `--uniq4-aux-groups 0` no new module is
constructed, no new branch is entered, and the arm is the frozen ST_LANG
configuration.

Seams: hiddens.FrozenVLM -> QueryTokVLM; amort.model.AmortModel -> AmortModelV4.
Rest passes verbatim to the frozen run_amort_arm (--arm UNIQ expected).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

#: set by run_uniq5_arm before delegating here (EPR-014); None/{} = untouched
EXTRA_HEAD_CLS: Any = None
EXTRA_HEAD_KWARGS: dict[str, Any] = {}
EXTRA_SETUP: dict[str, Any] = {}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--uniq4-qtok", type=int, default=8)
    ap.add_argument("--uniq4-lora", action="store_true")
    ap.add_argument("--uniq4-lora-r", type=int, default=16)
    # --- EPR-013: masked cross-attention refinement (default off) ----------
    ap.add_argument("--uniq4-refine-layers", type=int, default=0,
                    help="EPR-013: masked cross-attn refinement layers between "
                         "q_proj_in and to_mask (0 = baseline)")
    ap.add_argument("--uniq4-refine-anneal", action="store_true",
                    help="EPR-013 ablation: EoMT mask annealing")
    ap.add_argument("--uniq4-refine-no-self-attn", action="store_true",
                    help="EPR-013 ablation: drop the per-layer query self-attn")
    ap.add_argument("--uniq4-refine-thr-raw", action="store_true",
                    help="EPR-013 ablation: EoMT's raw>0 feedback threshold "
                         "instead of Mask2Former's sigmoid<0.5")
    ap.add_argument("--uniq4-refine-no-aux-loss", action="store_true",
                    help="EPR-013 ablation: supervise only the final field")
    # --- EPR-016: one-to-many auxiliary query groups (default off) ---------
    ap.add_argument("--uniq4-aux-groups", type=int, default=0,
                    help="EPR-016: m extra groups of K training-only query "
                         "tokens (0 = baseline; inference is unchanged)")
    ap.add_argument("--uniq4-aux-lambda", type=float, default=1.0,
                    help="EPR-016: weight on the auxiliary groups (H-DETR "
                         "lambda1); groups are averaged (Group DETR Eq.7)")
    ap.add_argument("--uniq4-aux-seed", type=int, default=20260813,
                    help="EPR-016: seed for the auxiliary token embeddings")
    own, rest = ap.parse_known_args(argv)

    import q3vl.whereb.amort.model as amodel
    import q3vl.whereb.hiddens as ahid
    from q3vl.whereb.amort import uniq4
    from q3vl.whereb.amort.uniq4b import LangQueryTokVLM

    head_kwargs: dict[str, Any] = {}
    if own.uniq4_refine_layers:
        head_kwargs.update(
            n_refine_layers=int(own.uniq4_refine_layers),
            refine_anneal=bool(own.uniq4_refine_anneal),
            refine_self_attn=not own.uniq4_refine_no_self_attn,
            refine_thr_raw=bool(own.uniq4_refine_thr_raw),
            refine_aux_loss=not own.uniq4_refine_no_aux_loss)
    if own.uniq4_aux_groups:
        head_kwargs.update(aux_groups=int(own.uniq4_aux_groups),
                           aux_lambda=float(own.uniq4_aux_lambda))
    head_kwargs.update(EXTRA_HEAD_KWARGS)

    uniq4.VARIANT4.update(n_qtok=own.uniq4_qtok,
                          lora_r=own.uniq4_lora_r,
                          head_cls=EXTRA_HEAD_CLS,
                          head_kwargs=head_kwargs)

    def _make(model, processor, device="cuda", **kw):
        vlm = LangQueryTokVLM(model, processor, device,
                                n_qtok=own.uniq4_qtok,
                                lora_r=own.uniq4_lora_r,
                                aux_groups=own.uniq4_aux_groups,
                                aux_seed=own.uniq4_aux_seed, **kw)
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
        (cfg_dir / "uniq4b_setup.json").write_text(json.dumps({
            "n_qtok": own.uniq4_qtok, "lora": "language-only",
            "lora_r": own.uniq4_lora_r,
            # EPR-013 / EPR-016 / EPR-014 flag record (frozen with the run).
            # These read from `head_kwargs`, i.e. what the head is ACTUALLY
            # built with: with `--uniq4-refine-layers 0` no refine module is
            # constructed at all, so recording the CLI's `--uniq4-refine-anneal`
            # as `true` would put a switch in the run record that the arm does
            # not have.  The raw flags are kept below under `cli_flags`.
            "refine_layers": int(head_kwargs.get("n_refine_layers", 0)),
            "refine_anneal": bool(head_kwargs.get("refine_anneal", False)),
            "refine_self_attn": bool(head_kwargs.get("refine_self_attn", False)),
            "refine_thr": (("raw>0" if head_kwargs.get("refine_thr_raw")
                            else "sigmoid<0.5")
                           if head_kwargs.get("n_refine_layers") else None),
            "refine_aux_loss": bool(head_kwargs.get("refine_aux_loss", False)),
            "aux_groups": int(head_kwargs.get("aux_groups", 0)),
            "aux_lambda": float(head_kwargs.get("aux_lambda", 1.0)),
            "aux_seed": (own.uniq4_aux_seed if own.uniq4_aux_groups else None),
            "cli_flags": {
                "uniq4_refine_layers": own.uniq4_refine_layers,
                "uniq4_refine_anneal": own.uniq4_refine_anneal,
                "uniq4_refine_no_self_attn": own.uniq4_refine_no_self_attn,
                "uniq4_refine_thr_raw": own.uniq4_refine_thr_raw,
                "uniq4_refine_no_aux_loss": own.uniq4_refine_no_aux_loss,
                "uniq4_aux_groups": own.uniq4_aux_groups,
                "uniq4_aux_lambda": own.uniq4_aux_lambda,
                "uniq4_aux_seed": own.uniq4_aux_seed,
            },
            "head_cls": getattr(EXTRA_HEAD_CLS, "__name__", "UniQ4Head"),
            "head_kwargs": head_kwargs,
            **EXTRA_SETUP,
            "uniq4_sha256": hashlib.sha256(u4.read_bytes()).hexdigest(),
            "wrapper_sha256": hashlib.sha256(here.read_bytes()).hexdigest(),
        }, indent=2), encoding="utf-8")
    print(f"UNIQ4B qtok={own.uniq4_qtok} lora=language-only "
          f"refine={own.uniq4_refine_layers} aux_groups={own.uniq4_aux_groups} "
          f"head={getattr(EXTRA_HEAD_CLS, '__name__', 'UniQ4Head')} "
          "(seams installed)", flush=True)

    from q3vl.whereb.scripts.run_amort_arm import main as base_main

    return base_main(rest)


if __name__ == "__main__":
    raise SystemExit(main())
