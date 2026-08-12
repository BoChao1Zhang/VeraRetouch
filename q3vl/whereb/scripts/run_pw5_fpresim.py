"""P-W5: is there word-specific localisation in the similarity channel?

The second readout source, and -- more to the point -- **plan C's own P0**.
PROPOSAL section 7 makes C's repaired version conditional on scoring in "the
merger output (2560-d, H/32) or the W_v-projected space", and P-W5 is the only
card that measures whether *that* basis carries word-specific spatial signal.  So
this is not a consolation probe after the attention arm died; it is the
feasibility reading for the fallback.

Mechanism: the merger output is what the LLM actually receives for each image
cell, and ``tie_word_embeddings=true`` (verified in checkpoint-4976's
``config.json``) means the token embedding matrix doubles as the unembedding.  So
``merger_out[cell] . E[noun]`` is a defensible similarity field.

**The basis caveat, stated honestly rather than hidden** (PROPOSAL section 4's own
instruction): the merger output lives in the LLM's *input* space, while an
unembedding is normally applied to a *post-36-layer, post-final-norm* hidden
state.  Tying makes the two use one matrix; it does not make them one space.
Both readings are therefore reported side by side and neither is called the
right one:

* ``dot``    -- raw ``merger_out . E[noun]`` (the logit-lens reading);
* ``cosine`` -- both sides length-normalised (the representation-similarity reading).

Criteria (registered): word-specific delta = soft-IoU(target noun field) −
soft-IoU(control word field), **paired within the same image**, > 0 with p < 0.05.
The centre-prior column and the ``a/(2-a)`` random floor are mandatory (E1 review
R5: without the floor an absolute soft-IoU on this evaluation set is unreadable).
No veto power -- this is an intelligence card.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

#: Function words and scope vocabulary that carry no referent.  Deliberately a
#: fixed list rather than a POS tagger: a tunable extractor on the query side is
#: the knob that lets one enumerate to a positive result.
STOP = {
    "the", "a", "an", "and", "or", "of", "in", "on", "at", "with", "its", "their",
    "his", "her", "together", "nearby", "surrounding", "central", "centre", "center",
    "left", "right", "upper", "lower", "top", "bottom", "middle", "front", "back",
    "subject", "edit", "scope", "stays", "within", "including", "includes", "area",
    "region", "side", "part", "parts", "seated", "standing", "perched", "reading",
    "flowering", "large", "small", "big", "little", "whole", "entire", "overall",
    "this", "that", "these", "those", "is", "are", "was", "were", "be", "it",
}

#: Control nouns with no referent in a photo-retouching corpus.  Used alongside
#: the stronger cross-image control (a real subject noun from a *different*
#: image), which matches the target distribution exactly.
IRRELEVANT_NOUNS = (
    "keyboard", "volcano", "penguin", "spreadsheet", "trombone", "asteroid",
    "vaccine", "algebra", "podcast", "escalator", "parliament", "hurricane",
)


def subject_nouns(where_text: str, max_n: int = 4) -> list[str]:
    """Head nouns of the ``subject:`` clause of a ``<where>`` span."""
    t = where_text.lower()
    m = re.search(r"subject\s*:\s*(.*?)(?:;|$)", t, flags=re.S)
    clause = m.group(1) if m else t
    out: list[str] = []
    for phrase in re.split(r",| and ", clause):
        words = [w for w in re.findall(r"[a-z]+", phrase) if w not in STOP and len(w) > 2]
        if words:
            # English NPs are head-final ("the glass vase" -> "vase")
            head = words[-1]
            if head not in out:
                out.append(head)
    return out[:max_n]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="/home/bc/data/runs/q3vl_base_sft_20260804/checkpoint-4976")
    ap.add_argument("--gt-export", required=True, help="reuse E1's GT merged grids")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="V_where")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=20260810)
    args = ap.parse_args(argv)

    t0 = time.time()
    from transformers import AutoProcessor

    from q3vl.train.modeling import load_model
    from q3vl.whereb.attnread import merged_grid
    from q3vl.whereb.attnprobe import field_scores, paired_wilcoxon
    from q3vl.whereb.data import open_dataset
    from q3vl.whereb.metrics import center_prior_field

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    gtx = Path(args.gt_export)

    proc = AutoProcessor.from_pretrained(args.checkpoint)
    tok = proc.tokenizer
    model = load_model(args.checkpoint, attn_implementation="eager",
                       dtype="bfloat16").to(args.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tied = bool(getattr(model.config, "tie_word_embeddings", False))
    E = model.get_input_embeddings().weight.detach()
    print(f"tie_word_embeddings={tied}  embedding {tuple(E.shape)}", flush=True)

    ds, _ = open_dataset(args.split, need_mask=False)
    rows = ds.meta_rows()
    local = [i for i, r in enumerate(rows) if r.get("render_mode") == "local"]
    if args.limit:
        local = local[: args.limit]

    # word -> embedding (multi-token words are mean-pooled; recorded per word)
    wcache: dict[str, torch.Tensor] = {}
    wtoks: dict[str, int] = {}

    def wemb(word: str) -> torch.Tensor:
        if word not in wcache:
            ids = tok(" " + word, add_special_tokens=False)["input_ids"]
            wtoks[word] = len(ids)
            v = E[torch.tensor(ids, device=E.device)].float().mean(dim=0)
            wcache[word] = v
        return wcache[word]

    # first pass: collect subject nouns so cross-image controls can be drawn
    per: list[dict[str, Any]] = []
    for i in local:
        rec = ds.record(i)
        sid = rec.get("sample_id") or ds.refs[i].sample_id
        nouns = subject_nouns(rec.get("where", ""))
        if nouns and (gtx / "gt" / f"{sid}.npy").exists():
            per.append({"idx": i, "sample_id": sid, "nouns": nouns})
    print(f"local samples with parsed nouns + GT: {len(per)}", flush=True)
    all_nouns = sorted({n for p in per for n in p["nouns"]})
    print(f"distinct subject nouns: {len(all_nouns)}  e.g. {all_nouns[:15]}", flush=True)

    results: list[dict[str, Any]] = []
    for n_i, p in enumerate(per):
        sample = ds[p["idx"]]
        sid = p["sample_id"]
        gh, gw = merged_grid(sample.geometry.out_h, sample.geometry.out_w)
        gt = np.load(gtx / "gt" / f"{sid}.npy").astype(np.float64).reshape(-1)

        enc = proc.image_processor(images=[sample.image], do_resize=False,
                                   return_tensors="pt")
        with torch.no_grad():
            feats, _ = model.model.get_image_features(
                enc["pixel_values"].to(args.device, torch.bfloat16),
                enc["image_grid_thw"].to(args.device))
        f = feats[0] if isinstance(feats, (list, tuple)) else feats
        f = f.reshape(-1, f.shape[-1]).float()                  # (n_img, 2560)
        if f.shape[0] != gh * gw:
            print(f"  !! {sid}: {f.shape[0]} cells != {gh*gw}", flush=True)
            continue

        # high-norm outlier cells (Registers 2309.16588), same MAD rule as E1
        nrm = f.norm(dim=-1).cpu().numpy().astype(np.float64)
        med = np.median(nrm)
        mad = np.median(np.abs(nrm - med))
        outlier = (nrm > med + 3.0 * mad) if mad > 0 else np.zeros_like(nrm, bool)
        valid_all = np.ones(gh * gw, dtype=bool)
        valid_ex = ~outlier
        if valid_ex.sum() < 4 or (gt[valid_ex] > 0.5).sum() < 1:
            valid_ex = valid_all

        fn = torch.nn.functional.normalize(f, dim=-1)

        def fields(word: str) -> dict[str, np.ndarray]:
            e = wemb(word).to(f.device)
            dot = (f @ e).cpu().numpy().astype(np.float64)
            cos = (fn @ torch.nn.functional.normalize(e, dim=0)).cpu().numpy().astype(np.float64)
            return {"dot": dot, "cosine": cos}

        # controls: (1) a subject noun from a DIFFERENT image, (2) irrelevant noun
        own = set(p["nouns"])
        pool = [w for w in all_nouns if w not in own]
        cross = str(rng.choice(pool)) if pool else IRRELEVANT_NOUNS[0]
        irrel = str(rng.choice(IRRELEVANT_NOUNS))
        target = p["nouns"][0]

        prior = center_prior_field(gh, gw).numpy().astype(np.float64).reshape(-1)
        a = float((gt[valid_all] > 0.5).sum() / valid_all.sum())
        row: dict[str, Any] = {
            "sample_id": sid, "grid": [gh, gw], "nouns": p["nouns"],
            "target": target, "control_cross": cross, "control_irrelevant": irrel,
            "n_outlier_cells": int(outlier.sum()),
            "area_frac": a, "random_floor": a / (2 - a) if a < 1 else 1.0,
        }
        for vname, v in (("all", valid_all), ("excl_outlier", valid_ex)):
            row[f"center_prior_{vname}"] = float(
                field_scores(prior[None, None, :], gt, v)[0, 0])
            for basis in ("dot", "cosine"):
                for label, word in (("target", target), ("cross", cross),
                                    ("irrelevant", irrel)):
                    fld = fields(word)[basis]
                    row[f"{basis}_{label}_{vname}"] = float(
                        field_scores(fld[None, None, :], gt, v)[0, 0])
        results.append(row)
        if (n_i + 1) % 50 == 0:
            print(f"  [{n_i+1}/{len(per)}] {time.time()-t0:.0f}s", flush=True)

    # ---- criteria -------------------------------------------------------
    def col(k):
        return np.asarray([r[k] for r in results], dtype=np.float64)

    def agg(x):
        x = np.asarray(x, float)
        return {"n": int(x.size), "mean": float(x.mean()), "median": float(np.median(x)),
                "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90))}

    summary: dict[str, Any] = {
        "card": "P-W5 (F_pre/merger x <where> noun similarity)",
        "veto_power": False,
        "n_samples": len(results),
        "tie_word_embeddings": tied,
        "basis_caveat": (
            "merger output is the LLM INPUT space; an unembedding normally acts on a "
            "post-36-layer, post-final-norm state. tie_word_embeddings makes them share "
            "one matrix, NOT one space. Both readings reported, neither privileged."),
        "multi_token_words": {w: n for w, n in sorted(wtoks.items()) if n > 1},
        "random_floor": agg(col("random_floor")),
        "area_frac": agg(col("area_frac")),
        "readings": {},
    }
    for vname in ("all", "excl_outlier"):
        summary["readings"][vname] = {
            "center_prior": agg(col(f"center_prior_{vname}")),
        }
        for basis in ("dot", "cosine"):
            t_ = col(f"{basis}_target_{vname}")
            c_ = col(f"{basis}_cross_{vname}")
            i_ = col(f"{basis}_irrelevant_{vname}")
            summary["readings"][vname][basis] = {
                "target": agg(t_), "control_cross_image": agg(c_),
                "control_irrelevant": agg(i_),
                "WORD_SPECIFIC_delta_vs_cross": paired_wilcoxon(t_, c_),
                "WORD_SPECIFIC_delta_vs_irrelevant": paired_wilcoxon(t_, i_),
                "vs_center_prior": paired_wilcoxon(t_, col(f"center_prior_{vname}")),
                "vs_random_floor": paired_wilcoxon(t_, col("random_floor")),
            }

    setup = {
        "checkpoint": args.checkpoint, "split": args.split, "seed": args.seed,
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                     text=True, cwd="/home/bc/VeraRetouch").stdout.strip(),
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "wall_seconds": time.time() - t0, "argv": sys.argv,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps({**summary, "per_sample": results, "run_setup": setup}, indent=2),
        encoding="utf-8")

    print("\n=== P-W5 word-specific readings ===")
    for vname, blk in summary["readings"].items():
        print(f"\n-- support: {vname}   centre prior {blk['center_prior']['median']:.4f}  "
              f"random floor {summary['random_floor']['median']:.4f}")
        for basis in ("dot", "cosine"):
            b = blk[basis]
            dc = b["WORD_SPECIFIC_delta_vs_cross"]
            di = b["WORD_SPECIFIC_delta_vs_irrelevant"]
            print(f"   {basis:7s} target={b['target']['median']:.4f} "
                  f"cross={b['control_cross_image']['median']:.4f} "
                  f"irrel={b['control_irrelevant']['median']:.4f}")
            print(f"           word-specific Δ vs cross={dc['delta_median']:+.4f} "
                  f"(p={dc['p_value']:.3g}) | vs irrelevant={di['delta_median']:+.4f} "
                  f"(p={di['p_value']:.3g})")
    print(f"\ndone in {(time.time()-t0)/60:.1f} min -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
