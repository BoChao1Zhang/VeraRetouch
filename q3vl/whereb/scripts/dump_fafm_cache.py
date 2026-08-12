"""One frozen-VLM pass -> everything the FAFM probe needs, after which training is
GPU-cheap and touches no large model.

Proposal B's probe (``RESEARCH_unified-field-prediction_2026-08-10`` section 3.6) needs four
things per sample.  Three of them cost a vision-tower forward, so they are taken
in the **same** pass (protocol 2.3), and the fourth is a ridge solve:

``cstar``   the regression target ``c* = (A^T A + eps I)^-1 A^T y`` on the H/16
            grid -- the canonical coarse coordinate the flow transports to.
``sem``     ``B @ F_pre`` on H/16, 64 channels -- the spatially-aligned visual
            condition ``V`` (concat channel, Marigold mechanism).
``sim``     word-region similarity fields on H/32, one per subject noun plus a
            sentence-pooled channel -- the condition ``S``.
``guide_q`` the Rec.709 luma guide at **quarter** resolution.

Why the quarter-resolution guide, and why it is not a corner cut
---------------------------------------------------------------
The FAFM loss metric is ``Lambda = lambda I + (1-lambda) A^T A / ||A||^2``; its
second term is ``||A_I (c_hat - c*)||^2``, i.e. it needs to *apply* ``A_I``, not to
store it.  Storing ``A^T A`` would be 9.4 MB/sample (188 GB for 20k) -- applying
``A_I`` to the residual costs one guided upsample.  Measured on 12 samples,
solving/rendering through a half- or quarter-resolution guide changes the
full-resolution reconstruction soft-IoU by 5e-5 / 2.8e-4 respectively (0.99695
full, 0.99690 half, 0.99667 quarter), so the reduced guide carries the metric
faithfully at 1/16 the memory.  ``cstar`` itself is solved at **half** resolution
for the same reason (3.7x faster, 5e-5 cost), and the equivalence is re-measured
on this run's own sample and written into the manifest rather than assumed.

Noun source (decision, recorded in NOTES)
-----------------------------------------
Subject nouns are parsed from the **instruction**, not from the GT ``<where>``
text.  At inference the model generates ``<where>`` itself, so ``<where>`` nouns
are legitimate in production -- but in a probe the GT ``<where>`` is a label, and
scoring a conditioning channel built from it would overstate the arm.  The
instruction is available at inference with no leak, so it is the conservative
default; ``<where>``-derived nouns are stored alongside as a record-only column.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

RIDGE_EPS = 1e-3
SOLVE_DIV = 2          # c* solved through a half-resolution guide
GUIDE_DIV = 4          # stored guide for the training-time Lambda term
N_SIM = 4              # top-3 nouns + 1 sentence-pooled channel

STOP = set("""
a an the this that these those and or but if then than so as of in on at to for from by with
without into onto over under above below up down out off again further once here there all any
both each few more most other some such no nor not only own same too very can will just should
now make makes making made let lets please want wants need needs give gives add adds
adjust adjusts change changes set sets turn turns keep keeps leave leaves put puts
more less slightly bit little much very really quite somewhat image photo picture shot
color colour tone tones look looks style feel feeling area region part parts whole entire
brighter darker warmer cooler lighter softer stronger sharper
""".split())


def content_nouns(text: str, max_n: int = 3) -> list[str]:
    """Head-final content words of the instruction.  No POS tagger, fixed stop list.

    Deliberately the same shape as ``run_pw5_fpresim.subject_nouns`` (which parses
    the ``subject:`` clause of a ``<where>`` span) so the two are comparable, but
    it reads the instruction instead.
    """
    out: list[str] = []
    for phrase in re.split(r",| and | of ", (text or "").lower()):
        words = [w for w in re.findall(r"[a-z]+", phrase)
                 if w not in STOP and len(w) > 2]
        if words:
            head = words[-1]
            if head not in out:
                out.append(head)
    return out[:max_n]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint",
                    default="/home/bc/data/runs/q3vl_base_sft_20260804/checkpoint-4976")
    ap.add_argument("--basis",
                    default="/mnt/nfs-ro/bc/data/datasets/where_a-20260805/basis/"
                            "BA-3-Joint/B.npy")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=20260811)
    ap.add_argument("--equiv-check", type=int, default=24,
                    help="samples on which the reduced-resolution solve is "
                         "re-verified against the full-resolution one")
    args = ap.parse_args(argv)

    t0 = time.time()
    from transformers import AutoProcessor, AutoTokenizer

    from q3vl.train.modeling import load_model
    from q3vl.where.fpre import FPreHook, grid_from_geometry
    from q3vl.where.upsample import area_resize, luma_guide
    from q3vl.whereb.attnread import merged_grid
    from q3vl.whereb.data import open_dataset
    from q3vl.whereb.hiddens import resolve_visual
    from q3vl.whereb.metrics import soft_iou_value
    from q3vl.whereb.scripts.run_amort_e5 import load_embeddings
    from q3vl.whereb.unifield import GuidedOp, gram_exact, load_families

    out = Path(args.out)
    (out / "cache").mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    B = torch.from_numpy(np.load(args.basis)).float()
    proc = AutoProcessor.from_pretrained(args.checkpoint)
    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    E = load_embeddings(args.checkpoint)
    model = load_model(args.checkpoint, attn_implementation="eager",
                       dtype="bfloat16").to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    visual = resolve_visual(model)
    print(f"model + embeddings {tuple(E.shape)} ready ({time.time()-t0:.0f}s)", flush=True)

    ds, ds_facts = open_dataset(args.split, need_mask=True)
    rows = ds.meta_rows()
    local = [i for i, r in enumerate(rows) if r.get("render_mode") == "local"]
    pick = sorted(rng.choice(len(local), size=min(args.n, len(local)),
                             replace=False).tolist())
    idxs = [local[i] for i in pick]
    fams = load_families(ds, idxs)
    print(f"{args.split}: {len(local)} local, taking {len(idxs)} "
          f"({time.time()-t0:.0f}s)", flush=True)

    wcache: dict[str, torch.Tensor] = {}

    def wemb(word: str) -> torch.Tensor:
        if word not in wcache:
            ids = tok(" " + word, add_special_tokens=False)["input_ids"]
            wcache[word] = E[torch.tensor(ids)].mean(dim=0)
        return wcache[word]

    def solve_cstar(guide_hi: torch.Tensor, gt_hi: torch.Tensor,
                    gh: int, gw: int, div: int) -> torch.Tensor:
        H, W = gt_hi.shape[-2:]
        if div > 1:
            g = area_resize(guide_hi, (H // div, W // div))
            y = area_resize(gt_hi[None, None], (H // div, W // div))[0, 0]
        else:
            g, y = guide_hi, gt_hi
        op = GuidedOp(g.to(dev), gh, gw)
        G = gram_exact(op).double()
        b = op.adjoint(y.to(dev).double().reshape(-1)).reshape(-1).double()
        eye = torch.eye(op.n_low, dtype=torch.float64, device=G.device)
        c = torch.linalg.solve(G + RIDGE_EPS * eye, b)
        del op, G, eye
        return c

    manifest: list[dict] = []
    equiv: list[float] = []
    cstar_cells: list[np.ndarray] = []
    n_fail = 0
    for n, i in enumerate(idxs):
        try:
            s = ds[i]
            sid = s.sample_id
            gh16, gw16 = grid_from_geometry(s.geometry.out_h, s.geometry.out_w)
            gh32, gw32 = merged_grid(s.geometry.out_h, s.geometry.out_w)

            enc = proc.image_processor(images=[s.image], do_resize=False,
                                       return_tensors="pt")
            grid_thw = enc["image_grid_thw"]
            hook = FPreHook(visual)
            with torch.no_grad(), hook.attached():
                feats, _ = model.model.get_image_features(
                    enc["pixel_values"].to(dev, torch.bfloat16), grid_thw.to(dev))
            fpre = hook.split(grid_thw)[0].float()
            if tuple(fpre.shape[:2]) != (gh16, gw16):
                raise RuntimeError(f"F_pre grid {tuple(fpre.shape[:2])} != {(gh16,gw16)}")
            sem = fpre.reshape(-1, fpre.shape[-1]).cpu() @ B.T          # (P16,64)

            m = feats[0] if isinstance(feats, (list, tuple)) else feats
            m = m.reshape(-1, m.shape[-1]).float().cpu()                # (P32,2560)
            if m.shape[0] != gh32 * gw32:
                raise RuntimeError(f"merger {m.shape[0]} != {gh32*gw32}")

            nouns = content_nouns(s.instruction, max_n=N_SIM - 1)
            mn = torch.nn.functional.normalize(m, dim=-1)
            chans = []
            for w in nouns:
                e = wemb(w)
                chans.append(mn @ torch.nn.functional.normalize(e, dim=0))
            while len(chans) < N_SIM - 1:
                chans.append(torch.zeros(m.shape[0]))
            ids = tok(s.instruction or "", add_special_tokens=False)["input_ids"]
            sent = (E[torch.tensor(ids)].mean(dim=0) if ids
                    else torch.zeros(E.shape[-1]))
            chans.append(mn @ torch.nn.functional.normalize(sent, dim=0))
            sim = torch.stack(chans, dim=-1)                            # (P32, N_SIM)

            gt_hi = s.mask_target_hi().double()
            guide_hi = luma_guide(s.image_tensor().double().unsqueeze(0))
            c = solve_cstar(guide_hi, gt_hi, gh16, gw16, SOLVE_DIV)

            if len(equiv) < args.equiv_check:
                c_full = solve_cstar(guide_hi, gt_hi, gh16, gw16, 1)
                op_f = GuidedOp(guide_hi.to(dev), gh16, gw16)
                a = soft_iou_value(op_f.render(c).cpu(), gt_hi)
                b_ = soft_iou_value(op_f.render(c_full).cpu(), gt_hi)
                equiv.append(b_ - a)
                del op_f

            H, W = gt_hi.shape[-2:]
            guide_q = area_resize(guide_hi, (max(8, H // GUIDE_DIV),
                                             max(8, W // GUIDE_DIV)))[0, 0]
            gt16 = area_resize(gt_hi[None, None], (gh16, gw16))[0, 0]

            np.savez(out / "cache" / f"{sid}.npz",
                     cstar=c.cpu().numpy().astype(np.float32),
                     sem=sem.numpy().astype(np.float16),
                     sim=sim.numpy().astype(np.float16),
                     guide_q=guide_q.numpy().astype(np.float16),
                     gt16=gt16.numpy().astype(np.float16))
            cstar_cells.append(c.cpu().numpy())
            manifest.append({
                "sample_id": sid, "grid16": [gh16, gw16], "grid32": [gh32, gw32],
                "out_h": s.geometry.out_h, "out_w": s.geometry.out_w,
                "guide_q_hw": list(guide_q.shape),
                "family": fams.get(sid, "unknown"),
                "instruction": s.instruction, "nouns": nouns,
                "winner_confidence": s.meta.get("winner_confidence"),
                "build": s.meta.get("build"),
                "source_image_id": s.meta.get("source_image_id"),
            })
        except Exception as exc:                                 # noqa: BLE001
            n_fail += 1
            if n_fail <= 5:
                print(f"  FAIL {i}: {type(exc).__name__}: {exc}", flush=True)
        if (n + 1) % 200 == 0:
            torch.cuda.empty_cache()
            el = time.time() - t0
            print(f"  [{n+1}/{len(idxs)}] {el:.0f}s  "
                  f"eta {el/(n+1)*(len(idxs)-n-1)/60:.0f} min  fails={n_fail}",
                  flush=True)

    allc = np.concatenate(cstar_cells) if cstar_cells else np.zeros(1)
    (out / "manifest.json").write_text(json.dumps({
        "split": args.split, "n": len(manifest), "n_failed": n_fail,
        "seed": args.seed, "ridge_eps": RIDGE_EPS,
        "solve_div": SOLVE_DIV, "guide_div": GUIDE_DIV, "n_sim_channels": N_SIM,
        "basis": args.basis, "checkpoint": args.checkpoint,
        "reduced_solve_equivalence": {
            "n": len(equiv),
            "mean_softiou_full_minus_reduced": float(np.mean(equiv)) if equiv else None,
            "max_softiou_full_minus_reduced": float(np.max(equiv)) if equiv else None,
            "note": "positive = the full-resolution solve is better; this run "
                    "re-measures the shortcut instead of trusting the pilot.",
        },
        "cstar_domain": {
            "arm": "fafm_cstar", "eps": RIDGE_EPS,
            "domain": [float(allc.min()), float(allc.max())],
            "p01": float(np.percentile(allc, 1)),
            "p50": float(np.percentile(allc, 50)),
            "p99": float(np.percentile(allc, 99)),
            "frac_below_0": float((allc < 0).mean()),
            "frac_above_1": float((allc > 1).mean()),
            "advice": "c* is NOT confined to [0,1]. Clamping the COORDINATE "
                      "destroys the ridge solution silently. Render as "
                      "clip(A_I c, 0, 1) -- clip the field, never the coordinate.",
        },
        "noun_source": "instruction (leak-free); <where> nouns deliberately not used",
        "dataset_facts": ds_facts,
        "samples": manifest,
    }, indent=2, default=str), encoding="utf-8")
    print(f"done {len(manifest)} samples ({n_fail} failed) in "
          f"{(time.time()-t0)/60:.1f} min -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
