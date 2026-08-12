"""E3 -- the three-part diagnostic that separates M3 (copying the prior) from M4
(the instruction never reached the head).

M1/M2/M3/M4 all predict the *same* phenotype on W01/W02 -- a systematically
over-covering trivial blob -- but they demand opposite repairs, so P0-E3 asks
three questions of the trained heads themselves:

  (i)   does the output field vary across samples at all, or is it a constant
        shape wearing different sample ids?
  (ii)  is the output better explained by the centre prior / the P-W5 similarity
        field than by the GT?
  (iii) does a *subject swap* move the field (conditioning), and does a
        colour-word flip leave it alone (the pre-registered invariance control)?

Verdict logic (registered in RESEARCH section 4, P0-E3):

  output ~ constant, or corr(output, prior) >> corr(output, GT)   -> M3
  subject-swap difference ~ 0                                    -> M4

CAUTION: the ``antonym`` context is NOT the M4 probe.  It flips only colour
direction words (darker<->brighter) and keeps the subject phrase, so Where's
mask is *supposed* to be invariant to it -- small |delta| is the PASS of an
invariance control, not evidence that conditioning is missing.

Either one landing means P1/P2/P3's change of loss is beside the point and the
repair belongs in the conditioning wiring instead.

What this card adds over the boards already on disk
---------------------------------------------------
``eval_final/per_sample.jsonl`` already carries seven contexts x 400 local
samples for both arms, which answers (iii) at the metric level.  It does *not*
carry the fields, so (i) and (ii) -- which are about the shape of the output,
not its score -- cannot be read off it.  This card dumps ``m_low`` / ``s_low``
per sample per context and does the field-level work, then folds the existing
board rows in for (iii).

Cost: a forward pass over 400 local samples x 3 contexts x 2 arms, no training,
no backward.  It coexists with whatever the queue is running.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

CONTEXTS = ("gt", "antonym", "shuffled")


def agg(x: Any) -> dict[str, float]:
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
            "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90)),
            "min": float(a.min()), "max": float(a.max())}


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float((a @ b) / (na * nb))


def to_common(field: np.ndarray, size: int = 16) -> np.ndarray:
    """Resample one field onto a common ``size x size`` grid.

    Cross-sample variance is only meaningful on a shared support, and the F_pre
    grids differ per image (aspect ratio).  Area interpolation, never a plain
    resize of an overlay -- this is a field-to-field resample, not a viz path.
    """
    t = torch.from_numpy(field.astype(np.float64)).reshape(1, 1, *field.shape)
    return torch.nn.functional.interpolate(
        t, size=(size, size), mode="area")[0, 0].numpy()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="*", default=["W01", "W02"])
    ap.add_argument("--run-root", default="/home/bc/data/runs/where_b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="V_where")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn", default="eager")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pw5-cache", default="/home/bc/data/runs/where_b/amort_cache_20260810")
    ap.add_argument("--checkpoint",
                    default="/home/bc/data/runs/q3vl_base_sft_20260804/checkpoint-4976")
    ap.add_argument("--common-grid", type=int, default=16)
    ap.add_argument("--n-viz", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260810)
    args = ap.parse_args(argv)

    t0 = time.time()
    from q3vl.train.modeling import load_model
    from q3vl.whereb.config import (
        BASIS_ARM, GENCTX_DIR, MODEL_DIR, ORACLE_NAMESPACE, WHERE_A_MASKVIEW_DIR,
        WHERE_A_ORACLE_DIR, arm_config,
    )
    from q3vl.whereb.context import ShuffleIndex
    from q3vl.whereb.data import BatchBuilder, open_dataset
    from q3vl.whereb.fields import load_basis, predict_fields
    from q3vl.whereb.hiddens import FrozenVLM
    from q3vl.whereb.metrics import (
        center_prior_field, grid_boundary_f1, gt_area_k, soft_iou_value, topk_mask,
    )
    from q3vl.whereb.model import WhereBModel
    from q3vl.whereb.prefetch import SamplePrefetcher, prefetch_warmers
    from q3vl.train.collator import Sft2SegCollator
    from q3vl.train.modeling import load_processor

    out_dir = Path(args.out)
    for sub in ("viz", "config", "logs", "fields"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    processor, special_ids = load_processor(str(MODEL_DIR), 2048)
    collator = Sft2SegCollator(processor, max_length=2048, system_prompt=None)
    vlm_model = load_model(args.checkpoint, attn_implementation=args.attn,
                          dtype=args.dtype).to(args.device).eval()
    vlm = FrozenVLM(vlm_model, processor, device=args.device)
    print(f"VLM loaded ({time.time()-t0:.0f}s)", flush=True)

    from q3vl.whereb.stores import GenContextStore, OracleStore
    basis = load_basis(BASIS_ARM).to(args.device)
    oracle_base = Path(WHERE_A_ORACLE_DIR) / BASIS_ARM / ORACLE_NAMESPACE

    ds, ds_info = open_dataset(args.split, limit=args.limit,
                               maskview_root=WHERE_A_MASKVIEW_DIR)
    oracle = OracleStore(oracle_base / args.split)
    genctx = GenContextStore(Path(GENCTX_DIR) / args.split)
    shuffle_index = ShuffleIndex(ds.shuffle_records(), seed=0)

    # ---- the similarity field E5/P-W5 use, for the "is it copying?" column ---
    pw5: dict[str, np.ndarray] = {}
    cachedir = Path(args.pw5_cache)
    if (cachedir / "manifest.json").exists():
        from transformers import AutoTokenizer

        from q3vl.whereb.scripts.run_amort_e5 import load_embeddings
        from q3vl.whereb.scripts.run_pw5_fpresim import subject_nouns
        tok = AutoTokenizer.from_pretrained(args.checkpoint)
        E = load_embeddings(args.checkpoint)
        man = json.loads((cachedir / "manifest.json").read_text())
        idx_of = {r.sample_id: i for i, r in enumerate(ds.refs)}
        for rec in man["samples"]:
            sid = rec["sample_id"]
            if sid not in idx_of:
                continue
            nouns = subject_nouns(ds.record(idx_of[sid]).get("where", "") or "")
            if not nouns:
                continue
            gh32, gw32 = rec["grid32"]
            m = torch.from_numpy(
                np.load(cachedir / "cache" / f"{sid}.npz")["merger_out"]).float()
            ids = tok(" " + nouns[0], add_special_tokens=False)["input_ids"]
            e = E[torch.tensor(ids)].mean(dim=0)
            pw5[sid] = (m @ e).reshape(gh32, gw32).numpy().astype(np.float64)
        print(f"P-W5 similarity fields: {len(pw5)} ({time.time()-t0:.0f}s)", flush=True)

    all_arms: dict[str, Any] = {}
    for arm in args.arms:
        cfg = arm_config(arm)
        ckpt = Path(args.run_root) / arm / "where_b_final.pt"
        if not ckpt.exists():
            print(f"!! {arm}: {ckpt} missing, skipping", flush=True)
            continue
        model = WhereBModel(cfg).to(args.device)
        state = torch.load(ckpt, map_location=args.device, weights_only=False)
        sd = state.get("model", state) if isinstance(state, dict) else state
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"   {arm} load_state_dict missing={len(missing)} "
                  f"unexpected={len(unexpected)}", flush=True)
        model.eval()
        builder = BatchBuilder(collator, vlm, basis, cfg, oracle=oracle,
                               genctx=genctx, shuffle_index=shuffle_index,
                               device=args.device)
        print(f"{arm}: loaded {ckpt} ({time.time()-t0:.0f}s)", flush=True)

        per_ctx: dict[str, dict[str, np.ndarray]] = {}
        meta: dict[str, dict[str, Any]] = {}
        for mode in CONTEXTS:
            fields: dict[str, np.ndarray] = {}
            idx = list(range(len(ds)))
            chunks = ([(i, mode) for i in idx[j:j + args.batch_size]]
                      for j in range(0, len(idx), args.batch_size))
            supply = SamplePrefetcher(ds, chunks, workers=6,
                                      warm=prefetch_warmers(builder))
            n = 0
            for _c, built in supply:
                sel = []
                for s in built:
                    if s.is_global:
                        continue
                    if mode == "shuffled" and shuffle_index.partner_of(s.sample_id) is None:
                        continue
                    sel.append(s)
                if not sel:
                    continue
                batch = builder.build(sel, [mode] * len(sel))
                with torch.no_grad():
                    out = model(**batch.inputs)
                for j, tgt in enumerate(batch.targets):
                    params = {k: v.float() for k, v in out.select(j).items()}
                    gh, gw = tgt["grid_h"], tgt["grid_w"]
                    f = predict_fields(
                        tgt["phi_dir"].float(), params, cfg.readout, gh, gw,
                        guide_hi=tgt["guide_hi"].float(), up_cfg=cfg.upsample,
                        require_dtype=torch.float32,
                    )
                    sid = tgt["sample_id"]
                    fields[sid] = f["m_low"].detach().reshape(gh, gw).float().cpu().numpy()
                    if mode == "gt":
                        meta[sid] = {
                            "grid": [gh, gw],
                            "gt": tgt["mask_low"].detach().reshape(gh, gw).float().cpu().numpy(),
                            "s_low": f["s_low"].detach().float().cpu().numpy(),
                            "source_image_id": tgt["meta"].get("source_image_id"),
                            "instruction": tgt["meta"].get("instruction"),
                        }
                    n += 1
            per_ctx[mode] = fields
            print(f"  {arm}/{mode}: {n} fields ({time.time()-t0:.0f}s)", flush=True)

        np.savez_compressed(out_dir / "fields" / f"{arm}_gt.npz",
                            **{k: v for k, v in per_ctx["gt"].items()})
        all_arms[arm] = _analyse(arm, per_ctx, meta, pw5, args,
                                 center_prior_field, gt_area_k, topk_mask,
                                 soft_iou_value, grid_boundary_f1)
        try:
            _viz(out_dir / "viz", arm, per_ctx, meta, args.n_viz)
        except Exception as exc:                                # pragma: no cover
            print(f"  viz failed: {exc}", flush=True)
        del model
        torch.cuda.empty_cache()

    summary = {
        "card": "E3 (W01/W02 output-field triage: M3 copy-the-prior vs M4 no conditioning)",
        "split": args.split,
        "contexts": list(CONTEXTS),
        "common_grid": args.common_grid,
        "arms": all_arms,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "config" / "run_setup.json").write_text(json.dumps({
        "arms": args.arms, "run_root": args.run_root, "split": args.split,
        "checkpoint": args.checkpoint, "device": args.device, "dtype": args.dtype,
        "attn": args.attn, "seed": args.seed, "dataset": ds_info,
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                     text=True, cwd="/home/bc/VeraRetouch").stdout.strip(),
        "python": platform.python_version(), "torch": torch.__version__,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== E3 ===", flush=True)
    for arm, a in all_arms.items():
        print(f"[{arm}]", flush=True)
        print(f"  (i)  cross-sample mean pairwise corr of output   "
              f"{a['i_constant_output']['mean_pairwise_corr_across_samples']:.4f}"
              f"   (GT reference {a['i_constant_output']['mean_pairwise_corr_gt']:.4f})",
              flush=True)
        print(f"       per-cell cross-sample std / GT std          "
              f"{a['i_constant_output']['std_ratio_output_over_gt']:.4f}", flush=True)
        print(f"  (ii) corr(out, centre prior) {a['ii_what_explains_it']['corr_centre']['median']:.4f}"
              f" | corr(out, GT) {a['ii_what_explains_it']['corr_gt']['median']:.4f}"
              f" | corr(out, P-W5) {a['ii_what_explains_it']['corr_pw5'].get('median', float('nan')):.4f}",
              flush=True)
        print(f"  (iii) paired |gt - antonym| {a['iii_conditioning']['gt_vs_antonym_L1']['median']:.5f}"
              f" | |gt - shuffled| {a['iii_conditioning']['gt_vs_shuffled_L1']['median']:.5f}"
              f" | identical-field rate {a['iii_conditioning']['frac_antonym_identical']:.3f}",
              flush=True)
        print(f"  VERDICT {json.dumps(a['verdict'])}", flush=True)
    print(f"done in {(time.time()-t0)/60:.1f} min -> {out_dir}", flush=True)
    return 0


def _analyse(arm, per_ctx, meta, pw5, args, center_prior_field, gt_area_k,
             topk_mask, soft_iou_value, grid_boundary_f1) -> dict[str, Any]:
    G = args.common_grid
    sids = sorted(meta)
    gt_fields = per_ctx["gt"]

    # ---- (i) is the output a constant shape? -------------------------------
    out_c = np.stack([to_common(gt_fields[s], G) for s in sids if s in gt_fields])
    gt_c = np.stack([to_common(meta[s]["gt"], G) for s in sids if s in gt_fields])

    def mean_pairwise_corr(stack: np.ndarray, n_pairs: int = 4000) -> float:
        rng = np.random.default_rng(args.seed)
        n = stack.shape[0]
        flat = stack.reshape(n, -1)
        flat = flat - flat.mean(axis=1, keepdims=True)
        nrm = np.linalg.norm(flat, axis=1)
        ok = nrm > 1e-12
        vals = []
        for _ in range(n_pairs):
            i, j = rng.integers(0, n, 2)
            if i == j or not (ok[i] and ok[j]):
                continue
            vals.append(float(flat[i] @ flat[j] / (nrm[i] * nrm[j])))
        return float(np.mean(vals)) if vals else float("nan")

    i_block = {
        "n": int(out_c.shape[0]),
        # 1.0 = every sample produces the same shape.  The GT column is the
        # reference: real targets are not identical either, so the output number
        # is only readable next to it.
        "mean_pairwise_corr_across_samples": mean_pairwise_corr(out_c),
        "mean_pairwise_corr_gt": mean_pairwise_corr(gt_c),
        "std_ratio_output_over_gt": float(
            out_c.std(axis=0).mean() / (gt_c.std(axis=0).mean() + 1e-12)),
        "per_sample_field_std": agg([float(f.std()) for f in out_c]),
        "per_sample_field_mean": agg([float(f.mean()) for f in out_c]),
    }

    # ---- (ii) what explains the output better? -----------------------------
    cc, cg, cp, iou_c, iou_g = [], [], [], [], []
    for s in sids:
        if s not in gt_fields:
            continue
        f = gt_fields[s]
        gh, gw = meta[s]["grid"]
        gt = meta[s]["gt"]
        cprior = center_prior_field(gh, gw).double().numpy()
        cc.append(corr(f, cprior))
        cg.append(corr(f, gt))
        if s in pw5:
            cp.append(corr(to_common(f, G), to_common(pw5[s], G)))
        k = gt_area_k(torch.from_numpy(gt))
        gtb = (torch.from_numpy(gt) > 0.5).double()
        iou_g.append(soft_iou_value(topk_mask(torch.from_numpy(f).double(), k), gtb))
        iou_c.append(soft_iou_value(topk_mask(torch.from_numpy(cprior), k), gtb))
    ii_block = {
        "corr_centre": agg(cc), "corr_gt": agg(cg), "corr_pw5": agg(cp),
        "softiou_output": agg(iou_g), "softiou_centre_prior": agg(iou_c),
        "corr_centre_minus_corr_gt": agg(np.asarray(cc) - np.asarray(cg)),
    }

    # ---- (iii) does the instruction change anything? -----------------------
    def diff(a_mode: str, b_mode: str) -> tuple[list[float], float]:
        d, ident = [], 0
        fa, fb = per_ctx[a_mode], per_ctx[b_mode]
        common = [s for s in sids if s in fa and s in fb]
        for s in common:
            v = float(np.abs(fa[s] - fb[s]).mean())
            d.append(v)
            ident += int(v < 1e-6)
        return d, (ident / len(common) if common else float("nan"))

    d_ant, frac_ant = diff("gt", "antonym")
    d_shf, _ = diff("gt", "shuffled")
    scale = float(np.mean([np.abs(gt_fields[s]).mean() for s in sids if s in gt_fields]))
    iii_block = {
        "gt_vs_antonym_L1": agg(d_ant),
        "gt_vs_shuffled_L1": agg(d_shf),
        "field_scale_mean_abs": scale,
        "gt_vs_antonym_L1_relative": float(np.median(d_ant) / (scale + 1e-12)),
        "gt_vs_shuffled_L1_relative": float(np.median(d_shf) / (scale + 1e-12)),
        "frac_antonym_identical": frac_ant,
    }

    verdict = {
        "M3_copies_geometric_prior": bool(
            np.median(cc) > np.median(cg)),
        "M3_output_near_constant": bool(
            i_block["mean_pairwise_corr_across_samples"] > 0.9
            and i_block["mean_pairwise_corr_across_samples"]
            > i_block["mean_pairwise_corr_gt"] + 0.2),
        # NOT an M4 probe: the antonym context flips only colour-direction words
        # and keeps the subject, so small |delta| is the PASS of a pre-registered
        # INVARIANCE control (metrics.py:282, threshold 0.05).  M4 has to be read
        # off the subject-swapping contexts (shuffled / irrelevant / fixed_phrase).
        "antonym_invariance_pass": bool(
            iii_block["gt_vs_antonym_L1_relative"] < 0.05),
        "conditioning_responds_to_subject_swap": bool(
            iii_block["gt_vs_shuffled_L1_relative"] > 0.05),
        "corr_centre_median": float(np.median(cc)),
        "corr_gt_median": float(np.median(cg)),
    }
    return {"i_constant_output": i_block, "ii_what_explains_it": ii_block,
            "iii_conditioning": iii_block, "verdict": verdict}


def _viz(viz_dir: Path, arm: str, per_ctx, meta, n: int) -> None:
    """Output under two opposite instructions, next to GT and the centre prior.

    Fixed 0..1 colour scale for every mask panel -- no per-image min-max
    (2026-08-05 discipline); the whole point is that the panels look the same.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from q3vl.whereb.metrics import center_prior_field, gt_area_k, soft_iou_value, topk_mask

    sids = [s for s in sorted(meta) if s in per_ctx["gt"] and s in per_ctx["antonym"]]
    scored = []
    for s in sids:
        gt = torch.from_numpy(meta[s]["gt"]).double()
        k = gt_area_k(gt)
        f = torch.from_numpy(per_ctx["gt"][s]).double()
        scored.append((soft_iou_value(topk_mask(f, k), (gt > 0.5).double()), s))
    scored.sort()
    picks = [("failure", s) for _, s in scored[:n]] + [("success", s) for _, s in scored[-n:]]
    for tag, s in picks:
        gh, gw = meta[s]["grid"]
        gt = meta[s]["gt"]
        fig, ax = plt.subplots(1, 5, figsize=(18, 3.2))
        for a in ax:
            a.set_xticks([]); a.set_yticks([])
        ax[0].imshow(gt, cmap="gray", vmin=0, vmax=1); ax[0].set_title("GT", fontsize=9)
        ax[1].imshow(per_ctx["gt"][s], cmap="magma", vmin=0, vmax=1)
        ax[1].set_title("output | real instruction", fontsize=9)
        ax[2].imshow(per_ctx["antonym"][s], cmap="magma", vmin=0, vmax=1)
        ax[2].set_title("output | ANTONYM instruction", fontsize=9)
        d = np.abs(per_ctx["gt"][s] - per_ctx["antonym"][s])
        ax[3].imshow(d, cmap="inferno", vmin=0, vmax=1)
        ax[3].set_title(f"|difference|  max={d.max():.4f}", fontsize=9)
        ax[4].imshow(center_prior_field(gh, gw).numpy(), cmap="viridis")
        ax[4].set_title("centre prior", fontsize=9)
        fig.suptitle(f"{arm} {tag}  {s}", fontsize=10)
        fig.tight_layout()
        fig.savefig(viz_dir / f"{tag}_{arm}_{s}.png", dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
