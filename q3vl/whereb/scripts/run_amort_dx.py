"""DX-1..6: zero-training attribution of analytic-family edge dirtiness.

Splits the dirty-edge total D_total across three mechanisms registered in
RESEARCH_analytic-edge-quality_2026-08-11.md §2:

    (a) the head carries no structural bias      -> DX-3
    (b) guided upsampling copies image texture   -> DX-1, DX-2
    (c) the loss cannot see it                   -> DX-4
    plus DX-5 (family separability, the floor under every gating scheme)
    and DX-6 (oracle routing headroom).

Nothing here trains anything.  DX-1 does not even involve the model: it pushes
the **GT** coarse field through the arms' own guided upsampler, so whatever
dirtiness comes out is attributable to the operator alone.

The s-space round trip used for the GT field is deliberately linear
(``s = 6(m - 1/2)``, exactly invertible, and inside the operator's declared
(-3, 3) domain), so any deviation from the GT at delivery resolution is the
operator's doing and not an artefact of the mapping.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ANALYTIC = ("radial", "linear", "band")


def _med(xs) -> float | None:
    v = [float(x) for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--p1-run", default="/home/bc/data/runs/where_b/amort_P1_20260810")
    ap.add_argument("--p3-run", default="/home/bc/data/runs/where_b/amort_P3prime_20260810")
    ap.add_argument("--checkpoint",
                    default="/home/bc/data/runs/q3vl_base_sft_20260804/checkpoint-4976")
    ap.add_argument("--split", default="V_where")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--attn", default="eager")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--probe-train-split", default="train")
    ap.add_argument("--probe-train-n", type=int, default=1500)
    ap.add_argument("--skip-dx5", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    try:
        import resource

        s_, h_ = resource.getrlimit(resource.RLIMIT_NOFILE)
        if s_ < h_:
            resource.setrlimit(resource.RLIMIT_NOFILE, (h_, h_))
    except Exception:
        pass

    import torch.nn.functional as F
    from transformers import AutoProcessor

    from q3vl.train.collator import Sft2SegCollator
    from q3vl.train.modeling import load_model
    from q3vl.where.config import UpsampleConfig
    from q3vl.where.readout import apply_readout
    from q3vl.where.upsample import area_resize, guided_upsample, luma_guide
    from q3vl.whereb.amort.data import AmortBatchBuilder, family_labels
    from q3vl.whereb.amort.model import AmortModel
    from q3vl.whereb.amort.simfield import SimFieldNorm, WordEmbedder
    from q3vl.whereb.context import ShuffleIndex
    from q3vl.whereb.data import open_dataset
    from q3vl.whereb.edgequal import (calibrate, edge_quality_row, e_hf,
                                      kappa_tilde, mvr, afr)
    from q3vl.whereb.fields import load_basis
    from q3vl.whereb.hiddens import FrozenVLM
    from q3vl.whereb.metrics import gt_area_k, hard_iou, topk_mask
    from q3vl.whereb.stores import GenContextStore
    from q3vl.whereb.config import GENCTX_DIR

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ucfg = UpsampleConfig()

    proc = AutoProcessor.from_pretrained(args.checkpoint)
    vlm_model = load_model(args.checkpoint, attn_implementation=args.attn,
                           dtype="bfloat16").to(args.device).eval()
    vlm = FrozenVLM(vlm_model, proc, device=args.device, want_merger=True)
    collator = Sft2SegCollator(proc, max_length=2048, system_prompt=None)
    basis = load_basis("BA-3-Joint").to(args.device)
    embedder = WordEmbedder.from_checkpoint(args.checkpoint, proc.tokenizer)
    setup = json.loads((Path(args.p3_run) / "config" / "run_setup.json").read_text())
    norm = SimFieldNorm.from_dict(setup["sim_norm"])

    ds, _ = open_dataset(args.split, need_mask=True)
    rows = ds.meta_rows()
    idx = [i for i, r in enumerate(rows) if r.get("render_mode") == "local"]
    if args.limit:
        idx = idx[: args.limit]
    fam = family_labels(ds, idx)
    try:
        genctx = GenContextStore(Path(GENCTX_DIR) / args.split)
    except Exception:
        genctx = None
    builder = AmortBatchBuilder(
        collator, vlm, basis, embedder=embedder, norm=norm,
        shuffle_index=ShuffleIndex(ds.shuffle_records(), seed=0), genctx=genctx,
        families=fam, id_to_index={ds.record(i)["sample_id"]: i for i in idx},
        dataset=ds, device=args.device, want_hi=True,
        attn_implementation=args.attn, checkpoint=args.checkpoint)

    arms = {}
    for tag, run in (("P1", args.p1_run), ("P3prime", args.p3_run)):
        m = AmortModel(tag if tag == "P1" else "P3prime").to(args.device)
        m.load_state_dict(torch.load(Path(run) / "amort_final.pt",
                                     map_location=args.device)["model"])
        m.eval()
        arms[tag] = m

    ctx = "generated" if genctx is not None else "gt"
    print(f"context={ctx}  n={len(idx)}", flush=True)

    # ---------------- pass 1: collect fields --------------------------------
    recs: list[dict[str, Any]] = []
    gt_lo_pool, gt_hi_pool = [], []
    with torch.no_grad():
        for n, i in enumerate(idx):
            s = ds[i]
            x = builder.build([s], [ctx])[0]
            gh, gw = x.grid_h, x.grid_w
            gt_hi = x.gt_hi.float().cpu()
            gt_lo = x.gt_low.float().cpu()
            guide = x.guide_hi
            hi_shape = tuple(gt_hi.shape)

            # GT coarse -> the operator, and -> bilinear.  Linear, exactly
            # invertible s-mapping so the operator is the only thing measured.
            s_gt = (gt_lo * 2.0 - 1.0) * 3.0
            s_gt_b = s_gt.reshape(1, 1, gh, gw).to(guide.device)
            up_g = guided_upsample(s_gt_b, guide, ucfg).reshape(hi_shape).cpu()
            up_b = F.interpolate(s_gt_b, size=hi_shape, mode="bilinear",
                                 align_corners=False).reshape(hi_shape).cpu()
            gt_guided = (up_g / 3.0 + 1.0) / 2.0
            gt_bilin = (up_b / 3.0 + 1.0) / 2.0

            rec: dict[str, Any] = {
                "sample_id": x.sample_id, "family": x.family,
                "winner_confidence": x.meta.get("winner_confidence"),
                "grid": [gh, gw], "hi": list(hi_shape),
                "_gt_hi": gt_hi.numpy(), "_gt_lo": gt_lo.numpy(),
                "_gt_guided": gt_guided.numpy(), "_gt_bilin": gt_bilin.numpy(),
            }

            for tag, m in arms.items():
                cond = m.cond_of(x.cond_h, x.cond_mask, x.word_ids, x.word_offsets)
                if x.route_semantic and m.sem is not None:
                    o = m.forward_sem(x.feat, cond, sim=x.sim, center=x.center,
                                      guide_hi=guide)
                    s_low = o["s_low"].reshape(1, 1, gh, gw)
                    to_m = m.sem.mask_of
                    m_g = o["m_hi"].reshape(hi_shape).float().cpu()
                else:
                    o = m.forward_geo(x.feat, cond, x.phi_dir, sim=x.sim,
                                      center=x.center, guide_hi=guide,
                                      grid_h=gh, grid_w=gw)
                    s_low = o["s_low"].reshape(1, 1, gh, gw)
                    m_g = o["m_hi"].reshape(hi_shape).float().cpu()
                    to_m = None
                s_bl = F.interpolate(s_low.float(), size=hi_shape, mode="bilinear",
                                     align_corners=False)
                if to_m is not None:
                    m_b = to_m(s_bl).reshape(hi_shape).float().cpu()
                elif tag == "P3prime":
                    m_b = m.geo.mask_of(s_bl).reshape(hi_shape).float().cpu()
                else:
                    rho = {k: v.float() for k, v in o["params"].items()
                           if k not in ("w0", "w_raw", "alpha_raw")}
                    m_b = apply_readout(m.readout, s_bl.reshape(-1), rho
                                        ).reshape(hi_shape).float().cpu()
                rec[f"_{tag}_lo"] = o["m_low"].float().cpu().numpy()
                rec[f"_{tag}_guided"] = m_g.numpy()
                rec[f"_{tag}_bilin"] = m_b.numpy()
            recs.append(rec)
            gt_lo_pool.append(rec["_gt_lo"])
            gt_hi_pool.append(rec["_gt_hi"])
            if (n + 1) % 50 == 0:
                print(f"  fields [{n+1}/{len(idx)}]", flush=True)

    # ---------------- arm-constant calibration ------------------------------
    ana = [r for r in recs if r["family"] in ANALYTIC]
    cal_hi = calibrate([r["_gt_hi"] for r in ana])
    cal_lo = calibrate([r["_gt_lo"] for r in ana])
    print(f"calibration hi: r_hi={cal_hi.r_hi:.4f} tau={cal_hi.tau:.5f}  "
          f"lo: r_hi={cal_lo.r_hi:.4f} tau={cal_lo.tau:.5f}", flush=True)

    # ---------------- DX-1 / DX-2 / DX-3 ------------------------------------
    per: list[dict[str, Any]] = []
    for r in recs:
        f_ = r["family"]
        row = {"sample_id": r["sample_id"], "family": f_,
               "winner_confidence": r["winner_confidence"],
               "analytic": f_ in ANALYTIC}
        gt_hi, gt_lo = r["_gt_hi"], r["_gt_lo"]
        # DX-1: the operator alone, model absent
        row["dx1_E_HF"] = e_hf(r["_gt_guided"], gt_hi, cal_hi)
        row["dx1_kappa_ratio"] = edge_quality_row(
            r["_gt_guided"], gt_hi, f_, cal_hi, with_afr=False)["kappa_ratio"]
        row["dx1_bilin_E_HF"] = e_hf(r["_gt_bilin"], gt_hi, cal_hi)
        k_gt = gt_area_k(torch.from_numpy(gt_hi))
        row["dx1_soft_iou"] = hard_iou(topk_mask(torch.from_numpy(r["_gt_guided"]), k_gt),
                                       topk_mask(torch.from_numpy(gt_hi), k_gt))
        for tag in ("P1", "P3prime"):
            g, b, lo = r[f"_{tag}_guided"], r[f"_{tag}_bilin"], r[f"_{tag}_lo"]
            row[f"{tag}_D_total"] = e_hf(g, gt_hi, cal_hi)      # dirty-edge total
            row[f"{tag}_E_HF_bilin"] = e_hf(b, gt_hi, cal_hi)   # DX-2
            row[f"{tag}_kappa_ratio"] = edge_quality_row(
                g, gt_hi, f_, cal_hi, with_afr=False)["kappa_ratio"]
            row[f"{tag}_iou_guided"] = hard_iou(
                topk_mask(torch.from_numpy(g), k_gt), topk_mask(torch.from_numpy(gt_hi), k_gt))
            row[f"{tag}_iou_bilin"] = hard_iou(
                topk_mask(torch.from_numpy(b), k_gt), topk_mask(torch.from_numpy(gt_hi), k_gt))
            # DX-3: the coarse field itself, against the GT-coarse floor
            row[f"{tag}_lo_E_HF"] = e_hf(lo, gt_lo, cal_lo)
            row[f"{tag}_lo_MVR"] = mvr(lo, gt_lo, f_, cal_lo)
            row[f"{tag}_lo_AFR"] = afr(lo)
        row["floor_lo_MVR"] = mvr(gt_lo, gt_lo, f_, cal_lo)
        row["floor_lo_AFR"] = afr(gt_lo)
        per.append(row)

    def agg(sel, key):
        return _med(x[key] for x in per if sel(x) and x.get(key) is not None)

    is_ana = lambda x: x["analytic"]           # noqa: E731
    is_sem = lambda x: not x["analytic"]       # noqa: E731

    res: dict[str, Any] = {"n": len(per), "context": ctx,
                           "calibration": {"hi": cal_hi.to_dict(),
                                           "lo": cal_lo.to_dict()}}
    dx = {}
    for name, sel in (("analytic", is_ana), ("semantic", is_sem)):
        d_tot = {t: agg(sel, f"{t}_D_total") for t in ("P1", "P3prime")}
        dx[name] = {
            "n": sum(1 for x in per if sel(x)),
            "D_total": d_tot,
            "dx1_operator_only_E_HF": agg(sel, "dx1_E_HF"),
            "dx1_bilinear_E_HF": agg(sel, "dx1_bilin_E_HF"),
            "dx1_kappa_ratio": agg(sel, "dx1_kappa_ratio"),
            "dx1_soft_iou": agg(sel, "dx1_soft_iou"),
            "dx2_E_HF_bilin": {t: agg(sel, f"{t}_E_HF_bilin") for t in ("P1", "P3prime")},
            "dx3_coarse": {t: {"E_HF": agg(sel, f"{t}_lo_E_HF"),
                               "MVR": agg(sel, f"{t}_lo_MVR"),
                               "AFR": agg(sel, f"{t}_lo_AFR")}
                           for t in ("P1", "P3prime")},
            "dx3_floor": {"MVR": agg(sel, "floor_lo_MVR"),
                          "AFR": agg(sel, "floor_lo_AFR")},
            "iou_guided": {t: agg(sel, f"{t}_iou_guided") for t in ("P1", "P3prime")},
            "iou_bilin": {t: agg(sel, f"{t}_iou_bilin") for t in ("P1", "P3prime")},
            "kappa_ratio_guided": {t: agg(sel, f"{t}_kappa_ratio")
                                   for t in ("P1", "P3prime")},
        }
        # delta_b: share of dirty edge removed by dropping the guided operator
        db = {}
        for t in ("P1", "P3prime"):
            cur, bl = d_tot[t], dx[name]["dx2_E_HF_bilin"][t]
            db[t] = None if (cur is None or bl is None or abs(cur) < 1e-12) \
                else float(1.0 - bl / cur)
        dx[name]["dx2_delta_b"] = db
    res["dx"] = dx
    # Per-family breakdown.  MVR uses a CONSTANT direction vector for
    # linear/band (the registered rule), but a `band` mask is non-monotone along
    # any constant direction by construction, so its MVR floor is high for a
    # convention reason rather than a quality reason.  Reporting per family
    # keeps that visible instead of averaging it into the analytic aggregate.
    byfam = {}
    for f_ in ("radial", "linear", "band", "semantic"):
        sel = lambda x, f_=f_: x["family"] == f_
        byfam[f_] = {
            "n": sum(1 for x in per if sel(x)),
            "D_total": {t: agg(sel, f"{t}_D_total") for t in ("P1", "P3prime")},
            "delta_b": {t: (None if (agg(sel, f"{t}_D_total") in (None, 0)
                                     or agg(sel, f"{t}_E_HF_bilin") is None)
                            else 1.0 - agg(sel, f"{t}_E_HF_bilin") / agg(sel, f"{t}_D_total"))
                        for t in ("P1", "P3prime")},
            "coarse_MVR": {t: agg(sel, f"{t}_lo_MVR") for t in ("P1", "P3prime")},
            "floor_MVR": agg(sel, "floor_lo_MVR"),
            "coarse_AFR": {t: agg(sel, f"{t}_lo_AFR") for t in ("P1", "P3prime")},
            "floor_AFR": agg(sel, "floor_lo_AFR"),
            "kappa_ratio": {t: agg(sel, f"{t}_kappa_ratio") for t in ("P1", "P3prime")},
            "iou_guided": {t: agg(sel, f"{t}_iou_guided") for t in ("P1", "P3prime")},
            "iou_bilin": {t: agg(sel, f"{t}_iou_bilin") for t in ("P1", "P3prime")},
        }
    res["by_family"] = byfam

    # ---------------- DX-6: oracle hard routing -----------------------------
    # analytic -> bilinear (family-neutral), semantic -> current guided operator
    orac = {}
    for t in ("P1", "P3prime"):
        gains, kap_g, kap_o = [], [], []
        for x in per:
            cur = x[f"{t}_iou_guided"]
            new = x[f"{t}_iou_bilin"] if x["analytic"] else cur
            if cur is not None and new is not None:
                gains.append((x["analytic"], new - cur))
        orac[t] = {
            "analytic_delta_iou": _med(g for a, g in gains if a),
            "semantic_delta_iou": _med(g for a, g in gains if not a),
            "analytic_kappa_ratio_guided": agg(is_ana, f"{t}_kappa_ratio"),
        }
    res["dx6_oracle_routing"] = orac

    # ---------------- DX-4: loss blind-spot audit (CPU) ---------------------
    from q3vl.whereb.amort.losses import (LossWeights, area_band, bce_soft,
                                          sdf_boundary, signed_distance_field)

    w = LossWeights()
    rng = np.random.default_rng(20260811)
    audit: dict[str, dict[str, float]] = {}
    tgt_e = _med(x["P3prime_D_total"] for x in per if x["analytic"]) or 0.05
    for art in ("contour_warp", "texture_engrave", "global_widen"):
        deltas = defaultdict(list)
        for r in recs:
            if r["family"] not in ANALYTIC:
                continue
            gt = torch.from_numpy(r["_gt_lo"]).float()
            gh, gw = gt.shape
            base = gt.clone()
            if art == "contour_warp":
                yy, xx = np.mgrid[0:gh, 0:gw]
                ph = np.sin(2 * np.pi * yy / max(gh, 1) * 3.0) * 0.5
                dirty = torch.from_numpy(
                    np.clip(r["_gt_lo"] + ph * 0.1, 0, 1)).float()
            elif art == "texture_engrave":
                noise = rng.normal(0, 1, size=(gh, gw))
                dirty = torch.from_numpy(np.clip(r["_gt_lo"] + noise * 0.08, 0, 1)).float()
            else:  # global_widen
                dirty = torch.clamp(gt * 1.0 + 0.12, 0, 1)
            # calibrate amplitude so the injected E_HF matches the measured total
            got = e_hf(dirty.numpy(), base.numpy(), cal_lo)
            if abs(got) > 1e-9:
                scale = float(np.clip(abs(tgt_e / got) ** 0.5, 0.2, 5.0))
                dirty = torch.clamp(base + (dirty - base) * scale, 0, 1)
            phi = signed_distance_field(base)
            terms = {"BCE": (w.bce, lambda m: bce_soft(m, base)),
                     "SDF_boundary": (w.sdf, lambda m: sdf_boundary(m, phi)),
                     "area_band": (w.area, lambda m: area_band(m, base, w.area_tau))}
            # The denominator is the TOTAL WEIGHTED loss, not the individual
            # term: `area_band` is exactly 0 at the GT (the GT matches its own
            # area), so a per-term Delta/base is 0/0 and says nothing.  The
            # question DX-4 actually asks is "does the optimiser see this
            # artefact at all", and that is a question about the total.
            tot0 = tot1 = 0.0
            for nm, (wt, fn) in terms.items():
                l0, l1 = float(fn(base)), float(fn(dirty))
                tot0 += wt * l0
                tot1 += wt * l1
                deltas[nm].append((wt * (l1 - l0), abs(wt * l0)))
            deltas["TOTAL"].append((tot1 - tot0, abs(tot0)))
        audit[art] = {}
        tot_base = _med(b for _, b in deltas["TOTAL"]) or 1e-9
        for nm, vals in deltas.items():
            d = _med(abs(a) for a, _ in vals) or 0.0
            b = _med(b for _, b in vals) or 0.0
            audit[art][nm] = {
                "abs_weighted_delta_median": d,
                "weighted_base_median": b,
                # share of the TOTAL training signal this artefact moves
                "rel_delta_vs_total": d / max(tot_base, 1e-9),
                "verdict": ("blind (<5%)" if d / max(tot_base, 1e-9) < 0.05
                            else "visible (>=20%)" if d / max(tot_base, 1e-9) >= 0.20
                            else "partial (5-20%)"),
            }
    res["dx4_loss_audit"] = audit

    # Persist DX-1/2/3/4/6 BEFORE the optional probe.  DX-5 has now aborted the
    # whole battery twice (missing sklearn, then an empty feature matrix), each
    # time discarding five completed experiments that had nothing to do with it.
    # A late optional stage must never be able to void what already succeeded.
    def _dump():
        (out / "metrics.json").write_text(json.dumps(res, indent=2, default=str),
                                          encoding="utf-8")
        with (out / "per_sample_dx.jsonl").open("w") as fh:
            for row in per:
                fh.write(json.dumps(row) + "\n")

    _dump()
    print(f"DX-1/2/3/4/6 written to {out}", flush=True)

    # ---------------- DX-5: family separability probe -----------------------
    if not args.skip_dx5:
        try:
            res["dx5_probe"] = _dx5(args, builder, ds, idx, fam, ctx)
        except Exception as exc:  # noqa: BLE001
            import traceback

            res["dx5_probe"] = {"error": f"{type(exc).__name__}: {exc}",
                                "traceback": traceback.format_exc()[-1500:]}
            print(f"DX-5 FAILED (the rest of the battery is unaffected): {exc}",
                  flush=True)
        _dump()
    print(json.dumps({k: v for k, v in res.items()
                      if k in ("dx", "dx4_loss_audit", "dx6_oracle_routing",
                               "dx5_probe")}, indent=2, default=str)[:4000])
    return 0


def _dx5(args, builder, ds, idx, fam, ctx) -> dict[str, Any]:
    """Family separability from frozen features + instruction embedding.

    Trained on the TRAIN split and tested on V_where, so the held-out accuracy
    is not a within-split fit.  This is the floor under every gating scheme in
    §4 -- if families are not cheaply separable, no router is admissible.
    """
    import torch

    from q3vl.whereb.amort.data import family_labels
    from q3vl.whereb.data import open_dataset

    FAMS = ["radial", "linear", "band", "semantic"]
    fidx = {f: i for i, f in enumerate(FAMS)}

    skipped: dict[str, int] = {}

    def feats(dset, indices, families, mode):
        X, Y = [], []
        with torch.no_grad():
            for n, i in enumerate(indices):
                s = dset[i]
                try:
                    x = builder.build([s], [mode])[0]
                except Exception as exc:  # noqa: BLE001
                    skipped[f"{type(exc).__name__}: {str(exc)[:70]}"] = (
                        skipped.get(f"{type(exc).__name__}: {str(exc)[:70]}", 0) + 1)
                    continue
                lab = families.get(s.sample_id)
                if lab not in fidx:
                    continue
                # pooled frozen visual feature + pooled <where> hidden
                v = x.feat.mean(dim=(-2, -1)).reshape(-1)
                t = (x.cond_h[0] * x.cond_mask[0].float().unsqueeze(-1)).sum(0) \
                    / x.cond_mask[0].float().sum().clamp_min(1)
                X.append(torch.cat([v, t]).float().cpu().numpy())
                Y.append(fidx[lab])
                if (n + 1) % 250 == 0:
                    print(f"    probe feats [{n+1}/{len(indices)}]", flush=True)
        if not X:
            raise RuntimeError(
                f"probe got 0 usable samples from {len(indices)} "
                f"(mode={mode!r}); skip reasons: "
                f"{sorted(skipped.items(), key=lambda kv: -kv[1])[:3]}")
        return np.stack(X), np.asarray(Y)

    # need_mask=True: the builder resolves a GT mask for every sample, so
    # need_mask=False made every build raise and left the probe with 0 rows
    tr_ds, _ = open_dataset(args.probe_train_split, need_mask=True,
                            exclude_low=True)
    tr_rows = tr_ds.meta_rows()
    tr_idx = [i for i, r in enumerate(tr_rows) if r.get("render_mode") == "local"]
    rng = np.random.default_rng(0)
    tr_idx = [tr_idx[i] for i in rng.choice(len(tr_idx),
                                            size=min(args.probe_train_n, len(tr_idx)),
                                            replace=False)]
    tr_fam = family_labels(tr_ds, tr_idx)
    # point every split-dependent field at the probe split, not just `dataset`
    saved = (builder.dataset, builder.id_to_index, builder.families,
             builder.shuffle_index, builder.genctx)
    builder.dataset = tr_ds
    builder.id_to_index = {tr_ds.record(i)["sample_id"]: i for i in tr_idx}
    builder.families = tr_fam
    builder.genctx = None                 # train genctx is a different store
    Xtr, Ytr = feats(tr_ds, tr_idx, tr_fam, "gt")
    (builder.dataset, builder.id_to_index, builder.families,
     builder.shuffle_index, builder.genctx) = saved
    Xte, Yte = feats(ds, idx, fam, ctx)
    print(f"  probe train {Xtr.shape} test {Xte.shape}", flush=True)

    # Multinomial logistic regression in torch -- sklearn is not installed in
    # this env, and a 20-line probe is a smaller dependency than a new package
    # on a machine that is mid-campaign.
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    xt = torch.from_numpy(((Xtr - mu) / sd).astype(np.float32))
    yt = torch.from_numpy(Ytr.astype(np.int64))
    xe = torch.from_numpy(((Xte - mu) / sd).astype(np.float32))
    lin = torch.nn.Linear(xt.shape[1], len(FAMS))
    opt = torch.optim.LBFGS(lin.parameters(), max_iter=300, history_size=20)
    lossf = torch.nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        l = lossf(lin(xt), yt) + 1e-4 * lin.weight.pow(2).sum()
        l.backward()
        return l

    opt.step(closure)
    with torch.no_grad():
        train_acc = float((lin(xt).argmax(1) == yt).float().mean())
        pred = lin(xe).argmax(1).numpy()
    print(f"  probe train acc {train_acc:.4f}", flush=True)
    acc = float((pred == Yte).mean())
    sem = fidx["semantic"]
    ana_mask = Yte != sem
    # the asymmetric error that matters: an ANALYTIC sample routed to the
    # semantic (edge-hugging) path is the worst case
    wrong_dir = float((pred[ana_mask] == sem).mean()) if ana_mask.any() else 0.0
    binary = float(((pred == sem) == (Yte == sem)).mean())
    return {"n_train": int(len(Ytr)), "n_test": int(len(Yte)),
            "train_acc_4way": train_acc,
            "acc_4way": acc, "acc_binary_analytic_vs_semantic": binary,
            "analytic_misrouted_to_semantic": wrong_dir,
            "gate_verdict": ("hard_routing_ok" if binary >= 0.995 and wrong_dir <= 0.005
                             else "soft_gate_only" if binary >= 0.95
                             else "gates_downgraded")}


if __name__ == "__main__":
    raise SystemExit(main())
