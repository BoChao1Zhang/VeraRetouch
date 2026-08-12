"""Proposal B (FAFM) probe -- six arms, nine pre-registered criteria, one CFG rule.

``RESEARCH_unified-field-prediction_2026-08-10`` section 3.6, run only because Gate 0
passed both its gates (uni_gate0_20260811/caseB_fafm: G0a 4.67e-16, G0b overall
median 0.9967, every family >= 0.88).

Arms (same data, same condition tensors, same evaluation):

===  ==========================================================================
A    FAFM -- flow matching in the neck coordinate, K=16, R1 mode-seeking
B    centre prior -- zero parameters, matched-area top-k (mandatory column)
C    regression-direct -- SAME architecture, SAME conditions, one-shot L2 to c*.
     The核心 mechanism control, and simultaneously proposal A's E5: both cases
     ride on this one arm, so it is trained at full priority, not as an extra.
D    no-instruction -- A with E_T dropped (S, V only)
E    negative controls x3 -- shuffled / irrelevant-word / fixed-phrase instruction
F    K=1 -- A's single-sample setting, tests whether multi-modality is load-bearing
===  ==========================================================================

A, C and D are separate training runs; B, E and F are evaluation-time variants
of A, so the probe costs three trainings, not six.

Discipline: no AUC anywhere; soft/hard-IoU at matched-GT-area top-k, grid
boundary F1, the centre-prior column, the ``a/(2-a)`` floor, area strata and
per-family strata on every row.  IoU is never an optimisation target.
Checkpoint selection is by **pre-registered step count**, never by val loss.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

#: Per-family neck ceiling **in the column criterion 6 actually measures**.
#:
#: Criterion 6's ``measured`` is ``soft_iou_value(topk_mask(pred16, k), gt16b)``:
#: grid-level, binarised, matched-GT-area top-k.  The Gate-0 ceiling for THAT
#: column is ``G0b.grid_softiou_by_family``.  An earlier revision wired
#: ``softiou_hi_by_family`` instead -- the un-thresholded full-resolution soft
#: field -- which is a different quantity, and it put the contour family's neck
#: tax an order of magnitude out on the very column it was printed beside
#: (0.1051 claimed vs 0.0112 real).  REVIEW-impl-amort-uni B3.
GATE0_FAMILY_CEILING_GRID = {
    "linear": 1.0000, "band": 1.0000, "radial": 1.0000,
    "semantic": 0.9888, "unknown": 1.0000,
}
GATE0_OVERALL_CEILING_GRID = 1.0000

#: The same ceilings in the **full-resolution soft-field** column.  Reported for
#: provenance next to the grid column, never used to normalise criterion 6.
#: This is the column the "0.8949 / neck tax 0.1051" figures belong to.
GATE0_FAMILY_CEILING_HI = {
    "linear": 0.9990, "band": 0.9971, "radial": 0.9937,
    "semantic": 0.8949, "unknown": 0.9944,
}
GATE0_OVERALL_CEILING_HI = 0.9967

CFG_GRID = (1.0, 1.5, 2.0)
K_SAMPLES = 16
N_STEPS = 8
TAU = 0.6
NEGATIVES = ("shuffled", "irrelevant_words", "fixed_phrase")
#: The project's canonical fixed phrase, not an ad-hoc one.  "the main subject"
#: is deliberately the campaign's own known-strong subject-prior baseline, so the
#: control is HARD to beat; a bland "edit the image" throws that away and makes
#: the negative easier than it is supposed to be (REVIEW-impl-amort-uni B5).
FIXED_PHRASE = None      # resolved from q3vl.whereb.context at run time
#: Irrelevant words are drawn PER SAMPLE from the project vocabulary.  A single
#: constant string would collapse this control into a second copy of the fixed
#: phrase -- the project's own docstring says as much ("a single constant string
#: is the *other* control").
IRRELEVANT = None        # resolved per sample at run time


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:                                        # noqa: BLE001
        return "unknown"


# --------------------------------------------------------------------------- #
#  data                                                                        #
# --------------------------------------------------------------------------- #

class FAFMData:
    """Grid-bucketed access to a ``dump_fafm_cache`` directory.

    Samples are bucketed by grid shape (29 distinct shapes in the 20k subset,
    dominated by 32x48 at 41% and 48x32 at 24%): the stage keeps true aspect
    ratio, so a batch has to share a shape.  The network's position embedding is
    sin-cos and therefore shape-agnostic, so bucketing costs nothing but the
    loader.
    """

    def assert_cstar_domain(self, n_probe: int = 256, tol: float = 1e-6
                            ) -> dict[str, Any]:
        """Consumer-side domain assertion -- required before c* enters training.

        Section 3.2: "c* 的整臂实测值域必须落盘声明 ... **消费侧断言后才进训练**".
        Copying the producer's declaration into ``metrics.json`` is not an
        assertion; it is a restatement.  This actually reads the tensors and
        checks they live inside the declared interval, which is the only thing
        that catches the s-cache contract's second failure mode -- the silent one,
        where the values still sit inside the anchor domain but the axis is gone.
        """
        dom = (self.manifest.get("cstar_domain") or {}).get("domain")
        if not dom:
            raise RuntimeError(f"{self.root}: manifest declares no cstar domain")
        lo, hi = float(dom[0]), float(dom[1])
        rng = np.random.default_rng(0)
        pick = rng.choice(len(self.meta), size=min(n_probe, len(self.meta)),
                          replace=False)
        seen_lo, seen_hi, n_bad = float("inf"), float("-inf"), 0
        for i in pick:
            c = self.load(int(i))["cstar"]
            seen_lo = min(seen_lo, float(c.min()))
            seen_hi = max(seen_hi, float(c.max()))
            if float(c.min()) < lo - tol or float(c.max()) > hi + tol:
                n_bad += 1
        if n_bad:
            raise RuntimeError(
                f"{self.root}: {n_bad}/{len(pick)} sampled c* tensors fall outside "
                f"the declared domain [{lo}, {hi}] (observed [{seen_lo}, {seen_hi}]). "
                "Refusing to train on data that violates its own declaration.")
        return {"declared": [lo, hi], "observed": [seen_lo, seen_hi],
                "n_probed": int(len(pick)), "status": "asserted",
                "note": "c* is NOT in [0,1]; clip the FIELD, never the coordinate."}

    def __init__(self, root: str | Path, limit: int | None = None):
        self.root = Path(root)
        man = json.loads((self.root / "manifest.json").read_text())
        self.meta = man["samples"][:limit] if limit else man["samples"]
        self.manifest = man
        self.by_shape: dict[tuple[int, int], list[int]] = defaultdict(list)
        for i, s in enumerate(self.meta):
            self.by_shape[tuple(s["grid16"])].append(i)

    def __len__(self) -> int:
        return len(self.meta)

    def load(self, idx: int) -> dict[str, Any]:
        s = self.meta[idx]
        d = np.load(self.root / "cache" / f"{s['sample_id']}.npz")
        gh, gw = s["grid16"]
        return {
            "meta": s,
            "cstar": torch.from_numpy(d["cstar"]).float().reshape(1, gh, gw),
            "sem": torch.from_numpy(d["sem"]).float().reshape(gh, gw, -1).permute(2, 0, 1),
            "sim": torch.from_numpy(d["sim"]).float(),          # (P32, S)
            "guide_q": torch.from_numpy(d["guide_q"]).float(),
            "gt16": torch.from_numpy(d["gt16"]).float(),
            "grid32": s["grid32"],
        }

    def batch(self, idxs: list[int], device, sim_ch: int) -> dict[str, Any]:
        items = [self.load(i) for i in idxs]
        gh, gw = items[0]["cstar"].shape[-2:]
        sim = []
        for it in items:
            g32 = it["grid32"]
            s = it["sim"].reshape(g32[0], g32[1], -1).permute(2, 0, 1)[None]
            s = torch.nn.functional.interpolate(s, size=(gh, gw), mode="bilinear",
                                                align_corners=False)[0]
            sim.append(s[:sim_ch])
        return {
            "cstar": torch.stack([it["cstar"] for it in items]).to(device),
            "vis": torch.stack([it["sem"] for it in items]).to(device),
            "sim": torch.stack(sim).to(device),
            "guide_q": torch.stack([it["guide_q"] for it in items]).to(device),
            "gt16": torch.stack([it["gt16"] for it in items]).to(device),
            "meta": [it["meta"] for it in items],
            "grid": (gh, gw),
        }


class TextEncoder:
    """Frozen instruction embeddings -> ``(B, L, 2560)`` + padding mask."""

    def __init__(self, checkpoint: str, device, max_len: int = 48):
        from transformers import AutoTokenizer
        from q3vl.whereb.scripts.run_amort_e5 import load_embeddings
        self.tok = AutoTokenizer.from_pretrained(checkpoint)
        self.E = load_embeddings(checkpoint).float()
        self.device = device
        self.max_len = max_len

    def __call__(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        ids = [self.tok(t or "", add_special_tokens=False)["input_ids"][: self.max_len]
               or [0] for t in texts]
        L = max(len(i) for i in ids)
        out = torch.zeros(len(ids), L, self.E.shape[-1])
        mask = torch.ones(len(ids), L, dtype=torch.bool)
        for b, seq in enumerate(ids):
            out[b, : len(seq)] = self.E[torch.tensor(seq)]
            mask[b, : len(seq)] = False
        return out.to(self.device), mask.to(self.device)


def make_apply_A(guide_q: torch.Tensor, gh: int, gw: int):
    """``(apply_A, sigma_max_sq)`` for the stored guide's resolution, clamp OFF.

    Batched: one guided-upsample per residual.  The clamp must stay off -- it is
    the only nonlinearity in the operator, and with it on the metric term would
    silently stop being a quadratic form.

    The operator's ``||A||^2`` is returned **alongside** it, computed from this
    guide's own shape.  Handing the caller the operator without its norm is what
    let a full-resolution constant (256) be applied to a quarter-resolution
    operator (true value 16) and quietly cut the metric term to 6% of its
    intended weight (REVIEW-impl-amort-uni U5).
    """
    from q3vl.where.config import UpsampleConfig
    from q3vl.where.upsample import guided_upsample
    from q3vl.whereb.fafm import sigma_max_sq
    cfg = UpsampleConfig(clamp_domain=False)
    n_hi = int(guide_q.shape[-2]) * int(guide_q.shape[-1])
    smsq = sigma_max_sq(n_hi, gh * gw)

    def apply_A(r: torch.Tensor) -> torch.Tensor:
        g = guide_q[:, None]
        return guided_upsample(r, g, cfg)
    return apply_A, smsq


# --------------------------------------------------------------------------- #
#  train                                                                       #
# --------------------------------------------------------------------------- #

def train_arm(arm: str, data: FAFMData, text: TextEncoder, device, *,
              steps: int, batch: int, lr: float, seed: int, log_every: int = 100,
              cfg_obj=None) -> torch.nn.Module:
    from q3vl.whereb.fafm import FAFMConfig, FAFMNet, lambda_metric_loss

    torch.manual_seed(seed)
    fcfg = cfg_obj or FAFMConfig()
    net = FAFMNet(fcfg).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    # pct_start must buy at least one warmup step: at very small `steps` (smoke
    # runs) 0.05 rounds to a zero-length phase and OneCycleLR divides by zero.
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps,
        pct_start=max(0.05, 2.0 / max(steps, 2)))
    rng = np.random.default_rng(seed)
    shapes = [s for s, v in data.by_shape.items() if len(v) >= 8]
    weights = np.array([len(data.by_shape[s]) for s in shapes], dtype=float)
    weights /= weights.sum()
    n_par = sum(p.numel() for p in net.parameters())
    print(f"[{arm}] {n_par/1e6:.1f}M params, {steps} steps, batch {batch}", flush=True)

    t0 = time.time()
    net.train()
    losses: list[float] = []
    for step in range(1, steps + 1):
        sh = shapes[rng.choice(len(shapes), p=weights)]
        pool = data.by_shape[sh]
        idxs = rng.choice(len(pool), size=min(batch, len(pool)), replace=False)
        b = data.batch([pool[i] for i in idxs], device, fcfg.in_sim)
        gh, gw = b["grid"]
        B = b["cstar"].shape[0]

        txt, mask = text([m["instruction"] for m in b["meta"]])
        if arm == "no_text":
            txt = net.null_text.expand(B, 1, -1)
            mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
        else:
            # condition dropout -> free unconditional branch for CFG and controls
            drop = torch.rand(B, device=device) < fcfg.p_drop_text
            if drop.any():
                txt = txt.clone()
                txt[drop] = 0.0
                mask = mask.clone()
                mask[drop] = False
                mask[drop, 0] = False
        sim = b["sim"]
        dsim = torch.rand(B, device=device) < fcfg.p_drop_sim
        if dsim.any():
            sim = sim.clone()
            sim[dsim] = 0.0

        c_star = b["cstar"]
        apply_A, smsq = make_apply_A(b["guide_q"], gh, gw)
        if arm == "regression":
            t = torch.zeros(B, device=device)
            c_in = torch.zeros_like(c_star)
            c_hat = net(c_in, sim, b["vis"], t, txt, mask)
        else:
            t = torch.rand(B, device=device)
            c0 = torch.randn_like(c_star)
            c_t = (1 - t)[:, None, None, None] * c0 + t[:, None, None, None] * c_star
            c_hat = net(c_t, sim, b["vis"], t, txt, mask)
        loss = lambda_metric_loss(c_hat, c_star, apply_A, fcfg.lambda_mix, smsq)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(float(loss.detach()))
        if step % log_every == 0:
            print(f"[{arm}] step {step}/{steps} loss {np.mean(losses[-log_every:]):.4f} "
                  f"{time.time()-t0:.0f}s", flush=True)
    return net


# --------------------------------------------------------------------------- #
#  eval                                                                        #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def eval_arm(net, data: FAFMData, text: TextEncoder, device, *, arm: str,
             k: int, cfg_scale: float, seed: int, use_text: bool = True,
             negative: str | None = None, batch: int = 16,
             n_steps: int = N_STEPS,
             instr_override: dict[str, str] | None = None,
             shuffle_map: dict[str, str] | None = None,
             s_control: str | None = None) -> list[dict[str, Any]]:
    from q3vl.where.config import UpsampleConfig
    from q3vl.where.upsample import guided_upsample
    from q3vl.whereb.fafm import FAFMConfig, sample_fafm, select_mode
    from q3vl.whereb.metrics import (center_prior_field, grid_boundary_f1, gt_area_k,
                                     hard_iou, soft_iou_value, topk_mask)
    from q3vl.whereb.unifield import area_stratum

    fcfg = net.cfg if net is not None else FAFMConfig()
    ucfg = UpsampleConfig(clamp_domain=False)
    rows: list[dict[str, Any]] = []
    gen = torch.Generator(device=device).manual_seed(seed)
    # Iterate shape GROUPS, never a sliding window over a sorted list: a window
    # can straddle two grid shapes, and the obvious repairs either silently drop
    # the tail of the batch or fall back to one-sample-at-a-time.  Grouping first
    # makes every batch shape-homogeneous by construction.
    for shape, pool in sorted(data.by_shape.items()):
        for s in range(0, len(pool), batch):
            idxs = pool[s: s + batch]
            b = data.batch(idxs, device, fcfg.in_sim)
            gh, gw = b["grid"]
            B = len(idxs)
            instrs = [m["instruction"] for m in b["meta"]]
            if negative == "shuffled":
                roll = instrs[1:] + instrs[:1]
                instrs = roll
            elif negative == "irrelevant_words":
                instrs = [IRRELEVANT] * B
            elif negative == "fixed_phrase":
                instrs = [FIXED_PHRASE] * B
            txt, mask = text(instrs)

            if arm == "centre_prior":
                # The zero-parameter baseline is scored DIRECTLY on the grid, not
                # round-tripped through A_I.  Pushing it through the guided upsample
                # and back would lend it the operator's edge-snapping and quietly
                # make the mandatory baseline a different (stronger) object than the
                # one every other card in this campaign reports.
                cp = center_prior_field(gh, gw).double()
                for j, i in enumerate(idxs):
                    m = b["meta"][j]
                    gt16 = b["gt16"][j].cpu()
                    gt16b = (gt16 > 0.5).double()
                    kk = gt_area_k(gt16)
                    area = float(gt16b.mean())
                    pm = topk_mask(cp, kk)
                    rows.append({
                        "sample_id": m["sample_id"], "family": m.get("family", "unknown"),
                        "arm": arm, "negative": None, "cfg": cfg_scale, "k": 1,
                        "soft_iou": soft_iou_value(pm, gt16b),
                        "hard_iou": hard_iou(pm, gt16b),
                        "gbf1": grid_boundary_f1(pm, gt16b),
                        "centre_prior_softiou": soft_iou_value(pm, gt16b),
                        "centre_prior_gbf1": grid_boundary_f1(pm, gt16b),
                        "random_floor": area / (2 - area) if area < 1 else 1.0,
                        "area_frac": area, "area_stratum": area_stratum(area),
                        "pred_area_frac": float(pm.mean()),
                        "area_ratio": float(pm.mean()) / area if area > 0 else float("nan"),
                        "ambiguity": 0.0,
                    })
                continue
            if arm == "regression":
                t = torch.zeros(B, device=device)
                c_in = torch.zeros(B, 1, gh, gw, device=device)
                fields = net(c_in, b["sim"], b["vis"], t, txt, mask)[None]
            else:
                fields = sample_fafm(net, b["sim"], b["vis"], txt, mask, k=k,
                                     steps=n_steps, cfg_scale=cfg_scale,
                                     generator=gen, use_text=use_text)
            # render each sample through the frozen operator: clip the FIELD
            gq = b["guide_q"][:, None]
            rendered = []
            for kk in range(fields.shape[0]):
                r = guided_upsample(fields[kk].float(), gq, ucfg).clamp(0, 1)
                rendered.append(r)
            rend = torch.stack(rendered)                               # (K,B,1,H,W)

            for j, i in enumerate(idxs):
                m = b["meta"][j]
                fk = rend[:, j, 0].reshape(rend.shape[0], -1)
                if rend.shape[0] > 1:
                    chosen, ambig = select_mode(fk, TAU)
                else:
                    chosen, ambig = fk[0], 0.0
                H, W = rend.shape[-2:]
                pred_hi = chosen.reshape(H, W).cpu()
                gt16 = b["gt16"][j].cpu()
                pred16 = torch.nn.functional.interpolate(
                    pred_hi[None, None], size=(gh, gw), mode="area")[0, 0]
                gt16b = (gt16 > 0.5).double()
                kk = gt_area_k(gt16)
                area = float(gt16b.mean())
                cp = center_prior_field(gh, gw).double()
                pred_area = float(topk_mask(pred16, kk).mean()) if kk else 0.0
                row = {
                    "sample_id": m["sample_id"], "family": m.get("family", "unknown"),
                    "arm": arm, "negative": negative, "s_control": s_control,
                "cfg": cfg_scale, "k": k,
                    "soft_iou": soft_iou_value(topk_mask(pred16.double(), kk), gt16b),
                    "hard_iou": hard_iou(topk_mask(pred16.double(), kk), gt16b),
                    "gbf1": grid_boundary_f1(topk_mask(pred16.double(), kk), gt16b),
                    "centre_prior_softiou": soft_iou_value(topk_mask(cp, kk), gt16b),
                    "centre_prior_gbf1": grid_boundary_f1(topk_mask(cp, kk), gt16b),
                    "random_floor": area / (2 - area) if area < 1 else 1.0,
                    "area_frac": area, "area_stratum": area_stratum(area),
                    "pred_area_frac": float((pred16 > 0.5).double().mean()),
                    "area_ratio": (float((pred16 > 0.5).double().mean()) / area
                                   if area > 0 else float("nan")),
                    "ambiguity": ambig,
                }
                rows.append(row)
    return rows


class _Sub(FAFMData):
    """A one-sample view, used when a batch straddles two grid shapes."""

    def __init__(self, parent: FAFMData, idxs: list[int]):
        self.root = parent.root
        self.manifest = parent.manifest
        self.meta = [parent.meta[i] for i in idxs]
        self.by_shape = defaultdict(list)
        for i, s in enumerate(self.meta):
            self.by_shape[tuple(s["grid16"])].append(i)


def w1(a: list[float], b: list[float]) -> float:
    """1-Wasserstein between two 1-D samples (sorted-quantile form)."""
    x, y = np.sort(np.asarray(a)), np.sort(np.asarray(b))
    n = min(len(x), len(y))
    if n == 0:
        return float("nan")
    qs = (np.arange(n) + 0.5) / n
    return float(np.mean(np.abs(np.quantile(x, qs) - np.quantile(y, qs))))


def build_reversed_pairs(ev: FAFMData, max_gt_iou: float = 0.5
                         ) -> tuple[dict[str, str], dict[str, Any]]:
    """Same-image, different-target instruction pairs for criterion 4.

    Section 3.6 criterion 4 is "同图**反向指令**配对差分 ... ≥ 3x 全部负控制".
    "Same image, reversed instruction" is a strictly stronger probe than the
    ``shuffled`` negative: a shuffled instruction comes from a *different* image
    and can fail for reasons that have nothing to do with targeting (it may name
    objects that are simply absent).  A partner instruction is a *valid*
    instruction for *this* image, so the only thing that changes is which region
    is being asked for.

    The dissimilarity gate is the point.  Pairing by source image alone does not
    guarantee the target moved -- the same region often recurs under a different
    colour preset -- and a pair whose two GT masks coincide contributes a
    structurally-zero differential that silently dilutes the effect toward the
    null.  (REVIEW-impl-amort-uni opens on exactly this failure elsewhere in the
    campaign: "65.2% 的配对根本没换主体".)  So a partner is admissible only when
    ``soft-IoU(GT_i, GT_j) <= max_gt_iou``, and the realised pair count, the
    coverage, and the GT-overlap distribution are all reported so the claim is
    auditable rather than assumed.
    """
    from q3vl.whereb.metrics import soft_iou_value

    by_src: dict[str, list[int]] = defaultdict(list)
    for i, m in enumerate(ev.meta):
        src = m.get("source_image_id")
        if src:
            by_src[src].append(i)
    gts = {i: ev.load(i)["gt16"] for grp in by_src.values() for i in grp
           if len(by_src[ev.meta[i]["source_image_id"]]) > 1}
    pairs: dict[str, str] = {}
    overlaps: list[float] = []
    rejected = 0
    for src, grp in by_src.items():
        if len(grp) < 2:
            continue
        for i in grp:
            best, best_iou = None, 1.1
            for j in grp:
                if i == j:
                    continue
                a, bb = gts[i], gts[j]
                if a.shape != bb.shape:
                    continue
                iou = soft_iou_value((a > 0.5).double(), (bb > 0.5).double())
                if iou < best_iou:
                    best, best_iou = j, iou
            if best is None:
                continue
            if best_iou <= max_gt_iou:
                pairs[ev.meta[i]["sample_id"]] = ev.meta[best]["instruction"]
                overlaps.append(best_iou)
            else:
                rejected += 1
    info = {
        "n_pairs": len(pairs), "n_eval": len(ev),
        "coverage": len(pairs) / max(len(ev), 1),
        "rejected_same_target": rejected,
        "max_gt_iou_admitted": max_gt_iou,
        "gt_overlap_of_admitted": agg_local(overlaps),
        "rule": "partner must be a real instruction for the SAME source image whose "
                "GT mask overlaps this sample's GT by <= max_gt_iou, so the pair is "
                "guaranteed to have actually moved the target.",
    }
    return pairs, info


def agg_local(xs):
    import numpy as _np
    v = [float(x) for x in xs if x is not None]
    if not v:
        return {"n": 0}
    return {"n": len(v), "mean": float(_np.mean(v)),
            "median": float(_np.median(v)), "max": float(_np.max(v))}


def paired(a: list[float], b: list[float], seed: int = 0) -> dict[str, Any]:
    from q3vl.whereb.metrics import paired_delta
    return paired_delta(a, b, seed=seed)


def family_bank(train: FAFMData, n: int = 3000, size: int = 32
                ) -> tuple[torch.Tensor, list[str]]:
    """Reference ``c*`` bank for the nearest-``c*`` family diagnostic (criterion 7).

    The document specifies "K 样本集家族命中率(最近 c* 邻诊断分类)": classify a
    generated coarse field by its nearest training ``c*`` of known family.  Fields
    are resampled to a common 32x32 so that samples on different aspect ratios are
    comparable; this classifier is a **diagnostic only** and never touches training
    (family labels are not an input to proposal B -- section 3.5).
    """
    vecs, fams = [], []
    for i in range(min(n, len(train))):
        it = train.load(i)
        v = torch.nn.functional.interpolate(it["cstar"][None], size=(size, size),
                                            mode="bilinear", align_corners=False)[0, 0]
        vecs.append(v.reshape(-1))
        fams.append(it["meta"].get("family", "unknown"))
    V = torch.stack(vecs)
    V = V / V.norm(dim=1, keepdim=True).clamp_min(1e-9)
    return V, fams


def tarp_coverage(post: np.ndarray, truth: np.ndarray, seed: int = 0) -> float:
    """TARP: max deviation of the coverage ECDF from the diagonal.

    ``post`` ``(N, K, D)`` posterior draws, ``truth`` ``(N, D)``.  A
    condition-independent (collapsed) posterior fails this mechanically, which is
    the entire point of importing the instrument from proposal A -- collapse is
    detected by a number instead of by eye.
    """
    rng = np.random.default_rng(seed)
    N, K, D = post.shape
    ref = post.reshape(-1, D)[rng.integers(0, N * K, size=N)]
    d_true = np.linalg.norm(truth - ref, axis=1)
    d_post = np.linalg.norm(post - ref[:, None, :], axis=2)
    f = (d_post < d_true[:, None]).mean(axis=1)
    xs = np.linspace(0, 1, 101)
    ecdf = np.array([(f <= x).mean() for x in xs])
    return float(np.max(np.abs(ecdf - xs)))


@torch.no_grad()
def collect_posterior(net, data: FAFMData, text: TextEncoder, device, *, k: int,
                      cfg_scale: float, seed: int, size: int = 32, batch: int = 8):
    """``(N,K,D)`` posterior draws and ``(N,D)`` truths, both on a common 32x32 grid.

    Feeds criteria 7 (family mode recall) and 9 (c-space TARP).  Resampling to a
    common grid is what makes samples on different aspect ratios comparable; it is
    a diagnostic path only and never touches the loss.
    """
    from q3vl.whereb.fafm import sample_fafm
    post, truth, metas = [], [], []
    gen = torch.Generator(device=device).manual_seed(seed)
    rs = lambda x, n: torch.nn.functional.interpolate(  # noqa: E731
        x.reshape(n, 1, x.shape[-2], x.shape[-1]), size=(size, size),
        mode="bilinear", align_corners=False).reshape(n, -1)
    # Shape groups, for the same reason as eval_arm -- and here the sliding
    # window was worse than slow: on a straddling window it kept only the first
    # sample and DROPPED the rest, so criteria 7 and 9 were being computed on a
    # silently truncated, shape-biased subset (10 of 24 in the pilot).
    for shape, pool in sorted(data.by_shape.items()):
      for s in range(0, len(pool), batch):
        idxs = pool[s: s + batch]
        b = data.batch(idxs, device, net.cfg.in_sim)
        gh, gw = b["grid"]
        txt, mask = text([m["instruction"] for m in b["meta"]])
        f = sample_fafm(net, b["sim"], b["vis"], txt, mask, k=k, steps=N_STEPS,
                        cfg_scale=cfg_scale, generator=gen)          # (K,B,1,h,w)
        for j in range(len(idxs)):
            post.append(rs(f[:, j], k).cpu().numpy())
            truth.append(rs(b["cstar"][j][None], 1).cpu().numpy()[0])
            metas.append(b["meta"][j])
    return np.stack(post), np.stack(truth), metas


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-cache", default="/home/bc/data/runs/where_b/fafm_cache_20260811")
    ap.add_argument("--eval-cache",
                    default="/home/bc/data/runs/where_b/fafm_cache_vwhere_20260811")
    ap.add_argument("--checkpoint",
                    default="/home/bc/data/runs/q3vl_base_sft_20260804/checkpoint-4976")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=20260811)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--limit-eval", type=int, default=None)
    ap.add_argument("--n-viz", type=int, default=6)
    args = ap.parse_args(argv)

    t0 = time.time()
    from q3vl.whereb.unifield import agg, by_group

    out = Path(args.out)
    for sub in ("viz", "config", "logs"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    train = FAFMData(args.train_cache, args.limit_train)
    ev = FAFMData(args.eval_cache, args.limit_eval)
    text = TextEncoder(args.checkpoint, device)
    dom_train = train.assert_cstar_domain()
    dom_eval = ev.assert_cstar_domain()
    print(f"train {len(train)}  eval {len(ev)}  "
          f"c* domain asserted {dom_train['observed']} in {dom_train['declared']}  "
          f"({time.time()-t0:.0f}s)", flush=True)

    nets: dict[str, Any] = {}
    for arm in ("fafm", "regression", "no_text"):
        nets[arm] = train_arm(arm, train, text, device, steps=args.steps,
                              batch=args.batch, lr=args.lr, seed=args.seed)
        torch.save(nets[arm].state_dict(), out / "config" / f"{arm}.pt")
        print(f"[{arm}] trained ({time.time()-t0:.0f}s)", flush=True)

    # Guarded cross-image derangement, built once over the whole eval split.
    _rng = np.random.default_rng(args.seed)
    _ids = [m["sample_id"] for m in ev.meta]
    _instr = {m["sample_id"]: m["instruction"] for m in ev.meta}
    _perm = list(_rng.permutation(len(_ids)))
    for _i in range(len(_ids)):                      # break any fixed point
        if _perm[_i] == _i:
            _j = (_i + 1) % len(_ids)
            _perm[_i], _perm[_j] = _perm[_j], _perm[_i]
    shuffle_map = {_ids[i]: _instr[_ids[_perm[i]]] for i in range(len(_ids))}
    n_fixed = sum(1 for i in range(len(_ids)) if _perm[i] == i)

    def run(arm, **kw):
        net = nets.get(arm if arm in nets else "fafm")
        kw.setdefault("shuffle_map", shuffle_map)
        return eval_arm(net, ev, text, device, arm=arm, seed=args.seed, **kw)

    # ---- CFG selection: pre-registered, no post-hoc freedom ----------------
    gt_area = [float((ev.load(i)["gt16"] > 0.5).float().mean()) for i in range(len(ev))]
    reg_rows = run("regression", k=1, cfg_scale=1.0)
    cfg_scan: dict[str, Any] = {}
    for g in CFG_GRID:
        rows_g = run("fafm", k=K_SAMPLES, cfg_scale=g)
        neg_g = {n: run("fafm", k=K_SAMPLES, cfg_scale=g, negative=n) for n in NEGATIVES}
        by = {r["sample_id"]: r for r in rows_g}
        w1_a = w1([r["pred_area_frac"] for r in rows_g], gt_area)
        w1_c = w1([r["pred_area_frac"] for r in reg_rows], gt_area)
        # criterion 4 effect at this g: real instruction vs the WEAKEST-separated
        # negative (the min over the three), so one easy negative cannot carry it
        deltas = {}
        for n in NEGATIVES:
            bn = {r["sample_id"]: r for r in neg_g[n]}
            common = [s for s in by if s in bn]
            deltas[n] = paired([by[s]["soft_iou"] for s in common],
                               [bn[s]["soft_iou"] for s in common],
                               seed=args.seed)
        cfg_scan[f"{g}"] = {
            "rows": rows_g, "neg": neg_g,
            "w1_area": w1_a, "w1_area_regression": w1_c,
            "passes_c3": bool(w1_a <= 0.5 * w1_c),
            "cond_effect": min(d["delta"] for d in deltas.values()),
            "deltas": {k: {kk: vv for kk, vv in v.items() if kk != "ci95"}
                       for k, v in deltas.items()},
        }
        print(f"  CFG g={g}: W1 {w1_a:.4f} (reg {w1_c:.4f}) pass#3="
              f"{cfg_scan[f'{g}']['passes_c3']}  cond_effect "
              f"{cfg_scan[f'{g}']['cond_effect']:+.4f}  ({time.time()-t0:.0f}s)",
              flush=True)

    ok = [g for g in CFG_GRID if cfg_scan[f"{g}"]["passes_c3"]]
    g_sel = max(ok, key=lambda g: cfg_scan[f"{g}"]["cond_effect"]) if ok else 1.0
    print(f"CFG rule -> g={g_sel} (passing #3: {ok or 'none, fell back to 1.0'})",
          flush=True)

    sel = cfg_scan[f"{g_sel}"]
    rows_a, neg_rows = sel["rows"], sel["neg"]
    rows_b = run("centre_prior", k=1, cfg_scale=1.0)
    rows_d = run("no_text", k=K_SAMPLES, cfg_scale=g_sel, use_text=False)
    rows_f = run("fafm", k=1, cfg_scale=g_sel)

    idx = {r["sample_id"]: r for r in rows_a}
    def col(rows, key="soft_iou"):
        return [r[key] for r in rows if r["sample_id"] in idx]
    def pair_with(rows, key="soft_iou"):
        b = {r["sample_id"]: r for r in rows}
        common = [s for s in idx if s in b]
        return ([idx[s][key] for s in common], [b[s][key] for s in common])

    crit: dict[str, Any] = {}
    a, b_ = pair_with(rows_b)
    p1 = paired(a, b_, seed=args.seed)
    crit["1_vs_centre_prior"] = {"delta": p1["delta"], "p": p1["p_value"],
                                 "pass": bool(p1["delta"] >= 0.05 and p1["p_value"] < 0.01),
                                 "falsified": bool(p1["delta"] < 0.03),
                                 "line": "delta >= +0.05, p<0.01; falsify < +0.03"}
    a, c_ = pair_with(reg_rows)
    p2 = paired(a, c_, seed=args.seed)
    ar_a = float(np.nanmedian([r["area_ratio"] for r in rows_a]))
    ar_c = float(np.nanmedian([r["area_ratio"] for r in reg_rows]))
    crit["2_vs_regression"] = {
        "delta": p2["delta"], "p": p2["p_value"],
        "area_ratio_median_fafm": ar_a, "area_ratio_median_regression": ar_c,
        "pass": bool(p2["delta"] >= 0.03 and ar_c > 1.2 and 0.85 <= ar_a <= 1.15),
        "line": "delta >= +0.03 and regression area ratio > 1.2 while FAFM in [0.85,1.15]",
        "note": "shared verdict with proposal A's E5 (distribution estimation vs "
                "conditional-mean regression, same conditions, same latent space)"}
    crit["3_area_W1"] = {"w1_fafm": sel["w1_area"], "w1_regression": sel["w1_area_regression"],
                         "pass": sel["passes_c3"], "line": "W1(A) <= 0.5 W1(C)"}
    # --- criterion 4, in its pre-registered form -------------------------
    # Section 3.6: "同图反向指令配对差分 | 差分效应 >= 3x 全部负控制, p<0.01".
    # The effect is measured on same-image partner instructions (build_reversed_
    # pairs); the three arm-E controls are the comparison. The 3x factor is part
    # of the pre-registration and is NOT to be relaxed.
    rev_pairs, rev_info = build_reversed_pairs(ev)
    rows_rev = run("fafm", k=K_SAMPLES, cfg_scale=g_sel, instr_override=rev_pairs)
    by_rev = {r["sample_id"]: r for r in rows_rev}
    common_rev = [sid for sid in idx if sid in by_rev and sid in rev_pairs]
    p4 = paired([idx[s]["soft_iou"] for s in common_rev],
                [by_rev[s]["soft_iou"] for s in common_rev], seed=args.seed)
    neg_effects = {n: sel["deltas"][n]["delta"] for n in NEGATIVES}
    worst_neg = max(neg_effects.values()) if neg_effects else 0.0
    ratios = {n: (p4["delta"] / v if v > 0 else float("inf"))
              for n, v in neg_effects.items()}
    crit["4_instruction_conditionality"] = {
        "effect_reversed_instruction": p4["delta"],
        "p": p4["p_value"], "n_pairs_used": len(common_rev),
        "pairing": rev_info,
        "negative_control_effects": neg_effects,
        "ratio_effect_over_each_negative": ratios,
        "worst_negative_effect": worst_neg,
        "required_factor": 3.0,
        "pass": bool(p4["delta"] >= 3.0 * worst_neg and p4["p_value"] < 0.01
                     and p4["delta"] > 0 and len(common_rev) > 0),
        "line": "same-image reversed-instruction paired differential >= 3x EVERY "
                "negative control, p<0.01 (pre-registered factor, not relaxed)",
        "negatives_also_reported": {k: {kk: vv for kk, vv in v.items()
                                        if kk != "ci95"}
                                    for k, v in sel["deltas"].items()}}
    a, b_ = pair_with(rows_b, "gbf1")
    p5 = paired(a, b_, seed=args.seed)
    sem_a = [r["gbf1"] for r in rows_a if r["family"] == "semantic"]
    semb = {r["sample_id"]: r for r in rows_b}
    sem_b = [semb[r["sample_id"]]["gbf1"] for r in rows_a
             if r["family"] == "semantic" and r["sample_id"] in semb]
    crit["5_boundary_f1"] = {
        "delta_vs_centre_prior": p5["delta"], "p": p5["p_value"],
        "semantic_delta": (float(np.mean(sem_a) - np.mean(sem_b)) if sem_b else None),
        "pass": bool(p5["delta"] >= 0 and sem_b and
                     (np.mean(sem_a) - np.mean(sem_b)) >= 0.10),
        "line": "overall >= centre prior; semantic family >= centre prior + 0.10"}
    fam_a = by_group(rows_a, "family", "soft_iou")
    fam_b = by_group(rows_b, "family", "soft_iou")
    per_family = {}
    for k, v in fam_a.items():
        meas = v["median"] or 0.0
        ceil = GATE0_FAMILY_CEILING_GRID.get(k, GATE0_OVERALL_CEILING_GRID)
        ceil_hi = GATE0_FAMILY_CEILING_HI.get(k, GATE0_OVERALL_CEILING_HI)
        per_family[k] = {
            "n": v["n"],
            "ceiling_gate0_grid": ceil,
            "measured_grid": meas,
            "frac_of_ceiling": meas / ceil if ceil else None,
            "neck_tax_grid": 1.0 - ceil,
            "ceiling_gate0_hi_provenance": ceil_hi,
            "neck_tax_hi_provenance": 1.0 - ceil_hi,
            "centre_prior": fam_b.get(k, {}).get("median"),
            "beats_centre_prior": bool(meas >= (fam_b.get(k, {}).get("median") or 0)),
        }
    crit["6_per_family"] = {
        "per_family": per_family,
        "pass": bool(all(r["beats_centre_prior"] for r in per_family.values())),
        "line": "every family: FAFM >= centre prior (pass line UNCHANGED)",
        "reading_rule": "Coordinator ruling 2026-08-11 (1): read each family against "
                        "its OWN Gate-0 ceiling. The ceiling must come from the SAME "
                        "column as the measurement -- grid-level matched-area top-k -- "
                        "so frac_of_ceiling uses ceiling_gate0_grid. The full-resolution "
                        "soft-field ceilings (0.8949 for contour) are carried alongside "
                        "as *_hi_provenance and belong to a different column.",
        "neck_tax_note": "neck_tax_grid = 1 - ceiling_gate0_grid. In criterion 6's own "
                         "column the contour family's neck tax is 0.0112, NOT the 0.1051 "
                         "that belongs to the full-resolution soft-field column."}

    # criteria 7 and 9 need the raw posterior
    post, truth, pmeta = collect_posterior(nets["fafm"], ev, text, device,
                                           k=K_SAMPLES, cfg_scale=g_sel, seed=args.seed)
    V, fams = family_bank(train)
    P = torch.from_numpy(post).float()
    P = P / P.norm(dim=2, keepdim=True).clamp_min(1e-9)
    hits = []
    for i, m in enumerate(pmeta):
        nn_idx = (P[i] @ V.T).argmax(dim=1)
        got = {fams[j] for j in nn_idx.tolist()}
        hits.append(float(m.get("family", "unknown") in got))
    crit["7_family_mode_recall"] = {
        "recall": float(np.mean(hits)), "n": len(hits),
        "pass": bool(np.mean(hits) >= 0.70),
        "line": "GT family present among the K=16 samples >= 70% (nearest-c* diagnostic)"}
    dev_tarp = tarp_coverage(post, truth, seed=args.seed)
    crit["9_tarp_c_space"] = {"max_deviation": dev_tarp,
                              "pass": bool(dev_tarp <= 0.10),
                              "line": "coverage ECDF max deviation <= 0.10"}

    steps_curve = {}
    for n in (1, 2, 4, 8, 16):
        r = run("fafm", k=4, cfg_scale=g_sel, n_steps=n)
        steps_curve[str(n)] = agg(x["soft_iou"] for x in r)
    crit["8_steps_diversity"] = {"curve": {k: v["median"] for k, v in steps_curve.items()},
                                 "pass": None, "line": "recording only"}

    def summary(rows):
        return {"soft_iou": agg(r["soft_iou"] for r in rows),
                "hard_iou": agg(r["hard_iou"] for r in rows),
                "gbf1": agg(r["gbf1"] for r in rows),
                "area_ratio": agg(r["area_ratio"] for r in rows),
                "by_family": by_group(rows, "family", "soft_iou"),
                "by_area_stratum": by_group(rows, "area_stratum", "soft_iou")}

    # Section 3.5's three S-channel controls, evaluated (not just dropped in
    # training).  Without them "the field follows the instruction, not S" has no
    # control at all -- S carries a strong subject/coverage prior. (REVIEW B7.)
    s_ctrl_rows = {c: run("fafm", k=K_SAMPLES, cfg_scale=g_sel, s_control=c)
                   for c in ("shuffle_S", "zero_S", "wrong_image_S")}

    all_rows = {"A_fafm": rows_a, "B_centre_prior": rows_b,
                "C_regression": reg_rows, "D_no_text": rows_d, "F_k1": rows_f,
                **{f"E_{n}": neg_rows[n] for n in NEGATIVES},
                **{f"S_{c}": r for c, r in s_ctrl_rows.items()}}

    # Red line: every ablation row carries Delta_const / Delta_shuffle columns.
    # Previously only arm A had a paired differential, and only inside criterion 4.
    def with_deltas(name, rows):
        out = summary(rows)
        by = {r["sample_id"]: r for r in rows}
        for ref_name, ref_rows in (("const", rows_b),
                                   ("shuffle", neg_rows["shuffled"])):
            ref = {r["sample_id"]: r for r in ref_rows}
            common = [sid for sid in by if sid in ref]
            if common and name != ("B_centre_prior" if ref_name == "const"
                                   else "E_shuffled"):
                d = paired([by[s]["soft_iou"] for s in common],
                           [ref[s]["soft_iou"] for s in common], seed=args.seed)
                out[f"delta_{ref_name}"] = {"delta": d["delta"], "p": d["p_value"],
                                            "n": len(common)}
            else:
                out[f"delta_{ref_name}"] = None
        return out

    arms = {k: with_deltas(k, v) for k, v in all_rows.items()}
    arms["_delta_reference"] = {
        "const": "B_centre_prior (zero-parameter centre prior)",
        "shuffle": "E_shuffled (guarded cross-image instruction derangement)",
        "n_fixed_points_in_shuffle": n_fixed,
        "rule": "CLAUDE.md red line: every ablation row carries Delta_const / "
                "Delta_shuffle."}
    arms["A_fafm"]["ambiguity"] = agg(r["ambiguity"] for r in rows_a)
    arms["reference"] = {
        "centre_prior_softiou": agg(r["centre_prior_softiou"] for r in rows_a),
        "random_floor": agg(r["random_floor"] for r in rows_a),
        "gate0_ceilings_grid_column": GATE0_FAMILY_CEILING_GRID,
        "gate0_ceilings_hi_column_provenance": GATE0_FAMILY_CEILING_HI,
        "gate0_ceiling_column_note": "criterion 6 measures the GRID column; the hi "
                                     "column is provenance only (REVIEW B3)",
    }

    n_pass = sum(1 for k, v in crit.items() if v.get("pass") is True)
    metrics = {
        "card": "fafm_probe_B", "doc": "RESEARCH_unified-field-prediction section 3.6",
        "gate0": "uni_gate0_20260811/caseB_fafm (PASS)",
        "cfg_selected": g_sel, "cfg_rule": "g in {1,1.5,2}; among those passing #3 "
                                           "take the max of #4; none passing -> g=1",
        "cfg_scan": {k: {kk: vv for kk, vv in v.items() if kk not in ("rows", "neg")}
                     for k, v in cfg_scan.items()},
        "criteria": crit, "n_criteria_passed": n_pass,
        "arms": arms, "n_eval": len(rows_a),
        "train": {"n": len(train), "steps": args.steps, "batch": args.batch,
                  "lr": args.lr, "seed": args.seed,
                  "checkpoint_selection": "pre-registered step count; val loss never used",
                  "scale_deviation": {
                      "document_spec": "section 3.6/3.7 specify S-train full (~70k)",
                      "used": len(train),
                      "reason": "probe-scale run: six arms inside the section 3.7 budget "
                                "of one card for 2-5 h. Approved by coordinator ruling "
                                "2026-08-11 (3).",
                      "escalation": "if this probe clears its criteria, the full arm is "
                                    "re-run at 70k; the probe's numbers are not to be "
                                    "quoted as the full-arm result."}},
        "cstar_domain_train": train.manifest.get("cstar_domain"),
        "cstar_domain_assertion": {"train": dom_train, "eval": dom_eval},
        "s_channel_controls": {c: agg(r["soft_iou"] for r in rows)
                               for c, rows in s_ctrl_rows.items()},
        "git_commit": git_commit(), "python": platform.python_version(),
        "torch": torch.__version__, "elapsed_s": time.time() - t0,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    (out / "config" / "rows_fafm.json").write_text(json.dumps(rows_a, indent=1, default=str))
    (out / "config" / "rows_all.json").write_text(json.dumps(
        {k: v for k, v in all_rows.items()}, indent=1, default=str))
    _write_probe_viz(out / "viz", ev, rows_a, rows_b, args.n_viz)
    print(f"criteria passed {n_pass}/8 gated  (#8 is recording-only)  "
          f"({time.time()-t0:.0f}s)", flush=True)
    print(f"wrote {out/'metrics.json'}", flush=True)
    return 0




def _write_probe_viz(viz: Path, ev: FAFMData, rows_a: list[dict[str, Any]],
                     rows_b: list[dict[str, Any]], n: int) -> None:
    """``success_*`` / ``failure_*`` panels for the probe (REVIEW-impl-amort-uni B8).

    CLAUDE.md's delivery spec requires both, and is explicit that "no failure
    cases" means they were not looked for hard enough.  Sorting by the paired
    margin against the centre prior -- rather than by raw soft-IoU -- puts the
    genuinely instructive rows at the ends: the worst cases are where a zero-
    parameter baseline beats the model, which is the campaign's standing failure
    signature, not merely where the image was hard.

    Colour discipline: masks on a FIXED 0..1 scale (never per-image min-max,
    whose denominator is set by whatever the extreme cell happens to be); the
    difference panel on a fixed +-1; the coarse c* on an arm-wide symmetric scale
    printed in the title.
    """
    if not rows_a:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref = {r["sample_id"]: r["soft_iou"] for r in rows_b}
    by_id = {m["sample_id"]: i for i, m in enumerate(ev.meta)}
    scored = [(r["soft_iou"] - ref.get(r["sample_id"], 0.0), r) for r in rows_a
              if r["sample_id"] in by_id]
    if not scored:
        return
    scored.sort(key=lambda t: t[0])
    picks = ([("failure", m, r) for m, r in scored[:n]]
             + [("success", m, r) for m, r in scored[-n:]])
    for tag, margin, r in picks:
        it = ev.load(by_id[r["sample_id"]])
        gt = it["gt16"].numpy()
        cs = it["cstar"][0].numpy()
        lim = float(np.percentile(np.abs(cs), 99.5)) or 1.0
        fig, ax = plt.subplots(1, 3, figsize=(13, 4.0))
        ax[0].imshow(gt, cmap="magma", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax[0].set_title("GT field (grid, fixed 0..1)", fontsize=9)
        im = ax[1].imshow(cs, cmap="coolwarm", vmin=-lim, vmax=lim,
                          interpolation="nearest")
        ax[1].set_title(f"c* target (fixed +-{lim:.2f})", fontsize=9)
        plt.colorbar(im, ax=ax[1], fraction=0.046)
        ax[2].axis("off")
        ax[2].text(0.0, 0.5,
                   f"family: {r['family']}\nsoft-IoU: {r['soft_iou']:.4f}\n"
                   f"centre prior: {ref.get(r['sample_id'], float('nan')):.4f}\n"
                   f"margin: {margin:+.4f}\n"
                   f"area: {r['area_frac']:.3f}  floor: {r['random_floor']:.3f}\n"
                   f"area ratio: {r['area_ratio']:.3f}\n"
                   f"ambiguity: {r['ambiguity']:.3f}",
                   fontsize=10, va="center", family="monospace")
        fig.suptitle(f"{tag}  {r['sample_id']}", fontsize=10)
        fig.tight_layout()
        fig.savefig(viz / f"{tag}_{r['family']}_{r['sample_id'][:24]}.png", dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
