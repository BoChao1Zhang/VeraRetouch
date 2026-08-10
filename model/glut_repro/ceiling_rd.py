"""Reachability pre-check: is "Delta_shuffle >= 3 dB" attainable on this level,
on this target variant, at all?

READ THIS FIRST -- what the number is and is NOT
------------------------------------------------
Delta*_shuffle below is **attained** by a concrete estimator that is fitted on
S-train and scored on S-val through the same harness call the trained arms use.
It is therefore a CERTIFIED LOWER BOUND on the ceiling, and it is only ever
used in that direction:

    Delta*_shuffle >= 3 dB   =>  the >= 3 dB criterion is PROVABLY reachable
                                 on this level  =>  the level is eligible for
                                 the s-axis criterion and for RD-E's veto.
    Delta*_shuffle small     =>  NOTHING is proven.  Not "unreachable".  The
                                 level is marked N/A, never FAIL.

It is NOT an upper bound and must never be quoted as "the ceiling": the fitted
arms beat it on in-mask PSNR (L1/fixed reference 30.3 dB vs RD-STD 41.9 dB at
2k steps), because the table is coarse along s (8 quantile buckets over a
strongly bimodal mask) where the arms are continuous.  This is exactly the trap
G3 fell into -- its "43.66 / 36.54 ceiling" was crossed in three places and has
since been renamed to "binned-LUT reference value" (main agent, 2026-08-03).
The only level here with a genuine UPPER bound is L0, and that bound is
algebraic rather than statistical: the mask is all-ones, so every image carries
the identical s field, the cross-image shuffle is the identity map, and
Delta_shuffle is exactly 0 for any model whatsoever.

Why this runs BEFORE any training (task card, G3 NOTES decision 1 / D-26):
D-CONSTRUCT's as-shipped targets mix up to 40 transform identities
(class|tier|sign) per level, while oracle s encodes only WHERE and never WHICH.
The best any f(x,s) can then do is the conditional mean E[y|x,s], and with
near-symmetric signs that mean regresses to near-identity.  G3 measured
Delta_shuffle* = +0.32 dB on the as-shipped L1+L4 -- so on that data BOTH arms
score below the 0.3 dB veto line no matter how good they are, and RD-E's
"< 0.3 dB -> stop all renderer work" gate would fire on a data artefact.
A criterion evaluated outside the range its data can express is not a criterion.

What is measured
----------------
A non-parametric REFERENCE estimator for the arm family: a piecewise-affine table
over (s quantile bucket) x (hierarchical RGB bin), fitted on S-train pixels and
scored on S-val images through the same tools/harness calls the trained models
use.  Two forms, both reported:

  3D form   one hierarchy over RGB, no s        -> PSNR_3D*
  4D form   each s bucket builds its OWN RGB hierarchy = "N 3D LUTs indexed by
            s", which is literally RD-E's function class -> PSNR_4D*

Estimator caliber = D-24: per-cell least-squares AFFINE, not the literal
per-cell conditional mean.  Local linearity is what a trilinear 3D LUT actually
does inside one lattice cell; the constant estimator has a hard ~41 dB
quantisation floor that swallows D-CONSTRUCT's edit magnitudes whole.

The table is strictly richer than either arm in RGB (thousands of cells vs 32
Gaussians / 17^3 lattice) and strictly POORER in s (8 buckets, piecewise
constant vs continuous) -- which is precisely why it under-reads and why only
the lower-bound direction above is sound.  The controls that keep it honest:
  * Delta_const* and Delta_shuffle* of the 3D form are 0 by construction
    (no s enters) -- printed as an implementation self-check.
  * The same fit/score split (S-train -> S-val) as the trained arms, so finite
    sample inflation shows up as generalisation loss, not as free ceiling.
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "tools", "harness"))

from model.glut_repro import data_rd as D                       # noqa: E402
import collapse_probes as cp                                    # noqa: E402
from metrics import masked_psnr, psnr_full                       # noqa: E402

LEVELS_RGB = (33, 17, 9)
N_MIN_SUPPORT = 60          # >= 12 dof per affine cell (delta_ceil N_MIN_AFFINE)
N_MIN_FIT = 24              # a slot whose *assigned* members fall below this
                            # redirects to its parent (delta_ceil does the same
                            # with its three support tiers)
N_S_BUCKETS = 8
RIDGE = 1e-6


def _color_code(x: torch.Tensor, k: int) -> torch.Tensor:
    q = torch.clamp((x * k).floor().to(torch.int64), 0, k - 1)
    return (q[:, 0] * k + q[:, 1]) * k + q[:, 2]


class CeilTable:
    """Piecewise-affine (s bucket x hierarchical RGB bin) lookup.

    Slot layout (a flat index space so assignment is one gather):
        0                        global
        1 + b                    bucket b
        base_k + b*k^3 + code    bucket b, RGB level k in (9, 17, 33)
    Assignment picks the FINEST supported slot, where "supported" = at least
    N_MIN_SUPPORT training pixels fell in that raw bin.  The same rule runs at
    fit and at predict time, so the val pixels are routed exactly as the train
    pixels were.
    """

    def __init__(self, n_buckets: int, levels=LEVELS_RGB,
                 n_min: int = N_MIN_SUPPORT, device: str = "cuda"):
        self.nb = int(n_buckets)
        self.levels = tuple(levels)
        self.n_min = int(n_min)
        self.device = device
        self.base = {}
        off = 1 + self.nb
        for k in sorted(self.levels):                 # 9, 17, 33
            self.base[k] = off
            off += self.nb * k ** 3
        self.n_slots = off
        self.edges: torch.Tensor | None = None

    # ------------------------------------------------------------------ misc
    def buckets(self, s: torch.Tensor) -> torch.Tensor:
        if self.nb == 1 or self.edges is None or self.edges.numel() == 0:
            return torch.zeros_like(s, dtype=torch.int64)
        return torch.bucketize(s, self.edges, right=False).clamp(0, self.nb - 1)

    def _slots_all_levels(self, x: torch.Tensor, b: torch.Tensor) -> list:
        return [(k, self.base[k] + b * (k ** 3) + _color_code(x, k))
                for k in sorted(self.levels, reverse=True)]     # 33, 17, 9

    def assign(self, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        slot = torch.full_like(b, -1)
        for _, sl in self._slots_all_levels(x, b):
            hit = self.supported[sl] & (slot < 0)
            slot = torch.where(hit, sl, slot)
        bslot = 1 + b
        hit = self.supported[bslot] & (slot < 0)
        slot = torch.where(hit, bslot, slot)
        return torch.where(slot < 0, torch.zeros_like(slot), slot)

    # ------------------------------------------------------------------- fit
    def fit(self, x: np.ndarray, s: np.ndarray, y: np.ndarray) -> "CeilTable":
        dev = self.device
        xt = torch.as_tensor(x, dtype=torch.float32, device=dev)
        st = torch.as_tensor(s, dtype=torch.float32, device=dev)
        yt = torch.as_tensor(y, dtype=torch.float32, device=dev)
        p = xt.shape[0]

        if self.nb > 1:
            qs = torch.linspace(0, 1, self.nb + 1, device=dev)[1:-1]
            ref = st
            if p > 1_000_000:
                g = torch.Generator(device=dev).manual_seed(0)
                ref = st[torch.randint(0, p, (1_000_000,), device=dev,
                                       generator=g)]
            self.edges = torch.unique(torch.quantile(ref, qs))
        else:
            self.edges = torch.zeros(0, device=dev)
        b = self.buckets(st)

        # raw bin counts -> supported mask
        self.supported = torch.zeros(self.n_slots, dtype=torch.bool, device=dev)
        cnt_all = torch.zeros(self.n_slots, device=dev)
        for _, sl in self._slots_all_levels(xt, b):
            cnt_all.index_add_(0, sl, torch.ones(p, device=dev))
        cnt_all.index_add_(0, 1 + b, torch.ones(p, device=dev))
        cnt_all[0] = p
        self.supported = cnt_all >= self.n_min

        slot = self.assign(xt, b)
        n_assigned = torch.zeros(self.n_slots, device=dev)
        n_assigned.index_add_(0, slot, torch.ones(p, device=dev))

        # least squares affine per slot (accumulated in pixel chunks: the
        # outer products are P x 16 doubles and would be ~1 GB in one go)
        gram = torch.zeros(self.n_slots, 16, device=dev, dtype=torch.float64)
        rhs = torch.zeros(self.n_slots, 12, device=dev, dtype=torch.float64)
        pchunk = 1 << 20
        for i in range(0, p, pchunk):
            j = min(i + pchunk, p)
            xh = torch.cat([xt[i:j], torch.ones_like(xt[i:j, :1])], 1).double()
            yd = yt[i:j].double()
            sl = slot[i:j]
            gram.index_add_(0, sl,
                            (xh.unsqueeze(2) * xh.unsqueeze(1)).reshape(-1, 16))
            rhs.index_add_(0, sl,
                           (xh.unsqueeze(2) * yd.unsqueeze(1)).reshape(-1, 12))
        eye = torch.eye(4, device=dev, dtype=torch.float64).unsqueeze(0)
        theta = torch.empty(self.n_slots, 4, 3, device=dev, dtype=torch.float32)
        chunk = 1 << 16
        for i in range(0, self.n_slots, chunk):
            j = min(i + chunk, self.n_slots)
            g4 = gram[i:j].reshape(-1, 4, 4) + RIDGE * eye
            theta[i:j] = torch.linalg.solve(
                g4, rhs[i:j].reshape(-1, 4, 3)).float()
        self.theta = theta

        # a slot that ended up with too few *assigned* members redirects to its
        # parent (the next coarser level, then its bucket, then global)
        valid = n_assigned >= N_MIN_FIT
        valid[0] = True
        self.redirect = torch.arange(self.n_slots, device=dev)
        parent = torch.arange(self.n_slots, device=dev)
        order = sorted(self.levels, reverse=True)                # 33, 17, 9
        for li, k in enumerate(order):
            idx = torch.arange(self.nb * k ** 3, device=dev)
            bb = idx // (k ** 3)
            code = idx % (k ** 3)
            r = (code // (k * k)).double() / k
            g = ((code // k) % k).double() / k
            bl = (code % k).double() / k
            centre = torch.stack([r, g, bl], 1).float() + 0.5 / k
            if li + 1 < len(order):
                kk = order[li + 1]
                par = self.base[kk] + bb * (kk ** 3) + _color_code(centre, kk)
            else:
                par = 1 + bb
            parent[self.base[k] + idx] = par
        parent[1:1 + self.nb] = 0
        parent[0] = 0
        for _ in range(len(self.levels) + 2):
            self.redirect = torch.where(valid[self.redirect], self.redirect,
                                        parent[self.redirect])
        self.n_assigned = n_assigned
        self.stats = {
            "n_train_px": int(p),
            "n_slots": int(self.n_slots),
            "n_slots_used": int((n_assigned > 0).sum()),
            "n_slots_valid_used": int(((n_assigned > 0) & valid).sum()),
            "n_s_buckets_effective": int(b.max().item()) + 1,
        }
        return self

    # --------------------------------------------------------------- predict
    @torch.no_grad()
    def predict(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        b = self.buckets(s)
        slot = self.redirect[self.assign(x, b)]
        xh = torch.cat([x, torch.ones_like(x[:, :1])], 1)
        return torch.einsum("pi,pij->pj", xh, self.theta[slot])

    @torch.no_grad()
    def render(self, image: np.ndarray, s_cache: np.ndarray,
               chunk: int = 1 << 18) -> np.ndarray:
        img = np.asarray(image, dtype=np.float32)
        h, w = img.shape[:2]
        s_full = (D.s32_to_full(np.asarray(s_cache, dtype=np.float32), (h, w))
                  if np.asarray(s_cache).shape != (h, w) else
                  np.asarray(s_cache, dtype=np.float32))
        flat = img.reshape(-1, 3)
        sf = s_full.reshape(-1)
        out = np.empty_like(flat)
        for i in range(0, flat.shape[0], chunk):
            j = min(i + chunk, flat.shape[0])
            xb = torch.from_numpy(flat[i:j]).to(self.device)
            sb = torch.from_numpy(sf[i:j]).to(self.device)
            out[i:j] = self.predict(xb, sb).clamp(0, 1).cpu().numpy()
        return out.reshape(img.shape)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_level(level: str, variant: str, device: str = "cuda",
              px_per_img: int = 30000, n_val: int = 24,
              n_buckets: int = N_S_BUCKETS, seed: int = 0,
              cache_dir: str = "/var/cache/veradata/rd") -> dict:
    tr = D.load_pairs("train", (level,))
    va = D.even_subset(D.load_pairs("val", (level,)), n_val)
    # exactly the same pixel cache the trained arms consume: same pairs, same
    # subsample seed, same s pipeline.  Reference and arms therefore see one
    # dataset, not two.
    cpath = os.path.join(cache_dir,
                         f"train_{level}_{variant}_c32_px{px_per_img}.npz")
    if os.path.exists(cpath):
        cache = D.load_cache(cpath)
    else:
        cache = D.build_pixel_cache(tr, variant, px_per_img=px_per_img, seed=0,
                                    verbose=False)
        D.save_cache(cache, cpath)
    samples = []
    for p in va:
        d = D.read_pair(p, variant)
        samples.append({"id": d["uid"], "image": d["x"], "gt": d["y"],
                        "s": d["s_cache"], "mask": d["mask"]})

    x = cache["x"].astype(np.float32)
    y = cache["y"].astype(np.float32)
    s = cache["s"].astype(np.float32)

    t4 = CeilTable(n_buckets, device=device).fit(x, s, y)
    t3 = CeilTable(1, device=device).fit(x, s, y)

    s_null = np.full_like(np.asarray(samples[0]["s"], dtype=np.float32),
                          float(np.mean([np.mean(sm["s"]) for sm in samples])))

    def probes(tbl):
        r = cp.run_probes(lambda im, ss: tbl.render(im, ss), samples,
                          s_null=s_null, seed=seed)
        mp = {k: [] for k in ("psnr_in", "psnr_band", "psnr_out")}
        for sm in samples:
            m = masked_psnr(tbl.render(sm["image"], sm["s"]), sm["gt"],
                            sm["mask"])
            for k in mp:
                mp[k].append(m[k])
        r.update({k: float(np.nanmean(v)) for k, v in mp.items()})
        return r

    p4, p3 = probes(t4), probes(t3)
    ident = {
        "psnr_full": float(np.mean([psnr_full(sm["image"], sm["gt"])
                                    for sm in samples])),
        "psnr_in": float(np.nanmean(
            [masked_psnr(sm["image"], sm["gt"], sm["mask"])["psnr_in"]
             for sm in samples])),
        "psnr_band": float(np.nanmean(
            [masked_psnr(sm["image"], sm["gt"], sm["mask"])["psnr_band"]
             for sm in samples])),
    }
    return {
        "level": level, "variant": variant,
        "n_train_pairs": len(tr), "n_val_pairs": len(samples),
        "identity": ident,
        "ceiling_4d": p4, "ceiling_3d": p3,
        "delta_shuffle_star": p4["delta_shuffle"],
        "delta_const_star": p4["delta_const"],
        "delta_vs3d_star": p4["psnr_true"] - p3["psnr_true"],
        "delta_vs3d_in_star": p4["psnr_in"] - p3["psnr_in"],
        "criterion_3db_proven_reachable": bool(p4["delta_shuffle"] >= 3.0),
        "s_gain_generalizes": bool(p4["psnr_true"] - p3["psnr_true"] > 0.0),
        "estimator_role": "LOWER bound on the ceiling (attained by a fitted "
                          "non-parametric reference); never an upper bound",
        "selfcheck_3d_delta_shuffle_is_zero":
            bool(abs(p3["delta_shuffle"]) < 1e-6),
        "selfcheck_3d_delta_const_is_zero":
            bool(abs(p3["delta_const"]) < 1e-6),
        "table_stats_4d": t4.stats, "table_stats_3d": t3.stats,
    }


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="*", default=list(D.LEVELS))
    ap.add_argument("--variants", nargs="*", default=["fixed", "mixed"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--px-per-img", type=int, default=30000)
    ap.add_argument("--n-val", type=int, default=24)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache-dir", default="/var/cache/veradata/rd")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows = []
    if os.path.exists(args.out):
        with open(args.out) as f:
            rows = json.load(f).get("rows", [])
    done = {(r["level"], r["variant"]) for r in rows}
    for variant in args.variants:
        for level in args.levels:
            if (level, variant) in done:
                print(f"[ceil] SKIP {level}/{variant}", flush=True)
                continue
            r = run_level(level, variant, device=args.device,
                          px_per_img=args.px_per_img, n_val=args.n_val,
                          cache_dir=args.cache_dir)
            rows.append(r)
            print(f"[ceil] {level}/{variant}: "
                  f"D_shuf* {r['delta_shuffle_star']:+.2f}  "
                  f"D_const* {r['delta_const_star']:+.2f}  "
                  f"vs3D* {r['delta_vs3d_star']:+.2f}  "
                  f"in-mask vs3D* {r['delta_vs3d_in_star']:+.2f}  "
                  f"3D-ctrl zero={r['selfcheck_3d_delta_shuffle_is_zero']}",
                  flush=True)
            with open(args.out, "w") as f:
                json.dump({"rows": rows}, f, indent=1)
    print("CEILING_DONE", flush=True)


if __name__ == "__main__":
    main()
