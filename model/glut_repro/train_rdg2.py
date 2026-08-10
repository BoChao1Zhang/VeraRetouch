"""RD-G Stage-2: the same generators on D-CONSTRUCT with an oracle s axis.

Stage-1 asks "how many dB is generator capacity worth when the target is a
global colour transform".  Stage-2 asks whether that ordering survives the move
to the actual renderer of the project: 4-D anchored Gaussians (R-2), oracle s
from the GT mask, the L0-L7 capacity ladder.

Task: condition = (I_in, I_target) at 128x128 (the same 6-channel stack as
Stage-1, so the generator is unchanged except for the wider primitive head);
output = one 27N+12 4-D GLUT parameter set per image; loss = L1 between
f(x, s) and I_target on sampled pixels.  s is the GT mask, read at the INF-5
production caliber (mask -> 32x32 BOX -> bilinear back up), never per-image
normalised (red line).

Delta_const / Delta_shuffle here are the s-axis versions of PLAN M1/M2:
  Delta_const   = PSNR(true s) - PSNR(s == 1 everywhere)
  Delta_shuffle = PSNR(true s) - PSNR(another image's s field)
Both are reported per level.  The generator-side condition dropout of Stage-1 is
kept as well, so a collapsed generator is still visible.

Red lines specific to this stage:
  * mu_s is a frozen K=6 grid and sigma_s is sigmoid-bounded (R-2) -- structural,
    not penalised.
  * NO smoothing regulariser on the s axis.
  * R-3's diversity hinge uses the coordinator's relative gamma_mu,
    0.4 * std(anchor grid), instead of the absolute 0.15 that silently idles
    when the s range is not [0,1].
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/home/bc/VeraRetouch")
from model.glut_repro.model_rdg import (                           # noqa: E402
    ARMS, K_ANCHOR, N_PRIM_OUT_4D, SIGMA_S_MAX, ParamHead4D, PairTokenizer,
    MLPGen, TransformerGen, delta_e00, render4d,
)

EXP = "/home/bc/VeraRetouch/experiments/RDG_transformer_20260803"
CONSTRUCT = ("/home/bc/VeraRetouch/experiments/tooling-wave1/"
             "T4_construct/sanity")
LEVELS = ("L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7")
S_SIZE = 32


# ---------------------------------------------------------------------------
class ConstructSet(torch.utils.data.Dataset):
    """D-CONSTRUCT L0-L7 at a fixed working resolution.

    Oracle s (RO-0 caliber, copied from data_rd.py's documented recipe):
    mask/255 -> PIL BOX to 32x32 -> bilinear back to the working resolution.
    No per-image min-max or softmax anywhere (red line).
    """

    def __init__(self, split: str, res: int = 128, levels=LEVELS):
        self.res, self.rows = res, []
        for lv in levels:
            for p in sorted(glob.glob(os.path.join(
                    CONSTRUCT, split, lv, "*_in.png"))):
                stem = p[:-7]
                if os.path.isfile(stem + "_out.png") and \
                        os.path.isfile(stem + "_mask.png"):
                    self.rows.append((lv, stem))
        # the whole ladder is ~1.8k triples; decoded once at 128x128 it is
        # ~270 MB, so it is held in RAM and the loader never touches disk again.
        # The decode itself is ~8 min of 1024px PNGs on this (loaded) host, so
        # it is memoised next to the experiment and reused by every arm.
        n = len(self.rows)
        memo = os.path.join(EXP, "cache", f"construct_{split}_{res}.npz")
        if os.path.isfile(memo):
            z = np.load(memo)
            self.x, self.y, self.s, self.lv = z["x"], z["y"], z["s"], z["lv"]
            return
        self.x = np.empty((n, res, res, 3), np.uint8)
        self.y = np.empty((n, res, res, 3), np.uint8)
        self.s = np.empty((n, res, res), np.float32)
        self.lv = np.empty(n, np.int64)
        for i, (lv, stem) in enumerate(self.rows):
            self.x[i] = np.asarray(Image.open(stem + "_in.png").convert("RGB")
                                   .resize((res, res), Image.BOX), np.uint8)
            self.y[i] = np.asarray(Image.open(stem + "_out.png").convert("RGB")
                                   .resize((res, res), Image.BOX), np.uint8)
            m = Image.open(stem + "_mask.png").convert("L")
            s32 = m.resize((S_SIZE, S_SIZE), Image.BOX)
            self.s[i] = np.asarray(s32.resize((res, res), Image.BILINEAR),
                                   np.float32) / 255.0
            self.lv[i] = LEVELS.index(lv)
        os.makedirs(os.path.dirname(memo), exist_ok=True)
        np.savez(memo, x=self.x, y=self.y, s=self.s, lv=self.lv)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return (torch.from_numpy(self.x[i]), torch.from_numpy(self.y[i]),
                torch.from_numpy(self.s[i]), int(self.lv[i]))


def to_gpu(batch, dev):
    x = torch.stack([b[0] for b in batch]).to(dev).permute(0, 3, 1, 2).float() / 255
    y = torch.stack([b[1] for b in batch]).to(dev).permute(0, 3, 1, 2).float() / 255
    s = torch.stack([b[2] for b in batch]).to(dev)
    lv = torch.tensor([b[3] for b in batch], device=dev)
    return x, y, s, lv


def load_s_arm_recipe(arm_dir: str) -> dict:
    """Read a scache arm's own `_ARM_INFO.json` and return its consume recipe.

    NEVER assume an s cache lives in [0,1].  The RO-3 fused arm's global domain
    is [-3.256, 8.805] with p50 = -0.097 and 58.8% of cells negative, and its
    own WARNING records that the ro9 arm was already silently destroyed by the
    default clamp=(0,1) (0.93% of cells survived, with no error raised).
    Measured here first-hand over 193 of its 772 entries:
      per-image frac<0      p50 0.613, max 0.879
      default clamp(0,1)    p50 38.7% of cells survive, worst entry 12.1%
    So the default path silently throws away ~60% of the field.

    The recipe implemented is the arm's own "鲁棒做法": winsorize at p1/p99 then
    map linearly onto [0,1].  Verified on entry 0: min 0.107, max 1.000,
    100% of cells non-zero.
    """
    with open(os.path.join(arm_dir, "_ARM_INFO.json")) as f:
        info = json.load(f)
    q = info.get("quantiles", {})
    lo = q.get("p1", info["domain_global"][0])
    hi = q.get("p99", info["domain_global"][1])
    return {"lo": float(lo), "hi": float(hi), "info": info,
            "no_per_image_norm": bool(info.get("no_per_image_norm", True))}


def s_to_unit(s, lo: float, hi: float):
    """Winsorize to [lo,hi] then map to [0,1].  GLOBAL constants only --
    per-image min-max on s is a project red line."""
    import numpy as _np
    if isinstance(s, torch.Tensor):
        return (s.clamp(lo, hi) - lo) / (hi - lo)
    return (_np.clip(s, lo, hi) - lo) / (hi - lo)


def assert_s_matches_declared_domain(raw, declared_lo: float,
                                     declared_hi: float,
                                     max_outside: float = 0.005) -> None:
    """The check that actually catches the ro9 trap.

    `assert_s_in_anchor_domain` catches "the field lives nowhere near the
    anchors".  It does NOT catch the ro9 failure, because a field clamped to
    (0,1) lands squarely INSIDE the anchor domain -- it is just mostly zero.
    Measured on the RO-3 fused arm: raw cells orphaned from the anchors are only
    1.6% at p50, so the orphan guard stays quiet, while a default clamp(0,1)
    silently destroys 61.3% of the field (p50 over 193 entries).

    So the consumer must DECLARE the domain it is consuming in, and the raw data
    must actually live there.  Oracle masks declare [0,1] and do live there.
    The RO-3 arm declares [-3.256, 8.805] and cannot be consumed as [0,1]
    without this raising.
    """
    import numpy as _np
    a = raw.detach().cpu().numpy() if isinstance(raw, torch.Tensor) else _np.asarray(raw)
    outside = float(((a < declared_lo) | (a > declared_hi)).mean())
    if outside > max_outside:
        raise ValueError(
            f"s cache does not live in the declared domain "
            f"[{declared_lo:.3f}, {declared_hi:.3f}]: {100*outside:.1f}% of "
            f"cells are outside it (raw range [{a.min():.3f}, {a.max():.3f}], "
            f"{100*(a < declared_lo).mean():.1f}% below). Consuming it anyway "
            f"would clamp them to a boundary and silently delete the field "
            f"(the ro9 failure: 0.93% of cells survived, no error raised). "
            f"Read the arm's _ARM_INFO.json consume_recipe and use "
            f"load_s_arm_recipe()+s_to_unit().")


def assert_s_in_anchor_domain(s: torch.Tensor, mu_s: torch.Tensor,
                              sigma_s_max: float = SIGMA_S_MAX,
                              max_orphan: float = 0.05) -> None:
    """Fail LOUDLY if the s field does not live where the anchors are.

    The failure this guards against is silent: with mu_s pinned to a K=6 grid on
    [0,1] and sigma_s <= 0.30, any pixel further than ~3 sigma from every anchor
    gets zero density from every Gaussian, the normaliser falls back to eps, and
    the renderer quietly emits the global-affine branch alone.  Loss still goes
    down, Delta_shuffle still looks fine, and the s axis is simply not there.
    Measured on the RO-3 arm fed raw: 1.6% of cells orphaned at p50 but 25.4%
    on the worst entry -- enough to corrupt a level without ever erroring.
    """
    d = (s.reshape(-1, 1) - mu_s.reshape(1, -1).to(s.device)).abs().min(dim=1).values
    orphan = float((d > 3.0 * sigma_s_max).float().mean())
    if orphan > max_orphan:
        raise ValueError(
            f"s field is outside the anchor domain: {100*orphan:.1f}% of cells "
            f"are further than 3*sigma_s_max={3*sigma_s_max:.2f} from every one "
            f"of the {mu_s.numel()} mu_s anchors "
            f"[{float(mu_s.min()):.3f}, {float(mu_s.max()):.3f}]; s range is "
            f"[{float(s.min()):.3f}, {float(s.max()):.3f}]. "
            f"Normalise with load_s_arm_recipe()+s_to_unit() first, or widen "
            f"the anchor grid -- do NOT let this run silently.")


class RDG4D(torch.nn.Module):
    def __init__(self, arm: str, n_gauss: int = 48, tok_dim: int = 256,
                 style: int = 128, anchors=None, free: bool = False):
        super().__init__()
        kind, kw = ARMS[arm]
        self.arm, self.N = arm, n_gauss
        self.tokenizer = PairTokenizer(tok_dim, style, 6)
        # the primitive head is 27 wide instead of 23 (mu_s is NOT generated:
        # R-2 anchoring is structural, not a prediction), still zero-init
        if kind == "mlp":
            self.gen = MLPGen(n_gauss, tok_dim=tok_dim, four_d=True, **kw)
        else:
            self.gen = TransformerGen(n_gauss, tok_dim=tok_dim, style=style,
                                      four_d=True, **kw)
        self.head = ParamHead4D(n_gauss, anchors=anchors, free=free)

    def params_from_image(self, img, drop=None):
        tok, sty, grid = self.tokenizer(img, drop)
        outs, extra = self.gen(tok, sty, grid)
        return [self.head(zp, zg) for zp, zg in outs], extra

    def diversity_hinge(self, p) -> torch.Tensor:
        """R-3, with the coordinator's RELATIVE gamma.

        gamma_mu = 0.15 was written for s in [0,1], where the K=6 anchor grid
        has std 0.351 (43% of it).  Expressed relatively it is
        0.4 * std(anchor grid), which is what is used here so the hinge cannot
        silently idle if the s range ever changes.  With R-2 the mu_s term is
        structurally zero (frozen grid); it is still computed and reported,
        because "the hinge is pinned" is R-3's own failure criterion.
        """
        grid = self.head.mu_s.float()
        gamma_mu = 0.4 * float(grid.std())
        gamma_lam = 0.4 * float(torch.log(torch.tensor(
            [0.025, 0.30])).std())
        h_mu = torch.relu(gamma_mu - grid.std())
        h_lam = torch.relu(gamma_lam - torch.log(p["sigma_s"]).std(dim=1).mean())
        return h_mu + h_lam, float(h_mu), float(h_lam), gamma_mu, gamma_lam

    def n_params(self):
        def c(m):
            return sum(q.numel() for q in m.parameters() if q.requires_grad)
        return {"tokenizer": c(self.tokenizer), "generator": c(self.gen),
                "total": c(self)}


def psnr_t(a, b):
    return 10.0 * torch.log10(1.0 / ((a - b) ** 2).mean(dim=(-1, -2))
                              .clamp(min=1e-12))


@torch.no_grad()
def evaluate(model, ds, dev, bs=32, n_px=8192, seed=777):
    model.eval()
    g = torch.Generator(device=dev)
    rows = {k: [] for k in ("psnr", "psnr_const", "psnr_shuf", "psnr_null",
                            "de", "lv", "sig_s", "psnr_ident")}
    perm = np.random.default_rng(seed).permutation(len(ds))
    for i in range(0, len(ds), bs):
        idx = list(range(i, min(i + bs, len(ds))))
        x, y, s, lv = to_gpu([ds[j] for j in idx], dev)
        B, _, R, _ = x.shape
        g.manual_seed(seed + i)
        pix = torch.randint(0, R * R, (B, n_px), generator=g, device=dev)
        xf = x.permute(0, 2, 3, 1).reshape(B, -1, 3)
        yf = y.permute(0, 2, 3, 1).reshape(B, -1, 3)
        sf = s.reshape(B, -1)
        xs = torch.gather(xf, 1, pix.unsqueeze(-1).expand(-1, -1, 3))
        ys = torch.gather(yf, 1, pix.unsqueeze(-1).expand(-1, -1, 3))
        ss = torch.gather(sf, 1, pix)
        img = torch.cat([x, y], 1)
        p = model.params_from_image(img)[0][0]
        pr = render4d(p, xs, ss).clamp(0, 1)
        rows["psnr"].append(psnr_t(pr, ys).cpu().numpy())
        rows["de"].append(delta_e00(pr, ys).mean(1).cpu().numpy())
        rows["psnr_ident"].append(psnr_t(xs, ys).cpu().numpy())
        # M1: s replaced by a constant (1 everywhere)
        rows["psnr_const"].append(psnr_t(
            render4d(p, xs, torch.ones_like(ss)).clamp(0, 1), ys).cpu().numpy())
        # M2: another image's s field
        sh = torch.as_tensor(perm[idx] % len(ds), device=dev)
        _, _, s2, _ = to_gpu([ds[int(j)] for j in perm[idx] % len(ds)], dev)
        ss2 = torch.gather(s2.reshape(B, -1), 1, pix)
        rows["psnr_shuf"].append(psnr_t(
            render4d(p, xs, ss2).clamp(0, 1), ys).cpu().numpy())
        del sh
        # generator-side condition dropout
        pn = model.params_from_image(
            img, torch.ones(B, dtype=torch.bool, device=dev))[0][0]
        rows["psnr_null"].append(psnr_t(
            render4d(pn, xs, ss).clamp(0, 1), ys).cpu().numpy())
        rows["lv"].append(lv.cpu().numpy())
        rows["sig_s"].append(p["sigma_s"].cpu().numpy())
    out = {k: np.concatenate(v) for k, v in rows.items()}
    res = {"overall": {
        "psnr": float(out["psnr"].mean()),
        "psnr_identity": float(out["psnr_ident"].mean()),
        "de00_p50": float(np.percentile(out["de"], 50)),
        "delta_const_db": float(out["psnr"].mean() - out["psnr_const"].mean()),
        "delta_shuffle_db": float(out["psnr"].mean() - out["psnr_shuf"].mean()),
        "delta_cond_null_db": float(out["psnr"].mean()
                                    - out["psnr_null"].mean()),
        "sigma_s_mean": float(out["sig_s"].mean()),
        "sigma_s_frac_at_max": float((out["sig_s"] >= 0.2973).mean()),
        "sigma_s_std_log_mean": float(np.log(out["sig_s"]).std(axis=1).mean()),
    }, "per_level": {}}
    for li, lv in enumerate(LEVELS):
        m = out["lv"] == li
        if not m.any():
            continue
        res["per_level"][lv] = {
            "n": int(m.sum()),
            "psnr": float(out["psnr"][m].mean()),
            "psnr_identity": float(out["psnr_ident"][m].mean()),
            "de00_p50": float(np.percentile(out["de"][m], 50)),
            "delta_const_db": float(out["psnr"][m].mean()
                                    - out["psnr_const"][m].mean()),
            "delta_shuffle_db": float(out["psnr"][m].mean()
                                      - out["psnr_shuf"][m].mean()),
        }
    model.train()
    return res, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--n-gauss", type=int, default=48)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--px", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--cond-drop", type=float, default=0.15)
    ap.add_argument("--w-aux", type=float, default=0.2)
    ap.add_argument("--w-hinge", type=float, default=1.0)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--free", action="store_true")
    ap.add_argument("--init-from", default=None,
                    help="Stage-1 best.pt to warm start the shared trunk from")
    ap.add_argument("--anchors", default=os.path.join(
        EXP, "config", "anchors_stage0_kmeans.npy"))
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mem-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--out", default=os.path.join(EXP, "runs2"))
    args = ap.parse_args()

    tag = args.tag or args.arm
    run_dir = os.path.join(args.out, tag)
    os.makedirs(run_dir, exist_ok=True)
    mfile = os.path.join(run_dir, "metrics.json")
    if os.path.isfile(mfile):
        print(f"[{tag}] SKIP", flush=True)
        return 0
    dev = args.device
    dev_i = int(dev.split(":")[1])
    torch.cuda.set_per_process_memory_fraction(args.mem_frac, dev_i)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    anchors = None
    if os.path.isfile(args.anchors):
        anchors = torch.from_numpy(np.load(args.anchors))
    model = RDG4D(args.arm, args.n_gauss, anchors=anchors,
                  free=args.free).to(dev)
    if args.init_from and os.path.isfile(args.init_from):
        sd = torch.load(args.init_from, map_location="cpu",
                        weights_only=False)["model"]
        own = model.state_dict()
        keep = {k: v for k, v in sd.items()
                if k in own and own[k].shape == v.shape}
        model.load_state_dict(keep, strict=False)
        print(f"[{tag}] warm start: {len(keep)}/{len(own)} tensors from "
              f"{args.init_from}", flush=True)
    counts = model.n_params()
    print(f"[{tag}] params {counts}", flush=True)

    tr = ConstructSet("train", args.res)
    va = ConstructSet("val", args.res)
    print(f"[{tag}] D-CONSTRUCT train {len(tr)} / val {len(va)}", flush=True)

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim <= 1 or "query" in n or "register" in n
         or "null_" in n else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95))

    def lr_at(s):
        if s < args.warmup:
            return (s + 1) / args.warmup
        t = (s - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * t)) * 0.99 + 0.01
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    loader = torch.utils.data.DataLoader(
        tr, batch_size=args.bs, shuffle=True, num_workers=args.workers,
        drop_last=True, collate_fn=lambda b: b,
        persistent_workers=args.workers > 0)
    g = torch.Generator(device=dev)
    g.manual_seed(args.seed)
    hist, best, traj = [], None, []
    t0 = time.time()
    step = 0
    it = iter(loader)
    while step < args.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        x, y, s, _ = to_gpu(batch, dev)
        B, _, R, _ = x.shape
        pix = torch.randint(0, R * R, (B, args.px), generator=g, device=dev)
        xf = x.permute(0, 2, 3, 1).reshape(B, -1, 3)
        yf = y.permute(0, 2, 3, 1).reshape(B, -1, 3)
        xs = torch.gather(xf, 1, pix.unsqueeze(-1).expand(-1, -1, 3))
        ys = torch.gather(yf, 1, pix.unsqueeze(-1).expand(-1, -1, 3))
        ss = torch.gather(s.reshape(B, -1), 1, pix)
        if step == 0:                      # fail loud before wasting a run
            assert_s_in_anchor_domain(ss, model.head.mu_s)
        drop = torch.rand(B, device=dev, generator=g) < args.cond_drop
        with torch.autocast("cuda", dtype=torch.bfloat16):
            plist, extra = model.params_from_image(torch.cat([x, y], 1), drop)
        loss = (render4d(plist[0], xs, ss) - ys).abs().mean()
        for p in plist[1:]:
            loss = loss + args.w_aux * (
                render4d(p, xs[:, :2048], ss[:, :2048]) - ys[:, :2048]
            ).abs().mean()
        hinge, h_mu, h_lam, gmu, glam = model.diversity_hinge(plist[0])
        loss = loss + args.w_hinge * hinge
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        sched.step()
        step += 1
        if step % 100 == 0:                    # G3-caliber sigma_s trajectory
            with torch.no_grad():
                sg = plist[0]["sigma_s"].float()
                traj.append({"step": step, "sigma_s_mean": float(sg.mean()),
                             "sigma_s_std": float(sg.std()),
                             "sigma_s_frac_at_max": float(
                                 (sg >= 0.2973).float().mean()),
                             "log_sigma_s_std": float(
                                 torch.log(sg).std(dim=1).mean()),
                             "hinge_mu": h_mu, "hinge_lambda": h_lam,
                             "gamma_mu_rel": gmu, "gamma_lambda_rel": glam,
                             "loss": float(loss.detach())})
        if step % 200 == 0:
            el = time.time() - t0
            print(f"[{tag}] {step}/{args.steps} loss {float(loss):.5f} "
                  f"{step/el:.2f} it/s mem "
                  f"{torch.cuda.max_memory_allocated(dev_i)/2**30:.1f}G",
                  flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            r, _ = evaluate(model, va, dev)
            hist.append({"step": step, "val": r,
                         "sec": round(time.time() - t0, 1)})
            print(f"[{tag}] EVAL {step}: PSNR {r['overall']['psnr']:.3f} "
                  f"dE p50 {r['overall']['de00_p50']:.3f} "
                  f"dconst {r['overall']['delta_const_db']:.2f} "
                  f"dshuf {r['overall']['delta_shuffle_db']:.2f}", flush=True)
            score = r["overall"]["de00_p50"]
            if best is None or score < best[0]:
                best = (score, step, r)
                torch.save({"model": model.state_dict(), "step": step,
                            "args": vars(args)},
                           os.path.join(run_dir, "best.pt"))
            with open(os.path.join(run_dir, "history.json"), "w") as f:
                json.dump({"hist": hist, "sigma_s_trajectory": traj}, f,
                          indent=1)
    out = {"arm": args.arm, "tag": tag, "free": args.free, "params": counts,
           "selected_step": int(best[1]), "steps": args.steps,
           "wall_sec": round(time.time() - t0, 1),
           "peak_mem_gb": torch.cuda.max_memory_allocated(dev_i) / 2 ** 30,
           "val": best[2], "history": hist, "sigma_s_trajectory": traj,
           "config": vars(args)}
    with open(mfile, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[{tag}] DONE {json.dumps(best[2]['overall'])}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
