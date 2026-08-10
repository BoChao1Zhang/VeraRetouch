"""RD-G Stage-1 training driver: generator capacity on D-RENDER.

One driver, five arms, identical everything except the map
`(I_in, after) -> 23N+12 renderer parameters`:

    mlp        CGLUT section 3.2 MLP generator, width 128 ("Large")   0.25 M
    mlp_wide   the same MLP widened to the transformer's budget       4.9  M
    gtiny      transformer d=192 L=2                                  1.6  M
    glite      G-Lite   d=256 L=4 cross@{1,3}   (PLAN 1.4)            4.4  M
    gbase      G-Base   d=384 L=6 cross@{1,3,5} (PLAN 1.4)           14.1  M

Recipe: PLAN section 3 "第二级 / Stage 1" verbatim where it is stated --
AdamW beta (0.9, 0.95), wd 0.05, lr 4e-4, warmup 2000, grad clip 1.0, bf16,
condition dropout p=0.15 to a LEARNABLE constant.  Loss = L_cube (1.0) +
per-tap aux (0.2) + L_prior (0.01) + a route-entropy floor.  L_param and
L_prequery of the PLAN's loss list are NOT included: both need Stage-0
gold-standard per-LUT parameters as targets and E1 dumped metrics only, so
there is nothing to regress onto (NOTES decision 2).

Explicitly absent, and why:
  * no L_hc / perceptual term.  A0 measured 10*L_hc at -10.73 dB, caused by a
    CIELab-vs-RGB[0,1] scale mismatch; the dossier's own number for the whole
    auxiliary loss package is +0.11 dB.  Any future perceptual term must pass a
    dimensional check first (A0 REPORT section 2).
  * no s-axis anything.  Stage-1 has no s axis, so there is no s smoothing
    regulariser to forbid; the red line is vacuously satisfied and is re-checked
    in Stage-2.
  * checkpoint selection never uses val loss (PLAN: "L1 最低 = 最保守平均 LUT").
    Selection is on val dE00 p50 with the variance ratio as a veto.

Every eval row carries Delta_const and Delta_shuffle, defined for a generator as
the metric gap between the true condition and (a) the learned null condition
(the condition-dropout constant) and (b) the condition permuted across the
evaluation set.  A generator that has collapsed to "one average LUT" scores 0 on
both, which is exactly the failure the red line is there to catch.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/bc/VeraRetouch")
from model.glut_repro import data_rdg as D                       # noqa: E402
from model.glut_repro.model_rdg import (                          # noqa: E402
    RDGModel, bake_cube, delta_e00, render, tetra_lookup,
)

EXP = "/home/bc/VeraRetouch/experiments/RDG_transformer_20260803"


# ---------------------------------------------------------------------------
def psnr(a, b):
    mse = ((a - b) ** 2).mean(dim=(-1, -2))
    return 10.0 * torch.log10(1.0 / mse.clamp(min=1e-12))


def quant(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, q))


# ---------------------------------------------------------------------------
class Evaluator:
    """Fixed, deterministic evaluation over a pool of cache rows.

    Colour caliber: `n_uni` uniform colours (headline, E1-comparable) plus
    `n_nat` colours drawn from the sample's own I_in (the ENNELUT track).  Both
    colour sets and the shuffle permutation are drawn from their own seeded
    generators, so every arm and every checkpoint sees literally the same
    numbers.
    """

    def __init__(self, cache, rows, bank_names, device, n_uni=8192, n_nat=4096,
                 bs=64, seed=777):
        self.c, self.rows, self.dev = cache, rows, device
        self.n_uni, self.n_nat, self.bs, self.seed = n_uni, n_nat, bs, seed
        self.bank_names = bank_names
        need = sorted({int(cache.preset[r]) for r in rows})
        self.remap = {p: i for i, p in enumerate(need)}
        self.bank = D.load_lut_bank([bank_names[p] for p in need], device)
        g = np.random.default_rng(seed)
        self.perm = g.permutation(len(rows))          # condition shuffle

    @torch.no_grad()
    def run(self, model, bake=False, bake_max=256):
        model.eval()
        dev = self.dev
        acc = {k: [] for k in ("de_true", "de_null", "de_shuf", "psnr_true",
                               "psnr_null", "psnr_shuf", "de_nat", "alive",
                               "de_ident", "psnr_ident")}
        preds, tgts = [], []
        bake_de, bake_de_gt = [], []
        for i in range(0, len(self.rows), self.bs):
            idx = self.rows[i:i + self.bs]
            img, src, pid = D.collate_to_gpu(
                [D.Loader(self.c, self.rows).__getitem__(j)
                 for j in range(i, min(i + self.bs, len(self.rows)))], dev)
            B = img.shape[0]
            lidx = torch.tensor([self.remap[int(p)] for p in pid.cpu()],
                                device=dev)
            tab = self.bank[lidx]
            xs = D.eval_colors(B, self.n_uni, self.seed + i, dev)
            yt = D.tri_lookup(tab, xs)
            gnat = torch.Generator(device=dev)
            gnat.manual_seed(self.seed + 100000 + i)
            xn = D.image_colors(src, self.n_nat, gnat)
            yn = D.tri_lookup(tab, xn)

            # true condition
            p_true = model.params_from_image(img)[0][0]
            pr = render(p_true, xs).clamp(0, 1)
            acc["de_true"].append(delta_e00(pr, yt).mean(1).cpu().numpy())
            acc["psnr_true"].append(psnr(pr, yt).cpu().numpy())
            acc["de_nat"].append(
                delta_e00(render(p_true, xn).clamp(0, 1), yn).mean(1)
                .cpu().numpy())
            acc["de_ident"].append(delta_e00(xs, yt).mean(1).cpu().numpy())
            acc["psnr_ident"].append(psnr(xs, yt).cpu().numpy())
            og = (p_true["opacity"] * p_true["gate"])
            acc["alive"].append((og > 0.05).float().mean(1).cpu().numpy())
            preds.append(pr.cpu())
            tgts.append(yt.cpu())

            # null condition (Delta_const)
            drop = torch.ones(B, dtype=torch.bool, device=dev)
            p_null = model.params_from_image(img, drop)[0][0]
            prn = render(p_null, xs).clamp(0, 1)
            acc["de_null"].append(delta_e00(prn, yt).mean(1).cpu().numpy())
            acc["psnr_null"].append(psnr(prn, yt).cpu().numpy())

            # shuffled condition (Delta_shuffle): same targets, others' images
            sh = self.perm[i:i + B] % len(self.rows)
            img_s, _, _ = D.collate_to_gpu(
                [D.Loader(self.c, self.rows).__getitem__(int(j)) for j in sh],
                dev)
            p_sh = model.params_from_image(img_s)[0][0]
            prs = render(p_sh, xs).clamp(0, 1)
            acc["de_shuf"].append(delta_e00(prs, yt).mean(1).cpu().numpy())
            acc["psnr_shuf"].append(psnr(prs, yt).cpu().numpy())

            if bake and i < bake_max:
                cube = bake_cube(p_true, 33)
                rb = tetra_lookup(cube.reshape(B, 33, 33, 33, 3), xs)
                bake_de.append(delta_e00(rb, pr).mean(1).cpu().numpy())
                bake_de_gt.append(delta_e00(rb, yt).mean(1).cpu().numpy())

        out = {k: np.concatenate(v) for k, v in acc.items() if v}
        pred = torch.cat(preds).float()
        tgt = torch.cat(tgts).float()
        # variance ratio: between-sample variance of the produced transform vs
        # of the target transform, on the shared colour grid.  A generator that
        # has collapsed onto one average LUT scores 0 here even when its L1 is
        # excellent -- this is the "L1 最低 = 最保守平均 LUT" detector.
        vp = pred.var(dim=0, unbiased=False).mean()
        vt = tgt.var(dim=0, unbiased=False).mean()
        var_ratio = float(vp / vt.clamp(min=1e-12))
        var_expl = float(1.0 - ((pred - tgt) ** 2).mean() / vt.clamp(min=1e-12))
        res = {
            "n": int(len(out["de_true"])),
            "de00_p50": quant(out["de_true"], 50),
            "de00_p90": quant(out["de_true"], 90),
            "de00_p99": quant(out["de_true"], 99),
            "de00_mean": float(out["de_true"].mean()),
            "de00_nat_p50": quant(out["de_nat"], 50),
            "psnr_mean": float(out["psnr_true"].mean()),
            "psnr_p50": quant(out["psnr_true"], 50),
            "identity_de00_p50": quant(out["de_ident"], 50),
            "identity_psnr_mean": float(out["psnr_ident"].mean()),
            "delta_const_db": float(out["psnr_true"].mean()
                                    - out["psnr_null"].mean()),
            "delta_shuffle_db": float(out["psnr_true"].mean()
                                      - out["psnr_shuf"].mean()),
            "delta_const_de00_p50": quant(out["de_null"], 50) -
            quant(out["de_true"], 50),
            "delta_shuffle_de00_p50": quant(out["de_shuf"], 50) -
            quant(out["de_true"], 50),
            "var_ratio": var_ratio,
            "var_explained": var_expl,
            "alive_frac": float(out["alive"].mean()),
            "dead_prim_frac": float(1.0 - out["alive"].mean()),
        }
        if bake_de:
            res["bake_de00_vs_direct_p50"] = quant(np.concatenate(bake_de), 50)
            res["bake_de00_vs_direct_p99"] = quant(np.concatenate(bake_de), 99)
            res["bake_de00_vs_gt_p50"] = quant(np.concatenate(bake_de_gt), 50)
            res["bake_psnr_penalty_db"] = None
        model.train()
        return res, out


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--n-gauss", type=int, default=48)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--bs", type=int, default=192)
    ap.add_argument("--p-uni", type=int, default=2048)
    ap.add_argument("--p-nat", type=int, default=2048)
    ap.add_argument("--p-aux", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--cond-drop", type=float, default=0.15)
    ap.add_argument("--w-aux", type=float, default=0.2)
    ap.add_argument("--w-prior", type=float, default=0.01)
    ap.add_argument("--w-route", type=float, default=0.01)
    ap.add_argument("--free", action="store_true",
                    help="PLAN's mandatory no-hardening free-regression control")
    ap.add_argument("--anchors", default=os.path.join(EXP, "runs", "ceiling",
                                                      "anchors.npy"))
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-n", type=int, default=1536)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mem-frac", type=float, default=0.34)
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--cache", default=D.CACHE)
    ap.add_argument("--out", default=os.path.join(EXP, "runs"))
    args = ap.parse_args()

    tag = args.tag or (args.arm + ("_free" if args.free else ""))
    run_dir = os.path.join(args.out, tag)
    os.makedirs(run_dir, exist_ok=True)
    mfile = os.path.join(run_dir, "metrics.json")
    if os.path.isfile(mfile):
        print(f"[{tag}] SKIP (metrics.json exists)", flush=True)
        return 0

    dev = args.device
    dev_i = int(dev.split(":")[1])
    torch.cuda.set_per_process_memory_fraction(args.mem_frac, dev_i)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cache = D.RenderCache(args.cache)
    tr_rows = cache.idx_of("train")
    ev_img = cache.idx_of("val_img")[:args.eval_n]
    ev_lut = cache.idx_of("val_lut")[:args.eval_n]
    print(f"[{tag}] train {len(tr_rows)} | val_img {len(ev_img)} | "
          f"val_lut {len(ev_lut)}", flush=True)

    anchors = None
    if os.path.isfile(args.anchors):
        anchors = torch.from_numpy(np.load(args.anchors))
        print(f"[{tag}] Stage-0 k-means anchors from {args.anchors}", flush=True)
    model = RDGModel(args.arm, args.n_gauss, anchors=anchors,
                     free=args.free).to(dev)
    counts = model.n_params()
    print(f"[{tag}] params {counts}", flush=True)

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

    # full-corpus LUT bank on device (3.5k x 33^3 x 3 fp32 ~ 1.5 GB)
    bank_names = cache.presets
    bank = D.load_lut_bank(bank_names, dev)
    print(f"[{tag}] LUT bank {tuple(bank.shape)} "
          f"({bank.numel()*4/2**30:.2f} GB)", flush=True)

    loader = torch.utils.data.DataLoader(
        D.Loader(cache, tr_rows), batch_size=args.bs, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
        persistent_workers=args.workers > 0, prefetch_factor=6
        if args.workers > 0 else None)

    ev_i = Evaluator(cache, ev_img, bank_names, dev)
    ev_l = Evaluator(cache, ev_lut, bank_names, dev)

    g = torch.Generator(device=dev)
    g.manual_seed(args.seed)
    hist, best = [], None
    t0 = time.time()
    step = 0
    data_wait = 0.0
    scaler_dtype = torch.bfloat16
    it = iter(loader)
    while step < args.steps:
        tw = time.time()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        img = torch.cat([batch[0].to(dev, non_blocking=True),
                         batch[1].to(dev, non_blocking=True)], -1)
        img = img.permute(0, 3, 1, 2).float() / 255.0
        src = img[:, :3]
        pid = batch[2].to(dev, non_blocking=True)
        data_wait += time.time() - tw

        B = img.shape[0]
        drop = torch.rand(B, device=dev, generator=g) < args.cond_drop
        # PLAN: sampling must contain a uniform grid; ENNELUT: must also
        # contain natural colours.  Half and half.
        xu = torch.rand(B, args.p_uni, 3, device=dev, generator=g)
        xn = D.image_colors(src, args.p_nat, g)
        x = torch.cat([xu, xn], 1)
        with torch.no_grad():
            y = D.tri_lookup_bank(bank, pid, x)

        with torch.autocast("cuda", dtype=scaler_dtype):
            plist, extra = model.params_from_image(img, drop)
        loss_main = None
        aux = 0.0
        na = min(args.p_aux, x.shape[1])
        for k, p in enumerate(plist):
            if k == 0:
                loss_main = (render(p, x) - y).abs().mean()
            else:
                aux = aux + (render(p, x[:, :na]) - y[:, :na]).abs().mean()
        prior = sum((p["z_prim"].float().pow(2).mean()
                     + p["z_glob"].float().pow(2).mean())
                    for p in plist[:1])
        loss = loss_main + args.w_aux * aux / max(1, len(plist) - 1) \
            + args.w_prior * prior
        if extra is not None:                      # route entropy floor
            hmin = 0.5 * math.log(3.0)
            loss = loss + args.w_route * torch.relu(hmin - extra)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        sched.step()
        step += 1

        if step % 200 == 0:
            el = time.time() - t0
            print(f"[{tag}] {step}/{args.steps} loss {float(loss):.5f} "
                  f"main {float(loss_main):.5f} gn {float(gn):.2f} "
                  f"lr {sched.get_last_lr()[0]:.2e} "
                  f"{step/el:.2f} it/s  data_wait {100*data_wait/el:.1f}% "
                  f"mem {torch.cuda.max_memory_allocated(dev_i)/2**30:.1f}G",
                  flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            r_i, _ = ev_i.run(model, bake=(step == args.steps))
            r_l, _ = ev_l.run(model, bake=(step == args.steps))
            row = {"step": step, "sec": round(time.time() - t0, 1),
                   "train_loss": float(loss_main), "val_img": r_i,
                   "val_lut": r_l,
                   "data_wait_frac": data_wait / (time.time() - t0)}
            hist.append(row)
            print(f"[{tag}] EVAL {step}: val_img dE p50 {r_i['de00_p50']:.4f} "
                  f"varr {r_i['var_ratio']:.3f} dconst {r_i['delta_const_db']:.2f} "
                  f"dshuf {r_i['delta_shuffle_db']:.2f} | val_lut dE p50 "
                  f"{r_l['de00_p50']:.4f}", flush=True)
            # checkpoint selection: NEVER val loss.  dE00 p50, var_ratio veto.
            score = r_i["de00_p50"] if r_i["var_ratio"] > 0.30 else 1e9
            if best is None or score < best[0]:
                best = (score, step, row)
                torch.save({"model": model.state_dict(), "step": step,
                            "args": vars(args), "counts": counts},
                           os.path.join(run_dir, "best.pt"))
            with open(os.path.join(run_dir, "history.json"), "w") as f:
                json.dump(hist, f, indent=1)
            # a partial, always-current metrics file so an interrupted run is
            # still a deliverable (and so `aggregate.py` can be run mid-flight)
            with open(os.path.join(run_dir, "metrics_partial.json"), "w") as f:
                json.dump({"arm": args.arm, "tag": tag, "free": args.free,
                           "params": counts, "n_gauss": args.n_gauss,
                           "n_render_params": 23 * args.n_gauss + 12,
                           "selected_step": int(best[1]), "steps": args.steps,
                           "wall_sec": round(time.time() - t0, 1),
                           "data_wait_frac": data_wait / (time.time() - t0),
                           "peak_mem_gb": torch.cuda.max_memory_allocated(dev_i)
                           / 2 ** 30,
                           "val_img": best[2]["val_img"],
                           "val_lut": best[2]["val_lut"],
                           "history": hist, "config": vars(args),
                           "PARTIAL": True}, f, indent=1)

    # final: reload the selected checkpoint and produce the reported numbers
    ck = torch.load(os.path.join(run_dir, "best.pt"), map_location=dev,
                    weights_only=False)
    model.load_state_dict(ck["model"])
    fin_i, raw_i = ev_i.run(model, bake=True, bake_max=512)
    fin_l, raw_l = ev_l.run(model, bake=True, bake_max=512)
    out = {
        "arm": args.arm, "tag": tag, "free": args.free,
        "params": counts, "n_gauss": args.n_gauss,
        "n_render_params": 23 * args.n_gauss + 12,
        "selected_step": int(ck["step"]), "steps": args.steps,
        "wall_sec": round(time.time() - t0, 1),
        "data_wait_frac": data_wait / (time.time() - t0),
        "peak_mem_gb": torch.cuda.max_memory_allocated(dev_i) / 2 ** 30,
        "val_img": fin_i, "val_lut": fin_l, "history": hist,
        "config": vars(args),
    }
    np.savez(os.path.join(run_dir, "per_sample.npz"),
             **{f"img_{k}": v for k, v in raw_i.items()},
             **{f"lut_{k}": v for k, v in raw_l.items()})
    with open(mfile, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[{tag}] DONE {json.dumps(fin_i)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
