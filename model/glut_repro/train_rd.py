"""RD-STD / RD-E driver: one level, one target variant, one arm, one seed.

  python -m model.glut_repro.train_rd --level L1 --variant fixed --arm gstd \
      --outdir experiments/RD_std_e_20260803/runs/fixed_L1_gstd_s0

Arms (model_rd.build_arm):
  g3d    3D Gaussian control (the PLAN ladder's "同 N 3D")
  gstd   RD-STD = R-1 + R-2 + R-3 + R-7 + R-10
  lut3d  3D LUT control (structurally s-free -> Delta_* identically 0)
  lut4d  RD-E = R-11 quadrilinear 4D LUT

Losses
------
  g3d / lut3d      L_rec  (+ RGB-axis TV for the LUT)
  lut4d            L_rec  + RGB-axis TV.  s-axis TV weight is 0: "s 轴禁平滑
                   正则" is a red line, and RD-E is the veto gate on the whole
                   renderer programme -- smoothing along s would make "no s
                   information" self-fulfilling.
  gstd             L_rec + w_div * R-3 hinge + GECO(R-7) * (tau - s_response),
                   the constraint being R-10(b)/(c) ("perturb s, the output must
                   move" / "penalise df/ds ~ 0").  R-10(a) BCE(Phi(s), mask) is
                   VACUOUS under RO-0 -- oracle s IS the mask, so Phi=identity
                   scores it at zero with nothing to learn; it is disabled and
                   recorded as such (NOTES).

Checkpoint selection: none.  The last step is taken, always (red line:
"checkpoint 选择禁用 val loss").

Every row of the output carries Delta_const AND Delta_shuffle (red line).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "tools", "harness"))

from model.glut_repro import data_rd as D                       # noqa: E402
from model.glut_repro.losses import l_rec                        # noqa: E402
from model.glut_repro.model_rd import (                          # noqa: E402
    ARM_DOC, ARMS, GECO, LUT4D, build_arm, has_geco, has_hinge, s_response,
    uses_s)
import collapse_probes as cp                                     # noqa: E402
from metrics import masked_psnr, psnr_full                        # noqa: E402


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", _REPO, "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# ---------------------------------------------------------------------------
# harness adapter
# ---------------------------------------------------------------------------

class Renderer:
    """render_fn(image, s) -> out, the tools/harness contract.

    s arrives in whatever shape the sample carries (32x32 cache, or full
    resolution for the s_res=full side grid) and is bilinearly resized to the
    image only when shapes differ -- the same path training used.  No per-image
    normalization of s anywhere (red line)."""

    def __init__(self, model, device: str, chunk: int = 1 << 16):
        self.m, self.device, self.chunk = model, device, chunk

    @torch.no_grad()
    def __call__(self, image: np.ndarray, s: np.ndarray) -> np.ndarray:
        img = np.asarray(image)
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        h, w = img.shape[:2]
        sa = np.asarray(s, dtype=np.float32)
        s_full = sa if sa.shape == (h, w) else D.s32_to_full(sa, (h, w))
        flat = img.reshape(-1, 3).astype(np.float32)
        sf = s_full.reshape(-1)
        out = np.empty_like(flat)
        self.m.eval()
        for i in range(0, flat.shape[0], self.chunk):
            j = min(i + self.chunk, flat.shape[0])
            xb = torch.from_numpy(flat[i:j]).to(self.device)
            sb = torch.from_numpy(sf[i:j]).to(self.device)
            out[i:j] = self.m.predict(xb, sb).float().cpu().numpy()
        self.m.train()
        return out.reshape(img.shape)


def shuffle_full(samples, seed: int = 0) -> dict:
    """Delta_shuffle for the s_res=full side grid, where the s fields have
    different shapes per image and collapse_probes (correctly) refuses to
    silently resize.  Same derangement recipe as collapse_probes.delta_shuffle
    (seeded permutation then roll by 1 -> no fixed point); the donor field is
    bilinearly resized to the recipient, which is the only way to move a
    full-resolution s across images at all.  Labelled as a different caliber in
    the report."""
    n = len(samples)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    assign = np.empty(n, dtype=int)
    assign[order] = np.roll(order, 1)
    return {i: assign[i] for i in range(n)}


def eval_all(model, device, samples, s_null, seed: int,
             s_res: str) -> dict:
    r = Renderer(model, device)
    if s_res == "c32":
        pr = cp.run_probes(r, samples, s_null=s_null, seed=seed)
    else:
        # per-image PSNR under true / null / donor s, harness psnr_full caliber
        asg = shuffle_full(samples, seed)
        pt, pn, ps = [], [], []
        for i, smp in enumerate(samples):
            h, w = smp["image"].shape[:2]
            pt.append(psnr_full(r(smp["image"], smp["s"]), smp["gt"]))
            pn.append(psnr_full(r(smp["image"], s_null * np.ones((h, w),
                                                                 np.float32)),
                                smp["gt"]))
            donor = np.asarray(samples[asg[i]]["s"], dtype=np.float32)
            ps.append(psnr_full(r(smp["image"], D.s32_to_full(donor, (h, w))),
                                smp["gt"]))
        pt, pn, ps = map(np.array, (pt, pn, ps))
        pr = {"delta_const": float(pt.mean() - pn.mean()),
              "delta_const_mode": "provided_constant(full-res)",
              "delta_shuffle": float(pt.mean() - ps.mean()),
              "psnr_true": float(pt.mean()), "psnr_null": float(pn.mean()),
              "psnr_shuffled": float(ps.mean()), "shuffle_seed": seed,
              "n": len(samples)}
    acc = {k: [] for k in ("psnr_in", "psnr_band", "psnr_out", "psnr_full")}
    for smp in samples:
        pred = r(smp["image"], smp["s"])
        mp = masked_psnr(pred, smp["gt"], smp["mask"])
        for k in ("psnr_in", "psnr_band", "psnr_out"):
            acc[k].append(mp[k])
        acc["psnr_full"].append(psnr_full(pred, smp["gt"]))
    pr.update({k: float(np.nanmean(v)) for k, v in acc.items()})
    return pr


def identity_baseline(samples) -> dict:
    acc = {k: [] for k in ("psnr_in", "psnr_band", "psnr_out", "psnr_full")}
    for smp in samples:
        mp = masked_psnr(smp["image"], smp["gt"], smp["mask"])
        for k in ("psnr_in", "psnr_band", "psnr_out"):
            acc[k].append(mp[k])
        acc["psnr_full"].append(psnr_full(smp["image"], smp["gt"]))
    return {k: float(np.nanmean(v)) for k, v in acc.items()}


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", required=True, choices=list(D.LEVELS))
    ap.add_argument("--variant", default="fixed", choices=["fixed", "mixed"])
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--s-res", default="c32", choices=["c32", "full"])
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n-gaussians", type=int, default=32)
    ap.add_argument("--lut-n-rgb", type=int, default=17)
    ap.add_argument("--lut-n-s", type=int, default=5)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lut-lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--px-per-img", type=int, default=30000)
    ap.add_argument("--probe-every", type=int, default=1000)
    ap.add_argument("--probe-n", type=int, default=8)
    ap.add_argument("--final-n", type=int, default=24)
    # R-3
    ap.add_argument("--w-div", type=float, default=0.1)
    # R-7 / R-10
    ap.add_argument("--geco-tau", type=float, default=0.02)
    ap.add_argument("--geco-alpha", type=float, default=0.99)
    ap.add_argument("--geco-step", type=float, default=0.01)
    ap.add_argument("--sens-delta", type=float, default=0.5)
    ap.add_argument("--geco-px", type=int, default=4096,
                    help="pixels used for the R-10 s-response constraint. It "
                         "is a scalar expectation, so a sub-sample of the "
                         "batch estimates it to well under its own noise "
                         "floor while keeping the extra forwards cheap.")
    # R-11
    ap.add_argument("--tv-rgb", type=float, default=1e-4)
    ap.add_argument("--tv-s", type=float, default=0.0,
                    help="MUST stay 0 for any headline RD-E number (red line: "
                         "s 轴禁平滑正则). Non-zero only for the appendix "
                         "'lambda -> inf degenerates to 3D' sweep.")
    ap.add_argument("--cache-dir", default="/var/cache/veradata/rd")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = args.device

    # ---- data ------------------------------------------------------------
    tr_pairs = D.load_pairs("train", (args.level,))
    va_pairs = D.load_pairs("val", (args.level,))
    tag = f"{args.level}_{args.variant}_{args.s_res}_px{args.px_per_img}"
    cpath = os.path.join(args.cache_dir, f"train_{tag}.npz")
    if os.path.exists(cpath):
        cache = D.load_cache(cpath)
    else:
        cache = D.build_pixel_cache(tr_pairs, args.variant,
                                    px_per_img=args.px_per_img, seed=0,
                                    s_res=args.s_res, verbose=False)
        D.save_cache(cache, cpath)

    def samples_of(pairs):
        out = []
        for p in pairs:
            d = D.read_pair(p, args.variant, s_res=args.s_res)
            out.append({"id": d["uid"], "image": d["x"], "gt": d["y"],
                        "s": d["s_cache"], "mask": d["mask"],
                        "level": d["level"], "alpha": d["alpha"]})
        return out

    probe_samples = samples_of(D.even_subset(va_pairs, args.probe_n))
    final_samples = samples_of(D.even_subset(va_pairs, args.final_n))
    mean_s = float(np.mean([np.mean(s["s"]) for s in final_samples]))
    # Delta_const's s_null: explicit CONSTANT field (D-10 second track).  Not
    # the harness default -- collapse_probes._mean_s_null treats the last axis
    # of a 2-D (H,W) field as channels and returns a per-column profile, not a
    # constant (D-28).
    s_null_probe = (np.full_like(np.asarray(probe_samples[0]["s"], np.float32),
                                 float(np.mean([np.mean(s["s"])
                                                for s in probe_samples])))
                    if args.s_res == "c32" else
                    float(np.mean([np.mean(s["s"]) for s in probe_samples])))
    s_null_final = (np.full_like(np.asarray(final_samples[0]["s"], np.float32),
                                 mean_s) if args.s_res == "c32" else mean_s)

    X = torch.from_numpy(cache["x"]).to(dev)
    Y = torch.from_numpy(cache["y"]).to(dev)
    S = torch.from_numpy(cache["s"]).to(dev)
    M_px = X.shape[0]
    print(f"[rd] {args.level}/{args.variant}/{args.arm}  "
          f"train {len(tr_pairs)} pairs / {M_px} px, val {len(va_pairs)} pairs",
          flush=True)

    # ---- model -----------------------------------------------------------
    model = build_arm(args.arm, args.n_gaussians, args.lut_n_rgb,
                      args.lut_n_s).to(dev)
    is_lut = isinstance(model, LUT4D)
    lr = args.lut_lr if is_lut else args.lr
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.steps, eta_min=lr * 0.01)
    gen = torch.Generator(device=dev)
    gen.manual_seed(args.seed + 7)
    geco = GECO(args.geco_tau, args.geco_alpha, args.geco_step) \
        if has_geco(args.arm) else None
    hinge_on = has_hinge(args.arm)
    print(f"[rd] arm={args.arm} ({ARM_DOC[args.arm][0]}: {ARM_DOC[args.arm][1]})"
          f" params={model.n_params()} lr={lr} geco={geco is not None} "
          f"hinge={hinge_on}", flush=True)

    trace_path = os.path.join(args.outdir, "trace.jsonl")
    tf = open(trace_path, "w")
    t0 = time.time()
    loss_ema_t = torch.zeros((), device=dev)

    def probe(step: int, extra: dict) -> dict:
        pr = eval_all(model, dev, probe_samples, s_null_probe, args.seed,
                      args.s_res)
        rec = {"step": step, "sec": round(time.time() - t0, 1),
               "lr": sched.get_last_lr()[0], **extra,
               "delta_const": pr["delta_const"],
               "delta_shuffle": pr["delta_shuffle"],
               "psnr_true": pr["psnr_true"], "psnr_in": pr["psnr_in"],
               "psnr_band": pr["psnr_band"], "psnr_out": pr["psnr_out"]}
        if hinge_on:
            rec["sigma_s"] = model.sigma_s_stats()
            rec["mu_s"] = model.mu_s_stats()
        if geco is not None:
            rec["geco"] = geco.state()
        tf.write(json.dumps(rec) + "\n")
        tf.flush()
        return rec

    probe(0, {"loss": float("nan")})

    # accumulated on GPU: reading these every step would force a synchronize
    hinge_active_t = torch.zeros((), device=dev)
    last: dict = {}
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, M_px, (args.bs,), device=dev, generator=gen)
        xb, yb, sb = X[idx].float(), Y[idx].float(), S[idx].float()
        pred = model(xb, sb)
        loss = l_rec(pred, yb)
        if is_lut:
            loss = loss + model.tv(args.tv_rgb, args.tv_s)
        if hinge_on:
            h, h_mu, h_lam = model.diversity_hinge()          # R-3
            loss = loss + args.w_div * h
            hinge_active_t = hinge_active_t + (h.detach() > 0).float()
            last = {"hinge": h.detach(), "hinge_mu": h_mu, "hinge_lam": h_lam}
        if geco is not None:
            gp = min(args.geco_px, args.bs)
            resp = s_response(model, xb[:gp], sb[:gp],
                              args.sens_delta)             # R-10(b)/(c)
            loss = loss + geco.term(geco.constraint(resp))     # R-7
            last["s_response"] = resp.detach()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if geco is not None:
            geco.update()
        with torch.no_grad():
            ld = loss.detach()
            loss_ema_t = ld if step == 1 else 0.99 * loss_ema_t + 0.01 * ld
        if step % args.probe_every == 0 or step == args.steps:
            rec = probe(step, {"loss": float(ld), "loss_ema": float(loss_ema_t),
                               **{k: float(v) for k, v in last.items()}})
            print(f"[rd] step {step:6d} loss {float(loss_ema_t):.5f} "
                  f"in {rec['psnr_in']:.2f} band {rec['psnr_band']:.2f} "
                  f"D_const {rec['delta_const']:+.2f} "
                  f"D_shuf {rec['delta_shuffle']:+.2f} ({rec['sec']:.0f}s)",
                  flush=True)
    tf.close()

    # ---- final -----------------------------------------------------------
    fin = eval_all(model, dev, final_samples, s_null_final, args.seed,
                   args.s_res)
    ident = identity_baseline(final_samples)
    extra: dict = {}
    if hinge_on:
        extra.update({"sigma_s": model.sigma_s_stats(),
                      "mu_s": model.mu_s_stats(),
                      "hinge_active_frac": float(hinge_active_t)
                      / max(args.steps, 1)})
    if geco is not None:
        extra["geco"] = geco.state()
        with torch.no_grad():
            extra["s_response_final"] = float(s_response(
                model, X[:4096].float(), S[:4096].float(), args.sens_delta))
    if args.arm == "ga":
        extra["payload"] = model.payload_stats()
    if is_lut:
        extra["cell_visit_frac"] = model.cell_visits(X[:200000].float(),
                                                    S[:200000].float())
    out = {
        "exp": "RD_std_e", "arm": args.arm,
        "arm_row": ARM_DOC[args.arm][0], "arm_mech": ARM_DOC[args.arm][1],
        "uses_s": uses_s(args.arm), "level": args.level,
        "variant": args.variant, "s_res": args.s_res, "seed": args.seed,
        "params": model.n_params(),
        "n_train_pairs": len(tr_pairs), "n_val_pairs": len(va_pairs),
        "n_final_eval": len(final_samples),
        "metrics": {**fin, **extra},
        "identity_baseline": ident,
        "mean_alpha": float(np.mean([s["alpha"] for s in final_samples])),
        "wallclock_sec": round(time.time() - t0, 1),
        "git_commit": git_commit(), "args": vars(args),
    }
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(out, f, indent=1)
    torch.save({"state_dict": model.state_dict(), "args": vars(args)},
               os.path.join(args.outdir, "ckpt.pt"))
    print(f"[rd] DONE {args.level}/{args.variant}/{args.arm}: "
          f"in {fin['psnr_in']:.2f} (id {ident['psnr_in']:.2f}) "
          f"band {fin['psnr_band']:.2f} full {fin['psnr_full']:.2f} "
          f"D_const {fin['delta_const']:+.2f} D_shuf {fin['delta_shuffle']:+.2f}",
          flush=True)
    print("RD_RUN_DONE", flush=True)


if __name__ == "__main__":
    main()
