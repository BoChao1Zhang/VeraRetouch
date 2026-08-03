"""G3 driver — Gate D3 collapse-channel verification (PLAN v2 section 1.2 / section 5).

Trains one global 4D GLUT on D-CONSTRUCT L1+L4 pairs with a PURE reconstruction
loss (L1, no s-axis regularizer of any kind -- "s 轴禁平滑正则" red line), and
records, every `--probe-every` steps:

  * sigma_s quantiles                 (the escape-channel trajectory)
  * mu_s distribution + std(mu_s)     (R-3 diversity statistic, reported only)
  * s-sensitivity  E||f(x,s+d) - f(x,s)||   on held-out pixels
  * Delta_const / Delta_shuffle       (tools/harness/collapse_probes, M1/M2)

Verdict (task card / EXPERIMENTS_v3 Gate D3):
  Delta_shuffle < 0.3 dB  -> collapse reproduced -> R-2 + R-3 are mandatory
  Delta_shuffle >= 3 dB   -> anti-collapse may be downgraded to optional

Usage:
  python -m model.glut_repro.train_g3 --arm naive --dataset fixed \
      --outdir experiments/G3_collapse_20260803/runs/fixed_naive_s0 --steps 20000
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

from model.glut_repro import data_construct as dc          # noqa: E402
from model.glut_repro.losses import l_rec                   # noqa: E402
from model.glut_repro.model4d_naive import GLUT4D           # noqa: E402
import collapse_probes as cp                                # noqa: E402
from metrics import masked_psnr, psnr_full                  # noqa: E402

LEVELS = ("L1", "L4")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", _REPO, "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# ---------------------------------------------------------------------------
# Renderer wrapper for the harness (render_fn(image, s) -> out)
# ---------------------------------------------------------------------------

class Renderer:
    """Adapts GLUT4D to the harness contract.  s arrives as the CACHED 32x32
    field (harness shuffles those across images); it is bilinearly upsampled to
    the image resolution here -- exactly the same path training used.
    NO per-image normalization of s anywhere (red line)."""

    def __init__(self, model: GLUT4D, device: str, chunk: int = 1 << 19):
        self.m, self.device, self.chunk = model, device, chunk

    @torch.no_grad()
    def __call__(self, image: np.ndarray, s: np.ndarray) -> np.ndarray:
        img = np.asarray(image)
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        h, w = img.shape[:2]
        s_full = dc.s32_to_full(np.asarray(s, dtype=np.float32), (h, w))
        flat = img.reshape(-1, 3).astype(np.float32)
        sf = s_full.reshape(-1)
        out = np.empty_like(flat)
        self.m.eval()
        for i in range(0, flat.shape[0], self.chunk):
            j = min(i + self.chunk, flat.shape[0])
            xb = torch.from_numpy(flat[i:j]).to(self.device)
            sb = torch.from_numpy(sf[i:j]).to(self.device)
            out[i:j] = self.m.predict(xb, sb).cpu().numpy()
        return out.reshape(img.shape)


def build_probe_samples(pairs, fixed_spec, n: int):
    """Harness sample dicts {'image','gt','s','id'} on n val pairs.
    's' is the 32x32 cache (uniform shape -> Delta_shuffle is legal).

    Pairs are picked EVENLY SPACED over the uid-sorted list, not as a prefix:
    uids sort as L1_val_* then L4_val_*, so a prefix of 8 would be 100% L1
    (semantic binary masks) and the trace would never see an L4 geometric
    soft mask.  Even spacing keeps both levels in every probe set.
    """
    pairs = list(pairs)
    if n < len(pairs):
        idx = np.linspace(0, len(pairs) - 1, n).round().astype(int)
        sel = [pairs[i] for i in dict.fromkeys(idx.tolist())]
    else:
        sel = pairs
    out = []
    for p in sel:
        x, y, s32, _ = dc.read_pair(p, fixed_spec=fixed_spec)
        out.append({"id": p.uid, "image": x, "gt": y, "s": s32,
                    "mask": dc.read_mask(p), "level": p.level})
    return out


def s_null_field(samples) -> np.ndarray:
    """Constant s field at the dataset mean.

    NOT the harness default: collapse_probes._mean_s_null treats the LAST axis
    of a 2-D (H,W) s field as channels, so its "null" is a per-column profile
    rather than a constant field.  PLAN M1 wants a constant s_null, so we pass
    one explicitly (NOTES decision 5)."""
    return np.full_like(np.asarray(samples[0]["s"], dtype=np.float32),
                        float(np.mean([np.mean(s["s"]) for s in samples])))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_probes(model, device, samples, s_null, seed: int) -> dict:
    r = Renderer(model, device)
    out = cp.run_probes(r, samples, s_null=s_null, seed=seed)
    return out


def eval_masked(model, device, samples) -> dict:
    r = Renderer(model, device)
    acc = {k: [] for k in ("psnr_in", "psnr_band", "psnr_out", "psnr_full")}
    for smp in samples:
        pred = r(smp["image"], smp["s"])
        mp = masked_psnr(pred, smp["gt"], smp["mask"])
        for k in ("psnr_in", "psnr_band", "psnr_out"):
            acc[k].append(mp[k])
        acc["psnr_full"].append(psnr_full(pred, smp["gt"]))
    return {k: float(np.nanmean(v)) for k, v in acc.items()}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["naive", "anchored"], required=True)
    ap.add_argument("--dataset", choices=["fixed", "tiered", "mixed"],
                    default="fixed",
                    help="fixed  = one pinned transform (s fully sufficient);"
                         " tiered = class+sign pinned, amplitude varies"
                         " (s partially sufficient);"
                         " mixed  = the as-shipped 40-transform targets"
                         " (s nearly worthless).  The three bracket the"
                         " s-benefit of the task, which is what decides"
                         " whether the escape channel is zero-COST.")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n-gaussians", type=int, default=32)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--px-per-img", type=int, default=30000)
    ap.add_argument("--probe-every", type=int, default=100)
    ap.add_argument("--probe-n", type=int, default=12,
                    help="val pairs used for the every-probe-every trace")
    ap.add_argument("--final-n", type=int, default=48,
                    help="val pairs used for the final reported numbers")
    ap.add_argument("--sens-delta", type=float, default=0.1)
    ap.add_argument("--cache-dir", default="/var/cache/veradata/g3")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    spec = {"fixed": dc.FIXED_SPEC, "tiered": dc.tiered_spec,
            "mixed": None}[args.dataset]

    # ---- data ------------------------------------------------------------
    tr_pairs = dc.load_pairs("train", LEVELS)
    va_pairs = dc.load_pairs("val", LEVELS)
    tag = f"{args.dataset}_px{args.px_per_img}_seed0"
    cpath = os.path.join(args.cache_dir, f"train_{tag}.npz")
    if os.path.exists(cpath):
        cache = dc.load_cache(cpath)
    else:
        cache = dc.build_pixel_cache(tr_pairs, px_per_img=args.px_per_img,
                                     seed=0, fixed_spec=spec)
        dc.save_cache(cache, cpath)
    print(f"[g3] train {len(tr_pairs)} pairs / {cache['x'].shape[0]} px; "
          f"val {len(va_pairs)} pairs; dataset={args.dataset}", flush=True)

    X = torch.from_numpy(cache["x"]).to(device)        # (M,3) f16
    Y = torch.from_numpy(cache["y"]).to(device)
    S = torch.from_numpy(cache["s"]).to(device)
    M_px = X.shape[0]

    probe_samples = build_probe_samples(va_pairs, spec, args.probe_n)
    final_samples = build_probe_samples(va_pairs, spec, args.final_n)
    s_null_probe = s_null_field(probe_samples)
    s_null_final = s_null_field(final_samples)

    # held-out pixels for s-sensitivity (from the val pairs, never trained on)
    rng = np.random.default_rng(args.seed + 5)
    sx, ss = [], []
    for smp in probe_samples:
        flat = smp["image"].reshape(-1, 3)
        sfull = dc.s32_to_full(smp["s"], smp["image"].shape[:2]).reshape(-1)
        idx = rng.choice(flat.shape[0], size=min(4000, flat.shape[0]),
                         replace=False)
        sx.append(flat[idx])
        ss.append(sfull[idx])
    sens_x = torch.from_numpy(np.concatenate(sx)).to(device)
    sens_s = torch.from_numpy(np.concatenate(ss)).to(device)

    # ---- model -----------------------------------------------------------
    model = GLUT4D(args.n_gaussians, arm=args.arm).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.steps, eta_min=args.lr * 0.01)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed + 7)
    print(f"[g3] arm={args.arm} N={args.n_gaussians} "
          f"params={model.n_params()}", flush=True)

    trace_path = os.path.join(args.outdir, "trace.jsonl")
    tf = open(trace_path, "w")
    t0 = time.time()
    # loss EMA is accumulated ON GPU: reading the loss every step with .item()
    # forces a synchronize, which stops the CPU running ahead of the launch
    # queue and costs ~3x wallclock on this launch-bound model.  We only sync
    # at probe steps.
    loss_ema_t = torch.zeros((), device=device)
    loss_ema = None

    def probe(step: int, loss_val: float) -> dict:
        pr = eval_probes(model, device, probe_samples, s_null_probe, args.seed)
        rec = {
            "step": step, "loss": loss_val, "loss_ema": loss_ema,
            "sec": round(time.time() - t0, 1),
            "lr": sched.get_last_lr()[0],
            "sigma_s": model.sigma_s_stats(),
            "mu_s": model.mu_s_stats(),
            "s_sensitivity": model.s_sensitivity(sens_x, sens_s,
                                                 args.sens_delta),
            "delta_const": pr["delta_const"],
            "delta_shuffle": pr["delta_shuffle"],
            "psnr_true": pr["psnr_true"], "psnr_null": pr["psnr_null"],
            "psnr_shuffled": pr["psnr_shuffled"],
        }
        tf.write(json.dumps(rec) + "\n")
        tf.flush()
        return rec

    r0 = probe(0, float("nan"))
    print(f"[g3] step 0  sigma_s q50={r0['sigma_s']['q50']:.4f}  "
          f"D_shuf={r0['delta_shuffle']:+.3f}", flush=True)

    for step in range(1, args.steps + 1):
        idx = torch.randint(0, M_px, (args.bs,), device=device, generator=gen)
        xb = X[idx].float()
        yb = Y[idx].float()
        sb = S[idx].float()
        loss = l_rec(model(xb, sb), yb)          # PURE L_rec, nothing else
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            ld = loss.detach()
            loss_ema_t = ld if step == 1 else 0.99 * loss_ema_t + 0.01 * ld
        if step % args.probe_every == 0 or step == args.steps:
            lv = float(ld)                       # the only per-step-loop sync
            loss_ema = float(loss_ema_t)
            rec = probe(step, lv)
            if step % (args.probe_every * 10) == 0 or step == args.steps:
                print(f"[g3] step {step:6d}  loss {loss_ema:.5f}  "
                      f"sig_s q50 {rec['sigma_s']['q50']:.4f} "
                      f"max {rec['sigma_s']['max']:.4f}  "
                      f"sens {rec['s_sensitivity']:.5f}  "
                      f"D_const {rec['delta_const']:+.3f}  "
                      f"D_shuf {rec['delta_shuffle']:+.3f}  "
                      f"({rec['sec']:.0f}s)", flush=True)
    tf.close()

    # ---- final report ----------------------------------------------------
    fin_probes = eval_probes(model, device, final_samples, s_null_final,
                             args.seed)
    fin_masked = eval_masked(model, device, final_samples)
    # identity baseline on the same pairs (context for the PSNR numbers)
    ident = {"psnr_full": float(np.mean(
        [psnr_full(s["image"], s["gt"]) for s in final_samples])),
        "psnr_in": float(np.nanmean(
            [masked_psnr(s["image"], s["gt"], s["mask"])["psnr_in"]
             for s in final_samples]))}
    verdict = ("collapsed" if fin_probes["delta_shuffle"] < 0.3 else
               "no_collapse" if fin_probes["delta_shuffle"] >= 3.0 else
               "inconclusive")
    out = {
        "exp": "G3_collapse", "arm": args.arm, "dataset": args.dataset,
        "fixed_spec": (spec if isinstance(spec, dict) else
                       ("per-pair tiered_spec(uid), exposure sign=+1, "
                        "tiers 0.15/0.30/0.60/1.20"
                        if spec is not None else None)),
        "levels": list(LEVELS),
        "n_train_pairs": len(tr_pairs), "n_val_pairs": len(va_pairs),
        "n_final_eval": len(final_samples),
        "criteria": {"delta_shuffle_collapsed": 0.3,
                     "delta_shuffle_no_collapse": 3.0,
                     "delta_const_collapsed": 0.05,
                     "m3_frac_at_upper": 0.80},
        "metrics": {**fin_probes, **fin_masked,
                    "s_sensitivity": model.s_sensitivity(
                        sens_x, sens_s, args.sens_delta),
                    "sigma_s": model.sigma_s_stats(),
                    "mu_s": model.mu_s_stats()},
        "identity_baseline": ident,
        "verdict": verdict,
        "params": model.n_params(),
        "wallclock_sec": round(time.time() - t0, 1),
        "git_commit": git_commit(), "args": vars(args),
    }
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(out, f, indent=1)
    torch.save({"state_dict": model.state_dict(), "arm": args.arm,
                "n_gaussians": args.n_gaussians, "args": vars(args)},
               os.path.join(args.outdir, "ckpt.pt"))
    print(json.dumps({k: out[k] for k in ("arm", "dataset", "verdict")},
                     indent=1), flush=True)
    print(f"[g3] D_shuffle={fin_probes['delta_shuffle']:+.3f} dB  "
          f"D_const={fin_probes['delta_const']:+.3f} dB  "
          f"sigma_s q50={out['metrics']['sigma_s']['q50']:.4f}  "
          f"verdict={verdict}", flush=True)
    print("G3_RUN_DONE", flush=True)


if __name__ == "__main__":
    main()
