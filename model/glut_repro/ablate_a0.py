"""A0 bug-hunt / gap-attribution ablation runner.

Isolates every ingredient the `full` arm adds on top of `rec`, plus two
optimization-recipe variants, on a fixed small LUT subset so every arm is a
PAIRED comparison (same LUTs, same seed, same GT).

Arms (``--arms`` comma list):
  rec        L_rec only, GLUT-original init (f(x)=2x @ init)     [baseline]
  hc         rec + 10 * L_hc                     (paper formula, as shipped)
  hc_fix     rec + 10 * L_hc_stable              (chroma-floored hue cosine)
  hc_w1      rec + 1  * L_hc                     (weight sensitivity)
  sparse     rec + 0.001 * R_sparse
  mining     rec + hard sample mining (ep5->20, 10%->40%)
  full       rec + all three (== run_a0 --arm full)
  g0         L_rec only, GLOBAL affine init G=0  (f(x)=x @ init)
  g0_full    g0 init + all three extras

Everything else (Adam 1e-3 cosine, bs 1024, 20 ep, N=32, color-space split)
is the frozen A0 recipe.

Diagnostics recorded per arm:
  * per-term loss traces (L_rec / L_hc / R_sparse) -- L_hc is monitored even
    when it is not in the objective;
  * total gradient-norm trace + spike statistics (p50/p99/max, #steps above
    100x the running median) -- this is what catches the near-neutral chroma
    blow-up in L_hc;
  * opacity statistics + weight-mass alive_frac at the end (R_sparse collapse
    check, PLAN red flag "effective primitives < 0.8 N");
  * eval on the fixed 2^21 held-out colour subsample: PSNR float/8bit, dE00.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from model.glut_repro import data, losses  # noqa: E402
from model.glut_repro.model import BatchedGLUT  # noqa: E402
from model.glut_repro.train_a0 import de00_worker, psnr_8bit, psnr_float  # noqa: E402


# ---------------------------------------------------------------------------
# Arm table
# ---------------------------------------------------------------------------

ARMS: dict[str, dict] = {
    "rec":     dict(hc=0.0, sparse=0.0, mining=False, init="glut", hc_mode="paper"),
    "hc":      dict(hc=10.0, sparse=0.0, mining=False, init="glut", hc_mode="paper"),
    "hc_fix":  dict(hc=10.0, sparse=0.0, mining=False, init="glut", hc_mode="stable"),
    "hc_w1":   dict(hc=1.0, sparse=0.0, mining=False, init="glut", hc_mode="paper"),
    "sparse":  dict(hc=0.0, sparse=0.001, mining=False, init="glut", hc_mode="paper"),
    "mining":  dict(hc=0.0, sparse=0.0, mining=True, init="glut", hc_mode="paper"),
    "full":    dict(hc=10.0, sparse=0.001, mining=True, init="glut", hc_mode="paper"),
    "g0":      dict(hc=0.0, sparse=0.0, mining=False, init="g0", hc_mode="paper"),
    "g0_full": dict(hc=10.0, sparse=0.001, mining=True, init="g0", hc_mode="stable"),
    # gradient-calibrated L_hc: lambda chosen so that |grad(lambda*L_hc)| is
    # ~10-15% of |grad(L_rec)| at the converged state (analysis/lhc_grad_probe:
    # 10*L_hc_stable sits at 184x L_rec, so lambda ~ 10*0.11/184 ~ 0.006).
    "hc_cal":  dict(hc=0.006, sparse=0.0, mining=False, init="glut", hc_mode="stable"),
    "full_fix": dict(hc=0.006, sparse=0.001, mining=True, init="glut", hc_mode="stable"),
    # bounded (sigmoid) opacity instead of raw+clamp -- see SigmoidOpacityGLUT
    "opac_sig": dict(hc=0.0, sparse=0.0, mining=False, init="glut",
                     hc_mode="paper", opacity="sigmoid"),
    "opac_sig_g0": dict(hc=0.0, sparse=0.0, mining=False, init="g0",
                        hc_mode="paper", opacity="sigmoid"),
}
for _a in ARMS.values():
    _a.setdefault("opacity", "clamp")


class SigmoidOpacityGLUT(BatchedGLUT):
    """BatchedGLUT with a BOUNDED opacity parameterization.

    The shipped model follows the dossier 2.4 ruling `raw parameter + clamp`.
    torch's clamp backward passes gradient only for min <= x <= max, so the
    moment Adam pushes opacity_raw past 0 or 1 the gradient is permanently
    zero: opacity is a ONE-WAY TRAPDOOR and a primitive that falls below 0 can
    never be revived (measured: alive_frac 0.854 in the rec arm, i.e. ~4.7 of
    32 primitives are dead weight).  sigmoid(logit) keeps o in (0,1) with a
    gradient everywhere; init logit 4.0 -> o = 0.982 ~ the paper's 1.0.
    """

    @torch.no_grad()
    def init_glut_original(self, sigma: float = 0.15) -> None:
        super().init_glut_original(sigma)
        self.opacity_raw.fill_(4.0)

    def opacity(self) -> torch.Tensor:
        return torch.sigmoid(self.opacity_raw)

    def weights(self, x: torch.Tensor) -> torch.Tensor:
        p = torch.exp(self.log_density(x))
        o = self.opacity().float().unsqueeze(-1)
        num = p * o
        return num / (num.sum(dim=1, keepdim=True) + 1e-6)


def l_hc_stable(pred: torch.Tensor, gt: torch.Tensor,
                c_floor: float = 1.0) -> torch.Tensor:
    """Chroma-weighted hue cosine distance with a chroma FLOOR.

    The shipped `losses.l_hc` divides by cp = sqrt(ap^2+bp^2+1e-6), i.e. the
    predicted chroma, whose minimum is 1e-3.  d(1-cos_h)/d ap ~ ag/(cp*cg), so
    a prediction that lands on the neutral axis produces a gradient up to
    ~1e3 x larger than a normal one; with weight 10 that is a 1e4 spike that
    poisons Adam's second moment for the following ~1/(1-beta2) steps.
    Flooring BOTH chroma norms at c_floor (1 Lab unit ~ 1 JND) removes the
    singularity while leaving the loss identical wherever chroma is
    perceptually meaningful.
    """
    lab_p = losses.srgb_to_lab_torch(pred)
    lab_g = losses.srgb_to_lab_torch(gt)
    ap, bp = lab_p[..., 1], lab_p[..., 2]
    ag, bg = lab_g[..., 1], lab_g[..., 2]
    cp = torch.sqrt(ap * ap + bp * bp + c_floor * c_floor)
    cg = torch.sqrt(ag * ag + bg * bg + c_floor * c_floor)
    cos_h = (ap * ag + bp * bg) / (cp * cg)
    w = (cg - c_floor).clamp(min=0.0)          # weight by GT chroma, 0 at neutral
    return (w * (1.0 - cos_h)).mean()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class AblationTrainer:
    def __init__(self, colors: np.ndarray, gts: np.ndarray, cfg: dict,
                 n_gaussians: int = 32, device: str = "cuda", seed: int = 0,
                 epochs: int = 20, bs: int = 1024, lr: float = 1e-3):
        torch.manual_seed(seed)
        self.cfg = cfg
        self.device = device
        self.epochs = epochs
        self.bs = bs
        self.B, self.P = gts.shape[0], colors.shape[0]
        self.x_all = torch.from_numpy(colors).to(device)
        self.y_all = torch.from_numpy(gts).to(device)
        cls = (SigmoidOpacityGLUT if cfg.get("opacity") == "sigmoid"
               else BatchedGLUT)
        self.model = cls(self.B, n_gaussians).to(device)
        if cfg["init"] == "g0":
            with torch.no_grad():
                self.model.G.zero_()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.steps_per_epoch = self.P // bs
        self.total_steps = self.steps_per_epoch * epochs
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=self.total_steps)
        if cfg["mining"]:
            self.err = torch.full((self.B, self.P), 1e3, device=device,
                                  dtype=torch.float16)
            self.top_pool: torch.Tensor | None = None
        self.g = torch.Generator(device=device)
        self.g.manual_seed(seed + 1)
        pr = torch.randperm(
            self.P, generator=torch.Generator().manual_seed(seed + 11))[:32768]
        self.probe_idx = pr.to(device)
        self.log: list[dict] = []
        # gradient norms kept on GPU (no per-step host sync)
        self.gnorm_buf = torch.zeros(self.total_steps, device=device)
        self.gnorms: np.ndarray | None = None

    # -- mining -------------------------------------------------------------
    def _mining_ratio(self, epoch: int) -> float:
        if not self.cfg["mining"] or epoch < 5:
            return 0.0
        return 0.1 + (0.4 - 0.1) * min(1.0, (epoch - 5) / 15.0)

    def _refresh_pool(self) -> None:
        k = max(1, self.P // 10)
        self.top_pool = torch.topk(self.err.float(), k, dim=1).indices

    def _batch_indices(self, rho: float) -> torch.Tensor:
        idx = torch.randint(0, self.P, (self.B, self.bs), device=self.device,
                            generator=self.g)
        if rho > 0.0 and self.top_pool is not None:
            k_mine = int(rho * self.bs)
            if k_mine > 0:
                sel = torch.randint(0, self.top_pool.shape[1],
                                    (self.B, k_mine), device=self.device,
                                    generator=self.g)
                idx[:, :k_mine] = torch.gather(self.top_pool, 1, sel)
        return idx

    # -- loss ---------------------------------------------------------------
    def _terms(self, f: torch.Tensor, y: torch.Tensor):
        rec = losses.l_rec(f, y)
        hc_fn = losses.l_hc if self.cfg["hc_mode"] == "paper" else l_hc_stable
        hc = hc_fn(f.clamp(0, 1), y) if self.cfg["hc"] > 0 else None
        if self.cfg["sparse"] > 0:
            o_raw = (torch.sigmoid(self.model.opacity_raw)
                     if self.cfg.get("opacity") == "sigmoid"
                     else self.model.opacity_raw)
            sp = losses.r_sparse(o_raw)
        else:
            sp = None
        total = rec
        if hc is not None:
            total = total + self.cfg["hc"] * hc
        if sp is not None:
            total = total + self.cfg["sparse"] * sp
        return total, rec, hc, sp

    def _grad_norm(self) -> torch.Tensor:
        """Total gradient L2 norm as a 0-dim GPU tensor (no host sync)."""
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        return torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads)))

    def train(self, log_every: int = 500) -> None:
        m = self.model
        step = 0
        t0 = time.time()
        for epoch in range(1, self.epochs + 1):
            rho = self._mining_ratio(epoch)
            for _ in range(self.steps_per_epoch):
                if self.cfg["mining"] and step % 200 == 0:
                    self._refresh_pool()
                idx = self._batch_indices(rho)
                x = self.x_all[idx]
                y = torch.gather(self.y_all, 1,
                                 idx.unsqueeze(-1).expand(-1, -1, 3)).float()
                f = m(x)
                total, rec, hc, sp = self._terms(f, y)
                self.opt.zero_grad(set_to_none=True)
                total.backward()
                gn = self._grad_norm()
                self.gnorm_buf[step] = gn
                self.opt.step()
                self.sched.step()
                if self.cfg["mining"]:
                    with torch.no_grad():
                        e = (f.detach().clamp(0, 1) - y).abs().mean(-1)
                        self.err.scatter_(1, idx, e.half())
                if step % log_every == 0:
                    with torch.no_grad():
                        hc_mon = float(losses.l_hc(f.detach().clamp(0, 1), y))
                    self.log.append({
                        "step": step, "epoch": epoch,
                        "total": float(total.detach()),
                        "l_rec": float(rec.detach()),
                        "l_hc_monitor": hc_mon,
                        "l_hc_used": None if hc is None else float(hc.detach()),
                        "r_sparse": None if sp is None else float(sp.detach()),
                        "gnorm": float(gn), "rho": rho,
                        "lr": float(self.opt.param_groups[0]["lr"]),
                        "sec": round(time.time() - t0, 1)})
                step += 1

    # -- diagnostics --------------------------------------------------------
    @torch.no_grad()
    def health(self) -> dict:
        m = self.model
        xp = self.x_all[self.probe_idx].unsqueeze(0).expand(self.B, -1, -1)
        mass = m.weight_mass(xp)                       # (B,N)
        alive = (mass >= (0.1 / m.N)).float().mean(-1)  # per-LUT alive share
        o = (torch.sigmoid(m.opacity_raw.detach())
             if self.cfg.get("opacity") == "sigmoid"
             else m.opacity_raw.detach().clamp(0, 1))
        gn = self.gnorm_buf.detach().float().cpu().numpy()
        self.gnorms = gn
        med = float(np.median(gn))
        return {
            "alive_frac_mean": float(alive.mean()),
            "alive_frac_min": float(alive.min()),
            "opacity_mean": float(o.mean()),
            "opacity_min": float(o.min()),
            "opacity_frac_below_0.05": float((o < 0.05).float().mean()),
            "opacity_frac_at_1": float((o >= 0.999).float().mean()),
            "gnorm_p50": med,
            "gnorm_p99": float(np.percentile(gn, 99)),
            "gnorm_max": float(gn.max()),
            "gnorm_spikes_100x": int((gn > 100 * med).sum()),
            "gnorm_spikes_1000x": int((gn > 1000 * med).sum()),
            "gnorm_nonfinite": int((~np.isfinite(gn)).sum()),
            "gnorm_trace_every100": [float(v) for v in gn[::100]],
        }

    @torch.no_grad()
    def predict_colors(self, colors: np.ndarray, chunk: int = 1 << 16):
        m = self.model.eval()
        out = np.empty((self.B, colors.shape[0], 3), dtype=np.float32)
        for i in range(0, colors.shape[0], chunk):
            j = min(i + chunk, colors.shape[0])
            x = torch.from_numpy(colors[i:j]).to(self.device)
            out[:, i:j] = m.predict(
                x.unsqueeze(0).expand(self.B, -1, -1)).cpu().numpy()
        self.model.train()
        return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", _REPO, "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def load_gt_sub(lut_ids, part_idx):
    out = []
    for lid in lut_ids:
        mm = np.load(data._cache_path("a0", lid, "eval"), mmap_mode="r")
        out.append(np.asarray(mm[part_idx]))
    return np.stack(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lut-list", required=True)
    ap.add_argument("--arms", default="rec,hc,hc_fix,sparse,mining,full,g0")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--de-workers", type=int, default=8)
    ap.add_argument("--tag", default="ablate")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    luts = []
    with open(args.lut_list) as f:
        for line in f:
            if line.strip():
                lid, path = line.rstrip("\n").split("\t")
                luts.append({"id": lid, "path": path})
    ids = [l["id"] for l in luts]
    print(f"[ablate] {len(ids)} LUTs, arms={args.arms}, epochs={args.epochs}",
          flush=True)

    jobs = [("a0", l["id"], l["path"], ("train", "eval")) for l in luts]
    errs = [e for _, e in data.build_gt_parallel(jobs, 8) if e]
    if errs:
        raise RuntimeError(f"GT build failures: {errs}")

    colors_train = data.train_colors()
    sub = data.evalsub_indices()
    colors_eval = data.eval_colors()[sub]
    gts_train = np.stack([np.load(data._cache_path("a0", lid, "train"))
                          for lid in ids])
    gt_eval = load_gt_sub(ids, sub)                     # (B,S,3) f16

    gt_dir = os.path.join(args.outdir, "gt_tmp")
    os.makedirs(gt_dir, exist_ok=True)
    gt_paths = []
    for k, lid in enumerate(ids):
        p = os.path.join(gt_dir, f"{lid}.gt.npy")
        if not os.path.exists(p):               # atomic: concurrent runs share gt_tmp
            tmp = f"{p}.{os.getpid()}.tmp.npy"
            np.save(tmp, gt_eval[k])
            os.replace(tmp, p)
        gt_paths.append(p)

    results = {}
    for arm in args.arms.split(","):
        arm = arm.strip()
        cfg = ARMS[arm]
        t0 = time.time()
        tr = AblationTrainer(colors_train, gts_train, cfg, n_gaussians=args.n,
                             seed=args.seed, epochs=args.epochs, bs=args.bs,
                             lr=args.lr)
        tr.train()
        health = tr.health()
        pred = tr.predict_colors(colors_eval)
        pred_dir = os.path.join(args.outdir, f"pred_tmp_{args.tag}")
        os.makedirs(pred_dir, exist_ok=True)
        de_jobs = []
        for k, lid in enumerate(ids):
            pp = os.path.join(pred_dir, f"{arm}__{lid}.npy")
            np.save(pp, pred[k].astype(np.float16))
            de_jobs.append((pp, gt_paths[k], slice(None)))
        from multiprocessing import Pool
        with Pool(args.de_workers) as pool:
            de_stats = pool.map(de00_worker, de_jobs)
        per_lut = []
        for k, lid in enumerate(ids):
            g = gt_eval[k].astype(np.float32)
            per_lut.append({
                "lut_id": lid,
                "psnr_float": psnr_float(pred[k], g),
                "psnr_8bit": psnr_8bit(pred[k], g),
                "de00_mean": de_stats[k]["mean"],
                "de00_p99": de_stats[k]["p99"]})
            os.remove(os.path.join(pred_dir, f"{arm}__{lid}.npy"))
        psnrs = np.array([r["psnr_float"] for r in per_lut])
        des = np.array([r["de00_mean"] for r in per_lut])
        res = {
            "arm": arm, "cfg": cfg, "epochs": args.epochs, "n": args.n,
            "n_luts": len(ids),
            "psnr_float_mean": float(psnrs.mean()),
            "psnr_float_min": float(psnrs.min()),
            "de00_mean": float(des.mean()),
            "de00_max": float(des.max()),
            "health": health,
            "per_lut": per_lut,
            "loss_log": tr.log,
            "wall_s": round(time.time() - t0, 1),
        }
        results[arm] = res
        print(f"[ablate] {arm}: PSNR {res['psnr_float_mean']:.3f} dB  "
              f"dE00 {res['de00_mean']:.4f}  alive {health['alive_frac_mean']:.3f}"
              f"  gspike100x {health['gnorm_spikes_100x']}  "
              f"({res['wall_s']:.0f}s)", flush=True)
        with open(os.path.join(args.outdir, f"{args.tag}.json"), "w") as f:
            json.dump({"git_commit": git_commit(), "args": vars(args),
                       "lut_ids": ids, "results": results}, f, indent=1)
    print("ABLATE_DONE", flush=True)


if __name__ == "__main__":
    main()
