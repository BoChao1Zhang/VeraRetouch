"""A0 batched GLUT trainer -- GLUT single-style recipe (dossier 2.2, sec 4.1 + A.1).

Recipe: Adam base lr 1e-3, cosine annealing over the whole run, 20 epochs,
bs=1024 per LUT, train set = the 128^3 hald colors (color-space split).
GLUT-original init (identity local AND global affines -> f(x)=2x at step 0).

Two arms:
  rec  : L_rec only, no mining              (dossier 2.3 step 2-3)
  full : L_rec + 10 L_hc + 0.001 R_sparse + hard sample mining
         (epoch 5->20, mining share of each batch 10%->40% linear,
         highest-L1 samples; total worth < +0.4 dB per the paper)

All B LUTs train simultaneously in one BatchedGLUT (per-LUT params, per-LUT
GT, shared color table).  Statistically identical to independent runs; the
per-step sample indices are drawn independently per LUT.

Loss is computed on UNCLAMPED f(x) (paper silent; clamped loss would have
zero gradient wherever f>1 at the 2x init -- see NOTES.md assumption list).
`--loss-on-clamped` flips this for a one-off ablation.
"""

from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import torch

from model.glut_repro.model import BatchedGLUT
from model.glut_repro import losses


def psnr_float(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = float(np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2))
    return 99.0 if mse <= 1e-12 else float(10.0 * math.log10(1.0 / mse))


def psnr_8bit(pred: np.ndarray, gt: np.ndarray) -> float:
    p = np.round(pred.astype(np.float64) * 255.0)
    g = np.round(gt.astype(np.float64) * 255.0)
    mse = float(np.mean((p - g) ** 2))
    return 99.0 if mse <= 1e-12 else float(10.0 * math.log10(255.0 ** 2 / mse))


class A0Trainer:
    def __init__(self, colors: np.ndarray, gts: np.ndarray, n_gaussians: int,
                 arm: str = "rec", device: str = "cuda", seed: int = 0,
                 epochs: int = 20, bs: int = 1024, lr: float = 1e-3,
                 loss_on_clamped: bool = False):
        """colors: (P,3) float32 shared train colors; gts: (B,P,3) float16."""
        assert arm in ("rec", "full")
        torch.manual_seed(seed)
        self.arm = arm
        self.device = device
        self.epochs = epochs
        self.bs = bs
        self.loss_on_clamped = loss_on_clamped
        self.B, self.P = gts.shape[0], colors.shape[0]
        self.x_all = torch.from_numpy(colors).to(device)          # (P,3) f32
        self.y_all = torch.from_numpy(gts).to(device)             # (B,P,3) f16
        self.model = BatchedGLUT(self.B, n_gaussians).to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.steps_per_epoch = self.P // bs
        total = self.steps_per_epoch * epochs
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=total)
        self.total_steps = total
        # hard-mining state (full arm)
        if arm == "full":
            self.err = torch.full((self.B, self.P), 1e3, device=device,
                                  dtype=torch.float16)
            self.top_pool: torch.Tensor | None = None   # (B,K) indices
        self.g = torch.Generator(device=device)
        self.g.manual_seed(seed + 1)
        self.log: list[dict] = []

    def _mining_ratio(self, epoch: int) -> float:
        if self.arm != "full" or epoch < 5:
            return 0.0
        return 0.1 + (0.4 - 0.1) * min(1.0, (epoch - 5) / 15.0)

    def _refresh_pool(self) -> None:
        k = max(1, self.P // 10)   # top-10% error pool
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

    def train(self, log_every: int = 500) -> list[dict]:
        m = self.model
        step = 0
        t0 = time.time()
        for epoch in range(1, self.epochs + 1):
            rho = self._mining_ratio(epoch)
            for _ in range(self.steps_per_epoch):
                if self.arm == "full" and step % 200 == 0:
                    self._refresh_pool()
                idx = self._batch_indices(rho)                     # (B,bs)
                x = self.x_all[idx]                                # (B,bs,3)
                y = torch.gather(
                    self.y_all, 1,
                    idx.unsqueeze(-1).expand(-1, -1, 3)).float()   # (B,bs,3)
                f = m(x)
                pred = f.clamp(0, 1) if self.loss_on_clamped else f
                if self.arm == "rec":
                    loss = losses.l_rec(pred, y)
                else:
                    # L_hc needs valid sRGB -> computed on the clamped output
                    loss = losses.l_rec(pred, y) \
                        + 10.0 * losses.l_hc(f.clamp(0, 1), y) \
                        + 0.001 * losses.r_sparse(m.opacity_raw)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                self.opt.step()
                self.sched.step()
                if self.arm == "full":
                    with torch.no_grad():
                        e = (f.detach().clamp(0, 1) - y).abs().mean(-1)
                        self.err.scatter_(1, idx, e.half())
                if step % log_every == 0:
                    self.log.append({
                        "step": step, "epoch": epoch,
                        "loss": float(loss.detach()),
                        "lr": float(self.opt.param_groups[0]["lr"]),
                        "rho": rho, "sec": round(time.time() - t0, 1)})
                step += 1
        return self.log

    @torch.no_grad()
    def predict_colors(self, colors: np.ndarray,
                       chunk: int = 1 << 16) -> np.ndarray:
        """(P,3) -> (B,P,3) float32 clamped predictions."""
        m = self.model.eval()
        out = np.empty((self.B, colors.shape[0], 3), dtype=np.float32)
        for i in range(0, colors.shape[0], chunk):
            j = min(i + chunk, colors.shape[0])
            x = torch.from_numpy(colors[i:j]).to(self.device)
            x = x.unsqueeze(0).expand(self.B, -1, -1)
            out[:, i:j] = m.predict(x).cpu().numpy()
        self.model.train()
        return out

    def save(self, path: str, lut_ids: list[str], extra: dict) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"state_dict": self.model.state_dict(),
                    "lut_ids": lut_ids, "arm": self.arm,
                    "n_gaussians": self.model.N, "extra": extra,
                    "log": self.log}, path)


def de00_worker(args):
    """CPU DeltaE00 via the protocol-authoritative cubelib path.

    Tiled over 2M-point slabs to bound per-worker RAM (colour's CIEDE2000 on
    14.7M float64 points would need several GB of intermediates)."""
    import sys as _sys
    _repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    _sys.path.insert(0, os.path.join(_repo, "tools", "cube"))
    from cubelib import delta_e00  # noqa: E402
    pred_path, gt_path, sl = args
    pred_mm = np.load(pred_path, mmap_mode="r")[sl]
    gt_mm = np.load(gt_path, mmap_mode="r")[sl]
    n = pred_mm.shape[0]
    tile = 1 << 21
    des = []
    for i in range(0, n, tile):
        j = min(i + tile, n)
        des.append(delta_e00(
            np.asarray(pred_mm[i:j], dtype=np.float32),
            np.asarray(gt_mm[i:j], dtype=np.float32)).astype(np.float32))
    de = np.concatenate(des)
    return {"mean": float(np.mean(de)), "p95": float(np.percentile(de, 95)),
            "p99": float(np.percentile(de, 99)), "max": float(np.max(de))}
