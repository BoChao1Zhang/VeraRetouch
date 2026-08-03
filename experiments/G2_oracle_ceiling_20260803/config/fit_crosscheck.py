"""G2 实验法互证：三臂 GLUT 拟合（复用 model/glut_repro 的 BatchedGLUT + E1 初始化）。

在 l 系 S-val normal 的 N 组上，同一批像素同时拟合 7 个独立 GLUT：

  arm_1lut   1 张 3D GLUT，全部像素     （3D 基线）
  arm_mask3  3 张 3D GLUT，按 **C_GT 掩膜** 分三桶，各桶查各自的表（真 4D 形态）
  arm_luma3  3 张 3D GLUT，按 **亮度** 分三桶                     （亮度伪 4D 对照）

判据侧读法：arm_mask3 − arm_1lut 应与解析法 Δ_ceil 同量级；
arm_luma3 − arm_1lut 就是「换个和语义无关的轴也能白捡多少」——
两者之差才是掩膜轴的净贡献（Δ_shuffle 的实验法版本）。

7 个模型共享一次 BatchedGLUT 前向（每个 batch 元素有自己的像素集与 GT），
初始化沿用 E1 的残差加权 k-means（fit_e1.weighted_kmeans_init）。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/bc/VeraRetouch")

from model.glut_repro.fit_e1 import weighted_kmeans_init  # noqa: E402
from model.glut_repro.losses import l_rec  # noqa: E402
from model.glut_repro.model import BatchedGLUT  # noqa: E402
from tools.ceiling.loader import load_triplet  # noqa: E402

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)   # Rec.709


def _psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    qp = np.round(np.clip(pred, 0, 1) * 255.0)
    qg = np.round(np.clip(gt, 0, 1) * 255.0)
    mse = float(np.mean((qp - qg) ** 2))
    return 100.0 if mse <= 0 else min(10.0 * math.log10(255.0 ** 2 / mse), 100.0)


def _tercile_labels(v: np.ndarray) -> np.ndarray:
    """按分位切三桶（并列值不拆，桶可能退化——这是正确行为）。"""
    e = np.unique(np.quantile(v, [1 / 3, 2 / 3]))
    return np.searchsorted(e, v, side="left").astype(np.int64)


def fit_arms(x: np.ndarray, y: np.ndarray, s: np.ndarray, n_gauss: int = 32,
             steps: int = 1200, bs: int = 8192, lr: float = 5e-3,
             p_sample: int = 131072, device: str = "cuda", seed: int = 0) -> dict:
    """返回三臂 PSNR 与诊断。x,y: (P,3) float32；s: (P,)。"""
    lum = x @ LUMA
    lab_m = _tercile_labels(s)
    lab_l = _tercile_labels(lum)

    groups: list[tuple[str, np.ndarray]] = [("1lut", np.arange(x.shape[0]))]
    for k in range(3):
        groups.append((f"mask3_{k}", np.flatnonzero(lab_m == k)))
    for k in range(3):
        groups.append((f"luma3_{k}", np.flatnonzero(lab_l == k)))

    rng = np.random.default_rng(seed)
    xs, ys, valid = [], [], []
    for name, idx in groups:
        if idx.size == 0:                       # 退化桶：占位（不参与评估）
            idx = np.arange(min(1024, x.shape[0]))
            valid.append(False)
        else:
            valid.append(True)
        pick = rng.choice(idx, size=p_sample, replace=idx.size < p_sample)
        xs.append(x[pick])
        ys.append(y[pick])
    xb = np.stack(xs)                            # (B,P,3)
    yb = np.stack(ys)
    b = xb.shape[0]

    torch.manual_seed(seed)
    model = BatchedGLUT(b, n_gauss).to(device)
    mus, biases = [], []
    for i in range(b):
        mu, bias = weighted_kmeans_init(xb[i], yb[i], n_gauss, seed=seed + i)
        mus.append(mu)
        biases.append(bias)
    model.init_e1_residual(torch.from_numpy(np.stack(mus)).to(device),
                           torch.from_numpy(np.stack(biases)).to(device), sigma=0.30)

    xt = torch.from_numpy(xb).to(device)
    yt = torch.from_numpy(yb).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.01)
    g = torch.Generator(device=device).manual_seed(seed + 7)
    sigma_hi, sigma_lo = 0.30, 0.02
    for step in range(steps):
        idx = torch.randint(0, p_sample, (b, bs), device=device, generator=g)
        xx = torch.gather(xt, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        yy = torch.gather(yt, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        loss = l_rec(model(xx), yy)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        t_an = 0.6 * steps
        floor = (sigma_hi * (sigma_lo / sigma_hi) ** (step / t_an)) if step <= t_an else 0.005
        with torch.no_grad():
            model.chol_log_diag.clamp_(float(np.log(floor)), float(np.log(2.0)))

    # 全量预测：每个 batch 元素在**自己负责的像素**上评估
    pred = {"1lut": np.empty_like(y), "mask3": np.empty_like(y), "luma3": np.empty_like(y)}
    with torch.no_grad():
        model.eval()
        def _pred(bi: int, idx: np.ndarray) -> np.ndarray:
            out = np.empty((idx.size, 3), dtype=np.float32)
            for i0 in range(0, idx.size, 1 << 18):
                sl = idx[i0:i0 + (1 << 18)]
                xx = torch.from_numpy(x[sl]).to(device).unsqueeze(0)
                pad = torch.zeros(b, xx.shape[1], 3, device=device)
                pad[bi] = xx[0]
                out[i0:i0 + sl.size] = model.predict(pad)[bi].cpu().numpy()
            return out
        pred["1lut"][:] = _pred(0, np.arange(x.shape[0]))
        for k in range(3):
            idx = np.flatnonzero(lab_m == k)
            if idx.size:
                pred["mask3"][idx] = _pred(1 + k, idx)
        for k in range(3):
            idx = np.flatnonzero(lab_l == k)
            if idx.size:
                pred["luma3"][idx] = _pred(4 + k, idx)

    res = {f"psnr_{k}": _psnr(v, y) for k, v in pred.items()}
    res["psnr_identity"] = _psnr(x, y)
    res["delta_mask3"] = res["psnr_mask3"] - res["psnr_1lut"]
    res["delta_luma3"] = res["psnr_luma3"] - res["psnr_1lut"]
    res["delta_net"] = res["delta_mask3"] - res["delta_luma3"]
    res["n_buckets_mask"] = int(lab_m.max()) + 1
    res["n_buckets_luma"] = int(lab_l.max()) + 1
    res["valid_buckets"] = sum(valid)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--n-gauss", type=int, default=32)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--max-pixels", type=int, default=1_000_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=20260803)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.index, encoding="utf-8")]
    per: list[dict] = []
    t0 = time.time()
    for r in rows:
        if len(per) >= args.limit:
            break
        try:
            t = load_triplet(r, max_pixels=args.max_pixels)
        except Exception as e:
            print(f"[warn] {r['candidate_id']}: {e}", flush=True)
            continue
        if t is None:
            continue
        res = fit_arms(t.x, t.y, t.s, n_gauss=args.n_gauss, steps=args.steps,
                       device=args.device, seed=args.seed)
        res.update({"id": r["candidate_id"], "build": r["build"], "pool": r["pool"],
                    "group_id": r["group_id"], "n_pixels": int(t.x.shape[0])})
        per.append(res)
        if len(per) % 10 == 0:
            print(f"[xcheck] {len(per)}/{args.limit} {time.time()-t0:.0f}s "
                  f"d_mask3={np.median([p['delta_mask3'] for p in per]):.2f} "
                  f"d_luma3={np.median([p['delta_luma3'] for p in per]):.2f}", flush=True)

    def q(key: str) -> dict:
        v = np.array([p[key] for p in per], dtype=np.float64)
        return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
                "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))}

    agg = {"n": len(per), "n_gauss": args.n_gauss, "steps": args.steps,
           "params_per_lut": 22 * args.n_gauss + 12,
           **{k: q(k) for k in ("psnr_identity", "psnr_1lut", "psnr_mask3", "psnr_luma3",
                                "delta_mask3", "delta_luma3", "delta_net")}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"agg": agg, "per_image": per}, f, ensure_ascii=False, indent=2)
    print(json.dumps(agg, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
