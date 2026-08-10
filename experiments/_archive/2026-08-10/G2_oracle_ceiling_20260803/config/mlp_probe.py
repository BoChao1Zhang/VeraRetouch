"""D0-4 MLP 容量探针（EXPERIMENTS_v3 §D0「当天三个数」第二个数）。

问题：**逐像素 4D 算子 f(r,g,b,s)→RGB 在信息论上够不够表达一个局部编辑？**
做法：在**单个样本**上把一个 ~200K 参的 MLP 直接过拟合到 (x, s) → y。
判据：≥ 45 dB。< 40 dB ⇒ 问题不在渲染器（连过拟合都做不到就是数据本身不是
4D 逐像素函数），全线转修数据。

网络：4 → 256×4 → 3（4 层隐藏，宽 256），参数量
  4*256+256 + 3*(256*256+256) + 256*3+3 = 199,427。
纯逐像素、无 (x,y) 坐标 / 无邻域 / 无排序（红线：逐像素算子禁空间输入）。
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

CONSTRUCT_ROOT = Path("/home/bc/VeraRetouch/experiments/tooling-wave1/T4_construct/sanity")


class PixelMLP(nn.Module):
    """f(r,g,b,s) -> RGB。in_dim=4（4D 臂）或 3（3D 对照臂）。"""

    def __init__(self, in_dim: int = 4, width: int = 256, depth: int = 4):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, width), nn.ReLU(inplace=True)]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.ReLU(inplace=True)]
        layers += [nn.Linear(width, 3)]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    qp = torch.round(pred.clamp(0, 1) * 255.0)
    qg = torch.round(gt.clamp(0, 1) * 255.0)
    mse = float(torch.mean((qp - qg) ** 2))
    return 100.0 if mse <= 0 else min(10.0 * math.log10(255.0 ** 2 / mse), 100.0)


def overfit_one(x: np.ndarray, y: np.ndarray, s: np.ndarray, in_dim: int = 4,
                steps: int = 6000, bs: int = 65536, lr: float = 2e-3,
                device: str = "cuda", seed: int = 0, log_every: int = 1000
                ) -> dict:
    torch.manual_seed(seed)
    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    st = torch.as_tensor(s, dtype=torch.float32, device=device).unsqueeze(1)
    z = torch.cat([xt, st], 1) if in_dim == 4 else xt
    p = z.shape[0]

    model = PixelMLP(in_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.01)
    g = torch.Generator(device=device).manual_seed(seed + 1)
    curve = []
    t0 = time.time()
    for step in range(steps):
        idx = torch.randint(0, p, (min(bs, p),), device=device, generator=g)
        loss = torch.mean((model(z[idx]) - yt[idx]) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % log_every == 0 or step == steps - 1:
            with torch.no_grad():
                pr = torch.cat([model(z[i:i + 1 << 20]) for i in range(0, p, 1 << 20)])
                curve.append({"step": step, "psnr": _psnr(pr, yt),
                              "loss": float(loss.detach())})
            print(f"    step {step:5d} psnr={curve[-1]['psnr']:.2f} "
                  f"loss={curve[-1]['loss']:.2e} {time.time()-t0:.0f}s", flush=True)
    with torch.no_grad():
        pr = torch.cat([model(z[i:i + 1 << 20]) for i in range(0, p, 1 << 20)])
    return {"psnr": _psnr(pr, yt), "n_params": model.n_params(),
            "in_dim": in_dim, "steps": steps, "curve": curve,
            "sec": round(time.time() - t0, 1)}


def load_construct(uid: str, split: str = "val"):
    from PIL import Image
    root = CONSTRUCT_ROOT / split
    for line in open(root / "manifest.jsonl", encoding="utf-8"):
        m = json.loads(line)
        if m["uid"] != uid:
            continue
        f = m["files"]
        x = np.asarray(Image.open(root / f["in"]).convert("RGB"), np.float32) / 255.0
        y = np.asarray(Image.open(root / f["out"]).convert("RGB"), np.float32) / 255.0
        s = np.asarray(Image.open(root / f["mask"]).convert("L"), np.float32) / 255.0
        return m, x.reshape(-1, 3), y.reshape(-1, 3), s.reshape(-1)
    raise KeyError(uid)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uids", nargs="*", default=None,
                    help="默认 L1/L4 各 3 个 val 单样本")
    ap.add_argument("--split", default="val")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--also-3d", action="store_true", default=True,
                    help="同时跑 3D 对照臂 f(r,g,b)（不给 s）")
    args = ap.parse_args()

    uids = args.uids or [f"L1_{args.split}_{i:04d}" for i in range(3)] + \
                        [f"L4_{args.split}_{i:04d}" for i in range(3)]
    out: list[dict] = []
    for uid in uids:
        m, x, y, s = load_construct(uid, args.split)
        print(f"[probe] {uid} level={m['level']} mask={m['mask_kind']} P={x.shape[0]}",
              flush=True)
        r4 = overfit_one(x, y, s, in_dim=4, steps=args.steps, device=args.device)
        row = {"uid": uid, "level": m["level"], "mask_kind": m["mask_kind"],
               "n_pixels": int(x.shape[0]), "psnr_4d": r4["psnr"],
               "n_params": r4["n_params"], "curve_4d": r4["curve"], "sec": r4["sec"]}
        if args.also_3d:
            r3 = overfit_one(x, y, s, in_dim=3, steps=args.steps, device=args.device)
            row["psnr_3d_mlp"] = r3["psnr"]
            row["curve_3d"] = r3["curve"]
        out.append(row)
        print(f"[probe] {uid}: 4D={row['psnr_4d']:.2f} dB"
              + (f" | 3D-only={row.get('psnr_3d_mlp', float('nan')):.2f} dB"
                 if args.also_3d else ""), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ps = [r["psnr_4d"] for r in out]
    summary = {"n": len(out), "n_params": out[0]["n_params"] if out else None,
               "psnr_4d_min": float(np.min(ps)), "psnr_4d_mean": float(np.mean(ps)),
               "psnr_4d_median": float(np.median(ps)),
               "criterion_ge_45db_all": bool(np.min(ps) >= 45.0),
               "criterion_lt_40db_any": bool(np.min(ps) < 40.0),
               "per_sample": out}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_sample"},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
