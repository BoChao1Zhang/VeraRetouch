"""G2 可视化：三档 Δ_ceil 直方图并排 + 高/低 Δ_ceil 典型样本图。

样本图六联：I_in | I_tar | s 场(C_GT) | |I_tar−I_in|×5 | 3D 天花板残差×5 |
4D 天花板残差×5 —— 直接看出「3D 解释不掉的那部分是不是恰好长在掩膜里」。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import sys  # noqa: E402
sys.path.insert(0, "/home/bc/VeraRetouch")
from tools.ceiling.delta_ceil import _group_affine, _hier_3d_affine, _refine, \
    s_quantile_buckets, N_MIN_AFFINE, N_MIN_OFFSET, S_BUCKETS  # noqa: E402


# --------------------------------------------------------------------------- 直方图

def hist_three_tracks(aggs: dict[str, list[float]], out: Path, key: str = "Δ_ceil",
                      criteria: dict[str, float] | None = None) -> None:
    tracks = list(aggs.keys())
    fig, axes = plt.subplots(1, len(tracks), figsize=(5.2 * len(tracks), 4.0), sharey=False)
    if len(tracks) == 1:
        axes = [axes]
    for ax, t in zip(axes, tracks):
        v = np.asarray(aggs[t], dtype=np.float64)
        v = v[np.isfinite(v)]
        lo, hi = float(np.min(v)), float(np.max(v))
        bins = np.linspace(min(lo, 0) - 0.2, hi + 0.2, 40)
        ax.hist(v, bins=bins, color="#3b6ea5", edgecolor="white", linewidth=0.4)
        med = float(np.median(v))
        ax.axvline(med, color="#c0392b", lw=2,
                   label=f"median {med:.2f} dB")
        if criteria and t in criteria:
            ax.axvline(criteria[t], color="#27ae60", lw=1.6, ls="--",
                       label=f"判据 {criteria[t]:g} dB")
        ax.set_title(f"{t}  (n={v.size})")
        ax.set_xlabel(f"{key} (dB)")
        ax.set_ylabel("images")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.suptitle(f"G2 oracle 天花板：{key} 三档并排", fontsize=13)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- 样本图

def _ceiling_maps(x, y, s, hw, device="cuda"):
    """返回 (pred3, pred4) 的残差幅值图（(H,W) float）。"""
    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    st = torch.as_tensor(s, dtype=torch.float32, device=device)
    cell3, pred3 = _hier_3d_affine(xt, yt, n_min=N_MIN_AFFINE)
    sb = s_quantile_buckets(st, S_BUCKETS)
    pred4, _ = _refine(cell3 * S_BUCKETS + sb, xt, yt, pred3,
                       N_MIN_AFFINE, N_MIN_OFFSET, "affine")
    h, w = hw
    r3 = (pred3 - yt).abs().mean(1).reshape(h, w).cpu().numpy()
    r4 = (pred4 - yt).abs().mean(1).reshape(h, w).cpu().numpy()
    return r3, r4


def sample_panel(x, y, s, hw, title: str, out: Path, device: str = "cuda") -> None:
    h, w = hw
    xi = x.reshape(h, w, 3)
    yi = y.reshape(h, w, 3)
    si = s.reshape(h, w)
    r3, r4 = _ceiling_maps(x, y, s, hw, device)
    diff = np.abs(yi - xi).mean(-1)
    panels = [(xi, "I_in", None), (yi, "I_tar", None), (si, "s = C_GT", "magma"),
              (np.clip(diff * 5, 0, 1), "|I_tar−I_in|×5", "inferno"),
              (np.clip(r3 * 5, 0, 1), "3D 天花板残差×5", "inferno"),
              (np.clip(r4 * 5, 0, 1), "4D 天花板残差×5", "inferno")]
    fig, axes = plt.subplots(1, 6, figsize=(21, 21 * h / (6 * w) + 0.9))
    for ax, (img, name, cmap) in zip(axes, panels):
        ax.imshow(img, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- CLI

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-image", nargs="+", type=Path, required=True,
                    help="per_image_<track>.jsonl，顺序 = 图上从左到右")
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--index", type=Path, required=True, help="l 系索引（取样本图）")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--key", default="delta_arm")
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    data: dict[str, list[dict]] = {}
    for lab, p in zip(args.labels, args.per_image):
        data[lab] = [json.loads(l) for l in open(p, encoding="utf-8")]

    hist_three_tracks({k: [r[args.key] for r in v] for k, v in data.items()},
                      args.out_dir / f"hist_{args.key}_three_tracks.png",
                      key=f"Δ_ceil ({args.key})",
                      criteria={"D-CONSTRUCT(S-val)": 8.0,
                                "D-SFT-L(S-val,normal) 真实档": 1.0,
                                "D-SFT-G(S-val) 对照": 0.0})

    # 高/低 Δ_ceil 典型样本（取真实档）
    real_label = next((l for l in args.labels if "SFT-L" in l or "real" in l), args.labels[0])
    rows = sorted(data[real_label], key=lambda r: r[args.key])
    idx = {r["candidate_id"]: r for r in
           (json.loads(l) for l in open(args.index, encoding="utf-8"))}
    from tools.ceiling.loader import load_triplet

    for tag, subset in (("success", rows[-args.n_samples:][::-1]),
                        ("failure", rows[:args.n_samples])):
        for j, r in enumerate(subset):
            ir = idx.get(r["id"])
            if ir is None:
                continue
            t = load_triplet(ir)
            if t is None:
                continue
            title = (f"[{tag}] {r['id'][:24]} pool={r.get('pool')} "
                     f"Δ_ceil(arm)={r['delta_arm']:.2f} dB  Δ_nested={r['delta']:.2f}  "
                     f"Δ_donor={r.get('delta_donor', float('nan')):.2f}  "
                     f"PSNR 3D={r['psnr_3d']:.2f}→4D_arm={r['psnr_4d_arm']:.2f}  "
                     f"s̄={t.s.mean():.3f}")
            sample_panel(t.x, t.y, t.s, t.hw, title,
                         args.out_dir / f"{tag}_{j}_{r['id'][-10:]}.png", args.device)
            print(f"[viz] {tag}_{j} {r['id'][-10:]} Δ={r[args.key]:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
