"""INF-1 真实数据冒烟测试 — 在落盘 build 的小样本上跑通全 harness.

数据: /mnt/nfs/bc/data/datasets/sft/<build>/batch-*/shards/*.tar,
样本三件套 {sid}.in.jpg (输入) / {sid}.jpg (输出 GT) / {sid}.cgt.png (软掩膜, 部分样本有).

跑什么:
1. metrics: 输入 vs GT 全指标 (masked 三分用 .cgt.png), 可选 LPIPS;
2. collapse probes: oracle 混合渲染器 (s=cgt 下采 32×32, render = image*(1-s)+gt*s)
   -> Δ_const / Δ_shuffle 应显著为正 (真实图上的 sanity);
3. stats: 配对比较 "oracle 渲染 vs 恒等渲染" 的逐图 PSNR;
4. 产出 metrics.json (README schema) + viz + leaderboard demo 到报告目录.

运行: /home/bc/miniconda3/bin/python3 tools/harness/smoke_real.py [--n 6] [--lpips]
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import metrics
import stats
from collapse_probes import run_probes
from leaderboard import build_leaderboard

SHARD = "/mnt/nfs/bc/data/datasets/sft/mini30-v52-20260730/batch-0000/shards/shard-00000.tar"
OUT_DIR = Path("/home/bc/VeraRetouch/experiments/tooling-wave1/harness")


def load_samples(shard: str, n: int) -> list:
    """取前 n 个带 .cgt.png 的样本. 返回 [{'id','image','gt','mask'}]."""
    samples = []
    with tarfile.open(shard) as tf:
        names = set(tf.getnames())
        # 只取三件齐全的样本 (部分样本无 .in.jpg, 见 NOTES.md 冒烟记录)
        sids = sorted(
            sid
            for sid in {nm[: -len(".cgt.png")] for nm in names if nm.endswith(".cgt.png")}
            if f"{sid}.in.jpg" in names and f"{sid}.jpg" in names
        )[:n]
        for sid in sids:
            def _img(suffix, mode="RGB"):
                fo = tf.extractfile(f"{sid}{suffix}")
                assert fo is not None, f"member {sid}{suffix} is not a regular file"
                data = fo.read()
                return np.asarray(Image.open(io.BytesIO(data)).convert(mode))

            img, gt, m = _img(".in.jpg"), _img(".jpg"), _img(".cgt.png", "L")
            if img.shape != gt.shape:  # 个别 build 输入/输出分辨率不同则对齐到 GT
                img = np.asarray(Image.fromarray(img).resize((gt.shape[1], gt.shape[0]), Image.Resampling.BILINEAR))
            if m.shape != gt.shape[:2]:
                m = np.asarray(Image.fromarray(m).resize((gt.shape[1], gt.shape[0]), Image.Resampling.BILINEAR))
            samples.append({"id": sid.split("_candidate_")[-1][:8], "image": img, "gt": gt, "mask": m})
    return samples


def s_from_mask(mask: np.ndarray, res: int = 32) -> np.ndarray:
    """oracle s 场: cgt 软掩膜下采到 res×res (INF-5 的 32×32 口径), HxWx1. 不做任何归一化."""
    s = np.asarray(Image.fromarray(mask).resize((res, res), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
    return s[..., None]


def oracle_render(image: np.ndarray, s: np.ndarray) -> np.ndarray:
    """render_fn(image, s): s 双线性上采回原图, out = image*(1-s) + 该图 GT*s.

    GT 经闭包绑定在 sample 上 (见 main 中的包装); 此处签名版仅用于文档.
    """
    raise NotImplementedError("wrapped per-sample in main()")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--lpips", action="store_true", help="附带 LPIPS (需 torchvision 权重)")
    ap.add_argument("--shard", default=SHARD)
    args = ap.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "viz").mkdir(exist_ok=True)
    samples = load_samples(args.shard, args.n)
    print(f"[smoke] loaded {len(samples)} samples with cgt masks from {args.shard}")

    # ---- 1. metrics: 输入 vs GT --------------------------------------
    batch = [{"pred": s["image"], "gt": s["gt"], "mask": s["mask"], "id": s["id"]} for s in samples]
    lpips_note = "skipped"
    try:
        res = metrics.evaluate_batch(batch, with_lpips=args.lpips)
        if args.lpips:
            lpips_note = "ok"
    except Exception as e:  # LPIPS 权重下载失败等 → 回退无 LPIPS
        print(f"[smoke] lpips failed ({e}); retrying without", file=sys.stderr)
        res = metrics.evaluate_batch(batch, with_lpips=False)
        lpips_note = f"failed: {e}"
    for r in res["per_image"]:
        print(
            f"  {r['id']}: full={r['psnr_full']:.2f} in={r['psnr_in']:.2f} "
            f"band={r['psnr_band']:.2f} out={r['psnr_out']:.2f} ssim={r['ssim']:.4f} "
            f"dE00={r['delta_e00']:.2f}"
        )

    # ---- 2. collapse probes: oracle 渲染 -----------------------------
    gt_by_key = {id(s["image"]): s["gt"] for s in samples}

    def render(image, s):
        gt = gt_by_key[id(image)]
        h, w = gt.shape[:2]
        s2 = np.asarray(
            Image.fromarray((np.clip(np.asarray(s)[..., 0], 0, 1) * 255).astype(np.uint8)).resize(
                (w, h), Image.Resampling.BILINEAR
            ),
            dtype=np.float64,
        )[..., None] / 255.0
        return metrics.to_float01(image) * (1 - s2) + metrics.to_float01(gt) * s2

    probe_samples = [
        {"id": s["id"], "image": s["image"], "gt": s["gt"], "s": s_from_mask(s["mask"])}
        for s in samples
    ]
    probes = run_probes(render, probe_samples, seed=0)
    print(
        f"[smoke] probes: Δ_const={probes['delta_const']:.3f} dB, "
        f"Δ_shuffle={probes['delta_shuffle']:.3f} dB (oracle 渲染, 应显著为正)"
    )

    # ---- 3. stats: oracle 渲染 vs 恒等渲染 配对 ----------------------
    psnr_oracle = [
        metrics.psnr_full(render(s["image"], ps["s"]), s["gt"])
        for s, ps in zip(samples, probe_samples)
    ]
    psnr_identity = [metrics.psnr_full(s["image"], s["gt"]) for s in samples]
    cmp = stats.compare(psnr_oracle, psnr_identity, n_boot=2000, seed=0)
    print(
        f"[smoke] paired ΔPSNR={cmp['delta_psnr_mean']:+.3f} dB, "
        f"CI95=[{cmp['ci95'][0]:+.3f}, {cmp['ci95'][1]:+.3f}], sign p={cmp['sign_test']['p_value']:.3g}"
    )

    # ---- 4. 落盘: metrics.json + viz + leaderboard demo --------------
    mjson = {
        "exp_id": "harness-smoke-oracle",
        "arm": "smoke",
        "date": "2026-08-02",
        "n_images": len(samples),
        "config": {"band_px": metrics.DEFAULT_BAND_PX, "mask_thr": metrics.DEFAULT_MASK_THR,
                   "cap_db": metrics.DEFAULT_CAP_DB, "shard": args.shard, "lpips": lpips_note},
        "metrics": {
            **{k: res["mean"][k] for k in
               ("psnr_in", "psnr_band", "psnr_out", "psnr_full", "ssim", "delta_e00")},
            **({"lpips": res["mean"]["lpips"]} if "lpips" in res["mean"] else {}),
            "delta_const": probes["delta_const"],
            "delta_shuffle": probes["delta_shuffle"],
        },
        "probes": probes,
        "ci": cmp,
        "per_image": res["per_image"],
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(mjson, indent=2, ensure_ascii=False))
    print(f"[smoke] wrote {OUT_DIR / 'metrics.json'}")

    md, warns = build_leaderboard([OUT_DIR / "metrics.json"])
    (OUT_DIR / "leaderboard_demo.md").write_text(md)
    for w in warns:
        print(w, file=sys.stderr)

    # viz: 前 2 样本五联图 输入/GT/软掩膜/三分区/|diff|
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # success: psnr_in 最好 2 样本; failure: psnr_in 最差样本 (真实数据里的困难案例)
    pairs = sorted(zip(samples, res["per_image"]), key=lambda p: p[1]["psnr_in"], reverse=True)
    to_plot = [(s, r, "success") for s, r in pairs[:2]] + [(pairs[-1][0], pairs[-1][1], "failure")]
    for s, r, kind in to_plot:
        reg = metrics.region_partition(s["mask"])
        reg_rgb = np.zeros(s["mask"].shape + (3,))
        reg_rgb[reg["in"]] = [0.9, 0.3, 0.2]
        reg_rgb[reg["band"]] = [0.95, 0.8, 0.2]
        reg_rgb[reg["out"]] = [0.2, 0.4, 0.8]
        diff = np.abs(metrics.to_float01(s["image"]) - metrics.to_float01(s["gt"])).mean(-1)
        fig, axes = plt.subplots(1, 5, figsize=(20, 4.2))
        panels = [(s["image"], "input (.in.jpg)"), (s["gt"], "GT (.jpg)"),
                  (s["mask"], "soft mask (.cgt.png)"), (reg_rgb, "in/band/out (band_px=3)"),
                  (diff, "|input-GT| mean")]
        for ax, (im, title) in zip(axes, panels):
            ax.imshow(im, cmap="gray" if im.ndim == 2 and title.startswith("soft") else
                      ("magma" if im.ndim == 2 else None))
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        fig.suptitle(
            f"{s['id']}  PSNR full={r['psnr_full']:.2f} in={r['psnr_in']:.2f} "
            f"band={r['psnr_band']:.2f} out={r['psnr_out']:.2f}  SSIM={r['ssim']:.3f}  "
            f"ΔE00={r['delta_e00']:.2f}", fontsize=11)
        fig.tight_layout()
        out_png = OUT_DIR / "viz" / f"{kind}_smoke_{s['id']}.png"
        fig.savefig(out_png, dpi=110)
        plt.close(fig)
        print(f"[smoke] wrote {out_png}")

    print("[smoke] DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
