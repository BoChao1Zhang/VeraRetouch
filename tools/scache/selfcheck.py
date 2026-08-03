"""T5 自检：prod-l1 前 100 组 写→读→上采样 往返 + IoU 报告 + 可视化。

判据（任务卡 T5 / EXPERIMENTS_v3 INF-5）：上采样恢复的掩膜与原 C_GT 掩膜
在 0.5 阈值下 IoU > 0.95（mean 口径判定，逐张分布一并报告）。

流程：
  1. oracle.build_oracle: 前 N 组（默认 100）C_GT -> 32x32 float16 缓存（写）；
  2. SCache.iter_entries 读回（读）；
  3. 按 meta.origin 的 (shard, offset) 引用回读全分辨率掩膜与输入图；
  4. upsample_s (bilinear + guided_blur, guide=原图) 恢复（上采样）；
  5. IoU@0.5 vs 原掩膜；对照组：纯 bilinear（无引导）；
  6. metrics.json + 3 张三列拼图（原掩膜 / 32x32 缓存 / 上采样恢复）。

CLI:
    python3 tools/scache/selfcheck.py --out experiments/tooling-wave1/scache \
        [--build ...] [--journal ...] [--groups 100] [--size 32] [--device cuda:0]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from api import SCache  # noqa: E402
from oracle import ORACLE_ARM, build_oracle, load_guide, load_mask  # noqa: E402
from upsample import DEFAULT_EPS, DEFAULT_SUBSAMPLE, iou_at, upsample_s  # noqa: E402

DEFAULT_BUILD = Path("/mnt/nfs/bc/data/datasets/sft/prod-l1-local17k-20260731")
DEFAULT_JOURNAL = Path(
    "/var/cache/veradata/annot_review/journal-archive/prod-l1-local17k-20260731"
)


def _to_u8(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def make_triptych(mask: np.ndarray, s_small: np.ndarray, restored: np.ndarray,
                  out_path: Path, label_h: int = 28) -> None:
    """三列拼图：原掩膜 / 32x32 缓存（最近邻放大）/ 上采样恢复。"""
    from PIL import ImageDraw

    H, W = mask.shape
    small_big = np.asarray(
        Image.fromarray(_to_u8(s_small), mode="L").resize((W, H), Image.Resampling.NEAREST),
        dtype=np.uint8,
    )
    cols = [_to_u8(mask), small_big, _to_u8(restored)]
    gap = 8
    canvas = Image.new("L", (W * 3 + gap * 2, H + label_h), color=64)
    for i, c in enumerate(cols):
        canvas.paste(Image.fromarray(c, mode="L"), (i * (W + gap), label_h))
    draw = ImageDraw.Draw(canvas)
    titles = ["C_GT (full res)", f"cache {s_small.shape[0]}x{s_small.shape[1]}", "guided upsample"]
    for i, t in enumerate(titles):
        draw.text((i * (W + gap) + 6, 6), t, fill=255)
    canvas.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="s 缓存往返自检（prod-l1 前 100 组）")
    ap.add_argument("--out", required=True, type=Path, help="输出目录（缓存/metrics/viz）")
    ap.add_argument("--build", type=Path, default=DEFAULT_BUILD)
    ap.add_argument("--journal", type=Path, default=DEFAULT_JOURNAL)
    ap.add_argument("--groups", type=int, default=100)
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--eps", type=float, default=DEFAULT_EPS)
    ap.add_argument("--subsample", type=int, default=DEFAULT_SUBSAMPLE)
    ap.add_argument("--kernel-size", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--viz-n", type=int, default=3)
    args = ap.parse_args()

    out_dir = args.out
    root = out_dir / "s_cache"
    viz_dir = out_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. 写 ----
    t0 = time.time()
    stats = build_oracle(
        args.build, root, journal_dir=args.journal,
        size=args.size, max_groups=args.groups,
    )
    t_write = time.time() - t0
    print(f"[selfcheck] oracle built: {stats} in {t_write:.1f}s", flush=True)

    # ---- 2..5. 读 + 上采样 + IoU ----
    cache = SCache(root, ORACLE_ARM, resolution=args.size)
    rows = []
    t1 = time.time()
    for i, entry in enumerate(cache.iter_entries()):
        origin = entry.meta["origin"]
        mask = load_mask(origin["mask_member"])                      # 全分辨率 C_GT [0,1]
        guide = load_guide(origin["input_member"], tuple(origin["mask_hw"]))
        s_small = entry.s.astype(np.float32)                         # 读回 float16 -> f32

        restored = upsample_s(
            s_small, guide, kernel_size=args.kernel_size,
            eps=args.eps, subsample=args.subsample, device=args.device,
        )
        # 对照：纯 bilinear（无引导）
        bil = np.asarray(
            Image.fromarray(s_small.astype(np.float32), mode="F").resize(
                (mask.shape[1], mask.shape[0]), Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        )
        rows.append({
            "img_id": entry.img_id,
            "instr_hash": entry.instr_hash,
            "group_id": origin["group_id"],
            "mask_hw": origin["mask_hw"],
            "mask_area_frac": float((mask > 0.5).mean()),
            "iou_guided": iou_at(restored, mask, 0.5),
            "iou_bilinear": iou_at(bil, mask, 0.5),
        })
        if (i + 1) % 100 == 0:
            print(f"[selfcheck] roundtrip {i + 1} done", flush=True)
    t_round = time.time() - t1

    ious = np.array([r["iou_guided"] for r in rows])
    ious_b = np.array([r["iou_bilinear"] for r in rows])
    per_group = {}
    for r in rows:
        per_group.setdefault(r["group_id"], []).append(r["iou_guided"])
    group_mean = np.array([float(np.mean(v)) for v in per_group.values()])

    metrics = {
        "spec": "EXPERIMENTS_v3 INF-5 / 任务卡 T5",
        "criterion": "IoU@0.5 (guided upsample vs C_GT) > 0.95",
        "build": str(args.build),
        "n_groups": stats["groups"],
        "n_entries": len(rows),
        "cache_size": args.size,
        "upsample": {
            "eps": args.eps, "subsample": args.subsample,
            "kernel_size": args.kernel_size or "auto",
        },
        "iou_guided": {
            "mean": float(ious.mean()), "median": float(np.median(ious)),
            "min": float(ious.min()), "max": float(ious.max()),
            "frac_gt_0.95": float((ious > 0.95).mean()),
            "frac_gt_0.90": float((ious > 0.90).mean()),
        },
        "iou_bilinear_baseline": {
            "mean": float(ious_b.mean()), "median": float(np.median(ious_b)),
            "min": float(ious_b.min()),
            "frac_gt_0.95": float((ious_b > 0.95).mean()),
        },
        "iou_group_mean": {
            "mean": float(group_mean.mean()), "min": float(group_mean.min()),
        },
        "pass_mean_gt_0.95": bool(ious.mean() > 0.95),
        "timing_sec": {"build_write": round(t_write, 1), "roundtrip": round(t_round, 1)},
        "per_entry": rows,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    # ---- 6. 可视化：best / median / worst ----
    order = np.argsort(ious)
    picks = {"worst": int(order[0]), "median": int(order[len(order) // 2]),
             "best": int(order[-1])}
    for tag, idx in picks.items():
        r = rows[idx]
        entry = cache.read(r["img_id"], r["instr_hash"])
        origin = entry.meta["origin"]
        mask = load_mask(origin["mask_member"])
        guide = load_guide(origin["input_member"], tuple(origin["mask_hw"]))
        restored = upsample_s(
            entry.s.astype(np.float32), guide, kernel_size=args.kernel_size,
            eps=args.eps, subsample=args.subsample, device=args.device,
        )
        out_png = viz_dir / f"roundtrip_{tag}_iou{r['iou_guided']:.3f}_{r['img_id']}.png"
        make_triptych(mask, entry.s.astype(np.float32), restored, out_png)
        print(f"[selfcheck] viz {tag}: {out_png.name}", flush=True)

    summary = {k: v for k, v in metrics.items() if k != "per_entry"}
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
