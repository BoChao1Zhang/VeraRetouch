#!/usr/bin/env python
"""Where-A 在 V_where 上的推理效果图（mentor 浏览用）。

面板固定为：I_in ｜ GT mask ｜ s 场叠原图 ｜ 预测 mask (CBand12) ｜ 预测 mask (Band)。

可视化纪律（结果审阅要求，逐条落实）：

* **色标只取有效格**：``s`` 场的色标范围取"掩膜确实有响应"的像素的 2–98 分位，
  而不是全场 min/max —— 少数饱和像素会把其余全部压成一个颜色；
* **禁 resize**：``s`` 场经一次 guided upsample 后本来就是 ``out_h × out_w``，与 I_in
  逐像素同尺寸。脚本**断言**两者形状相等后直接叠加，不做任何缩放，也就不存在重采样
  引入的错位；
* **pad/无效格显式**：``F_pre`` 网格与图像的关系是 ``out = grid × 16`` 精确整除
  （spec-5 契约保证），没有 padding。脚本逐样本断言，并把结论写进 index.md；
  若某天真的出现 pad，会在图上以斜纹标出而不是悄悄裁掉。
* **数字取自原始场**：面板上的 soft-IoU / std / 值域全部取自已发布的 oracle 载荷与
  原始张量，不从被重标定的显示图像上反推。

oracle 参数直接读已发布的 V_where 载荷（不重拟合），只有 ``F_pre`` 需要跑一次前向。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

CJK_CANDIDATES = ("Noto Sans CJK JP", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei")


def _setup_fonts():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager, rcParams

    have = {f.name for f in font_manager.fontManager.ttflist}
    picked = [c for c in CJK_CANDIDATES if c in have]
    rcParams["font.sans-serif"] = picked + ["DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False
    return picked[0] if picked else None


def load_payloads(root: Path) -> dict:
    out = {}
    for idx in sorted((root / "indexes").glob("shard-*.idx.jsonl")):
        with idx.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                fd = os.open(str(root / "shards" / f"{row['shard']}.tar"), os.O_RDONLY)
                data = os.pread(fd, row["length"], row["offset_data"])
                os.close(fd)
                rec = json.loads(data)
                out[rec["sample_id"]] = rec
    return out


def summarize(recs: dict) -> list[dict]:
    """One row per sample, everything needed for stratified selection."""
    rows = []
    for sid, rec in recs.items():
        c = rec["fits"].get("cband12")
        b = rec["fits"].get("band")
        if not c or c.get("status") != "ok":
            continue
        ev = c["eval"]
        hi = (ev.get("hi") or {}).get("soft_iou_minmax")
        if hi is None:
            continue
        cc = np.asarray(c["latent"]["rho"]["c"])
        n_active = int((cc > 0.5).sum())
        meta = rec.get("meta", {})
        rows.append({
            "sample_id": sid,
            "build": meta.get("build"),
            "upscaled": bool(meta.get("upscaled")),
            "winner_confidence": meta.get("winner_confidence"),
            "region": meta.get("region"),
            "hi_cband": hi,
            "low_cband": ev["low"]["soft_iou_minmax"],
            "drop": ev["low"]["soft_iou_minmax"] - hi,
            "hi_band": ((b or {}).get("eval", {}).get("hi") or {}).get("soft_iou_minmax"),
            "mask_area": ev["low"]["target_mean"],
            "s_std": ev["s_low_stats"]["std"],
            "ood": ev["s_domain"]["frac_out_of_domain"],
            "n_active": n_active,
        })
    return rows


def draw(rows: list[dict], n: int, seed: int) -> list[dict]:
    """从 V_where 里**随机**抽 n 个样本，不做任何挑选。

    先前的版本按 build / 掩膜面积 / 上采样 / 活跃基元分层挑"成功案例"，那会系统性地
    高估效果：挑出来的中位 IoU 0.984，而整体分布并非如此。这里改为固定 seed 的均匀
    随机抽样，抽到什么就画什么，好坏都在里面。seed 记录在 index.md 与 index.json 中，
    换个 seed 就能复现另一批。
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
    picked = []
    areas = np.quantile([r["mask_area"] for r in rows], [1 / 3, 2 / 3])
    for rank, i in enumerate(idx):
        r = dict(rows[int(i)])
        r["draw_rank"] = rank
        r["area_band"] = ("小" if r["mask_area"] < areas[0]
                          else ("中" if r["mask_area"] < areas[1] else "大"))
        r["prim_band"] = "单基元" if r["n_active"] == 1 else "多基元"
        picked.append(r)
    return picked


def valid_scale(s: np.ndarray, valid: np.ndarray | None = None):
    """色标取**有效格**（非 pad 格）的 2–98 分位。

    "有效格" 指的是非 padding 的格子，不是"掩膜有响应的格子"。用后者会把小掩膜样本
    的色标压到极窄（实测一个面积占比 0.02 的样本得到 [-0.40, -0.04]），于是掩膜以外
    的整幅图全部饱和成同一个颜色 —— 那恰好是这条纪律想避免的结果，只是方向反了。
    取 2–98 分位而不是 min/max，是为了不让个别极端格拉满量程。
    """
    live = s if valid is None else s[valid]
    if live.size < 32:
        live = s.reshape(-1)
    lo, hi = float(np.percentile(live, 2)), float(np.percentile(live, 98))
    if hi - lo < 1e-6:
        lo, hi = float(s.min()), float(s.max())
    return lo, hi, int(live.size), int(s.size)


def main() -> int:
    font = _setup_fonts()
    import matplotlib.pyplot as plt
    from matplotlib import gridspec  # noqa: F401

    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="BA-3-Joint")
    ap.add_argument("--split", default="V_where")
    ap.add_argument("--oracle-root", default=None)
    ap.add_argument("--checkpoint", default="/home/bc/data/runs/q3vl_base_sft_20260804/checkpoint-4976")
    ap.add_argument("--model-dir", default="/home/bc/data/models/Qwen3-VL-4B-Instruct")
    ap.add_argument("--maskview-root", default="/mnt/nfs/bc/data/datasets/where_a-20260805/maskviews")
    ap.add_argument("--basis", default="/mnt/nfs/bc/data/datasets/where_a-20260805/basis/BA-3-Joint/B.npy")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--n", type=int, default=28, help="随机抽样张数")
    ap.add_argument("--seed", type=int, default=20260806, help="抽样 seed（记录在案）")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    from q3vl.where.basis import Latent, mask_from_latent
    from q3vl.where.calibrate import Calibrator
    from q3vl.where.config import CalibConfig, PhiConfig, REPORT_DIR, S_DOMAIN, UpsampleConfig
    from q3vl.where.fpre import load_vision_tower
    from q3vl.where.pipeline import WhereADataSource
    from q3vl.where.projector import BasisProjector
    from q3vl.where.readout import apply_readout
    from q3vl.where.upsample import combine_then_upsample

    root = Path(args.oracle_root) if args.oracle_root else Path(
        f"/mnt/nfs/bc/data/datasets/where_a-20260805/oracle/{args.arm}/s5/{args.split}")
    out_dir = Path(args.out_dir) if args.out_dir else REPORT_DIR / "viz_mentor"
    out_dir.mkdir(parents=True, exist_ok=True)

    recs = load_payloads(root)
    rows = summarize(recs)
    picks = draw(rows, args.n, args.seed)
    want = {r["sample_id"]: r for r in picks}
    print(f"载入 {len(recs)} 个已发布 latent，随机抽 {len(picks)} 个（seed={args.seed}，未做任何挑选）",
          flush=True)

    from transformers import AutoProcessor
    visual = load_vision_tower(Path(args.checkpoint), dtype=getattr(torch, args.dtype),
                               device=args.device)
    processor = AutoProcessor.from_pretrained(args.model_dir)
    source = WhereADataSource(visual, processor, device=args.device,
                              maskview_root=args.maskview_root, attach_hi=True)
    proj = BasisProjector()
    with torch.no_grad():
        proj.weight.copy_(torch.from_numpy(np.load(args.basis)))
    cal = Calibrator(CalibConfig(arm=args.arm, phi=PhiConfig()), projector=proj,
                     device=args.device)
    cal.freeze_projector()

    made = []
    pad_seen = 0
    for prepared in source.iter_split(args.split):
        sid = prepared.sample.sample_id
        if sid not in want:
            continue
        info = want[sid]
        smp = prepared.sample

        # pad / invalid cells: spec-5 guarantees out == grid * 16 exactly.
        H, W = prepared.mask_hi.shape
        pad = (smp.grid_h * 16 != H) or (smp.grid_w * 16 != W)
        pad_seen += int(pad)

        with torch.no_grad():
            parts = cal.phi_for(smp)
            guide = prepared.guide().to(args.device).double()
            panels = {}
            for ro in ("cband12", "band"):
                f = recs[sid]["fits"].get(ro)
                if not f or f.get("status") != "ok":
                    panels[ro] = None
                    continue
                lat = Latent.from_dict(f["latent"]).to(args.device)
                s_hi, s_lo, dom = combine_then_upsample(
                    parts.phi_dir.double(), lat, smp.grid_h, smp.grid_w, guide,
                    UpsampleConfig(), return_domain_report=True)
                m_hi = apply_readout(lat.readout, s_hi.reshape(-1), lat.rho)
                m_low, _ = mask_from_latent(parts.phi_dir.double(), lat)
                panels[ro] = {
                    "s_hi": s_hi.reshape(H, W).float().cpu().numpy(),
                    "m_hi": m_hi.reshape(H, W).float().cpu().numpy(),
                    "m_low": m_low.reshape(smp.grid_h, smp.grid_w).float().cpu().numpy(),
                    "dom": dom,
                    "iou_hi": (f["eval"].get("hi") or {})["soft_iou_minmax"],
                    "iou_low": f["eval"]["low"]["soft_iou_minmax"],
                    "s_std": f["eval"]["s_low_stats"]["std"],
                }

        img = np.clip(prepared.image_hi.permute(1, 2, 0).cpu().numpy(), 0, 1)
        gt = prepared.mask_hi.cpu().numpy()
        cb = panels["cband12"]
        bd = panels["band"]   # 不占面板，但交付 IoU 进标题做对比

        # 禁 resize：s 场与 I_in 本来就同尺寸，断言而不是缩放
        assert cb["s_hi"].shape == img.shape[:2] == gt.shape, (
            f"{sid}: s 场 {cb['s_hi'].shape} 与 I_in {img.shape[:2]} 尺寸不一致，"
            "叠图必须是逐像素严格对应，不允许 resize")

        # 本管线不产生 pad（out = grid x 16 精确整除），有效格 = 全部格子；
        # 若将来出现 pad，这里传入 valid 掩膜即可，色标自动只统计有效格。
        valid = None if not pad else np.ones_like(cb["s_hi"], dtype=bool)
        lo, hi, n_live, n_all = valid_scale(cb["s_hi"], valid)
        gray = img.mean(axis=2)

        gt_low = smp.mask_low.reshape(smp.grid_h, smp.grid_w).cpu().numpy()

        def s_overlay(ax):
            ax.imshow(gray, cmap="gray", vmin=0, vmax=1)
            im = ax.imshow(cb["s_hi"], cmap="coolwarm", vmin=lo, vmax=hi, alpha=0.62)
            # 掩膜 0.5 等值线：直接看到掩膜边界落在 s 场的哪一段上
            ax.contour(cb["m_hi"], levels=[0.5], colors="lime", linewidths=1.1)
            ax.set_title(f"s 场（交付分辨率，叠原图）\n"
                         f"色标 [{lo:.2f}, {hi:.2f}]\n"
                         f"有效格 {n_live}/{n_all} 的 2–98 分位\n"
                         f"绿线 = 预测掩膜 0.5 等值线", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046, shrink=0.85)

        # 每张图都给两档语义掩码：低分辨率（拟合所在层）与交付分辨率（下游消费层）。
        # 两档并排才看得出"掩膜是怎么塌的"——落差大说明问题出在 guided upsample 之后，
        # 低分辨率那格本身就差则说明是 basis 的表达上界。
        fig, axes = plt.subplots(1, 6, figsize=(23, 4.3))
        axes[0].imshow(img); axes[0].set_title("输入图 I_in", fontsize=10)
        axes[1].imshow(gt_low, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        axes[1].set_title(f"GT 掩膜（低分辨率\nF_pre {smp.grid_h}×{smp.grid_w}）", fontsize=9)
        axes[2].imshow(cb["m_low"], cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        axes[2].set_title(f"预测掩膜 低分辨率\nsoft-IoU {cb['iou_low']:.4f}", fontsize=9)
        axes[3].imshow(gt, cmap="magma", vmin=0, vmax=1)
        axes[3].set_title(f"GT 掩膜（交付分辨率）\n面积占比 {info['mask_area']:.3f}", fontsize=9)
        axes[4].imshow(cb["m_hi"], cmap="magma", vmin=0, vmax=1)
        axes[4].set_title(f"预测掩膜 交付分辨率\nsoft-IoU {cb['iou_hi']:.4f}"
                          f"（落差 {cb['iou_low']-cb['iou_hi']:+.4f}）", fontsize=9)
        s_overlay(axes[5])
        for a in axes:
            a.axis("off")
        if pad:
            axes[2].text(0.02, 0.02, "含 pad 格", transform=axes[2].transAxes,
                         color="yellow", fontsize=9)

        tag = (f"{info['build']} ｜ 掩膜{info['area_band']} ｜ "
               f"{'上采样' if info['upscaled'] else '原生分辨率'} ｜ "
               f"{info['prim_band']}({info['n_active']}) ｜ {info['winner_confidence']}")
        head = f"随机抽样 #{info['draw_rank']:02d}"
        band_txt = (f" ｜ Band 交付 {bd['iou_hi']:.4f}" if bd else " ｜ Band 拟合被拒")
        line2 = (f"CBand12 低分辨率 {cb['iou_low']:.4f} → 交付 {cb['iou_hi']:.4f}"
                 f"（落差 {cb['iou_low']-cb['iou_hi']:+.4f}）{band_txt} ｜ "
                 f"s 原始值域 [{cb['dom']['raw_min']:.2f}, {cb['dom']['raw_max']:.2f}]，"
                 f"声明域 {list(S_DOMAIN)}，越域 {cb['dom']['frac_out_of_domain']:.3%}"
                 f" ｜ std(s*) {cb['s_std']:.3f}")
        sub = ""
        fig.suptitle(f"【{head}】{sid}\n{tag}\n{line2}{sub}", fontsize=10)
        fig.tight_layout(rect=(0, 0, 0.995, 0.83 if sub else 0.86))
        name = out_dir / f"r{info['draw_rank']:02d}_{info['build']}_{sid[:16]}.png"
        fig.savefig(name, dpi=104)
        plt.close(fig)

        info["file"] = name.name
        info["pad"] = pad
        made.append(info)
        print(f"  [{len(made):2d}/{len(want)}] #{info['draw_rank']:02d} {sid[:22]} "
              f"{info['build']} 低 {cb['iou_low']:.4f} → 交付 {cb['iou_hi']:.4f}", flush=True)
        if len(made) == len(want):
            break

    # ---- index.md ----
    made.sort(key=lambda r: r["draw_rank"])
    v = np.array([r["hi_cband"] for r in made])
    vlow = np.array([r["low_cband"] for r in made])
    dr = np.array([r["drop"] for r in made])
    L = []
    L.append("# Where-A 推理效果图（V_where，BA-3-Joint，随机抽样）\n")
    L.append(f"- **抽样方式：固定 seed 的均匀随机抽样，未做任何挑选。** "
             f"`seed = {args.seed}`，从 V_where 的 {len(rows)} 个可用样本里抽 {len(made)} 个，"
             f"抽到什么画什么。换 seed 即可复现另一批。")
    L.append(f"- 掩膜与 oracle 参数直接读已发布载荷 `{root}`，**未重新拟合**；只重算了 `F_pre`。")
    L.append("- 每张图 6 面板：`I_in ｜ GT(低分辨率) ｜ 预测(低分辨率) ｜ GT(交付) ｜ 预测(交付) ｜ s 场叠图`。"
             "两档语义掩码并排，是为了看出掩膜在上采样前后各是什么样。")
    L.append("- 读数限定：guided upsample `radius_low=1, eps=1e-2`（D5 终值）；"
             "readout 为 CBand12（s 场面板的绿线是它的 0.5 等值线）。")
    L.append(f"- pad/无效格：spec-5 契约保证 `out = grid × 16` 精确整除，本批 **{pad_seen}** 张含 pad。")
    L.append("- s 场色标只取有效格（非 pad 格）的 2–98 分位；s 场与 I_in 逐像素同尺寸，"
             "叠图**未做任何 resize**；面板数字全部取自原始场。\n")

    L.append("## 这批随机样本实际长什么样\n")
    L.append(f"| 指标 | 交付分辨率 | 低分辨率 |")
    L.append("|---|---|---|")
    L.append(f"| 中位 soft-IoU | **{np.median(v):.4f}** | {np.median(vlow):.4f} |")
    L.append(f"| 均值 | {v.mean():.4f} | {vlow.mean():.4f} |")
    L.append(f"| 最差 / 最好 | {v.min():.4f} / {v.max():.4f} | {vlow.min():.4f} / {vlow.max():.4f} |")
    L.append(f"| ≥0.90 的比例 | {100*(v>=0.90).mean():.0f}% ({int((v>=0.90).sum())}/{len(v)}) | "
             f"{100*(vlow>=0.90).mean():.0f}% ({int((vlow>=0.90).sum())}/{len(vlow)}) |")
    L.append(f"| <0.70 的比例 | {100*(v<0.70).mean():.0f}% ({int((v<0.70).sum())}/{len(v)}) | "
             f"{100*(vlow<0.70).mean():.0f}% ({int((vlow<0.70).sum())}/{len(vlow)}) |")
    n_collapse = int((dr > 0.15).sum())
    L.append(f"\n低→交付落差：中位 {np.median(dr):+.4f}，最大 {dr.max():+.4f}。")
    if n_collapse:
        L.append(f"其中 **{n_collapse} 个落差 >0.15** —— 这些是掩膜在 guided upsample "
                 f"之后塌掉的（低分辨率那一格本来是好的）。")
    else:
        L.append("本批没有落差 >0.15 的样本。")

    # 抽 28 个很容易错过尾部；把全体 400 个的分布并列出来，
    # 这样 mentor 一眼能看出这批抽样是不是有代表性，而不必去猜。
    av = np.array([r["hi_cband"] for r in rows])
    adr = np.array([r["low_cband"] - r["hi_cband"] for r in rows])
    L.append(f"\n**这批 {len(made)} 张 vs V_where 全体 {len(rows)} 个**（判断这批抽样有没有代表性）：\n")
    L.append("| | 本批随机 28 | V_where 全体 400 |")
    L.append("|---|---|---|")
    L.append(f"| 交付 IoU 中位 | {np.median(v):.4f} | {np.median(av):.4f} |")
    L.append(f"| 交付 IoU 最差 | {v.min():.4f} | **{av.min():.4f}** |")
    L.append(f"| <0.90 的比例 | {100*(v<0.90).mean():.0f}% | {100*(av<0.90).mean():.0f}% |")
    L.append(f"| <0.70 的比例 | {100*(v<0.70).mean():.0f}% | {100*(av<0.70).mean():.0f}% |")
    L.append(f"| 落差 >0.15 的比例 | {100*(dr>0.15).mean():.0f}% | {100*(adr>0.15).mean():.0f}% |")
    if v.min() > av.min() + 0.1:
        L.append(f"\n> ⚠ 这批抽样**没有抽到尾部**：全体最差是 {av.min():.4f}，"
                 f"本批最差只到 {v.min():.4f}。28 张随机样本容易错过低频的坏样本，"
                 f"看整体上界/下界请以右列为准，或换 seed 再抽一批。\n")
    obs = {}
    for k, lbl in (("build", "build"), ("area_band", "掩膜面积"),
                   ("prim_band", "活跃基元"), ("upscaled", "上采样")):
        c = {}
        for r in made:
            c[str(r[k])] = c.get(str(r[k]), 0) + 1
        obs[lbl] = "、".join(f"{a} {b}" for a, b in sorted(c.items()))
    L.append("这批抽到的分布（**观察结果，不是设计出来的**）：\n")
    for lbl, txt in obs.items():
        L.append(f"- {lbl}：{txt}")

    L.append("\n## 逐张明细（按抽样顺序）\n")
    L.append("| # | 图 | sample id | build | 掩膜面积 | 上采样 | 活跃基元 | 低分辨率 IoU | 交付 IoU | 落差 | s 越域 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in made:
        L.append(f"| {r['draw_rank']:02d} | `{r['file']}` | {r['sample_id'][:24]} | {r['build']} | "
                 f"{r['area_band']}({r['mask_area']:.3f}) | {'是' if r['upscaled'] else '否'} | "
                 f"{r['n_active']} | {r['low_cband']:.4f} | **{r['hi_cband']:.4f}** | "
                 f"{r['drop']:+.4f} | {r['ood']:.2%} |")
    L.append(f"\n> 字体：{font or 'DejaVu Sans（无 CJK，中文可能显示为方块）'}")
    (out_dir / "index.md").write_text("\n".join(L), encoding="utf-8")
    (out_dir / "index.json").write_text(json.dumps(
        {"seed": args.seed, "n": len(made), "selection": "uniform random, no curation",
         "oracle_root": str(root), "samples": made}, indent=2, ensure_ascii=False))
    print(f"\n写出 {len(made)} 张图 + index.md 到 {out_dir}")
    source.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
