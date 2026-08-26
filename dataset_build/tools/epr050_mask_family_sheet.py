#!/usr/bin/env python3
"""EPR-050 v3.5: one sheet per source showing the four MAIN-CHAIN mask families.

semantic / radial / band / linear are drawn by the main chain's own code -- this
tool only calls dataset_build/src/construct and the EPR build module, it defines
no shape of its own.  The bottom row shows the v3.1-v3.4 `offset_ramp` linear for
reference: that is the shape the user flagged as a hard-edged narrow transition.

Each row: alpha field | alpha profile along the frame's long axis | the geometry
step applied to the source.  The degradation is a single LUT at the given
strength (not the six-step chain), so the panel isolates the mask.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import epr050_build_degradation as B  # noqa: E402
from epr050_build_degradation import (BANK, _font, apply_lut, geo_mask,  # noqa: E402
                                      gray_png, load_config, mainchain_geom,
                                      open_source, python_rng, to_png)
from q3vl.whatb.lutdata import mix_alpha  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
FAMILIES = ("semantic", "radial", "band", "linear")


def profile_plot(m: torch.Tensor, size=(300, 200)) -> Image.Image:
    """Mean alpha along whichever image axis the mask actually varies on (larger
    profile variance), so a horizontal and a vertical gradient are both legible.
    The axis used is printed under the plot."""
    a = m.cpu().numpy()
    px, py = a.mean(0), a.mean(1)
    use_x = float(px.var()) >= float(py.var())
    prof = px if use_x else py
    axis = "x (width)" if use_x else "y (height)"
    n = len(prof)
    W, H = size
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    f = _font(11)
    x0, y0, x1, y1 = 28, 8, W - 8, H - 30
    d.rectangle([x0, y0, x1, y1], outline=(180, 180, 180))
    for v in (0.0, 0.5, 1.0):
        y = y1 - v * (y1 - y0)
        d.line([x0, y, x1, y], fill=(228, 228, 228))
        d.text((4, y - 6), f"{v:.1f}", fill=(90, 90, 90), font=f)
    d.line([(x0 + i / (n - 1) * (x1 - x0), y1 - float(prof[i]) * (y1 - y0))
            for i in range(n)], fill=(200, 30, 30), width=2)
    d.text((x0 + 2, y1 + 4), f"mean alpha along {axis}", fill=(90, 90, 90), font=f)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="a v3.5 config (geom_sampler = mainchain_weights)")
    ap.add_argument("--source", default=None,
                    help="source_path; default = taken from --journal row --row")
    ap.add_argument("--subject", default=None, help="subject.png for that source")
    ap.add_argument("--journal",
                    default="/home/bc/data/builds/epr050-degrade-20260825/"
                            "v3.5_smoke/pairs.jsonl")
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--lut", default=None)
    ap.add_argument("--major", default="暖调高亮")
    ap.add_argument("--attempt", type=int, default=1,
                    help="attempt index fed to the content-keyed geometry rng")
    ap.add_argument("--offset-ramp-feather", type=float, default=0.10,
                    help="reference row: v3.4 feather, fraction of the SHORT side")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(
        REPO / "docs/assets/epr050_degrade_20260825/mask_families_v3.5.jpg"))
    a = ap.parse_args()

    cfg = load_config(Path(a.config))
    max_side = int(cfg["render"]["max_side"])
    dev = a.device

    src_path, subj_path, sid = a.source, a.subject, "sheet"
    if src_path is None:
        rec = [json.loads(l) for l in open(a.journal)][a.row]
        src_path, subj_path, sid = rec["source_path"], rec["subject_png"], rec["id"]
    src = open_source(src_path, max_side)
    h, w = src.shape[:2]
    x0 = torch.from_numpy(src.astype(np.float32) / 255.0).to(dev)

    subj = B.load_subject(Path(subj_path), h, w, dev)
    hard = (subj > 0.5).float().cpu().numpy()
    ys_, xs_ = np.nonzero(hard > 0.5)
    bbox = (float(xs_.min()) / w, float(ys_.min()) / h,
            float(xs_.max() + 1) / w, float(ys_.max() + 1) / h)
    area = float(hard.mean())

    lut = a.lut
    if lut is None:
        pool = json.loads(B.POOL_JSON.read_text())
        cand = sorted(r["name"] for r in pool if r["major"] == a.major
                      and r["recovered"] >= B.POOL_REC_MIN
                      and r["clip"] <= B.POOL_CLIP_MAX)
        if not cand:
            raise SystemExit(f"no main-pool LUT for major {a.major!r}")
        lut = cand[0]
    z = np.load(f"{BANK}/luts.npz")
    vol = torch.from_numpy(z[lut][None]).to(dev, torch.float32) \
        .permute(0, 4, 1, 2, 3).contiguous()

    rows = []
    for kind in FAMILIES:
        gp, m = mainchain_geom(kind, hard, bbox, area,
                               python_rng(sid, "geom", a.attempt), h, w, dev)
        if m is None:
            rows.append((f"{kind}  (main chain)", "sampler returned None "
                         "(geometry infeasible for this subject)", None, ""))
            continue
        g = gp.get("geom")
        sub = (f"mask_type={gp.get('mask_type')}  "
               + (json.dumps(g, ensure_ascii=False) if g else
                  f"subject_area={gp['subject_area']}"))
        if gp.get("width_tries"):
            sub += (f"   [v3.6 width={gp['width_value']} after {gp['width_tries']}"
                    f" draw(s), ok={gp['width_ok']}]")
        if kind == "semantic" and B.SEMANTIC_COVER_MIN:
            keep = area >= B.SEMANTIC_COVER_MIN
            sub += (f"   [v3.6 semantic gate: subject_area={area:.4f} "
                    f"{'>=' if keep else '<'} {B.SEMANTIC_COVER_MIN} -> "
                    f"{'kept' if keep else 'DROPPED from the family draw'}]")
        rows.append((f"{kind}  (main chain)", sub, m, ""))
    # reference: the v3.1-v3.4 shape, drawn by the legacy rasteriser
    ref = geo_mask(h, w, "linear",
                   dict(angle=np.pi / 2,
                        feather_px=round(a.offset_ramp_feather * min(h, w), 3),
                        offset=0.5), dev)
    rows.append(("offset_ramp  (v3.1-v3.4 linear, reference)",
                 f"feather = {a.offset_ramp_feather:.2f} x short side, offset = 0.50",
                 ref, ""))

    tiles = []
    for name, sub, m, _ in rows:
        if m is None:
            tiles.append((name, sub, None, "n/a"))
            continue
        y = mix_alpha(x0.reshape(1, -1, 3), apply_lut(vol, x0.reshape(1, -1, 3)),
                      (m * a.strength).reshape(1, -1, 1)).reshape(h, w, 3)
        # v3.6: the three numbers the criteria are written in -- frame coverage,
        # and the full-strength / transition split measured INSIDE the mask.
        st = B.geom_shape_stats(m)
        nums = (f"cover(a>{B.COVER_EPS}) = {st['coverage']:.4f} "
                f"(min {B.COVER_MIN})   "
                f"full_frac_of_mask(a>={B.FULL_EPS}) = "
                f"{st['full_frac_of_mask']:.4f} "
                f"(band [{B.FULL_OF_MASK_MIN}, {B.FULL_OF_MASK_MAX}])   "
                f"mid_frac_of_mask = {st['mid_frac_of_mask']:.4f}   "
                f"frac(a==0) = {st['frac_zero']:.4f}"
                + ("  [zero_band exempt]" if name.split()[0] in B.ZERO_BAND_EXEMPT
                   else "")
                + f"   span_long = {B.span_long_frac(m):.4f}")
        tiles.append((name, sub, [gray_png(m), profile_plot(m), to_png(y)], nums))

    panel_w = 300

    def fit(im, wid=panel_w):
        s = wid / im.width
        return im.resize((wid, max(1, round(im.height * s))), Image.LANCZOS)

    src_im = fit(to_png(x0))
    ph = max([src_im.height] + [max(fit(i).height for i in t[2])
                                for t in tiles if t[2]])
    hdr, cap = 30, 52
    W, H = panel_w * 4, hdr + len(tiles) * (ph + cap)
    sheet = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(sheet)
    fb, fh = _font(14), _font(11)
    d.text((4, 4), f"{sid}: main-chain mask families, same source and same LUT "
                   f"({lut}, major {a.major}), strength {a.strength:.2f}",
           fill="black", font=fb)
    titles = ["before (shared)", "alpha field", "alpha profile",
              "geometry step applied"]
    y = hdr
    for r, (name, sub, row, nums) in enumerate(tiles):
        panels = [src_im] + ([fit(i) for i in row] if row else [])
        for k, t in enumerate(panels):
            sheet.paste(t, (k * panel_w, y))
            d.rectangle([k * panel_w, y, k * panel_w + panel_w - 1,
                         y + t.height - 1], outline=(150, 150, 150))
            if r == 0:
                d.text((k * panel_w + 4, y - 13), titles[k], fill=(90, 90, 90),
                       font=fh)
        if not row:
            d.text((panel_w + 10, y + ph // 2), "n/a", fill="gray", font=fb)
        d.text((4, y + ph + 3), name, fill="black", font=fb)
        d.text((4, y + ph + 20), f"{sub}", fill=(60, 60, 60), font=fh)
        d.text((4, y + ph + 31), nums, fill="black", font=fh)
        y += ph + cap

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=94)
    print(f"wrote {out}  ({W}x{H})  source={sid}  lut={lut}")
    for name, sub, _, nums in tiles:
        print(f"  {name}\n      {sub}\n      {nums}")


if __name__ == "__main__":
    main()
