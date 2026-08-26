#!/usr/bin/env python3
"""EPR-050: one per-sample sheet showing the restoration unwinding step by step.

Why a separate file: `epr050_build_degradation.py`'s sha256 is frozen into
`run_args.json`, so editing it after a run makes every later resume of the same
journal fail the identity guard.  This tool imports that module and calls its
functions; it never modifies it.  Every experiment-semantic number is taken from
the TOML recorded in `<out>/run_args.json` (or `--config`), never hard-coded.

Layout, per sample, two aligned rows of 7 panels, ordered along the RESTORATION
direction, i.e. the REVERSE of `[run] step_order` -- the column order is derived
from the config, never hard-coded, so it follows the chain order of the run being
drawn (v3.2: geom last, so it is unwound first; v3.3: geom first, so it is
unwound last):

  row 1   after | alpha of step N | ... | alpha of step 1 | before
  row 2   after |     y_{N-1}     | ... |     x_hat       | err

  column j (1..N) pairs the alpha field of the step being inverted with the
  image obtained right after inverting it.  The alpha shown is the ACTUAL field
  used, i.e. already multiplied by the calibrated global strength s.
  The error panel is |x_hat - before| on the same absolute colour scale
  (0 .. err_vmax 8-bit levels) as every other EPR-050 sheet.

The intermediate restored images are not on disk (the build tool only saves the
final x_hat), so they are recomputed here from what pairs.jsonl records: the five
preset ids, the mask parameters, and s.  The recomputation goes through the build
module's own functions, and `--verify` asserts that the recomputed x_hat renders
bit-for-bit identically to the `restored_e.png` asset the build run wrote.
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

# labels come from the build module so the two sheets can never disagree about
# what a step kind is called (and a new kind cannot be forgotten here)
def unwind_order() -> list[int]:
    """Chain positions in restoration order = the reverse of [run] step_order."""
    return list(reversed(range(B.N_STEPS)))


def alpha_title(k: int) -> str:
    return f"step{k+1} alpha ({B.KIND_LABEL[B.STEP_KIND[k]]})"


def img_title(k: int) -> str:
    tail = " = x_hat" if k == 0 else ""
    return f"after unwind step{k+1} ({B.KIND_LABEL[B.STEP_KIND[k]]}){tail}"


def rebuild(row: dict, z, dev: str, max_side: int):
    """Recompute the alpha fields, the forward chain and the full unwind.

    Returns (x0, fields, ys, est, iters) with fields/ys/est indexed by step.
    Every parameter comes from the recorded row; nothing is resampled.
    """
    gid = row["id"]
    src = B.open_source(row["source_path"], max_side)
    h, w = src.shape[:2]
    if [h, w] != list(row["size"]):
        B.die(f"{gid}: size drift {[h, w]} vs recorded {row['size']}")
    x0 = torch.from_numpy(src.astype(np.float32) / 255.0).to(dev)

    m = row["mask"]
    bands, (t_lo, t_hi) = B.lum_bands(x0, m["q_lo"], m["q_hi"], m["width"])
    if abs(t_lo - m["t_lo"]) > 1e-6 or abs(t_hi - m["t_hi"]) > 1e-6:
        B.die(f"{gid}: luminance threshold drift "
              f"({t_lo}, {t_hi}) vs recorded ({m['t_lo']}, {m['t_hi']})")
    # v3.5/v3.6 rows carry the MAIN CHAIN's geometry dict, so the alpha is
    # replayed by the build module's rebuilder (raster_geometry on the recorded
    # LR parameters, or _semantic_alpha on the hard subject mask with the row's
    # own content key).  Legacy rows still go through geo_mask() inside it.
    hard = None
    if m["geom"] in ("subject", "semantic"):
        subj = B.load_subject(Path(row["subject_png"]), h, w, dev)
        hard = ((subj > 0.5).float().cpu().numpy() if m["geom"] == "semantic"
                else subj.cpu().numpy())
    m_geo = B.geom_alpha_from_row(row, h, w, dev, hard)

    # chain position -> alpha field, ordered by [run] step_order (v3.3); the
    # fields themselves are order-independent (all derived from x0)
    by_kind = dict(lum_high=bands["lum_high"], lum_mid=bands["lum_mid"],
                   lum_shadow=bands["lum_shadow"],
                   **{"global": torch.ones((h, w), device=dev)}, geom=m_geo)
    if "hue" in B.STEP_KIND:
        # v3.4: content-determined, no sampled parameters, same function as build
        by_kind["hue"] = B.hue_mask(x0)
    fields = [by_kind[k] for k in B.STEP_KIND]
    if [s["kind"] for s in row["steps"]] != list(B.STEP_KIND):
        B.die(f"{gid}: config step_order {list(B.STEP_KIND)} disagrees with the "
              f"recorded chain {[s['kind'] for s in row['steps']]}")
    s_glob = row["calib"]["s"]
    if s_glob != 1.0:
        fields = [f * s_glob for f in fields]
    alphas = [f.reshape(1, -1, 1).contiguous() for f in fields]

    vols = [torch.from_numpy(z[n][None]).to(dev, torch.float32)
            .permute(0, 4, 1, 2, 3).contiguous() for n in row["luts"]]

    xf = x0.reshape(1, -1, 3)
    ys = [xf]
    for k in range(B.N_STEPS):
        prev = ys[-1]
        ys.append(B.mix_alpha(
            prev, B.chunked(lambda t, v=vols[k]: B.apply_lut(v, t), prev), alphas[k]))

    est = [None] * (B.N_STEPS + 1)
    est[B.N_STEPS] = ys[-1]
    iters = [0] * B.N_STEPS
    for k in range(B.N_STEPS - 1, -1, -1):
        xh, it = B.chunked(lambda t, a, v=vols[k]: B.invert_blend(v, t, a),
                           est[k + 1], alphas[k])
        est[k], iters[k] = xh, it
    del vols
    return x0, fields, ys, est, iters


def sheet(row: dict, x0, fields, est, iters, out_path: Path, panel_w: int):
    h, w = row["size"]
    scale = panel_w / w
    ph = max(1, round(h * scale))
    hdr, cap, foot = 20, 18, 46
    ncol = B.N_STEPS + 2
    W = panel_w * ncol
    H = hdr + ph + cap + hdr + ph + cap + foot
    sheet_im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(sheet_im)
    fb, fs = B._font(13), B._font(11)

    after_im = B.to_png(est[B.N_STEPS].reshape(h, w, 3))
    err = (est[0] - x0.reshape(1, -1, 3)).abs().amax(-1)[0] * B.LEVEL
    err_im = Image.fromarray(B.colorize(err.reshape(h, w).cpu().numpy()))

    order = unwind_order()
    top = [(after_im, "after (degraded)")]
    bot = [(after_im, "after (start)")]
    for k in order:
        top.append((B.gray_png(fields[k]), alpha_title(k)))
        bot.append((B.to_png(est[k].reshape(h, w, 3)), img_title(k)))
    top.append((B.to_png(x0), "before (source)"))
    bot.append((err_im, f"err |x_hat - before| 0..{B.ERR_VMAX:.0f} lv"))

    def draw_row(cells, y0):
        for j, (im, title) in enumerate(cells):
            x = j * panel_w
            d.text((x + 4, y0 + 3), title, fill="black", font=fb)
            p = im.resize((panel_w, ph), Image.LANCZOS)
            sheet_im.paste(p, (x, y0 + hdr))
            # step 4's alpha is 1 everywhere (a flat panel); a frame keeps it
            # distinguishable from an empty cell
            d.rectangle([x, y0 + hdr, x + panel_w - 1, y0 + hdr + ph - 1],
                        outline=(150, 150, 150))
        return y0 + hdr + ph + cap

    y = draw_row(top, 0)
    y = draw_row(bot, y)

    # per-column numbers, aligned under their column
    for j, k in enumerate(order):
        st = row["steps"][k]
        e = (est[k] - x0.reshape(1, -1, 3)).abs().amax(-1)[0] * B.LEVEL
        p50 = float(torch.quantile(e.float(), 0.5))
        d.text(((j + 1) * panel_w + 4, y - cap + 2),
               f"cov={st['coverage']:.4f} errp50={p50:.4g} it={iters[k]}",
               fill=(60, 60, 60), font=fs)
    c = row["calib"]
    d.text((4, y + 2),
           f"{row['id']} pool={row['pool']} major={row['major']} "
           f"rec_band={row['rec_band']} geom={row['mask']['geom']} "
           f"subject={row['mask']['uses_subject']} "
           f"luts={','.join(x[-6:] for x in row['luts'])}",
           fill="black", font=fs)
    d.text((4, y + 16),
           f"s={c['s']:.4f} de00_target={c['de00_target']:.3f} "
           f"de00_after={c['de00_acted_after']:.3f} "
           f"| err |x_hat-before| p50={row['err_E_all']['p50']:.4g} "
           f"p95={row['err_E_all']['p95']:.4g} max={row['err_E_all']['max']:.4g} "
           f"(8-bit levels, per-pixel RGB L-inf) "
           f"| conv_fail_chain={row['conv_fail_frac_chain']:.2e}",
           fill="black", font=fs)
    chain = " -> ".join(f"step{k+1} {B.KIND_LABEL[B.STEP_KIND[k]]}" for k in order)
    d.text((4, y + 30),
           f"columns run along the restoration direction (reverse of [run] "
           f"step_order): {chain}; "
           "alpha shown is the field actually applied (already scaled by s)",
           fill=(90, 90, 90), font=fs)
    sheet_im.save(out_path, quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/bc/data/builds/epr050-degrade-20260825/v3.2")
    ap.add_argument("--montage-dir", default=None,
                    help="default <repo>/docs/assets/epr050_degrade_20260825/recovery")
    ap.add_argument("--config", default=None,
                    help="defaults to the config recorded in <out>/run_args.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--panel-w", type=int, default=300)
    ap.add_argument("--ids", default=None, help="comma-separated subset")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--verify", type=int, default=0,
                    help="assert bit-exact vs restored_e.png for the first N rows; "
                         "-1 = all rows (the recomputation is done anyway)")
    ap.add_argument("--redo", action="store_true", help="redraw existing sheets")
    a = ap.parse_args()

    out = Path(a.out)
    cfg = a.config
    if cfg is None:
        man = json.loads((out / "run_args.json").read_text())
        cfg = man.get("config_path")
        if not cfg:
            raise SystemExit("no --config and run_args.json has no config_path")
    cfg_path = Path(cfg)
    if not cfg_path.is_absolute():
        cfg_path = B.REPO / cfg_path
    cfg_d = B.load_config(cfg_path)
    max_side = B._int(cfg_d, "render", "max_side")

    md = Path(a.montage_dir) if a.montage_dir else \
        B.REPO / "docs/assets/epr050_degrade_20260825/recovery"
    md.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(out / "pairs.jsonl")]
    if a.ids:
        want = set(a.ids.split(","))
        rows = [r for r in rows if r["id"] in want]
    if a.limit:
        rows = rows[: a.limit]

    dev = a.device if torch.cuda.is_available() else "cpu"
    z = np.load(f"{B.BANK}/luts.npz")
    n_verify = len(rows) if a.verify == -1 else a.verify
    verify: list[dict] = []
    made = 0
    for i, r in enumerate(rows):
        p = md / f"{r['id']}.jpg"
        if p.exists() and not a.redo and i >= n_verify:
            continue
        x0, fields, ys, est, iters = rebuild(r, z, dev, max_side)
        if i < n_verify:
            h, w = r["size"]
            ref = np.asarray(Image.open(out / "assets" / f"{r['id']}.restored_e.png")
                             .convert("RGB"))
            got = np.asarray(B.to_png(est[0].reshape(h, w, 3)))
            ref_a = np.asarray(Image.open(out / "assets" / f"{r['id']}.after.png")
                               .convert("RGB"))
            got_a = np.asarray(B.to_png(ys[-1].reshape(h, w, 3)))
            v = dict(id=r["id"],
                     restored_e_bit_exact=bool(np.array_equal(ref, got)),
                     restored_e_max_abs_diff=int(np.abs(
                         ref.astype(np.int32) - got.astype(np.int32)).max()),
                     after_bit_exact=bool(np.array_equal(ref_a, got_a)),
                     after_max_abs_diff=int(np.abs(
                         ref_a.astype(np.int32) - got_a.astype(np.int32)).max()))
            verify.append(v)
            if not (v["restored_e_bit_exact"] and v["after_bit_exact"]):
                B.die(f"{r['id']}: recomputation drifted from the stored assets: {v}")
        if not p.exists() or a.redo:
            sheet(r, x0, fields, est, iters, p, a.panel_w)
            made += 1
        print(f"  [{i+1}/{len(rows)}] {r['id']} -> {p.name}"
              f"{'  verified bit-exact' if i < n_verify else ''}", flush=True)

    if verify:
        vp = out / "recovery_verify.json"
        vp.write_text(json.dumps(dict(
            n=len(verify),
            all_restored_e_bit_exact=all(v["restored_e_bit_exact"] for v in verify),
            all_after_bit_exact=all(v["after_bit_exact"] for v in verify),
            rows=verify), ensure_ascii=False, indent=1))
        print(f"verify: {len(verify)} rows, restored_e bit-exact "
              f"{sum(v['restored_e_bit_exact'] for v in verify)}/{len(verify)}, "
              f"after bit-exact {sum(v['after_bit_exact'] for v in verify)}/{len(verify)} "
              f"-> {vp}", flush=True)

    # minimal browsing page: image tags only, no per-sample fields
    idx = md / "index.html"
    imgs = sorted(x.name for x in md.glob("*.jpg"))
    idx.write_text(
        "<!doctype html>\n<meta charset=utf-8>\n<title>EPR-050 recovery</title>\n"
        "<style>body{margin:0;background:#fff}img{display:block;width:100%;"
        "margin:0 0 12px}</style>\n"
        + "".join(f'<img src="{n}" loading="lazy">\n' for n in imgs))
    print(f"=== {made} sheets written, {len(imgs)} in {md} ===", flush=True)
    print(f"index: {idx}", flush=True)


if __name__ == "__main__":
    main()
