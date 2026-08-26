#!/usr/bin/env python3
"""EPR-050 per-sample quality filter -- derives a clean subset from a finished journal.

Read-only with respect to the run: it never touches pairs.jsonl or assets/, and it
does not import any build-time state beyond the shared drawing helpers.  Kept out of
epr050_build_degradation.py on purpose -- that tool's sha256 is frozen into
run_args.json and editing it would break the identity guard of any resume.

Criterion (thresholds live in the experiment TOML, [quality]):

    reject  <=>  err_E_all.p99 > p99_max  OR  conv_fail_frac_chain > conv_fail_max

Both columns measure *spread-out* recovery error.  err_E_all.max is deliberately
NOT a criterion: it is a single-pixel extremum, and thresholding on it kills rows
whose p99 is ~1e-3 (visually indistinguishable from a perfect recovery).  max is
carried in the output as a reference column only.

Control-pool rows (pool != "main") are flagged "ctrl" and take no part in the
clean/reject decision -- that pool exists to show what an ill-posed degradation
looks like and is reported alongside by design.

Outputs, written next to the journal:
  quality_flags.json  -- per row: flag + the values that triggered it
  clean_index.json    -- the clean id list + retention by rec_band / major / geom

Optionally draws a contact sheet of the rejected rows (before | after | restore E |
err heat map) using the build tool's own absolute 0..err_vmax colour scale.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

FLAGS = ("clean", "reject_p99", "reject_convfail", "reject_both", "ctrl")


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_thresholds(cfg_path: Path) -> tuple[float, float, dict]:
    """Strict: a missing [quality] key raises rather than falling back to a default."""
    try:
        import tomllib
    except ModuleNotFoundError:                     # pragma: no cover
        import tomli as tomllib                     # type: ignore
    cfg = tomllib.loads(cfg_path.read_text())
    if "quality" not in cfg:
        raise SystemExit(f"config has no [quality] section: {cfg_path}")
    q = cfg["quality"]
    for k in ("p99_max", "conv_fail_max"):
        if k not in q:
            raise SystemExit(f"config: [quality] missing key {k!r}")
    return float(q["p99_max"]), float(q["conv_fail_max"]), cfg


def flag_row(r: dict, p99_max: float, cf_max: float) -> tuple[str, float, float, float]:
    p99 = float(r["err_E_all"]["p99"])
    mx = float(r["err_E_all"]["max"])
    cf = float(r["conv_fail_frac_chain"])
    if r["pool"] != "main":
        return "ctrl", p99, mx, cf
    bad_p99, bad_cf = p99 > p99_max, cf > cf_max
    if bad_p99 and bad_cf:
        return "reject_both", p99, mx, cf
    if bad_p99:
        return "reject_p99", p99, mx, cf
    if bad_cf:
        return "reject_convfail", p99, mx, cf
    return "clean", p99, mx, cf


def run_journal(journal: Path, p99_max: float, cf_max: float) -> list[dict]:
    out = []
    for line in open(journal):
        r = json.loads(line)
        flag, p99, mx, cf = flag_row(r, p99_max, cf_max)
        out.append({
            "id": r["id"],
            "pool": r["pool"],
            "rec_band": r["rec_band"],
            "major": r["major"],
            "geom": r["mask"]["geom"],
            "uses_subject": r["mask"]["uses_subject"],
            "flag": flag,
            "err_E_all_p99": p99,
            "err_E_all_max": mx,          # reference column, NOT a criterion
            "conv_fail_frac_chain": cf,
            "trip_p99": bool(r["pool"] == "main" and p99 > p99_max),
            "trip_conv_fail": bool(r["pool"] == "main" and cf > cf_max),
        })
    return out


def strata(recs: list[dict], key) -> "OrderedDict[str, dict]":
    """Retention over main-pool rows only; ctrl rows are excluded from every cell."""
    main = [x for x in recs if x["pool"] == "main"]
    tab: "OrderedDict[str, dict]" = OrderedDict()
    for k in sorted({str(key(x)) for x in main}):
        rows = [x for x in main if str(key(x)) == k]
        clean = sum(x["flag"] == "clean" for x in rows)
        tab[k] = {
            "n": len(rows),
            "clean": clean,
            "rejected": len(rows) - clean,
            "keep_rate": round(clean / len(rows), 6) if rows else None,
            "by_flag": dict(Counter(x["flag"] for x in rows)),
        }
    return tab


def montage_rejects(recs, journal_dir: Path, cfg_path: Path, out_path: Path,
                    panel_w: int = 300):
    from PIL import Image, ImageDraw
    from epr050_build_degradation import _font, colorize, load_config
    import numpy as np

    load_config(cfg_path)                 # sets ERR_VMAX used by colorize()
    import epr050_build_degradation as B
    vmax = B.ERR_VMAX

    COLS = [("src", "before"), ("after", "after"),
            ("restored_e", "restore E"), ("err_e", "err heat map")]
    rows = [x for x in recs if x["flag"].startswith("reject")]
    rows.sort(key=lambda x: -x["err_E_all_p99"])
    if not rows:
        return None, 0

    assets = journal_dir / "assets"
    fh, fb = _font(12), _font(14)
    tiles, hs = [], []
    for r in rows:
        row = []
        for c, _ in COLS:
            p = assets / f"{r['id']}.{c}.png"
            if not p.exists():
                row.append(None)
                continue
            im = Image.open(p).convert("RGB")
            s = panel_w / im.width
            row.append(im.resize((panel_w, max(1, round(im.height * s))), Image.LANCZOS))
        hs.append(max(i.height for i in row if i is not None))
        tiles.append(row)

    hdr, cap = 24, 40
    W, H = panel_w * len(COLS), hdr + sum(h + cap for h in hs)
    sheet = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(sheet)
    for k, (_, t) in enumerate(COLS):
        d.text((k * panel_w + 4, 4), t, fill="black", font=fb)
    bar = colorize(np.linspace(0, vmax, 96)[None].repeat(11, 0), vmax)
    x0 = (len(COLS) - 1) * panel_w + panel_w - 150
    sheet.paste(Image.fromarray(bar), (x0, 6))
    d.text((x0 - 10, 4), "0", fill="black", font=fh)
    d.text((x0 + 100, 4), f"{vmax:.0f}+ lv", fill="black", font=fh)

    y = hdr
    for r, row, h in zip(rows, tiles, hs):
        for k, im in enumerate(row):
            if im is None:
                d.text((k * panel_w + panel_w // 2 - 12, y + h // 2), "n/a",
                       fill="gray", font=fb)
            else:
                sheet.paste(im, (k * panel_w, y))
                d.rectangle([k * panel_w, y, k * panel_w + panel_w - 1,
                             y + im.height - 1], outline=(150, 150, 150))
        d.text((4, y + h + 3),
               f"{r['id']}  pool={r['pool']}  rec_band={r['rec_band']}  "
               f"major={r['major']}  geom={r['geom']}  subject={r['uses_subject']}",
               fill="black", font=fh)
        d.text((4, y + h + 19),
               f"err_E_all p99={r['err_E_all_p99']:.4g} lv  max={r['err_E_all_max']:.4g} lv "
               f"(max is a reference column, not a criterion)  "
               f"conv_fail_frac_chain={r['conv_fail_frac_chain']:.6g}  flag={r['flag']}",
               fill="black", font=fh)
        y += h + cap

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)
    return out_path, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/bc/data/builds/epr050-degrade-20260825/v3.4",
                    help="build directory holding pairs.jsonl (outputs land here)")
    ap.add_argument("--config", default=None,
                    help="defaults to the config recorded in <out>/run_args.json")
    ap.add_argument("--compare", default=None,
                    help="a second build dir to flag with the SAME thresholds, "
                         "reported side by side (nothing is written there)")
    ap.add_argument("--montage", default=None,
                    help="path of the rejected-rows contact sheet (.jpg)")
    ap.add_argument("--tag", default=None, help="label for this journal in stdout")
    a = ap.parse_args()

    out = Path(a.out)
    cfg_path = Path(a.config) if a.config else None
    if cfg_path is None:
        man = json.loads((out / "run_args.json").read_text())
        cfg_path = Path(man["config_path"])
    p99_max, cf_max, _ = load_thresholds(cfg_path)
    tag = a.tag or out.name

    recs = run_journal(out / "pairs.jsonl", p99_max, cf_max)
    dist = dict(Counter(x["flag"] for x in recs))
    ident = {
        "journal": str(out / "pairs.jsonl"),
        "journal_sha256": sha256_file(out / "pairs.jsonl"),
        "config": str(cfg_path),
        "config_sha256": sha256_file(cfg_path),
        "filter_tool": str(Path(__file__).resolve()),
        "filter_tool_sha256": sha256_file(Path(__file__).resolve()),
        "thresholds": {"p99_max": p99_max, "conv_fail_max": cf_max},
        "criterion": ("reject iff err_E_all.p99 > p99_max OR "
                      "conv_fail_frac_chain > conv_fail_max; err_E_all.max is a "
                      "reference column and is NOT a criterion; pool != 'main' "
                      "is flagged 'ctrl' and excluded from the decision"),
    }

    (out / "quality_flags.json").write_text(json.dumps({
        **ident,
        "n_rows": len(recs),
        "flag_counts": {k: dist.get(k, 0) for k in FLAGS},
        "rows": recs,
    }, ensure_ascii=False, indent=2))

    clean_ids = [x["id"] for x in recs if x["flag"] == "clean"]
    main_n = sum(x["pool"] == "main" for x in recs)
    (out / "clean_index.json").write_text(json.dumps({
        **ident,
        "n_rows": len(recs),
        "n_main": main_n,
        "n_ctrl": len(recs) - main_n,
        "n_clean": len(clean_ids),
        "keep_rate_main": round(len(clean_ids) / main_n, 6) if main_n else None,
        "flag_counts": {k: dist.get(k, 0) for k in FLAGS},
        "by_rec_band": strata(recs, lambda x: x["rec_band"]),
        "by_major": strata(recs, lambda x: x["major"]),
        "by_geom": strata(recs, lambda x: x["geom"]),
        "by_uses_subject": strata(recs, lambda x: x["uses_subject"]),
        "rejected_ids": [x["id"] for x in recs if x["flag"].startswith("reject")],
        "clean_ids": clean_ids,
    }, ensure_ascii=False, indent=2))

    print(f"[{tag}] thresholds p99_max={p99_max} conv_fail_max={cf_max}")
    print(f"[{tag}] flags {json.dumps({k: dist.get(k, 0) for k in FLAGS})}"
          f"  clean/main = {len(clean_ids)}/{main_n}")
    for name, key in (("rec_band", lambda x: x["rec_band"]),
                      ("geom", lambda x: x["geom"]),
                      ("major", lambda x: x["major"])):
        print(f"[{tag}] by {name}:")
        for k, v in strata(recs, key).items():
            print(f"    {k:<16} n={v['n']:<4} clean={v['clean']:<4} "
                  f"keep={v['keep_rate']}")
    print(f"[{tag}] wrote {out / 'quality_flags.json'}")
    print(f"[{tag}] wrote {out / 'clean_index.json'}")

    if a.compare:
        c = Path(a.compare)
        crecs = run_journal(c / "pairs.jsonl", p99_max, cf_max)
        cdist = Counter(x["flag"] for x in crecs)
        cmain = sum(x["pool"] == "main" for x in crecs)
        cclean = cdist.get("clean", 0)
        print(f"[compare {c.name}] flags "
              f"{json.dumps({k: cdist.get(k, 0) for k in FLAGS})}"
              f"  clean/main = {cclean}/{cmain}  (nothing written)")
        for k, v in strata(crecs, lambda x: x["rec_band"]).items():
            print(f"    {k:<16} n={v['n']:<4} clean={v['clean']:<4} "
                  f"keep={v['keep_rate']}")

    if a.montage:
        p, n = montage_rejects(recs, out, cfg_path, Path(a.montage))
        print(f"[{tag}] montage: {p}  rows={n}")


if __name__ == "__main__":
    main()
