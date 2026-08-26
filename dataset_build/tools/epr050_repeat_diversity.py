#!/usr/bin/env python3
"""EPR-050 same-source repeat test: chain diversity vs recovery identity.

Reads a journal built with --repeat-salt (rows `<source_id>.rep<k>`) plus the
already-written assets, and reports, per source:

  diversity   -- distinct majors / LUT ids / geom kinds / luminance-band draws /
                 calibration strength s over the K repeats;
                 pairwise CIEDE2000 median between the K `after` images
                 (K*(K-1)/2 pairs); per-pixel cross-chain std of `after`.
  identity    -- each chain's restore error against the SAME before image,
                 pairwise max difference between the K restored images, and how
                 many chains pass the [quality] thresholds.

Nothing is recomputed through the LUT chain: every number comes from the journal
or from the PNGs the build already wrote.  The `before` PNG is asserted to be
byte-identical across the repeats of one source, so "the same original" is a
checked fact and not an assumption.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from epr050_quality_filter import flag_row, load_thresholds  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def base_id(rid: str) -> str:
    return rid.split(".rep")[0]


def rep_idx(rid: str) -> int:
    return int(rid.split(".rep")[1]) if ".rep" in rid else -1


def q(v, ks=(0, 50, 100)):
    a = np.asarray(v, dtype=np.float64)
    return {f"p{k}": round(float(np.percentile(a, k)), 6) for k in ks} | {
        "mean": round(float(a.mean()), 6)}


def load_u8(p: Path) -> np.ndarray:
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)


def lab_of(rgb_u8: np.ndarray) -> np.ndarray:
    from skimage.color import rgb2lab
    return rgb2lab(rgb_u8.astype(np.float64) / 255.0)


def de00_med(la: np.ndarray, lb: np.ndarray) -> float:
    from skimage.color import deltaE_ciede2000
    d = deltaE_ciede2000(la.reshape(-1, 3), lb.reshape(-1, 3))
    return float(np.median(d))


def colorize_abs(err: np.ndarray, vmax: float) -> np.ndarray:
    """Absolute [0, vmax] viridis ramp -- the build tool's scale rule, never
    per-image min-max."""
    from epr050_build_degradation import _VIRIDIS
    t = np.clip(err / vmax, 0.0, 1.0) * (len(_VIRIDIS) - 1)
    i = np.floor(t).astype(np.int32).clip(0, len(_VIRIDIS) - 2)
    f = (t - i)[..., None]
    return (_VIRIDIS[i] * (1 - f) + _VIRIDIS[i + 1] * f).astype(np.uint8)


def _clean_only(rows, rests, rest_err, flags, before):
    keep = [i for i, f in enumerate(flags) if f == "clean"]
    if len(keep) < 2:
        return {"n": len(keep), "note": "fewer than 2 clean chains"}
    dif = [np.abs(rests[i].astype(np.int16) - rests[j].astype(np.int16)).max(-1)
           for i, j in itertools.combinations(keep, 2)]
    pr = [int(x.max()) for x in dif]
    return dict(
        n=len(keep), n_pairs=len(pr),
        pairwise_restored_max_diff_levels=q(pr),
        pairwise_restored_p99_diff_levels=q(
            [float(np.percentile(x, 99)) for x in dif]),
        pairwise_restored_p50_diff_levels=q(
            [float(np.percentile(x, 50)) for x in dif]),
        journal_err_E_p99=q([rows[i]["err_E_all"]["p99"] for i in keep]),
        journal_err_E_max=q([rows[i]["err_E_all"]["max"] for i in keep]),
        png_restore_err_max=q([float(rest_err[i].max()) for i in keep]),
        png_restore_err_p99=q([float(np.percentile(rest_err[i], 99)) for i in keep]),
    )


def analyse(sid, rows, assets: Path, p99_max, cf_max, std_vmax):
    rows = sorted(rows, key=lambda r: rep_idx(r["id"]))
    ids = [r["id"] for r in rows]
    K = len(rows)

    # ---- the before image must be the same file for every repeat ------------
    src_sha = {hashlib.sha256((assets / f"{i}.src.png").read_bytes()).hexdigest()
               for i in ids}
    if len(src_sha) != 1:
        raise SystemExit(f"{sid}: before image differs across repeats ({src_sha})")
    before = load_u8(assets / f"{ids[0]}.src.png")

    afters = [load_u8(assets / f"{i}.after.png") for i in ids]
    rests = [load_u8(assets / f"{i}.restored_e.png") for i in ids]

    # ---- diversity ---------------------------------------------------------
    lut_sets = [set(r["luts"]) for r in rows]
    union = set().union(*lut_sets)
    labs = [lab_of(a) for a in afters]
    pairs = list(itertools.combinations(range(K), 2))
    de_pairs = [de00_med(labs[i], labs[j]) for i, j in pairs]

    stack = np.stack([a.astype(np.float32) for a in afters], 0)
    std_map = stack.std(0, ddof=1).max(-1)          # 8-bit levels, max over RGB
    de_vs_before = [de00_med(lab_of(before), labs[i]) for i in range(K)]

    # ---- identity ----------------------------------------------------------
    rest_err = [np.abs(r.astype(np.int16) - before.astype(np.int16)).max(-1)
                for r in rests]
    pair_rest_dif = [np.abs(rests[i].astype(np.int16)
                            - rests[j].astype(np.int16)).max(-1) for i, j in pairs]
    pair_rest_max = [int(x.max()) for x in pair_rest_dif]
    pair_rest_p99 = [float(np.percentile(x, 99)) for x in pair_rest_dif]
    pair_rest_p50 = [float(np.percentile(x, 50)) for x in pair_rest_dif]
    flags = [flag_row(r, p99_max, cf_max)[0] for r in rows]

    per_chain = []
    for r, e, f, d in zip(rows, rest_err, flags, de_vs_before):
        per_chain.append(dict(
            id=r["id"], rep=rep_idx(r["id"]), major=r["major"],
            geom=r["mask"]["geom"], uses_subject=r["mask"]["uses_subject"],
            luts=r["luts"], s=round(r["calib"]["s"], 6),
            de00_target=round(r["calib"]["de00_target"], 4),
            de00_after=round(r["calib"]["de00_acted_after"], 4),
            de00_med_vs_before=round(d, 4),
            q_lo=round(r["mask"]["q_lo"], 4), q_hi=round(r["mask"]["q_hi"], 4),
            width=round(r["mask"]["width"], 4),
            journal_err_E_p50=r["err_E_all"]["p50"],
            journal_err_E_p99=r["err_E_all"]["p99"],
            journal_err_E_max=r["err_E_all"]["max"],
            conv_fail_frac_chain=r["conv_fail_frac_chain"],
            png_restore_err_p50=float(np.percentile(e, 50)),
            png_restore_err_p99=float(np.percentile(e, 99)),
            png_restore_err_max=int(e.max()),
            flag=f,
        ))

    out = dict(
        source_id=sid, n_repeats=K, ids=ids,
        before_sha256=src_sha.pop(), size=list(before.shape[:2]),
        diversity=dict(
            distinct_major=len({r["major"] for r in rows}),
            major_census=dict(Counter(r["major"] for r in rows)),
            geom_census=dict(Counter(r["mask"]["geom"] for r in rows)),
            uses_subject_census=dict(Counter(str(r["mask"]["uses_subject"])
                                             for r in rows)),
            luts_per_chain=len(rows[0]["luts"]),
            distinct_luts_union=len(union),
            distinct_luts_max_possible=K * len(rows[0]["luts"]),
            identical_lut_sets=sum(
                1 for i, j in pairs if lut_sets[i] == lut_sets[j]),
            # set vs ORDER: with a major whose main pool has exactly chain_len
            # LUTs every chain must draw the same set, so the only LUT-side
            # freedom left is the order they are applied in.
            distinct_lut_sets=len({frozenset(s) for s in lut_sets}),
            distinct_lut_orders=len({tuple(r["luts"]) for r in rows}),
            identical_lut_orders=sum(
                1 for i, j in pairs if rows[i]["luts"] == rows[j]["luts"]),
            lut_set_jaccard=q([len(lut_sets[i] & lut_sets[j])
                               / len(lut_sets[i] | lut_sets[j]) for i, j in pairs]),
            s=q([r["calib"]["s"] for r in rows]),
            de00_target=q([r["calib"]["de00_target"] for r in rows]),
            de00_acted_after=q([r["calib"]["de00_acted_after"] for r in rows]),
            q_lo=q([r["mask"]["q_lo"] for r in rows]),
            q_hi=q([r["mask"]["q_hi"] for r in rows]),
            width=q([r["mask"]["width"] for r in rows]),
            n_pairs=len(pairs),
            pairwise_after_de00_med=q(de_pairs),
            after_de00_med_vs_before=q(de_vs_before),
            after_cross_chain_std_levels=dict(
                mean=round(float(std_map.mean()), 4),
                p50=round(float(np.percentile(std_map, 50)), 4),
                p99=round(float(np.percentile(std_map, 99)), 4),
                max=round(float(std_map.max()), 4),
                colour_scale_vmax=std_vmax),
        ),
        identity=dict(
            journal_err_E_p50=q([r["err_E_all"]["p50"] for r in rows]),
            journal_err_E_p99=q([r["err_E_all"]["p99"] for r in rows]),
            journal_err_E_max=q([r["err_E_all"]["max"] for r in rows]),
            conv_fail_frac_chain=q([r["conv_fail_frac_chain"] for r in rows]),
            png_restore_err_p50=q([float(np.percentile(e, 50)) for e in rest_err]),
            png_restore_err_p99=q([float(np.percentile(e, 99)) for e in rest_err]),
            png_restore_err_max=q([float(e.max()) for e in rest_err]),
            pairwise_restored_max_diff_levels=q(pair_rest_max),
            pairwise_restored_p99_diff_levels=q(pair_rest_p99),
            pairwise_restored_p50_diff_levels=q(pair_rest_p50),
            flag_census=dict(Counter(flags)),
            n_pass_quality=sum(f == "clean" for f in flags),
            # same columns restricted to the chains that pass [quality]; the two
            # views are reported side by side, neither replaces the other
            clean_only=_clean_only(rows, rests, rest_err, flags, before),
        ),
        per_chain=per_chain,
    )
    return out, std_map, before, afters, rests, per_chain


def draw_sheet(sid, std_map, before, afters, rests, per_chain, assets: Path,
               out_path: Path, err_vmax: float, std_vmax: float, panel_w=200):
    from epr050_build_degradation import _font
    K = len(afters)
    fh, fb = _font(11), _font(14)
    ids = [c["id"] for c in per_chain]
    errs = [Image.open(assets / f"{i}.err_e.png").convert("RGB") for i in ids]

    def fit(im):
        s = panel_w / im.width
        return im.resize((panel_w, max(1, round(im.height * s))), Image.LANCZOS)

    r1 = [fit(Image.fromarray(before))] + [fit(Image.fromarray(a)) for a in afters]
    r2 = [fit(Image.fromarray(colorize_abs(std_map, std_vmax)))] \
        + [fit(Image.fromarray(r)) for r in rests]
    r3 = [None] + [fit(e) for e in errs]
    rows = [r1, r2, r3]
    labels = ["before + K x after",
              f"cross-chain std of after (0-{std_vmax:.0f} lv) + K x restore E",
              f"K x |restore - before| (0-{err_vmax:.0f} lv)"]

    ph = max(i.height for i in r1)
    hdr, capr, capb = 22, 18, 46
    W = panel_w * (K + 1)
    H = hdr + sum(capr + ph for _ in rows) + capb
    sheet = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(sheet)
    d.text((4, 3), f"{sid}   K = {K} independent chains on one source", "black", fb)

    y = hdr
    for row, lab in zip(rows, labels):
        d.text((4, y + 3), lab, fill="black", font=fh)
        y += capr
        for k, im in enumerate(row):
            if im is None:
                continue
            sheet.paste(im, (k * panel_w, y))
            d.rectangle([k * panel_w, y, k * panel_w + panel_w - 1,
                         y + im.height - 1], outline=(150, 150, 150))
        y += ph
    for k, c in enumerate(per_chain, start=1):
        x = k * panel_w + 3
        d.text((x, y + 2), f"rep{c['rep']} {c['major']}", fill="black", font=fh)
        d.text((x, y + 14), f"{c['geom']} s={c['s']:.3f}", fill="black", font=fh)
        d.text((x, y + 26), f"E p99={c['journal_err_E_p99']:.3g} {c['flag']}",
               fill="black", font=fh)
    d.text((3, y + 2), "before (shared)", fill="black", font=fh)
    d.text((3, y + 14), "std map: absolute", fill="black", font=fh)
    d.text((3, y + 26), "scale, no min-max", fill="black", font=fh)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",
                    default="/home/bc/data/builds/epr050-degrade-20260825/v3.4_diversity")
    ap.add_argument("--config", default=None,
                    help="defaults to the config recorded in <out>/run_args.json")
    ap.add_argument("--sheet-dir",
                    default=str(REPO / "docs/assets/epr050_degrade_20260825/diversity_v3.4"))
    ap.add_argument("--tier", default=None,
                    help="label for this run in the comparison block "
                         "(e.g. 'major free' / 'same major, LUTs may vary')")
    ap.add_argument("--compare-with", default=None,
                    help="comma-separated diversity.json paths of other tiers; "
                         "their per-source numbers are merged into a comparison "
                         "block written next to this run's diversity.json")
    ap.add_argument("--std-vmax", type=float, default=64.0,
                    help="absolute colour-scale ceiling of the cross-chain std map, "
                         "in 8-bit levels")
    a = ap.parse_args()

    out = Path(a.out)
    cfg_path = Path(a.config) if a.config else Path(
        json.loads((out / "run_args.json").read_text())["config_path"])
    p99_max, cf_max, cfg = load_thresholds(cfg_path)
    err_vmax = float(cfg["render"]["err_vmax"])

    rows = [json.loads(l) for l in open(out / "pairs.jsonl")]
    groups: "OrderedDict[str, list]" = OrderedDict()
    for r in rows:
        groups.setdefault(base_id(r["id"]), []).append(r)

    report = dict(
        journal=str(out / "pairs.jsonl"),
        journal_sha256=hashlib.sha256((out / "pairs.jsonl").read_bytes()).hexdigest(),
        config=str(cfg_path),
        config_sha256=hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
        tool=str(Path(__file__).resolve()),
        tool_sha256=hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
        quality_thresholds={"p99_max": p99_max, "conv_fail_max": cf_max},
        err_vmax=err_vmax, std_vmax=a.std_vmax,
        repeat_runs=json.loads((out / "repeat_runs.json").read_text())
        if (out / "repeat_runs.json").exists() else None,
        sources=[],
    )
    for sid, grp in groups.items():
        rep, std_map, before, afters, rests, per_chain = analyse(
            sid, grp, out / "assets", p99_max, cf_max, a.std_vmax)
        p = draw_sheet(sid, std_map, before, afters, rests, per_chain,
                       out / "assets", Path(a.sheet_dir) / f"{sid}.jpg",
                       err_vmax, a.std_vmax)
        rep["sheet"] = str(p)
        Image.fromarray(colorize_abs(std_map, a.std_vmax)).save(
            out / f"{sid}.after_std.png")
        rep["std_map_png"] = str(out / f"{sid}.after_std.png")
        report["sources"].append(rep)
        dv, idt = rep["diversity"], rep["identity"]
        print(f"[{sid}] K={rep['n_repeats']} majors={dv['distinct_major']} "
              f"luts_union={dv['distinct_luts_union']}/{dv['distinct_luts_max_possible']} "
              f"geom={dv['geom_census']}")
        print(f"    pairwise after dE00 med: {dv['pairwise_after_de00_med']}")
        print(f"    after std lv: {dv['after_cross_chain_std_levels']}")
        print(f"    restore E p99 over chains: {idt['journal_err_E_p99']}")
        print(f"    pairwise restored max diff lv: "
              f"{idt['pairwise_restored_max_diff_levels']}")
        print(f"    clean-only ({idt['clean_only']['n']} chains) pairwise restored "
              f"max diff lv: {idt['clean_only'].get('pairwise_restored_max_diff_levels')}")
        print(f"    flags {idt['flag_census']}  sheet {p}")

    report["tier"] = a.tier
    (out / "diversity.json").write_text(json.dumps(report, ensure_ascii=False,
                                                   indent=2))
    print(f"wrote {out / 'diversity.json'}")

    if a.compare_with:
        tiers = [report] + [json.loads(Path(p).read_text())
                            for p in a.compare_with.split(",") if p]
        cmp = {"columns": ["tier", "source_id", "distinct_major",
                           "distinct_lut_sets", "distinct_lut_orders",
                           "pairwise_after_de00_med_p50",
                           "pairwise_after_de00_med_mean",
                           "after_cross_chain_std_mean_levels",
                           "n_clean", "clean_pairwise_restored_p99_diff_max"],
               "rows": []}
        for t in tiers:
            for s in t["sources"]:
                dv, idt = s["diversity"], s["identity"]
                co = idt["clean_only"]
                cmp["rows"].append([
                    t.get("tier"), s["source_id"], dv["distinct_major"],
                    dv.get("distinct_lut_sets"), dv.get("distinct_lut_orders"),
                    dv["pairwise_after_de00_med"]["p50"],
                    dv["pairwise_after_de00_med"]["mean"],
                    dv["after_cross_chain_std_levels"]["mean"],
                    idt["n_pass_quality"],
                    (co.get("pairwise_restored_p99_diff_levels") or {}).get("p100"),
                ])
        p = out / "tier_comparison.json"
        p.write_text(json.dumps(cmp, ensure_ascii=False, indent=2))
        print(f"wrote {p}")
        w = [max(len(str(r[i])) for r in [cmp["columns"]] + cmp["rows"])
             for i in range(len(cmp["columns"]))]
        for r in [cmp["columns"]] + cmp["rows"]:
            print("  " + " | ".join(str(x).ljust(w[i]) for i, x in enumerate(r)))


if __name__ == "__main__":
    main()
