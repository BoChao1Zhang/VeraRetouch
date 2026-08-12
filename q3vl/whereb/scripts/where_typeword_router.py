"""Zero-training mask-type router from the ``<where>`` text, and its accuracy.

The P1/P3 routing plan starts with a **rule over type words** rather than a
learned classifier.  Whether that is viable is an empirical question with a very
specific shape: DELTA section 7 downgraded the language-side evidence because
``<where>`` is verbatim-correct on 0/30 local samples -- but *verbatim* accuracy
is the wrong bar for routing.  Routing only needs the **family word** to be
right, which is a far coarser target.  This card measures that directly.

Ground truth family comes from ``slot_id`` in the build's ``.vrmeta.json``
(radial / band / linear / semantic) -- the only place the label survives.

The rule is not tuned to the predictions: the keyword table is derived from the
**GT** ``<where>`` text by in-family-vs-out-of-family lift, then frozen and
applied unchanged to the generated text.  Tuning it on predictions is exactly
the "enumerate to a positive result" move the probe discipline forbids.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

#: Frozen keyword table, derived from GT scope text by lift (see module docstring).
#: Each family scores +1 per distinct hit; ties break by the FAMILY_ORDER below.
KEYWORDS: dict[str, tuple[str, ...]] = {
    "radial":   ("oval", "falloff", "centered", "centred", "corners", "unchanged",
                 "surrounding", "elliptical", "radial", "vignette"),
    "linear":   ("toward", "spanning", "gradient", "opposite", "fading",
                 "strongest", "horizontal", "vertical", "diagonal"),
    "band":     ("band", "both", "edges", "running", "strip", "beyond",
                 "extending", "through"),
    "semantic": ("within", "stays", "itself", "confined", "only the subject",
                 "subject only"),
}
#: applied on ties only; semantic first because its two markers are the most
#: specific (in-family 0.91, out-of-family 0.00) and it is the smallest class
FAMILY_ORDER = ("semantic", "radial", "linear", "band")


def scope_of(where_text: str) -> str:
    t = (where_text or "").lower()
    return t.split("edit scope:")[-1] if "edit scope:" in t else t


def route(where_text: str) -> tuple[str, dict[str, int]]:
    """``(family, per-family hit counts)``.  ``"unknown"`` when nothing matches."""
    scope = scope_of(where_text)
    words = set(re.findall(r"[a-z]+", scope))
    score = {}
    for fam, keys in KEYWORDS.items():
        n = 0
        for k in keys:
            n += int(k in words) if " " not in k else int(k in scope)
        score[fam] = n
    best = max(score.values())
    if best == 0:
        return "unknown", score
    winners = [f for f in FAMILY_ORDER if score.get(f, 0) == best]
    return winners[0], score


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="V_where")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args(argv)

    from q3vl.where.maskdata import MaskResolver
    from q3vl.whereb.config import GENCTX_DIR
    from q3vl.whereb.data import open_dataset
    from q3vl.whereb.stores import GenContextStore

    ds, _ = open_dataset(args.split, need_mask=False)
    rows = ds.meta_rows()
    local = [i for i, r in enumerate(rows) if r.get("render_mode") == "local"]
    vres = MaskResolver(verify="none", suffix=".vrmeta.json")
    gs = GenContextStore(Path(GENCTX_DIR) / args.split)

    def one(i):
        rec = ds.record(i)
        sid = rec["sample_id"]
        try:
            slot = json.loads(vres.read_bytes(vres.resolve(rec)).decode()).get("slot_id", "")
        except Exception:
            return None
        gen = ""
        try:
            g = gs.record(sid)
            gen = g.get("where_text") or g.get("text") or ""
        except Exception:
            gen = ""
        return sid, slot.rsplit("-", 1)[0], rec.get("where", ""), gen

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        data = [r for r in ex.map(one, local) if r]
    print(f"{args.split}: {len(data)} local samples", flush=True)
    n_gen = sum(1 for _, _, _, g in data if g.strip())
    print(f"  generated <where> text available for {n_gen}", flush=True)

    fams = list(KEYWORDS) + ["unknown"]
    out: dict[str, object] = {"split": args.split, "n": len(data),
                              "n_with_generated_text": n_gen,
                              "keywords": {k: list(v) for k, v in KEYWORDS.items()}}

    for source in ("gt", "generated"):
        cm = collections.Counter()
        n_used = 0
        for sid, fam, gt_txt, gen_txt in data:
            txt = gt_txt if source == "gt" else gen_txt
            if source == "generated" and not txt.strip():
                continue
            n_used += 1
            cm[(fam, route(txt)[0])] += 1
        truths = sorted({t for t, _ in cm})
        acc = sum(cm[(t, t)] for t in truths) / n_used if n_used else 0.0
        per_class = {}
        for t in truths:
            n_t = sum(v for (tt, _), v in cm.items() if tt == t)
            n_p = sum(v for (_, pp), v in cm.items() if pp == t)
            tp = cm[(t, t)]
            per_class[t] = {
                "n_true": n_t, "recall": tp / n_t if n_t else None,
                "precision": tp / n_p if n_p else None,
            }
        out[f"{source}_accuracy"] = acc
        out[f"{source}_n"] = n_used
        out[f"{source}_per_class"] = per_class
        out[f"{source}_confusion"] = {f"{t}->{p}": v for (t, p), v in sorted(cm.items())}

        print(f"\n=== source = {source} <where> text  (n={n_used}) ===", flush=True)
        print(f"  overall family accuracy: {acc:.4f}", flush=True)
        hdr = "  true\\pred    " + "".join(f"{p:>10}" for p in fams)
        print(hdr, flush=True)
        for t in truths:
            print(f"  {t:<12}" + "".join(f"{cm[(t, p)]:>10}" for p in fams), flush=True)
        for t, v in per_class.items():
            print(f"    {t:<10} recall {v['recall']:.3f}  precision "
                  f"{v['precision'] if v['precision'] is None else round(v['precision'],3)}",
                  flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
