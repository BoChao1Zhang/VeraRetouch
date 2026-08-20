"""DATA-MASKCUT-02 (was DATA-BANDCUT-01): quantify a set of
``family == F && mean(GT alpha) < T_F`` rules, then emit a filtered copy of the
split indices.

Two stages, both read-only against the source dataset:

* **quantify** -- for every ``task_type == "local"`` row of each split, fetch the
  mask family from the construction-side ``.vrmeta.json`` via
  :func:`q3vl.whereb.amort.data.family_labels` (``q3vl/whereb/amort/data.py:172``),
  then for the rows whose family carries a rule read the GT alpha through
  :class:`q3vl.whatb.evaldata.SampleStore` (``q3vl/whatb/evaldata.py:56``) and
  take its mean **at the alpha's own resolution** (no resize).  Every per-sample
  number is written to ``mask_alpha_mean.jsonl``.
* **filter** -- copy each ``<split>.index.jsonl`` (and, when present, each
  ``<split>_sft_ids.txt``) line by line, dropping only the excluded
  ``sample_id`` s.  Kept lines are written back as the **original bytes** (no
  json re-serialisation) and the byte identity is asserted after the write.

Rules are given as ``--rule FAMILY=THRESHOLD`` (repeatable, comma-separable).
The single-rule spelling ``--family F --threshold T`` is kept as an exact
shorthand for ``--rule F=T`` so the DATA-BANDCUT-01 command line still
reproduces its per-sample numbers, its exclusion list and its index bytes.

``style`` rows carry no mask member (``q3vl/whatb/evaldata.py:96``: ``alpha == 1``
everywhere by dataset convention) and are therefore outside this filter entirely.
A family with no rule (e.g. ``linear``) is counted but never has its GT alpha
read and never contributes an exclusion.

Nothing here writes to ``/mnt/nfs`` or to the source dataset root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

DEFAULT_SPLITS = ("train", "V_what", "V_where", "T_final", "T_lut_unseen")
DEFAULT_OUT = "/home/bc/data/datasets/sft2seg-20260804-maskcut"
QUANTILES = (("min", 0.0), ("p05", 0.05), ("p25", 0.25), ("p50", 0.50),
             ("p75", 0.75), ("p95", 0.95), ("max", 1.0))
#: the label range of q3vl/whereb/scripts/mask_type_stats.py:38 family_of()
KNOWN_FAMILIES = ("band", "radial", "semantic", "linear", "unknown")


# --------------------------------------------------------------------------- #
# a `ds` shim for family_labels: it only ever calls `ds.record(i)`
# --------------------------------------------------------------------------- #
class RecordShim:
    """``.record(i)`` over a list of :class:`q3vl.whatb.splits.IndexRow`.

    ``family_labels`` reads records from 32 threads; the shard handles are
    therefore thread-local (one open fd per shard per thread, reused) rather
    than a per-call open/seek/close on the soft mount.
    """

    def __init__(self, rows: Sequence[Any]) -> None:
        self.rows = list(rows)
        self._local = threading.local()

    def _handles(self) -> dict[str, Any]:
        h = getattr(self._local, "handles", None)
        if h is None:
            h = self._local.handles = {}
        return h

    def record(self, i: int) -> dict[str, Any]:
        from q3vl.whatb.splits import ro_path

        member = self.rows[i].raw["members"]["record"]
        key = str(member["shard"])
        handles = self._handles()
        fh = handles.get(key)
        if fh is None:
            fh = handles[key] = ro_path(key).open("rb")
        fh.seek(int(member["offset"]))
        blob = fh.read(int(member["length"]))
        if len(blob) != int(member["length"]):
            raise IOError(f"short read of {key}:{member['offset']}")
        return json.loads(blob)

    def close(self) -> None:
        for h in (getattr(self._local, "handles", None) or {}).values():
            h.close()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_rules(specs: Sequence[str] | None, legacy_family: str | None,
                legacy_threshold: float | None) -> dict[str, float]:
    """``--rule FAMILY=THRESHOLD`` (repeatable, comma-separable) -> mapping.

    ``--family`` / ``--threshold`` are the single-rule shorthand and may not be
    combined with ``--rule``: two spellings of the same thing in one command
    line is exactly how a rule set silently loses a member.
    """
    legacy = legacy_family is not None or legacy_threshold is not None
    if specs and legacy:
        raise SystemExit(
            "--rule and --family/--threshold are two spellings of the same "
            "thing; give one or the other, not both")
    if not specs:
        fam = "band" if legacy_family is None else legacy_family
        thr = 0.60 if legacy_threshold is None else legacy_threshold
        specs = [f"{fam}={thr}"]
    rules: dict[str, float] = {}
    for spec in specs:
        for part in str(spec).split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise SystemExit(f"--rule takes FAMILY=THRESHOLD, got {part!r}")
            name, _, val = part.partition("=")
            name = name.strip()
            if name not in KNOWN_FAMILIES:
                raise SystemExit(
                    f"--rule: unknown family {name!r}; the label range of "
                    f"family_of() is {KNOWN_FAMILIES}")
            try:
                thr = float(val.strip())
            except ValueError:
                raise SystemExit(
                    f"--rule {name}: {val.strip()!r} is not a float")
            if not 0.0 <= thr <= 1.0:
                raise SystemExit(
                    f"--rule {name}: threshold {thr} is outside [0, 1]")
            if name in rules and rules[name] != thr:
                raise SystemExit(
                    f"--rule {name} given twice with different thresholds "
                    f"({rules[name]} and {thr})")
            rules[name] = thr
    if not rules:
        raise SystemExit("--rule: no rule parsed")
    return rules


def quantiles(values: Sequence[float]) -> dict[str, float]:
    import numpy as np

    if not values:
        return {name: None for name, _ in QUANTILES}
    a = np.asarray(sorted(values), dtype=np.float64)
    return {name: float(np.quantile(a, q)) for name, q in QUANTILES}


# --------------------------------------------------------------------------- #
# stage A
# --------------------------------------------------------------------------- #
def split_report(*, split: str, rules: dict[str, float],
                 ok: Sequence[dict[str, Any]], bad: Sequence[dict[str, Any]],
                 below: Sequence[dict[str, Any]], below_ids: set[str],
                 fam_counts: Counter, fam_counts_normal: Counter,
                 conf_all: Counter, conf_local: Counter, conf_below: Counter,
                 n_total: int, n_local: int, t0: float) -> dict[str, Any]:
    """The per-split stage-A report.  One body, both the fresh-read path and
    the ``--alpha-jsonl`` reuse path, so the two cannot drift."""
    per_family: dict[str, Any] = {}
    for fam in sorted(set(fam_counts) | set(rules)):
        fam_ok = [r for r in ok if r["family"] == fam]
        fam_bad = [r for r in bad if r.get("family") == fam]
        fam_below = [r for r in fam_ok
                     if r["alpha_mean"] < rules.get(fam, -1.0)]
        n_fam = fam_counts.get(fam, 0)
        per_family[fam] = {
            "n_local_rows": n_fam,
            "n_local_rows_normal": fam_counts_normal.get(fam, 0),
            "threshold": rules.get(fam),
            "n_alpha_read": len(fam_ok),
            "n_alpha_failed": len(fam_bad),
            "n_excluded": len(fam_below),
            "n_excluded_normal": sum(
                1 for r in fam_below if r["winner_confidence"] == "normal"),
            "frac_of_family": (len(fam_below) / n_fam) if n_fam else None,
            "alpha_mean_quantiles": quantiles(
                [r["alpha_mean"] for r in fam_ok]),
            "threshold_percentile_within_family": (
                float(100.0 * len(fam_below) / len(fam_ok))
                if fam_ok else None),
            "alpha_resolutions": dict(sorted(Counter(
                f"{r['alpha_h']}x{r['alpha_w']}" for r in fam_ok).items(),
                key=lambda kv: -kv[1])),
        }

    n_below = len(below)
    n_normal = conf_all.get("normal", 0)
    n_below_normal = conf_below.get("normal", 0)
    return {
        "n_total": n_total,
        "n_local": n_local,
        "n_style": n_total - n_local,
        "n_alpha_read": len(ok),
        "n_alpha_failed": len(bad),
        "n_excluded": n_below,
        "n_excluded_normal": n_below_normal,
        "n_after": n_total - n_below,
        "n_normal": n_normal,
        "n_after_normal": n_normal - n_below_normal,
        "frac_of_split": n_below / n_total if n_total else None,
        "frac_of_local": n_below / n_local if n_local else None,
        "family_counts": dict(sorted(fam_counts.items())),
        "family_counts_normal": dict(sorted(fam_counts_normal.items())),
        "per_family": per_family,
        "alpha_mean_quantiles": quantiles([r["alpha_mean"] for r in ok]),
        "winner_confidence": {
            "split": dict(sorted(conf_all.items())),
            "local": dict(sorted(conf_local.items())),
            "alpha_read": dict(sorted(Counter(
                r["winner_confidence"] for r in ok).items())),
            "excluded": dict(sorted(conf_below.items())),
        },
        "n_excluded_dedup_check": len(below_ids),
        "seconds": round(time.time() - t0, 1),
    }


def load_reuse(alpha_jsonl: Path, stats_ref: Path, splits: Sequence[str],
               rules: dict[str, float]) -> tuple[dict[str, list], dict[str, Any]]:
    """Per-sample alphas + threshold-independent counts of an earlier stage A.

    ``alpha_mean`` is a property of the GT alpha alone -- it does not depend on
    any threshold -- so a rule set over the *same* families can be evaluated
    against an existing ``mask_alpha_mean.jsonl`` without opening a mask again.
    The counts that stage A takes from the index rather than from the mask
    (``family_counts``, ``winner_confidence``, ``n_total`` / ``n_local``) are
    threshold-independent in the same way and are read back from the reference
    ``stage_a_stats.json``.

    Guards, because a silently mismatched reuse would be indistinguishable from
    a fresh read: the reference must cover every requested split, its family set
    must be exactly the rule set, and every row of a ruled family must have had
    its alpha read (``n_alpha_read == family_counts[fam]``).
    """
    ref = json.loads(Path(stats_ref).read_text("utf-8"))
    by_split: dict[str, list[dict[str, Any]]] = {}
    fams_seen: set[str] = set()
    with Path(alpha_jsonl).open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            fams_seen.add(r["family"])
            by_split.setdefault(r["split"], []).append(r)
    if fams_seen != set(rules):
        raise AssertionError(
            f"--alpha-jsonl covers families {sorted(fams_seen)} but the rule "
            f"set is {sorted(rules)}; a family with a rule but no per-sample "
            f"alpha would be silently un-filtered")
    for split in splits:
        if split not in ref.get("per_split", {}):
            raise AssertionError(f"--stats-ref has no per_split entry for {split!r}")
        if split not in by_split:
            raise AssertionError(f"--alpha-jsonl has no rows for split {split!r}")
        pf = ref["per_split"][split]["per_family"]
        for fam in rules:
            if pf.get(fam, {}).get("n_alpha_read") != pf.get(fam, {}).get("n_local_rows"):
                raise AssertionError(
                    f"--stats-ref {split}/{fam}: n_alpha_read "
                    f"{pf.get(fam, {}).get('n_alpha_read')} != n_local_rows "
                    f"{pf.get(fam, {}).get('n_local_rows')}; the reused file does "
                    f"not cover the whole family")
            got = sum(1 for r in by_split[split] if r["family"] == fam)
            if got != pf[fam]["n_alpha_read"]:
                raise AssertionError(
                    f"--alpha-jsonl {split}/{fam}: {got} row(s) but --stats-ref "
                    f"says {pf[fam]['n_alpha_read']}")
    return by_split, ref


def quantify(splits: Sequence[str], root: Path, mask_root: str, workers: int,
             rules: dict[str, float], out_dir: Path,
             alpha_jsonl: Path | None = None,
             stats_ref: Path | None = None) -> dict[str, Any]:
    import numpy as np

    reuse: dict[str, list[dict[str, Any]]] | None = None
    ref: dict[str, Any] | None = None
    if alpha_jsonl is not None:
        reuse, ref = load_reuse(alpha_jsonl, stats_ref, splits, rules)
    else:
        from q3vl.whatb.evaldata import SampleStore
        from q3vl.whatb.splits import load_index
        from q3vl.whereb.amort.data import family_labels

    per_split: dict[str, Any] = {}
    alpha_rows: list[dict[str, Any]] = []
    alpha_failures: list[dict[str, str]] = []

    for split in splits:
        t0 = time.time()
        if reuse is not None:
            rs = ref["per_split"][split]
            n_total, n_local = int(rs["n_total"]), int(rs["n_local"])
            fam_counts = Counter(rs["family_counts"])
            fam_counts_normal = Counter(rs["family_counts_normal"])
            conf_all = Counter(rs["winner_confidence"]["split"])
            conf_local = Counter(rs["winner_confidence"]["local"])
            ok = [r for r in reuse[split] if r["family"] in rules]
            bad: list[dict[str, Any]] = []
            print(f"{split}: {n_total} rows, {n_local} local (reused)", flush=True)
            print(f"  families: {dict(sorted(fam_counts.items()))} "
                  f"(reused, 0 mask reads)", flush=True)
            alpha_rows.extend(ok)
            below = [r for r in ok if r["alpha_mean"] < rules[r["family"]]]
            below_ids = {r["sample_id"] for r in below}
            conf_below = Counter(r["winner_confidence"] for r in below)
            per_split[split] = split_report(
                split=split, rules=rules, ok=ok, bad=bad, below=below,
                below_ids=below_ids, fam_counts=fam_counts,
                fam_counts_normal=fam_counts_normal, conf_all=conf_all,
                conf_local=conf_local, conf_below=conf_below,
                n_total=n_total, n_local=n_local, t0=t0)
            print(f"  excluded {len(below)} of {len(ok)} alpha-read rows "
                  f"({ {f: per_split[split]['per_family'][f]['n_excluded'] for f in sorted(rules)} })",
                  flush=True)
            continue

        rows = load_index(split, root)
        local_idx = [i for i, r in enumerate(rows) if r.task_type == "local"]
        print(f"{split}: {len(rows)} rows, {len(local_idx)} local", flush=True)

        shim = RecordShim(rows)
        try:
            fams = family_labels(shim, local_idx, workers=workers)
        finally:
            shim.close()
        fam_counts = Counter(fams.values())
        fam_counts_normal = Counter(
            fams[rows[i].sample_id] for i in local_idx
            if rows[i].winner_confidence == "normal")
        print(f"  families: {dict(sorted(fam_counts.items()))} "
              f"({time.time()-t0:.0f}s)", flush=True)

        # one pass in index order, so the single-rule spelling emits exactly the
        # rows (and the order) DATA-BANDCUT-01 emitted
        target = [rows[i] for i in local_idx
                  if fams.get(rows[i].sample_id) in rules]
        store = SampleStore(split, root=root, mask_root=mask_root)
        store._load_mask_index()          # pre-warm: the lazy build is not thread-safe

        def one(row) -> dict[str, Any]:
            fam = fams[row.sample_id]
            try:
                a = store.alpha(row.sample_id, row.task_type)
                arr = a.numpy() if hasattr(a, "numpy") else np.asarray(a)
                arr = np.asarray(arr, dtype=np.float64).reshape(
                    arr.shape[-2], arr.shape[-1])
                return {"sample_id": row.sample_id, "split": split,
                        "task_type": row.task_type, "family": fam,
                        "alpha_mean": float(arr.mean()),
                        "winner_confidence": row.winner_confidence,
                        "alpha_h": int(arr.shape[0]), "alpha_w": int(arr.shape[1])}
            except Exception as exc:                       # noqa: BLE001
                return {"sample_id": row.sample_id, "split": split,
                        "family": fam,
                        "error": f"{type(exc).__name__}: {exc}"}

        got: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for n, r in enumerate(ex.map(one, target)):
                got.append(r)
                if (n + 1) % 5000 == 0:
                    print(f"  alpha [{n+1}/{len(target)}] {time.time()-t0:.0f}s",
                          flush=True)

        ok = [r for r in got if "error" not in r]
        bad = [r for r in got if "error" in r]
        alpha_rows.extend(ok)
        alpha_failures.extend(bad)

        below = [r for r in ok if r["alpha_mean"] < rules[r["family"]]]
        below_ids = {r["sample_id"] for r in below}
        conf_all = Counter(r.winner_confidence for r in rows)
        conf_local = Counter(rows[i].winner_confidence for i in local_idx)
        conf_below = Counter(r["winner_confidence"] for r in below)

        per_split[split] = split_report(
            split=split, rules=rules, ok=ok, bad=bad, below=below,
            below_ids=below_ids, fam_counts=fam_counts,
            fam_counts_normal=fam_counts_normal, conf_all=conf_all,
            conf_local=conf_local, conf_below=conf_below,
            n_total=len(rows), n_local=len(local_idx), t0=t0)
        print(f"  excluded {len(below)} of {len(ok)} alpha-read rows "
              f"({ {f: per_split[split]['per_family'][f]['n_excluded'] for f in sorted(rules)} }) "
              f"({time.time()-t0:.0f}s)", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    jl = out_dir / "mask_alpha_mean.jsonl"
    with jl.open("w", encoding="utf-8") as fh:
        for r in alpha_rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")

    excluded = sorted(r["sample_id"] for r in alpha_rows
                      if r["alpha_mean"] < rules[r["family"]])
    if len(set(excluded)) != len(excluded):
        raise AssertionError("a sample_id was excluded twice; the family label "
                             "is supposed to be single-valued per sample")
    (out_dir / "excluded_sample_ids.txt").write_text(
        "".join(s + "\n" for s in excluded), encoding="utf-8")

    pooled_family: dict[str, Any] = {}
    for fam in sorted(rules):
        fam_ok = [r for r in alpha_rows if r["family"] == fam]
        fam_below = [r for r in fam_ok if r["alpha_mean"] < rules[fam]]
        pooled_family[fam] = {
            "threshold": rules[fam],
            "n_alpha_read": len(fam_ok),
            "n_excluded": len(fam_below),
            "alpha_mean_quantiles": quantiles([r["alpha_mean"] for r in fam_ok]),
            "threshold_percentile_within_family": (
                float(100.0 * len(fam_below) / len(fam_ok)) if fam_ok else None),
        }
    all_means = [r["alpha_mean"] for r in alpha_rows]
    return {
        "rules": dict(sorted(rules.items())),
        "per_split": per_split,
        "pooled": {
            "n_alpha_read": len(alpha_rows),
            "n_excluded": len(excluded),
            "per_family": pooled_family,
            "alpha_mean_quantiles": quantiles(all_means),
        },
        "alpha_failures": alpha_failures,
        "n_excluded": len(excluded),
    }


# --------------------------------------------------------------------------- #
# stage B
# --------------------------------------------------------------------------- #
def filter_indices(splits: Sequence[str], root: Path, out_dir: Path,
                   excluded: set[str]) -> dict[str, Any]:
    dst_dir = out_dir / "splits"
    dst_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {}

    def copy_filtered(src: Path, dst: Path, sid_of) -> dict[str, Any]:
        kept: list[bytes] = []
        n_in = n_drop = 0
        with src.open("rb") as fh, dst.open("wb") as wh:
            for raw in fh:
                if not raw.strip():
                    continue
                n_in += 1
                if sid_of(raw) in excluded:
                    n_drop += 1
                    continue
                kept.append(raw)
                wh.write(raw)

        # byte identity: the file we just wrote must be exactly the kept source
        # lines, unchanged and in order.
        with dst.open("rb") as fh:
            back = [ln for ln in fh]
        identical = (len(back) == len(kept)
                     and all(a == b for a, b in zip(back, kept)))
        if not identical:
            raise AssertionError(f"{dst}: kept lines are not byte-identical to source")
        return {
            "src": str(src), "src_sha256": sha256_file(src),
            "dst": str(dst), "dst_sha256": sha256_file(dst),
            "n_before": n_in, "n_after": n_in - n_drop, "n_dropped": n_drop,
            "kept_lines_byte_identical": True,
        }

    for split in splits:
        src = root / "splits" / f"{split}.index.jsonl"
        dst = dst_dir / f"{split}.index.jsonl"
        info = copy_filtered(src, dst, lambda raw: str(json.loads(raw)["sample_id"]))
        print(f"{split}: {info['n_before']} -> {info['n_after']} "
              f"(-{info['n_dropped']}), byte-identical OK", flush=True)

        # `<split>_sft_ids.txt` -- one sample_id per line, same drop list
        ids_src = root / "splits" / f"{split}_sft_ids.txt"
        if ids_src.is_file():
            ids_dst = dst_dir / f"{split}_sft_ids.txt"
            info["sft_ids"] = copy_filtered(
                ids_src, ids_dst, lambda raw: raw.decode("utf-8").strip())
            print(f"{split}_sft_ids.txt: {info['sft_ids']['n_before']} -> "
                  f"{info['sft_ids']['n_after']} "
                  f"(-{info['sft_ids']['n_dropped']}), byte-identical OK",
                  flush=True)
        else:
            info["sft_ids"] = None
        out[split] = info
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", nargs="*", default=list(DEFAULT_SPLITS))
    ap.add_argument("--src-root",
                    default="/mnt/nfs-ro/bc/data/datasets/sft2seg-20260804")
    ap.add_argument("--mask-root",
                    default="/mnt/nfs-ro/bc/data/datasets/where_a-20260805/maskviews")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--rule", action="append", default=None,
                    help="FAMILY=THRESHOLD; repeatable and comma-separable. "
                         "Drop every local row whose mask family is FAMILY and "
                         "whose GT alpha mean is < THRESHOLD.  A family with no "
                         "rule is counted but never read and never dropped")
    ap.add_argument("--family", default=None,
                    help="DATA-BANDCUT-01 shorthand for a single --rule; "
                         "--family F --threshold T == --rule F=T.  Cannot be "
                         "combined with --rule")
    ap.add_argument("--threshold", type=float, default=None,
                    help="see --family (default 0.60 when --family is used)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--alpha-jsonl", default=None,
                    help="reuse the per-sample GT alpha means of an earlier "
                         "stage A (`mask_alpha_mean.jsonl`) instead of opening "
                         "a single mask again.  `alpha_mean` does not depend on "
                         "any threshold, so a new rule set over the SAME "
                         "families is a pure re-thresholding of this file.  "
                         "Requires --stats-ref for the counts stage A takes "
                         "from the index (family_counts, winner_confidence, "
                         "n_total / n_local), which are threshold-independent "
                         "in the same way")
    ap.add_argument("--stats-ref", default=None,
                    help="the `stage_a_stats.json` written next to "
                         "--alpha-jsonl; required with it, rejected without it")
    ap.add_argument("--stamp", required=True,
                    help="generation timestamp, passed in from outside")
    ap.add_argument("--stage", choices=["all", "quantify", "filter"], default="all")
    args = ap.parse_args(argv)

    t0 = time.time()
    rules = parse_rules(args.rule, args.family, args.threshold)
    root = Path(args.src_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"rules: {dict(sorted(rules.items()))}", flush=True)

    if (args.alpha_jsonl is None) != (args.stats_ref is None):
        raise SystemExit("--alpha-jsonl and --stats-ref go together: the "
                         "per-sample alphas alone do not carry the index-side "
                         "counts of stage A")
    if args.alpha_jsonl is not None and Path(args.alpha_jsonl).resolve() == \
            (out_dir / "mask_alpha_mean.jsonl").resolve():
        raise SystemExit("--alpha-jsonl is the file this run would overwrite")

    stats: dict[str, Any] | None = None
    if args.stage in ("all", "quantify"):
        stats = quantify(args.splits, root, args.mask_root, args.workers,
                         rules, out_dir,
                         alpha_jsonl=(Path(args.alpha_jsonl)
                                      if args.alpha_jsonl else None),
                         stats_ref=(Path(args.stats_ref)
                                    if args.stats_ref else None))
        if args.alpha_jsonl is not None:
            # the reuse is a re-thresholding, not a re-measurement: the
            # per-sample file this run emits must be the one it read.
            a = sha256_file(Path(args.alpha_jsonl))
            b = sha256_file(out_dir / "mask_alpha_mean.jsonl")
            if a != b:
                raise AssertionError(
                    f"reused {args.alpha_jsonl} ({a}) but re-emitted a "
                    f"different mask_alpha_mean.jsonl ({b})")
            print(f"reuse: mask_alpha_mean.jsonl sha256 {a} (unchanged)",
                  flush=True)
        (out_dir / "stage_a_stats.json").write_text(
            json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.stage == "quantify":
        print(f"done in {time.time()-t0:.0f}s")
        return 0

    if stats is None:
        stats = json.loads((out_dir / "stage_a_stats.json").read_text("utf-8"))
    excluded = set((out_dir / "excluded_sample_ids.txt")
                   .read_text("utf-8").split())
    files = filter_indices(args.splits, root, out_dir, excluded)

    # runtime assertion: the exclusion list is not merely defined, it landed.
    n_dropped = sum(v["n_dropped"] for v in files.values())
    if n_dropped != len(excluded):
        raise AssertionError(
            f"{len(excluded)} sample_id(s) were listed for exclusion but "
            f"{n_dropped} line(s) were dropped across {args.splits}")

    manifest = {
        "stamp": args.stamp,
        "criterion": ("drop every task_type == 'local' row whose mask family F "
                      "carries a rule and whose GT alpha mean < T_F, with "
                      f"{ {k: v for k, v in sorted(rules.items())} }; the mean is "
                      "taken over the GT alpha at its own resolution, no resize; "
                      "families with no rule are never read and never dropped; "
                      "task_type == 'style' rows have no mask member and are out "
                      "of scope"),
        "rules": dict(sorted(rules.items())),
        "family_source": "q3vl/whereb/amort/data.py:172 family_labels() -> "
                         "q3vl/whereb/scripts/mask_type_stats.py:38 family_of("
                         ".vrmeta.json slot_id)",
        "alpha_source": "q3vl/whatb/evaldata.py:56 SampleStore.alpha() -> "
                        "where_a-20260805/maskviews/<split>/*.maskhi.png (mode L)",
        "alpha_reuse": None if args.alpha_jsonl is None else {
            "alpha_jsonl": str(Path(args.alpha_jsonl).resolve()),
            "alpha_jsonl_sha256": sha256_file(Path(args.alpha_jsonl)),
            "stats_ref": str(Path(args.stats_ref).resolve()),
            "stats_ref_sha256": sha256_file(Path(args.stats_ref)),
            "note": ("no mask was opened by this run; alpha_mean is "
                     "threshold-independent, so the rule set was applied to "
                     "the per-sample alphas of the referenced stage A and the "
                     "index-side counts (family_counts, winner_confidence, "
                     "n_total / n_local) were read back from its stats"),
        },
        "src_root": str(root),
        "mask_root": args.mask_root,
        "splits": files,
        "n_excluded_sample_ids": len(excluded),
        "excluded_sample_ids_sha256": hashlib.sha256(
            (out_dir / "excluded_sample_ids.txt").read_bytes()).hexdigest(),
        "mask_alpha_mean_jsonl_sha256": hashlib.sha256(
            (out_dir / "mask_alpha_mean.jsonl").read_bytes()).hexdigest(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "argv": vars(args),
        "stage_a": {k: v for k, v in stats.items() if k != "alpha_failures"},
        "alpha_failures": stats.get("alpha_failures", []),
        "seconds": round(time.time() - t0, 1),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"done in {time.time()-t0:.0f}s -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
