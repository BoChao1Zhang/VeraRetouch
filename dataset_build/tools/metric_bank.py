"""WP15b: candidate perceptual-direction metrics computed from archived pixels.

Nothing here is imported by the build.  It is a measurement bank: for one
before/after/alpha triple it emits every candidate statistic WP15 wants to score
by ROC, each as a *signed* number whose sign is the direction it would report.

Families (the numbering follows the WP15 brief)

1. ``bucket.*``   -- 11 basic colour-name buckets, area share + alpha-weighted
                     dL / da / db / dC inside each bucket.
2. ``noclip.*``   -- means after dropping pixels near 0 or 255 in either image;
                     ``clip.*`` carries the clipped fractions on their own.
3. ``chromatic.hue_*`` -- the joint (da, db) vector on chromatic pixels: length
                     and angle, plus the warm/cool and green/magenta projections.
4. ``chromatic.*``  -- chroma statistics restricted to pixels chromatic in
                     *either* image (mean and median).
5. ``hi.*``       -- the same statistics on the high-alpha core (alpha >= 0.75),
                     against ``full.*`` as its control.
6. ``full.*``     -- the production statistic, unchanged, as the baseline arm.
7. ``top.*``      -- per-bucket medians and the top-decile coherent change.

Support sets are named by prefix and every prefix carries the same statistic
names, so a ROC sweep is a cross product of (support x statistic).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "/home/bc/VeraRetouch")
sys.path.insert(0, "/home/bc/VeraRetouch/dataset_build/src")
from construct.visibility import srgb_to_lab  # noqa: E402

# --- colour-name buckets ---------------------------------------------------
#
# van de Weijer's w2c LUT is not obtainable without a download, so this is the
# sanctioned fallback: an explicit rule set over CIELAB that assigns each pixel
# one of the 11 Berlin-Kay basic colour terms.  A plain nearest-centroid in Lab
# was tried first and rejected: the canonical sRGB primaries sit at chroma 80+
# while real photographic surfaces sit at chroma 10-40, so every natural pixel
# fell into the achromatic centroids.  Assignment is therefore hue-angle based
# for chromatic pixels and lightness based for achromatic ones, with brown and
# pink split off by lightness as the colour-naming literature does.
ACHROMATIC_C = 12.0          # C* below this reads as black / grey / white
HUE_BINS = (                 # (name, lo_deg, hi_deg) over h = atan2(b*, a*)
    ("red", 345.0, 25.0),
    ("orange", 25.0, 60.0),
    ("yellow", 60.0, 100.0),
    ("green", 100.0, 180.0),
    ("blue", 180.0, 285.0),
    ("purple", 285.0, 325.0),
    ("pink", 325.0, 345.0),
)
BUCKETS = ["black", "grey", "white", "red", "orange", "yellow", "green",
           "blue", "purple", "pink", "brown"]
CHROMATIC_BUCKETS = BUCKETS[3:]
VIVID_C = 20.0               # "chromatic pixel" threshold, WP11/WP13 definition
HI_ALPHA = 0.75


def bucket_labels(lab: np.ndarray) -> np.ndarray:
    """Per-pixel index into ``BUCKETS``."""
    lightness, a_star, b_star = np.moveaxis(lab, -1, 0)
    chroma = np.hypot(a_star, b_star)
    hue = np.degrees(np.arctan2(b_star, a_star)) % 360.0
    out = np.full(lightness.shape, BUCKETS.index("grey"), dtype=np.int8)

    achromatic = chroma < ACHROMATIC_C
    out[achromatic & (lightness < 25.0)] = BUCKETS.index("black")
    out[achromatic & (lightness >= 75.0)] = BUCKETS.index("white")

    chromatic = ~achromatic
    for name, lo, hi in HUE_BINS:
        if lo > hi:                      # the red bin wraps through 0
            sel = chromatic & ((hue >= lo) | (hue < hi))
        else:
            sel = chromatic & (hue >= lo) & (hue < hi)
        out[sel] = BUCKETS.index(name)
    # brown is dark orange/yellow; pink is light, moderate-chroma red
    warm = chromatic & (hue >= 20.0) & (hue < 100.0)
    out[warm & (lightness < 45.0)] = BUCKETS.index("brown")
    reddish = chromatic & ((hue >= 325.0) | (hue < 25.0))
    out[reddish & (lightness >= 60.0) & (chroma < 50.0)] = BUCKETS.index("pink")
    return out


# --- weighted helpers ------------------------------------------------------

def _mean(values: np.ndarray, weight: np.ndarray, total: float) -> float:
    return float((values * weight).sum() / total)


def _std(values: np.ndarray, weight: np.ndarray, total: float) -> float:
    mean = _mean(values, weight, total)
    return math.sqrt(max(0.0, _mean((values - mean) ** 2, weight, total)))


def _wquantile(values: np.ndarray, weight: np.ndarray, q: float) -> float:
    """Weighted quantile; ``values``/``weight`` are flat and weight sums > 0."""
    order = np.argsort(values, kind="stable")
    v, w = values[order], weight[order]
    cum = np.cumsum(w)
    cutoff = q * cum[-1]
    index = int(np.searchsorted(cum, cutoff, side="left"))
    return float(v[min(index, len(v) - 1)])


class Pair:
    """A before/after/alpha triple, resampled onto the AFTER grid."""

    def __init__(self, before: Path, after: Path, cgt: Path | None) -> None:
        after_image = Image.open(after).convert("RGB")
        size = after_image.size
        after_rgb = np.asarray(after_image, np.float32) / 255.0
        before_image = Image.open(before).convert("RGB")
        if before_image.size != size:
            before_image = before_image.resize(size, Image.Resampling.BILINEAR)
        before_rgb = np.asarray(before_image, np.float32) / 255.0

        if cgt is not None:
            alpha_image = Image.open(cgt).convert("L")
            if alpha_image.size != size:
                alpha_image = alpha_image.resize(size, Image.Resampling.BILINEAR)
            alpha = np.asarray(alpha_image, np.float32).astype(np.float64) / 255.0
        else:
            alpha = np.ones(before_rgb.shape[:2], np.float64)

        self.alpha = alpha
        self.lab1 = srgb_to_lab(before_rgb).astype(np.float64)
        self.lab2 = srgb_to_lab(after_rgb).astype(np.float64)
        self.l1, self.a1, self.b1 = np.moveaxis(self.lab1, -1, 0)
        self.l2, self.a2, self.b2 = np.moveaxis(self.lab2, -1, 0)
        self.c1 = np.hypot(self.a1, self.b1)
        self.c2 = np.hypot(self.a2, self.b2)
        # saturation as the eye reads it: chroma relative to lightness
        self.s1 = self.c1 / np.maximum(self.l1, 1.0)
        self.s2 = self.c2 / np.maximum(self.l2, 1.0)

        rgb1_255, rgb2_255 = before_rgb * 255.0, after_rgb * 255.0
        self.clipped = (
            (rgb1_255.max(axis=-1) >= 250.0) | (rgb1_255.min(axis=-1) <= 5.0)
            | (rgb2_255.max(axis=-1) >= 250.0) | (rgb2_255.min(axis=-1) <= 5.0)
        )
        self.clip_hi_1 = rgb1_255.max(axis=-1) >= 250.0
        self.clip_hi_2 = rgb2_255.max(axis=-1) >= 250.0
        # Two readings of "a coloured pixel", and they are not interchangeable.
        # ``chromatic`` (either image) counts a pixel that the edit *turned*
        # colourful; ``chromatic_before`` counts only what was already coloured
        # before the edit.  WP11/WP13's diagnostic used the before-only form, and
        # on e424ffd1 the two disagree in sign (-13.3 vs +6.4), so both are here
        # and the ROC decides.
        self.chromatic = (self.c1 >= VIVID_C) | (self.c2 >= VIVID_C)
        self.chromatic_before = self.c1 >= VIVID_C
        self.chromatic_after = self.c2 >= VIVID_C
        self.hi_alpha = alpha >= HI_ALPHA * max(1e-9, float(alpha.max()))

    # -- one support set ---------------------------------------------------
    def stats(self, mask: np.ndarray | None) -> dict[str, float] | None:
        weight = self.alpha if mask is None else self.alpha * mask
        total = float(weight.sum())
        if total <= 1e-6:
            return None
        d_l = _mean(self.l2 - self.l1, weight, total)
        d_a = _mean(self.a2 - self.a1, weight, total)
        d_b = _mean(self.b2 - self.b1, weight, total)
        d_c = _mean(self.c2 - self.c1, weight, total)
        out = {
            "weight_frac": total / float(self.alpha.sum()),
            "d_L": d_l,
            "d_a": d_a,
            "d_b": d_b,
            "d_C": d_c,
            "d_sat_x100": 100.0 * _mean(self.s2 - self.s1, weight, total),
            "d_contrast": (_std(self.l2, weight, total) - _std(self.l1, weight, total)),
            "hue_len": math.hypot(d_a, d_b),
            "hue_deg": math.degrees(math.atan2(d_b, d_a)) % 360.0,
        }
        flat_w = weight.reshape(-1)
        keep = flat_w > 1e-9
        flat_w = flat_w[keep]
        for name, field in (("d_L", self.l2 - self.l1), ("d_b", self.b2 - self.b1),
                            ("d_a", self.a2 - self.a1), ("d_C", self.c2 - self.c1)):
            flat_v = field.reshape(-1)[keep]
            out[f"{name}_median"] = _wquantile(flat_v, flat_w, 0.5)
        l1f, l2f = self.l1.reshape(-1)[keep], self.l2.reshape(-1)[keep]
        out["d_p90p10"] = ((_wquantile(l2f, flat_w, 0.9) - _wquantile(l2f, flat_w, 0.1))
                           - (_wquantile(l1f, flat_w, 0.9) - _wquantile(l1f, flat_w, 0.1)))
        # top-decile coherent change: do the pixels that moved most agree in sign?
        for name, field in (("d_C", self.c2 - self.c1), ("d_L", self.l2 - self.l1),
                            ("d_b", self.b2 - self.b1), ("d_a", self.a2 - self.a1)):
            flat_v = field.reshape(-1)[keep]
            order = np.argsort(np.abs(flat_v), kind="stable")[::-1]
            cum = np.cumsum(flat_w[order])
            cut = int(np.searchsorted(cum, 0.10 * cum[-1], side="left")) + 1
            picked = order[:cut]
            out[f"{name}_top_decile"] = float(
                (flat_v[picked] * flat_w[picked]).sum() / flat_w[picked].sum())
        return out

    # -- colour-name buckets ----------------------------------------------
    def buckets(self) -> dict:
        labels1 = bucket_labels(self.lab1)
        labels2 = bucket_labels(self.lab2)
        total = float(self.alpha.sum())
        out: dict[str, dict] = {}
        for index, name in enumerate(BUCKETS):
            sel = labels1 == index
            weight = self.alpha * sel
            mass = float(weight.sum())
            entry = {
                "area_before": mass / total,
                "area_after": float((self.alpha * (labels2 == index)).sum()) / total,
            }
            if mass > 1e-6:
                entry.update({
                    "d_L": _mean(self.l2 - self.l1, weight, mass),
                    "d_a": _mean(self.a2 - self.a1, weight, mass),
                    "d_b": _mean(self.b2 - self.b1, weight, mass),
                    "d_C": _mean(self.c2 - self.c1, weight, mass),
                    "d_C_median": _wquantile(
                        (self.c2 - self.c1).reshape(-1)[weight.reshape(-1) > 1e-9],
                        weight.reshape(-1)[weight.reshape(-1) > 1e-9], 0.5),
                })
            out[name] = entry
        return out


def bucket_scores(buckets: dict) -> dict[str, float]:
    """Row-level signed scores derived from the per-bucket table."""
    chromatic = [(name, buckets[name]) for name in CHROMATIC_BUCKETS
                 if "d_C" in buckets[name]]
    out: dict[str, float] = {}
    area = sum(entry["area_before"] for _, entry in chromatic)
    out["chromatic_area"] = area
    if not chromatic or area <= 1e-9:
        return out
    for field in ("d_L", "d_a", "d_b", "d_C"):
        out[f"area_weighted_{field}"] = sum(
            entry["area_before"] * entry[field] for _, entry in chromatic) / area
    out["area_weighted_d_C_median"] = sum(
        entry["area_before"] * entry["d_C_median"] for _, entry in chromatic) / area
    # the bucket whose change is most likely to be the one a viewer reports:
    # large change on a bucket that occupies real estate
    salience = max(chromatic,
                   key=lambda kv: abs(kv[1]["d_C"]) * math.sqrt(kv[1]["area_before"]))
    out["dominant_bucket"] = BUCKETS.index(salience[0])
    for field in ("d_L", "d_a", "d_b", "d_C"):
        out[f"dominant_{field}"] = salience[1][field]
    out["dominant_area"] = salience[1]["area_before"]
    largest = max(chromatic, key=lambda kv: kv[1]["area_before"])
    out["largest_bucket"] = BUCKETS.index(largest[0])
    for field in ("d_L", "d_a", "d_b", "d_C"):
        out[f"largest_{field}"] = largest[1][field]
    # how much bucket membership itself moved (a hue rotation the means miss)
    out["bucket_area_churn"] = 0.5 * sum(
        abs(buckets[name]["area_after"] - buckets[name]["area_before"])
        for name in BUCKETS)
    return out


SUPPORTS = {
    "full": None,
    "noclip": "not_clipped",
    "chromatic": "chromatic",
    "chromatic_noclip": "chromatic_noclip",
    "hi": "hi_alpha",
    "hi_chromatic": "hi_chromatic",
}


def measure(before: Path, after: Path, cgt: Path | None) -> dict:
    pair = Pair(before, after, cgt)
    masks = {
        "full": None,
        "noclip": ~pair.clipped,
        "chromatic": pair.chromatic,
        "chromatic_noclip": pair.chromatic & ~pair.clipped,
        "chrombefore": pair.chromatic_before,
        "chrombefore_noclip": pair.chromatic_before & ~pair.clipped,
        "chromafter": pair.chromatic_after,
        "hi": pair.hi_alpha,
        "hi_chromatic": pair.hi_alpha & pair.chromatic,
        "hi_chrombefore": pair.hi_alpha & pair.chromatic_before,
    }
    record: dict = {"support": {}}
    for name, mask in masks.items():
        stats = pair.stats(mask)
        if stats is not None:
            record["support"][name] = stats
    total = float(pair.alpha.sum())
    record["clip"] = {
        "frac_before": float((pair.alpha * pair.clip_hi_1).sum()) / total,
        "frac_after": float((pair.alpha * pair.clip_hi_2).sum()) / total,
        "frac_either": float((pair.alpha * pair.clipped).sum()) / total,
    }
    record["clip"]["delta"] = record["clip"]["frac_after"] - record["clip"]["frac_before"]
    record["chromatic_frac"] = float((pair.alpha * pair.chromatic).sum()) / total
    record["chrombefore_frac"] = float((pair.alpha * pair.chromatic_before).sum()) / total
    record["chromafter_frac"] = float((pair.alpha * pair.chromatic_after).sum()) / total
    record["hi_alpha_frac"] = float((pair.alpha * pair.hi_alpha).sum()) / total
    record["alpha_mean"] = total / pair.alpha.size
    record["buckets"] = pair.buckets()
    record["bucket_scores"] = bucket_scores(record["buckets"])
    return record


# --- row sources -----------------------------------------------------------

REVIEW = Path("/var/cache/veradata/annot_review/fresh200-v4a1-20260729")
EVIDENCE = Path("/var/cache/veradata/annot_review/hints_mean_bias_2026-07-29")
BUILD = Path("/mnt/nfs/bc/data/builds/fresh200-v4a1-20260729")


def panel_rows() -> list[dict]:
    gdirs = {}
    for directory in sorted(REVIEW.glob("g[0-9][0-9][0-9]_*")):
        gdirs[json.loads((directory / "annot.json").read_text())["group_id"]] = directory
    modes = {}
    for line in (REVIEW / "blind" / "sample_index.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            modes[row["sft_id"]] = row["slot_mode"]
    stored = {}
    groups_path = BUILD / "groups.jsonl"
    if groups_path.is_file():
        for line in groups_path.read_text().splitlines():
            if not line.strip():
                continue
            group = json.loads(line)
            for candidate in group["candidates"]:
                stored[candidate["candidate_id"]] = candidate.get("objective_hints")
    out = []
    for line in (REVIEW / "run" / "sft.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        directory = gdirs[row["group_id"]]
        rank = row["winner_rank"]
        cgt = directory / f"cgt_rank{rank}.png"
        out.append({
            "row_id": row["sft_id"], "source": "fresh200_panel",
            "sft_id": row["sft_id"], "group_id": row["group_id"],
            "candidate_id": row["candidate_id"], "task_type": row["task_type"],
            "slot_mode": modes.get(row["sft_id"]),
            "before": directory / "before.jpg",
            "after": directory / f"after_rank{rank}.jpg",
            "cgt": cgt if cgt.is_file() else None,
            "stored_hints": stored.get(row["candidate_id"]),
        })
    return out


def evidence_rows() -> list[dict]:
    out = []
    for directory in sorted(EVIDENCE.glob("fresh*")):
        if not directory.is_dir():
            continue
        hints = json.loads((directory / "hints.json").read_text())
        cgt = directory / "cgt.png"
        out.append({
            "row_id": directory.name, "source": "evidence22",
            "sft_id": hints["sft_id"], "group_id": hints.get("group_id"),
            "candidate_id": hints.get("candidate_id"),
            "task_type": hints.get("task_type"), "slot_mode": hints.get("slot_mode"),
            "before": directory / "before.jpg", "after": directory / "after.jpg",
            "cgt": cgt if cgt.is_file() else None,
            "stored_hints": hints.get("archived_objective_hints"),
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("which", choices=["panel", "evidence", "both"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = []
    if args.which in ("panel", "both"):
        rows += panel_rows()
    if args.which in ("evidence", "both"):
        rows += evidence_rows()
    if args.limit:
        rows = rows[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.is_file():
        for line in args.out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["row_id"])
    with args.out.open("a", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            if row["row_id"] in done:
                continue
            record = measure(row["before"], row["after"], row["cgt"])
            payload = {key: (str(value) if isinstance(value, Path) else value)
                       for key, value in row.items()}
            payload["metrics"] = record
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{index + 1}/{len(rows)} {row['row_id']}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
