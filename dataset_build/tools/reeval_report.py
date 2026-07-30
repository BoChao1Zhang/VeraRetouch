"""Stage D of the WP5 re-evaluation: turn the stage A-C logs into statistics.

Every number the report quotes is produced here, so the report can be re-derived
from the JSONL logs without re-running a single relay call.

The decomposition that matters is in ``compare_old_vs_new``.  The first-round
conclusion ("44 fail -> 0 fail") mixed three effects that this tool separates:

  measurement effect  = blind re-score of the OLD text  -  first-round score of
                        the same OLD text.  The text is identical, so whatever
                        moves is method plus regression to the mean.
  prompt effect       = blind score of the NEW text  -  blind score of the OLD
                        text, same request, sides randomised.  This is the only
                        part attributable to prompt-v2/v3.
  judge noise         = stage B2 test-retest on unchanged text.  Nothing smaller
                        than this counts as an effect at all.

No scipy: the sign test is an exact binomial sum and the proportion intervals
are Wilson, both a dozen lines.

Usage:
  python -m dataset_build.tools.reeval_report
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REVIEW_ROOT = Path("/var/cache/veradata/annot_review/eval100-annotqa-20260727")
BUILD_ROOT = Path("/mnt/nfs/bc/data/builds/eval100-annotqa-20260727")
DEFAULT_OUT = REVIEW_ROOT / "wp5_reeval"
DIMENSIONS = ("consistency", "reasoning_alignment", "language", "usability")
# first-round dimension names, for the old/new bridge
OLD_EQUIV = {"consistency": "consistency", "reasoning_alignment": "reasoning",
             "language": "language", "usability": "usability"}


def read_ok(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    latest: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("ok") or row["key"] not in latest:
            latest[row["key"]] = row
    return [row for row in latest.values() if row.get("ok")]


def read_all(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def sd(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mu = sum(values) / len(values)
    return round(math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1)), 4)


def binom_two_sided(successes: int, trials: int, p: float = 0.5) -> float:
    """Exact two-sided binomial p-value (point-probability method)."""
    if trials == 0:
        return 1.0
    def pmf(k: int) -> float:
        return math.comb(trials, k) * p ** k * (1 - p) ** (trials - k)
    observed = pmf(successes)
    total = min(1.0, sum(pmf(k) for k in range(trials + 1)
                         if pmf(k) <= observed * (1 + 1e-9)))
    # keep three significant digits rather than rounding tiny p-values to 0.0
    return float(f"{total:.3g}")


def wilson(successes: int, trials: int, z: float = 1.96) -> list[float] | None:
    if trials == 0:
        return None
    phat = successes / trials
    denom = 1 + z * z / trials
    centre = (phat + z * z / (2 * trials)) / denom
    half = z * math.sqrt(phat * (1 - phat) / trials + z * z / (4 * trials * trials)) / denom
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 3:
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return round(num / (dx * dy), 4) if dx and dy else None


def dist(values: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(Counter(values).items(), key=lambda kv: str(kv[0])))


# --------------------------------------------------------------------------- #
# stage summaries
# --------------------------------------------------------------------------- #

def summarize_b1(rows: list[dict]) -> dict[str, Any]:
    def block(subset: list[dict]) -> dict[str, Any]:
        scores = {d: [r["result"]["scores"][d] for r in subset] for d in DIMENSIONS}
        return {
            "n": len(subset),
            "mean": {d: mean(v) for d, v in scores.items()},
            "sd": {d: sd(v) for d, v in scores.items()},
            "score_hist": {d: dist(v) for d, v in scores.items()},
            "verdict": dist(r["result"]["verdict"] for r in subset),
        }
    out = {"all": block(rows)}
    for task_type in sorted({r["task_type"] for r in rows}):
        out[task_type] = block([r for r in rows if r["task_type"] == task_type])
    # verdict vs usability anchor consistency
    anchor = Counter()
    for row in rows:
        usability = row["result"]["scores"]["usability"]
        implied = "pass" if usability >= 4 else "borderline" if usability == 3 else "fail"
        anchor[f"{row['result']['verdict']}|{implied}"] += 1
    out["verdict_vs_usability_anchor"] = dict(sorted(anchor.items()))
    return out


FAILURE_MODES = {
    # keyed on what the judge complained about, so the fix is actionable
    "region_extent_mismatch": (
        "region", "extent", "broader", "wider", "narrower", "only the", "confined",
        "limited to", "whole image", "entire image", "background also", "not limited",
        "mask covers", "region map",
    ),
    "magnitude_overstated": (
        "overstate", "exaggerat", "stronger than", "larger than", "more dramatic",
        "too strong", "subtle", "barely", "imperceptib", "hard to see", "minimal change",
        "no visible", "negligible",
    ),
    "direction_inverted": (
        "opposite", "inverted", "reversed", "contradict", "actually brighter",
        "actually darker", "actually warmer", "actually cooler", "wrong direction",
    ),
    "invented_problem": (
        "not visible", "not apparent", "cannot verify", "unsupported", "invent",
        "no evidence", "does not appear", "not present in the before",
    ),
    "subject_misidentified": (
        "misidentif", "wrong subject", "not the", "different subject", "identifies",
        "mislabel",
    ),
}


def classify_reasons(row: Mapping[str, Any]) -> list[str]:
    text = " ".join(str(v) for v in row["result"].get("reasons", {}).values()).lower()
    return sorted(mode for mode, keys in FAILURE_MODES.items()
                  if any(key in text for key in keys))


def summarize_failures(rows: list[dict]) -> dict[str, Any]:
    buckets: Counter = Counter()
    by_verdict: dict[str, Counter] = defaultdict(Counter)
    listing = []
    for row in rows:
        modes = classify_reasons(row)
        verdict = row["result"]["verdict"]
        buckets.update(modes)
        by_verdict[verdict].update(modes)
        if verdict != "pass":
            listing.append({
                "sft_id": row["sft_id"], "group_id": row["group_id"],
                "task_type": row["task_type"], "verdict": verdict,
                "scores": row["result"]["scores"], "modes": modes,
                "consistency_reason": row["result"]["reasons"]["consistency"],
            })
    return {
        "mode_counts_all": dict(buckets.most_common()),
        "mode_counts_by_verdict": {k: dict(v.most_common()) for k, v in by_verdict.items()},
        "non_pass_n": len(listing),
        "fail_ids": sorted(r["sft_id"] for r in listing if r["verdict"] == "fail"),
        "borderline_ids": sorted(r["sft_id"] for r in listing if r["verdict"] == "borderline"),
        "listing": sorted(listing, key=lambda r: (r["verdict"], sum(r["scores"].values()))),
    }


def summarize_b2(b1: list[dict], b2: list[dict]) -> dict[str, Any]:
    first = {r["sft_id"]: r["result"] for r in b1}
    pairs = [(first[r["sft_id"]], r["result"]) for r in b2 if r["sft_id"] in first]
    out: dict[str, Any] = {"n": len(pairs), "by_dimension": {}}
    for dimension in DIMENSIONS:
        a = [p[0]["scores"][dimension] for p in pairs]
        b = [p[1]["scores"][dimension] for p in pairs]
        deltas = [y - x for x, y in zip(a, b)]
        out["by_dimension"][dimension] = {
            "exact_agreement": round(sum(1 for d in deltas if d == 0) / len(deltas), 4) if deltas else None,
            "within_1": round(sum(1 for d in deltas if abs(d) <= 1) / len(deltas), 4) if deltas else None,
            "mean_abs_delta": mean([abs(d) for d in deltas]),
            "mean_delta": mean(deltas),
            "sd_delta": sd(deltas),
            "pearson_r": pearson(a, b),
            "mean_pass1": mean(a), "mean_pass2": mean(b),
            "delta_hist": dist(deltas),
        }
    va = [p[0]["verdict"] for p in pairs]
    vb = [p[1]["verdict"] for p in pairs]
    same = sum(1 for x, y in zip(va, vb) if x == y)
    out["verdict"] = {
        "exact_agreement": round(same / len(pairs), 4) if pairs else None,
        "confusion": dist(f"{x}->{y}" for x, y in zip(va, vb)),
        "pass1": dist(va), "pass2": dist(vb),
    }
    # the noise floor a claimed improvement must clear, per dimension
    out["noise_floor_mean_abs_delta"] = {
        d: out["by_dimension"][d]["mean_abs_delta"] for d in DIMENSIONS
    }
    return out


def summarize_b3(rows: list[dict]) -> dict[str, Any]:
    """Un-blind the A/B sides and count wins for the NEW text."""
    out: dict[str, Any] = {"n": len(rows), "by_dimension": {}, "old_scores": {},
                           "new_scores": {}}
    per_dim_delta: dict[str, list[int]] = defaultdict(list)
    old_by_dim: dict[str, list[int]] = defaultdict(list)
    new_by_dim: dict[str, list[int]] = defaultdict(list)
    prefs: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        side = row["side_map"]           # {"A": "old"|"new", "B": ...}
        letter_of = {v: k for k, v in side.items()}
        old_scores = row["result"][f"scores_{letter_of['old']}"]
        new_scores = row["result"][f"scores_{letter_of['new']}"]
        for dimension in DIMENSIONS:
            old_by_dim[dimension].append(old_scores[dimension])
            new_by_dim[dimension].append(new_scores[dimension])
            per_dim_delta[dimension].append(new_scores[dimension] - old_scores[dimension])
        for dimension in (*DIMENSIONS, "overall"):
            choice = row["result"]["preference"][dimension]
            if choice == "tie":
                prefs[dimension]["tie"] += 1
            else:
                prefs[dimension]["new" if side[choice] == "new" else "old"] += 1
    for dimension in (*DIMENSIONS, "overall"):
        wins = prefs[dimension]["new"]
        losses = prefs[dimension]["old"]
        ties = prefs[dimension]["tie"]
        decided = wins + losses
        entry = {
            "new_wins": wins, "old_wins": losses, "ties": ties,
            "win_rate_excl_ties": round(wins / decided, 4) if decided else None,
            "win_rate_ci95": wilson(wins, decided),
            "sign_test_p": binom_two_sided(wins, decided),
        }
        if dimension in DIMENSIONS:
            deltas = per_dim_delta[dimension]
            improved = sum(1 for d in deltas if d > 0)
            regressed = sum(1 for d in deltas if d < 0)
            entry.update({
                "mean_old": mean(old_by_dim[dimension]),
                "mean_new": mean(new_by_dim[dimension]),
                "mean_delta": mean(deltas),
                "sd_delta": sd(deltas),
                "improved": improved, "unchanged": len(deltas) - improved - regressed,
                "regressed": regressed,
                "score_sign_test_p": binom_two_sided(improved, improved + regressed),
                "delta_hist": dist(deltas),
            })
        out["by_dimension"][dimension] = entry
    out["old_scores"] = {d: {"mean": mean(old_by_dim[d]), "hist": dist(old_by_dim[d])}
                         for d in DIMENSIONS}
    out["new_scores"] = {d: {"mean": mean(new_by_dim[d]), "hist": dist(new_by_dim[d])}
                         for d in DIMENSIONS}
    # side-position bias check: did the judge favour position A?
    a_wins = sum(1 for row in rows if row["result"]["preference"]["overall"] == "A")
    b_wins = sum(1 for row in rows if row["result"]["preference"]["overall"] == "B")
    out["position_bias"] = {
        "A_overall_wins": a_wins, "B_overall_wins": b_wins,
        "sign_test_p": binom_two_sided(a_wins, a_wins + b_wins),
    }
    return out


def compare_old_vs_new(b1: list[dict], b3: list[dict],
                       b1o: list[dict] | None = None) -> dict[str, Any]:
    """Split the first-round headline into measurement effect and prompt effect."""
    old_review = {s["sft_id"]: s for s in
                  json.loads((REVIEW_ROOT / "scored_samples.json").read_text())}
    reannotated = {row["sft_id"] for row in b3}
    comparison = json.loads(
        (REVIEW_ROOT / "reannot_ab" / "final_comparison.json").read_text())
    first_round_after = {c["sft_id"]: c["after"]["scores"] for c in comparison}

    control, treated = [], []
    for row in b1:
        sid = row["sft_id"]
        if sid not in old_review:
            continue
        (treated if sid in reannotated else control).append(
            (old_review[sid]["scores"], row["result"]["scores"], sid))

    def bridge(pairs: list[tuple[dict, dict, str]]) -> dict[str, Any]:
        block: dict[str, Any] = {"n": len(pairs)}
        for new_dim, old_dim in OLD_EQUIV.items():
            olds = [p[0][old_dim] for p in pairs]
            news = [p[1][new_dim] for p in pairs]
            deltas = [n - o for o, n in zip(olds, news)]
            block[new_dim] = {
                "mean_first_round": mean(olds), "mean_blind": mean(news),
                "mean_delta": mean(deltas), "sd_delta": sd(deltas),
                "improved": sum(1 for d in deltas if d > 0),
                "regressed": sum(1 for d in deltas if d < 0),
                "pearson_r": pearson(olds, news),
            }
        block["verdict_first_round"] = dist(
            old_review[p[2]]["verdict"] for p in pairs)
        return block

    out = {
        "control_not_reannotated": bridge(control),
        "treated_reannotated_current_text": bridge(treated),
    }

    # measurement effect on frozen text: first-round score of the OLD text vs the
    # blind re-score of that same OLD text inside stage B3.
    old_first = {sid: old_review[sid]["scores"] for sid in reannotated if sid in old_review}
    old_blind: dict[str, dict] = {}
    new_blind: dict[str, dict] = {}
    for row in b3:
        letter_of = {v: k for k, v in row["side_map"].items()}
        old_blind[row["sft_id"]] = row["result"][f"scores_{letter_of['old']}"]
        new_blind[row["sft_id"]] = row["result"][f"scores_{letter_of['new']}"]
    shared = sorted(set(old_first) & set(old_blind))
    decomposition = {}
    for new_dim, old_dim in OLD_EQUIV.items():
        first = [old_first[s][old_dim] for s in shared]
        blind_old = [old_blind[s][new_dim] for s in shared]
        blind_new = [new_blind[s][new_dim] for s in shared]
        first_new = [first_round_after[s][old_dim] for s in shared
                     if s in first_round_after]
        decomposition[new_dim] = {
            "first_round_old_text": mean(first),
            "blind_old_text": mean(blind_old),
            "measurement_effect": round((mean(blind_old) or 0) - (mean(first) or 0), 4),
            "blind_new_text": mean(blind_new),
            "prompt_effect": round((mean(blind_new) or 0) - (mean(blind_old) or 0), 4),
            "first_round_new_text": mean(first_new),
            "first_round_claimed_gain": round(
                (mean(first_new) or 0) - (mean(first) or 0), 4),
        }
    out["decomposition_paired_b3"] = {"n": len(shared), "by_dimension": decomposition}

    # Same-format decomposition.  b1o scores the OLD text with the single-item b1
    # protocol, so first-round -> b1o is a method+selection shift on frozen text
    # measured the same way as the control group's shift, and b1o -> b1 is the
    # prompt effect measured without the paired-request contrast effect.
    if b1o:
        old_blind_single = {row["sft_id"]: row["result"] for row in b1o}
        new_blind_single = {row["sft_id"]: row["result"] for row in b1
                            if row["sft_id"] in old_blind_single}
        ids = sorted(set(old_blind_single) & set(new_blind_single) & set(old_first))
        same_format: dict[str, Any] = {}
        control_shift = out["control_not_reannotated"]
        for new_dim, old_dim in OLD_EQUIV.items():
            first = [old_first[s][old_dim] for s in ids]
            blind_old = [old_blind_single[s]["scores"][new_dim] for s in ids]
            blind_new = [new_blind_single[s]["scores"][new_dim] for s in ids]
            frozen_shift = round((mean(blind_old) or 0) - (mean(first) or 0), 4)
            control_delta = control_shift[new_dim]["mean_delta"] or 0
            prompt_deltas = [n - o for o, n in zip(blind_old, blind_new)]
            improved = sum(1 for d in prompt_deltas if d > 0)
            regressed = sum(1 for d in prompt_deltas if d < 0)
            same_format[new_dim] = {
                "first_round_old_text": mean(first),
                "blind_old_text_single": mean(blind_old),
                "frozen_text_shift": frozen_shift,
                "control_group_shift": round(control_delta, 4),
                "regression_to_mean_component": round(frozen_shift - control_delta, 4),
                "blind_new_text_single": mean(blind_new),
                "prompt_effect_unpaired": mean(prompt_deltas),
                "prompt_effect_sd": sd(prompt_deltas),
                "prompt_improved": improved, "prompt_regressed": regressed,
                "prompt_unchanged": len(prompt_deltas) - improved - regressed,
                "prompt_sign_test_p": binom_two_sided(improved, improved + regressed),
            }
        verdicts_old = dist(old_blind_single[s]["verdict"] for s in ids)
        verdicts_new = dist(new_blind_single[s]["verdict"] for s in ids)
        out["decomposition_same_format_b1o"] = {
            "n": len(ids),
            "by_dimension": same_format,
            "verdict_first_round_old_text": dist(
                old_review[s]["verdict"] for s in ids),
            "verdict_blind_old_text": verdicts_old,
            "verdict_blind_new_text": verdicts_new,
            "verdict_first_round_new_text": dist(
                ("pass" if first_round_after[s]["usability"] >= 4
                 else "borderline" if first_round_after[s]["usability"] == 3 else "fail")
                for s in ids if s in first_round_after),
        }
    return out


def summarize_c(rows: list[dict]) -> dict[str, Any]:
    by_slot: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        chose_winner = row["result"]["choice"] == row["winner_side"]
        row["_win"] = chose_winner
        by_slot[row["opponent_slot"]].append(row)
    out: dict[str, Any] = {"n": len(rows), "by_slot": {}, "by_mode": {}}
    for slot, subset in sorted(by_slot.items()):
        wins = sum(1 for r in subset if r["_win"])
        gaps = Counter(r["result"]["gap"] for r in subset)
        oa_gap = [r["winner_onealign"] - r["opponent_onealign"] for r in subset]
        out["by_slot"][slot] = {
            "n": len(subset), "winner_wins": wins,
            "win_rate": round(wins / len(subset), 4),
            "ci95": wilson(wins, len(subset)),
            "p_vs_chance": binom_two_sided(wins, len(subset)),
            "gap_judged": dict(gaps),
            "mean_onealign_gap": mean(oa_gap),
            "win_rate_when_clear": (
                round(sum(1 for r in subset if r["_win"] and r["result"]["gap"] == "clear")
                      / gaps["clear"], 4) if gaps["clear"] else None),
        }
    for mode in sorted({r["render_mode"] for r in rows}):
        subset = [r for r in rows if r["render_mode"] == mode]
        wins = sum(1 for r in subset if r["_win"])
        out["by_mode"][mode] = {
            "n": len(subset), "winner_wins": wins,
            "win_rate": round(wins / len(subset), 4),
            "ci95": wilson(wins, len(subset)),
            "p_vs_chance": binom_two_sided(wins, len(subset)),
            "by_slot": {
                slot: {
                    "n": len([r for r in subset if r["opponent_slot"] == slot]),
                    "wins": sum(1 for r in subset
                                if r["opponent_slot"] == slot and r["_win"]),
                }
                for slot in sorted(by_slot)
            },
        }
    wins_all = sum(1 for r in rows if r["_win"])
    out["overall"] = {
        "winner_wins": wins_all, "n": len(rows),
        "win_rate": round(wins_all / len(rows), 4) if rows else None,
        "ci95": wilson(wins_all, len(rows)),
        "p_vs_chance": binom_two_sided(wins_all, len(rows)),
        "gap_judged": dist(r["result"]["gap"] for r in rows),
    }
    # side-position sanity: with sides randomised, "left" should land near 50%
    left = sum(1 for r in rows if r["result"]["choice"] == "left")
    out["position_bias"] = {
        "left_chosen": left, "n": len(rows),
        "left_rate": round(left / len(rows), 4) if rows else None,
        "p_vs_chance": binom_two_sided(left, len(rows)),
        "winner_placed_left": sum(1 for r in rows if r["winner_side"] == "left"),
    }
    # does agreement scale with the OneAlign margin?
    margins = [r["winner_onealign"] - r["opponent_onealign"] for r in rows]
    outcomes = [1.0 if r["_win"] else 0.0 for r in rows]
    out["onealign_margin_vs_agreement"] = {
        "pearson_r": pearson(margins, outcomes),
        "by_margin_band": {},
    }
    bands = [(0, 2), (2, 5), (5, 10), (10, 1e9)]
    for low, high in bands:
        subset = [r for r, m in zip(rows, margins) if low <= m < high]
        if not subset:
            continue
        wins = sum(1 for r in subset if r["_win"])
        out["onealign_margin_vs_agreement"]["by_margin_band"][f"{low}-{high:g}"] = {
            "n": len(subset), "wins": wins, "win_rate": round(wins / len(subset), 4),
            "p_vs_chance": binom_two_sided(wins, len(subset)),
        }
    # groups whose candidates the judge could not separate at all
    per_group: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        per_group[row["group_id"]].append(row)
    degenerate = [gid for gid, subset in per_group.items()
                  if len(subset) >= 3 and all(r["result"]["gap"] == "negligible"
                                              for r in subset)]
    out["groups"] = {
        "n": len(per_group),
        "degenerate_all_negligible": len(degenerate),
        "degenerate_ids": sorted(degenerate),
        "swept_winner_lost_all": sorted(
            gid for gid, subset in per_group.items()
            if len(subset) >= 3 and not any(r["_win"] for r in subset)),
        "winner_won_all": sorted(
            gid for gid, subset in per_group.items()
            if len(subset) >= 3 and all(r["_win"] for r in subset)),
    }
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    b1 = read_ok(args.out / "b1_blind.jsonl")
    b1o = read_ok(args.out / "b1o_oldtext.jsonl")
    b2 = read_ok(args.out / "b2_retest.jsonl")
    b3 = read_ok(args.out / "b3_pairs.jsonl")
    c = read_ok(args.out / "c_iaa.jsonl")

    report: dict[str, Any] = {
        "counts": {
            "b1_ok": len(b1), "b1_rows": len(read_all(args.out / "b1_blind.jsonl")),
            "b1o_ok": len(b1o), "b2_ok": len(b2), "b3_ok": len(b3), "c_ok": len(c),
            "b1o_failed": len([r for r in read_all(args.out / "b1o_oldtext.jsonl")
                               if not r.get("ok")]),
            "b1_failed": len([r for r in read_all(args.out / "b1_blind.jsonl")
                              if not r.get("ok")]),
            "b2_failed": len([r for r in read_all(args.out / "b2_retest.jsonl")
                              if not r.get("ok")]),
            "b3_failed": len([r for r in read_all(args.out / "b3_pairs.jsonl")
                              if not r.get("ok")]),
            "c_failed": len([r for r in read_all(args.out / "c_iaa.jsonl")
                             if not r.get("ok")]),
        },
        "mech": json.loads((args.out / "mech_summary.json").read_text())
        if (args.out / "mech_summary.json").is_file() else None,
    }
    if b1:
        report["b1"] = summarize_b1(b1)
        report["b1_failure_modes"] = summarize_failures(b1)
    if b1 and b2:
        report["b2_retest"] = summarize_b2(b1, b2)
    if b3:
        report["b3_pairs"] = summarize_b3(b3)
    if b1o:
        report["b1o_old_text"] = summarize_b1(b1o)
    if b1 and b3:
        report["old_vs_new"] = compare_old_vs_new(b1, b3, b1o)
    if c:
        report["c_iaa"] = summarize_c(c)

    (args.out / "wp5_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
