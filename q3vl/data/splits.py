"""Isolated evaluation splits and the group-level LUT reserve (METACANVAS 2.2).

Four mutually exclusive evaluation sets are carved out of the frozen ``eval``
authority, and one set of LUT identities is removed from the *training* manifest
so that ``T_lut_unseen`` really is unseen:

    T_lut_unseen  every sample whose ``lut_id`` was reserved (never trained on)
    T_final       held-out test, seen LUTs
    V_where       selection set for the Where arms
    V_what        selection set for the What arms

Isolation rules implemented here, quoted from METACANVAS 2.2:

* a source image never crosses the train / select / test boundary.  The frozen
  split authority already gives train-vs-eval source disjointness (measured: 0
  shared sources); this module additionally keeps every source whole and gives
  the select role (``V_where`` + ``V_what``) and the test role (``T_final`` +
  ``T_lut_unseen``) disjoint source sets.  ``T_final`` and ``T_lut_unseen`` are
  both test-role and deliberately share sources -- they are separated by LUT
  identity, which is the axis their comparison is about.
* a LUT identity never crosses ``T_lut_unseen`` and any training set;
* groups -- ``(source_id, lut_id, build)`` -- are never split;
* everything is deterministic: selection order is by a fixed key, ties are
  broken by a SHA-256 of the group key, and no RNG is consulted.

Why the source partition is *optimised* rather than random: reserving a LUT is
expensive (each reserved LUT costs ~48 training samples per eval sample gained),
so the budget only buys ~700 reserved eval samples.  A random third of the
sources would strand two thirds of them in the select role, where they cannot be
used.  Choosing the test sources to concentrate the reserved samples recovers
most of them at no cost to the other sets' sizes.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

from .config import LUT_RESERVE_TARGET_EVAL, LUT_RESERVE_TRAIN_BUDGET

SPLITS = ("V_where", "V_what", "T_final", "T_lut_unseen")


def _digest_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class _Union:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # deterministic: the lexicographically smaller root wins
            lo, hi = (ra, rb) if ra < rb else (rb, ra)
            self.parent[hi] = lo


def source_groups(rows: Iterable[dict[str, Any]]) -> dict[str, str]:
    """``sft_id -> source group key``, merging sources that share an input file.

    A single physical photograph can be registered under two ``source_id``s in
    two builds; grouping only by ``source_id`` would then let the same picture
    land in two different evaluation sets.  Identical ``i_in_path`` values are
    unioned so that cannot happen.
    """
    union = _Union()
    by_path: dict[str, str] = {}
    rows = list(rows)
    for row in rows:
        sid = row["source_id"] or f"path:{row['i_in_path']}"
        union.find(sid)
        path = row["i_in_path"]
        if path:
            other = by_path.get(path)
            if other is None:
                by_path[path] = sid
            else:
                union.union(sid, other)
    return {row["sft_id"]: union.find(row["source_id"] or f"path:{row['i_in_path']}")
            for row in rows}


@dataclass
class ReserveResult:
    lut_ids: set[str] = field(default_factory=set)
    eval_gain: int = 0
    train_cost: int = 0
    train_total: int = 0
    per_major: list[dict[str, Any]] = field(default_factory=list)

    @property
    def train_fraction(self) -> float:
        return self.train_cost / self.train_total if self.train_total else 0.0


def choose_lut_reserve(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    target_eval: int = LUT_RESERVE_TARGET_EVAL,
    budget_fraction: float = LUT_RESERVE_TRAIN_BUDGET,
) -> ReserveResult:
    """Pick LUT identities to hold out, stratified by taxonomy ``major``."""
    train_count: dict[str, int] = defaultdict(int)
    for row in train_rows:
        train_count[row["lut_id"]] += 1
    eval_count: dict[str, int] = defaultdict(int)
    lut_major: dict[str, str] = {}
    for row in eval_rows:
        eval_count[row["lut_id"]] += 1
        lut_major.setdefault(row["lut_id"], row.get("major") or "unknown")

    budget = int(budget_fraction * len(train_rows))
    eval_total = len(eval_rows)

    by_major: dict[str, list[str]] = defaultdict(list)
    for lut in eval_count:
        by_major[lut_major[lut]].append(lut)

    result = ReserveResult(train_total=len(train_rows))
    major_eval = {m: sum(eval_count[l] for l in luts) for m, luts in by_major.items()}

    def cost_of(lut: str) -> int:
        return train_count.get(lut, 0)

    def ordered(luts: list[str]) -> list[str]:
        # cheapest eval sample first; SHA-256 of the id breaks ties so the
        # choice never depends on dict ordering.
        return sorted(luts, key=lambda l: (cost_of(l) / eval_count[l], -eval_count[l],
                                           _digest_key(l)))

    # Pass 1: every major gets its proportional slice of *both* budgets up front.
    # Spending the budget major-by-major in size order (an earlier version of
    # this function) exhausted it after six of the ten taxonomy majors and left
    # T_lut_unseen with no coverage of the other four -- which would have made
    # "unseen LUT generalisation" a claim about six styles, not about the corpus.
    picked_by_major: dict[str, list[str]] = {}
    gained_by_major: dict[str, int] = {}
    cost_by_major: dict[str, int] = {}
    for major in sorted(by_major, key=lambda m: (-major_eval[m], m)):
        share = major_eval[major] / eval_total
        major_target = target_eval * share
        major_budget = budget * share
        gained = cost = 0
        picked: list[str] = []
        for lut in ordered(by_major[major]):
            if gained >= major_target:
                break
            if cost + cost_of(lut) > major_budget:
                continue
            picked.append(lut)
            gained += eval_count[lut]
            cost += cost_of(lut)
        picked_by_major[major] = picked
        gained_by_major[major] = gained
        cost_by_major[major] = cost

    result.lut_ids = {lut for luts in picked_by_major.values() for lut in luts}
    result.eval_gain = sum(gained_by_major.values())
    result.train_cost = sum(cost_by_major.values())

    # Pass 2: redistribute whatever the proportional slices left unspent, by
    # global cost-effectiveness, until the eval target or the budget is reached.
    remaining = [l for l in ordered(list(eval_count)) if l not in result.lut_ids]
    for lut in remaining:
        if result.eval_gain >= target_eval:
            break
        if result.train_cost + cost_of(lut) > budget:
            continue
        result.lut_ids.add(lut)
        result.eval_gain += eval_count[lut]
        result.train_cost += cost_of(lut)
        major = lut_major[lut]
        picked_by_major[major].append(lut)
        gained_by_major[major] += eval_count[lut]
        cost_by_major[major] += cost_of(lut)

    for major in sorted(by_major, key=lambda m: (-major_eval[m], m)):
        result.per_major.append({
            "major": major,
            "eval_pool": major_eval[major],
            "target": round(target_eval * major_eval[major] / eval_total, 1),
            "luts_reserved": len(picked_by_major[major]),
            "eval_reserved": gained_by_major[major],
            "train_removed": cost_by_major[major],
        })
    return result


@dataclass
class SplitPlan:
    assignment: dict[str, str]          # sft_id -> split name
    unused: dict[str, str]              # sft_id -> reason
    group_role: dict[str, str]          # source group -> select/test
    sizes: dict[str, int]
    group_counts: dict[str, int]


def partition_eval(
    eval_rows: list[dict[str, Any]],
    reserved: set[str],
    groups: dict[str, str],
) -> SplitPlan:
    """Assign every eval survivor to one of the four sets (or mark it unused)."""
    per_group: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"open": [], "reserved": []})
    for row in eval_rows:
        group = groups[row["sft_id"]]
        bucket = "reserved" if row["lut_id"] in reserved else "open"
        per_group[group][bucket].append(row["sft_id"])

    total_open = sum(len(g["open"]) for g in per_group.values())
    test_open_target = total_open / 3.0

    # test role first: maximise reserved capture per unit of the T_final quota
    # a group consumes, then fill the quota.  Deterministic order, no RNG.
    ordered = sorted(
        per_group.items(),
        key=lambda kv: (-len(kv[1]["reserved"]) / (len(kv[1]["open"]) + 1),
                        -len(kv[1]["reserved"]), _digest_key(kv[0])),
    )
    role: dict[str, str] = {}
    test_open = 0
    for group, buckets in ordered:
        if not buckets["reserved"]:
            continue
        if test_open + len(buckets["open"]) > test_open_target and test_open > 0:
            continue
        role[group] = "test"
        test_open += len(buckets["open"])

    remaining = [(g, b) for g, b in ordered if g not in role]
    # top up T_final with the open-only groups, largest first
    for group, buckets in sorted(remaining, key=lambda kv: (-len(kv[1]["open"]), _digest_key(kv[0]))):
        if test_open >= test_open_target:
            break
        role[group] = "test"
        test_open += len(buckets["open"])

    select_groups = [(g, b) for g, b in ordered if g not in role]
    # balanced deterministic 1:1 split of the select role
    bucket_sizes = {"V_where": 0, "V_what": 0}
    for group, buckets in sorted(select_groups,
                                 key=lambda kv: (-len(kv[1]["open"]), _digest_key(kv[0]))):
        target = min(bucket_sizes, key=lambda k: (bucket_sizes[k], k))
        role[group] = target
        bucket_sizes[target] += len(buckets["open"])

    assignment: dict[str, str] = {}
    unused: dict[str, str] = {}
    for group, buckets in per_group.items():
        where = role[group]
        if where == "test":
            for sft_id in buckets["open"]:
                assignment[sft_id] = "T_final"
            for sft_id in buckets["reserved"]:
                assignment[sft_id] = "T_lut_unseen"
        else:
            for sft_id in buckets["open"]:
                assignment[sft_id] = where
            for sft_id in buckets["reserved"]:
                # a reserved-LUT sample cannot sit in a selection set: it would
                # make Where/What checkpoints be chosen partly on unseen LUTs.
                unused[sft_id] = "reserved_lut_in_select_source"

    sizes = {name: sum(1 for v in assignment.values() if v == name) for name in SPLITS}
    group_counts = {
        "test": sum(1 for v in role.values() if v == "test"),
        "V_where": sum(1 for v in role.values() if v == "V_where"),
        "V_what": sum(1 for v in role.values() if v == "V_what"),
        "total": len(role),
    }
    return SplitPlan(assignment, unused, role, sizes, group_counts)


def audit(rows_by_id: dict[str, dict[str, Any]], assignment: dict[str, str],
          train_ids: set[str], groups: dict[str, str]) -> dict[str, Any]:
    """Every isolation claim this module makes, re-derived from the output."""
    by_split: dict[str, set[str]] = {name: set() for name in SPLITS}
    for sft_id, name in assignment.items():
        by_split[name].add(sft_id)

    def field_set(ids: set[str], key: str) -> set[str]:
        return {rows_by_id[i][key] for i in ids if rows_by_id[i].get(key)}

    def group_set(ids: set[str]) -> set[str]:
        return {groups[i] for i in ids}

    def triple_set(ids: set[str]) -> set[tuple[str, str, str]]:
        """The protocol's own group key: (source_image_id, lut_id, build)."""
        return {(groups[i], rows_by_id[i]["lut_id"], rows_by_id[i]["build"]) for i in ids}

    names = list(SPLITS)
    id_overlaps = {}
    source_overlaps = {}
    group_overlaps = {}
    triple_overlaps = {}
    lut_overlaps = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            id_overlaps[f"{a}|{b}"] = len(by_split[a] & by_split[b])
            source_overlaps[f"{a}|{b}"] = len(field_set(by_split[a], "source_id")
                                              & field_set(by_split[b], "source_id"))
            group_overlaps[f"{a}|{b}"] = len(group_set(by_split[a]) & group_set(by_split[b]))
            triple_overlaps[f"{a}|{b}"] = len(triple_set(by_split[a]) & triple_set(by_split[b]))
            lut_overlaps[f"{a}|{b}"] = len(field_set(by_split[a], "lut_id")
                                           & field_set(by_split[b], "lut_id"))

    train_luts = field_set(train_ids, "lut_id")
    train_sources = field_set(train_ids, "source_id")
    train_groups = group_set(train_ids)
    return {
        "sample_id_overlap": id_overlaps,
        "source_overlap": source_overlaps,
        "source_group_overlap": group_overlaps,
        "protocol_group_overlap_source_lut_build": triple_overlaps,
        "lut_overlap_between_eval_sets": lut_overlaps,
        "train_x_eval_sample_overlap": {
            name: len(train_ids & by_split[name]) for name in names
        },
        "train_x_eval_source_overlap": {
            name: len(train_sources & field_set(by_split[name], "source_id")) for name in names
        },
        "train_x_eval_source_group_overlap": {
            name: len(train_groups & group_set(by_split[name])) for name in names
        },
        "train_x_eval_protocol_group_overlap": {
            name: len(triple_set(train_ids) & triple_set(by_split[name])) for name in names
        },
        "train_x_T_lut_unseen_lut_overlap": len(train_luts & field_set(by_split["T_lut_unseen"], "lut_id")),
        "select_vs_test_source_overlap": len(
            (field_set(by_split["V_where"], "source_id") | field_set(by_split["V_what"], "source_id"))
            & (field_set(by_split["T_final"], "source_id") | field_set(by_split["T_lut_unseen"], "source_id"))
        ),
        "n_train_luts": len(train_luts),
        "n_reserved_luts": len(field_set(by_split["T_lut_unseen"], "lut_id")),
    }
