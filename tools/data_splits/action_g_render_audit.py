#!/usr/bin/env python3
"""Action item G: are the rendered candidates of abstained groups (winner
margin < 1.0, no SFT row) landed and usable, and what is the total usable
render-pair count (the D-RENDER scale number)?

Method (per completed build):
  1. groups.jsonl (journal): every group carries exactly its final rendered
     candidates; classify groups by winner_confidence
     (normal / low / abstain / unannotated=null).
  2. shards index scan: groups/<build>/batch-*/indexes/*.idx.jsonl gives the
     candidate after-images (.jpg) and, for l-builds, per-candidate masks
     (.cgt.png) that actually landed in the tar shards.
  3. I_in availability: the journal source_path points at build-machine local
     paths; usability is checked against the NFS img banks
     (img/unknown/<pool>, matching on basename stem) or, for pools whose
     paths exist on this host (mmart_ppr10k, fivek_gold), os.path.exists.
  4. failures.jsonl: rendering-stage failure events, for the attrition
     ledger (candidates that never reached the journal).

Usable render pair := candidate whose after .jpg landed AND whose source
image is retrievable (bank hit or local path).

Outputs: experiments/tooling-wave1/data_splits/action_g_report.{json,md}
Usage:  python3 action_g_render_audit.py [--limit N] [--builds b1,b2]
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vr_common as C  # noqa: E402

CONF_CLASSES = ("normal", "low", "abstain", "unannotated")

# journal pool -> (img bank dirs, accepted roles)
BANK_FOR_POOL = {
    "unsplash": (["unsplash", "unsplash_work"], {"primary"}),
    "awards": (["awards"], {"primary"}),
    "quandian": (["quandian"], {"primary"}),
    "korean": (["korean"], {"primary"}),
    "greysky": (["greysky"], {"primary"}),
    "ppr10k": (["ppr10k"], {"source"}),
    "raise6k": (["raise6k"], {"preview", "raw"}),
    # fivek_gold bank uses the member file name as role and stores the exact
    # original path; matched on full source_path (see load_bank_stems)
    "fivek_gold": (["fivek_gold"], {"before.jpg"}),
}
FULL_PATH_BANKS = {"fivek_gold"}
LOCAL_PATH_POOLS = {"mmart_ppr10k"}


def stem(path: str) -> str:
    b = os.path.basename(path)
    return os.path.splitext(b)[0].lower()


_BANK_CACHE: dict[str, set[str]] = {}


def load_bank_stems(pools: set[str]) -> dict[str, set[str]]:
    """pool -> set of basename stems available in the img banks (cached)."""
    out: dict[str, set[str]] = {}
    for pool in pools:
        if pool not in BANK_FOR_POOL:
            continue
        if pool in _BANK_CACHE:
            out[pool] = _BANK_CACHE[pool]
            continue
        dirs, roles = BANK_FOR_POOL[pool]
        stems: set[str] = set()
        for d in dirs:
            meta = os.path.join(C.IMG_BANK_ROOT, d, "metadata.jsonl")
            if not os.path.isfile(meta):
                continue
            with open(meta) as f:
                for line in f:
                    r = json.loads(line)
                    if r.get("role") in roles:
                        sp = r.get("source_path", "")
                        stems.add(sp if pool in FULL_PATH_BANKS else stem(sp))
        _BANK_CACHE[pool] = stems
        out[pool] = stems
    return out


def _scan_idx_file(idx: str) -> tuple[set[str], set[str]]:
    jpg: set[str] = set()
    cgt: set[str] = set()
    with open(idx) as f:
        for line in f:
            r = json.loads(line)
            suf = r.get("suffix")
            if suf not in (".jpg", ".cgt.png"):
                continue
            sid = r["sample_id"]
            p = sid.rfind("candidate_")
            if p < 0:
                continue
            (jpg if suf == ".jpg" else cgt).add(sid[p:])
    return jpg, cgt


def landed_candidates(build: str) -> tuple[set[str], set[str]]:
    """(after .jpg candidate_ids, .cgt.png candidate_ids) landed in
    groups/<build>. Threaded: the l-builds have >2,000 small idx files on NFS
    and the scan is open-latency-bound."""
    jpg: set[str] = set()
    cgt: set[str] = set()
    pattern = os.path.join(C.DATASETS_ROOT, "groups", build, "batch-*", "indexes", "*.idx.jsonl")
    files = sorted(glob.glob(pattern))
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
        for j, c in ex.map(_scan_idx_file, files):
            jpg |= j
            cgt |= c
    return jpg, cgt


def render_failures(build: str) -> collections.Counter:
    out = collections.Counter()
    path = os.path.join(C.JOURNAL_ROOT, build, "failures.jsonl")
    if not os.path.isfile(path):
        return out
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("stage") == "rendering":
                out[r.get("error_code", "?")] += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--builds", type=str, default=None)
    args = ap.parse_args()

    builds = C.completed_builds()
    if args.builds:
        want = set(args.builds.split(","))
        builds = [b for b in builds if b in want]

    report: dict = {"builds": {}, "limit": args.limit}
    tot = collections.Counter()

    for b in builds:
        is_local = "-l" in b.split("global")[0] or "local" in b
        # pass 1: journal
        groups_by_conf = collections.Counter()
        cand_conf: dict[str, str] = {}
        cand_src: dict[str, str] = {}          # candidate_id -> source_id
        group_sources: dict[str, tuple[str, str]] = {}  # source_id -> (pool, path)
        pools_seen: set[str] = set()
        for g in C.iter_groups(b, limit=args.limit):
            conf = C.group_conf_class(g)
            groups_by_conf[conf] += 1
            pool = C.pool_of(g.get("source_path", ""))
            pools_seen.add(pool)
            group_sources[g["source_id"]] = (pool, g.get("source_path", ""))
            for c in g["candidates"]:
                cand_conf[c["candidate_id"]] = conf
                cand_src[c["candidate_id"]] = g["source_id"]

        # source availability
        banks = load_bank_stems(pools_seen)
        src_ok: dict[str, bool] = {}
        src_ok_by_pool = collections.Counter()
        src_n_by_pool = collections.Counter()
        for sid, (pool, path) in group_sources.items():
            if pool in banks:
                key = path if pool in FULL_PATH_BANKS else stem(path)
                ok = key in banks[pool]
            elif pool in LOCAL_PATH_POOLS:
                ok = os.path.exists(path)
            else:
                ok = False
            src_ok[sid] = ok
            src_n_by_pool[pool] += 1
            src_ok_by_pool[pool] += int(ok)

        cand_src_ok = {cid: src_ok[sid] for cid, sid in cand_src.items()}

        landed_jpg, landed_mask = landed_candidates(b)
        fails = render_failures(b)

        per_conf = {}
        for conf in CONF_CLASSES:
            cids = [cid for cid, cf in cand_conf.items() if cf == conf]
            n_land = sum(1 for cid in cids if cid in landed_jpg)
            n_mask = sum(1 for cid in cids if cid in landed_mask) if is_local else None
            n_usable = sum(1 for cid in cids if cid in landed_jpg and cand_src_ok[cid])
            per_conf[conf] = {
                "groups": groups_by_conf.get(conf, 0),
                "candidates": len(cids),
                "after_landed": n_land,
                "mask_landed": n_mask,
                "usable_pairs": n_usable,
            }
            tot[f"{conf}:candidates"] += len(cids)
            tot[f"{conf}:usable"] += n_usable
            tot["usable_total"] += n_usable
            tot["candidates_total"] += len(cids)

        report["builds"][b] = {
            "is_local": is_local,
            "groups_total": sum(groups_by_conf.values()),
            "per_confidence": per_conf,
            "journal_candidates": len(cand_conf),
            "landed_after_jpg": len(landed_jpg),
            "landed_after_not_in_journal": len(landed_jpg - set(cand_conf)),
            "landed_cgt_png": len(landed_mask) if is_local else None,
            "sources": {
                "distinct": len(group_sources),
                "i_in_available": sum(src_ok.values()),
                "by_pool": {p: [src_ok_by_pool[p], src_n_by_pool[p]] for p in sorted(src_n_by_pool)},
            },
            "render_failure_events": dict(fails.most_common()),
        }

    report["totals"] = {
        "candidates_total": tot["candidates_total"],
        "usable_render_pairs_total": tot["usable_total"],
        "usable_by_confidence": {c: tot[f"{c}:usable"] for c in CONF_CLASSES},
        "candidates_by_confidence": {c: tot[f"{c}:candidates"] for c in CONF_CLASSES},
        "abstain_usable_pairs": tot["abstain:usable"],
        "d_render_scale_note": "usable = after .jpg landed in groups dataset AND "
                               "source retrievable (img bank / local path)",
    }

    os.makedirs(C.REPORT_DIR, exist_ok=True)
    suffix = ".sample" if args.limit is not None else ""
    with open(os.path.join(C.REPORT_DIR, f"action_g_report{suffix}.json"), "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # markdown
    md = ["# 行动项 G：弃权组渲染产物落盘可用性核查\n"]
    md.append("| build | 组数 | journal 候选 | 落盘 after | 弃权组候选 | 弃权组可用对 | null 组候选 | null 组可用对 | I_in 可用源 |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for b, r in report["builds"].items():
        pc = r["per_confidence"]
        md.append(
            f"| {b} | {r['groups_total']} | {r['journal_candidates']} | {r['landed_after_jpg']} "
            f"| {pc['abstain']['candidates']} | {pc['abstain']['usable_pairs']} "
            f"| {pc['unannotated']['candidates']} | {pc['unannotated']['usable_pairs']} "
            f"| {r['sources']['i_in_available']}/{r['sources']['distinct']} |")
    t = report["totals"]
    md.append("")
    md.append(f"- **可用渲染对总数（D-RENDER 规模数）：{t['usable_render_pairs_total']:,}**"
              f"（journal 候选 {t['candidates_total']:,}）")
    md.append(f"- 其中弃权组（abstain）可用对：**{t['abstain_usable_pairs']:,}**；"
              f"按置信度：{json.dumps(t['usable_by_confidence'], ensure_ascii=False)}")
    md.append(f"- 口径：{t['d_render_scale_note']}")
    with open(os.path.join(C.REPORT_DIR, f"action_g_report{suffix}.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print(json.dumps(report["totals"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
