"""G2: 从 journal 归档 + shard 索引装配 D-SFT-L / D-SFT-G 的 S-val 条目清单。

流程（避免逐样本 tar 读 vrmeta，那样 50 万样本要跑几小时）：
  1) journal `groups.jsonl` 逐行先用正则取 source_id → 只对 S-split==val 的行做
     完整 json.loads（S-split 是 source_id 的纯哈希函数，等价于读 T1 旁表；
     旁表另做一致性核对，见 --verify-side-table）。
  2) journal `sft.jsonl` 给出 D-SFT-L/G 的**行定义**（candidate_id + winner_confidence
     + instruction + local.C_GT），据此筛 normal 置信度。
  3) `rg -F -f <candidate_ids>` 一次扫过 build 的 *.idx.jsonl 取成员 ranged-read 引用。

输出 JSONL：每行一个候选，含 source_path / pool / 成员引用 / 掩膜引用。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/home/bc/VeraRetouch/tools/bgr_check")
import common as C  # noqa: E402  (BankResolver / pool_of / preprocess_source_bytes)

JOURNAL_ROOT = Path("/var/cache/veradata/annot_review/journal-archive")
SFT_ROOT = Path("/mnt/nfs/bc/data/datasets/sft")
SPLIT_SEED = "verasplit-v1"

L_BUILDS = [
    "prod-l1-local17k-20260731", "prod-l2-local17k-20260731",
    "prod-l3-local17k-20260731", "prod-l4-local17k-20260801",
    "prod-l5-local17k-20260801", "prod-l6-local17k-20260801",
]
G_BUILDS = [
    "prod-g1-global25k-20260731", "prod-g2-global25k-20260731",
    "prod-g3-global25k-20260801", "prod-g4-global25k-20260801",
]

_SRC_RE = re.compile(rb'"source_id"\s*:\s*"([^"]+)"')


def s_split(source_id: str) -> str:
    h = hashlib.sha1(f"{SPLIT_SEED}:{source_id}".encode()).hexdigest()
    b = int(h[:8], 16) % 100
    return "train" if b < 90 else ("val" if b < 95 else "test")


# --------------------------------------------------------------------------- journal

def scan_groups(build: str, want_split: str = "val") -> dict[str, dict]:
    """candidate_id -> group-level info，只保留 S-split==want_split 的组。"""
    path = JOURNAL_ROOT / build / "groups.jsonl"
    out: dict[str, dict] = {}
    with open(path, "rb") as f:
        for raw in f:
            m = _SRC_RE.search(raw)
            if m is None:
                continue
            sid = m.group(1).decode()
            if s_split(sid) != want_split:
                continue
            g = json.loads(raw)
            base = {
                "build": build, "group_id": g.get("group_id"),
                "source_id": g.get("source_id"), "source_path": g.get("source_path"),
                "render_mode": g.get("render_mode", ""),
                "pool": C.pool_of(str(g.get("source_path", ""))),
            }
            for c in g.get("candidates", []) or []:
                cid = c.get("candidate_id")
                if not cid:
                    continue
                rec = dict(base)
                rec["candidate_id"] = cid
                rec["preset_id"] = (c.get("recipe") or {}).get("preset_id", "")
                rec["rank"] = c.get("rank")
                out[cid] = rec
    return out


def scan_sft(build: str) -> dict[str, dict]:
    """candidate_id -> SFT 行信息（D-SFT-L/G 的行定义）。"""
    path = JOURNAL_ROOT / build / "sft.jsonl"
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            cid = d.get("candidate_id")
            if not cid:
                continue
            loc = d.get("local") or {}
            out[cid] = {
                "winner_confidence": d.get("winner_confidence"),
                "task_type": d.get("task_type"),
                "instruction": d.get("instruction"),
                "mask_id": loc.get("mask_id"),
                "region": loc.get("region"),
                "mask_area": (loc.get("subject") or {}).get("area"),
                "amount": loc.get("amount"),
            }
    return out


# --------------------------------------------------------------------------- shard refs

def find_members(build: str, candidate_ids: list[str]) -> dict[str, dict]:
    """candidate_id -> {suffix: ref}，一次 rg -F 扫 build 的 *.idx.jsonl。"""
    root = SFT_ROOT / build
    out: dict[str, dict] = {}
    if not candidate_ids:
        return out
    wanted = set(candidate_ids)
    with tempfile.NamedTemporaryFile("w", suffix=".pats", delete=False) as tf:
        tf.write("\n".join(candidate_ids) + "\n")
        pats = tf.name
    try:
        proc = subprocess.run(
            ["rg", "-I", "--no-heading", "-F", "-f", pats, "-j", "16",
             "-g", "*.idx.jsonl", str(root)],
            capture_output=True, text=True, check=False)
        for line in proc.stdout.splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = row.get("sample_id", "")
            pos = sid.rfind("candidate_")
            if pos < 0:
                continue
            cid = sid[pos:]
            if cid not in wanted:
                continue
            parts = row["logical_path"].split("/")
            batch = parts[2]
            tar = str(root / batch / "shards" / (row["shard"] + ".tar"))
            out.setdefault(cid, {})[row["suffix"]] = {
                "tar": tar, "offset_data": int(row["offset_data"]),
                "length": int(row["length"]), "sha256": row.get("sha256", ""),
                "sample_id": sid,
            }
    finally:
        os.unlink(pats)
    return out


# --------------------------------------------------------------------------- 主流程

def build_index(builds: list[str], line: str, need_mask: bool,
                want_confidence: str | None, per_build_cap: int | None,
                one_per_group: bool, want_split: str = "val") -> tuple[list[dict], dict]:
    resolver = C.BankResolver()
    rows: list[dict] = []
    stats: dict = {"per_build": {}, "skip": {}}

    def skip(reason: str) -> None:
        stats["skip"][reason] = stats["skip"].get(reason, 0) + 1

    for build in builds:
        groups = scan_groups(build, want_split)
        sft = scan_sft(build)
        cands = []
        seen_groups: set[str] = set()
        for cid, g in groups.items():
            s = sft.get(cid)
            if s is None:
                skip("not_sft_row")
                continue
            if want_confidence is not None and s["winner_confidence"] != want_confidence:
                skip(f"conf={s['winner_confidence']}")
                continue
            if one_per_group and g["group_id"] in seen_groups:
                skip("dup_group")
                continue
            seen_groups.add(str(g["group_id"]))
            cands.append((cid, g, s))
        cands.sort(key=lambda t: t[0])          # 确定序
        if per_build_cap is not None:
            cands = cands[:per_build_cap * 3]    # 留冗余给后续 skip

        members = find_members(build, [c[0] for c in cands])
        kept = 0
        for cid, g, s in cands:
            if per_build_cap is not None and kept >= per_build_cap:
                break
            mem = members.get(cid)
            if not mem:
                skip("member_missing")
                continue
            tar_ref = mem.get(".jpg") or mem.get(".png")
            mask_ref = mem.get(".cgt.png")
            if tar_ref is None:
                skip("after_missing")
                continue
            if need_mask and mask_ref is None:
                skip("cgt_missing")
                continue
            src = resolver.resolve(g["pool"], str(g["source_path"]))
            if src is None:
                skip(f"source_unresolved:{g['pool']}")
                continue
            rows.append({
                "line": line, "build": build, "group_id": g["group_id"],
                "source_id": g["source_id"], "source_path": g["source_path"],
                "pool": g["pool"], "render_mode": g["render_mode"],
                "candidate_id": cid, "preset_id": g["preset_id"], "rank": g["rank"],
                "winner_confidence": s["winner_confidence"],
                "task_type": s["task_type"], "instruction": s["instruction"],
                "mask_id": s["mask_id"], "region": s["region"],
                "mask_area": s["mask_area"], "amount": s["amount"],
                "split": want_split,
                "tar_ref": tar_ref, "mask_ref": mask_ref, "source_ref": src,
            })
            kept += 1
        stats["per_build"][build] = {"val_groups": len(seen_groups),
                                     "candidates": len(cands), "kept": kept}
        print(f"[index] {build}: val_groups={len(seen_groups)} cands={len(cands)} kept={kept}",
              flush=True)
    stats["total"] = len(rows)
    return rows, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--line", choices=["l", "g"], required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--split", default="val")
    ap.add_argument("--confidence", default="normal")
    ap.add_argument("--per-build-cap", type=int, default=None)
    ap.add_argument("--builds", nargs="*", default=None)
    args = ap.parse_args()

    builds = args.builds or (L_BUILDS if args.line == "l" else G_BUILDS)
    rows, stats = build_index(
        builds, args.line, need_mask=(args.line == "l"),
        want_confidence=(None if args.confidence == "any" else args.confidence),
        per_build_cap=args.per_build_cap, one_per_group=True, want_split=args.split)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(str(args.out) + ".stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps({"n": len(rows), "out": str(args.out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
