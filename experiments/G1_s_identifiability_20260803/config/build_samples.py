"""G1 采样器（DATA_ASSIGNMENT §3.1 G1 行）。

S-val 源 300 张（unsplash/awards/ppr10k 各 1/3），每源 3 指令：
  syn_a = SFT 行 instruction / syn_b = instruction_short / opp = 方向词取反。

指令来源：journal 归档 groups.jssonl 的原 source_path 已失效（/home/bc/data/datasets
旧盘已迁移）——经 img 银行回取（BankResolver，复用 tools/bgr_check/common.py，
与 T1 action_g_render_audit 同链），字节抽到 STAGE_DIR 下的平面文件供推理直读。
优先级：local 行 > global 行（区域指向类从 D-SFT-L 抽）；normal > low（待决策 D4）。
反义不可构造（无方向词）的源跳过并记账。确定性：seed=20260803。

输出：g1_samples.json + g1_sampling_report.json（同目录）+ STAGE_DIR 源图。
"""
from __future__ import annotations

import json
import random
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "readout"))
sys.path.insert(0, str(REPO / "tools" / "bgr_check"))
from ro9_gl_attention import make_antonym  # noqa: E402

from common import BankResolver, read_member  # noqa: E402  (tools/bgr_check)

STAGE_DIR = Path("/var/cache/veradata/g1_srcimg_20260803")
IMG_BANK = Path("/mnt/nfs/bc/data/datasets/img/unknown")


def resolve_ppr10k(source_path: str) -> dict | None:
    """ppr10k 银行的 member 后缀是 '.source.png'（多角色打包），BankResolver 的
    通用后缀匹配（.png/.jpg）取不到——本地补一个专用解析（full-path 精确匹配）。"""
    bank = IMG_BANK / "ppr10k"
    meta = bank / "metadata.jsonl"
    if not hasattr(resolve_ppr10k, "_cache"):
        by_full = {}
        with open(meta, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("role") == "source":
                    by_full[r.get("source_path", "")] = r["sample_id"]
        resolve_ppr10k._cache = by_full  # type: ignore[attr-defined]
    sid = resolve_ppr10k._cache.get(source_path)  # type: ignore[attr-defined]
    if sid is None:
        return None
    con = sqlite3.connect(bank / "indexes" / "catalog.sqlite3")
    try:
        cols = [r[1] for r in con.execute("PRAGMA table_info(members)")]
        ln = "length" if "length" in cols else "size"
        row = con.execute(
            f"SELECT shard, offset_data, {ln} FROM members "
            "WHERE sample_id=? AND suffix='.source.png'", (sid,)).fetchone()
    finally:
        con.close()
    if row is None:
        return None
    return {"kind": "bank", "tar": str(bank / "shards" / f"{row[0]}.tar"),
            "offset_data": int(row[1]), "length": int(row[2]),
            "bank": "ppr10k", "sample_id": sid, "suffix": ".png",
            "matched_by": "full_path"}

SPLITS_DB = REPO / "tools" / "data_splits" / "splits.sqlite3"
JOURNAL = Path("/var/cache/veradata/annot_review/journal-archive")
POOLS = ("unsplash", "awards", "ppr10k")
PER_POOL = 100
SEED = 20260803
CONF_RANK = {"normal": 0, "low": 1}
OUT = Path(__file__).parent / "g1_samples.json"
REPORT = Path(__file__).parent / "g1_sampling_report.json"


def main() -> None:
    con = sqlite3.connect(SPLITS_DB)
    val_pool = {sid: pool for sid, pool in con.execute(
        "SELECT source_id, pool FROM sources WHERE split='val' AND pool IN (?,?,?)",
        POOLS)}
    con.close()

    # source_id -> best sft row
    rows_by_src: dict[str, list[dict]] = defaultdict(list)
    builds = sorted(p.name for p in JOURNAL.iterdir()
                    if p.name.startswith("prod-") and (p / "sft.jsonl").is_file()
                    and (p / "groups.jsonl").is_file())
    for build in builds:
        g2src: dict[str, tuple[str, str]] = {}
        with open(JOURNAL / build / "groups.jsonl", encoding="utf-8") as f:
            for line in f:
                g = json.loads(line)
                sid = g.get("source_id")
                if sid in val_pool:
                    g2src[g["group_id"]] = (sid, g.get("source_path", ""))
        with open(JOURNAL / build / "sft.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                hit = g2src.get(r.get("group_id"))
                if hit is None:
                    continue
                sid, spath = hit
                instr, short = r.get("instruction"), r.get("instruction_short")
                if not instr or not short or instr == short:
                    continue
                rows_by_src[sid].append({
                    "build": build,
                    "task_type": r.get("task_type", ""),
                    "conf": r.get("winner_confidence", "low"),
                    "sft_id": r.get("sft_id"),
                    "instruction": instr,
                    "instruction_short": short,
                    "img_path": spath,
                })

    rng = random.Random(SEED)
    resolver = BankResolver()
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    samples, report = [], {"builds": builds, "per_pool": {}, "skipped": defaultdict(int)}
    for pool in POOLS:
        cands = sorted(sid for sid, p in val_pool.items() if p == pool and sid in rows_by_src)
        rng.shuffle(cands)
        picked = 0
        stats = {"val_total": sum(1 for p in val_pool.values() if p == pool),
                 "with_sft": len(cands), "no_antonym": 0, "no_img": 0, "picked": 0}
        for sid in cands:
            if picked >= PER_POOL:
                break
            rows = sorted(
                rows_by_src[sid],
                key=lambda r: (0 if r["task_type"] == "local" else 1,
                               CONF_RANK.get(r["conf"], 2)))
            row = rows[0]
            opp, n_swaps = make_antonym(row["instruction"])
            if opp is None:
                stats["no_antonym"] += 1
                continue
            # img 银行回取 -> STAGE_DIR 平面文件
            ref = (resolve_ppr10k(row["img_path"]) if pool == "ppr10k"
                   else resolver.resolve(pool, row["img_path"]))
            if ref is None:
                stats["no_img"] += 1
                continue
            if ref["kind"] == "local":
                staged = Path(ref["path"])
            else:
                staged = STAGE_DIR / f"{sid}{ref['suffix'].lower()}"
                if not staged.is_file():
                    staged.write_bytes(
                        read_member(ref["tar"], ref["offset_data"], ref["length"]))
            row = dict(row, img_path=str(staged))
            assert "__" not in sid, sid
            samples.append({
                "img_id": sid, "img_path": row["img_path"], "pool": pool,
                "build": row["build"], "sft_id": row["sft_id"],
                "task_type": row["task_type"], "winner_confidence": row["conf"],
                "n_antonym_swaps": n_swaps,
                "instructions": {"syn_a": row["instruction"],
                                 "syn_b": row["instruction_short"],
                                 "opp": opp},
            })
            picked += 1
        stats["picked"] = picked
        report["per_pool"][pool] = stats

    rng.shuffle(samples)  # 池间交错，冒烟前 N 个也均衡
    OUT.write_text(json.dumps(samples, ensure_ascii=False, indent=1))
    report["total"] = len(samples)
    report["conf_dist"] = dict(defaultdict(int))
    conf_dist: dict[str, int] = defaultdict(int)
    task_dist: dict[str, int] = defaultdict(int)
    for s in samples:
        conf_dist[s["winner_confidence"]] += 1
        task_dist[s["task_type"]] += 1
    report["conf_dist"] = dict(conf_dist)
    report["task_dist"] = dict(task_dist)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
