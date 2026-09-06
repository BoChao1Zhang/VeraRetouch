# 源自 experiments/prs/EPR-051_masked-restore-production/stage0/sprf/vlmsft/guards.py @ 2026-09-06（原件未入 git；原 sha256 见 veraretouch_sprf/PROVENANCE.md），逐字复制/仅改导入与路径解析（EPR-052）。
"""EPR-051 / SFT+ADAPT -- contamination guards.

G4   no held-out ID (snapshot_newdata_v3.heldout_ids.json) in train/val.
G4b  no key from the eval-only held-out CoT directory in train/val.

G4b exists because of the 2026-09-02 ruling that approved annotating ~1,464
held-out d6 samples for the R1 / oracle_text row.  Those labels are EVAL ONLY:
they carry GT CoT text for samples the SPRF held-out headline is computed on, so
a single one of them leaking into SFT or adapter training would make the R1 row
self-referential.  The guard reads the directory rather than trusting that the
annotation campaign stayed on its own side of the split, and it records the key
count so "0 intersection" can never be a silent consequence of reading nothing.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

EVAL_ONLY_DIR = Path(os.environ.get("VR_EVAL_ONLY_DIR", "/home/bc/data/builds/epr051_cot_heldout_eval"))  # EPR-052：可用环境变量覆盖，默认值不变
EXPECTED_CONTRACT = "eval_only_heldout"


def die(m: str):
    print(f"FATAL: {m}", flush=True)
    raise SystemExit(2)


def scan_eval_only(directory: Path = EVAL_ONLY_DIR) -> dict:
    """-> dict(present, dir, n_files, n_records, keys:set, ids:set, contracts:set)"""
    keys: set[str] = set()
    ids: set[str] = set()
    contracts: set[str] = set()
    files: list[str] = []
    n_records = 0
    if not directory.exists():
        return dict(present=False, dir=str(directory), n_files=0, n_records=0,
                    keys=keys, ids=ids, contracts=contracts, files=files)
    for p in sorted(directory.rglob("*.jsonl")):
        files.append(str(p))
        raw = p.read_bytes()
        cut = raw.rfind(b"\n")                      # ignore a half-written tail
        for line in raw[: cut + 1].split(b"\n"):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            n_records += 1
            if r.get("key"):
                keys.add(r["key"])
            if r.get("id"):
                ids.add(r["id"])
            if r.get("contract"):
                contracts.add(r["contract"])
    return dict(present=True, dir=str(directory), n_files=len(files),
                n_records=n_records, keys=keys, ids=ids, contracts=contracts,
                files=files)


def assert_no_eval_only_contamination(train_keys, val_keys, key_to_id,
                                      directory: Path = EVAL_ONLY_DIR) -> dict:
    """G4b.  Returns a provenance record; dies on any overlap."""
    scan = scan_eval_only(directory)
    used_keys = set(train_keys) | set(val_keys)
    used_ids = {key_to_id[k] for k in used_keys if k in key_to_id}

    bad_keys = sorted(used_keys & scan["keys"])
    bad_ids = sorted(used_ids & scan["ids"])
    if bad_keys:
        die(f"G4b eval-only CoT keys found in training data: {len(bad_keys)}, "
            f"e.g. {bad_keys[:5]}")
    if bad_ids:
        die(f"G4b eval-only CoT source ids found in training data: {len(bad_ids)}, "
            f"e.g. {bad_ids[:5]}")
    if scan["present"] and scan["contracts"] - {EXPECTED_CONTRACT}:
        die(f"G4b eval-only dir carries unexpected contracts: "
            f"{sorted(scan['contracts'])} (expected only {EXPECTED_CONTRACT!r})")

    rec = dict(guard="G4b", dir=scan["dir"], present=scan["present"],
               n_files=scan["n_files"], n_records=scan["n_records"],
               n_eval_only_keys=len(scan["keys"]),
               n_eval_only_ids=len(scan["ids"]),
               contracts=sorted(scan["contracts"]),
               key_intersection=0, id_intersection=0)
    if not scan["present"]:
        rec["note"] = ("eval-only directory does not exist yet; the intersection is "
                       "vacuously empty and MUST be re-asserted once it lands")
    print(f"[G4b] {json.dumps(rec)}", flush=True)
    return rec
