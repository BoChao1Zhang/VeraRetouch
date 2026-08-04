#!/usr/bin/env bash
# =============================================================================
# Post-training durability step for main-agent ruling D-J1 (2026-08-05).
#
# Training writes to LOCAL disk. This script copies the deliverable artefacts to
# the NFS durable path and records BOTH sides' paths and sha256 digests into
#   experiments/Q3VL_metacanvas_where_what_20260804/s0_base_sft/checkpoint_sync_record.json
#
# Copied:
#   checkpoint-2488  (0.5 epoch, protected)
#   checkpoint-4976  (1.0 epoch, protected)
#   the final model + trainer state at the output_dir root
#   train.log / job.marker / run_setup.json
# Rolling (non-protected) checkpoints are NOT copied: they are transient by
# design (save_total_limit 3) and cost ~55 GiB each.
#
# RUN THIS ONLY AFTER TRAINING HAS FINISHED. Expect ~120 GiB over NFS at the
# measured ~103 MB/s, i.e. roughly 20 minutes.
#
#   bash SYNC_PROTECTED_TO_NFS.sh              # copy + verify + record
#   VERIFY_ONLY=1 bash SYNC_PROTECTED_TO_NFS.sh  # re-verify an existing copy
# =============================================================================
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/bc/VeraRetouch}"
LOCAL_DIR="${LOCAL_DIR:-/home/bc/data/runs/q3vl_base_sft_20260804}"
NFS_DIR="${NFS_DIR:-/mnt/nfs/bc/runs/q3vl_base_sft_20260804}"
RECORD="${RECORD:-${REPO_ROOT}/experiments/Q3VL_metacanvas_where_what_20260804/s0_base_sft/checkpoint_sync_record.json}"
PROTECTED_STEPS="${PROTECTED_STEPS:-2488 4976}"
VERIFY_ONLY="${VERIFY_ONLY:-0}"

[ -d "${LOCAL_DIR}" ] || { echo "no such local run dir: ${LOCAL_DIR}" >&2; exit 1; }

if [ ! -f "${LOCAL_DIR}/trainer_state.json" ]; then
  echo "REFUSING: ${LOCAL_DIR}/trainer_state.json is absent -- training has not" >&2
  echo "finished, so there is nothing durable to publish yet." >&2
  exit 1
fi

mkdir -p "${NFS_DIR}"

if [ "${VERIFY_ONLY}" != "1" ]; then
  for step in ${PROTECTED_STEPS}; do
    src="${LOCAL_DIR}/checkpoint-${step}"
    if [ ! -d "${src}" ]; then
      echo "WARNING: ${src} missing -- skipping" >&2
      continue
    fi
    echo "rsync checkpoint-${step} ..."
    rsync -a --info=progress2 "${src}" "${NFS_DIR}/"
  done
  echo "rsync root-level artefacts ..."
  rsync -a --info=progress2 --exclude 'checkpoint-*' "${LOCAL_DIR}/" "${NFS_DIR}/"
fi

echo "computing digests (both sides) ..."
/home/bc/envs/q3vl_sft/bin/python - "$LOCAL_DIR" "$NFS_DIR" "$RECORD" "$PROTECTED_STEPS" <<'PY'
import hashlib, json, os, subprocess, sys
from pathlib import Path

local, nfs, record, steps = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4].split()

def sha256(p, chunk=1 << 24):
    h = hashlib.sha256()
    with p.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def digest_tree(root: Path):
    """One digest per file, plus a single tree digest over (relpath, size, sha256)."""
    files, total = {}, 0
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(root))
            d = sha256(p)
            files[rel] = {"bytes": p.stat().st_size, "sha256": d}
            total += p.stat().st_size
    tree = hashlib.sha256()
    for rel, meta in sorted(files.items()):
        tree.update(f"{rel}\0{meta['bytes']}\0{meta['sha256']}\0".encode())
    return {"n_files": len(files), "total_bytes": total,
            "tree_sha256": tree.hexdigest(), "files": files}

out = {
    "ruling": "main-agent D-J1 (2026-08-05): train on local disk, publish protected "
              "checkpoints to the NFS durable path afterwards",
    "local_run_dir": str(local),
    "nfs_durable_dir": str(nfs),
    "protected_steps": [int(s) for s in steps],
    "artefacts": {},
    "all_match": True,
}
targets = [f"checkpoint-{s}" for s in steps] + ["__root__"]
for name in targets:
    if name == "__root__":
        lp, np_ = local, nfs
        ldig = {"n_files": 0, "total_bytes": 0, "tree_sha256": None, "files": {}}
        ndig = dict(ldig)
        # root-level files only (checkpoints handled separately)
        def root_digest(root: Path):
            files, total = {}, 0
            for p in sorted(root.iterdir()):
                if p.is_file():
                    files[p.name] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
                    total += p.stat().st_size
            tree = hashlib.sha256()
            for rel, meta in sorted(files.items()):
                tree.update(f"{rel}\0{meta['bytes']}\0{meta['sha256']}\0".encode())
            return {"n_files": len(files), "total_bytes": total,
                    "tree_sha256": tree.hexdigest(), "files": files}
        ldig, ndig = root_digest(lp), (root_digest(np_) if np_.exists() else None)
    else:
        lp, np_ = local / name, nfs / name
        if not lp.exists():
            out["artefacts"][name] = {"status": "absent_locally"}
            continue
        ldig = digest_tree(lp)
        ndig = digest_tree(np_) if np_.exists() else None
    match = bool(ndig) and ldig["tree_sha256"] == ndig["tree_sha256"]
    out["all_match"] = out["all_match"] and match
    out["artefacts"][name] = {
        "local_path": str(lp), "nfs_path": str(np_),
        "local": ldig, "nfs": ndig, "tree_digests_match": match,
    }

record.parent.mkdir(parents=True, exist_ok=True)
record.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: (v.get("tree_digests_match") if isinstance(v, dict) else v)
                  for k, v in out["artefacts"].items()}, indent=2))
print("all_match:", out["all_match"])
print("record written to", record)
sys.exit(0 if out["all_match"] else 2)
PY
