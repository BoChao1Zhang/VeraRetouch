"""Frozen-input provenance: the Where checkpoint must be the same for all arms.

    "The Where checkpoint is exactly the same and frozen for every What arm."
    -- protocol 6

Twelve arms start in four waves over several days (protocol 11's ``T1``-``C2``),
each as a separate process with the checkpoint path on the command line.  That is
precisely the shape of mistake nothing else would catch: a typo, a re-selected
Where checkpoint after a re-run, a stale symlink -- and the resulting board would
compare arms conditioned on two different Where models while every per-arm log
looked healthy.

So the digest is recorded, and every later arm checks it against every earlier
one.  Two rules, both hard stops (review blocker B-6):

1. **a declared digest must still hold** -- if a previous ``run_setup.json``
   recorded digest ``X`` and this arm's checkpoint hashes to ``Y``, stop;
2. **a declared digest must exist** -- if a previous run declared one and this arm
   cannot produce one at all, stop.  "We lost the provenance" is not a lesser
   failure than "the provenance disagrees".

The no-where controls ``C01``/``C02`` are *not* exempt: amendment A-3 makes the
frozen Where checkpoint's ``m_pred`` the supervision mask of all twelve arms, so
all twelve are conditioned on it and all twelve must agree about which one.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

__all__ = ["file_sha256", "collect_where_digests", "assert_where_consistency",
           "WhereProvenanceError"]

_CHUNK = 1 << 20


class WhereProvenanceError(RuntimeError):
    """Raised when two arms would run on different frozen Where checkpoints."""


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            block = fh.read(_CHUNK)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def collect_where_digests(run_root: str | Path, *, exclude_arm: str | None = None
                          ) -> dict[str, dict[str, Any]]:
    """``arm -> {digest, path, run_dir}`` from every published ``run_setup.json``.

    A malformed or partially written setup file is reported rather than skipped:
    a run that crashed before writing its provenance is exactly the run whose
    checkpoint nobody can vouch for.
    """
    out: dict[str, dict[str, Any]] = {}
    root = Path(run_root)
    if not root.exists():
        return out
    for setup_path in sorted(root.glob("*/run_setup.json")):
        try:
            setup = json.loads(setup_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            out[f"<unreadable:{setup_path.parent.name}>"] = {
                "digest": None, "path": str(setup_path), "error": str(exc)}
            continue
        arm = setup.get("arm") or setup_path.parent.name
        if exclude_arm is not None and arm == exclude_arm:
            continue
        where = setup.get("where") or {}
        out[arm] = {
            "digest": where.get("checkpoint_sha256"),
            "path": where.get("path"),
            "where_arm": where.get("where_arm"),
            "step": where.get("step"),
            "run_dir": str(setup_path.parent),
        }
    return out


def assert_where_consistency(run_root: str | Path, arm: str,
                             digest: str | None, checkpoint: str | None = None
                             ) -> dict[str, Any]:
    """Hard stop when this arm's frozen Where checkpoint differs from a previous one.

    Returns the provenance record that belongs in ``run_setup.json``.
    """
    previous = collect_where_digests(run_root, exclude_arm=arm)
    unreadable = [k for k in previous if k.startswith("<unreadable:")]
    declared = {a: v for a, v in previous.items()
                if not a.startswith("<unreadable:") and v.get("digest")}

    if unreadable:
        raise WhereProvenanceError(
            f"{arm}: cannot read the provenance of {unreadable}.  Protocol 6 needs "
            "every arm to be on one frozen Where checkpoint, and an unreadable "
            "run_setup.json means one arm's conditioning is unaccounted for.  "
            "Repair or remove those run directories before starting another arm."
        )
    if declared and not digest:
        raise WhereProvenanceError(
            f"{arm}: {sorted(declared)} already declared a frozen Where checkpoint "
            f"digest, but this arm has none.  Protocol 6 forbids running an arm on "
            "an unverifiable Where checkpoint, and amendment A-3 makes the frozen "
            "m_pred part of the loss for every arm including C01-C04."
        )
    mismatched = {a: v["digest"] for a, v in declared.items() if v["digest"] != digest}
    if digest and mismatched:
        lines = "\n".join(
            f"  {a}: {d}  ({previous[a].get('path')})" for a, d in sorted(mismatched.items()))
        raise WhereProvenanceError(
            f"{arm}: frozen Where checkpoint digest {digest} "
            f"({checkpoint}) disagrees with previously started arms:\n{lines}\n"
            "Protocol 6: 'the Where checkpoint is exactly the same and frozen for "
            "every What arm'.  Refusing to start -- a board built from two Where "
            "checkpoints is not the experiment the protocol defines."
        )
    return {
        "checkpoint_sha256": digest,
        "path": checkpoint,
        "n_previous_arms_checked": len(declared),
        "previous_arms": sorted(declared),
        "consistent": True,
    }
