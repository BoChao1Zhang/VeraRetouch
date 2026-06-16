"""One-off: extract the '2008-2021 获奖摄影师作品合集' nested .zip/.rar archives
(~43GB, double-nested: outer archive -> inner archive -> images) into a flat
scratch tree the registry can scan. Parallel 7z, recursive until no archives
remain. Idempotent-ish (re-run skips already-extracted dirs that are non-empty).

Run: python -m dataset_build.source_qa._extract_awards
"""
from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SRC = "/home/bc/retouching/presets/2008-2021 获奖摄影师作品合集"
DST = "/home/bc/data/datasets/_scratch/awards"
ARCH = {".zip", ".rar", ".7z"}
IMG = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
WORKERS = 8


def _7z(archive: str, outdir: str) -> bool:
    os.makedirs(outdir, exist_ok=True)
    r = subprocess.run(["7z", "x", "-y", "-bd", f"-o{outdir}", archive],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return r.returncode == 0


def _extract_to_sibling(p: Path) -> bool:
    """Extract p into <parent>/<stem>/, delete p on success, .failed on failure."""
    out = p.parent / p.stem
    if out.is_dir() and any(out.iterdir()):
        try: p.unlink()
        except OSError: pass
        return True
    if _7z(str(p), str(out)):
        try: p.unlink()
        except OSError: pass
        return True
    try: p.rename(p.with_suffix(p.suffix + ".failed"))
    except OSError: pass
    return False


def main() -> None:
    os.makedirs(DST, exist_ok=True)
    # Phase A: outer archives mirrored into DST/<year>/<stem>/
    outer = [p for p in Path(SRC).rglob("*") if p.suffix.lower() in ARCH]
    print(f"[awards] {len(outer)} outer archives", file=sys.stderr)

    def do_outer(p):
        rel = p.relative_to(SRC)
        out = Path(DST) / rel.parent / p.stem
        if out.is_dir() and any(out.iterdir()):
            return True
        return _7z(str(p), str(out))

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(do_outer, outer))

    # Phase B: recursively extract nested archives in DST until none remain
    rounds = 0
    while True:
        nested = [p for p in Path(DST).rglob("*")
                  if p.suffix.lower() in ARCH and not p.name.endswith(".failed")]
        if not nested:
            break
        rounds += 1
        print(f"[awards] round {rounds}: {len(nested)} nested archives", file=sys.stderr)
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            results = list(ex.map(_extract_to_sibling, nested))
        if not any(results):
            print("[awards] no progress; stopping (remaining archives failed)", file=sys.stderr)
            break

    imgs = sum(1 for p in Path(DST).rglob("*") if p.suffix.lower() in IMG)
    failed = sum(1 for p in Path(DST).rglob("*.failed"))
    print(f"[awards] DONE: {imgs} images under {DST} ({failed} failed archives)")


if __name__ == "__main__":
    main()
