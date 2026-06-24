"""Fresh re-extract awards into a NEW scratch dir (the original _scratch/awards was
emptied by the 2026-06-22 migration, so _extract_awards.py's non-empty-dir skip
defeats recovery). Extract → recursively unnest → verify image validity (PIL).
Run: python -m dataset_build.source_qa._reextract_awards_v2
"""
from __future__ import annotations
import os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SRC = "/home/bc/retouching/presets/2008-2021 获奖摄影师作品合集"
DST = "/home/bc/data/datasets/_scratch/awards_v2"
ARCH = {".zip", ".rar", ".7z"}
IMG = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
WORKERS = 8


def _7z(a, o):
    os.makedirs(o, exist_ok=True)
    return subprocess.run(["7z", "x", "-y", "-bd", f"-o{o}", a],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def _nest(p: Path):
    out = p.parent / p.stem
    if out.is_dir() and any(out.iterdir()):
        p.unlink(missing_ok=True); return True
    if _7z(str(p), str(out)):
        p.unlink(missing_ok=True); return True
    p.rename(p.with_suffix(p.suffix + ".failed")); return False


def main():
    os.makedirs(DST, exist_ok=True)
    outer = [p for p in Path(SRC).rglob("*") if p.suffix.lower() in ARCH]
    print(f"[awards_v2] {len(outer)} outer archives", file=sys.stderr, flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(lambda p: _7z(str(p), str(Path(DST) / p.relative_to(SRC).parent / p.stem)), outer))
    rounds = 0
    while True:
        nested = [p for p in Path(DST).rglob("*") if p.suffix.lower() in ARCH and not p.name.endswith(".failed")]
        if not nested:
            break
        rounds += 1
        print(f"[awards_v2] round {rounds}: {len(nested)} nested", file=sys.stderr, flush=True)
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            if not any(ex.map(_nest, nested)):
                break
    # verify
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    imgs = [p for p in Path(DST).rglob("*") if p.suffix.lower() in IMG]
    valid = empty = corrupt = 0
    for p in imgs:
        if p.stat().st_size == 0:
            empty += 1; continue
        try:
            Image.open(p).verify(); valid += 1
        except Exception:
            corrupt += 1
    print(f"[awards_v2] DONE: {len(imgs)} imgs | valid={valid} empty={empty} corrupt={corrupt} "
          f"failed_archives={sum(1 for _ in Path(DST).rglob('*.failed'))}")


if __name__ == "__main__":
    main()
