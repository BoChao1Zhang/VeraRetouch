#!/usr/bin/env python3
"""Migrate + sequentially-rename every source corpus into ~/data/datasets.

Goal (see plan home-bc-retouching-img-crispy-treasure.md): consolidate the
scattered source pools (retouching/presets, monetGPT/data, /home/bc/datasets)
under a single ``/home/bc/data/datasets/<corpus>/`` tree, renaming every file to
``<corpus>_<NNNNNN>.<ext>``.

Key invariant: the unit of renaming is a *pair group* (one original image plus
all of its paired sub-assets), NOT a single file. Every asset in a group gets
the SAME sequence number, so ppr10k ``source / target_{a,b,c} / xmp / mask`` stay
aligned and a fivek_gold sample's 4 expert dirs share one base number.

Everything lives on the same ``/home`` mount, so moves are zero-cost
``rename(2)`` metadata ops (verified per-asset via ``st_dev``). Exceptions that
genuinely cost disk: fivek raw (extracted from a tar) and RAISE (downloaded
separately by download_raise.py). awards are moved from the already-extracted
``_scratch/awards`` tree.

Idempotent + resumable: a per-dataset manifest (JSONL) + state file act as a
ledger. A group is written ``pending`` before the move and flipped to ``moved``
after; a re-run skips groups already ``moved`` and rolls back a half-moved group.

Usage:
    python -m dataset_build.tools.migrate_datasets plan                 # dry-run, write planned manifests
    python -m dataset_build.tools.migrate_datasets apply --datasets ppr10k,unsplash
    python -m dataset_build.tools.migrate_datasets apply                # all (except fivek raw unless --extract-fivek)
    python -m dataset_build.tools.migrate_datasets verify
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

# Reuse the registry's path helpers / filters so migration and scanning agree.
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))  # repo root, so `dataset_build` imports
from dataset_build import registry as reg  # noqa: E402

DEST_ROOT = "/home/bc/data/datasets"
RETOUCH_PRESETS = "/home/bc/retouching/presets"
MONET_DATA = "/home/bc/retouching/monetGPT/data"
FIVEK_TAR = "/home/bc/datasets/fivek_dataset.tar"
FIVEK_RAW_GLOB_RE = re.compile(r"raw_photos/HQa[^/]+/photos/[^/]+\.dng$", re.IGNORECASE)

IMAGE_EXTS = reg.IMAGE_EXTS
AWARD_IMG_EXTS = IMAGE_EXTS | {".tif", ".tiff", ".webp", ".bmp"}
RECIPE_EXTS = {".xmp", ".lrtemplate", ".cube", ".3dl"}
SEQ_WIDTH = 6


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Asset:
    role: str          # e.g. "source", "target_a", "mask_360p", "xmp", "dir"
    src: str           # absolute source path (file or dir)
    sub: str           # destination subdir under the dataset root ("" = root)
    ext: str = ""      # output extension incl. dot; "" keeps the source ext / dir
    is_dir: bool = False


@dataclass
class Group:
    orig_id: str               # stable, unique-per-dataset key used for idempotency
    assets: List[Asset]
    pair_kind: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Dataset:
    name: str                  # corpus / manifest name
    prefix: str                # filename prefix => <prefix>_<NNNNNN>
    root: str                  # dataset root relative to DEST_ROOT
    planner: Callable[[], Iterator[Group]]


# --------------------------------------------------------------------------- #
# Planners — each yields pair groups for one corpus
# --------------------------------------------------------------------------- #
def _exists(p: str) -> bool:
    try:
        return os.path.exists(p)
    except OSError:
        return False


def plan_ppr10k() -> Iterator[Group]:
    root = Path(MONET_DATA) / "ppr10k"
    src_dir = root / "source"
    if not src_dir.is_dir():
        return
    mask_360 = root / "masks" / "360p" / "masks_360p"
    mask_full = root / "masks" / "full" / "masks_full"
    id_map = root / "manifests" / "id_map.csv"
    id_to_base: Dict[str, str] = {}
    if id_map.is_file():
        with open(id_map, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                nid = (row.get("new_id") or "").strip()
                ob = (row.get("orig_base") or "").strip()
                if nid and ob:
                    id_to_base[nid] = ob
    for src_png in sorted(src_dir.glob("*.png")):
        sid = src_png.stem
        orig_base = id_to_base.get(sid)
        if not orig_base:
            continue
        assets = [Asset("source", str(src_png), "source", ".png")]
        for e in ("a", "b", "c"):
            t = root / f"target_{e}" / f"{sid}.png"
            x = root / "xmp" / f"target_{e}" / f"{sid}.xmp"
            if t.is_file():
                assets.append(Asset(f"target_{e}", str(t), f"target_{e}", ".png"))
            if x.is_file():
                assets.append(Asset(f"xmp_target_{e}", str(x), f"xmp/target_{e}", ".xmp"))
        xsrc = root / "xmp" / "source" / f"{sid}.xmp"
        if xsrc.is_file():
            assets.append(Asset("xmp_source", str(xsrc), "xmp/source", ".xmp"))
        m360 = mask_360 / f"{orig_base}.png"
        if m360.is_file():
            assets.append(Asset("mask_360p", str(m360), "masks/360p", ".png"))
        mfull = mask_full / f"{orig_base}.png"
        if mfull.is_file():
            assets.append(Asset("mask_full", str(mfull), "masks/full", ".png"))
        yield Group(orig_id=sid, assets=assets, pair_kind="ppr10k",
                    meta={"orig_id": sid, "orig_base": orig_base})


def plan_fivek_gold() -> Iterator[Group]:
    """20000 dirs <base>_<A|B|D|E>; group by base, move the 4 expert dirs whole."""
    root = Path(MONET_DATA) / "fivek_mmart_like" / "train_global"
    if not root.is_dir():
        return
    by_base: Dict[str, List[Path]] = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        m = re.match(r"^(.*)_([A-E])$", d.name)
        base = m.group(1) if m else d.name
        by_base.setdefault(base, []).append(d)
    for base in sorted(by_base):
        assets = []
        for d in sorted(by_base[base]):
            m = re.match(r"^(.*)_([A-E])$", d.name)
            expert = m.group(2) if m else "X"
            assets.append(Asset(f"dir_{expert}", str(d), "train_global", f"_{expert}", is_dir=True))
        yield Group(orig_id=base, assets=assets, pair_kind="fivek_gold",
                    meta={"base_name": base})


def plan_fivek_raw(extract: bool) -> Iterator[Group]:
    """5000 DNG inside fivek_dataset.tar. Extract member -> fivek5k/raw/<seq>.dng."""
    if not extract:
        return
    if not _exists(FIVEK_TAR):
        return
    with tarfile.open(FIVEK_TAR, "r") as tf:
        members = [m for m in tf.getmembers()
                   if m.isfile() and FIVEK_RAW_GLOB_RE.search(m.name)]
    for m in sorted(members, key=lambda x: x.name):
        stem = Path(m.name).stem
        yield Group(orig_id=stem,
                    assets=[Asset("dng", "tar://" + m.name, "raw", ".dng")],
                    pair_kind="fivek_raw",
                    meta={"orig_stem": stem, "tar_member": m.name})


def plan_unsplash() -> Iterator[Group]:
    img_dir = Path(DEST_ROOT) / "unsplash-lite" / "images" / "original"
    if not img_dir.is_dir():
        return
    for f in sorted(img_dir.iterdir()):
        if f.is_file() and reg._ext(f) in IMAGE_EXTS:
            yield Group(orig_id=f.stem,
                        assets=[Asset("image", str(f), "images", reg._ext(f))],
                        pair_kind="single", meta={"unsplash_id": f.stem})


def plan_raise() -> Iterator[Group]:
    raw = Path(DEST_ROOT) / "RAISE-6k" / "raw"
    if not raw.is_dir():
        return
    for f in sorted(raw.iterdir()):
        if f.is_file() and f.suffix.lower() == ".nef" and not f.name.startswith("RAISE-6k_"):
            yield Group(orig_id=f.stem,
                        assets=[Asset("nef", str(f), "raw", ".nef")],
                        pair_kind="single", meta={"raise_id": f.stem})


# GREYSKY collection layout: <collection>/{DNG 原图参数文件, JPG 预览文件, XMP 预设文件}/<stem>.{dng,jpg,xmp}
_GREYSKY_JPG_DIR = "JPG 预览文件"   # expert preview == the gold 'after'
_GREYSKY_XMP_DIR = "XMP 预设文件"   # the expert preset


def _greysky_siblings(dng: Path):
    """Resolve a GREYSKY DNG's sibling expert JPG (gold 'after') and XMP (preset)
    by stem within its collection, tolerating CJK/whitespace in folder names."""
    col = dng.parent.parent

    def _norm(p: Path) -> str:
        return re.sub(r"\s+", "", p.stem).lower()

    target = _norm(dng)

    def _find(subdir: str, ext: str):
        d = col / subdir
        if not d.is_dir():
            return None
        exact = d / f"{dng.stem}{ext}"
        if exact.is_file():
            return str(exact)
        try:
            for f in d.iterdir():
                if f.is_file() and f.suffix.lower() == ext and _norm(f) == target:
                    return str(f)
        except OSError:
            pass
        return None

    return _find(_GREYSKY_JPG_DIR, ".jpg"), _find(_GREYSKY_XMP_DIR, ".xmp")


def plan_greysky() -> Iterator[Group]:
    root = Path(RETOUCH_PRESETS) / "GREYSKY 老斯基的预设专车"
    if not root.is_dir():
        return
    for f in reg._iter_files(root):
        if reg._ext(f) != ".dng":
            continue
        sz = reg._safe_stat_size(f)
        if sz is not None and sz < 2_000_000:   # preset-dng, not a genuine raw
            continue
        jpg, xmp = _greysky_siblings(f)
        assets = [Asset("dng", str(f), "raw", ".dng")]
        if jpg:
            assets.append(Asset("preview", jpg, "preview", ".jpg"))
        if xmp:
            assets.append(Asset("xmp", xmp, "xmp", ".xmp"))
        yield Group(orig_id=str(f.relative_to(root)), assets=assets, pair_kind="greysky",
                    meta={"collection": f.parent.parent.name, "orig_stem": f.stem})


def plan_korean() -> Iterator[Group]:
    root = _glob_one(RETOUCH_PRESETS, "211*")
    if root is None:
        return
    min_bytes = reg._DEFAULT_MIN_SOURCE_BYTES
    for f in reg._iter_files(root):
        if reg._ext(f) not in IMAGE_EXTS:
            continue
        if _is_junk(f, root):
            continue
        sz = reg._safe_stat_size(f)
        if sz is not None and sz < min_bytes:
            continue
        yield Group(orig_id=str(f.relative_to(root)),
                    assets=[Asset("image", str(f), "", reg._ext(f))],
                    pair_kind="single", meta={"orig_rel": str(f.relative_to(root))})


def plan_quandian_sources() -> Iterator[Group]:
    root = Path(RETOUCH_PRESETS) / "全店素材"
    if not root.is_dir():
        return
    min_bytes = reg._DEFAULT_MIN_SOURCE_BYTES
    for f in reg._iter_files(root):
        ext = reg._ext(f)
        if _is_junk(f, root):
            continue
        pack_id = reg.SourceRegistry._pack_id_for(f, root)
        if ext in IMAGE_EXTS:
            sz = reg._safe_stat_size(f)
            if sz is None or sz < min_bytes:
                continue
            yield Group(orig_id=str(f.relative_to(root)),
                        assets=[Asset("image", str(f), "", ext)],
                        pair_kind="single",
                        meta={"pack_id": pack_id, "orig_rel": str(f.relative_to(root))})
        elif ext in (".cr2", ".arw"):
            yield Group(orig_id=str(f.relative_to(root)),
                        assets=[Asset("raw", str(f), "", ext)],
                        pair_kind="single",
                        meta={"pack_id": pack_id, "orig_rel": str(f.relative_to(root)), "raw": True})


def plan_awards() -> Iterator[Group]:
    root = Path(DEST_ROOT) / "_scratch" / "awards"
    if not root.is_dir():
        return
    for f in reg._iter_files(root):
        if reg._ext(f) not in AWARD_IMG_EXTS:
            continue
        rel = f.relative_to(root).parts
        year = rel[0] if rel and rel[0].isdigit() else None
        yield Group(orig_id=str(f.relative_to(root)),
                    assets=[Asset("image", str(f), "", reg._ext(f))],
                    pair_kind="single", meta={"award_year": year, "orig_rel": str(f.relative_to(root))})


def _recipe_planner(tree: str, exts: set) -> Callable[[], Iterator[Group]]:
    def _plan() -> Iterator[Group]:
        root = Path(tree)
        if not root.is_dir():
            return
        for f in reg._iter_files(root):
            if reg._ext(f) not in exts:
                continue
            if _is_junk(f, root):
                continue
            yield Group(orig_id=str(f.relative_to(root)),
                        assets=[Asset("recipe", str(f), "", reg._ext(f))],
                        pair_kind="recipe", meta={"orig_rel": str(f.relative_to(root))})
    return _plan


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _glob_one(parent: str, pattern: str) -> Optional[Path]:
    p = Path(parent)
    if not p.is_dir():
        return None
    matches = sorted(p.glob(pattern))
    return matches[0] if matches else None


_JUNK_PATH_RE = re.compile(reg._DEFAULT_JUNK_PATH_RE, re.IGNORECASE)


def _is_junk(path: Path, root: Path) -> bool:
    if reg._ext(path) in reg._DEFAULT_JUNK_EXT:
        return True
    try:
        rel = str(path.relative_to(root))
    except ValueError:
        rel = str(path)
    return bool(_JUNK_PATH_RE.search(rel))


def _same_device(a: str, b_parent: str) -> bool:
    try:
        return os.stat(a).st_dev == os.stat(b_parent).st_dev
    except OSError:
        return False


# --------------------------------------------------------------------------- #
# Executor
# --------------------------------------------------------------------------- #
class Migrator:
    def __init__(self, dest_root: str, dry_run: bool, seq_width: int, resume: bool):
        self.dest_root = dest_root
        self.dry_run = dry_run
        self.seq_width = seq_width
        self.resume = resume
        self.mig_dir = Path(dest_root) / "_migration"
        self.manifest_dir = self.mig_dir / "manifests"
        self.state_path = self.mig_dir / "migrate_state.json"
        self.log_path = self.mig_dir / "migrate.log"
        if not dry_run:
            self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        if self.resume and self.state_path.is_file():
            try:
                return json.loads(self.state_path.read_text())
            except (OSError, json.JSONDecodeError):
                pass
        return {}

    def _save_state(self) -> None:
        if self.dry_run:
            return
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=False))
        os.replace(tmp, self.state_path)

    def _log(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
        print(line, flush=True)
        if not self.dry_run:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def _build_plan(self, ds: Dataset, ds_root: Path) -> List[Dict[str, Any]]:
        """Return the immutable seq->assets plan for a dataset, resuming/extending
        any existing manifest. Seq numbers, once assigned, never change — so a crash
        mid-move can't desync seq<->orig (we re-read the manifest, never re-enumerate
        sources that may already be moved away)."""
        existing: List[Dict[str, Any]] = []
        seen_orig: set = set()
        max_seq = 0
        real_manifest = self.manifest_dir / f"{ds.name}.jsonl"
        if not self.dry_run and self.resume and real_manifest.is_file():
            with open(real_manifest, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    existing.append(rec)
                    seen_orig.add(rec["orig_id"])
                    max_seq = max(max_seq, int(rec["seq"]))
        # Enumerate (new) groups and assign continuing seq numbers.
        new_recs: List[Dict[str, Any]] = []
        counter = max_seq
        for grp in ds.planner():
            if grp.orig_id in seen_orig:
                continue
            counter += 1
            new_id = f"{ds.prefix}_{counter:0{self.seq_width}d}"
            assets = [{"role": a.role, "src": a.src, "is_dir": a.is_dir,
                       "dst": str(ds_root / a.sub / f"{new_id}{a.ext}")}
                      for a in grp.assets]
            new_recs.append({"seq": counter, "new_id": new_id, "dataset": ds.name,
                             "orig_id": grp.orig_id, "pair_kind": grp.pair_kind,
                             "meta": grp.meta, "assets": assets})
        # Durably append the new plan records BEFORE any move (apply mode).
        if not self.dry_run and new_recs:
            with open(real_manifest, "a", encoding="utf-8") as f:
                for rec in new_recs:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return existing + new_recs

    def run_dataset(self, ds: Dataset) -> Dict[str, int]:
        ds_root = Path(self.dest_root) / ds.root
        plan = self._build_plan(ds, ds_root)
        counts = {"planned": len(plan), "moved": 0, "skip": 0, "fail": 0, "missing_src": 0}

        if self.dry_run:
            plan_p = self.mig_dir / "manifests_dryrun"
            plan_p.mkdir(parents=True, exist_ok=True)
            with open(plan_p / f"{ds.name}.jsonl", "w", encoding="utf-8") as f:
                for rec in plan:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._log(f"[{ds.name}] " + " ".join(f"{k}={v}" for k, v in counts.items() if v))
            return counts

        st = self.state.setdefault(ds.name, {"done_seqs": []})
        done = set(st["done_seqs"])
        flush_every, since_flush = 200, 0
        for rec in plan:
            if rec["seq"] in done:
                counts["skip"] += 1
                continue
            assets = rec["assets"]
            # A source that is neither present nor already at dst can't be moved.
            if any(not a["src"].startswith("tar://") and not _exists(a["src"]) and not _exists(a["dst"])
                   for a in assets):
                counts["missing_src"] += 1
                continue
            moved: List[tuple] = []
            try:
                for a in assets:
                    self._move_asset(a)
                    moved.append(a)
                done.add(rec["seq"])
                counts["moved"] += 1
                since_flush += 1
                if since_flush >= flush_every:
                    st["done_seqs"] = sorted(done)
                    self._save_state()
                    since_flush = 0
            except Exception as exc:  # noqa: BLE001 - rollback then record
                for a in reversed(moved):
                    try:
                        if a["src"].startswith("tar://"):
                            if os.path.exists(a["dst"]):
                                os.remove(a["dst"])
                        else:
                            os.replace(a["dst"], a["src"])
                    except OSError:
                        pass
                counts["fail"] += 1
                self._log(f"[{ds.name}] FAIL seq={rec['seq']} orig={rec['orig_id']}: {exc}")
        st["done_seqs"] = sorted(done)
        self._save_state()
        self._log(f"[{ds.name}] " + " ".join(f"{k}={v}" for k, v in counts.items() if v))
        return counts

    def _move_asset(self, a: Dict[str, Any]) -> None:
        self._move(Asset(role=a["role"], src=a["src"], sub="", ext="", is_dir=a["is_dir"]), a["dst"])

    def _tar_member(self, name: str):
        """Open the fivek tar once + build a name->TarInfo index (one full scan),
        so each extract is an O(1) lookup instead of re-scanning the 50GB archive."""
        idx = getattr(self, "_tar_index", None)
        if idx is None:
            tf = tarfile.open(FIVEK_TAR, "r")
            self._tarfile = tf
            idx = {m.name: m for m in tf.getmembers()}
            self._tar_index = idx
        ti = idx.get(name)
        if ti is None:
            return None
        return self._tarfile.extractfile(ti)

    def _move(self, a: Asset, dst: str) -> None:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            return  # idempotent: already in place
        if a.src.startswith("tar://"):
            member = a.src[len("tar://"):]
            src_f = self._tar_member(member)
            if src_f is None:
                raise IOError(f"cannot extract {member}")
            tmp = dst + ".part"
            with open(tmp, "wb") as out:
                while True:
                    chunk = src_f.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
            os.replace(tmp, dst)
            return
        if not _same_device(a.src, os.path.dirname(dst)):
            raise IOError(f"cross-device move refused: {a.src} -> {dst}")
        os.rename(a.src, dst)


# --------------------------------------------------------------------------- #
# Dataset registry
# --------------------------------------------------------------------------- #
def build_datasets(extract_fivek: bool) -> List[Dataset]:
    return [
        Dataset("ppr10k", "ppr10k", "ppr10k", plan_ppr10k),
        Dataset("fivek_gold", "fivek_gold", "fivek_gold", plan_fivek_gold),
        Dataset("fivek", "fivek", "fivek5k", lambda: plan_fivek_raw(extract_fivek)),
        Dataset("unsplash", "unsplash", "unsplash-lite", plan_unsplash),
        Dataset("raise", "raise", "RAISE-6k", plan_raise),
        Dataset("greysky", "greysky", "presets_sources/greysky", plan_greysky),
        Dataset("korean", "korean", "presets_sources/korean", plan_korean),
        Dataset("quandian", "quandian", "presets_sources/quandian", plan_quandian_sources),
        Dataset("awards", "awards", "presets_sources/awards", plan_awards),
        # recipe banks (look assets; sequenced for later LLM meta-tagging)
        Dataset("quandian_recipes", "quandian", "recipes/quandian",
                _recipe_planner(str(Path(RETOUCH_PRESETS) / "全店素材"), RECIPE_EXTS)),
        Dataset("e18_recipes", "e18", "recipes/e18",
                _recipe_planner(str(Path(RETOUCH_PRESETS) / "E18 《300+广告级LUT》"), {".cube", ".3dl"})),
    ]


def cmd_verify(dest_root: str) -> int:
    mig = Path(dest_root) / "_migration" / "manifests"
    if not mig.is_dir():
        print("no manifests to verify (run apply first)")
        return 1
    rc = 0
    for mf in sorted(mig.glob("*.jsonl")):
        n = ok = missing = pending = 0
        with open(mf, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                n += 1
                grp_missing = 0
                for a in rec["assets"]:
                    if os.path.exists(a["dst"]):
                        ok += 1
                    else:
                        missing += 1
                        grp_missing += 1
                # A group whose source still exists but dst doesn't = not yet applied.
                if grp_missing:
                    if any(not a["src"].startswith("tar://") and os.path.exists(a["src"]) for a in rec["assets"]):
                        pending += 1
                    else:
                        rc = 1
        print(f"[verify] {mf.stem}: groups={n} assets_present={ok} assets_missing={missing} pending={pending}")
    return rc


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Migrate + sequentially rename source corpora into ~/data/datasets")
    ap.add_argument("command", choices=["plan", "apply", "verify"])
    ap.add_argument("--dest-root", default=DEST_ROOT)
    ap.add_argument("--datasets", default="", help="comma list to restrict (default: all)")
    ap.add_argument("--seq-width", type=int, default=SEQ_WIDTH)
    ap.add_argument("--extract-fivek", action="store_true", help="extract fivek raw DNG from tar (+50G)")
    ap.add_argument("--no-resume", action="store_true", help="ignore prior state/manifests")
    args = ap.parse_args(argv)

    if args.command == "verify":
        return cmd_verify(args.dest_root)

    datasets = build_datasets(args.extract_fivek)
    if args.datasets:
        want = {s.strip() for s in args.datasets.split(",") if s.strip()}
        datasets = [d for d in datasets if d.name in want]
        if not datasets:
            raise SystemExit(f"no matching datasets in {want}")

    m = Migrator(args.dest_root, dry_run=(args.command == "plan"),
                 seq_width=args.seq_width, resume=not args.no_resume)
    total: Dict[str, int] = {}
    for ds in datasets:
        c = m.run_dataset(ds)
        for k, v in c.items():
            total[k] = total.get(k, 0) + v
    m._log("[TOTAL] " + " ".join(f"{k}={v}" for k, v in total.items() if v))
    if args.command == "plan":
        print(f"\nDry-run manifests written to {m.mig_dir / 'manifests_dryrun'}/ — review before `apply`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
