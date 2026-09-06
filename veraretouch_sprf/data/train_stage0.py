#!/usr/bin/env python3
# 源自 experiments/prs/EPR-051_masked-restore-production/stage0/train_stage0.py @ 2026-09-06（原件未入 git；原 sha256 见 veraretouch_sprf/PROVENANCE.md），逐字复制/仅改导入与路径解析（EPR-052）。
"""EPR-051 stage 0 (chain C): G4D restoration accuracy, single-head decoder.

Four elements of the experiment card live in ``TASK_CARD.md``; this file is the
implementation of exactly those four and nothing else.

  data      chain-B clean prefix pairs ``(y_d, before)``; held-out by a sha1 rule
            on the SOURCE id (never the row id, so repeat-salted repeats of one
            source stay on one side).  Clean-ness is per (id, depth): the
            ``emits[].flag`` of that depth decides, so a clean d4 enters training
            even when the full chain was rejected (IMPLEMENTATION 2.1b).
  model     frozen SigLIP2-so400m-patch14-384 on the DEGRADED image only ->
            shared trunk MLP -> ONE G4D parameter head (27N+12, N=48) plus the
            six alpha-hat mixing weights.  P0 picked the single-head form.
            alpha is oracle: the six journal alpha fields, rebuilt from the
            recorded parameters, composed into one scalar field alpha-hat.
  loss      sampled-pixel L1, ``|G(y_d, alpha_hat; theta) - before|``.
  optimiser AdamW + cosine, see ``[optim]``; the step budget is
            ``min(max_steps, ceil(n_train / batch))`` when ``one_pass_cap``.

Everything experiment-semantic comes from the TOML (single source of truth, no
default fallbacks -- a missing key stops the run).  Nothing about the data law is
re-implemented here: the alpha fields go through
``epr050_build_degradation``'s own functions (``lum_bands`` / ``hue_mask`` /
``geom_alpha_from_row``), and ``--verify-alpha N`` re-asserts on N chains that
they agree with ``epr050_recovery_montage.rebuild`` (the rebuilder that was
verified bit-exact against the stored assets, recovery_verify 200/200).

Guards that are wired, not just declared:
  * the eval's metric functions are counted; ``assert_wired()`` at the end of
    every eval raises unless every pre-registered column was computed once per
    evaluated sample ("defined but never connected" has cost this campaign three
    times).  ``--sabotage-metric COL`` skips one column on purpose so that the
    assertion can be shown to fire.
  * every journal row is asserted to be on the campaign's TRAIN side
    (``split_rule`` and ``split_bucket``): the last line of defence against
    V_where / V_what / T_final leaking in through a rebuilt source pool.
  * the held-out ids and the training ids are asserted disjoint, and the
    training sampler is asserted never to yield a held-out id.
  * the upper-bound column's provenance must be single-valued in one run, and a
    missing upper bound is counted and reported, never silently dropped.
  * the sha256 of this tool, of the config, of ``epr050_build_degradation.py``
    and of ``q3vl/whatb/arms/g4d.py`` are frozen into ``run_args.json`` and
    checked against ``[guard]`` before the first step.
  * a non-finite loss stops the run at that step (no silent NaN).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath
import io

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from veraretouch_sprf import _paths as _P  # EPR-052：替代原 sys.path 注入块
_P.ensure_sys_path()
REPO = _P.REPO

import epr050_build_degradation as B  # noqa: E402
from q3vl.whatb.arms import g4d as G4DMOD  # noqa: E402
from q3vl.whatb.arms.g4d import (  # noqa: E402
    G4DConfig,
    G4DGenerator,
    G4DParams,
    Glut4DCarrier,
    n_params_g4d,
)

LEVEL = 255.0
_QUANTILE_MAX = 1 << 24
SELECT_METRICS = ("heldout_model_linf8_p50",)


# --------------------------------------------------------------------------- #
# config: strict, no defaults (same discipline as epr050_build_degradation)
# --------------------------------------------------------------------------- #
def die(msg: str):
    raise SystemExit(f"train_stage0: {msg}")


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Cfg:
    """A TOML with no default values: every read is declared and checked."""

    def __init__(self, path: Path):
        try:
            import tomllib
        except ModuleNotFoundError:                       # pragma: no cover
            import tomli as tomllib                       # type: ignore
        if not path.exists():
            die(f"config not found: {path}")
        self.path = path
        self.text = path.read_text()
        self.sha256 = hashlib.sha256(self.text.encode()).hexdigest()
        self.d = tomllib.loads(self.text)
        self.read: dict[str, object] = {}

    def _get(self, sec: str, key: str):
        if sec not in self.d:
            die(f"config: missing section [{sec}]")
        if key not in self.d[sec]:
            die(f"config: missing key [{sec}] {key}")
        v = self.d[sec][key]
        self.read[f"{sec}.{key}"] = v
        return v

    def num(self, sec, key) -> float:
        v = self._get(sec, key)
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            die(f"config: [{sec}] {key} must be a number, got {v!r}")
        return float(v)

    def int_(self, sec, key) -> int:
        v = self._get(sec, key)
        if not isinstance(v, int) or isinstance(v, bool):
            die(f"config: [{sec}] {key} must be an integer, got {v!r}")
        return int(v)

    def str_(self, sec, key, allowed=None) -> str:
        v = self._get(sec, key)
        if not isinstance(v, str):
            die(f"config: [{sec}] {key} must be a string, got {v!r}")
        if allowed is not None and v not in allowed:
            die(f"config: [{sec}] {key} must be one of {allowed}, got {v!r}")
        return v

    def bool_(self, sec, key) -> bool:
        v = self._get(sec, key)
        if not isinstance(v, bool):
            die(f"config: [{sec}] {key} must be true/false, got {v!r}")
        return v

    def int_or_auto(self, sec, key) -> int | None:
        """An explicit step count, or ``"auto"`` -> None (derived from the budget)."""
        v = self._get(sec, key)
        if v == "auto":
            return None
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            die(f"config: [{sec}] {key} must be a non-negative integer or \"auto\", "
                f"got {v!r}")
        return int(v)

    def list_(self, sec, key, kind=str) -> list:
        v = self._get(sec, key)
        if not isinstance(v, list) or not all(isinstance(x, kind) for x in v):
            die(f"config: [{sec}] {key} must be a list of {kind.__name__}, got {v!r}")
        return list(v)


# --------------------------------------------------------------------------- #
# split: a sha1 rule of the campaign's family, on the SOURCE id
# --------------------------------------------------------------------------- #
def source_id_of(row_id: str) -> str:
    """``"<source>.rep<k>" -> "<source>"``; repeats of one source never straddle."""
    return row_id.partition(".rep")[0]


def split_bucket(source_id: str, salt: str) -> int:
    """``int(sha1("<salt>:<source_id>")[:8], 16) % 100`` -- the family the campaign
    already uses (``tools/data_splits/README.md``, epr050 ``split_bucket``).

    The salt is NOT ``verasplit-v1``: the degradation source pool is already
    filtered to that rule's TRAIN buckets (<= 89), so re-using it would put zero
    rows on the held-out side.  A fresh salt splits inside the train pool without
    ever touching V_where / V_what / T_final, which are excluded upstream and
    re-asserted row by row in :func:`load_shards`.
    """
    return int(hashlib.sha1(f"{salt}:{source_id}".encode()).hexdigest()[:8], 16) % 100


def is_heldout(row_id: str, salt: str, bucket_min: int) -> bool:
    return split_bucket(source_id_of(row_id), salt) >= bucket_min


# --------------------------------------------------------------------------- #
# sample index + the compact per-chain record the workers actually need
# --------------------------------------------------------------------------- #
_MASK_KEYS = ("q_lo", "q_hi", "width", "t_lo", "t_hi", "geom", "attempts")


def compact_row(r: dict) -> bytes:
    """Only the fields the alpha rebuild reads, as JSON bytes.

    A worker holding the full journal costs ~10 kB/chain of live Python objects;
    at 10w chains x 12 workers that is the machine's memory.  The compact record
    is stored as bytes (one refcount, no nested objects) and parsed per item --
    ~10 us against the item's ~150 ms, i.e. free.
    """
    m = r["mask"]
    rec = dict(id=r["id"], size=r["size"], subject_png=r.get("subject_png"),
               calib={"s": r["calib"]["s"]},
               steps=[{"kind": s["kind"]} for s in r["steps"]],
               mask={k: m[k] for k in _MASK_KEYS if k in m})
    rec["mask"]["geom_params"] = m.get("geom_params")
    return json.dumps(rec, ensure_ascii=False).encode()


def load_split_table(p: Path) -> dict[str, str]:
    """``tools/data_splits/splits_sources.csv`` -- the campaign's FROZEN split table.

    It, not the sha1 bucket, is authoritative for a source it knows about; the
    build tool records ``split_rule = "frozen_table"`` for exactly those rows.
    """
    import csv
    if not p.exists():
        die(f"[data] split_table {p} does not exist")
    with open(p, newline="") as f:
        return {r["source_id"]: r["split"] for r in csv.DictReader(f)}


def assert_train_side(r: dict, want_rules: set[str], train_max: int,
                      table: dict[str, str], table_path: Path) -> None:
    """Per-row last line of defence: this row is on the campaign's TRAIN side.

    Two authorities, each checked against its own source of truth -- a row whose
    rule the config did not authorise stops the run rather than being trusted.
      ``frozen_table``       the CSV must know the source and say ``train``.
      ``sha1_verasplit-v1``  the recorded bucket must be inside the train range.
    """
    rule = r.get("split_rule")
    if rule not in want_rules:
        die(f"{r['id']}: split_rule {rule!r} is not in [data] require_split_rules "
            f"{sorted(want_rules)} -- the campaign's split authorities are not "
            "negotiable")
    sid = source_id_of(r["id"])
    if rule == "frozen_table":
        got = table.get(sid)
        if got is None:
            die(f"{r['id']}: split_rule says frozen_table but {sid} is absent from "
                f"{table_path}")
        if got != "train":
            die(f"{r['id']}: the frozen split table puts {sid} in {got!r}; "
                "V_where/V_what/T_final must never reach training")
        return
    if not isinstance(r.get("split_bucket"), int) or r["split_bucket"] > train_max:
        die(f"{r['id']}: split_bucket {r.get('split_bucket')} is outside the train "
            f"range (<= {train_max}); V_where/V_what/T_final must never reach "
            "training")


def read_clean_ids(d: Path, names: list[str]) -> set[str]:
    """The shard's whole-chain clean list, under whichever name it was written.

    Only consulted for rows WITHOUT ``emits`` (pre-v3.7): with ``emits`` the
    per-depth flag is the clean criterion (editor's ruling, IMPLEMENTATION 2.1b).
    """
    for n in names:
        p = d / n
        if p.exists():
            blob = json.loads(p.read_text())
            if "clean_ids" in blob:
                return set(blob["clean_ids"])
            die(f"{p}: no 'clean_ids' key; the clean-list schema changed")
    die(f"{d}: none of {names} exists -- [data] clean_index_names is out of date")


def load_shards(cfg: Cfg) -> tuple[list[dict], dict[str, bytes], dict]:
    """One flat list of prefix-pair samples over every shard directory.

    A shard is a build output directory: ``pairs.jsonl`` + a clean list +
    ``assets/``.  With ``[emit]`` (v3.7) each row carries ``emits[]`` and one row
    yields several samples, one per emitted depth; without it (v3.6.1) the row
    yields the single full-chain depth whose asset is ``<id>.after.png``.
    """
    shards = cfg.list_("data", "shards")
    clean_only = cfg.bool_("data", "clean_only")
    clean_names = cfg.list_("data", "clean_index_names")
    flags_ok = set(cfg.list_("data", "depth_flags_ok"))
    salt = cfg.str_("data", "split_salt")
    bmin = cfg.int_("data", "holdout_bucket_min")
    want_rules = set(cfg.list_("data", "require_split_rules"))
    train_max = cfg.int_("data", "train_bucket_max")
    table_path = Path(cfg.str_("data", "split_table"))
    split_table = load_split_table(table_path) if "frozen_table" in want_rules else {}
    tool_allowed = set(cfg.list_("guard", "build_tool_sha256_allowed"))
    if not 0 < bmin <= 100:
        die("config: [data] holdout_bucket_min must be in (0, 100]")
    if not shards:
        die("config: [data] shards is empty -- point it at the chain-B shards")

    samples: list[dict] = []
    blobs: dict[str, bytes] = {}
    manifests = []
    cfg_sha = None
    n_flag_dropped = 0
    for sd in shards:
        d = Path(sd)
        rp = d / "run_args.json"
        if not rp.exists():
            die(f"{d}: no run_args.json -- not a build shard")
        man = json.loads(rp.read_text())
        if man["tool_sha256"] not in tool_allowed:
            die(f"{d} was built by tool sha {man['tool_sha256'][:12]}, which is not "
                f"in [guard] build_tool_sha256_allowed")
        # the journal's own sha256, as recorded by the quality filter: this is
        # what freezes THIS run's dataset against a shard that keeps growing.
        jsha, jname = None, None
        for n in clean_names:
            if (d / n).exists():
                jsha = json.loads((d / n).read_text()).get("journal_sha256")
                jname = n
                break
        manifests.append(dict(shard=str(d), config_sha256=man["config_sha256"],
                              tool_sha256=man["tool_sha256"],
                              config_path=man["config_path"],
                              clean_index=jname, journal_sha256=jsha,
                              journal_bytes=(d / "pairs.jsonl").stat().st_size))
        if cfg_sha is None:
            cfg_sha = man["config_sha256"]
        elif man["config_sha256"] != cfg_sha:
            die(f"shard {d} was built with a different config "
                f"({man['config_sha256'][:12]} vs {cfg_sha[:12]}); one training set "
                "must not mix data laws")
        legacy_clean = None
        for line in open(d / "pairs.jsonl"):
            r = json.loads(line)
            # last line of defence: the source pool is built train-only, and a
            # rebuilt pool that quietly changed rule or bucket would leak V/T.
            assert_train_side(r, want_rules, train_max, split_table, table_path)
            if r.get("winner_confidence") == "low":       # campaign-wide rule
                continue
            base = dict(shard=str(d), id=r["id"], geom=r["mask"]["geom"],
                        major=r.get("major"), rec_band=r.get("rec_band"),
                        pool=r.get("pool"),
                        heldout=is_heldout(r["id"], salt, bmin))
            got = []
            if r.get("emits"):
                # per-(id, depth) clean-ness: the depth's own flag decides
                for e in r["emits"]:
                    if clean_only and e.get("flag") not in flags_ok:
                        n_flag_dropped += 1
                        continue
                    got.append(dict(base, depth=int(e["depth"]),
                                    after_asset=e["asset"],
                                    exact_err=e.get("err_E_all")))
            else:
                if clean_only:
                    if legacy_clean is None:
                        legacy_clean = read_clean_ids(d, clean_names)
                    if r["id"] not in legacy_clean:
                        continue
                got.append(dict(base, depth=None,          # = full chain
                                after_asset=f"{r['id']}.after.png",
                                exact_err=r.get("err_E_all")))
            if got:
                samples += got
                blobs[r["id"]] = compact_row(r)
    if not samples:
        die("no samples after filtering")
    tr = {s["id"] for s in samples if not s["heldout"]}
    ho = {s["id"] for s in samples if s["heldout"]}
    if tr & ho:
        die(f"train/held-out id overlap: {sorted(tr & ho)[:5]}")
    blob_bytes = sum(len(v) for v in blobs.values())
    meta = dict(shards=manifests, build_config_sha256=cfg_sha,
                split_rule=f'int(sha1("{salt}:"+source_id)[:8],16) % 100 '
                           f'>= {bmin} -> held-out',
                train_guard=f"split_rule in {sorted(want_rules)}; frozen_table rows "
                            f"checked against {table_path} (must be 'train'), "
                            f"sha1 rows checked against split_bucket <= {train_max}",
                clean_criterion="per (id, depth) emits[].flag when emits exist, "
                                "else the shard's whole-chain clean list",
                n_samples=len(samples), n_chains=len(blobs),
                n_train_rows=len(tr), n_heldout_rows=len(ho),
                n_depth_flag_dropped=n_flag_dropped,
                compact_index_bytes=blob_bytes,
                compact_bytes_per_chain=round(blob_bytes / max(1, len(blobs)), 1))
    return samples, blobs, meta


def bind_build_config(samples: list[dict], allowed_cfg_sha: list[str]) -> dict:
    """Load the build TOML into ``epr050_build_degradation``'s globals.

    ``B.STEP_KIND`` / ``B.N_STEPS`` / ``render.max_side`` are experiment-semantic
    and only exist after this call; the dataset workers call it too.

    The build TOML on disk must still be the one the shards were built from, or
    an explicitly vetted successor listed in ``[guard]
    build_config_sha256_allowed``.  It fails CLOSED: a config nobody vetted
    stops the run, exactly like an unknown build-tool sha.
    """
    d = Path(samples[0]["shard"])
    man = json.loads((d / "run_args.json").read_text())
    p = Path(man["config_path"])
    if not p.is_absolute():
        p = B.REPO / p
    build_cfg = B.load_config(p)
    # the build TOML on disk must still be the one the shard was built from:
    # the alpha rebuild reads STEP_KIND / N_STEPS / max_side out of it, so an
    # edited build config is the same class of hazard as an edited build tool.
    on_disk = sha256_file(p)
    if on_disk != man["config_sha256"] and on_disk not in set(allowed_cfg_sha):
        die(f"build config {p} sha {on_disk[:12]} != the shard manifest's "
            f"config_sha256 {man['config_sha256'][:12]} and is not in [guard] "
            "build_config_sha256_allowed -- the alpha rebuild would run on a "
            "different data law than the one that built the shards")
    if on_disk != man["config_sha256"]:
        print(f"[guard] build config drifted to {on_disk[:12]} (shards built with "
              f"{man['config_sha256'][:12]}); allowed by [guard] "
              "build_config_sha256_allowed -- see NOTES N25 for the "
              "--verify-alpha evidence that admitted it", flush=True)
    return dict(step_kind=list(B.STEP_KIND), n_steps=int(B.N_STEPS),
                max_side=B._int(build_cfg, "render", "max_side"),
                config_path=str(p), config_sha256=man["config_sha256"])


# --------------------------------------------------------------------------- #
# oracle alpha: the journal's own rebuilders, never re-derived here
# --------------------------------------------------------------------------- #
class TarAssets:
    """Shard assets out of an indexed tar (``indexed_tar`` schema v2).

    The index rows carry ``offset_data`` and ``size``, so a member is one
    ``os.pread`` -- the same access pattern as
    ``dataset_build/tools/archive_reader.py::ArchiveReader.read``.  Deliberately
    NOT sqlite-backed: one fd plus a plain dict is thread-safe under the
    ``READ_FANOUT`` read pool, whereas a shared sqlite handle across threads is
    a trap this campaign has already paid for.
    """

    IDX_NAMES = ("assets.idx.jsonl", "assets.index.jsonl", "assets.tar.idx.jsonl")

    def __init__(self, shard: Path):
        self.tar = shard / "assets.tar"
        idx = next((shard / n for n in self.IDX_NAMES if (shard / n).is_file()), None)
        if idx is None:
            die(f"{shard}: assets.tar present but no index next to it "
                f"(looked for {', '.join(self.IDX_NAMES)})")
        self.index: dict[str, tuple[int, int, str]] = {}
        with open(idx, encoding="utf-8") as fh:
            for ln, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                sv = r.get("schema_version")
                if sv != 2:
                    die(f"{idx}:{ln}: unsupported index schema_version {sv!r} "
                        "(this tool reads indexed_tar schema v2)")
                if r["offset"] != r["offset_data"]:
                    die(f"{idx}:{ln}: offset != offset_data, refusing to guess")
                # address by file name, which is what the rest of this tool has
                # always used ("<id>.src.png", "<id>.d4.after.png", ...)
                name = PurePosixPath(str(r["logical_path"])).name
                rec = (int(r["offset_data"]), int(r["size"]),
                       str(r["member"]))
                for key in {name, str(r["member"])}:
                    if self.index.setdefault(key, rec) != rec:
                        die(f"{idx}: duplicate index key {key!r} with different "
                            "offsets; the shard index is ambiguous")
        self.fd = os.open(str(self.tar), os.O_RDONLY)

    def read(self, name: str) -> bytes:
        rec = self.index.get(name)
        if rec is None:
            raise KeyError(f"{name} is not in {self.tar}")
        off, size, member = rec

        def once() -> bytes:
            # check the tar's own header before trusting the index, exactly as
            # archive_reader.read does.  An index whose offsets are wrong reads
            # plausible-looking bytes from the wrong place and would otherwise
            # be trained on silently -- this guard caught precisely that bug in
            # the harness that built the first test shard.
            head = os.pread(self.fd, 512, off - 512)
            if len(head) != 512:
                raise OSError(f"short header read for {name}")
            try:
                info = tarfile.TarInfo.frombuf(head, encoding="utf-8",
                                               errors="strict")
            except Exception as exc:
                die(f"{self.tar}: no valid tar header at offset {off - 512} for "
                    f"{name} ({exc}) -- the index does not match the archive")
            if info.name != member or info.size != size:
                die(f"{self.tar}: index says {name} ({size} B) at {off} but the "
                    f"archive header says {info.name!r} ({info.size} B; "
                    f"index member {member!r}) -- "
                    "refusing to read from a mismatched index")
            buf = os.pread(self.fd, size, off)
            if len(buf) != size:
                raise OSError(f"short read: {len(buf)} of {size} bytes")
            return buf

        return with_read_retry(once, f"tar member {name} in {self.tar}")


# one reader per shard per process; workers inherit nothing and open their own
_TAR_ASSETS: dict[str, TarAssets] = {}
# run-constant, set once in main() from [data] asset_source before the DataLoader
# forks its workers, so every process agrees without threading it through.
ASSET_MODE = "dir"


def asset_bytes(shard: str, name: str) -> bytes:
    """Raw bytes of one shard asset, from the directory or the indexed tar."""
    if ASSET_MODE == "dir" or (ASSET_MODE == "auto"
                               and not (Path(shard) / "assets.tar").is_file()):
        def once() -> bytes:
            return (Path(shard) / "assets" / name).read_bytes()
        return with_read_retry(once, f"asset {name}")
    reader = _TAR_ASSETS.get(shard)
    if reader is None:
        reader = _TAR_ASSETS[shard] = TarAssets(Path(shard))
    return reader.read(name)


def load_pair(shard: str, row: dict, after_asset: str):
    """``(before, after)`` as float tensors in [0,1] from the shard's assets.

    ``<id>.src.png`` is byte-identical to ``open_source(source_path, max_side)``
    (asserted in ``--verify-alpha``), so reading the asset instead of the source
    JPEG is the same image at a fraction of the decode cost.
    """
    def png(name: str) -> np.ndarray:
        # the read itself retries inside asset_bytes; the decode is wrapped too
        # because a truncated read surfaces as a PIL error, not an OSError.
        try:
            with Image.open(io.BytesIO(asset_bytes(shard, name))) as im:
                return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        except ReadFailed:
            raise
        except Exception as exc:
            raise ReadFailed(f"asset {name}: {exc!r}") from exc

    x0 = png(f"{row['id']}.src.png")
    y = png(after_asset)
    if x0.shape != y.shape:
        die(f"{row['id']}: before {x0.shape} and after {y.shape} disagree")
    return torch.from_numpy(x0), torch.from_numpy(y)


def alpha_fields(row: dict, x0: torch.Tensor) -> torch.Tensor:
    """The six journal alpha fields, in chain order, already scaled by ``s``.

    Every field is produced by the build module's own function, with the row's
    recorded parameters; nothing is resampled and nothing is re-implemented.
    """
    h, w = row["size"]
    if tuple(x0.shape[:2]) != (h, w):
        die(f"{row['id']}: asset {tuple(x0.shape[:2])} vs journal size {(h, w)}")
    m = row["mask"]
    bands, (t_lo, t_hi) = B.lum_bands(x0, m["q_lo"], m["q_hi"], m["width"])
    if abs(t_lo - m["t_lo"]) > 1e-6 or abs(t_hi - m["t_hi"]) > 1e-6:
        die(f"{row['id']}: luminance threshold drift ({t_lo}, {t_hi}) vs "
            f"recorded ({m['t_lo']}, {m['t_hi']})")
    hard = None
    if m["geom"] in ("subject", "semantic"):
        subj = B.load_subject(Path(row["subject_png"]), h, w, "cpu")
        hard = ((subj > 0.5).float().numpy() if m["geom"] == "semantic"
                else subj.numpy())
    by_kind = dict(bands)
    by_kind["geom"] = B.geom_alpha_from_row(row, h, w, "cpu", hard)
    by_kind["global"] = torch.ones((h, w))
    if "hue" in B.STEP_KIND:
        by_kind["hue"] = B.hue_mask(x0)
    if [s["kind"] for s in row["steps"]] != list(B.STEP_KIND):
        die(f"{row['id']}: config step_order {list(B.STEP_KIND)} disagrees with "
            f"the recorded chain {[s['kind'] for s in row['steps']]}")
    f = torch.stack([by_kind[k] for k in B.STEP_KIND], 0)
    s = row["calib"]["s"]
    return f * s if s != 1.0 else f


def depth_mask(depth: int | None, n_steps: int) -> torch.Tensor:
    """``(n_steps,)`` 1.0 for the steps that were actually applied to this pair."""
    d = n_steps if depth is None else int(depth)
    if not 1 <= d <= n_steps:
        die(f"depth {d} outside 1..{n_steps}")
    m = torch.zeros(n_steps)
    m[:d] = 1.0
    return m


def compose_alpha(alphas: torch.Tensor, mask: torch.Tensor,
                  w: torch.Tensor | None, mode: str) -> torch.Tensor:
    """The composite scalar field alpha-hat, P0's two definitions (TASK_CARD 5).

    ``alphas`` (B, K, P), ``mask`` (B, K), ``w`` (B, K) or None.
      prod        alpha_hat = 1 - prod_k (1 - m_k alpha_k)
      wsum_pred   alpha_hat = clamp(sum_k m_k w_k alpha_k, 0, 1), w predicted
    """
    m = mask.unsqueeze(-1)
    if mode == "prod":
        return 1.0 - torch.prod(1.0 - m * alphas, dim=1)
    if mode == "wsum_pred":
        if w is None:
            die("alpha_hat = wsum_pred needs the predicted weights")
        return (m * w.unsqueeze(-1) * alphas).sum(1).clamp(0.0, 1.0)
    die(f"unknown alpha_hat mode {mode!r}")


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
def feature_key(s: dict) -> str:
    return f"{s['id']}|{'full' if s['depth'] is None else s['depth']}"


class PrefixPixels(Dataset):
    """One item = ``n_pix`` random pixels of one prefix pair, plus its alphas."""

    def __init__(self, samples: list[dict], blobs: dict[str, bytes],
                 n_pix: int, n_steps: int):
        self.samples = samples
        self.blobs = blobs
        self.n_pix = int(n_pix)
        self.n_steps = int(n_steps)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        row = json.loads(self.blobs[s["id"]])
        x0, y = load_pair(s["shard"], row, s["after_asset"])
        a = alpha_fields(row, x0)
        npx = x0.shape[0] * x0.shape[1]
        # the worker's own RNG (DataLoader seeds it from the main process's
        # generator, which `[run] seed` fixes): fresh pixels every visit, and the
        # whole run still replays from one seed.
        idx = torch.randint(0, npx, (self.n_pix,))
        return dict(
            x=x0.reshape(-1, 3)[idx],
            y=y.reshape(-1, 3)[idx],
            alpha=a.reshape(self.n_steps, -1)[:, idx],
            mask=depth_mask(s["depth"], self.n_steps),
            key=feature_key(s), index=i)


def collate(batch: list[dict]) -> dict:
    out = {k: torch.stack([b[k] for b in batch]) for k in ("x", "y", "alpha", "mask")}
    out["key"] = [b["key"] for b in batch]
    out["index"] = torch.tensor([b["index"] for b in batch])
    return out


# --------------------------------------------------------------------------- #
# frozen encoder + per-shard feature cache
# --------------------------------------------------------------------------- #
class FrozenSiglip:
    def __init__(self, path: str, dtype: str, device: str, readout: str):
        from transformers import AutoProcessor, SiglipVisionModel
        td = {"bf16": torch.bfloat16, "fp16": torch.float16,
              "fp32": torch.float32}[dtype]
        self.model = SiglipVisionModel.from_pretrained(
            path, local_files_only=True, dtype=td).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.proc = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.device, self.td, self.readout = device, td, readout
        self.dim = int(self.model.config.hidden_size)

    @torch.no_grad()
    def encode(self, images: list[Image.Image]) -> torch.Tensor:
        """``(B, D)`` for the pooled read-outs, ``(B, L, D)`` for ``tokens``.

        The cache dtype is a function of the read-out, not a separate key:
        ``pooler`` / ``mean`` keep the fp32 they have always written (their
        cache fingerprints therefore do not move), ``tokens`` writes fp16
        because the whole ``last_hidden_state`` is kept -- 729 x 1152 per
        sample, no pooling, no compression (EPR-051 chain C ruling).
        """
        px = self.proc(images=images, return_tensors="pt").pixel_values
        out = self.model(px.to(self.device, self.td))
        if self.readout == "pooler":
            return out.pooler_output.float().cpu()
        if self.readout == "mean":
            return out.last_hidden_state.mean(1).float().cpu()
        if self.readout == "tokens":
            return out.last_hidden_state.to(torch.float16).cpu()
        die(f"unknown [encoder] readout {self.readout!r}")


def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


READ_FANOUT = 8         # concurrent pread()s per batch; see NOTES N21
READ_RETRY_SLEEPS = (5, 15, 45)     # NFS soft-mount jitter; see NOTES N27


class ReadFailed(RuntimeError):
    """A cache/asset read that survived every retry.  Ends the run cleanly."""


def with_read_retry(fn, what: str):
    """Run ``fn``; on an OS-level read error back off and retry, then give up.

    The token cache and the shard assets may live on a soft-mounted NFS export,
    where a server hiccup surfaces as EIO/ESTALE rather than a hang.  A retried
    read is fine; a *silently wrong* read is not, so after the last attempt this
    raises ``ReadFailed`` and the caller checkpoints and dies instead of
    training on whatever came back.  Sleeps are 5 / 15 / 45 s (4 attempts).
    """
    last = None
    for i, nap in enumerate((0,) + READ_RETRY_SLEEPS):
        if nap:
            time.sleep(nap)
        try:
            return fn()
        except OSError as exc:                     # EIO / ESTALE / ENODEV / ...
            last = exc
            print(f"[io] read failed ({what}) attempt {i + 1}/"
                  f"{len(READ_RETRY_SLEEPS) + 1}: {exc!r}", flush=True)
    raise ReadFailed(f"{what}: {len(READ_RETRY_SLEEPS) + 1} attempts failed, "
                     f"last error {last!r}")


class TokenStore:
    """``key -> (L, D)`` fp16 patch tokens, one file per shard, read with pread.

    The token cache is 1,679,616 B per sample (729 x 1152 x fp16); the 13-shard
    training set is 91.9 GB and the box has 125 GB of RAM, so the cache lives on
    disk and is read per sample instead of being held as a dict of tensors.
    This is NOT a compression: every one of the 729 tokens is stored at full
    1152-d fp16 and ``__getitem__`` returns exactly the bytes the encoder wrote.

    ``os.preadv`` rather than ``np.memmap`` indexing: preadv takes no shared file
    offset (so one fd is safe from many threads) and it releases the GIL for the
    duration of the read, which is what lets ``prefetched`` fan a batch out over
    ``READ_FANOUT`` threads.  A memmap page-faults with the GIL held and does not
    parallelise.  Byte-for-byte the same data either way.
    """

    def __init__(self) -> None:
        self.parts: list[tuple[int, int, int]] = []     # (fd, ntok, dim)
        self.index: dict[str, tuple[int, int]] = {}
        self.bytes = 0

    def add(self, path: Path, keys: list[str], shape: tuple[int, int, int]) -> None:
        fd = os.open(str(path), os.O_RDONLY)
        pi = len(self.parts)
        self.parts.append((fd, int(shape[1]), int(shape[2])))
        for i, k in enumerate(keys):
            self.index[k] = (pi, i)
        self.bytes += int(np.prod(shape)) * 2

    def __len__(self) -> int:
        return len(self.index)

    def __contains__(self, k: str) -> bool:
        return k in self.index

    def __getitem__(self, k: str) -> torch.Tensor:
        pi, i = self.index[k]
        fd, ntok, dim = self.parts[pi]
        nbytes = ntok * dim * 2

        def read_once() -> bytearray:
            buf = bytearray(nbytes)
            got = os.preadv(fd, [buf], i * nbytes)
            if got != nbytes:
                # a short read is not an exception, so raise one: on a soft
                # mount it is the same class of event as EIO and must retry,
                # never be zero-padded into the batch.
                raise OSError(f"short read: {got} of {nbytes} bytes")
            return buf

        buf = with_read_retry(read_once, f"token cache {k}")
        return torch.from_numpy(
            np.frombuffer(buf, dtype=np.float16).reshape(ntok, dim))


def stack_feats(feats, keys: list[str], tokens: int, pool=None) -> torch.Tensor:
    """The disk-bound half: ``(B, in_dim)`` or ``(B, L, D)``, still on the CPU.

    ``pool`` fans the per-sample reads out over threads.  ``Executor.map``
    yields in ARGUMENT order, so the stack is assembled in ``keys`` order no
    matter which read finishes first: the result is bit-identical to the serial
    comprehension, and so is the order in which samples are consumed.
    """
    parts = ([feats[k] for k in keys] if pool is None
             else list(pool.map(feats.__getitem__, keys)))
    t = torch.stack(parts)
    return t if tokens else t.reshape(len(keys), -1)


def gather_feats(feats, keys: list[str], tokens: int, device: str) -> torch.Tensor:
    """``(B, in_dim)`` for the pooled cache, ``(B, L, D)`` for the token cache."""
    return stack_feats(feats, keys, tokens).to(device, non_blocking=True).float()


def prefetched(loader, feats, tokens: int, device: str):
    """Yield ``(batch, features)`` with the NEXT batch's features already read.

    One training batch is 64 x 729 x 1152 fp16 = 107 MB of the token cache on a
    rotational disk, so two things are overlapped with the previous step's
    compute: the batch itself (one buffer thread, depth 1) and, inside it, the
    64 per-sample ``pread``s over ``READ_FANOUT`` threads.  Measured cold, one
    batch: 2.56 s serial, 1.60 s at fan-out 8.

    Only the CPU-side read is threaded: the ``.to(device)`` stays on the main
    thread so every CUDA call in this process is still issued from one thread.
    Two pools, not one: the buffer thread blocks on the read pool, and a task
    that waits on its own pool can deadlock it.

    The values yielded, and the order samples are consumed in, are exactly what
    the serial ``gather_feats`` gives -- this is a scheduling change only.
    """
    from concurrent.futures import ThreadPoolExecutor
    fan = READ_FANOUT if tokens else 0
    with ThreadPoolExecutor(max_workers=1) as buf:
        pool = ThreadPoolExecutor(max_workers=fan) if fan else None
        try:
            pending = None
            for b in loader:
                fut = buf.submit(stack_feats, feats, b["key"], tokens, pool)
                if pending is not None:
                    pb, pf = pending
                    yield pb, pf.result().to(device, non_blocking=True).float()
                pending = (b, fut)
            if pending is not None:
                pb, pf = pending
                yield pb, pf.result().to(device, non_blocking=True).float()
        finally:
            if pool is not None:
                pool.shutdown(wait=True)


def build_feature_cache(cfg: Cfg, samples: list[dict], device: str,
                        inputs: list[str]) -> tuple[dict, dict]:
    """``key -> (len(inputs), D)`` features of the frozen encoder, one file per shard.

    The encoder is frozen for the whole of stage 0, so its output is a pure
    function of the image bytes: it is computed once and cached.  Each file is
    fingerprinted by (encoder path, readout, dtype, inputs, that shard's key
    list); a mismatch recomputes that shard only, so an interrupted cache build
    does not throw away the shards already done.
    """
    path = cfg.str_("encoder", "path")
    readout = cfg.str_("encoder", "readout", ("pooler", "mean", "tokens"))
    dtype = cfg.str_("encoder", "dtype", ("bf16", "fp16", "fp32"))
    bs = cfg.int_("encoder", "cache_batch")
    cache_dir = Path(cfg.str_("encoder", "cache_dir"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    by_shard: dict[str, list[dict]] = {}
    for s in samples:
        by_shard.setdefault(s["shard"], []).append(s)

    if readout == "tokens":
        return _build_token_cache(by_shard, path, readout, dtype, bs, cache_dir,
                                  inputs, device)

    feats: dict[str, torch.Tensor] = {}
    enc = None
    dim = None
    files = []
    t0 = time.time()
    n_encoded = 0
    for shard, ss in sorted(by_shard.items()):
        keys = sorted(feature_key(s) for s in ss)
        fp = hashlib.sha256(json.dumps(
            [path, readout, dtype, inputs, keys]).encode()).hexdigest()
        fn = cache_dir / f"siglip_{Path(shard).name}_{fp[:12]}.pt"
        if fn.exists():
            blob = torch.load(fn, map_location="cpu")
            if blob.get("fingerprint") == fp:
                feats.update(blob["features"])
                dim = blob["dim"]
                files.append(dict(shard=shard, file=str(fn), reused=True,
                                  n=len(blob["features"])))
                continue
        if enc is None:
            enc = FrozenSiglip(path, dtype, device, readout)
            dim = enc.dim
        part: dict[str, torch.Tensor] = {}
        for i in range(0, len(ss), bs):
            chunk = ss[i:i + bs]
            imgs: list[Image.Image] = []
            for s in chunk:
                a = Path(s["shard"]) / "assets"
                for which in inputs:
                    name = (f"{s['id']}.src.png" if which == "before"
                            else s["after_asset"])
                    imgs.append(Image.open(
                        io.BytesIO(asset_bytes(s["shard"], name))).convert("RGB"))
            z = enc.encode(imgs).reshape(len(chunk), len(inputs), -1)
            for j, s in enumerate(chunk):
                part[feature_key(s)] = z[j].clone()
            for im in imgs:
                im.close()
            n_encoded += len(imgs)
            if (i // bs) % 50 == 0:
                print(f"  encoder cache {Path(shard).name} {i + len(chunk)}/{len(ss)} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        _atomic_save(dict(fingerprint=fp, features=part, dim=dim), fn)
        feats.update(part)
        files.append(dict(shard=shard, file=str(fn), reused=False, n=len(part)))
    if enc is not None:
        del enc
        torch.cuda.empty_cache()
    if dim is None:
        die("feature cache produced no dimension -- no samples?")
    return feats, dict(dir=str(cache_dir), files=files, dim=dim, inputs=list(inputs),
                       tokens=0, bytes=0,
                       images_encoded=n_encoded, seconds=round(time.time() - t0, 1))


def _build_token_cache(by_shard: dict, path: str, readout: str, dtype: str, bs: int,
                       cache_dir: Path, inputs: list[str],
                       device: str) -> tuple[TokenStore, dict]:
    """``key -> (L, D)`` SigLIP2 ``last_hidden_state``, one memmap per shard.

    Same contract as the pooled cache: the frozen encoder's output is a pure
    function of the image bytes, each shard file is fingerprinted by (encoder
    path, read-out, dtype, inputs, that shard's key list), and an interrupted
    build only loses the shard it was on.  The ``.json`` sidecar is written
    AFTER the ``.bin`` is renamed into place, so its presence is the "this
    shard is complete" flag.

    No pooling and no dimensionality reduction: all 729 tokens x 1152 dims are
    stored (fp16, 1,679,616 B per sample).  See NOTES N15 for the budget.
    """
    import shutil

    store = TokenStore()
    enc = None
    dim = None
    ntok = None
    files: list[dict] = []
    t0 = time.time()
    n_encoded = 0
    n_new = 0

    todo = sum(len(ss) for ss in by_shard.values())
    free_gb = shutil.disk_usage(cache_dir).free / 2 ** 30
    print(f"  token cache: <= {todo} samples x 1,679,616 B (729x1152 fp16) "
          f"= {todo * 1679616 / 2 ** 30:.1f} GB worst case; "
          f"{free_gb:.1f} GB free on {cache_dir}", flush=True)

    for shard, ss in sorted(by_shard.items()):
        order = sorted(ss, key=feature_key)
        keys = [feature_key(s) for s in order]
        if len(set(keys)) != len(keys):
            die(f"{shard}: duplicate feature keys -- the token memmap is indexed "
                "by key and cannot hold two rows under one name")
        fp = hashlib.sha256(json.dumps(
            [path, readout, dtype, inputs, keys]).encode()).hexdigest()
        stem = cache_dir / f"tokens_{Path(shard).name}_{fp[:12]}"
        binp, metap = Path(f"{stem}.bin"), Path(f"{stem}.json")
        if metap.exists() and binp.exists():
            meta = json.loads(metap.read_text())
            shape = tuple(int(x) for x in meta.get("shape", ()))
            if (meta.get("fingerprint") == fp and len(shape) == 3
                    and binp.stat().st_size == int(np.prod(shape)) * 2):
                store.add(binp, meta["keys"], shape)
                ntok, dim = shape[1], shape[2]
                files.append(dict(shard=shard, file=str(binp), reused=True,
                                  n=len(meta["keys"]), shape=list(shape)))
                continue
        if enc is None:
            enc = FrozenSiglip(path, dtype, device, readout)
            dim = enc.dim
        tmp = Path(f"{stem}.bin.tmp")
        mm = None
        for i in range(0, len(order), bs):
            chunk = order[i:i + bs]
            imgs: list[Image.Image] = []
            for s in chunk:
                a = Path(s["shard"]) / "assets"
                for which in inputs:
                    name = (f"{s['id']}.src.png" if which == "before"
                            else s["after_asset"])
                    imgs.append(Image.open(
                        io.BytesIO(asset_bytes(s["shard"], name))).convert("RGB"))
            z = enc.encode(imgs)                       # (n_imgs, L, D) fp16
            z = z.reshape(len(chunk), len(inputs) * z.shape[-2], z.shape[-1])
            if mm is None:
                ntok = int(z.shape[1])
                need = len(order) * ntok * dim * 2
                if need > shutil.disk_usage(cache_dir).free:
                    die(f"token cache for {Path(shard).name} needs "
                        f"{need / 2 ** 30:.1f} GB and the disk has "
                        f"{shutil.disk_usage(cache_dir).free / 2 ** 30:.1f} GB free")
                mm = np.memmap(tmp, dtype=np.float16, mode="w+",
                               shape=(len(order), ntok, dim))
            mm[i:i + len(chunk)] = z.numpy()
            n_encoded += len(imgs)
            n_new += len(chunk)
            if (i // bs) % 50 == 0:
                el = time.time() - t0
                print(f"  token cache {Path(shard).name} {i + len(chunk)}/{len(order)} "
                      f"({el:.0f}s, {n_new / max(el, 1e-9):.1f} sample/s)", flush=True)
        if mm is None:
            die(f"{shard}: no samples to encode")
        mm.flush()
        del mm
        os.replace(tmp, binp)
        meta = dict(fingerprint=fp, keys=keys, shape=[len(order), ntok, dim],
                    dtype="float16")
        mtmp = Path(f"{stem}.json.tmp")
        mtmp.write_text(json.dumps(meta))
        os.replace(mtmp, metap)
        store.add(binp, keys, (len(order), ntok, dim))
        files.append(dict(shard=shard, file=str(binp), reused=False, n=len(order),
                          shape=[len(order), ntok, dim]))
    if enc is not None:
        del enc
        torch.cuda.empty_cache()
    if dim is None or ntok is None:
        die("token cache produced no shape -- no samples?")
    el = round(time.time() - t0, 1)
    print(f"  token cache ready: {len(store)} samples, {ntok} tokens x {dim} dims, "
          f"{store.bytes / 2 ** 30:.1f} GB on disk, {n_new} newly encoded in {el}s",
          flush=True)
    return store, dict(dir=str(cache_dir), files=files, dim=dim, inputs=list(inputs),
                       tokens=ntok, bytes=store.bytes,
                       images_encoded=n_encoded, seconds=el)


# --------------------------------------------------------------------------- #
# model: shared trunk -> ONE G4D head (+ the alpha-hat weights)
# --------------------------------------------------------------------------- #
class ConfidencePool(nn.Module):
    """FC4's confidence-weighted pooling in its one-layer degenerate form.

    Anchor (opened 2026-08-27): yuanming-hu/fc4, ``config.py`` --
    ``WEIGHTED_POOLING = True``, ``FCN_INPUT_SIZE = 512``: every spatial
    position emits its own estimate plus a learned confidence, and the global
    answer is the confidence-weighted sum instead of a mean.  Here the
    "positions" are SigLIP2 patch tokens and the weighted sum produces the same
    1152-d vector the pooled arm feeds to the trunk, so this is the ablation
    row that isolates "which tokens" from "how many queries".
    """

    def __init__(self, in_dim: int):
        super().__init__()
        self.conf = nn.Linear(in_dim, 1)

    def forward(self, tok: torch.Tensor) -> torch.Tensor:      # (B,L,D) -> (B,D)
        a = torch.softmax(self.conf(tok).squeeze(-1), dim=-1)
        return torch.einsum("bl,bld->bd", a, tok)


class TokenReadout(nn.Module):
    """M learnable queries reading the 729 patch tokens through one MHA layer.

    BACKBONE_SURVEY 4 改法二.  Anchors, both opened 2026-08-27:
      * AttentionLut (arXiv:2401.01569) -- "the attention fusion module
        integrates the image feature with the priori attention feature obtained
        during training", i.e. learned query tokens read an image feature set;
      * SA-LUT (arXiv:2506.13465) -- "a Context Generator using content-style
        cross-attention to produce a context map".
    Query ``M-1`` is the w query (it feeds ``w_head``); queries ``0..M-2`` are
    the theta queries and are concatenated into the trunk's input.
    """

    def __init__(self, in_dim: int, d: int, n_query: int, n_heads: int):
        super().__init__()
        if n_query < 2:
            die(f"[model] n_query = {n_query}: at least one theta query and one "
                "w query are needed")
        if d % n_heads:
            die(f"[model] d_readout {d} is not divisible by n_heads {n_heads}")
        self.proj = nn.Linear(in_dim, d)
        self.query = nn.Parameter(torch.empty(n_query, d))
        nn.init.normal_(self.query, std=d ** -0.5)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.n_query, self.d = n_query, d
        self.theta_in = (n_query - 1) * d
        self.w_in = d

    def forward(self, tok: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        kv = self.proj(tok)                                   # (B,L,d)
        q = self.query.unsqueeze(0).expand(tok.shape[0], -1, -1)
        o, _ = self.attn(q, kv, kv, need_weights=False)        # (B,M,d)
        return o[:, :-1].reshape(tok.shape[0], -1), o[:, -1]


class Stage0Model(nn.Module):
    """SigLIP2 features -> read-out -> trunk MLP -> one ``G4DParams`` + six w.

    K = 1 (single head) is the P0 verdict; the six-head composite stays an
    ablation and is not built here.

    ``[model] readout`` picks where the trunk's input comes from:
      ``pooled``    the pooled 1152-d vector (the baseline; the encoder cache
                    must be a pooled one), trunk in = 1152, w_head in = cond;
      ``mha``       ``TokenReadout`` over the 729 patch tokens, trunk in =
                    (n_query - 1) * d_readout, w_head in = d_readout;
      ``confpool``  ``ConfidencePool`` over the same tokens, trunk in = 1152,
                    w_head in = cond.
    In every mode the G4D head's last layers are zero-weighted and ``w_head``
    is zero-weight / bias 1/K, so the step-0 output is the identity map and is
    bit-identical across the three read-outs.
    """

    def __init__(self, in_dim: int, cfg: Cfg, n_steps: int, tokens: int = 0):
        super().__init__()
        hidden = cfg.int_("model", "trunk_hidden")
        layers = cfg.int_("model", "trunk_layers")
        cond = cfg.int_("model", "cond_dim")
        mode = cfg.str_("model", "readout", ("pooled", "mha", "confpool"))
        d_read = cfg.int_("model", "d_readout")
        n_query = cfg.int_("model", "n_query")
        n_heads = cfg.int_("model", "n_heads")
        g4d = G4DConfig(mode=cfg.str_("model", "mode"),
                        n_gauss=cfg.int_("model", "n_gauss"),
                        cond_dim=cond, hidden=cfg.int_("model", "gen_hidden"))
        if mode == "pooled" and tokens:
            die("[model] readout = 'pooled' but [encoder] readout = 'tokens': "
                "the pooled trunk cannot eat a (L, D) cache")
        if mode != "pooled" and not tokens:
            die(f"[model] readout = {mode!r} needs [encoder] readout = 'tokens'")
        # module creation order is load-bearing: in 'pooled' mode nothing is
        # built before the trunk, so the baseline's RNG stream (and therefore
        # its weights) are unchanged by this file's token support.
        if mode == "pooled":
            self.readout = None
            trunk_in, w_in = in_dim, cond
        elif mode == "confpool":
            self.readout = ConfidencePool(in_dim)
            trunk_in, w_in = in_dim, cond
        else:
            self.readout = TokenReadout(in_dim, d_read, n_query, n_heads)
            trunk_in, w_in = self.readout.theta_in, self.readout.w_in
        seq: list[nn.Module] = []
        d = trunk_in
        for _ in range(layers):
            seq += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        seq += [nn.Linear(d, cond), nn.ReLU()]
        self.trunk = nn.Sequential(*seq)
        self.head = G4DGenerator(g4d)
        self.w_head = nn.Linear(w_in, n_steps)
        nn.init.zeros_(self.w_head.weight)
        with torch.no_grad():                       # init at P0's 1/K weights
            self.w_head.bias.fill_(1.0 / n_steps)
        self.readout_mode = mode
        self.g4d_cfg = g4d
        self.theta_dim = n_params_g4d(g4d.n_gauss, g4d.mode)

    def trunk_forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Everything up to (not including) the two heads.

        Returns ``(cond for the G4D head, feature for w_head)``; the two are the
        same tensor in every mode except ``mha``, where the w query has its own
        output.  Split out so the training loop can keep this part under
        autocast while the G4D head stays in fp32 (``autocast_scope``).
        """
        if self.readout is None:
            h = self.trunk(feat)
            return h, h
        if self.readout_mode == "confpool":
            h = self.trunk(self.readout(feat))
            return h, h
        theta_in, w_in = self.readout(feat)
        return self.trunk(theta_in), w_in

    def heads(self, h: torch.Tensor, wf: torch.Tensor):
        return self.head(h), self.w_head(wf)

    def forward(self, feat: torch.Tensor) -> tuple[G4DParams, torch.Tensor]:
        h, wf = self.trunk_forward(feat)
        return self.heads(h, wf)


def restore(carrier: Glut4DCarrier, y: torch.Tensor, s: torch.Tensor,
            params: G4DParams, point_chunk: int) -> torch.Tensor:
    return carrier(y, s, params, point_chunk=point_chunk)


# --------------------------------------------------------------------------- #
# metrics: counted, so "defined but never connected" cannot happen quietly
# --------------------------------------------------------------------------- #
def quantile(t: torch.Tensor, p: float) -> float:
    f = t.reshape(-1).float()
    if f.numel() > _QUANTILE_MAX:
        f = f[:: (f.numel() + _QUANTILE_MAX - 1) // _QUANTILE_MAX]
    return float(torch.quantile(f, p))


def linf8(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """8-bit L-inf over RGB -- the campaign's one restoration-error unit."""
    return (pred - tgt).abs().amax(-1) * LEVEL


def err_stats(e: torch.Tensor) -> dict:
    return dict(n=int(e.numel()), p50=quantile(e, 0.5), p95=quantile(e, 0.95),
                p99=quantile(e, 0.99), mean=float(e.mean()), max=float(e.max()))


class MetricLedger:
    """Counts every pre-registered column's evaluations and refuses a silent skip."""

    def __init__(self, columns: list[str], strata: list[str]):
        self.columns = list(columns)
        self.strata = list(strata)
        self.calls: dict[str, int] = {}
        self.missing: dict[str, int] = {}
        self.sources: dict[str, set] = {}
        self.expected = 0
        self.armed = False

    def arm(self, n_samples: int) -> None:
        self.calls = {c: 0 for c in self.columns}
        self.missing = {c: 0 for c in self.columns}
        self.sources = {c: set() for c in self.columns}
        self.expected = int(n_samples)
        self.armed = True

    def note(self, column: str, source: str | None = None,
             missing: bool = False) -> None:
        if not self.armed:
            die("metric ledger used before arm(); the eval was not wired")
        if column not in self.calls:
            die(f"metric column {column!r} is not pre-registered {self.columns}")
        self.calls[column] += 1
        if missing:
            self.missing[column] += 1
        if source is not None:
            self.sources[column].add(source)

    def stats(self, column: str, pred: torch.Tensor, tgt: torch.Tensor,
              source: str | None = None) -> dict:
        self.note(column, source=source)
        return err_stats(linf8(pred, tgt))

    def assert_wired(self, out: dict) -> None:
        """The run-time assertion the pre-registration demands."""
        if not self.armed:
            die("eval finished without arming the metric ledger")
        bad = {c: n for c, n in self.calls.items() if n != self.expected}
        if bad:
            die("pre-registered metric functions were not called once per "
                f"evaluated sample: {bad} (expected {self.expected} each). "
                "A criterion that is defined but not connected is a blocker.")
        for c, src in self.sources.items():
            if len(src) > 1:
                die(f"column {c!r} mixed provenances {sorted(src)} inside one run; "
                    "two different measurement domains must not share a column")
        for c, n in self.missing.items():
            if n and n == self.expected:
                die(f"column {c!r} is missing on every evaluated sample -- the "
                    "column is not available for this data, do not report it empty")
        missing = [s for s in self.strata if s not in out.get("by", {})]
        if missing:
            die(f"pre-registered strata {missing} are absent from the eval output")


# --------------------------------------------------------------------------- #
# eval
# --------------------------------------------------------------------------- #
def median_of(recs: list[dict], column: str, field: str) -> float:
    v = [r["cols"][column][field] for r in recs if column in r["cols"]
         and r["cols"][column].get(field) is not None]
    return float(np.median(np.asarray(v, dtype=float))) if v else float("nan")


def eval_subset(samples: list[dict], k: int, salt: str) -> list[dict]:
    """A fixed, hash-ordered, (geom, rec_band, depth)-stratified subset of size k.

    Not a sorted prefix: within each stratum the order is
    ``sha1("<salt>:<key>")`` and the strata are drawn round-robin, so the subset
    is stable across steps, covers every stratum present, and is not a function
    of the id's alphabet.
    """
    if k <= 0 or k >= len(samples):
        return list(samples)
    groups: dict[tuple, list] = {}
    for s in samples:
        groups.setdefault((str(s["geom"]), str(s["rec_band"]), str(s["depth"])),
                          []).append(s)
    for g in groups.values():
        g.sort(key=lambda s: hashlib.sha1(
            f"{salt}:{feature_key(s)}".encode()).hexdigest())
    keys = sorted(groups)
    out: list[dict] = []
    i = 0
    while len(out) < k:
        moved = False
        for kk in keys:
            if i < len(groups[kk]):
                out.append(groups[kk][i])
                moved = True
                if len(out) == k:
                    break
        if not moved:
            break
        i += 1
    return out


@torch.no_grad()
def quick_eval(model, carrier, samples, blobs, feats, cfg, device, ledger,
               n_steps, alpha_mode, point_chunk, exact_source, tokens=0,
               sabotage=None) -> dict:
    """Held-out reconstruction error, only numbers, stratified as pre-registered."""
    n_pix = cfg.int_("eval", "pixels_per_sample")
    ledger.arm(len(samples))
    model.eval()
    recs: list[dict] = []
    for s in samples:
        row = json.loads(blobs[s["id"]])
        x0, y = load_pair(s["shard"], row, s["after_asset"])
        a = alpha_fields(row, x0)
        npx = x0.shape[0] * x0.shape[1]
        idx = (torch.arange(npx) if n_pix <= 0 else
               torch.arange(0, npx, max(1, npx // n_pix))[:n_pix])
        xb = x0.reshape(-1, 3)[idx].to(device).unsqueeze(0)
        yb = y.reshape(-1, 3)[idx].to(device).unsqueeze(0)
        ab = a.reshape(n_steps, -1)[:, idx].to(device).unsqueeze(0)
        mb = depth_mask(s["depth"], n_steps).to(device).unsqueeze(0)
        feat = gather_feats(feats, [feature_key(s)], tokens, device)

        params, w = model(feat)
        s_hat = compose_alpha(ab, mb, w, alpha_mode)
        pred = restore(carrier, yb, s_hat, params, point_chunk).float()

        cols: dict[str, dict] = {}
        if sabotage != "model":
            cols["model"] = ledger.stats("model", pred, xb, source="uint8_asset")
        if sabotage != "identity":
            cols["identity"] = ledger.stats("identity", yb, xb, source="uint8_asset")
        if sabotage != "exact_solve":
            cols["exact_solve"] = exact_column(s, idx, xb, ledger, exact_source,
                                               device)
        z = float((s_hat == 0).float().mean())
        recs.append(dict(id=s["id"], depth=s["depth"] or n_steps, geom=s["geom"],
                         rec_band=s["rec_band"], major=s["major"],
                         n_eval_pixels=int(idx.numel()), alpha_hat_zero_frac=z,
                         alpha_hat_mean=float(s_hat.mean()),
                         w=[float(x) for x in w[0].detach().cpu()], cols=cols))
    model.train()

    columns = ledger.columns
    overall = {c: {f: median_of(recs, c, f) for f in ("p50", "p95", "p99")}
               for c in columns}
    by: dict[str, dict] = {}
    for stratum in ledger.strata:
        groups: dict[str, list] = {}
        for r in recs:
            groups.setdefault(str(r[stratum]), []).append(r)
        by[stratum] = {k: dict(n=len(v),
                               **{c: {f: median_of(v, c, f) for f in ("p50", "p95")}
                                  for c in columns})
                       for k, v in sorted(groups.items())}
    out = dict(n=len(recs), metric="8-bit L-inf over RGB, |x_hat - before| * 255; "
                                   "per-sample p50/p95/p99, then the MEDIAN across "
                                   "evaluated samples",
               exact_solve_source=exact_source,
               column_missing={c: n for c, n in ledger.missing.items()},
               column_sources={c: sorted(v) for c, v in ledger.sources.items()},
               overall=overall, by=by, per_sample=recs)
    ledger.assert_wired(out)
    return out


def exact_column(s: dict, idx: torch.Tensor, xb: torch.Tensor, ledger: MetricLedger,
                 source: str, device: str) -> dict:
    """The upper-bound column, from one single provenance per run.

    ``journal``  the per-depth well-posedness certificate ``emits[].err_E_all``
                 (float domain, computed from the true ``y_d``; the production
                 reading, IMPLEMENTATION 2.1b).
    ``asset``    ``<id>.restored_e.png`` recomputed in the uint8 domain; only the
                 full-chain depth has such an asset.
    A sample with no upper bound is COUNTED, never dropped.
    """
    if source == "journal":
        e = s.get("exact_err")
        if not e or "p50" not in e:
            ledger.note("exact_solve", source="journal_certificate", missing=True)
            return dict(source="journal_certificate", missing=True)
        ledger.note("exact_solve", source="journal_certificate")
        return dict(e, source="journal_certificate")
    if source == "asset":
        name = f"{s['id']}.restored_e.png"
        try:
            raw = asset_bytes(s["shard"], name) if s["depth"] is None else None
        except (FileNotFoundError, KeyError):
            raw = None
        if raw is None:
            ledger.note("exact_solve", source="uint8_asset", missing=True)
            return dict(source="uint8_asset", missing=True)
        ex = torch.from_numpy(
            np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.float32)
            / 255.0
        ).reshape(-1, 3)[idx].to(device).unsqueeze(0)
        d = ledger.stats("exact_solve", ex, xb, source="uint8_asset")
        d["source"] = "uint8_asset"
        return d
    die(f"unknown [eval] exact_solve_source {source!r}")


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def lr_at(step: int, base: float, warmup: int, total: int, schedule: str) -> float:
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    if schedule == "cosine":
        t = (step - warmup) / max(1, total - warmup)
        return base * 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))
    if schedule == "constant":
        return base
    die(f"unknown [optim] schedule {schedule!r}")


def worker_init(_):
    torch.set_num_threads(1)


def derive_cadence(value: int | None, budget: int, divisor: int, name: str,
                   allow_zero: bool = False) -> int:
    """``"auto"`` -> ``max(1, budget // divisor)``; an explicit value passes through.

    The cadences must scale with the EFFECTIVE budget: a 2000-step log cadence on
    a 220-step budget writes one row and an 8 h unattended run has no curve.
    """
    if divisor <= 0:
        die(f"config: the auto divisor for {name} must be positive, got {divisor}")
    v = max(1, budget // divisor) if value is None else value
    if v == 0 and allow_zero:
        return 0
    if v < 1:
        die(f"{name} = {v} would never fire; give a positive value or \"auto\"")
    if v >= budget:
        die(f"{name} = {v} is not smaller than the effective step budget {budget}; "
            "the run would produce no intermediate row -- fix the config, the tool "
            "does not silently downgrade it")
    return v


def rng_state() -> dict:
    return dict(torch=torch.get_rng_state(),
                cuda=(torch.cuda.get_rng_state_all()
                      if torch.cuda.is_available() else []),
                numpy=np.random.get_state(), python=random.getstate())


def set_rng_state(st: dict) -> None:
    # the states must be CPU ByteTensors whatever device the checkpoint was
    # loaded onto (torch.load(map_location=cuda) would hand back CUDA tensors)
    torch.set_rng_state(st["torch"].cpu().to(torch.uint8))
    if st.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([t.cpu().to(torch.uint8) for t in st["cuda"]])
    np.random.set_state(st["numpy"])
    random.setstate(st["python"])


def append_jsonl(path: Path, rec: dict) -> None:
    """One line, flushed and fsynced: an 8 h unattended run must never lose the
    numbers it already produced because it was killed mid-write."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--verify-alpha", type=int, default=0,
                    help="assert on N chains that the alpha fields and the assets "
                         "agree with epr050_recovery_montage.rebuild")
    ap.add_argument("--sabotage-metric", default=None,
                    help="skip one pre-registered column on purpose; the run-time "
                         "assertion must fire (guard verification only)")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint to evaluate (default: ckpt_best then ckpt_last)")
    ap.add_argument("--allow-untrained", action="store_true",
                    help="--eval-only on a freshly initialised model (guard drills)")
    ap.add_argument("--resume", action="store_true",
                    help="continue from <out_dir>/ckpt_last.pt (model + optimiser "
                         "+ step + RNG)")
    ap.add_argument("--stop-after", type=int, default=0,
                    help="operational: stop after N steps of THIS process (a "
                         "resume drill), the step budget itself is unchanged")
    a = ap.parse_args()

    cfg = Cfg(Path(a.config))
    out_dir = Path(cfg.str_("run", "out_dir"))
    device = cfg.str_("run", "device")
    seed = cfg.int_("run", "seed")
    cfg_max_steps = cfg.int_("run", "max_steps")
    one_pass_cap = cfg.bool_("run", "one_pass_cap")
    # cadences may be "auto": derived from the EFFECTIVE budget further down,
    # once n_train and the batch size are known.
    cfg_log_every = cfg.int_or_auto("run", "log_every")
    cfg_eval_every = cfg.int_or_auto("run", "eval_every")
    cfg_ckpt_every = cfg.int_or_auto("run", "ckpt_every")
    log_div = cfg.int_("run", "auto_log_divisor")
    eval_div = cfg.int_("run", "auto_eval_divisor")
    # extra FULL held-out evals at explicit steps, on top of the derived
    # cadence: this is how a run whose budget is 30k lands a headline-comparable
    # table exactly on the step another run stopped at (step-matched, U4).
    eval_at = sorted({int(x) for x in cfg.list_("run", "eval_at", int)})
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False

    # ---- frozen provenance (M7) -------------------------------------------- #
    build_tool_sha = sha256_file(Path(B.__file__))
    g4d_sha = sha256_file(Path(G4DMOD.__file__))
    allowed = set(cfg.list_("guard", "build_tool_sha256_allowed"))
    if build_tool_sha not in allowed:
        die(f"epr050_build_degradation.py sha {build_tool_sha[:12]} is not in "
            "[guard] build_tool_sha256_allowed -- the alpha rebuild would run on "
            "a different data law than the one that built the shards")
    want_g4d = cfg.str_("guard", "g4d_sha256")
    if g4d_sha != want_g4d:
        die(f"q3vl/whatb/arms/g4d.py sha {g4d_sha[:12]} != [guard] g4d_sha256 "
            f"{want_g4d[:12]} -- the carrier changed under the run")

    global ASSET_MODE
    ASSET_MODE = cfg.str_("data", "asset_source", ("dir", "tar", "auto"))
    samples, blobs, index_meta = load_shards(cfg)
    law = bind_build_config(samples, cfg.list_("guard", "build_config_sha256_allowed"))
    n_steps = law["n_steps"]
    train = [s for s in samples if not s["heldout"]]
    held = [s for s in samples if s["heldout"]]
    if not held:
        die("held-out split is empty; check [data] split_salt / holdout_bucket_min")
    train.sort(key=lambda s: (s["id"], s["depth"] or 0))
    held.sort(key=lambda s: (s["id"], s["depth"] or 0))
    print(f"samples: {len(samples)} ({len(train)} train / {len(held)} held-out), "
          f"{index_meta['n_chains']} chains, compact index "
          f"{index_meta['compact_index_bytes'] / 2 ** 20:.1f} MiB "
          f"({index_meta['compact_bytes_per_chain']} B/chain), chain length "
          f"{n_steps}, step_order {law['step_kind']}", flush=True)

    if a.verify_alpha:
        verify_alpha(samples, blobs, law, a.verify_alpha, device)

    inputs = cfg.list_("encoder", "inputs")
    for w in inputs:
        if w not in ("before", "after"):
            die(f"[encoder] inputs may only contain before/after, got {w!r}")
    feats, cache_meta = build_feature_cache(cfg, samples, device, inputs)
    tokens = int(cache_meta["tokens"])
    # token cache: the L axis already carries both inputs, so in_dim is the
    # per-token width; pooled cache: the inputs are concatenated into in_dim.
    in_dim = cache_meta["dim"] * (1 if tokens else len(inputs))

    alpha_mode = cfg.str_("model", "alpha_hat", ("wsum_pred", "prod"))
    point_chunk = cfg.int_("model", "point_chunk")
    model = Stage0Model(in_dim, cfg, n_steps, tokens=tokens).to(device)
    carrier = Glut4DCarrier(model.g4d_cfg).to(device)
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    lr = cfg.num("optim", "lr")
    wd = cfg.num("optim", "weight_decay")
    betas = tuple(cfg.list_("optim", "betas", float))
    batch = cfg.int_("optim", "batch")
    cfg_warmup = cfg.int_or_auto("optim", "warmup_steps")
    warmup_frac = cfg.num("optim", "warmup_frac")
    warmup_max = cfg.int_("optim", "warmup_max")
    schedule = cfg.str_("optim", "schedule", ("cosine", "constant"))
    clip = cfg.num("optim", "grad_clip")
    amp = cfg.str_("optim", "autocast", ("bf16", "fp16", "off"))
    scope = cfg.str_("optim", "autocast_scope", ("trunk_only", "trunk_and_head", "off"))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=betas)

    # "~30k step or one pass over the data, whichever comes first" (editor's
    # ruling): the literal reading, and the cosine schedule runs to it.
    one_pass = math.ceil(len(train) / batch)
    max_steps = min(cfg_max_steps, one_pass) if one_pass_cap else cfg_max_steps
    # every cadence and the warm-up are fractions of the EFFECTIVE budget, so a
    # budget that shrinks with the data does not silently produce a curve with
    # one point (or a warm-up longer than the run).
    log_every = derive_cadence(cfg_log_every, max_steps, log_div, "[run] log_every")
    eval_every = derive_cadence(cfg_eval_every, max_steps, eval_div,
                                "[run] eval_every")
    ckpt_every = derive_cadence(cfg_ckpt_every, max_steps, eval_div,
                                "[run] ckpt_every", allow_zero=True)
    warmup = (min(warmup_max, int(warmup_frac * max_steps)) if cfg_warmup is None
              else cfg_warmup)
    if warmup >= max_steps:
        die(f"[optim] warmup_steps = {warmup} is not smaller than the effective "
            f"budget {max_steps}; the cosine leg would never run")
    for e in eval_at:
        if e < 1 or e > max_steps:
            die(f"[run] eval_at contains {e}, which is outside 1..{max_steps} "
                "(the effective budget); a step-matched eval that can never run "
                "is not a criterion")
    cadence = dict(effective_max_steps=max_steps,
                   log_every=log_every, eval_every=eval_every,
                   ckpt_every=ckpt_every, warmup_steps=warmup, eval_at=eval_at,
                   config=dict(log_every=cfg_log_every, eval_every=cfg_eval_every,
                               ckpt_every=cfg_ckpt_every, warmup_steps=cfg_warmup,
                               auto_log_divisor=log_div, auto_eval_divisor=eval_div,
                               warmup_frac=warmup_frac, warmup_max=warmup_max))

    columns = cfg.list_("eval", "columns")
    strata = cfg.list_("eval", "strata")
    ledger = MetricLedger(columns, strata)
    select_metric = cfg.str_("eval", "select_metric", SELECT_METRICS)
    exact_source = cfg.str_("eval", "exact_solve_source", ("journal", "asset"))
    interval_n = cfg.int_("eval", "interval_max_samples")
    final_n = cfg.int_("eval", "final_max_samples")
    subset_salt = cfg.str_("eval", "subset_salt")
    interval_set = eval_subset(held, interval_n, subset_salt)
    final_set = eval_subset(held, final_n, subset_salt)

    out_dir.mkdir(parents=True, exist_ok=True)
    tool_sha = sha256_file(Path(__file__))
    started = time.strftime("%Y%m%dT%H%M%S%z")
    # provenance is append-only: a resume / --eval-only must not overwrite the
    # manifest of the process that produced the checkpoints.
    rap = out_dir / "run_args.json"
    if rap.exists():
        rap = out_dir / f"run_args.{started}.json"
    rap.write_text(json.dumps(dict(
        epr="EPR-051/stage0", tool=str(Path(__file__).resolve()),
        tool_sha256=tool_sha, config_path=str(cfg.path), config_sha256=cfg.sha256,
        config=cfg.d, data_law=law, index=index_meta, encoder=cache_meta,
        frozen_sha256=dict(train_stage0=tool_sha,
                           epr050_build_degradation=build_tool_sha,
                           q3vl_whatb_arms_g4d=g4d_sha),
        invocation=dict(resume=bool(a.resume), eval_only=bool(a.eval_only),
                        stop_after=int(a.stop_after), ckpt=a.ckpt,
                        sabotage_metric=a.sabotage_metric),
        schedule=dict(config_max_steps=cfg_max_steps, one_pass_steps=one_pass,
                      one_pass_cap=one_pass_cap, effective_max_steps=max_steps,
                      cadence=cadence),
        eval=dict(select_metric=select_metric, exact_solve_source=exact_source,
                  interval_n=len(interval_set), final_n=len(final_set),
                  subset_rule="strata (geom, rec_band, depth) round-robin, "
                              f'within-stratum order sha1("{subset_salt}:"+key)'),
        model=dict(in_dim=in_dim, tokens=tokens, readout=model.readout_mode,
                   theta_dim=model.theta_dim,
                   trainable_params=n_train_params, heads=1,
                   alpha_hat=alpha_mode, g4d=model.g4d_cfg.as_dict()),
        started=started), ensure_ascii=False, indent=1))
    print(f"model: readout={model.readout_mode} in_dim={in_dim} tokens={tokens} "
          f"theta_dim={model.theta_dim} "
          f"trainable={n_train_params}; step budget {max_steps} "
          f"(config {cfg_max_steps}, one pass {one_pass}, cap={one_pass_cap}); "
          f"cadence log/eval/ckpt = {log_every}/{eval_every}/{ckpt_every}, "
          f"warmup {warmup}; eval interval/final = {len(interval_set)}/"
          f"{len(final_set)} of {len(held)}; run_args -> {rap.name}", flush=True)

    def run_eval(step: int, which: str, sabotage=None) -> dict:
        subset = interval_set if which == "interval" else final_set
        r = quick_eval(model, carrier, subset, blobs, feats, cfg, device, ledger,
                       n_steps, alpha_mode, point_chunk, exact_source,
                       tokens=tokens, sabotage=sabotage)
        r["step"] = step
        r["eval_kind"] = which
        return r

    def save_ckpt(path: Path, step: int) -> None:
        _atomic_save(dict(step=step, model=model.state_dict(),
                          optimizer=opt.state_dict(), rng=rng_state(),
                          config_sha256=cfg.sha256, tool_sha256=tool_sha), path)

    if a.eval_only:
        cp = Path(a.ckpt) if a.ckpt else None
        if cp is None:
            for name in ("ckpt_best.pt", "ckpt_last.pt"):
                if (out_dir / name).exists():
                    cp = out_dir / name
                    break
        if cp is None:
            if not a.allow_untrained:
                die("--eval-only found no checkpoint; pass --ckpt or "
                    "--allow-untrained (a randomly initialised eval is a drill, "
                    "not a number)")
        else:
            blob = torch.load(cp, map_location="cpu", weights_only=False)
            model.load_state_dict(blob["model"])
            print(f"loaded {cp} (step {blob.get('step')})", flush=True)
        r = run_eval(0, "final", sabotage=a.sabotage_metric)
        (out_dir / "metrics_eval_only.json").write_text(
            json.dumps(r, ensure_ascii=False, indent=1))
        print_eval(r)
        return

    n_pix = cfg.int_("loss", "pixels_per_sample")
    ds = PrefixPixels(train, blobs, n_pix, n_steps)
    nw = cfg.int_("data", "num_workers")
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=nw,
                    collate_fn=collate, drop_last=len(train) > batch,
                    worker_init_fn=worker_init,
                    persistent_workers=nw > 0, pin_memory=True)
    heldout_ids = {s["id"] for s in held}

    steps_path = out_dir / "steps.jsonl"
    inter_path = out_dir / "metrics_intermediate.jsonl"
    step = 0
    best_key = None
    if a.resume:
        cp = out_dir / "ckpt_last.pt"
        if not cp.exists():
            die(f"--resume: {cp} does not exist")
        blob = torch.load(cp, map_location="cpu", weights_only=False)
        if blob["config_sha256"] != cfg.sha256:
            die("--resume: the checkpoint was written under a different config")
        model.load_state_dict(blob["model"])
        opt.load_state_dict(blob["optimizer"])
        set_rng_state(blob["rng"])
        step = int(blob["step"])
        if inter_path.exists():
            done = [json.loads(x) for x in open(inter_path)]
            # ONLY the interval rows: selection runs on one population for the
            # whole run, and an `eval_at` row is a FULL held-out table (a
            # different, easier/harder population) that must not become the
            # best-so-far key on resume.
            keys = [d["overall"]["model"]["p50"] for d in done
                    if "overall" in d and d.get("eval_kind") == "interval"]
            best_key = min(keys) if keys else None
        # the `seconds` and `epoch` axes restart with this process; the break is
        # marked in the log rather than reconstructed.
        append_jsonl(steps_path, dict(
            event="resume", from_step=step, started=started,
            run_args=rap.name, ckpt=str(cp),
            note="seconds/epoch restart at this line; step is continuous"))
        print(f"resumed from {cp} at step {step} "
              f"(best {select_metric} so far {best_key})", flush=True)
    elif steps_path.exists():
        die(f"{steps_path} already exists and --resume was not given; move it "
            "aside or resume (D-20: 'file exists' is not noise)")

    # the extra step-matched evals still owed by THIS process; anything already
    # behind the resume point is recorded as skipped rather than silently lost.
    eval_at_todo = {e for e in eval_at if e > step}
    eval_at_skipped = [e for e in eval_at if e <= step]
    if eval_at_skipped:
        print(f"[run] eval_at {eval_at_skipped} is at or behind the resume step "
              f"{step}; those evals are in the earlier process's "
              "metrics_intermediate.jsonl", flush=True)

    t0 = time.time()
    stop_at = step + a.stop_after if a.stop_after else None
    history: list[dict] = []
    epoch = 0
    interrupted = False
    try:
        while step < max_steps and not interrupted:
            for b, feat in prefetched(dl, feats, tokens, device):
                if step >= max_steps or interrupted:
                    break
                for i in b["index"].tolist():          # the split is asserted, not assumed
                    if train[i]["id"] in heldout_ids:
                        die(f"held-out id {train[i]['id']} entered the training batch")
                for g in opt.param_groups:
                    g["lr"] = lr_at(step, lr, warmup, max_steps, schedule)
                x = b["x"].to(device, non_blocking=True)
                y = b["y"].to(device, non_blocking=True)
                al = b["alpha"].to(device, non_blocking=True)
                mk = b["mask"].to(device, non_blocking=True)
                # `feat` came from the prefetch thread one step early; it is
                # bit-for-bit gather_feats(feats, b["key"], tokens, device).
                amp_dt = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp)
                trunk_ctx = (torch.autocast("cuda", dtype=amp_dt)
                             if amp_dt is not None and scope != "off"
                             else torch.autocast("cuda", enabled=False))
                if scope == "trunk_and_head":
                    with trunk_ctx:
                        params, w = model(feat)
                else:                                   # trunk_only / off
                    with trunk_ctx:
                        h, wf = model.trunk_forward(feat)
                    h, wf = h.float(), wf.float()
                    params, w = model.heads(h, wf)
                s_hat = compose_alpha(al, mk, w, alpha_mode)
                pred = restore(carrier, y, s_hat, params, point_chunk)
                loss = (pred - x).abs().mean()
                if not torch.isfinite(loss):
                    die(f"step {step}: loss is {float(loss)} -- NaN/Inf stops the run")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip))
                if not math.isfinite(gn):
                    die(f"step {step}: grad norm is {gn} -- NaN/Inf stops the run")
                opt.step()
                step += 1

                if step % log_every == 0 or step == 1 or step == max_steps:
                    with torch.no_grad():
                        e = linf8(pred.detach().float(), x)
                        rec = dict(step=step, epoch=epoch, loss=float(loss),
                                   train_linf8_p50=quantile(e, 0.5),
                                   train_linf8_p95=quantile(e, 0.95),
                                   lr=opt.param_groups[0]["lr"], grad_norm=gn,
                                   alpha_hat_mean=float(s_hat.mean()),
                                   w_mean=[float(v) for v in w.mean(0).detach().cpu()],
                                   seconds=round(time.time() - t0, 1),
                                   gpu_peak_gb=round(
                                       torch.cuda.max_memory_allocated() / 2 ** 30, 3))
                        history.append(rec)
                    append_jsonl(steps_path, rec)
                    print(f"  step {step}/{max_steps} loss={rec['loss']:.6f} "
                          f"L8p50={rec['train_linf8_p50']:.3f} lr={rec['lr']:.2e} "
                          f"gn={gn:.3f} peak={rec['gpu_peak_gb']}GB", flush=True)
                if step in eval_at_todo:
                    # a step-matched row against another run: FULL held-out, so it
                    # is comparable with that run's headline table and not with the
                    # interval subset (TASK_CARD 6 forbids mixing the two).  It is
                    # deliberately NOT used for checkpoint selection: selection runs
                    # on one population for the whole run.
                    eval_at_todo.discard(step)
                    r = run_eval(step, "final")
                    r["eval_at_match"] = True
                    print_eval(r)
                    append_jsonl(inter_path, {k: v for k, v in r.items()
                                              if k != "per_sample"})
                    (out_dir / f"metrics_step{step}.json").write_text(
                        json.dumps(r, ensure_ascii=False, indent=1))
                if eval_every and step % eval_every == 0:
                    r = run_eval(step, "interval")
                    print_eval(r)
                    append_jsonl(inter_path, {k: v for k, v in r.items()
                                              if k != "per_sample"})
                    key = r["overall"]["model"]["p50"]
                    if best_key is None or key < best_key:
                        best_key = key
                        save_ckpt(out_dir / "ckpt_best.pt", step)
                if ckpt_every and step % ckpt_every == 0:
                    save_ckpt(out_dir / "ckpt_last.pt", step)
                if stop_at is not None and step >= stop_at:
                    save_ckpt(out_dir / "ckpt_last.pt", step)
                    print(f"--stop-after: stopping at step {step}, ckpt_last written",
                          flush=True)
                    interrupted = True
                    break
            epoch += 1
    except ReadFailed as exc:
        # soft-mount / cache read that survived every retry: land the
        # results we already have, then stop.  Never train on a partial
        # read, and never leave the run without a resumable checkpoint.
        save_ckpt(out_dir / "ckpt_last.pt", step)
        append_jsonl(steps_path, dict(
            event="io_abort", step=step, started=started, error=str(exc),
            note="checkpoint written at this step; resume with --resume"))
        die(f"read error at step {step}: {exc}. ckpt_last.pt written at "
            f"step {step}; rerun with --resume once the mount is healthy")

    save_ckpt(out_dir / "ckpt_last.pt", step)
    if interrupted:
        print("interrupted by --stop-after; no final eval (resume to continue)",
              flush=True)
        return
    # pre-registered criterion with a run-time assertion: a step-matched eval
    # that was asked for and never fired is a dead criterion, not a warning.
    if eval_at_todo:
        die(f"[run] eval_at {sorted(eval_at_todo)} never fired although the run "
            f"reached step {step} -- the step-matched comparison has no row")
    final = run_eval(step, "final", sabotage=a.sabotage_metric)
    print_eval(final)
    (out_dir / "metrics.json").write_text(json.dumps(dict(
        epr="EPR-051/stage0", generated=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        config_sha256=cfg.sha256, tool_sha256=tool_sha,
        frozen_sha256=dict(epr050_build_degradation=build_tool_sha,
                           q3vl_whatb_arms_g4d=g4d_sha),
        select_metric=select_metric, best_select_value=best_key,
        n_train=len(train), n_heldout=len(held), steps=step,
        effective_max_steps=max_steps, one_pass_steps=one_pass, cadence=cadence,
        run_args=rap.name,
        resumed_from_steps=[json.loads(x)["from_step"] for x in open(steps_path)
                            if '"event"' in x],
        wall_seconds=round(time.time() - t0, 1),
        gpu_peak_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 3),
        train_history=history, heldout=final), ensure_ascii=False, indent=1))
    print(f"metrics -> {out_dir / 'metrics.json'}", flush=True)


def print_eval(r: dict) -> None:
    print(f"\n=== held-out 8-bit L-inf, median across {r['n']} samples "
          f"({r.get('eval_kind')} eval, step {r.get('step')}) ===", flush=True)
    print(f"{'column':14s} {'p50':>9s} {'p95':>9s} {'p99':>9s}  missing")
    for c, v in r["overall"].items():
        print(f"{c:14s} {v['p50']:9.3f} {v['p95']:9.3f} {v['p99']:9.3f}"
              f"  {r['column_missing'].get(c, 0)}")
    for stratum, groups in r["by"].items():
        print(f"\n-- by {stratum}")
        for k, d in groups.items():
            cells = "  ".join(f"{c}={d[c]['p50']:.3f}/{d[c]['p95']:.3f}"
                              for c in r["overall"])
            print(f"   {k:14s} n={d['n']:<3d} {cells}")


def verify_alpha(samples: list[dict], blobs: dict, law: dict, n: int,
                 device: str) -> None:
    """Assert the light path == the verified rebuilder, on N chains."""
    import epr050_recovery_montage as M
    z = np.load(f"{B.BANK}/luts.npz")
    # pick the target chains FIRST, then parse only the shards they live in (and
    # only the lines that matter): a full-journal parse of every shard is minutes
    # of start-up on a 10w-chain training set.
    targets: dict[str, list[str]] = {}
    seen: set[str] = set()
    for s in samples:
        if s["id"] in seen or len(seen) >= n:
            continue
        seen.add(s["id"])
        targets.setdefault(s["shard"], []).append(s["id"])
    rows: dict[str, dict] = {}
    for sd, ids in targets.items():
        want = set(ids)
        for line in open(Path(sd) / "pairs.jsonl"):
            r = json.loads(line)
            if r["id"] in want:
                rows[r["id"]] = r
                want.discard(r["id"])
                if not want:
                    break
    checked = 0
    for s in samples:
        if s["id"] not in rows or checked >= n:
            continue
        row = rows.pop(s["id"])
        x0g, fg, ys, est, _ = M.rebuild(row, z, device, law["max_side"])
        x0, y = load_pair(s["shard"], json.loads(blobs[s["id"]]),
                          f"{s['id']}.after.png")
        af = alpha_fields(json.loads(blobs[s["id"]]), x0)
        h, w = row["size"]
        d_alpha = max(float((g.cpu() - c).abs().max()) for g, c in zip(fg, af))
        d_x0 = float((x0g.cpu() - x0).abs().max())
        after_ok = np.array_equal(
            np.asarray(B.to_png(ys[-1].reshape(h, w, 3))),
            (y.numpy() * 255).round().astype(np.uint8))
        print(f"  verify-alpha {s['id']}: alpha_maxdiff={d_alpha:.3e} "
              f"src_maxdiff={d_x0:.3e} after_bit_exact={after_ok}", flush=True)
        if d_x0 != 0.0 or not after_ok:
            die(f"{s['id']}: the assets disagree with the rebuilder")
        if d_alpha > 1e-5:
            die(f"{s['id']}: alpha fields drifted by {d_alpha} from the rebuilder")
        checked += 1
    print(f"  verify-alpha: {checked} chains agree with "
          "epr050_recovery_montage.rebuild", flush=True)


if __name__ == "__main__":
    main()
