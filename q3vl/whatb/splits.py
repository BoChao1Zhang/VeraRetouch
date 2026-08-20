"""Split indices, records, and the B3 bucket pools.

Read-only access to ``sft2seg-20260804``.  Two levels:

* the **index** (``splits/<split>.index.jsonl``) carries ``sample_id``,
  ``lut_id``, ``source_image_id``, ``task_type``, ``winner_confidence`` and the
  tar coordinates of the image and the record.  Everything the training set is
  defined by (``split == "train" and winner_confidence == "normal"``) is in the
  index, so an epoch can be planned without opening a shard.

  There is more than one index口径 (:class:`DatasetVersion`,
  :data:`DATASET_VERSIONS`): ``v20260804`` is the published index (train normal
  n = 93,934) and ``cut-p45`` is that index with the small-area band / radial /
  semantic masks dropped (n = 80,269).  Both point at the *same* shards and keep
  the same ``sha1`` split assignment; they differ only in which index lines
  survive.  ``cut-p45`` is the default (:data:`DEFAULT_DATASET_VERSION`) and a
  runner selects the other with ``--dataset-version v20260804``.  Every
  ``root=`` parameter below defaults to the active口径.
* the **record** (``<sample_id>.rec.json`` inside ``records/shards``) carries
  ``instruction`` / ``where`` / ``color`` / ``major`` / ``minor`` /
  ``preset_path`` / ``mask_id``.  ``minor`` is what the ``B3_bucket_retrieval``
  bucket is -- the frozen block says to take it from the record and **not** from
  ``tools/data_splits/splits_presets.csv`` (that CSV's ``major`` disagrees with
  the record's on 706 of 800 sampled rows).

Paths in the index point at ``/mnt/nfs`` (the hard mount).  Reads are rewritten
to ``/mnt/nfs-ro`` by :func:`ro_path`: the campaign rule is that reads go to the
soft mount and only writes go through ``nfsx`` on the hard one.

A second training source: L8
----------------------------
``prod-l8-local400k-20260812`` has no split index of its own -- its rows live in
``l8_train.manifest.jsonl``, one JSON object per usable sample, written by the z
cache producer (``zcache_l8/_src/build_manifest_l8.py``).  The manifest row is
already the *whole* index row: ``sample_id`` / ``lut_id`` / ``source_image_id``
/ ``task_type`` / ``winner_confidence`` plus the ``where`` / ``color`` texts that
``q3vl.data.twoseg.convert`` produced from the seven-tag reasoning, so
:class:`IndexRow` carries them inline and no record shard has to be opened for
an L8 row (see :attr:`IndexRow.color_text`).

**The training population is measured, never written down.**
:func:`train_normal_rows` counts each source at start-up and asserts it against
that source's *own* on-disk declaration -- the sft2seg part against the active
口径's :attr:`DatasetVersion.normal_n`, the L8 part against
``l8_train.manifest.report.json``.  Adding a third source therefore adds a
source, not a new literal to keep in sync.  L8 is **not** filtered by any
sft2seg口径: its rows are addressed by ``l8_train.manifest.jsonl``, which no
口径 touches.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

__all__ = [
    "DATASET_ROOT",
    "SFT2SEG_SHARD_ROOT",
    "SPLITS",
    "TRAIN_NORMAL_N",
    "DatasetVersion",
    "DATASET_VERSIONS",
    "DATASET_VERSION_CHOICES",
    "DEFAULT_DATASET_VERSION",
    "dataset_version",
    "version_for_root",
    "active_dataset_version",
    "active_root",
    "use_dataset_version",
    "dataset_version_facts",
    "dataset_available",
    "L8_MANIFEST",
    "L8_ZCACHE_ROOT",
    "L8_SPLIT",
    "DATA_SOURCES",
    "DATA_CHOICES",
    "IndexRow",
    "ro_path",
    "split_index_path",
    "load_index",
    "load_index_cached",
    "normal_only",
    "read_record",
    "iter_records",
    "color_texts_of",
    "bucket_pools",
    "split_facts",
    "l8_manifest_report",
    "load_l8_index",
    "extra_train_rows",
    "train_normal_rows",
    "train_normal_n",
    "train_source_facts",
]

SPLITS: tuple[str, ...] = ("train", "V_what", "V_where", "T_final", "T_lut_unseen")

#: soft (read-only) mount that holds the record / image / mask shards.  Every
#: index口径 points its rows at these shards (the ``members`` paths are absolute),
#: so a filtered index directory carries ``splits/`` and nothing else.
SFT2SEG_SHARD_ROOT = Path("/mnt/nfs-ro/bc/data/datasets/sft2seg-20260804")


# --------------------------------------------------------------------------- #
# the dataset口径 (index version)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DatasetVersion:
    """One index口径: a ``splits/`` directory plus the counts measured in it.

    A口径 is *not* a new dataset.  Every version indexes the same shards
    (:data:`SFT2SEG_SHARD_ROOT`); they differ only in which rows the index keeps,
    and the ``sha1`` split rule family is untouched (no row changes split).

    ``n`` / ``normal_n`` are **measured** off the index files and written down
    here so :func:`train_normal_rows` has something to assert against -- the same
    role the single ``TRAIN_NORMAL_N`` literal used to play, one per口径.
    """

    name: str
    root: Path
    #: split -> rows in ``<root>/splits/<split>.index.jsonl``
    n: Mapping[str, int]
    #: split -> rows with ``winner_confidence == "normal"``
    normal_n: Mapping[str, int]
    rule: str
    measured_on: str
    parent: str | None = None
    excluded_ids: Path | None = None
    excluded_n: int = 0
    excluded_sha256: str | None = None

    @property
    def train_normal_n(self) -> int:
        return int(self.normal_n["train"])

    def facts(self) -> dict[str, Any]:
        """The block ``run_setup.json`` carries so two boards cannot be confused."""
        return {
            "name": self.name,
            "root": str(self.root),
            "parent": self.parent,
            "rule": self.rule,
            "measured_on": self.measured_on,
            "shard_root": str(SFT2SEG_SHARD_ROOT),
            "n": dict(self.n),
            "normal_n": dict(self.normal_n),
            "train_normal_n": self.train_normal_n,
            "excluded_ids": (str(self.excluded_ids) if self.excluded_ids else None),
            "excluded_n": int(self.excluded_n),
            "excluded_sha256": self.excluded_sha256,
        }


#: every registered index口径.  Counts measured with ``wc -l`` / ``jq`` on the
#: index files on 2026-08-18; ``excluded_sha256`` is ``sha256sum`` of that口径's
#: ``excluded_sample_ids.txt``.
DATASET_VERSIONS: dict[str, DatasetVersion] = {
    "v20260804": DatasetVersion(
        name="v20260804",
        root=SFT2SEG_SHARD_ROOT,
        n={"train": 159215, "V_what": 897, "V_where": 896,
           "T_final": 918, "T_lut_unseen": 433},
        normal_n={"train": 93934, "V_what": 567, "V_where": 515,
                  "T_final": 533, "T_lut_unseen": 252},
        rule="unfiltered -- the index sft2seg-20260804 was published with",
        measured_on="2026-08-18 (train normal 93934 = the frozen block's item 1)",
    ),
    "cut-p45": DatasetVersion(
        name="cut-p45",
        root=Path("/home/bc/data/datasets/sft2seg-20260804-cut-p45"),
        parent="v20260804",
        n={"train": 135697, "V_what": 777, "V_where": 781,
           "T_final": 806, "T_lut_unseen": 371},
        normal_n={"train": 80269, "V_what": 496, "V_where": 430,
                  "T_final": 483, "T_lut_unseen": 221},
        rule=("drop family==band & alpha_mean<0.4695485997094125, "
              "radial<0.26050587805813546, semantic<0.10028177653808375; "
              "linear unthresholded.  area = GT alpha mean.  Kept lines are "
              "byte-identical to the parent's and no row changed split."),
        measured_on="2026-08-18 (index built 2026-08-18 10:19)",
        excluded_ids=Path("/home/bc/data/datasets/sft2seg-20260804-cut-p45/"
                          "excluded_sample_ids.txt"),
        excluded_n=23927,
        excluded_sha256=("28d645dce0ea32e56c4a23773feee74e0b807dcbea4"
                         "da5b8ae8b7ec250216563"),
    ),
}

DATASET_VERSION_CHOICES: tuple[str, ...] = tuple(DATASET_VERSIONS)

#: the口径 every runner uses unless ``--dataset-version`` says otherwise
DEFAULT_DATASET_VERSION = "cut-p45"

#: the default口径's index root.  Kept under the old name because that is what
#: ``--dataset-root`` defaults, ``codec/lutcode`` and the test gates read; the
#:口径 a call actually resolves to is :func:`active_root`.
DATASET_ROOT = DATASET_VERSIONS[DEFAULT_DATASET_VERSION].root

#: frozen block item 1.  No longer the assertion -- it is the *original*口径's
#: train count, kept so the "frozen block" records in the runners stay pinned to
#: the number the published boards were run on.  The assertion is per口径
#: (:attr:`DatasetVersion.normal_n`).
TRAIN_NORMAL_N = DATASET_VERSIONS["v20260804"].train_normal_n

_ACTIVE_VERSION: DatasetVersion = DATASET_VERSIONS[DEFAULT_DATASET_VERSION]
_INDEX_READ = False


def dataset_version(name: str) -> DatasetVersion:
    """The registered口径 called ``name``."""
    try:
        return DATASET_VERSIONS[str(name)]
    except KeyError:
        raise ValueError(
            f"unknown dataset version {name!r}; registered: "
            f"{list(DATASET_VERSION_CHOICES)}") from None


def version_for_root(root: str | Path) -> DatasetVersion:
    """The口径 whose ``root`` is ``root`` -- an unregistered root is a refusal.

    A root nobody measured has no ``n`` to assert against, and an index without
    an expected count is exactly the data drift the assertion exists to catch.
    """
    p = Path(root).resolve()
    for ver in DATASET_VERSIONS.values():
        if Path(ver.root).resolve() == p:
            return ver
    raise AssertionError(
        f"{root} is not a registered dataset version.  Registered roots: "
        + ", ".join(f"{v.name}={v.root}" for v in DATASET_VERSIONS.values())
        + ".  Register it in q3vl/whatb/splits.py DATASET_VERSIONS with its "
          "measured per-split n before training or scoring on it.")


def active_dataset_version() -> DatasetVersion:
    """The口径 this process resolves ``root=None`` to."""
    return _ACTIVE_VERSION


def active_root() -> Path:
    return Path(_ACTIVE_VERSION.root)


def use_dataset_version(name: str | DatasetVersion, *, force: bool = False
                        ) -> DatasetVersion:
    """Select the process-wide口径.  Called once, before any index is read.

    Switching after an index has already been parsed is refused: the parsed rows
    are cached per ``(split, root)`` but the *derived* numbers a runner has
    already computed are not, so a mid-run switch would silently mix two口径.
    ``force=True`` (tests) drops the parse cache and allows it.
    """
    global _ACTIVE_VERSION, _INDEX_READ
    ver = name if isinstance(name, DatasetVersion) else dataset_version(name)
    if ver.name == _ACTIVE_VERSION.name:
        return ver
    if _INDEX_READ and not force:
        raise AssertionError(
            f"the active dataset version is already {_ACTIVE_VERSION.name} and an "
            f"index has been read from it; refusing to switch to {ver.name} "
            "mid-process")
    _ACTIVE_VERSION = ver
    _INDEX_READ = False
    _load_index_cached.cache_clear()
    return ver


def _resolve_root(root: str | Path | None) -> Path:
    return active_root() if root is None else Path(root)


def dataset_version_facts(root: str | Path | None = None) -> dict[str, Any]:
    """:meth:`DatasetVersion.facts` of the口径 ``root`` names (default: active)."""
    return version_for_root(_resolve_root(root)).facts()


def dataset_available(root: str | Path | None = None) -> bool:
    """True when both the index口径 and the shard mount are readable.

    Test gates need both: a口径 root can be on local disk while the records and
    images it points at still live on the soft mount.
    """
    try:
        return (split_index_path("train", root).is_file()
                and SFT2SEG_SHARD_ROOT.is_dir())
    except (OSError, ValueError):
        return False

#: the L8 manifest written next to its z cache (one JSON object per usable row)
L8_MANIFEST = Path("/home/bc/data/runs/whatb/zcache_l8/l8_train.manifest.jsonl")

#: root of the L8 z cache; the two contexts live under ``generated/`` / ``teacher/``
L8_ZCACHE_ROOT = Path("/home/bc/data/runs/whatb/zcache_l8")

#: the split namespace the L8 rows and their z cache are addressed by
L8_SPLIT = "l8_train"

#: ``--data`` -> the extra sources unioned onto the frozen sft2seg train split
DATA_SOURCES: dict[str, tuple[str, ...]] = {
    "v2seg": (),
    "v2seg+l8": ("l8",),
}

DATA_CHOICES: tuple[str, ...] = tuple(DATA_SOURCES)


def ro_path(path: str | Path) -> Path:
    """Rewrite a ``/mnt/nfs`` path to the soft read-only mount."""
    s = str(path)
    if s.startswith("/mnt/nfs/"):
        return Path("/mnt/nfs-ro/" + s[len("/mnt/nfs/"):])
    return Path(s)


def split_index_path(split: str, root: str | Path | None = None) -> Path:
    """``root`` defaults to the active口径 (:func:`active_root`)."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    return _resolve_root(root) / "splits" / f"{split}.index.jsonl"


@dataclass(frozen=True)
class IndexRow:
    """One index line, with the fields whatb reads named explicitly."""

    sample_id: str
    split: str
    lut_id: str
    source_image_id: str
    task_type: str
    winner_confidence: str
    raw: Mapping[str, Any]

    @classmethod
    def from_json(cls, obj: Mapping[str, Any]) -> "IndexRow":
        return cls(sample_id=str(obj["sample_id"]), split=str(obj.get("split", "")),
                   lut_id=str(obj.get("lut_id", "")),
                   source_image_id=str(obj.get("source_image_id", "")),
                   task_type=str(obj.get("task_type", "")),
                   winner_confidence=str(obj.get("winner_confidence", "")),
                   raw=obj)

    @property
    def is_normal(self) -> bool:
        return self.winner_confidence == "normal"

    @property
    def is_style(self) -> bool:
        """``style`` = global edit; ``alpha == 1`` everywhere (no mask member)."""
        return self.task_type == "style"

    @property
    def has_record(self) -> bool:
        """True when the row addresses a ``*.rec.json`` in a shard.

        False for an L8 manifest row: L8 publishes no record shard, and the two
        fields whatb reads off a record (``color`` for the colour-span assertion,
        ``minor`` for the B3 bucket) are carried inline instead / not at all.
        """
        return "record" in (self.raw.get("members") or {})

    @property
    def color_text(self) -> str | None:
        """The ``<color>`` segment when the row carries it inline, else ``None``.

        The L8 manifest stores what ``q3vl.data.twoseg.convert`` produced (spec
        4.2: ``where`` = the region_scope body, ``color`` = the other six
        segments in order).  Reusing it is the single-source rule -- two
        conversions of the same reasoning is how a text cache and a z cache
        drift apart.
        """
        v = self.raw.get("color")
        return str(v) if v else None

    @property
    def source(self) -> str:
        """``"v2seg"`` or ``"l8"`` -- which corpus this row came from."""
        return str(self.raw.get("_source", "v2seg"))


def load_index(split: str, root: str | Path | None = None) -> list[IndexRow]:
    """All rows of one split's index, in file order (active口径 by default)."""
    global _INDEX_READ
    path = split_index_path(split, root)
    rows: list[IndexRow] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(IndexRow.from_json(json.loads(line)))
    _INDEX_READ = True
    return rows


@lru_cache(maxsize=None)
def _load_index_cached(split: str, root: Path) -> tuple[IndexRow, ...]:
    return tuple(load_index(split, root))


def load_index_cached(split: str, root: str | Path | None = None
                      ) -> tuple[IndexRow, ...]:
    """:func:`load_index`, parsed once per process, keyed by the resolved root.

    ``train.index.jsonl`` is 159,215 lines (v20260804) / 135,697 (cut-p45); the
    training population, the B3 bucket pool and ``Lib_tr`` all read it, and a run
    that parses it three times pays for it three times.
    """
    return _load_index_cached(split, _resolve_root(root))


load_index_cached.cache_clear = _load_index_cached.cache_clear   # type: ignore[attr-defined]
load_index_cached.cache_info = _load_index_cached.cache_info     # type: ignore[attr-defined]


def normal_only(rows: Iterable[IndexRow]) -> list[IndexRow]:
    """``winner_confidence == "normal"`` -- the only rows that may be trained on
    or scored as GT (campaign data discipline)."""
    return [r for r in rows if r.is_normal]


def read_record(row: IndexRow | Mapping[str, Any], *, verify: bool = False
                ) -> dict[str, Any]:
    """The ``*.rec.json`` of one index row, read by (shard, offset, length)."""
    obj = row.raw if isinstance(row, IndexRow) else row
    member = obj["members"]["record"]
    path = ro_path(member["shard"])
    with path.open("rb") as fh:
        fh.seek(int(member["offset"]))
        blob = fh.read(int(member["length"]))
    if len(blob) != int(member["length"]):
        raise IOError(f"short read of {path}:{member['offset']}")
    if verify and member.get("sha256"):
        got = hashlib.sha256(blob).hexdigest()
        if got != member["sha256"]:
            raise IOError(f"checksum mismatch for {path}:{member['offset']}")
    return json.loads(blob)


def iter_records(rows: Sequence[IndexRow], *, verify: bool = False
                 ) -> Iterator[dict[str, Any]]:
    """Records of ``rows``, in order.  One open fd per shard, reused."""
    handles: dict[str, Any] = {}
    try:
        for row in rows:
            member = row.raw["members"]["record"]
            key = str(member["shard"])
            fh = handles.get(key)
            if fh is None:
                fh = handles[key] = ro_path(key).open("rb")
            fh.seek(int(member["offset"]))
            blob = fh.read(int(member["length"]))
            if verify and member.get("sha256"):
                if hashlib.sha256(blob).hexdigest() != member["sha256"]:
                    raise IOError(f"checksum mismatch for {key}")
            yield json.loads(blob)
    finally:
        for fh in handles.values():
            fh.close()


def color_texts_of(rows: Sequence[IndexRow]) -> list[str]:
    """The ``<color>`` text of each row, whatever source it came from.

    A row that carries the segment inline (L8) is read straight off the manifest;
    one that does not (sft2seg) is read from its record shard.  Empty texts are
    dropped -- the colour-span assertion compares encodings, and there is nothing
    to compare on an empty string.
    """
    inline = [r for r in rows if r.color_text]
    from_shard = [r for r in rows if not r.color_text and r.has_record]
    out = [str(r.color_text) for r in inline]
    out += [str(rec.get("color", "")) for rec in iter_records(from_shard)]
    return [t for t in out if t]


# --------------------------------------------------------------------------- #
# the second training source: L8
# --------------------------------------------------------------------------- #
def l8_manifest_report(manifest: str | Path = L8_MANIFEST) -> dict[str, Any]:
    """The producer's ``*.manifest.report.json`` next to the manifest."""
    path = Path(str(manifest).replace(".jsonl", "") + ".report.json")
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist; it is the L8 manifest's own declaration of "
            "how many rows it holds, and the row count is asserted against it "
            "rather than against a literal in this file")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def load_l8_index(manifest: str | Path = L8_MANIFEST) -> tuple[IndexRow, ...]:
    """The L8 manifest as :class:`IndexRow` s, counted against its own report.

    Both ``winner_confidence`` values are in the manifest on purpose (the two
    consumers disagree: Where-B keeps ``low``, whatb does not).  This returns
    every row; :func:`train_normal_rows` is what applies the campaign's
    normal-only rule.
    """
    path = Path(manifest)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist; --data v2seg+l8 needs the L8 manifest that "
            "the z cache producer wrote next to zcache_l8/")
    rows: list[IndexRow] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = dict(json.loads(line))
            obj["split"] = L8_SPLIT
            obj["_source"] = "l8"
            rows.append(IndexRow.from_json(obj))
    report = l8_manifest_report(path)
    counts = report.get("counts") or {}
    declared_n = int(report.get("n_manifest", -1))
    declared_normal = int(counts.get("usable_final_normal", -1))
    n_normal = sum(1 for r in rows if r.is_normal)
    if declared_n >= 0 and len(rows) != declared_n:
        raise AssertionError(
            f"{path}: {len(rows)} rows but the producer's report declares "
            f"{declared_n}.  A manifest that no longer matches its own report is "
            "a half-written manifest, not a bigger training set.")
    if declared_normal >= 0 and n_normal != declared_normal:
        raise AssertionError(
            f"{path}: {n_normal} winner_confidence=normal rows but the report "
            f"declares {declared_normal}")
    ids = {r.sample_id for r in rows}
    if len(ids) != len(rows):
        raise AssertionError(
            f"{path}: {len(rows)} rows but only {len(ids)} distinct sample_id; a "
            "duplicated id would silently take one z vector for two samples")
    return tuple(rows)


def extra_train_rows(data: str) -> list[IndexRow]:
    """The normal-only rows the non-sft2seg sources of ``--data`` contribute."""
    if data not in DATA_SOURCES:
        raise ValueError(f"--data must be one of {DATA_CHOICES}, got {data!r}")
    out: list[IndexRow] = []
    for name in DATA_SOURCES[data]:
        if name == "l8":
            out.extend(normal_only(load_l8_index()))
        else:                                            # pragma: no cover
            raise ValueError(f"unknown training source {name!r}")
    return out


def train_normal_rows(data: str = "v2seg", *, split: str = "train",
                      root: str | Path | None = None) -> list[IndexRow]:
    """The training population of ``--data``: measured, then asserted.

    Every source is counted here and checked against that source's own on-disk
    declaration -- the sft2seg part against the **口径's** own
    :attr:`DatasetVersion.normal_n` (resolved from ``root``), the L8 part against
    ``l8_train.manifest.report.json``.  The **total** is deliberately not written
    down anywhere: a literal for the merged n is a number that has to be edited
    every time a corpus is added, and the edit is exactly what gets forgotten.
    """
    resolved = _resolve_root(root)
    ver = version_for_root(resolved)
    rows = list(normal_only(load_index_cached(split, resolved)))
    want = ver.normal_n.get(split)
    if want is None:
        raise AssertionError(
            f"dataset version {ver.name!r} has no measured normal-only count for "
            f"split {split!r}; measure it and register it in DATASET_VERSIONS "
            "before training or scoring on it")
    if len(rows) != want:
        raise AssertionError(
            f"dataset version {ver.name!r} declares {want} normal rows for split "
            f"{split!r} ({ver.root}); this index gives {len(rows)}")
    seen = {r.sample_id for r in rows}
    for row in extra_train_rows(data):
        if row.sample_id in seen:
            raise AssertionError(
                f"{row.sample_id} appears in two training sources; the z caches "
                "are keyed by sample_id, so one of the two z vectors would be "
                "silently unreachable")
        seen.add(row.sample_id)
        rows.append(row)
    return rows


def train_normal_n(data: str = "v2seg", *, split: str = "train",
                   root: str | Path | None = None) -> int:
    """``len(train_normal_rows(data))`` -- what ``steps_per_epoch`` is derived from."""
    return len(train_normal_rows(data, split=split, root=root))


def train_source_facts(data: str = "v2seg", *, split: str = "train",
                       root: str | Path | None = None) -> dict[str, Any]:
    """Per-source counts for ``run_setup.json`` (no total is hard-coded anywhere)."""
    per: dict[str, Any] = {}
    for row in train_normal_rows(data, split=split, root=root):
        d = per.setdefault(row.source, {"n_normal": 0, "lut_id": set(),
                                        "source_image_id": set()})
        d["n_normal"] += 1
        d["lut_id"].add(row.lut_id)
        d["source_image_id"].add(row.source_image_id)
    out = {k: {"n_normal": v["n_normal"], "uniq_lut_id": len(v["lut_id"]),
               "uniq_source_image_id": len(v["source_image_id"])}
           for k, v in sorted(per.items())}
    return {"data": data, "sources": out,
            "n_normal_total": sum(v["n_normal"] for v in out.values())}


def bucket_pools(records: Iterable[Mapping[str, Any]], *, key: str = "minor",
                 ) -> dict[str, list[str]]:
    """``minor -> sorted unique lut_id`` over the given records.

    The ``B3_bucket_retrieval`` pool.  Built from **train** records only (the
    baseline draws a train LUT), and from the record's own label -- frozen block:
    ``splits_presets.csv`` is not used.  Measured on train: 77 buckets, pool size
    min 1 / median 17 / max 285, 3149 ids in total.
    """
    pools: dict[str, set[str]] = {}
    for rec in records:
        bucket = rec.get(key)
        lut_id = rec.get("lut_id")
        if bucket is None or not lut_id:
            continue
        pools.setdefault(str(bucket), set()).add(str(lut_id))
    return {k: sorted(v) for k, v in sorted(pools.items())}


def split_facts(rows: Sequence[IndexRow]) -> dict[str, Any]:
    """Counts that go into ``run_setup.json`` next to the frozen numbers."""
    normal = [r for r in rows if r.is_normal]
    return {
        "n": len(rows),
        "n_normal": len(normal),
        "n_low": len(rows) - len(normal),
        "n_style": sum(1 for r in rows if r.is_style),
        "n_local": sum(1 for r in rows if not r.is_style),
        "n_normal_style": sum(1 for r in normal if r.is_style),
        "n_normal_local": sum(1 for r in normal if not r.is_style),
        "uniq_lut_id": len({r.lut_id for r in rows}),
        "uniq_lut_id_normal": len({r.lut_id for r in normal}),
        "uniq_source": len({r.source_image_id for r in rows}),
    }
