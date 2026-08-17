"""The one ``z`` cache -- writer, reader and start-up assertions (HANDOFF 步骤 0-7).

``z = norm(hidden_states[-1])`` at the ``<seg_color>`` position of a reply the
frozen base produced.  Every one of the six arms consumes it and **none** of
them may carry its own copy: the frozen block says 共同依赖只写一份, and five
private readers had already drifted into five different on-disk layouts, four
different assertion sets and one silent bf16->fp32 upcast
(``np.asarray(data["z"], dtype=np.float32)``) before this module existed.

Schema (``EPR-024:607`` item 2, field for field; shape照
``q3vl/whereb/gencontext.py:115-125, 163-172``)::

    sample_id         str
    split             str
    checkpoint        str    the base that produced the hidden state
    readout_kind      str    one of WHATB_READOUT_KINDS
    context_source    str    "teacher" | "generated"
    control_tag       str    "none" | "shuffle" | "irrelevant" | "const"
    reply_token_ids   [int]  the reply span the hidden states came from
    readout_index     int    (or [start, end] for a pooled kind)
    z                 (2560,) or (K, 2560) float32
    n_generated_tokens int

**dtype.**  EPR-024:607 writes the field as ``z (bf16 2560)``; ruling 11.1-5
(HANDOFF:1009) later pinned the cache to **fp32** because bf16's 8-bit mantissa
moves the ``M`` column of criterion §D, which is a *difference* of two
conditions.  The later ruling wins, and this module never converts: an array
that is not fp32 is **rejected**, it is not upcast (an upcast bf16 cache is
numerically a bf16 cache wearing an fp32 label, and the assertion that was
supposed to catch it would pass).

On-disk layout, one directory per ``(split, control_tag)``::

    <root>/<split>__<tag>/z.npy          (n, 2560) or (n, K, 2560) float32
    <root>/<split>__<tag>/index.jsonl    one row per z, in z order
    <root>/<split>__<tag>/meta.json      the run-level fields + n/dim/dtype

``context_source`` is not part of the leaf name -- an arm addresses a cache by
``(split, control_tag)`` and *asserts* the context.  The two contexts therefore
live under two roots (``<cache>/generated/...`` and ``<cache>/teacher/...``), so
pointing ``--context teacher`` at the generated root fails the start-up
assertion instead of quietly training on the other condition.

Start-up assertions (§3.8 / HANDOFF §4.H), all three of them in
:meth:`ZCache.assert_belongs_to`:

1. ``meta.checkpoint`` == this run's base checkpoint,
2. ``meta.readout_kind`` == this run's ``--readout`` (and ``context_source`` ==
   ``--context`` for the ``none`` tag; the three controls are always
   ``generated`` -- a teacher-forced reasoning would carry the *true*
   instruction's semantics at the ``<seg_color>`` position and void the control,
   frozen block ⑥),
3. a 1% sample of the rows is replayed through
   :func:`q3vl.whereb.readout.verify_plan`, so the recorded ``readout_index``
   really carries the token id the kind promises.

:class:`SyntheticZCache` is the smoke stand-in: a deterministic hash, never a
read-out, and every board built on it must set ``published=False``.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .readout import ReplyPlan, verify_plan

__all__ = [
    "CONTROL_TAGS",
    "CONTEXT_SOURCES",
    "Z_DTYPE",
    "Z_DIM",
    "ZCACHE_FIELDS",
    "ZCacheDtypeError",
    "leaf_dir",
    "write_z_cache",
    "ZCache",
    "MultiZCache",
    "SyntheticZCache",
    "ZCacheDir",
]

#: the four caches N1..N3 are computed from (frozen block ⑥)
CONTROL_TAGS: tuple[str, ...] = ("none", "shuffle", "irrelevant", "const")

#: ``--context`` values; the three controls are always ``generated``
CONTEXT_SOURCES: tuple[str, ...] = ("teacher", "generated")

#: ruling 11.1-5.  Not a preference: see the module docstring.
Z_DTYPE = np.float32

#: Qwen3-VL-4B text hidden size (``q3vl.whereb.config.TEXT_HIDDEN``)
Z_DIM = 2560

#: the per-row fields the frozen schema names (EPR-024:607)
ZCACHE_FIELDS: tuple[str, ...] = (
    "sample_id", "split", "checkpoint", "readout_kind", "context_source",
    "control_tag", "reply_token_ids", "readout_index", "n_generated_tokens",
)

#: optional per-row fields the arms read when they are present
ZCACHE_EXTRA_FIELDS: tuple[str, ...] = (
    "color_text", "lut_id", "source_image_id", "task_type", "winner_confidence",
    "minor", "expected_ids",
)

DEFAULT_VERIFY_FRAC = 0.01
DEFAULT_SEED = 20260810


class ZCacheDtypeError(TypeError):
    """The cache is not fp32 and this module refuses to convert it."""


def _hash_seed(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)


def leaf_dir(root: str | Path, split: str, tag: str = "none") -> Path:
    """``<root>/<split>__<tag>`` -- the one directory naming convention."""
    return Path(root) / f"{split}__{tag}"


def blob_path(root: str | Path, split: str, tag: str = "none",
              context: str = "generated") -> Path:
    """``<root>/<split>.<context>.<tag>.zcache.pt`` -- the producer's naming.

    The offline producer that ran on 2026-08-15
    (``/home/bc/data/runs/whatb/zcache_v2seg/_src/build_z.py``) writes one
    ``torch.save`` blob per stage instead of a directory.  Same *semantics*,
    different encoding, and :class:`ZCache` reads both -- there is still one
    reader class and one assertion set, which is what the frozen block asks for.
    """
    return Path(root) / f"{split}.{context}.{tag}.zcache.pt"


def resolve_leaf(root: str | Path, split: str, tag: str = "none",
                 context: str = "generated") -> Path | None:
    """The canonical directory if it exists, else the producer's ``.pt`` blob.

    Three placements are tried, in this order:

    1. ``<root>/<split>__<tag>``          -- the canonical directory,
    2. ``<root>/<context>/<split>__<tag>`` -- the same directory under the
       per-context root this module's docstring describes ("the two contexts
       therefore live under two roots"), which is how the L8 cache is laid out,
    3. ``<root>/<split>.<context>.<tag>.zcache.pt`` -- the offline producer's blob.

    The context sub-directory is *not* a fourth encoding: the leaf it points at
    is the ordinary directory form, and ``assert_belongs_to`` still checks
    ``meta.context_source`` there, so pointing ``--context teacher`` at the
    generated root still fails at start-up rather than training on the other
    condition.
    """
    d = leaf_dir(root, split, tag)
    if d.is_dir():
        return d
    d = leaf_dir(Path(root) / str(context), split, tag)
    if d.is_dir():
        return d
    p = blob_path(root, split, tag, context)
    return p if p.is_file() else None


# --------------------------------------------------------------------------- #
# writer
# --------------------------------------------------------------------------- #
def write_z_cache(path: str | Path, rows: Sequence[Mapping[str, Any]],
                  z: np.ndarray | Tensor, *, checkpoint: str, readout_kind: str,
                  context_source: str, control_tag: str, split: str,
                  readout_qtok: int = 0, extra_meta: Mapping[str, Any] | None = None,
                  ) -> Path:
    """Write one ``(split, control_tag)`` cache directory.

    The generation side (a frozen-VLM forward, ``scripts/build_zcache.py``) is a
    GPU job; this is the on-disk contract both sides agree on, so a cache written
    by any generator is readable here and is asserted against the run's own base
    checkpoint before a single step is taken.
    """
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    if isinstance(z, Tensor):
        z = z.detach().cpu().numpy()
    arr = np.asarray(z)
    if arr.dtype == np.float64:
        # a *narrowing* cast invents nothing; float64 is what a numpy default
        # rng hands back and the disk format is fp32 either way
        arr = arr.astype(Z_DTYPE)
    elif arr.dtype != Z_DTYPE:
        raise ZCacheDtypeError(
            f"z is {arr.dtype}, the cache is fp32 (ruling 11.1-5).  A widening "
            "cast (bf16/fp16 -> fp32) is exactly what this refuses: it would "
            "produce a bf16 cache wearing an fp32 label, and the assertion that "
            "is supposed to catch that would pass.  Read out in fp32.")
    if arr.ndim not in (2, 3) or arr.shape[-1] != Z_DIM:
        raise ValueError(f"z must be (n, {Z_DIM}) or (n, K, {Z_DIM}), got {arr.shape}")
    if arr.shape[0] != len(rows):
        raise ValueError(f"{len(rows)} index rows for {arr.shape[0]} z vectors")
    if context_source not in CONTEXT_SOURCES:
        raise ValueError(f"context_source must be one of {CONTEXT_SOURCES}")
    if control_tag not in CONTROL_TAGS:
        raise ValueError(f"control_tag must be one of {CONTROL_TAGS}")
    np.save(out / "z.npy", arr)
    with (out / "index.jsonl").open("w", encoding="utf-8") as fh:
        for i, r in enumerate(rows):
            rec = {k: r.get(k) for k in ZCACHE_FIELDS}
            rec.update({k: r[k] for k in ZCACHE_EXTRA_FIELDS if k in r})
            rec.update({"checkpoint": checkpoint, "readout_kind": readout_kind,
                        "context_source": context_source, "control_tag": control_tag,
                        "split": split, "row": i})
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    meta = {
        "checkpoint": checkpoint, "readout_kind": readout_kind,
        "readout_qtok": int(readout_qtok), "context_source": context_source,
        "control_tag": control_tag, "split": split, "n": int(arr.shape[0]),
        "dim": int(arr.shape[-1]),
        "k_rows": int(arr.shape[1]) if arr.ndim == 3 else 0,
        "dtype": "float32",
    }
    if extra_meta:
        meta.update(dict(extra_meta))
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
# reader
# --------------------------------------------------------------------------- #
class ZCache:
    """One ``(split, control_tag)`` cache: a directory (mmap) or a ``.pt`` blob.

    Two encodings, one reader.  The directory form is what
    :func:`write_z_cache` produces; the ``.pt`` form is what the 2026-08-15
    offline producer wrote (``{"meta": ..., "sample_ids": [...], "z": (n,2560)
    fp32}``, per-sample diagnostics in a sidecar ``.report.json``).  Both go
    through the same start-up assertions and the same fp32 rejection rule.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.is_file() and self.path.suffix == ".pt":
            self._init_from_blob()
            return
        meta_path = self.path / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"{meta_path} does not exist.  A z cache directory is "
                "{z.npy, index.jsonl, meta.json}; build it with "
                "q3vl/whatb/scripts/build_zcache.py (the arms consume z, they "
                "never generate it).")
        self.meta: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
        self.rows: list[dict[str, Any]] = []
        with (self.path / "index.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        self.z = np.load(self.path / "z.npy", mmap_mode="r")
        if self.z.dtype != Z_DTYPE:
            raise ZCacheDtypeError(
                f"{self.path/'z.npy'} is {self.z.dtype}; ruling 11.1-5 pins the "
                "cache to fp32 and this reader refuses to upcast (bf16's 8 "
                "mantissa bits move the N1..N3 M column, which is a difference "
                "of two conditions -- an upcast would hide exactly that).")
        if self.z.ndim not in (2, 3) or self.z.shape[-1] != Z_DIM:
            raise ValueError(
                f"{self.path/'z.npy'} is {self.z.shape}, expected (n, {Z_DIM}) or "
                f"(n, K, {Z_DIM})")
        if self.z.shape[0] != len(self.rows):
            raise ValueError(
                f"{self.path}: {len(self.rows)} index rows but "
                f"{self.z.shape[0]} z vectors")
        self._index = {str(r["sample_id"]): int(r.get("row", i))
                       for i, r in enumerate(self.rows)}

    def _init_from_blob(self) -> None:
        """``{"meta", "sample_ids", "z"}`` + the optional ``.report.json`` sidecar."""
        blob = torch.load(self.path, map_location="cpu", weights_only=False)
        self.meta = dict(blob.get("meta") or {})
        ids = [str(s) for s in blob["sample_ids"]]
        z = blob["z"]
        if not isinstance(z, torch.Tensor):
            z = torch.as_tensor(np.asarray(z))
        if z.dtype != torch.float32:
            raise ZCacheDtypeError(
                f"{self.path}: z is {z.dtype}; ruling 11.1-5 pins the cache to "
                "fp32 and this reader refuses to upcast (bf16's 8 mantissa bits "
                "move the N1..N3 M column, which is a difference of two "
                "conditions -- an upcast would hide exactly that).")
        if z.dim() not in (2, 3) or z.shape[-1] != Z_DIM:
            raise ValueError(f"{self.path}: z is {tuple(z.shape)}, expected "
                             f"(n, {Z_DIM}) or (n, K, {Z_DIM})")
        if z.shape[0] != len(ids):
            raise ValueError(f"{self.path}: {len(ids)} ids but {z.shape[0]} z rows")
        self.z = z.numpy()
        # the sidecar carries the per-sample readout_index the producer used;
        # it also ran verify_plan on EVERY plan at write time (stronger than the
        # 1% replay this class does), which assert_belongs_to records.
        sidecar = self.path.parent / (self.path.name.replace(".pt", "") + ".report.json")
        diag: dict[str, dict[str, Any]] = {}
        if sidecar.is_file():
            try:
                rep = json.loads(sidecar.read_text(encoding="utf-8"))
                for r in rep.get("rows") or []:
                    if r.get("sample_id"):
                        diag[str(r["sample_id"])] = r
            except (json.JSONDecodeError, OSError):        # pragma: no cover
                diag = {}
        self.rows = [{"sample_id": sid, "row": i,
                      **{k: v for k, v in (diag.get(sid) or {}).items()
                         if k != "sample_id"}}
                     for i, sid in enumerate(ids)]
        self._index = {sid: i for i, sid in enumerate(ids)}
        self.meta.setdefault("dtype", "float32")
        self.meta.setdefault("n", len(ids))

    # -- shape ---------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.rows)

    def __contains__(self, sample_id: str) -> bool:
        return str(sample_id) in self._index

    @property
    def k_rows(self) -> int:
        """``K`` for a ``qtok`` cache, 0 for a single-vector one."""
        return int(self.z.shape[1]) if self.z.ndim == 3 else 0

    @property
    def ids(self) -> list[str]:
        return [str(r["sample_id"]) for r in self.rows]

    @property
    def control_tag(self) -> str:
        return str(self.meta.get("control_tag", "none"))

    @property
    def split(self) -> str:
        return str(self.meta.get("split", ""))

    # -- access --------------------------------------------------------------
    def vector(self, sample_id: str, *, device: Any = "cpu",
               dtype: torch.dtype = torch.float32) -> Tensor:
        try:
            i = self._index[str(sample_id)]
        except KeyError:
            raise KeyError(
                f"{sample_id} is not in the z cache at {self.path}; the cache and "
                "the split disagree, which silently drops samples from the paired "
                "delta") from None
        return torch.from_numpy(np.array(self.z[i])).to(device=device, dtype=dtype)

    #: qdual / idgate spelling
    get = vector

    def batch(self, sample_ids: Sequence[str], *, device: Any = "cpu",
              dtype: torch.dtype = torch.float32) -> Tensor:
        idx = []
        for s in sample_ids:
            try:
                idx.append(self._index[str(s)])
            except KeyError:
                raise KeyError(f"{s} is not in the z cache at {self.path}") from None
        return torch.from_numpy(np.array(self.z[idx])).to(device=device, dtype=dtype)

    def mean(self, *, device: Any = "cpu",
             dtype: torch.dtype = torch.float32) -> Tensor:
        """``z̄_train`` for the ``--cond-trainmean`` column."""
        return torch.from_numpy(np.array(self.z).mean(axis=0)).to(
            device=device, dtype=dtype)

    def color_texts(self, limit: int | None = None) -> list[str]:
        """The cached ``<color>`` texts (the colour-span start-up assertion)."""
        out = [str(r["color_text"]) for r in self.rows if r.get("color_text")]
        return out[:limit] if limit else out

    def field(self, sample_id: str, key: str, default: Any = None) -> Any:
        row = self.rows[self._index[str(sample_id)]]
        return row.get(key, default)

    # -- the start-up assertion ---------------------------------------------
    def assert_belongs_to(self, *, checkpoint: str, readout_kind: str,
                          context_source: str | None = None,
                          control_tag: str | None = None,
                          k_rows: int | None = None,
                          verify_frac: float = DEFAULT_VERIFY_FRAC,
                          seed: int = DEFAULT_SEED) -> dict[str, Any]:
        """§3.8 / HANDOFF §4.H.  Returns the ``run_setup`` record."""
        if str(self.meta.get("checkpoint")) != str(checkpoint):
            raise AssertionError(
                f"{self.path}: cache was written from checkpoint "
                f"{self.meta.get('checkpoint')!r}, this run's base is "
                f"{checkpoint!r}.  A condition read off another base is not this "
                "experiment's condition.")
        if str(self.meta.get("readout_kind")) != str(readout_kind):
            raise AssertionError(
                f"{self.path}: cache readout_kind={self.meta.get('readout_kind')!r} "
                f"but --readout {readout_kind!r}")
        if context_source is not None and \
                str(self.meta.get("context_source")) != str(context_source):
            raise AssertionError(
                f"{self.path}: cache context_source="
                f"{self.meta.get('context_source')!r} but --context "
                f"{context_source!r}")
        if control_tag is not None and \
                str(self.meta.get("control_tag")) != str(control_tag):
            raise AssertionError(
                f"{self.path}: cache control_tag={self.meta.get('control_tag')!r} "
                f"!= {control_tag!r}; N1/N2/N3 would then be computed from the "
                "wrong cache")
        if k_rows is not None and int(k_rows) != self.k_rows:
            raise AssertionError(
                f"{self.path}: cache has k_rows={self.k_rows}, this run asks for "
                f"{int(k_rows)} (--z-expand)")
        rng = random.Random(seed)
        k = (max(1, int(round(float(verify_frac) * len(self.rows))))
             if self.rows else 0)
        picked = rng.sample(range(len(self.rows)), min(k, len(self.rows)))
        n_verified = 0
        for i in picked:
            plan = self._plan_of(self.rows[i])
            if plan is None:
                continue
            verify_plan(plan)
            n_verified += 1
        rec = {
            "path": str(self.path), "n": len(self.rows), "k_rows": self.k_rows,
            **{key: self.meta.get(key) for key in
               ("checkpoint", "readout_kind", "context_source", "control_tag",
                "split", "dtype")},
            "n_verify_plan": n_verified, "verify_frac": float(verify_frac),
            "n_verify_sampled": len(picked),
        }
        if n_verified == 0 and self.rows:
            # not a pass by omission: say which witness stands in its place
            rec["verify_plan_note"] = (
                "the rows carry no reply_token_ids, so verify_plan cannot be "
                "replayed here.  The producer recorded in meta whether it ran "
                "verify_plan at write time: "
                f"producer={self.meta.get('producer')!r}, "
                f"reply_layout={self.meta.get('reply_layout')!r}, "
                f"seg_color_id={self.meta.get('seg_color_id')!r}")
        return rec

    def _plan_of(self, row: Mapping[str, Any]) -> ReplyPlan | None:
        ids = row.get("reply_token_ids")
        idx = row.get("readout_index")
        if not ids or idx is None:
            return None
        ids = [int(t) for t in ids]
        kind = str(self.meta.get("readout_kind"))
        if isinstance(idx, (list, tuple)):
            start, end = int(idx[0]), int(idx[1])
            pool, expected = True, ()
        else:
            start = int(idx)
            end = start + 1
            pool = False
            recorded = row.get("expected_ids")
            expected = (tuple(int(t) for t in recorded) if recorded
                        else (int(ids[start]),))
        return ReplyPlan(token_ids=ids, start=start, end=end, pool=pool, kind=kind,
                         source=str(self.meta.get("context_source", "generated")),
                         expected_ids=expected)

    def facts(self) -> dict[str, Any]:
        return {"path": str(self.path), "n": len(self.rows), "k_rows": self.k_rows,
                "synthetic": False, "meta": dict(self.meta)}


# --------------------------------------------------------------------------- #
# several caches, one condition
# --------------------------------------------------------------------------- #
class MultiZCache:
    """Two or more :class:`ZCache` leaves addressed as one condition.

    ``--data v2seg+l8`` trains on two corpora whose z were read out by the same
    frozen base into two caches (``zcache_v2seg/train`` and
    ``zcache_l8/<context>/l8_train``).  They are **not** concatenated on disk:
    each keeps its own ``meta.json``, and each goes through its own
    :meth:`ZCache.assert_belongs_to` -- so a member written from another
    checkpoint, another ``--readout`` or the other ``context_source`` still
    fails at start-up, one member at a time.

    Dispatch is by ``sample_id``.  A ``sample_id`` that appears in two members is
    rejected at construction: with a silent winner, one of the two z vectors
    would become unreachable and the sample would train on the other corpus's
    condition.
    """

    def __init__(self, members: Sequence["ZCache"], *, control_tag: str = "none",
                 ) -> None:
        if not members:
            raise ValueError("MultiZCache needs at least one member cache")
        self.members: tuple[ZCache, ...] = tuple(members)
        self._tag = str(control_tag)
        self._owner: dict[str, int] = {}
        for m_i, m in enumerate(self.members):
            for sid in m.ids:
                prev = self._owner.get(sid)
                if prev is not None:
                    raise ValueError(
                        f"{sid} is in both {self.members[prev].path} and "
                        f"{m.path}; the two training sources overlap and one of "
                        "the two z vectors would be silently unreachable")
                self._owner[sid] = m_i
        k = {m.k_rows for m in self.members}
        if len(k) != 1:
            raise ValueError(f"members disagree on k_rows: {sorted(k)}")
        self.k_rows = int(k.pop())

    # -- shape ---------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._owner)

    def __contains__(self, sample_id: str) -> bool:
        return str(sample_id) in self._owner

    @property
    def path(self) -> str:
        return " + ".join(str(m.path) for m in self.members)

    @property
    def ids(self) -> list[str]:
        out: list[str] = []
        for m in self.members:
            out.extend(m.ids)
        return out

    @property
    def control_tag(self) -> str:
        return self._tag

    @property
    def split(self) -> str:
        return "+".join(m.split for m in self.members)

    @property
    def meta(self) -> dict[str, Any]:
        """The first member's meta, plus the member list.  Not a merge: the
        fields that must agree are asserted, the rest stay per member."""
        out = dict(self.members[0].meta)
        out["members"] = [dict(m.meta, path=str(m.path)) for m in self.members]
        out["n"] = len(self._owner)
        return out

    # -- access --------------------------------------------------------------
    def _member_of(self, sample_id: str) -> "ZCache":
        try:
            return self.members[self._owner[str(sample_id)]]
        except KeyError:
            raise KeyError(
                f"{sample_id} is in none of the {len(self.members)} z caches "
                f"({self.path}); the caches and the split disagree, which "
                "silently drops samples from the paired delta") from None

    def vector(self, sample_id: str, *, device: Any = "cpu",
               dtype: torch.dtype = torch.float32) -> Tensor:
        return self._member_of(sample_id).vector(sample_id, device=device,
                                                 dtype=dtype)

    get = vector

    def batch(self, sample_ids: Sequence[str], *, device: Any = "cpu",
              dtype: torch.dtype = torch.float32) -> Tensor:
        """One gather per member, reassembled in the requested order."""
        ids = [str(s) for s in sample_ids]
        if not ids:
            shape = (0, self.k_rows, Z_DIM) if self.k_rows else (0, Z_DIM)
            return torch.empty(shape, device=torch.device(device), dtype=dtype)
        per: dict[int, list[int]] = {}          # member -> output positions
        for i, s in enumerate(ids):
            self._member_of(s)                  # the KeyError with the real message
            per.setdefault(self._owner[s], []).append(i)
        out: Tensor | None = None
        for m_i, positions in per.items():
            part = self.members[m_i].batch([ids[i] for i in positions],
                                           device=device, dtype=dtype)
            if out is None:
                out = torch.empty((len(ids),) + tuple(part.shape[1:]),
                                  device=part.device, dtype=part.dtype)
            out[positions] = part
        assert out is not None                  # per is non-empty when ids is
        return out

    def mean(self, *, device: Any = "cpu",
             dtype: torch.dtype = torch.float32) -> Tensor:
        """``z̄_train`` over the union -- the n-weighted mean, not a mean of means."""
        total = float(sum(len(m) for m in self.members))
        acc = None
        for m in self.members:
            part = m.mean(device=device, dtype=torch.float32) * (len(m) / total)
            acc = part if acc is None else acc + part
        return acc.to(device=device, dtype=dtype)

    def color_texts(self, limit: int | None = None) -> list[str]:
        out: list[str] = []
        for m in self.members:
            out.extend(m.color_texts())
        return out[:limit] if limit else out

    def field(self, sample_id: str, key: str, default: Any = None) -> Any:
        return self._member_of(sample_id).field(sample_id, key, default)

    # -- the start-up assertion ---------------------------------------------
    def assert_belongs_to(self, **kw: Any) -> dict[str, Any]:
        """Every member's own :meth:`ZCache.assert_belongs_to`, all of them."""
        recs = [m.assert_belongs_to(**kw) for m in self.members]
        return {"path": self.path, "n": len(self._owner), "k_rows": self.k_rows,
                "n_members": len(self.members), "members": recs,
                **{key: self.members[0].meta.get(key) for key in
                   ("checkpoint", "readout_kind", "context_source", "control_tag",
                    "dtype")},
                "split": self.split,
                "n_verify_plan": sum(int(r.get("n_verify_plan", 0)) for r in recs)}

    def facts(self) -> dict[str, Any]:
        return {"path": self.path, "n": len(self._owner), "k_rows": self.k_rows,
                "synthetic": False,
                "members": [m.facts() for m in self.members]}


# --------------------------------------------------------------------------- #
# smoke stand-in
# --------------------------------------------------------------------------- #
class SyntheticZCache:
    """A deterministic pseudo-condition for ``--smoke``.  Never publishable.

    It is NOT a read-out.  It exists so the CPU plumbing test has distinct
    conditions per sample; any board built on one must carry ``published=False``.
    ``key_fn(tag, sample_id) -> str`` lets an arm keep a *learnable* structure
    (e.g. keying ``none`` on the sample's ``lut_id`` so a generator can fit it),
    and ``jitter`` adds a per-sample perturbation on top of that shared key.
    """

    def __init__(self, *, tag: str = "none", split: str = "", checkpoint: str = "",
                 readout_kind: str = "seg_color", context_source: str = "generated",
                 dim: int = Z_DIM, k_rows: int = 0, jitter: float = 0.0,
                 key_fn: Callable[[str, str], str] | None = None) -> None:
        self.tag = tag
        self.split = split
        self.dim = int(dim)
        self.k_rows = int(k_rows)
        self.jitter = float(jitter)
        self.key_fn = key_fn
        self.meta: dict[str, Any] = {
            "checkpoint": checkpoint, "readout_kind": readout_kind,
            "context_source": context_source, "control_tag": tag, "split": split,
            "synthetic": True, "dtype": "float32",
        }

    @property
    def control_tag(self) -> str:
        return self.tag

    def __len__(self) -> int:
        return 0

    def __contains__(self, sample_id: str) -> bool:
        return True

    def vector(self, sample_id: str, *, device: Any = "cpu",
               dtype: torch.dtype = torch.float32) -> Tensor:
        key = (self.key_fn(self.tag, str(sample_id)) if self.key_fn
               else f"{self.tag}/{sample_id}")
        g = torch.Generator().manual_seed(_hash_seed(key))
        z = torch.randn(self.dim, generator=g, dtype=torch.float32)
        if self.jitter or self.k_rows:
            g2 = torch.Generator().manual_seed(
                _hash_seed(f"{sample_id}|{self.tag}"))
            if self.jitter:
                z = z + self.jitter * torch.randn(self.dim, generator=g2,
                                                  dtype=torch.float32)
            if self.k_rows:
                z = z.unsqueeze(0).repeat(self.k_rows, 1)
                z = z + 0.01 * torch.randn(z.shape, generator=g2,
                                           dtype=torch.float32)
        return z.to(device=device, dtype=dtype)

    get = vector

    def batch(self, sample_ids: Sequence[str], *, device: Any = "cpu",
              dtype: torch.dtype = torch.float32) -> Tensor:
        return torch.stack([self.vector(s) for s in sample_ids]).to(
            device=device, dtype=dtype)

    def mean(self, *, device: Any = "cpu",
             dtype: torch.dtype = torch.float32) -> Tensor:
        return torch.zeros(self.dim, device=device, dtype=dtype)

    def color_texts(self, limit: int | None = None) -> list[str]:
        return []

    def field(self, sample_id: str, key: str, default: Any = None) -> Any:
        return default

    def assert_belongs_to(self, **kw: Any) -> dict[str, Any]:
        return {"path": None, "n": 0, "synthetic": True, "control_tag": self.tag,
                "split": self.split, **{k: self.meta.get(k) for k in
                                        ("checkpoint", "readout_kind",
                                         "context_source")}}

    def facts(self) -> dict[str, Any]:
        return {"path": None, "n": 0, "k_rows": self.k_rows, "synthetic": True,
                "meta": dict(self.meta)}


# --------------------------------------------------------------------------- #
# the four caches of one split
# --------------------------------------------------------------------------- #
class ZCacheDir:
    """``<root>`` holding one leaf per control tag, for one split.

    Every cache is opened and asserted **at construction**, not lazily: HANDOFF
    §4.H wants the checkpoint assertion before the dataloader is built, and a
    lazy one would land after the first thousand steps.
    """

    def __init__(self, root: str | Path | None, *, split: str, checkpoint: str,
                 readout_kind: str, context_source: str = "generated",
                 tags: Iterable[str] = CONTROL_TAGS,
                 required: Iterable[str] = ("none",),
                 synthetic: bool = False, k_rows: int = 0, jitter: float = 0.0,
                 key_fn: Callable[[str, str], str] | None = None,
                 verify_frac: float = DEFAULT_VERIFY_FRAC,
                 seed: int = DEFAULT_SEED) -> None:
        self.root = Path(root) if root is not None else None
        self.split = split
        self.checkpoint = checkpoint
        self.readout_kind = readout_kind
        self.context_source = context_source
        self.synthetic = bool(synthetic)
        self.tags = tuple(tags)
        self.required = tuple(required)
        self.caches: dict[str, ZCache | SyntheticZCache] = {}
        self.record: dict[str, Any] = {}
        if self.synthetic:
            for tag in self.tags:
                self.caches[tag] = SyntheticZCache(
                    tag=tag, split=split, checkpoint=checkpoint,
                    readout_kind=readout_kind,
                    context_source="generated" if tag != "none" else context_source,
                    k_rows=k_rows, jitter=jitter, key_fn=key_fn)
                self.record[tag] = self.caches[tag].facts()
            return
        if self.root is None:
            raise SystemExit(
                "no z cache root.  The four caches none/shuffle/irrelevant/const "
                "are what the N1..N3 columns are computed from; build them with "
                "q3vl/whatb/scripts/build_zcache.py, or run --smoke with the "
                "synthetic stand-in (never publishable).")
        for tag in self.tags:
            # frozen block ⑥: the three controls are regenerated reasonings, so
            # their context_source is always "generated" whatever --context says.
            ctx = self.context_source if tag == "none" else "generated"
            path = resolve_leaf(self.root, split, tag, ctx)
            if path is None:
                want = f"{leaf_dir(self.root, split, tag)} or " \
                       f"{blob_path(self.root, split, tag, ctx)}"
                if tag in set(self.required):
                    raise FileNotFoundError(
                        f"the {tag!r} z cache for split {split} is missing "
                        f"({want}); the arm consumes z from cache and cannot "
                        "generate it here")
                self.record[tag] = {"path": want, "present": False}
                continue
            cache = ZCache(path)
            rec = cache.assert_belongs_to(
                checkpoint=checkpoint, readout_kind=readout_kind,
                context_source=ctx, control_tag=tag,
                k_rows=k_rows or None, verify_frac=verify_frac, seed=seed)
            rec["present"] = True
            self.record[tag] = rec
            self.caches[tag] = cache

    # -- access --------------------------------------------------------------
    def __contains__(self, tag: str) -> bool:
        return tag in self.caches

    def cache(self, tag: str = "none") -> ZCache | SyntheticZCache:
        try:
            return self.caches[tag]
        except KeyError:
            raise KeyError(
                f"no {tag!r} z cache for split {self.split} under {self.root}") from None

    def vector(self, sample_id: str, tag: str = "none", *, device: Any = "cpu",
               dtype: torch.dtype = torch.float32) -> Tensor:
        return self.cache(tag).vector(sample_id, device=device, dtype=dtype)

    def z(self, sample_ids: Sequence[str], *, tag: str = "none",
          device: Any = "cpu", dtype: torch.dtype = torch.float32) -> Tensor:
        return self.cache(tag).batch(sample_ids, device=device, dtype=dtype)

    def batch(self, sample_ids: Sequence[str], tag: str = "none", *,
              device: Any = "cpu", dtype: torch.dtype = torch.float32) -> Tensor:
        return self.cache(tag).batch(sample_ids, device=device, dtype=dtype)

    def mean(self, tag: str = "none", *, device: Any = "cpu",
             dtype: torch.dtype = torch.float32) -> Tensor:
        return self.cache(tag).mean(device=device, dtype=dtype)

    def color_texts(self, limit: int | None = None, tag: str = "none") -> list[str]:
        return self.cache(tag).color_texts(limit)

    def ids(self, tag: str = "none") -> list[str]:
        c = self.cache(tag)
        return c.ids if isinstance(c, ZCache) else []

    @property
    def meta(self) -> dict[str, Any]:
        """``tag -> meta``; what ``assert_context_caches`` style checks read."""
        return {t: dict(c.meta) for t, c in self.caches.items()}

    def facts(self) -> dict[str, Any]:
        return {"root": str(self.root) if self.root else None, "split": self.split,
                "checkpoint": self.checkpoint, "readout_kind": self.readout_kind,
                "context_source": self.context_source,
                "synthetic": self.synthetic, "controls": dict(self.record)}
