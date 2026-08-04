"""Indexed-tar-shard random-access reader.

Consumption side of the storage contract in spec 3.3 / METACANVAS 2.3:

    index records at least ``shard / member / offset / length / size / checksum``,
    a schema version, a shard digest and a sample count; resume authority comes
    from the terminal manifest and the shard index, never from directory mtime.

The producing task (S0-DATA) runs in parallel with this one, so the exact JSONL
row layout is not yet frozen. Three layouts are accepted and auto-detected;
anything else raises with the observed keys listed, so a schema drift shows up
as a loud failure at load time rather than as silent garbage:

  A. nested   -- ``{"sample_id": ..., "members": {"image": {...}, "record": {...}}}``
  B. per-member flat -- ``{"sample_id": ..., "key": "image", "shard": ..., ...}``
                        (rows grouped by ``sample_id``)
  C. single-member flat -- ``{"sample_id": ..., "shard": ..., ...}`` with the
                        role inferred from the member's file extension.

Field aliases are configurable so a name mismatch is a config fix, not a patch.
"""

from __future__ import annotations

import hashlib
import json
import os
import zlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# alias -> canonical, applied to member dicts
MEMBER_FIELD_ALIASES: dict[str, str] = {
    "shard": "shard",
    "shard_name": "shard",
    "shard_path": "shard",
    "tar": "shard",
    "member": "member",
    "member_name": "member",
    "name": "member",
    "key": "member",
    "offset": "offset",
    "member_offset": "offset",
    "data_offset": "offset",
    "length": "length",
    "member_length": "length",
    "nbytes": "length",
    "size": "size",
    "member_size": "size",
    "checksum": "checksum",
    "sha256": "checksum",
    "crc32": "checksum",
    "digest": "checksum",
}

_ROLE_BY_SUFFIX = {
    ".jpg": "image", ".jpeg": "image", ".png": "image", ".webp": "image", ".bmp": "image",
    ".json": "record", ".jsonl": "record", ".txt": "record",
}


class ShardIndexError(RuntimeError):
    pass


class ShardIntegrityError(RuntimeError):
    pass


@dataclass(frozen=True)
class MemberRef:
    shard: str
    member: str
    offset: int
    length: int
    size: int | None = None
    checksum: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard": self.shard, "member": self.member, "offset": self.offset,
            "length": self.length, "size": self.size, "checksum": self.checksum,
        }


@dataclass
class SampleRef:
    sample_id: str
    members: dict[str, MemberRef]
    meta: dict[str, Any] = field(default_factory=dict)


def _normalise_member(raw: dict[str, Any]) -> MemberRef:
    canon: dict[str, Any] = {}
    for k, v in raw.items():
        target = MEMBER_FIELD_ALIASES.get(k)
        if target is None:
            continue
        # do not let a generic alias overwrite an exact-name hit
        if target in canon and k != target:
            continue
        canon[target] = v
    missing = [f for f in ("shard", "member", "offset", "length") if f not in canon]
    if missing:
        raise ShardIndexError(
            f"member entry missing {missing}; observed keys={sorted(raw)}. "
            f"Extend MEMBER_FIELD_ALIASES if the producer uses other names."
        )
    checksum = canon.get("checksum")
    if checksum is not None and not isinstance(checksum, str):
        checksum = str(checksum)
    return MemberRef(
        shard=str(canon["shard"]),
        member=str(canon["member"]),
        offset=int(canon["offset"]),
        length=int(canon["length"]),
        size=int(canon["size"]) if canon.get("size") is not None else None,
        checksum=checksum,
    )


def _role_from_member_name(name: str) -> str:
    return _ROLE_BY_SUFFIX.get(Path(name).suffix.lower(), "blob")


def _sample_id_of(row: dict[str, Any]) -> str:
    for k in ("sample_id", "sft_id", "id", "key", "stable_id"):
        v = row.get(k)
        if isinstance(v, str) and v:
            return v
    raise ShardIndexError(f"index row has no sample id; observed keys={sorted(row)}")


class ShardIndex:
    """Random-access index over indexed tar shards."""

    def __init__(self, samples: list[SampleRef], source: str = "", layout: str = ""):
        self.samples = samples
        self.source = source
        self.layout = layout
        self._by_id = {s.sample_id: i for i, s in enumerate(samples)}
        if len(self._by_id) != len(samples):
            raise ShardIndexError("duplicate sample_id in index")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> SampleRef:
        return self.samples[i]

    def by_id(self, sample_id: str) -> SampleRef:
        return self.samples[self._by_id[sample_id]]

    @property
    def sample_ids(self) -> list[str]:
        return [s.sample_id for s in self.samples]

    def shards(self) -> set[str]:
        return {m.shard for s in self.samples for m in s.members.values()}

    def filter_ids(self, keep: set[str]) -> "ShardIndex":
        kept = [s for s in self.samples if s.sample_id in keep]
        return ShardIndex(kept, source=self.source, layout=self.layout)

    # -- loading ------------------------------------------------------------
    @classmethod
    def load(cls, path: str | os.PathLike, split: str | None = None) -> "ShardIndex":
        path = Path(path)
        rows = list(_iter_json_rows(path))
        if not rows:
            raise ShardIndexError(f"index {path} is empty")
        if split is not None:
            rows = [r for r in rows if r.get("split") in (None, split)]
            if not rows:
                raise ShardIndexError(f"index {path} has no rows for split={split!r}")
        layout = cls._detect_layout(rows[0])
        builder = {
            "nested": cls._build_nested,
            "per_member": cls._build_per_member,
            "single_flat": cls._build_single_flat,
        }[layout]
        samples = builder(rows)
        return cls(samples, source=str(path), layout=layout)

    @staticmethod
    def _detect_layout(row: dict[str, Any]) -> str:
        if isinstance(row.get("members"), (dict, list)):
            return "nested"
        has_member_fields = all(
            any(a in row for a, t in MEMBER_FIELD_ALIASES.items() if t == canon)
            for canon in ("shard", "member", "offset", "length")
        )
        if not has_member_fields:
            raise ShardIndexError(
                "cannot detect index layout; a row must either carry a 'members' "
                f"mapping or flat shard/member/offset/length fields. observed keys={sorted(row)}"
            )
        # 'key'/'role' present alongside flat fields => one row per member
        if "role" in row or ("key" in row and "member" in row):
            return "per_member"
        return "single_flat"

    @staticmethod
    def _build_nested(rows: list[dict[str, Any]]) -> list[SampleRef]:
        out = []
        for row in rows:
            members_raw = row["members"]
            if isinstance(members_raw, list):
                members = {}
                for m in members_raw:
                    role = m.get("role") or m.get("key") or _role_from_member_name(
                        str(m.get("member") or m.get("name") or "")
                    )
                    members[str(role)] = _normalise_member(m)
            else:
                members = {str(k): _normalise_member(v) for k, v in members_raw.items()}
            meta = {k: v for k, v in row.items() if k != "members"}
            out.append(SampleRef(_sample_id_of(row), members, meta))
        return out

    @staticmethod
    def _build_per_member(rows: list[dict[str, Any]]) -> list[SampleRef]:
        grouped: dict[str, SampleRef] = {}
        order: list[str] = []
        for row in rows:
            sid = _sample_id_of(row)
            role = str(row.get("role") or row.get("key") or _role_from_member_name(str(row.get("member", ""))))
            if sid not in grouped:
                grouped[sid] = SampleRef(sid, {}, {k: v for k, v in row.items()})
                order.append(sid)
            grouped[sid].members[role] = _normalise_member(row)
        return [grouped[s] for s in order]

    @staticmethod
    def _build_single_flat(rows: list[dict[str, Any]]) -> list[SampleRef]:
        out = []
        for row in rows:
            ref = _normalise_member(row)
            role = str(row.get("role") or _role_from_member_name(ref.member))
            out.append(SampleRef(_sample_id_of(row), {role: ref}, dict(row)))
        return out


def _iter_json_rows(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix in (".jsonl", ".ndjson"):
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)
    elif path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for key in ("samples", "rows", "index", "entries"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
        if not isinstance(data, list):
            raise ShardIndexError(f"{path}: expected a list of index rows")
        yield from data
    else:
        raise ShardIndexError(f"unsupported index file type: {path}")


def _digest(data: bytes, alg: str) -> str:
    if alg == "crc32":
        return format(zlib.crc32(data) & 0xFFFFFFFF, "08x")
    return hashlib.new(alg, data).hexdigest()


def parse_checksum(value: str) -> tuple[str, str]:
    """``'sha256:ab..'`` / bare hex -> ``(algorithm, hexdigest)``."""
    v = value.strip().lower()
    if ":" in v:
        alg, _, hexd = v.partition(":")
        return alg, hexd
    return {64: "sha256", 40: "sha1", 32: "md5", 8: "crc32"}.get(len(v), "sha256"), v


class ShardStore:
    """Positional reads into unbundled tar shards, with checksum verification.

    ``verify`` -- ``"checksum"`` (default), ``"length"`` or ``"none"``.
    File descriptors are cached per (pid, shard) so dataloader workers that
    fork after construction reopen their own handles.
    """

    def __init__(
        self,
        shard_root: str | os.PathLike,
        verify: str = "checksum",
        max_open: int = 32,
    ):
        self.shard_root = Path(shard_root)
        if verify not in ("checksum", "length", "none"):
            raise ValueError(f"verify must be checksum|length|none, got {verify!r}")
        self.verify = verify
        self.max_open = max_open
        self._fds: dict[str, int] = {}
        self._pid = os.getpid()
        self.n_reads = 0
        self.n_checksum_verified = 0

    def _fd(self, shard: str) -> int:
        if os.getpid() != self._pid:  # forked dataloader worker
            self._fds.clear()
            self._pid = os.getpid()
        fd = self._fds.get(shard)
        if fd is None:
            path = Path(shard)
            if not path.is_absolute():
                path = self.shard_root / shard
            if not path.exists():
                raise ShardIntegrityError(f"shard not found: {path}")
            if len(self._fds) >= self.max_open:
                _, old = self._fds.popitem()
                os.close(old)
            fd = os.open(path, os.O_RDONLY)
            self._fds[shard] = fd
        return fd

    def read(self, ref: MemberRef, verify: str | None = None) -> bytes:
        mode = self.verify if verify is None else verify
        fd = self._fd(ref.shard)
        data = os.pread(fd, ref.length, ref.offset)
        self.n_reads += 1
        if len(data) != ref.length:
            raise ShardIntegrityError(
                f"short read for {ref.shard}:{ref.member} -- got {len(data)} of {ref.length} "
                f"bytes at offset {ref.offset}"
            )
        if ref.size is not None and ref.size != ref.length:
            raise ShardIntegrityError(
                f"{ref.shard}:{ref.member} size {ref.size} != length {ref.length}; the "
                f"contract stores members uncompressed, so they must agree"
            )
        if mode == "checksum":
            if not ref.checksum:
                raise ShardIntegrityError(
                    f"{ref.shard}:{ref.member} has no checksum but verify='checksum'"
                )
            alg, expected = parse_checksum(ref.checksum)
            got = _digest(data, alg)
            if got != expected:
                raise ShardIntegrityError(
                    f"{alg} mismatch for {ref.shard}:{ref.member} -- expected {expected}, got {got}"
                )
            self.n_checksum_verified += 1
        return data

    def close(self) -> None:
        for fd in self._fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        self._fds.clear()

    def __del__(self):  # best effort
        self.close()


# --- terminal manifest -----------------------------------------------------

_N_EFFECTIVE_ALIASES = (
    "n_effective", "N_effective", "num_effective", "n_samples", "num_samples",
    "count", "n_train", "sample_count",
)
_DIGEST_ALIASES = ("digest", "manifest_digest", "sha256", "content_digest")


@dataclass
class TerminalManifest:
    path: str
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: str | os.PathLike) -> "TerminalManifest":
        p = Path(path)
        return cls(path=str(p), raw=json.loads(p.read_text(encoding="utf-8")))

    def _lookup(self, aliases: tuple[str, ...], scope: str | None = None) -> Any:
        containers = [self.raw]
        for key in ("counts", "splits", "summary", "stats"):
            if isinstance(self.raw.get(key), dict):
                containers.append(self.raw[key])
                if scope and isinstance(self.raw[key].get(scope), dict):
                    containers.insert(0, self.raw[key][scope])
        if scope and isinstance(self.raw.get(scope), dict):
            containers.insert(0, self.raw[scope])
        for c in containers:
            for a in aliases:
                if c.get(a) is not None:
                    return c[a]
        return None

    def n_effective(self, split: str = "train") -> int:
        v = self._lookup(_N_EFFECTIVE_ALIASES, scope=split)
        if v is None:
            raise ShardIndexError(
                f"terminal manifest {self.path} has no N_effective for split={split!r}; "
                f"top-level keys={sorted(self.raw)}. Refusing to guess -- spec 8.1 forbids "
                f"hardcoding 2645/5290."
            )
        return int(v)

    def digest(self) -> str | None:
        v = self._lookup(_DIGEST_ALIASES)
        return str(v) if v is not None else None

    def schema_version(self) -> str | None:
        for k in ("schema_version", "version", "schema"):
            if self.raw.get(k) is not None:
                return str(self.raw[k])
        return None

    def summary(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "schema_version": self.schema_version(),
            "digest": self.digest(),
            "top_level_keys": sorted(self.raw),
        }
