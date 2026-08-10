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
import threading
import zlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --- read-path mount rewrite (W-B1, 2026-08-10) ---------------------------
# The frozen split indexes bake ABSOLUTE shard paths under the hard mount:
#     "shard": "/mnt/nfs/bc/data/datasets/sft2seg-20260804/images/shards/..."
# so moving the *root constants* to /mnt/nfs-ro (D-B17) never touched the bytes
# the training loop actually streams -- 100% of it stayed on /mnt/nfs (nfs4,
# hard), where one stalled read parks the process in unkillable D state and
# takes the whole box with it (CLAUDE.md 2026-08-10 incident).
#
# ShardStore is read-only by construction (O_RDONLY + pread) and both mounts
# export the same tree (172.25.76.194:/rwq), so rewriting the prefix at open
# time is behaviour-preserving.  It is on by default rather than opt-in
# precisely because the D-B17 miss was an opt-in that nobody remembered to
# apply; it goes inert when /mnt/nfs-ro is not mounted, so a box without the
# read mirror keeps working exactly as before.
READ_PREFIX_REWRITE: tuple[tuple[str, str], ...] = (("/mnt/nfs/", "/mnt/nfs-ro/"),)
_ro_mounted: bool | None = None


def _read_mirror_mounted() -> bool:
    """True if the rewrite targets are mounted. Local read of /proc/mounts only."""
    global _ro_mounted
    if _ro_mounted is None:
        try:
            with open("/proc/mounts", "r", encoding="utf-8") as fh:
                fields = [ln.split() for ln in fh]
            points = {f[1] for f in fields if len(f) > 1}
        except OSError:                                   # pragma: no cover
            points = set()
        _ro_mounted = all(dst.rstrip("/") in points for _, dst in READ_PREFIX_REWRITE)
    return _ro_mounted


def rewrite_read_path(path: str | os.PathLike) -> str:
    """Map an absolute read path onto the soft read mirror of the same export."""
    s = str(path)
    if not _read_mirror_mounted():
        return s
    for src, dst in READ_PREFIX_REWRITE:
        if s.startswith(src):
            return dst + s[len(src):]
    return s


# --- local shard cache (PERF-1, 2026-08-10) -------------------------------
# Measured on this box while W01/W02 were streaming (see PERF-1):
#
#   nfs-ro random 152 KiB pread, 1 thread : p50 17.0 ms, 53 reads/s,   8 MB/s
#   local ext4, page cache warm           : p50  0.02 ms, 45k reads/s
#   local ext4, page cache dropped        : p50  6.5 ms, 145 reads/s
#   nfs-ro sequential copy                : 106 MB/s  (27.5 GiB -> ~4.5 min)
#
# Where-B walks a shuffled 1-epoch permutation, so *every* member read is a cold
# random read: 78 ms of the 627 ms it took to assemble one micro-batch of 8 was
# nothing but NFS round-trip latency (<5% of it was transferring bytes).  The
# whole consumed corpus is 27.5 GiB against 1.1 TiB of free local disk and
# 113 GiB of free RAM, so a one-off sequential copy converts every one of those
# round trips into a page-cache hit.
#
# The cache is a pure read-path detour: the same shard bytes, at the same
# offsets, and every member is still sha256-verified on read by ``ShardStore``
# (``verify="checksum"``) and by the published stores, so a corrupt cached shard
# fails loudly at the exact member rather than training on garbage.  It is
# keyed by the path *relative to the NFS export*, so a cache entry can only ever
# stand in for the file it was copied from.
SHARD_CACHE_ENV = "Q3VL_SHARD_CACHE"
DEFAULT_SHARD_CACHE = Path("/home/bc/data/shard_cache")
CACHE_MANIFEST_NAME = "cache_manifest.json"
#: absolute prefixes whose tails are usable as cache keys (both mounts export
#: the same tree, so ``/mnt/nfs/x`` and ``/mnt/nfs-ro/x`` share one key)
CACHEABLE_PREFIXES: tuple[str, ...] = tuple(dict.fromkeys(
    [src for src, _ in READ_PREFIX_REWRITE] + [dst for _, dst in READ_PREFIX_REWRITE]
))

_cache_lock = threading.Lock()
_cache_state: dict[str, Any] | None = None


def _load_cache_state() -> dict[str, Any]:
    """Read the cache manifest once per process.  Local IO only, never NFS."""
    raw = os.environ.get(SHARD_CACHE_ENV)
    if raw is not None and raw.strip().lower() in ("", "0", "off", "no", "none", "disabled"):
        return {"enabled": False, "root": None, "reason": f"{SHARD_CACHE_ENV} disables it",
                "entries": {}, "hits": 0, "misses": 0}
    root = Path(raw.strip()) if raw else DEFAULT_SHARD_CACHE
    manifest = root / CACHE_MANIFEST_NAME
    try:
        entries = json.loads(manifest.read_text(encoding="utf-8"))["entries"]
    except (OSError, KeyError, ValueError) as exc:
        return {"enabled": False, "root": str(root),
                "reason": f"no usable {manifest}: {type(exc).__name__}",
                "entries": {}, "hits": 0, "misses": 0}
    return {"enabled": True, "root": str(root), "reason": None,
            "entries": {k: int(v["bytes"]) for k, v in entries.items()},
            "hits": 0, "misses": 0}


def cache_state() -> dict[str, Any]:
    """Memoised cache manifest + hit counters.  Local IO only, never NFS."""
    global _cache_state
    if _cache_state is None:
        with _cache_lock:
            if _cache_state is None:
                _cache_state = _load_cache_state()
    return _cache_state


def reset_shard_cache() -> None:
    """Forget the memoised cache/mount state (tests and warm jobs only)."""
    global _cache_state, _ro_mounted
    with _cache_lock:
        _cache_state = None
        _ro_mounted = None


def cache_key_for(path: str | os.PathLike) -> str | None:
    """The cache-relative key of an export path, or ``None`` if not cacheable."""
    s = str(path)
    for prefix in CACHEABLE_PREFIXES:
        if s.startswith(prefix):
            return s[len(prefix):]
    return None


def resolve_read_path(path: str | os.PathLike) -> str:
    """``rewrite_read_path`` + local cache.  This is what open() should be given.

    A cache entry is used only when the manifest knows the key *and* the local
    file is exactly the size the manifest recorded -- so a half-copied shard (or
    one truncated by a full disk) is ignored rather than read.
    """
    s = rewrite_read_path(path)
    st = cache_state()
    if not st["enabled"]:
        return s
    key = cache_key_for(s)
    want = st["entries"].get(key) if key is not None else None
    if want is not None:
        local = os.path.join(st["root"], key)
        try:
            if os.stat(local).st_size == want:
                st["hits"] += 1
                return local
        except OSError:
            pass
    st["misses"] += 1
    return s


def shard_cache_facts() -> dict[str, Any]:
    """Provenance for ``run_setup.json``: which cache, how many entries, hit rate."""
    st = cache_state()
    return {"enabled": st["enabled"], "root": st["root"], "reason": st["reason"],
            "n_entries": len(st["entries"]),
            "cached_bytes": sum(st["entries"].values()),
            "open_hits": st["hits"], "open_misses": st["misses"]}


class _FdPool:
    """Per-(pid, thread) read-only fd cache.

    ``os.pread`` is positional, so a shared fd would be safe to *read* through --
    but the LRU eviction is not: one thread closing an fd another thread is about
    to pread yields EBADF, or worse, a read against whatever the number was
    recycled into.  Each thread therefore keeps its own table, which also makes
    the fork check (dataloader workers) a per-thread no-op.
    """

    def __init__(self, max_open: int = 32):
        self.max_open = max_open
        self._local = threading.local()
        self._tables: list[dict[str, int]] = []
        self._lock = threading.Lock()

    def table(self) -> dict[str, int]:
        pid = os.getpid()
        table = getattr(self._local, "table", None)
        if table is None or getattr(self._local, "pid", None) != pid:
            table = {}
            self._local.table = table
            self._local.pid = pid
            with self._lock:
                self._tables.append(table)
        return table

    def fd(self, key: str, path: str) -> int:
        table = self.table()
        fd = table.get(key)
        if fd is None:
            fd = os.open(path, os.O_RDONLY)
            if len(table) >= self.max_open:
                _, old = table.popitem()
                try:
                    os.close(old)
                except OSError:                                # pragma: no cover
                    pass
            table[key] = fd
        return fd

    def close(self) -> None:
        with self._lock:
            tables, self._tables = list(self._tables), []
        for table in tables:
            for fd in table.values():
                try:
                    os.close(fd)
                except OSError:
                    pass
            table.clear()
        self._local = threading.local()


class SliceReader:
    """``pread`` byte ranges out of shard tars, with fd caching + cache resolution.

    The published stores used to ``os.open``/``os.close`` per member; on NFSv4
    that made OPEN+CLOSE 72% of the client's RPC mix.  One fd per shard per
    thread removes both round trips without changing a single byte that is read.
    """

    def __init__(self, max_open: int = 16):
        self._pool = _FdPool(max_open)

    def read(self, path: str | os.PathLike, offset: int, length: int) -> bytes:
        fd = self._pool.fd(str(path), resolve_read_path(path))
        return os.pread(fd, length, offset)

    def close(self) -> None:
        self._pool.close()

    def __del__(self):                                          # best effort
        try:
            self.close()
        except Exception:                                       # pragma: no cover
            pass


class BoundedBytesCache:
    """FIFO byte cache with a hard budget.  Values are immutable ``bytes``.

    Used to let a prefetch thread pay for a published-store member read and have
    the training thread find it already there.  Nothing derived is cached (the
    JSON is re-parsed per call), so no caller can ever mutate another's object.
    """

    def __init__(self, max_bytes: int):
        self.max_bytes = int(max_bytes)
        self._data: dict[Any, bytes] = {}
        self._bytes = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: Any) -> bytes | None:
        with self._lock:
            hit = self._data.get(key)
            if hit is None:
                self.misses += 1
            else:
                self.hits += 1
            return hit

    def put(self, key: Any, data: bytes) -> None:
        if self.max_bytes <= 0 or len(data) > self.max_bytes:
            return
        with self._lock:
            if key in self._data:
                return
            self._data[key] = data
            self._bytes += len(data)
            while self._bytes > self.max_bytes and len(self._data) > 1:
                oldest = next(iter(self._data))
                self._bytes -= len(self._data.pop(oldest))

    def facts(self) -> dict[str, Any]:
        with self._lock:
            return {"max_bytes": self.max_bytes, "bytes": self._bytes,
                    "n_entries": len(self._data), "hits": self.hits,
                    "misses": self.misses}


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
    File descriptors are cached per (pid, thread, shard) so dataloader workers
    that fork after construction reopen their own handles, and so a prefetch
    thread can never close an fd another thread is about to pread (PERF-1).
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
        self._pool = _FdPool(max_open)
        self._counters = threading.Lock()
        self.n_reads = 0
        self.n_checksum_verified = 0

    @property
    def _fds(self) -> dict[str, int]:
        """This thread's open shards.  Kept as an attribute name for inspection."""
        return self._pool.table()

    def _fd(self, shard: str) -> int:
        table = self._pool.table()
        fd = table.get(shard)
        if fd is not None:
            return fd
        path = Path(shard)
        if not path.is_absolute():
            path = self.shard_root / shard
        # after joining, so a shard_root on the hard mount is covered too
        path = Path(resolve_read_path(path))
        if not path.exists():
            raise ShardIntegrityError(f"shard not found: {path}")
        return self._pool.fd(shard, str(path))

    def read(self, ref: MemberRef, verify: str | None = None) -> bytes:
        mode = self.verify if verify is None else verify
        fd = self._fd(ref.shard)
        data = os.pread(fd, ref.length, ref.offset)
        with self._counters:
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
            with self._counters:
                self.n_checksum_verified += 1
        return data

    def close(self) -> None:
        self._pool.close()

    def __del__(self):  # best effort
        try:
            self.close()
        except Exception:                                       # pragma: no cover
            pass


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
