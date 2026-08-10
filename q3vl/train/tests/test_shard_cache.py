"""The local read cache and the thread-safe fd pool (PERF-1).

The cache is on the read path of every unattended arm, so the properties that
matter are the ones that keep a *wrong* cache from being used silently:

* a key identifies the file it was copied from (export-relative), so a cache
  entry can never stand in for a different shard;
* a size that disagrees with the manifest is ignored, so a half-copied shard is
  a slow read and not a corrupt one;
* the published digest is checked at warm time, and the per-member digest is
  still checked at read time.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

import pytest

from q3vl.train import shardcache as SC
from q3vl.train import shards as S
from q3vl.train.shards import MemberRef, ShardStore


@pytest.fixture
def export(tmp_path, monkeypatch):
    """A fake NFS export + a cache root, with the module wired to both."""
    src = tmp_path / "export"
    (src / "ds" / "shards").mkdir(parents=True)
    cache = tmp_path / "cache"
    monkeypatch.setattr(S, "CACHEABLE_PREFIXES", (str(src) + "/",))
    monkeypatch.setattr(S, "READ_PREFIX_REWRITE", ())
    monkeypatch.setattr(S, "DEFAULT_SHARD_CACHE", cache)
    monkeypatch.delenv(S.SHARD_CACHE_ENV, raising=False)
    S.reset_shard_cache()
    yield src, cache
    S.reset_shard_cache()


def _warm(entries, cache, **kw):
    """Warm with the free-space floor disabled -- the tmp filesystem the tests
    run on is smaller than the production floor, and the floor has its own
    test."""
    kw.setdefault("free_floor", 0)
    return SC.warm(entries, cache, log=lambda *_: None, **kw)


def _publish(root: Path, name: str, payload: bytes) -> Path:
    tar = root / "ds" / "shards" / f"{name}.tar"
    tar.write_bytes(payload)
    manifest = root / "ds" / "manifest.json"
    shards = json.loads(manifest.read_text())["shards"] if manifest.exists() else []
    shards.append({"shard_id": name, "tar": f"shards/{name}.tar",
                   "tar_bytes": len(payload),
                   "tar_sha256": hashlib.sha256(payload).hexdigest()})
    manifest.write_text(json.dumps({"status": "complete", "shards": shards}))
    return tar


# -- path resolution --------------------------------------------------------

def test_no_manifest_means_the_cache_is_inert(export):
    src, _cache = export
    tar = _publish(src, "shard-00000", b"x" * 64)
    assert S.resolve_read_path(tar) == str(tar)
    assert S.shard_cache_facts()["enabled"] is False


def test_a_warmed_shard_is_read_from_the_cache(export):
    src, cache = export
    tar = _publish(src, "shard-00000", os.urandom(4096))
    _warm(SC.plan_dataset(src / "ds"), cache)
    S.reset_shard_cache()
    assert S.resolve_read_path(tar) == str(cache / "ds/shards/shard-00000.tar")
    assert S.shard_cache_facts()["n_entries"] == 1


def test_a_truncated_cache_entry_is_ignored_not_read(export):
    """A shard the disk filled up on must fall back, never be read short."""
    src, cache = export
    tar = _publish(src, "shard-00000", os.urandom(4096))
    _warm(SC.plan_dataset(src / "ds"), cache)
    S.reset_shard_cache()
    local = cache / "ds/shards/shard-00000.tar"
    with local.open("r+b") as fh:
        fh.truncate(100)
    assert S.resolve_read_path(tar) == str(tar)
    assert S.shard_cache_facts()["open_misses"] == 1


def test_a_path_outside_the_export_is_never_cacheable(export):
    _src, _cache = export
    assert S.cache_key_for("/somewhere/else/shard.tar") is None
    assert S.resolve_read_path("/somewhere/else/shard.tar") == "/somewhere/else/shard.tar"


def test_the_env_switch_disables_the_cache(export, monkeypatch):
    src, cache = export
    tar = _publish(src, "shard-00000", os.urandom(4096))
    _warm(SC.plan_dataset(src / "ds"), cache)
    monkeypatch.setenv(S.SHARD_CACHE_ENV, "0")
    S.reset_shard_cache()
    assert S.resolve_read_path(tar) == str(tar)


# -- warming ----------------------------------------------------------------

def test_warm_verifies_the_published_digest_and_refuses_a_mismatch(export):
    src, cache = export
    _publish(src, "shard-00000", os.urandom(4096))
    entries = SC.plan_dataset(src / "ds")
    bad = [SC.CacheEntry(source=entries[0].source, key=entries[0].key,
                         bytes=entries[0].bytes, sha256="0" * 64)]
    with pytest.raises(RuntimeError, match="sha256"):
        _warm(bad, cache)
    assert not (cache / bad[0].key).exists()          # no half-trusted leftover


def test_warm_is_idempotent_and_skips_what_is_already_there(export):
    src, cache = export
    _publish(src, "shard-00000", os.urandom(4096))
    entries = SC.plan_dataset(src / "ds")
    first = _warm(entries, cache)
    second = _warm(entries, cache)
    assert first["copied"] == 1 and second["copied"] == 0 and second["skipped"] == 1


def test_warm_refuses_to_break_the_free_space_floor(export):
    src, cache = export
    _publish(src, "shard-00000", os.urandom(4096))
    with pytest.raises(RuntimeError, match="floor"):
        SC.warm(SC.plan_dataset(src / "ds"), cache,
                free_floor=1 << 62, log=lambda *_: None)


def test_verify_reports_a_corrupted_cache_entry(export):
    src, cache = export
    _publish(src, "shard-00000", os.urandom(4096))
    _warm(SC.plan_dataset(src / "ds"), cache)
    assert SC.verify(cache, deep=True, log=lambda *_: None)["n_bad"] == 0
    local = cache / "ds/shards/shard-00000.tar"
    data = bytearray(local.read_bytes())
    data[0] ^= 0xFF
    local.write_bytes(bytes(data))
    assert SC.verify(cache, deep=True, log=lambda *_: None)["n_bad"] == 1
    assert SC.verify(cache, deep=False, log=lambda *_: None)["n_bad"] == 0  # size still ok


# -- ShardStore over the cache ----------------------------------------------

def _ref(shard: Path, offset: int, data: bytes) -> MemberRef:
    return MemberRef(shard=str(shard), member="m", offset=offset, length=len(data),
                     size=len(data), checksum=hashlib.sha256(data).hexdigest())


def test_shard_store_reads_identical_bytes_through_the_cache(export):
    src, cache = export
    payload = os.urandom(8192)
    tar = _publish(src, "shard-00000", payload)
    store = ShardStore("/")
    direct = store.read(_ref(tar, 512, payload[512:1024]))
    _warm(SC.plan_dataset(src / "ds"), cache)
    S.reset_shard_cache()
    store2 = ShardStore("/")
    assert store2.read(_ref(tar, 512, payload[512:1024])) == direct
    assert str(cache) in os.readlink(f"/proc/self/fd/{store2._fds[str(tar)]}")


def test_a_corrupt_cached_shard_fails_at_the_member_not_silently(export):
    src, cache = export
    payload = os.urandom(8192)
    tar = _publish(src, "shard-00000", payload)
    _warm(SC.plan_dataset(src / "ds"), cache)
    S.reset_shard_cache()
    local = cache / "ds/shards/shard-00000.tar"
    blob = bytearray(local.read_bytes())
    blob[600] ^= 0xFF
    local.write_bytes(bytes(blob))                    # same size -> cache still used
    with pytest.raises(S.ShardIntegrityError, match="mismatch"):
        ShardStore("/").read(_ref(tar, 512, payload[512:1024]))


# -- the fd pool ------------------------------------------------------------

def test_each_thread_gets_its_own_descriptors(export):
    src, _cache = export
    payload = os.urandom(8192)
    tar = _publish(src, "shard-00000", payload)
    store = ShardStore("/")
    ref = _ref(tar, 0, payload[:256])
    store.read(ref)
    mine = dict(store._fds)
    seen: list[dict] = []

    def worker():
        store.read(ref)
        seen.append(dict(store._fds))

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert seen and seen[0][str(tar)] != mine[str(tar)]


def test_concurrent_reads_agree_with_serial_reads(export):
    """max_open=1 forces eviction on every open: the pre-PERF-1 shared dict
    would have closed another thread's fd here."""
    src, _cache = export
    blobs = {}
    for i in range(4):
        payload = os.urandom(4096)
        blobs[_publish(src, f"shard-{i:05d}", payload)] = payload
    store = ShardStore("/", max_open=1)
    refs = [_ref(t, 128, p[128:256]) for t, p in blobs.items()]
    expect = [p[128:256] for p in blobs.values()]

    out: list[list[bytes]] = []
    errs: list[BaseException] = []

    def worker():
        try:
            out.append([store.read(r) for _ in range(20) for r in refs][:len(refs)])
        except BaseException as exc:                             # noqa: BLE001
            errs.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errs, errs
    assert all(got == expect for got in out)


# -- the bounded byte cache -------------------------------------------------

def test_bounded_bytes_cache_evicts_oldest_and_stays_under_budget():
    c = S.BoundedBytesCache(300)
    for i in range(5):
        c.put(i, bytes(100))
    assert c.facts()["bytes"] <= 300
    assert c.get(0) is None and c.get(4) is not None


def test_bounded_bytes_cache_never_stores_an_oversized_payload():
    c = S.BoundedBytesCache(100)
    c.put("big", bytes(101))
    assert c.get("big") is None
