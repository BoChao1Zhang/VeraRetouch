# Canonical Databuild Viewer

Inspection of canonical databuild artifacts. The backend prefers the
`canonical_*` PostgreSQL projection and falls back to `groups.jsonl`, `sft.jsonl`,
`failures.jsonl`, and `manifest.json` when PostgreSQL is unavailable. Image assets
may remain only in immutable indexed NFS shards; selecting a group locates its
members first and materializes only those members into a bounded disposable cache.

The inspector exposes every complete eight-candidate group, source and rendered
images, local `C_GT` and mask overlay views, visibility and OneAlign metrics,
subject/region metadata, top-2 SFT annotations, and structured failure events.
Filters cover build, render mode, preset format and taxonomy, annotation queue,
failure state, and winner rank.

## Backend

Use the same owner-only canonical TOML as the build. Its
`[viewer].postgres_dsn` supplies the preferred projection connection. The JSONL
fallback always defaults to the durable `/mnt/nfs/bc/data/builds` ledger;
`output_root` is tmpfs by contract and is never used as fallback authority.

```bash
cd /home/bc/VeraRetouch
python -m pip install -e '.[viewer]'
python -m databuild_viewer.backend.app \
  --config /absolute/path/to/databuild.toml
```

The server listens on `127.0.0.1:8077` by default. `--host` and `--port` are
explicit overrides. The selfcheck is read-only:

```bash
python -m databuild_viewer.backend.app \
  --config /absolute/path/to/databuild.toml --selfcheck
```

The current NFS ledger is the no-flag default. The backend reads only its small
manifests for health/build discovery, then serially parses the explicitly selected
build after releasing the previous build's records. Duplicate `build_id` values
from different resolved directories fail closed; overlapping roots that discover
the same directory are deduplicated. `--nfs-ledger` may select a single build or
another ledger root. It may be combined with `--config`:
PostgreSQL remains preferred and NFS remains the authoritative fallback:

```bash
python -m databuild_viewer.backend.app \
  --config /absolute/path/to/databuild.toml \
  --nfs-ledger /mnt/nfs/bc/data/builds/mini30-v52-20260730 \
  --catalog /var/cache/veradata/global.sqlite3
```

For another JSONL-only location, use `--build-root`. Add historical logical image
roots only when they are outside the canonical defaults:

```bash
python -m databuild_viewer.backend.app \
  --build-root /absolute/path/to/build \
  --allow-root /absolute/historical/image/root
```

The local member cache defaults to
`${XDG_CACHE_HOME:-~/.cache}/databuild_viewer` and 32 GiB. It never defaults to
NFS and never extracts an entire shard or build. One process owns a cache root at
a time; after acquiring that owner lock, startup removes every dot-prefixed
regular residue in the viewer-owned disposable `members/` directory, including
current, legacy, and unknown partial names. Every archive cache hit is verified
against the indexed size and SHA-256, and the byte limit includes all regular
files under `members/` including active partials. Constructor failures release
the owner lock before returning an error. The owner lock lives
outside `members/` and is not charged as payload. Configure the cache with
`--cache-root` / `DBV_CACHE_ROOT` and `--cache-bytes` / `DBV_CACHE_BYTES`.
`--catalog` / `DBV_CATALOG` selects the existing global archive index without
copying it. `DBV_CONFIG`, `DBV_BUILD_ROOTS`, and `DBV_ALLOWED_ROOTS` retain their
existing process-start behavior. `DBV_CONFIG` contains only a path; credentials
remain inside the ignored `0600` TOML.

The prepare protocol is versioned. `build_id` is required on group detail and
prepare calls. `POST /api/groups/{group_id}/prepare?build_id=...` starts or
deduplicates a job; `GET` on the same path returns `queued`, `locating`,
`materializing`, `ready`, or `failed` with file and byte counters. A failed or
lost job restarts only with `POST ...?build_id=...&retry=true`. Archived `/img`
reads are cache-only and are served only after a canonical group is `ready`;
they never bypass preparation with a direct shard read. Authorization returns one
canonical path string which prepare, catalog lookup, cache naming, and full/thumb
reads share. Shutdown and reconfigure cooperatively cancel queued/running work
and wait for the single worker before releasing cache ownership.

Two operational limits are intentional: the bounded single-worker FIFO can make
a newly selected group wait behind older queued work, and local ordinary-file
serving assumes allowed local roots are trusted against replacement between path
validation and open. Hard-mounted NFS reads may also delay cooperative shutdown
until the current kernel read returns.

## Frontend

```bash
cd databuild_viewer/frontend
npm install
npm run dev
```

Vite listens on `127.0.0.1:5173` and proxies `/api` and `/img` to
`http://127.0.0.1:8077`. For a single-port production-style run, build first and
then start the backend:

```bash
npm run build
```

The compiled SPA is included in the Python wheel; rebuild it only after frontend
source changes.

## Verification

```bash
uv run --no-project --no-progress --with fastapi==0.136.1 --with Pillow==12.2.0 \
  python -m unittest \
    databuild_viewer.backend.test_app \
    databuild_viewer.backend.test_materialize -v
uv run --no-project --no-progress --with Pillow==12.2.0 \
  python -m unittest \
    dataset_build.tests.test_prefetch_and_sft_pack.PrefetchTests -v
cd databuild_viewer/frontend && npm run build
cd databuild_viewer/frontend && npm run test:e2e   # Playwright desktop/mobile screenshots
uv build --wheel --out-dir /tmp/veraretouch-wheel
unzip -l /tmp/veraretouch-wheel/*.whl | rg 'databuild_viewer/frontend/dist/'
```
