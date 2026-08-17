# DBV-NFS-1 Implementation Notes

## Scope

Repair `databuild_viewer` as an indexed-shard consumer of the canonical NFS
ledger and immutable `groups` datasets. This is tooling work, not a training or
evaluation experiment.

## Contracts Read Before Implementation

- `CLAUDE.md`: a tooling task must record source verification, assumptions, and
  decisions before implementation. The s-cache numeric-domain and long-job
  rules do not apply to this viewer request.
- `docs/_archive/2026-08-12/july-docs/DATABUILD_IMPLEMENTATION_2026-07-27.md` section 1 and stages
  D/E: the four canonical JSONL files plus `manifest.json` are the ledger;
  PostgreSQL is only a rebuildable viewer projection. Landed asset paths remain
  their historical staging keys and must be resolved through the archive index.
- `docs/_archive/2026-08-12/july-docs/DATABUILD_IO_REFACTOR_2026-07-27.md` section 4 step 4 and
  section 6: `datasets/groups/<build_id>` is authoritative for every candidate
  and `C_GT`; `datasets/sft/<build_id>` is a winner-only derivative. A group's
  members are physically adjacent. Published datasets are uncompressed USTAR
  plus durable JSONL indexes and a derived SQLite catalog.
- `docs/DATA_ASSIGNMENT_2026-08-02.md` section 1: archived `.in.jpg` is a VLM
  preview, not canonical `I_in`. The viewer may display recorded source/preview
  assets but must not relabel `.in.jpg` as reconstructed `I_in`.
- `build-indexed-datasets/SKILL.md` Consumer Contract: resolve random reads by
  index and seek; ordinary paths may use only a bounded extraction cache; the
  terminal contract must not become a long-lived extracted small-file tree.

## Repository API Verification

- `dataset_build.tools.prefetch.prefetch(source_paths, dest, db_path=...)`
  deduplicates requested historical paths, resolves them through the global
  SQLite catalog, orders rows by `(root, shard, offset_data)`, validates the tar
  header, and publishes each flat cache member with `os.replace`.
- `dataset_build.tools.archive_reader.read_bytes()` is local-first, then the
  configured prefetch directory, then indexed archive read. `prefetch_name()`
  is the stable SHA-256 cache key for a historical logical path.
- `databuild_viewer.backend.repository.ViewerRepository.group(group_id, build_id)`
  returns the build-scoped canonical group, eight candidates, SFT rows, and
  related failures. Those records contain every image path needed for one
  selected-group prepare job.
- Existing `/img` requires `Path.resolve(strict=True)` and therefore rejects
  correctly indexed historical paths after their staging files are retired.

## Official Sources Opened

All external facts used below were checked against the original documentation
on 2026-08-04; no unvisited URL is relied on.

- Python `os.replace`: https://docs.python.org/3/library/os.html#os.replace
  states that an existing file destination is replaced silently and a
  successful rename is atomic when source and destination are on the same
  filesystem. Cache partials and final entries therefore remain in one cache
  directory/filesystem.
- Python `ThreadPoolExecutor`:
  https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor
  documents asynchronous callable execution and executor shutdown semantics.
  The viewer owns one bounded executor and explicitly shuts it down when
  reconfigured in tests or process teardown.
- React `useEffect`: https://react.dev/reference/react/useEffect documents that
  cleanup runs before an effect with changed dependencies and after unmount.
  The retained polling hook therefore clears its timer and ignores late results
  when the selected group changes.
- FastAPI `BackgroundTasks`:
  https://fastapi.tiangolo.com/tutorial/background-tasks/ was opened to verify
  its response-lifecycle behavior. The implementation deliberately uses a
  viewer-owned executor instead because jobs need cross-request deduplication,
  status snapshots, and cache lifecycle ownership.

## Assumptions And Conservative Defaults

- "Pre-unpack the shard" means index-first materialization of only the selected
  group's required members, in archive physical order. Whole-shard or
  whole-build extraction is out of scope.
- The canonical catalog defaults to `/var/cache/veradata/global.sqlite3`; it is
  never copied into the viewer cache. `DBV_CATALOG` and CLI `--catalog` may
  override it.
- The ledger defaults to `/mnt/nfs/bc/data/builds`. Existing `--config` and
  explicit `--build-root` behavior remains available.
- The disposable cache defaults to
  `${XDG_CACHE_HOME:-~/.cache}/databuild_viewer`, with a 32 GiB byte limit.
  `DBV_CACHE_ROOT`, `DBV_CACHE_BYTES`, `--cache-root`, and `--cache-bytes`
  override it.
- Polling is used instead of SSE/WebSocket. Snapshots are schema-versioned and
  the frontend retains the last successful snapshot while refreshing.
- Cache eviction is least-recently-used by completed entry mtime. Active paths
  are pinned; after acquiring the exclusive owner lock, every dot-prefixed
  regular residue in the disposable member directory is removed, covering
  current, legacy, and unknown partial names. All remaining member-directory
  files count toward the hard byte limit, and strict cache hits/final readiness
  verify indexed size and SHA-256. Constructor failures release the owner lock.
- Authorization runs before any cache lookup. The original supplied path must
  use absolute normalized syntax (no tilde, relative form, `..`, `//`, NUL, or
  symlink escape), have an allowed image suffix, and resolve under an explicitly
  configured allowed root. The returned canonical path string is the sole key
  used by prepare, catalog lookup, cache naming, and full/thumb reads. Catalog
  membership alone grants no access.
- The durable NFS ledger is the default fallback even when a TOML supplies the
  preferred PostgreSQL DSN. Build discovery reads manifests only; canonical
  JSONL load/switch operations are serialized, release the old active build
  before parsing the next, and retain one active build. Different directories
  declaring one `build_id` fail closed; overlapping roots resolving to the same
  directory are deduplicated.
- The cache root has one cross-process `flock` owner. Reconfigure and FastAPI
  shutdown set cooperative cancellation, wait for the worker to exit, and only
  then release ownership; a closed job cannot transition to `ready`.
- Archived `/img` is cache-only and requires a fully verified entry owned by a
  ready canonical group. Group-rail rows intentionally use icon placeholders so
  they cannot launch random archive reads before selection.
- Residual operational risks: the bounded one-worker FIFO does not reprioritize
  newer selections; local ordinary-file serving trusts allowed roots against a
  replace-after-validation race; and a hard-mounted NFS `pread` cannot observe
  cooperative cancellation until the kernel call returns.
- Runtime model inventory had no Opus entry (`pi --list-models opus` returned no
  matches); this worker uses the available runtime as approved in the task card.

## Pending Main-Agent Decisions

None. The task card already approved the materialization granularity, cache
location/limit, polling transport, and source-image semantics.
