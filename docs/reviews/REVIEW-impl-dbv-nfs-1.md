# REVIEW-impl-dbv-nfs-1 - DBV-NFS-1 independent implementation review

**Current final verdict (Round 3, 2026-08-04): APPROVED - 0 BLOCKER, 3 NIT.**

## Round 3 Review

- Correct: all five Round 2 blockers are repaired. B2, B3, B7, B9, and B10 passed direct adversarial reproductions against the current worktree, and the full B1-B8 regression surface passed.
- Fixed: none by this reviewer. This remained review-only; no implementation or project file was changed.
- Blocker: none. No B11 or later blocker was found.
- Note: approval retains three non-blocking nits for FIFO latency, trusted-local TOCTOU, and permanent-test completeness/timing sensitivity. The operational NFS and large-catalog risks remain explicit below.

## Round 3 Metadata

| Field | Value |
|---|---|
| Date | 2026-08-04 |
| Review type | Third and final independent implementation review; current diff and behavior checked directly; worker handoff not trusted |
| Reviewer runtime | No Opus runtime was available. The review used the available reviewer model; this requested-runtime limitation is recorded explicitly. |
| Verdict | **APPROVED** |
| Findings | **0 BLOCKER, 3 NIT** |
| Git action | No commit; no staging |
| Modified by reviewer | Only `docs/reviews/REVIEW-impl-dbv-nfs-1.md` |

The indexed dataset skill was reread in full. `CLAUDE.md`, the DataBuild implementation and IO storage/publication sections, and the `.in.jpg` rule in `DATA_ASSIGNMENT_2026-08-02.md` were rechecked before the diff. No new external semantic claim was needed; Round 1's official-source record remains preserved below.

## Round 3 Contract Matrix

| Contract area | Status | Round 3 evidence / disposition |
|---|---|---|
| No flags / config only / config + root override / env only | **PASS** | Actual `0600` `databuild.eval100.toml` loading retained a non-empty PG DSN while no-flags and config-only selected `/mnt/nfs/bc/data/builds`; explicit and two-root env overrides selected only their requested roots. An actual empty-root selfcheck returned 0/0. A temporary `0644` config raised redacted `ConfigError`. |
| PG-down JSONL fallback and secret redaction | **PASS** | A credential-bearing forced connection failure fell back with `source=jsonl`, `fallback_reason=postgres_unavailable`, and 30 mini-ledger groups. Neither the DSN nor marker appeared in serialized output. |
| Full-ledger manifest-only health/build discovery | **PASS** | With `_read_jsonl` patched to raise, the real ledger returned 34 builds and 203,075 manifest-estimated groups from 2,606,415 manifest bytes, kept `active_build=null`, and used 28,480 KiB max RSS. Direct `api_health()` and `api_builds()` likewise made zero JSONL calls. |
| One selected build resident / blocking concurrency / stale publication | **PASS (B2 cleared)** | A blocked B load followed by a fast C request produced maximum parser concurrency 1; C did not enter until B released, every B/C read observed no old active object, completion order was B then C, and final active was C. A forced parse failure left no active ID/data. |
| Build-scoped groups/facets/detail/prepare and refresh | **PASS** | API calls require `build_id`; PG and JSONL detail remain build-scoped; a single-build no-ID repository call remains compatible; focused refresh and malformed-ledger tests passed. |
| Duplicate manifest IDs / overlapping roots / recovery | **PASS (B10 cleared)** | Two distinct directories declaring `duplicate-id` raised exactly `duplicate_build_id:duplicate-id`, exposed no root, cleared both manifests and a previously active build, then recovered after one manifest was repaired. Overlapping discovery of the same resolved directory reported one build. |
| Current/legacy/unknown residue recovery | **PASS (B3 cleared)** | After exclusive ownership, current UUID partial, legacy PID partial, unknown dot file, and unknown dot symlink were all unlinked. The external symlink target, root `.owner.lock`, and a checksum-checked legal final remained intact. |
| Hard byte bound / write peak / checksum / LRU / active pin | **PASS** | Independent probes retained the recent LRU entry, evicted the oldest, retained an active pinned entry while evicting unpinned data, and held measured write peak/final use at 80/80 bytes under a 150-byte limit. Equal-size corruption, shard corruption, ready-read digest invalidation, oversize rejection, and real-NFS checksum reads passed. |
| Constructor unwind / ownership / idempotent close | **PASS (B9 cleared)** | Forced failure separately at cleanup, capacity enforcement, and executor construction returned `/proc/self/fd` to 4 each time, immediately reacquired the same cache, and returned to 4 after two `close()` calls. |
| Queued/active close / reconfigure / lifespan | **PASS** | Reconfigure blocked until the active worker exited, active and queued jobs remained failed, no materializer thread survived, the same cache was immediately reacquired, close was idempotent, and lifespan invoked close exactly once. |
| Rail zero-read / cache-only archived `/img` | **PASS** | Playwright observed zero `/img` requests at 4/10 files; archived cache miss stayed 409; ready full/thumb decode passed. No direct archive-read fallback exists in `/img`. |
| Path authorization and one canonical key | **PASS (B7 cleared)** | Prepare, materializer/catalog/cache key, full read, and thumbnail read all received the identical resolved in-root alias string. Relative, tilde, lexical/decoded traversal, double slash, prefix collision, symlink escape/loop, NUL, and unsupported suffix cases failed closed with controlled 4xx responses. |
| Lost job / 503 / stale response / unmount | **PASS** | Final Playwright run passed all cases: GET 404 stops polling and one explicit retry POST recovers; temporary 503 recovers with bounded backoff; a late old-group failure cannot replace the new group; unmount stops polling. |
| Production `prefetch()` compatibility and `.in.jpg` | **PASS** | `PrefetchResult` remains a `dict` subclass; all three real `_SourcePrefetch` callers passed. Asset collection excludes SFT `I_in`, including `.in.jpg`, and uses canonical group `source_path`; the viewer never presents `.in.jpg` as canonical input. |
| Frontend source/dist/screenshots | **PASS with N3 timing note** | Independent Vite output was byte-identical to checked-in dist. The final full Playwright run passed 15/15 and the progress case passed 5/5 targeted repeats. Latest desktop/mobile screenshots were opened directly and showed no incoherent overlap or document-width escape. |
| Real NFS cold/resume selected-group path | **PASS** | A disposable-cache smoke materialized and decoded 16/16 mini30 assets. Cold cache used 16 publishes and 2,900,111 bytes peak/final under 128 MiB; reopen/resume decoded 16/16 with zero publish calls and the same final usage. No production cache or NFS path was written. |

## Round 3 B2-B10 Disposition

| Finding | Status | Disposition |
|---|---|---|
| B2 build loading / residency | **PASS** | `repository.py:363-533` serializes manifest refresh and the entire load/switch, releases old active state before reads, clears on failure, and cannot publish a stale concurrent parser result. Manifest-only full-ledger discovery passed independently. |
| B3 cache residue / bound | **PASS** | `materialize.py:129-161,353-413` cleans all dot file/symlink residue only after exclusive ownership, accounts all regular member bytes, preserves legal finals/owner lock, and retains peak/LRU/pin behavior. |
| B4 cache and shard integrity | **PASS** | Indexed SHA-256 remains enforced on shard payload, existing final, ready state, and served cache read. Wrong-size and equal-size corruption paths passed. |
| B5 lifecycle | **PASS** | Active/queued close, reconfigure wait, worker exit, ownership transfer, idempotent close, and FastAPI lifespan close passed. |
| B6 archive gating | **PASS** | Rail is image-free until selected-group readiness and archived `/img` is verified-cache-only. |
| B7 authorization / canonical identity | **PASS** | `app.py:216-323` now propagates `_authorized_image_path()`'s canonical return value through prepare and both `/img` variants; the complete security matrix remained fail-closed. |
| B8 frontend recovery | **PASS** | Lost-job, transient-503, explicit retry, stale-response, rapid-switch, and unmount behavior passed Playwright. |
| B9 failed constructor ownership | **PASS** | `materialize.py:129-161` unwinds executor, flock, descriptor, and state for every tested post-lock constructor failure site; immediate reacquire and descriptor stability passed. |
| B10 duplicate build identity | **PASS** | `repository.py:378-445` deduplicates one resolved directory, rejects distinct-directory ID conflicts without paths, clears partial/stale state, and refreshes after repair. |

## Round 3 Findings

### NIT N1 - Bounded FIFO can still delay the newest interactive selection

- Location: `databuild_viewer/backend/materialize.py:21-22,203-220`; `databuild_viewer/frontend/src/Inspector.jsx:288-344`.
- Evidence: outstanding and retained jobs are bounded and one worker preserves the one-reader contract. Navigation cancels frontend polling but does not cancel or reprioritize an already queued backend job, so a current selection can still wait behind up to seven abandoned jobs.
- Minimum improvement: coalesce queued jobs no longer selected or define latest-selection priority with multi-client fairness. This is latency/ergonomics, not a correctness blocker.

### NIT N2 - Trusted local-file serving remains TOCTOU-sensitive

- Location: `databuild_viewer/backend/app.py:216-243,299-308`.
- Evidence: syntax, containment, suffix, symlink escape, and malformed input fail closed. For an allowed ordinary local file, validation/stat and `FileResponse` or thumbnail open remain separate operations, so a writable trusted tree can be replaced between them.
- Minimum improvement: serve from a no-follow opened descriptor with post-open containment/stat validation, or retain the documented immutable same-user local-tree trust assumption.

### NIT N3 - Some exhaustive regression variants exist only in review probes, and one progress test is timing-sensitive

- Location: `databuild_viewer/backend/test_materialize.py:154-179,221-238`; `databuild_viewer/backend/test_app.py:357-472,728-772`; `databuild_viewer/frontend/e2e/viewer.spec.js:388-398`.
- Evidence: checked-in tests cover residue cleanup, one cleanup-site constructor failure, serialized switching, duplicate recovery, and canonical alias identity. Capacity-site/executor-site unwind, preservation of a preexisting active build across duplicate failure, actual LRU/active-pin variants, and the complete path matrix were proven by independent Round 3 probes but are not all permanent tests. The first full Playwright run missed the short intermediate progress window (14/15); the same case then passed 5/5 targeted repeats and the final full run passed 15/15.
- Minimum improvement: promote the independent matrices into permanent tests and make progress states deterministic in the fixture rather than time-window dependent. This does not block approval because all variants reproduced correctly.

## Round 3 Validation Commands

| Command / probe | Result |
|---|---|
| `DBV_CACHE_ROOT=/tmp/... uv run --no-project --no-progress --with fastapi==0.136.1 --with Pillow==12.2.0 python -m unittest databuild_viewer.backend.test_app databuild_viewer.backend.test_materialize dataset_build.tests.test_prefetch_and_sft_pack -v` | **PASS**, 55 tests |
| `PYTHONPATH=dataset_build/src /home/bc/.venvs/iaa437/bin/python -m unittest dataset_build.tests.test_land_integration.PrefetchWiringTests -v` | **PASS**, 3 actual production-caller tests |
| Independent B2/B10 temporary-ledger blocking concurrency, parse failure, duplicate/redaction/repair/overlap probe | **PASS**, parser max 1; old active absent before reads; final active C; parse failure none; duplicate state cleared/recovered; overlap count 1 |
| Independent B3 current/legacy/unknown/symlink residue + legal final/owner + LRU/pin/peak/checksum probe | **PASS**, all residue removed; owner/final retained; LRU/pin correct; 80-byte peak/final under 150-byte limit |
| Independent B9 cleanup/capacity/executor constructor-failure matrix with `/proc/self/fd` | **PASS**, 4 -> 4 descriptors for all three sites; immediate reacquire; double close and second reacquire passed |
| Independent B7 alias/canonical full/thumb plus traversal/security matrix | **PASS**, one identical canonical key; all malformed/escape variants controlled 4xx |
| Independent active+queued close, same-cache reconfigure, thread, idempotent close, lifespan probe | **PASS**, both jobs failed, configure waited, zero worker threads, same-cache reacquire, lifespan close count 1 |
| Actual config/default/override/private-mode + PG-down mini-ledger probe, plus a fresh process with two-root `DBV_BUILD_ROOTS` | **PASS**, durable and env roots correct; PG retained/fallback sanitized; 30 groups; `0644` rejected |
| `python -m databuild_viewer.backend.app --nfs-ledger /mnt/nfs/bc/data/builds/mini30-v52-20260730 --catalog /var/cache/veradata/global.sqlite3 --selfcheck` with temp cache | **PASS**, JSONL source; 1 build; 30 groups; 0 malformed |
| `/usr/bin/time -v` real full-ledger manifest-only store probe with `_read_jsonl` forbidden | **PASS**, 34 builds; 2,606,415 manifest bytes; 203,075 estimated groups; 28,480 KiB max RSS; no active build |
| Direct full-ledger `api_health()` / `api_builds()` with `_read_jsonl` forbidden and temp cache | **PASS**, 34/34; zero JSONL calls; no active build |
| Real NFS mini30 cold prepare + close/reopen resume + checksum read + Pillow decode using a temporary cache | **PASS**, 16/16 twice; cold/resume 0.077/0.026 s; 16/0 publishes; 2,900,111-byte peak/final under 128 MiB |
| `npm --prefix databuild_viewer/frontend run build -- --outDir /tmp/dbv-round3-dist --emptyOutDir` + `diff -rq` + SHA-256 | **PASS**, 1,580 modules; source build and checked-in dist byte-identical |
| `npm --prefix databuild_viewer/frontend run test:e2e` | **PASS final**, 15/15. First full run was 14/15 due to one missed intermediate progress window; correct targeted command then passed 5/5 and final full rerun passed. |
| Direct visual reads of latest desktop overlay/materialized and mobile mask/materializing/error screenshots | **PASS**, UI nonblank and contained; no incoherent overlap; mobile long detail and progress stayed within viewport |
| Focused `.in.jpg` asset-collection test | **PASS**, preview `I_in` excluded; canonical source/candidates/masks retained |
| `git diff --check -- databuild_viewer dataset_build/tools/prefetch.py dataset_build/tests/test_prefetch_and_sft_pack.py docs/reviews/REVIEW-impl-dbv-nfs-1.md` | **PASS** after the Round 3 review edit |

Three preliminary custom-probe attempts had reviewer harness mistakes: a `Path` concatenation type error, a symlink-loop fixture that accidentally checked a different missing filename, and a direct `npm exec` invocation that omitted the project base URL. A timed command also initially used an unavailable bare `python` name. Corrected probes above passed. These are not implementation failures. The first full Playwright timing miss is separately retained in N3 rather than hidden.

## Round 3 Confirmed Correct Behavior

- The durable NFS ledger and PG projection/fallback relationship now matches the authoritative storage contract in every startup mode tested.
- Full-ledger discovery is manifest-only and bounded, while selected JSONL parsing is serialized with old-state release and fail-empty behavior.
- Cache ownership, crash-residue recovery, byte reservation, checksums, LRU, active pins, synchronous shutdown, and failed-constructor unwind behave coherently.
- Duplicate build identity and image-path authorization fail closed without root/credential disclosure; accepted aliases use one canonical catalog/cache/read key.
- Archived images cannot bypass selected-group preparation, the rail performs no pre-ready reads, and `.in.jpg` is not treated as canonical `I_in`.
- Frontend recovery, stale-response suppression, responsive containment, checked-in dist identity, real NFS cold/resume decoding, and production prefetch callers all passed.

## Round 3 Residual Risks

- A hard-mounted NFS `os.pread` can block in the kernel; cooperative cancellation and `shutdown(wait=True)` cannot bound shutdown latency until that call returns.
- The immutable global SQLite catalog is 19,629,056,000 bytes. It is not copied and normal access is one-worker/immutable, but cold mmap/page-fault latency remains deployment-dependent.
- JSONL refresh signatures use `(mtime_ns, size)` and assume atomic canonical publication; a deliberately in-place rewrite preserving both can evade refresh.
- Cache ownership is one process per root. Multi-worker deployment must use distinct roots or preserve the explicit failure-on-second-owner behavior.
- Full digest verification adds payload reads at cache trust and serve boundaries. Correctness passed; large-group and multi-client cold latency were not benchmarked.
- Local ordinary files remain outside the extraction-cache budget and rely on the trusted local-tree assumption in N2.

## Round 2 Review

- Correct: B1, B4, B5, B6, and B8 are behaviorally repaired. The durable NFS default/config/PG fallback matrix, checksum repair and rejection, synchronous close/reconfigure/lifespan behavior, cache-only archived image boundary, frontend recovery, source/dist identity, and real-NFS cold/resume prepare all passed.
- Fixed: none. This was a review-only task; no implementation or project file was modified.
- Blocker: B2, B3, and B7 remain open. New blockers B9 and B10 cover a leaked cache owner after constructor failure and silent duplicate-manifest shadowing.
- Note: the implementation remains fail-closed on cache capacity, but old/unknown dot entries can make the dedicated disposable cache permanently unusable. That is an availability blocker, not an acceptable steady-state fail-closed condition.

## Round 2 Metadata

| Field | Value |
|---|---|
| Date | 2026-08-04 |
| Review type | Second independent implementation review; current diff and behavior rechecked directly; fix-worker handoff not trusted |
| Reviewer runtime | No Opus runtime was available. The review used the available reviewer model; this requested-runtime limitation is recorded explicitly. |
| Verdict | **BLOCKED** |
| Findings | **5 BLOCKER, 3 NIT** |
| Git action | No commit; no staging |
| Modified by reviewer | Only `docs/reviews/REVIEW-impl-dbv-nfs-1.md` |

No new external semantic claim was needed in Round 2. Findings below are based on repository contracts, direct code inspection, local executions, real NFS reads, and temporary-cache reproductions. Round 1's official-source record remains preserved below.

## Round 2 Contract Matrix

| Contract area | Status | Round 2 evidence / disposition |
|---|---|---|
| No flags / config only / config + root override / env only | **PASS (B1 cleared)** | Default and config-only selected `/mnt/nfs/bc/data/builds`; config retained a non-null PG DSN; `DBV_BUILD_ROOTS` selected both env roots; config + explicit empty build root selfchecked 0 builds rather than scanning NFS. A `0644` config was rejected with `ConfigError`; no DSN/password appeared in output. |
| PG-down JSONL fallback and secret redaction | **PASS (B1 cleared)** | Forced connection failure returned `source=jsonl`, `fallback_reason=postgres_unavailable`, one mini build, and 30 groups; serialized output did not contain the credential marker. |
| Full-ledger `/api/health` and `/api/builds` | **PASS (B2 partly fixed)** | Real 34-build ledger opened 2,606,415 manifest bytes, made zero `_read_jsonl` calls, left `active_build=null`, used 21,372 KiB max RSS, and reported `read_bytes` delta 0. Direct API calls likewise made zero JSONL calls. |
| One selected build resident / build switching / JSONL thread safety | **BLOCKER (B2 remains)** | While build B's three JSONLs were being parsed, `store._active_id` remained `build-a` for every read. `_load` parses outside the lock, so concurrent requests can parse multiple large builds and race publication. |
| `build_id` scope: groups/facets/detail/prepare | **PASS except duplicate identity B10** | APIs reject missing `build_id`; PG candidate/SFT/failure queries include build scope; JSONL loads the requested build; materializer job keys include build ID; E2E asserts every scoped frontend request includes it. Before build selection, frontend resolves empty data and makes no group request. |
| Single-build no-`build_id`, JSONL mtime refresh | **PASS** | Single-build `group(..., build_id=None)` remained compatible. Rewriting `failures.jsonl` changed malformed count from 0 to 1 on the next load. |
| Duplicate build manifests | **BLOCKER (B10)** | Two different directories declaring `build_id=duplicate` produced one reported build and silently selected the lexicographically first directory. |
| Current stale partial cleanup and hard byte accounting | **PASS only for current naming** | Current UUID partials are removed; every regular member file is counted; capacity failure is fail-closed; real cold prepare peak/final usage stayed under the bound. |
| Legacy/unknown partial recovery | **BLOCKER (B3 remains)** | Old `.<hash>.<pid>.tmp` and `.unknown-v0-partial` files survived startup, were counted, were excluded from eviction, and prevented initialization even after the only normal entry was evicted. |
| Wrong-size/equal-size cache corruption | **PASS (B4 cleared)** | Independent wrong-size probe repaired 3 bytes to the expected 904 bytes; focused equal-size corruption test restored the indexed payload. |
| Corrupt shard/header/payload and ready-read integrity | **PASS (B4 cleared)** | Corrupt payload test failed with `payload checksum mismatch` before publication. Strict prefetch returns index size/digest metadata; final readiness and `/img` cache read revalidate size plus SHA-256 and invalidate corrupt ready jobs. |
| Write peak, post-publish usage, LRU, active pin | **PASS** | `bytes_needed` reserves pending payload bytes after invalid finals are removed; post-publish usage is asserted. Real 16-asset peak/final use was 2,900,111 bytes under 128 MiB. LRU kept touched A and evicted B; active-pin probe evicted old unpinned content while retaining the active path. |
| Same-cache second owner | **PASS in normal lifecycle; B9 on failed init** | A live owner rejects a second manager and a clean close permits replacement. Failed construction leaks ownership; see B9. |
| Queued/locating close, reconfigure, lifespan, worker lock | **PASS (B5 cleared)** | Close marked active and queued jobs failed, cancelled the queued future, waited for the active call, left no member/thread, and permitted a new owner. Reconfigure blocked until old work exited and then installed a new owner. Lifespan closed its manager. No close/worker-lock deadlock reproduced. |
| Hard NFS read interruption | **PASS with residual risk** | Cooperative checks occur around locate/read/publish boundaries and close waits synchronously. A kernel-blocked `pread` cannot observe cancellation until the hard NFS read returns; retained as an operational residual risk. |
| Rail / pre-ready network behavior | **PASS (B6 cleared)** | Rail uses icon placeholders. Playwright observed zero `/img` requests while preparation remained at 4/10; candidate/detail image rendering stayed absent until `ready`. |
| Archived `/img` cache-only and canonical prepared-ready ownership | **PASS except B7 canonical alias** | Archive miss returned 409 without `ArchiveReader`; only paths in a ready job's verified metadata can be read. Legal local and archived full/thumb responses decoded with expected response/MIME behavior. |
| Path authorization variants | **PASS for security boundary** | Tilde, relative, lexical `..`, double slash, prefix collision, and symlink escape were rejected; NUL returned controlled 400. URL-decoded `%2e` becomes lexical traversal and is rejected by the same normalization check. |
| One canonical string across prepare/cache/catalog/read | **BLOCKER (B7 remains)** | An in-root symlink alias authorizes to its resolved path, but `_prepare_detail` discards that result and submits the unresolved alias; `/img` later looks up the resolved string. |
| GET 404 job loss / retry | **PASS (B8 cleared)** | Playwright retained 3/10, stopped GET polling on 404, exposed Retry, issued exactly one retry POST, then rendered ready candidates. |
| Temporary 503/network recovery / rapid switch / unmount | **PASS (B8 cleared)** | Structured status errors drive bounded exponential retry; a 503 recovered on the second GET. Late old-group failure did not replace the new group, and navigation/unmount stopped polling updates/timers. |
| `.in.jpg` caveat | **PASS** | Asset collection still excludes SFT `I_in`; source uses canonical group `source_path`. No `.in.jpg` is presented as canonical `I_in`. |
| `prefetch()` compatibility and checksum boundary | **PASS** | Result remains a `dict` subclass with original mapping iteration/value behavior; the actual producer prefetch wiring suite passed 3 tests. SHA-256 is performed when trusting an existing cache entry, reading a shard payload, establishing ready state, and serving a verified ready entry. |
| Frontend source/dist/screenshots | **PASS** | Independent Vite build was byte-identical to checked-in dist; all 15 Playwright tests passed. Six screenshots existed, decoded nonblank, and three were visually read; desktop/mobile progress and mobile detail had no incoherent overlap or document-width overflow. |

## Round 2 B1-B8 Disposition

| Finding | Status | Disposition |
|---|---|---|
| B1 source-of-truth startup | **PASS** | Durable NFS default now holds across no-flag, config, env, override, and PG-down behavior; config privacy and DSN redaction passed. |
| B2 eager/full-ledger fallback | **BLOCKER** | Manifest-only discovery is fixed, but build switching and concurrent loads violate the one-build residency/thread-safety contract. |
| B3 cache bound/partials | **BLOCKER** | Correct for current partial names and peak accounting; legacy/unknown dot files permanently wedge the disposable cache. |
| B4 cache/shard checksum | **PASS** | Wrong/equal-size corruption, shard corruption, and ready-read digest paths behave correctly. |
| B5 lifecycle | **PASS** | Close/reconfigure/lifespan behavior is synchronous and no post-return ready/write/thread was observed. Failed-constructor ownership is separately B9. |
| B6 archive read gating | **PASS** | Rail is image-free, pre-ready `/img` count is zero, and archived `/img` is cache-only. |
| B7 authorization | **BLOCKER** | Input rejection is repaired, but canonical string identity is not maintained between prepare and `/img`. |
| B8 frontend recovery | **PASS** | Lost job, transient failure, retry, stale response, and unmount behavior passed. |

## Round 2 Findings

### BLOCKER B2 - Switching builds still overlaps full parsed build state and is not serialized

- Location: `databuild_viewer/backend/repository.py:438-493`.
- Reproduction: load empty build A, patch `_read_jsonl`, then request build B. All three B reads recorded `active_during_switch_read=build-a`; only after B was completely parsed did `_active` switch. The read/parse block is outside `self._lock`. Two concurrent requests can therefore parse two builds simultaneously and race which `_BuildData` becomes active.
- Impact: the full-ledger discovery OOM was fixed, but switching away from a large build temporarily holds old parsed A plus B's rows/derived structures. Concurrent clients can multiply that peak. This directly violates “at most one build resident” and undermines the fallback's bounded-memory claim.
- Minimum fix: serialize `_load` end to end, clear/release the old active object before parsing a different build, and prevent a stale concurrent load from overwriting a newer selection. Add a blocking two-build concurrency test that asserts only one parser enters and old active data is released before the new JSONL read begins.

### BLOCKER B3 - Legacy and unknown dot entries are counted but permanently non-reclaimable

- Location: `databuild_viewer/backend/materialize.py:24`, `materialize.py:316-328`, `materialize.py:330-376`.
- Reproduction: place a 70-byte Round-1-format `.<sha256>.123.tmp`, a 50-byte `.unknown-v0-partial`, and a 20-byte normal entry in a cache capped at 100 bytes. Startup evicted the normal entry, retained both dot files, then failed `cache limit cannot be met: need to free 40 bytes, freed 20`.
- Impact: accounting is fail-closed, so the hard byte bound is not exceeded; however, a crash from the immediately previous implementation can make the dedicated disposable cache permanently unavailable. Because every dot entry is excluded from LRU, retry/restart cannot recover without manual deletion. For this cache's availability contract, that is a blocker rather than an acceptable fail-closed state.
- Minimum fix: recognize and remove the prior `.<hash>.<pid>.tmp` format, and define a safe startup quarantine/deletion policy for all non-owner regular files in the exclusively owned disposable `members/` directory. Keep all files charged until removal. Add current, legacy, and unknown-name recovery tests.

### BLOCKER B7 - Authorization canonicalizes, but prepare submits a different cache/catalog key

- Location: `databuild_viewer/backend/app.py:216-253`, `app.py:293-310`.
- Reproduction: an allowed in-root alias `/allowed/alias/archived.png` resolved to `/allowed/inside/archived.png`. `_prepare_detail` returned the unresolved alias (`canonical_strings_equal=false`), while `/img` converted it to the resolved string before `read_cached`. The security escape tests passed; this is an identity/availability failure.
- Impact: a legal alias can be prepared under one SHA-256 cache key and requested under another, yielding 409 or a strict catalog miss even after successful preparation. Authorization, catalog lookup, cache naming, and read ownership do not use one canonical string as required.
- Minimum fix: canonicalize once and pass the returned canonical strings to materialization/catalog/cache/read, or reject every symlink alias whose supplied string differs from the accepted canonical string. Add an in-root symlink alias integration test that proves prepare and both full/thumb `/img` use the identical logical key.

### BLOCKER B9 - Constructor failure after `flock` leaks the owner lock

- Location: `databuild_viewer/backend/materialize.py:124-136`.
- Reproduction: the B3 capacity failure occurred after `_owner_fd` acquired `LOCK_EX`. After manually removing the offending files and forcing GC, a second manager in the same process still failed `materialization cache already has an owner`.
- Impact: one recoverable startup validation/capacity error wedges that cache root for the lifetime of the server process. Reconfigure cannot recover even after an operator fixes the cache contents, and the leaked descriptor accumulates across retries.
- Minimum fix: wrap every post-open/post-flock initialization step, including cleanup, capacity enforcement, and executor construction, in exception cleanup that unlocks and closes `_owner_fd`. Add a test that forces constructor failure, repairs the cause, and immediately acquires the same cache in the same process.

### BLOCKER B10 - Duplicate manifest build IDs silently select one ledger

- Location: `databuild_viewer/backend/repository.py:405-421`.
- Reproduction: two distinct directories each declared `build_id=duplicate`; `builds()` returned one row and selected the lexicographically first artifact root without an error or health warning.
- Impact: `--build-root` is repeatable and `DBV_BUILD_ROOTS` supports multiple roots. A duplicated canonical ID can silently expose the wrong snapshot and hide the other ledger, defeating build-scoped inspection and fail-closed authority selection.
- Minimum fix: detect duplicate IDs across distinct resolved directories during manifest refresh and raise a sanitized `StoreUnavailable`/health error naming only the duplicate ID and roots. Add duplicate-ID tests, while continuing to deduplicate the same resolved directory discovered through overlapping roots.

### NIT N1 - Bounded FIFO still makes the newest selection wait behind abandoned work

- Location: `databuild_viewer/backend/materialize.py:22-23`, `materialize.py:177-184`, `materialize.py:378-385`.
- Evidence: queue and retained-job maps are now bounded, clearing the unbounded-memory half of Round 1 N1. Work remains one-worker FIFO, and frontend navigation does not cancel or reprioritize stale selected groups, so the current selection can wait behind up to seven abandoned jobs.
- Minimum improvement: coalesce/cancel queued jobs no longer selected or explicitly prioritize the latest interactive request while preserving the one-active-NFS-reader constraint and multi-client fairness.

### NIT N2 - Local-file authorization/open remains TOCTOU-sensitive

- Location: `databuild_viewer/backend/app.py:216-243`, `app.py:297-306`.
- Evidence: malformed NUL is now controlled and traversal/symlink escape checks pass. Local authorization still resolves/stats a path separately from `FileResponse` or thumbnail open, so a writable trusted tree can be replaced between validation and open.
- Minimum improvement: serve from an opened no-follow descriptor with post-open containment/stat validation, or document the same-user immutable-local-tree trust requirement.

### NIT N3 - Tests and operator notes still overstate two blocker properties

- Location: `databuild_viewer/backend/test_app.py:327-353`, `backend/test_materialize.py:153-173`; `databuild_viewer/README.md:37-40,62-64`; `databuild_viewer/NOTES.md:87-102`.
- Evidence: tests prove manifest-only discovery and current UUID partial cleanup, but do not switch while observing old residency, race concurrent loads, cover duplicate build IDs, cover legacy/unknown partial names, or test failed-constructor lock release. README/NOTES consequently claim one-build residency and startup partial recovery more broadly than behavior supports.
- Minimum improvement: add the reproductions above and narrow the documentation until they pass.

## Round 2 Validation Commands

| Command / probe | Result |
|---|---|
| `uv run --no-project --no-progress --with fastapi==0.136.1 --with Pillow==12.2.0 python -m unittest databuild_viewer.backend.test_app databuild_viewer.backend.test_materialize dataset_build.tests.test_prefetch_and_sft_pack -v` with temp `DBV_CACHE_ROOT` | **PASS**, 50 tests |
| `PYTHONPATH=.../dataset_build/src /home/bc/.venvs/iaa437/bin/python -m unittest dataset_build.tests.test_land_integration.PrefetchWiringTests -v` | **PASS**, 3 production-caller compatibility tests |
| `npm --prefix databuild_viewer/frontend run build -- --outDir /tmp/dbv-round2-dist.* --emptyOutDir` + `diff -rq` + SHA-256 | **PASS**, 1,580 modules; source build and checked-in dist byte-identical |
| `npm --prefix databuild_viewer/frontend run test:e2e` | **PASS**, 15 Playwright tests |
| `git diff --check -- databuild_viewer dataset_build/tools/prefetch.py dataset_build/tests/test_prefetch_and_sft_pack.py docs/reviews/REVIEW-impl-dbv-nfs-1.md` | **PASS** after the Round 2 review edit; the untracked review file also has no trailing whitespace |
| Real full-ledger manifest-only `JsonlStore.builds()/health()` with `_read_jsonl` forbidden and `/usr/bin/time -v` | **PASS**, 34 builds; 2,606,415 manifest bytes; zero JSONL calls; 21,372 KiB max RSS; 0 storage `read_bytes` delta |
| Direct full-ledger `api_health()` / `api_builds()` with `_read_jsonl` forbidden and temp cache | **PASS**, 34/34 builds; zero JSONL calls; no active build |
| Actual config/default/env/override/private-mode matrix using a temporary `0600`/`0644` example config | **PASS**, durable default and env roots correct; config DSN retained but not printed; non-private config rejected |
| Forced PG-down mini ledger fallback with a credential-bearing failure | **PASS**, sanitized JSONL fallback, 30 groups, explicit build-scoped page |
| `python -m databuild_viewer.backend.app --nfs-ledger .../mini30-v52-20260730 --catalog /var/cache/veradata/global.sqlite3 --selfcheck` with temp cache | **PASS**, 1 build, 30 groups, 0 malformed |
| Real NFS `MaterializationManager` cold prepare + close/reopen resume + SHA-256 ready reads + Pillow decode | **PASS**, 16/16 decoded; cold 0.151 s; resume 0.050 s; zero resume writes; 2,900,111 bytes peak/final under 128 MiB; no worker after close |
| Real `ArchiveReader(..., verify_checksum=True)` indexed read | **PASS**, 739,925-byte source at shard offset 1,816,606,208 in 0.0056 s |
| Wrong-size/equal-size cache and corrupt-shard payload probes | **PASS**, wrong-size repaired; focused equal-size and corrupt-payload tests passed |
| Legacy/unknown stale partial + failed-constructor reacquire probe | **FAIL**, reproduced B3 and B9 |
| Build switch residency + duplicate manifest probes | **FAIL**, reproduced B2 and B10 |
| Queued/locating close, reconfigure, lifespan, LRU, active pin probes | **PASS**, synchronous close/reconfigure, no deadlock/thread, correct eviction/pin |
| Path matrix plus local/archive full/thumb decode | **PASS** for authorization/serving; **FAIL** canonical-string equality, reproduced B7 |
| Screenshot existence, Pillow pixel decode/statistics, and direct visual reads | **PASS**, six nonblank screenshots; reviewed desktop materialized, mobile materializing, and mobile mask/detail |

Two preliminary compatibility invocations failed before tests ran because the stripped environment first lacked the `construct` source path and then NumPy. The same test target passed under the project's actual `iaa437` environment. The first direct ArchiveReader reporting probe also used attribute access on its dict result and was corrected; the corrected checksum read passed. These were reviewer-harness errors, not implementation failures.

## Round 2 Confirmed Correct Behavior

- Durable NFS is now the source-of-truth fallback for all normal startup modes; tmpfs `output_root` is not reused as ledger authority.
- Full-ledger health/build discovery is genuinely manifest-only and bounded in the measured environment.
- Checksums are carried from the catalog through prefetch, publish, ready validation, resume trust, and `/img` serving. Corruption no longer becomes `ready`.
- Current-format partials, hard capacity, peak reservation, post-publish assertion, LRU, active pins, and normal one-owner lifecycle behave coherently.
- Close/reconfigure/lifespan wait for worker completion and do not release cache ownership early. Closed jobs did not transition back to ready and no post-return writes/threads were observed.
- Rail rows cannot cause archive reads; archived `/img` cannot bypass the verified ready cache.
- Frontend job-loss and transient-error recovery are status-aware, bounded, retryable, and resistant to stale group/unmount updates.
- `prefetch()` remains compatible with its production caller and `.in.jpg` remains excluded from canonical source display/materialization.

## Round 2 Residual Risks

- Hard-mounted NFS can block inside `os.pread`; cooperative cancellation and `shutdown(wait=True)` cannot finish until the kernel call returns. No bounded shutdown latency can be claimed.
- The 19,629,056,000-byte immutable global SQLite catalog is not copied and one worker limits normal concurrent access, but cold mmap/page-fault latency remains deployment-dependent.
- JSONL signatures still rely on `(mtime_ns, size)` and assume atomic canonical publication. In-place rewrites preserving both values can evade refresh.
- Cache coordination is one process per root. Multi-worker deployment must use distinct roots or retain the cross-process ownership contract; otherwise additional workers fail startup by design.
- Checksum verification adds full-payload reads at trust/serve boundaries. The 16-asset mini smoke was fast, but cold large-group and concurrent-client latency were not benchmarked.
- Local ordinary files remain outside the extraction-cache byte budget and rely on the trusted local-tree assumption described in N2.

---

## Round 1 History

The complete first-round review is preserved below as historical evidence. Its verdict and counts are superseded by the Round 2 verdict at the top of this file.

<details>
<summary>Round 1 independent review (2026-08-04)</summary>

## Review

- Correct: explicit `--config ... --nfs-ledger /mnt/nfs/bc/data/builds ...` wiring keeps the PostgreSQL projection preferred and uses the NFS ledger as JSONL fallback; the small real-NFS selfcheck passed for 30 groups.
- Correct: archive resolution is index-first and ordered by `(root, shard, offset_data)`; tar header name/size validation, strict missing-path rejection, per-member atomic `os.replace`, progress retention, stale-group suppression, and `.in.jpg` non-use are implemented coherently.
- Fixed: none. This was a review-only task; no implementation file was modified.
- Blocker: 8 findings remain. Cache bounding/integrity, process lifecycle, source-of-truth defaults, full-ledger fallback, archived image gating, authorization, and restart recovery are not ready for merge.
- Note: 43 focused Python tests, the production frontend build, 11 Playwright tests, source/dist comparison, `git diff --check`, and a checksum-verifying real-NFS read all passed. The passing tests do not exercise several blocker paths listed below.

## Metadata

| Field | Value |
|---|---|
| Date | 2026-08-04 |
| Review type | Independent implementation review; worker handoff not trusted |
| Reviewer runtime | Available reviewer model. No Opus entry was available at runtime, so the requested Opus reviewer restriction could not be met; this limitation is recorded here as required. |
| Verdict | **BLOCKED** |
| Findings | **8 BLOCKER, 3 NIT** |
| Git action | No commit; no staging |

## Scope

Reviewed only the actual worktree state in:

- `databuild_viewer/**`, including untracked `backend/materialize.py`, `backend/test_materialize.py`, `NOTES.md`, and rebuilt `frontend/dist` assets;
- `dataset_build/tools/prefetch.py`;
- `dataset_build/tests/test_prefetch_and_sft_pack.py`.

All other dirty-worktree changes were excluded. The only review output written is this file.

## Authoritative Contracts

- `CLAUDE.md`: data discipline, implementation-review gate, indexed storage expectations, and no experiment execution before blockers clear.
- `/home/bc/.pi/agent/skills/build-indexed-datasets/SKILL.md`, read in full: direct indexed seek for random reads, bounded ordinary-path cache, immutable published shards, checksum-bearing index, and no extracted hot tree.
- `docs/archive/DATABUILD_IMPLEMENTATION_2026-07-27.md` section 1 and stages D/E: canonical JSONL ledger, PG as projection only, historical staging paths as logical keys, and authoritative `groups/<build_id>` shards.
- `docs/archive/DATABUILD_IO_REFACTOR_2026-07-27.md` section 4 step 4 and section 6: physical member order, uncompressed USTAR, durable indexes, `groups` authority, and `sft` derivation.
- `docs/DATA_ASSIGNMENT_2026-08-02.md` section 1, especially line 35: `.in.jpg` is a VLM preview and must not be presented as canonical `I_in`.
- `databuild_viewer/NOTES.md`: read only as a verification checklist, not accepted as correctness evidence.
- Official semantics opened online: Python `Executor.shutdown` and `os.replace` at `docs.python.org`, and React `useEffect` cleanup at `react.dev`. Python documents that `wait=False` does not cancel running futures and the interpreter still waits for pending work; React documents cleanup before changed-dependency setup and after unmount.

## Validation Commands

| Command / probe | Result |
|---|---|
| `uv run --no-project --no-progress --with fastapi==0.136.1 --with Pillow==12.2.0 python -m unittest databuild_viewer.backend.test_app databuild_viewer.backend.test_materialize dataset_build.tests.test_prefetch_and_sft_pack -v` | **PASS**, 43 tests |
| `npm --prefix databuild_viewer/frontend run build -- --outDir /tmp/dbv-nfs-1-review-dist --emptyOutDir` | **PASS**, 1,580 modules transformed |
| `diff -rq /tmp/dbv-nfs-1-review-dist databuild_viewer/frontend/dist` plus SHA-256 comparison | **PASS**, source build and checked-in dist byte-identical |
| `npm --prefix databuild_viewer/frontend run test:e2e` | **PASS**, 11 Playwright tests |
| `git diff --check -- databuild_viewer dataset_build/tools/prefetch.py dataset_build/tests/test_prefetch_and_sft_pack.py` | **PASS** |
| NFS CLI: `python -m databuild_viewer.backend.app --nfs-ledger .../mini30-v52-20260730 --catalog /var/cache/veradata/global.sqlite3 --cache-root /tmp/... --selfcheck` under the same `uv run` environment | **PASS**, JSONL source, 1 build, 30 groups, 0 malformed |
| Real NFS indexed read, `ArchiveReader(..., verify_checksum=True)` on one selected mini30 candidate | **PASS**, `groups/mini30-v52-20260730/batch-0000`, `shard-00000`, offset 106,297,856, 242,273 bytes; locate 0.0023 s, read+SHA-256 0.0757 s |
| Read-only NFS ledger stat inventory | 34 builds; canonical files total 4,320,164,225 bytes: groups 3,536,960,299; SFT 603,026,662; failures 177,570,849; manifests 2,606,415 |
| Stale-partial capacity probe | **REPRODUCED**: configured max 100, reported usage 90, actual directory usage 240; the 150-byte partial survived capacity enforcement |
| Equal-size corruption probe using the real prefetch fixture | **REPRODUCED**: a 904-byte cache entry replaced by 904 corrupt bytes remained corrupt after strict prefetch; original was not restored |
| Executor close probe | **REPRODUCED**: one worker thread remained alive after `close()`; it wrote a final member and changed the job to `ready` after close |
| Authorization probe | **REPRODUCED**: `~/data/image.jpg` was accepted under allowed `/home/bc/data`; an embedded NUL raised uncaught `ValueError` |
| Three worker screenshots | **PASS for existence/visual inspection**: desktop 1440x900, mobile progress 390x844, mobile error 390x844; progress UI is contained and readable, with no document-width overflow |

The first frontend invocations were accidentally run from the repository root and failed with `Missing script`; the corrected `npm --prefix databuild_viewer/frontend ...` commands above passed. No project file was changed by either invocation.

## Contract Matrix

| Area | Status | Evidence / conclusion |
|---|---|---|
| NFS source of truth and PG fallback wiring | **BLOCKER** | Explicit CLI wiring works, but default/no-flag and `DBV_CONFIG` paths still select the retired local/tmpfs output root; see B1. |
| Full-ledger JSONL fallback | **BLOCKER** | The fallback eagerly retains every parsed record from 4.32 GB of source JSONL; see B2. |
| Index-first lookup and physical order | **PASS** | `prefetch.py:39-47,50-70,210-229` performs one indexed locate and ordered per-shard reads; the real NFS probe resolved the expected `groups` batch. |
| USTAR/header and strict missing path | **PASS** | `prefetch.py:96-110,176-182` checks tar header/member/size and rejects unknown retired paths before materializing any located row in strict mode. |
| Atomic publication | **PASS with blocker caveats** | `prefetch.py:113-123` uses same-directory temporary files and `os.replace`; capacity/lifecycle handling around partials is not correct (B3/B5). |
| Progress monotonicity and terminal state | **BLOCKER** | Normal counters are monotonic and exceptions become `failed`, but size-only readiness can report corrupt bytes as `ready`; see B4. |
| Cache bound, LRU, concurrency, retry, reconfigure, exit | **BLOCKER** | Single-live-manager reads are serialized and completed-file mtime LRU works, but stale partials exceed the bound and closed managers continue running; see B3/B5. |
| `/img` fail-closed authorization | **BLOCKER** | Root/suffix/prefix-collision/`..`/symlink resolution generally pass before catalog lookup, but tilde paths violate the absolute-path contract; see B7. |
| `/img` full/thumb and archive miss | **BLOCKER** | Decoding and MIME behavior work, but archive cache misses directly read shards and let the rail bypass selected-group preparation; see B6. |
| Prepare API asset ownership/dedup/errors | **PASS** | `app.py:227-251` fetches an exposed canonical group, collects only source/candidate/C_GT/SFT target assets, authorizes before submit, deduplicates, returns 404/403/415/503 appropriately, and does not include SFT `.in.jpg`. |
| Task exceptions and retry | **PASS with lifecycle blocker** | `_run` catches ordinary exceptions and publishes `failed`; explicit retry is required. Executor shutdown and queued-job ownership remain wrong (B5). |
| Frontend retained polling/stale group/unmount | **PASS** | `Inspector.jsx:278-316` retains the last snapshot, clears timers, and ignores late old-group results; Playwright group-switch and mobile containment cases pass. |
| Frontend failure recovery | **BLOCKER** | POST-declared failure is retryable, but GET 404 after backend restart loops forever without Retry; see B8. |
| Candidate reads before ready | **PASS** | Candidate tiles/detail images are not rendered until prepare says `ready`; Playwright verifies the candidate strip is absent during progress. The group rail source bypass is separate B6. |
| `.in.jpg` caveat | **PASS** | `collect_asset_paths` deliberately ignores SFT `I_in`; UI displays `group.source_path` as “Before” and never labels `.in.jpg` canonical `I_in`. |
| Tests, dist, README/NOTES, credentials | **NIT** | Tests/build/dist pass and no credential is present, but mocks hide blockers and NOTES makes two false lifecycle/default claims; see N3. |
| Performance: 19.6 GB catalog | **PASS as residual risk** | Catalog is opened immutable and not copied; one materializer worker bounds concurrent catalog connections. Repeated large-catalog opens/mmap remain an operational risk, not a demonstrated correctness failure. |
| Performance: 50 rail thumbnails / one-worker queue | **BLOCKER + NIT** | Rail archive reads are B6; stale selections and an unbounded one-worker queue are N1. |

## Findings

### BLOCKER B1 - NFS is opt-in; existing/default startup still falls back to retired local or tmpfs roots

- Location: `databuild_viewer/backend/app.py:31-33`, `app.py:67-74`, `app.py:399-413`; contradictory checklist statement at `databuild_viewer/NOTES.md:78-79`.
- Evidence: `DEFAULT_NFS_LEDGER_ROOT` is only the optional `--nfs-ledger` flag's `const`. With no flag, `_build_roots` remains `/home/bc/data/datasets/vera_directionA_1M/builds`. With `DBV_CONFIG` or `--config` but no `--nfs-ledger`, `output_root` is used, which the authoritative D-stage contract keeps on `/mnt/ramstage`. The explicit combined CLI was independently proven to select NFS and retain a non-null PG DSN, but existing startup behavior was not repaired by default.
- Impact: after PG failure, an operator using the old documented config startup can silently see an empty/retired tmpfs projection instead of the canonical NFS ledger. This violates source-of-truth behavior.
- Minimum fix: make `/mnt/nfs/bc/data/builds` the default JSONL fallback whenever no explicit `--build-root`/root override is supplied, including the `DBV_CONFIG` path, while keeping the config's PG DSN preferred. Add startup tests for no flags, config-only, config+override, env-only, and PG-down fallback roots. Correct NOTES.

### BLOCKER B2 - Whole-NFS JSONL fallback is an eager multi-gigabyte in-memory load

- Location: `databuild_viewer/backend/repository.py:331-350`, `repository.py:401-455`, `repository.py:484-505`, `repository.py:915-928`.
- Evidence: `_read_jsonl` builds a Python list for each complete file; `_refresh` then retains all canonical groups, SFT rows, failures, and nested candidate payloads for every discovered build. A read-only stat of the current 34-build ledger measured 4,320,164,225 raw bytes. Parsed Python objects add substantial overhead. Any PG query failure immediately enters this path.
- Impact: `--nfs-ledger` without a build value and the required PG fallback can incur multi-gigabyte NFS scans, long request stalls, and likely process OOM. Thus the advertised whole-ledger fallback is not operationally credible.
- Minimum fix: build/use a bounded on-disk ledger projection with paged queries and incremental per-build signatures, or require a bounded build selection for JSONL fallback and return an explicit service error rather than loading the entire root. PG fallback must remain authoritative, observable, and bounded.

### BLOCKER B3 - Cache accounting excludes stale partials and does not reserve replacement peak space

- Location: `databuild_viewer/backend/materialize.py:238-255`, `materialize.py:265-298`; `dataset_build/tools/prefetch.py:113-123`, `prefetch.py:188-195`.
- Evidence: `_usage` excludes every dot-prefixed file. `_discard_partials` only removes partials for the current failed job, so process-killed/orphan partials survive indefinitely. Reproduction with max=100 reported 90 bytes while 240 bytes were actually present, and `_ensure_capacity` left the 150-byte partial untouched. During replacement, `bytes_needed=max(0,new-old)` budgets the final delta while `_publish` temporarily holds both old final and full new partial, so instantaneous use can also exceed the configured cap.
- Impact: the approved 32 GiB limit is not a hard bound across crashes, retries, or replacement; disk exhaustion remains possible and health reporting is false.
- Minimum fix: serialize cache ownership, clean or quarantine stale partials on startup, count all regular cache bytes, reserve the full temporary-write peak, use collision-resistant per-attempt partial names, and assert actual post-publish usage. Add crash-leftover and wrong-size replacement tests.

### BLOCKER B4 - Payload checksum is ignored; equal-size corruption is accepted and reported ready

- Location: `dataset_build/tools/prefetch.py:39-47`, `prefetch.py:96-110`, `prefetch.py:188-195`, `prefetch.py:211-243`; `databuild_viewer/backend/materialize.py:205-212`, `materialize.py:222-235`.
- Evidence: locate SQL does not select `m.sha256`; `_read_member` validates only tar header name/size; existing cache entries are hits solely when `stat().st_size == size`; `_all_available` requires only nonzero size. In the repository's real prefetch fixture, replacing a valid 904-byte cache member with 904 corrupt bytes and rerunning strict prefetch retained the corrupt bytes and did not restore the original.
- Impact: disk corruption, stale same-size files, or shard payload drift can become terminal `ready` and be served indefinitely. The index's required checksum is unused precisely where durable cache trust is established.
- Minimum fix: select `sha256`, hash bytes before atomic publish, validate equal-size hits before touching LRU time, delete/refetch mismatches, and have final readiness validate expected size+digest. Add cache-corruption and shard-corruption tests.

### BLOCKER B5 - `close()` does not stop running work; reconfigure/exit can write after ownership transfer

- Location: `databuild_viewer/backend/materialize.py:100-110`, `materialize.py:118-132`, `materialize.py:188-220`; `databuild_viewer/backend/app.py:120-130`. The contrary claim appears at `databuild_viewer/NOTES.md:55-59`.
- Evidence: `shutdown(wait=False, cancel_futures=True)` cancels only not-started futures. `_run` has no cancellation check tied to `_closed`, and the app has no lifespan shutdown hook. The independent probe observed one live worker after `close()`; after release, it wrote a final cache member and changed the old job to `ready`. Official Python documentation confirms running futures are not cancelled and the interpreter does not exit until pending futures complete.
- Impact: reconfiguration can leave old/new managers racing on one cache, close can publish after ownership transfer, and server shutdown can wait on a blocked NFS read. This also undermines B3's capacity serialization.
- Minimum fix: add app lifespan ownership, cooperative cancellation checked before capacity/write/final publication, cancel queued work, wait for the active worker to leave its critical section, and prevent a closed job from reaching `ready`. Use a per-cache process lock or explicitly reject simultaneous managers. Add reconfigure and graceful-exit tests.

### BLOCKER B6 - Group rail and `/img` bypass selected-group materialization with direct archive reads

- Location: `databuild_viewer/frontend/src/Inspector.jsx:230-270`; `frontend/src/lib/ui.jsx:64-75`; `databuild_viewer/backend/app.py:270-301`; masking test behavior at `frontend/e2e/viewer.spec.js:176-183`.
- Evidence: every group row renders `AssetImage(group.source_path)`, generating `/img` requests before selection/prepare. On cache miss, `/img` calls `read_bytes` directly, which performs a random indexed archive read and does not materialize into the bounded viewer cache. The E2E suite intercepts every `/img` and returns a fixture immediately, so it cannot detect this behavior. The same backend fallback permits any authorized archived image request to bypass prepare.
- Impact: a page of 50 groups can cause many random source-shard reads, contrary to “selected group only”; ready is not an enforced backend boundary; these reads bypass cache capacity/LRU/integrity controls.
- Minimum fix: make archived `/img` cache-only for assets belonging to a prepared/ready canonical group, and remove archived rail thumbnails or provide a separately indexed, explicitly bounded thumbnail cache/batch. Add a network assertion that zero archived candidate/source member reads happen before selection/ready and that rail rendering cannot trigger direct NFS reads.

### BLOCKER B7 - The absolute-path authorization contract accepts tilde-relative inputs

- Location: `databuild_viewer/backend/app.py:201-217`.
- Evidence: the function calls `Path(value).expanduser()` before `is_absolute()`. With allowed root `/home/bc/data`, `~/data/image.jpg` was accepted and normalized to `/home/bc/data/image.jpg`, despite the contract requiring the supplied historical logical path itself to be absolute. Prefix collisions and explicit `..` traversal were correctly denied.
- Impact: authorization does not exactly implement the fail-closed input contract and aliases can diverge from canonical catalog/cache keys.
- Minimum fix: reject unless `Path(value).is_absolute()` before expansion/normalization; canonicalize exactly once and use that same string for authorization, cache key, and catalog lookup. Add tilde, relative, UNC/double-slash, symlink-parent, prefix-collision, and encoded traversal tests.

### BLOCKER B8 - A lost backend job leaves retained frontend progress unrecoverable

- Location: `databuild_viewer/frontend/src/api.js:1-13`; `frontend/src/Inspector.jsx:287-307`, `Inspector.jsx:321-358`.
- Evidence: if POST succeeds and the backend restarts, in-memory job status is lost and GET returns 404. The catch retains `last`; because a running snapshot exists, `failed` is false, no Retry button appears, and the hook schedules the same GET every 500 ms forever. `requestJson` collapses HTTP status into a plain string, preventing status-aware recovery. Existing Playwright covers a POST response whose state is already `failed`, not a GET 404 after a retained running snapshot.
- Impact: a routine backend restart produces a permanently stuck `queued/materializing` UI that cannot re-POST without navigation/reload.
- Minimum fix: preserve structured HTTP status; treat GET 404 as “job lost” and expose/reissue an explicit retry POST, while retaining snapshot and backoff for transient network/5xx errors. Add 404-after-progress, temporary 503 recovery, retry, and unmount tests.

### NIT N1 - One FIFO worker and an unbounded retained job map penalize the newest selection

- Location: `databuild_viewer/backend/materialize.py:100-103`, `materialize.py:118-137`.
- Evidence: every newly selected group submits a future to a single FIFO executor; stale selections are not cancelled or deprioritized, and `_jobs` is never pruned. Rapid navigation or multiple clients can make the currently viewed group wait behind abandoned groups and grow memory/future state without bound.
- Minimum improvement: cap outstanding jobs and retained terminal snapshots, prioritize or coalesce the latest interactive request, expose queue position, and define multi-client fairness. Keeping one active NFS reader is reasonable; keeping an unbounded stale FIFO is not.

### NIT N2 - Malformed-path handling and trusted-local TOCTOU should be hardened

- Location: `databuild_viewer/backend/app.py:201-224`, `app.py:274-283`.
- Evidence: an embedded NUL raises uncaught `ValueError` rather than a controlled 4xx. Local authorization and later `FileResponse`/open are separate path operations, leaving a replace-with-symlink race if the allowed local tree is writable by another principal.
- Minimum improvement: catch `ValueError`, return a stable 4xx, and serve from a descriptor opened with no-follow semantics plus post-open containment/stat validation. This is lower severity under the current same-user/trusted-local deployment, but should be documented if retained.

### NIT N3 - Tests and NOTES overstate coverage/lifecycle behavior

- Location: `databuild_viewer/backend/test_materialize.py:77-163`; `backend/test_app.py:477-549`; `frontend/e2e/viewer.spec.js:176-183,355-399`; `databuild_viewer/NOTES.md:55-59,78-79,86-88`.
- Evidence: materializer tests use a fake writer and do not cover stale partials, equal-size corruption, close/reconfigure, process exit, wrong-size peak usage, ready-entry eviction across clients, or local-existing TOCTOU. API tests mock archive reads; Playwright universally fulfills `/img`. NOTES incorrectly says the ledger defaults to NFS and that the executor is shut down at process teardown.
- Minimum improvement: correct NOTES and add integration tests that use the real prefetch/cache path and assert disk usage, digest integrity, lifecycle completion, and zero pre-ready archive reads.

## Confirmed Correct Behavior

- Explicit CLI source wiring: the combined config+NFS probe selected `/mnt/nfs/bc/data/builds` while retaining PG configuration; a real NFS small-build selfcheck passed.
- PG fallback semantics at the repository boundary remain fail-closed with sanitized `postgres_unavailable`; no DSN secret is exposed.
- `collect_asset_paths` restricts prepare inputs to the repository's canonical group detail and deduplicates paths. It ignores SFT `I_in`, satisfying the `.in.jpg` caveat.
- `_locate` uses a temporary keyed table, avoids SQLite host-parameter limits, and returns archive physical order. One shard descriptor is advanced through ordered rows.
- Strict unknown retired paths fail before materialization. Header mismatch/truncation becomes `failed`, not `ready`.
- Normal progress file/byte counters are real and monotonic in focused tests; local-existing assets start as completed; terminal exceptions are surfaced with a message.
- Within one live manager, `read_cached` and eviction share a lock, completed-member LRU mtime is touched on read/hit, and atomic replacement prevents readers seeing a normal in-process partial.
- Frontend polling retains the last snapshot during refresh, clears timers on dependency change/unmount, rejects late old-group results, gates candidate rendering until `ready`, and presents a usable retry for explicit failed snapshots.
- The three requested screenshots exist and are visually coherent. Mobile progress is contained; desktop candidate overflow is intentional horizontal scrolling, confirmed by CSS and Playwright.
- Frontend `dist` is byte-identical to an independent production build. No credential or secret value was found in README/NOTES/source.

## Residual Risks

- The global catalog is 19,629,056,000 bytes. Immutable SQLite plus a single active worker avoids copying it and limits connection concurrency, but cold page faults/repeated opens should be monitored on the deployment host.
- One selected mini30 group required 16 unique displayed assets; a checksum-verifying member read took 0.0757 s in the smoke environment. End-to-end cold-group latency and concurrent-client latency were not benchmarked because the review was restricted to read-only small-NFS smoke.
- Cache coordination is process-local. Even after B3/B5, multi-worker/multi-process deployment needs an explicit cache ownership/locking contract or separate cache roots.
- Local-existing assets are intentionally outside the viewer cache byte budget. Their storage lifecycle and trust boundary must remain separate from the bounded extraction-cache claim.
- The JSONL signature uses `(mtime_ns, size)`. An in-place rewrite preserving both could retain a stale snapshot; canonical atomic ledger publication makes this unlikely, but the assumption should be stated.

## Verdict

**BLOCKED.** DBV-NFS-1 must not proceed while any blocker remains. The implementation demonstrates the intended index-first path and a sound frontend progress presentation, but it does not yet enforce the approved source-of-truth, bounded-cache, integrity, lifecycle, authorization, and recovery contracts end to end.

</details>
