# Databuild Canonical Refactor

- Status: approved design, ready for implementation
- Date: 2026-07-20
- Scope: `dataset_build`, its local GPU renderer integration, annotation transport,
  and `databuild_viewer`

This document is the implementation authority for consolidating the repository's
databuild code into one production path. If existing comments, runbooks, scripts,
or historical schemas conflict with this document, this document wins. Low-level
implementation details may change when tests expose a better approach, but the
behavioral contracts and deletion boundaries below must not change silently.

## 1. Outcome

The repository will expose exactly one production command:

```bash
python -m construct.agent run --config /absolute/path/to/databuild.toml
```

That command owns the complete lifecycle:

```text
SAM3-ready sources
  -> scene-stratified source allocation
  -> configurable local/global split
  -> taxonomy-aware preset selection
  -> local GPU rendering of eight candidates
  -> before/after visibility gate
  -> OneAlign ranking and top-2 selection
  -> durable OpenAI Responses annotation queue
  -> SFT output and viewer projection
```

There will be no alternate producer, manual backfill path, DPO path, degradation
track, farm fallback, CPU operator fallback, or Chat Completions annotation path.

## 2. Goals And Non-Goals

### Goals

- One TOML-driven orchestrator for global and SAM3-aware local samples.
- Only `.xmp`, `.lrtemplate`, and LUT presets that the local GPU stack can render.
- Deterministic, resumable, balanced traversal of taxonomy major/minor categories.
- One instance-level SAM3 subject protocol and one fixed eight-slot mask protocol.
- A measurable before/after visibility gate before aesthetic ranking.
- One OpenAI Responses client abstraction for one or more external key lanes and local vLLM.
- SFT-only output with durable queues, idempotent resume, and auditable failures.
- A first-class viewer for inspecting sources, all eight candidates, masks, QA,
  and final annotations.
- Physical removal of obsolete databuild code after the canonical path is wired.

### Non-goals

- Do not delete Git branches, Git history, existing dataset directories, or
  historical PostgreSQL rows.
- Do not delete the shared farm implementation from `core/render_backend.py` if it
  remains useful outside databuild. Databuild must simply be unable to call it.
- Do not redesign OneAlign, the LUT math, the SAM3 instance selector, or the model
  training/export format beyond changes required by this contract.
- Do not reintroduce source QA as a source admission gate.
- Do not migrate the local model beyond the existing `Qwen3.5-35B-A3B` deployment.

## 3. Canonical Configuration

### 3.1 Source Of Truth

- The CLI accepts only `--config`; behavior-affecting environment overrides are
  removed from the production path.
- The real TOML contains secrets and credentials, including relay API keys and
  potentially credential-bearing PostgreSQL DSNs. It must be ignored by Git, must
  be checked for owner-only permissions (`0600`), and must never be dumped into
  logs, exceptions, JSONL, other persistent artifacts, PostgreSQL, or the manifest.
  Redaction must cover URI userinfo, credential query parameters, tokens, and keys
  rather than matching only the `api_key` field name.
- Commit a complete `databuild.example.toml` containing placeholders.
- The top-level `schema_version` is required and initially equals `1`.
- Missing required fields, unknown fields, invalid enum values, an empty external endpoint
  lane list, duplicate lane IDs, invalid paths, and ratios
  that do not sum to one are startup errors.

### 3.2 Recommended Schema

The exact Python types and loader layout are implementation details, but the TOML
must cover this semantic surface:

```toml
schema_version = 1
build_id = "example-build"
seed = 0
target_groups = 10000
output_root = "/home/bc/data/datasets/vera_directionA_1M/builds/example-build"
preset_filter = "all" # xmp | lrtemplate | lut | all

[mix]
local = 0.70
global = 0.30

[sources]
subject_cache = "/home/bc/data/datasets/vera_directionA_1M/subject_cache"
postgres_dsn = "postgresql://..."

[presets]
bank_dir = "/home/bc/data/datasets/vera_directionA_1M/preset_bank_full"
taxonomy = "/home/bc/data/datasets/vera_directionA_1M/preset_bank_full/taxonomy.jsonl"
fidelity_de_max = 6.0
disabled_formats = [] # any of: xmp, lrtemplate, lut

[render]
short_edge = 1024
jpeg_quality = 95
gpu_concurrency = 2
diff_short_edge = 512
visible_de_min = 2.5
visible_fraction_de = 2.3
visible_fraction_min = 0.50

[masks]
linear_target_alpha_mass = 0.50
sam3_relabel_attempts = 2

[annotation]
external_model = "configured-by-operator"
image_long_edge = 768
image_jpeg_quality = 90
external_reasoning_effort = "low"
external_max_output_tokens = 6000
transport_attempts_per_round = 4
queue_rounds = 3
# true = exhaust both external quotas before local vGate; false = external-only.
local_fallback = true

[[annotation.external_endpoints]]
id = "relay-a"
base_url = "https://relay-a.example/v1"
api_key = "secret"
concurrency = 16

[[annotation.external_endpoints]]
id = "relay-b"
base_url = "https://relay-b.example/v1"
api_key = "secret"
concurrency = 16

[annotation.local]
base_url = "http://127.0.0.1:8003/v1"
api_key = "EMPTY"
model = "qwen3_5-35b-a3b"
temperature = 0.2
enable_thinking = false
max_output_tokens = 2048

[viewer]
postgres_dsn = "postgresql://..."
```

The committed example must document every field. Defaults may exist in the example,
but production must not acquire hidden behavior from old YAML files or environment
variables.

## 4. Source Eligibility And Allocation

### 4.1 Eligibility

A source is eligible when all of the following are true:

1. `<subject_cache>/<path_key>/subject.json` exists and has `status == "ready"`.
2. `subject.png` exists, is readable, is non-empty, and passes the existing
   instance-mask area and integrity guards.
3. The source image exists and can be decoded.

Source QA verdicts do not filter the pool. They may remain as viewer/audit metadata.
Both global and local groups draw from this same SAM3-ready pool.

Only the instance-level SAM3 protocol remains:

```text
subject.json + subject.png
```

Concept-level masks, `regions.json`, `sam3_masks`, multi-concept unions, and live
fallback segmentation are not part of the canonical path.

### 4.2 Allocation

- Retain the existing scene-stratification behavior, including an explicit unknown
  scene bucket when metadata is missing.
- Deterministically shuffle within strata using `build_id + seed`.
- Convert configured local/global ratios into exact integer group targets with a
  deterministic largest-remainder allocation. The sum must equal `target_groups`.
- A source may produce at most one group in a build and belongs to exactly one mode.
- A terminal local/SAM3 failure is replaced from the unused source sequence. It is
  never converted into a global group.
- If the eligible source pool is exhausted, finish as `complete_with_failures` and
  report the target shortfall explicitly.

## 5. Preset Eligibility And Selection

### 5.1 Format Filter

The selector accepts exactly:

```text
xmp | lrtemplate | lut | all
```

- `xmp` does not include `.lrtemplate`.
- `[presets].disabled_formats` is a distinct, duplicate-free subset of `xmp`,
  `lrtemplate`, and `lut`. It is the operator control for excluding a renderer class
  known to be faulty; invalid values are startup errors.
- `all` is the natural union of the non-disabled inventories. It does not rebalance
  formats; format proportions follow eligible inventory.
- If a specifically requested format is disabled or empty, startup fails. If `all`
  leaves no eligible format, startup also fails.
- In `all`, disabled formats are removed before coverage state is built. Record both
  the disabled and effective format sets in the manifest.

### 5.2 GPU Capability

Eligible presets must:

- pass the local GPU capability scan;
- have no embedded local correction that would create nested mask semantics;
- require neither farm rendering nor CPU operator fallback;
- for XMP/LRTemplate calibration, satisfy existing `mean DeltaE00 <= 6` fidelity;
- have taxonomy major/minor labels;
- for global mode, have a non-empty display/style name.

Global-name-less presets remain eligible for local mode. Runtime unsupported
operations use `fallback=skip`; databuild never calls farm or CPU fallback.

Preflight must instantiate the canonical renderer through a dedicated local-GPU-only
factory and assert both an available CUDA device and the preset capability result.
Every production render call carries that immutable backend choice and fails closed
if it is not GPU-backed. No environment variable, generic backend default, or shared
renderer fallback may redirect canonical databuild to farm or CPU execution. CPU
rendering may exist only as an explicitly test-only oracle.

The bank is treated as static for a build. Production rows need only persist
`preset_id` plus the selected format/category metadata; renderer and content hashes
are not mandatory resume gates.

### 5.3 Coverage Algorithm

Coverage state is namespaced by:

```text
(build_id, render_mode, preset_filter)
```

Within each namespace:

1. Major categories use a seeded shuffled bag. Every eligible major is selected once
   before the next major cycle. Inventory size does not weight category frequency.
2. One group is locked to one major category.
3. Inside that major, minor categories use the same equal-frequency shuffled-bag
   rule. Draw round-robin across minors until eight slots are filled.
4. Inside each minor, draw without replacement from a seeded shuffled bag. In
   addition, maintain a provisional group-scoped `used_preset_ids` set: a preset may
   be reserved only when its ID is absent from that set, regardless of which minor
   produced it. This cross-minor check guarantees eight group-wide distinct preset
   IDs even if taxonomy records overlap. Only accepted output advances successful
   coverage.
5. A failed render or diff gate is logged as an attempt, then replaced from the same
   minor. If exhausted, continue with the next minor in the same major.
6. If a major cannot yield eight distinct accepted presets, release its provisional
   reservations and restart the entire group in the next major.
7. Never duplicate a preset, lower a quality threshold, or emit a group with fewer
   than eight candidates.

Reservations and commits must be concurrency-safe and resumable. A crash cannot
advance successful coverage without a corresponding accepted candidate record.

## 6. Rendering Contract

### 6.1 Common Input And Output

- Apply EXIF orientation, resize to short edge 1024 while preserving aspect ratio,
  then render all formats on the same pixel grid.
- Emit JPEG quality 95. The pre-encode tensor/array is authoritative for diff gates.
- C_GT is a single-channel PNG at exactly the rendered image dimensions. Canonical
  config and environment variables expose no storage-downscale bypass.
- Keep candidate result order aligned with selector slot order.

### 6.2 Global

Global candidates apply the complete preset to the whole image through the local GPU
pipeline. No mask is used.

### 6.3 Local

For XMP, LRTemplate, and LUT alike:

```text
edited = render_complete_preset(before, preset)
after  = before * (1 - effective_alpha) + edited * effective_alpha
```

- Composite in sRGB using exact alpha lerp.
- `alpha == 0` is bit-exact before encoding. JPEG may introduce small full-frame
  quantization differences after persistence; this is not mask leakage.
- Do not translate complete presets into Lightroom `Local*` parameter sets.
- The renderer may batch or group work by format internally, but the observable API
  is one list of eight `(preset, mask)` pairs.

## 7. Canonical Mask Protocol

### 7.1 Eight Candidate Slots

Every local group has exactly:

| Mode | Candidate slots | Physical masks |
| --- | ---: | ---: |
| radial | 2 | 2 independently sampled geometries |
| semantic | 2 | 1 SAM3 alpha reused by two different presets |
| band | 2 | 2 independently sampled geometries |
| linear | 2 | 2 independently sampled geometries |

This produces eight candidate slots and seven physical alpha assets. The semantic
candidates share `mask_id` and C_GT path but have distinct `slot_id`, `candidate_id`,
and `preset_id`.

The two geometry samples of a mode need not pass a pairwise similarity test. They
must be sampled independently, and every slot receives a different preset.

### 7.2 Subject Semantics

- All masks affect the subject region or the side containing the subject.
- Do not randomly invert masks to target the background.
- Remove `linear_bisect`, generic center masks, concept masks, and cross-mode fallback.
- If any required geometry cannot be generated after bounded deterministic resampling,
  the local group enters the SAM3 relabel queue. No other mask type may fill its slot.

### 7.3 Pairing

Shuffle the eight masks with a deterministic permutation derived from
`build_id + seed + source_id`, then zip one-to-one with the eight selected presets.
Do not render an 8x8 Cartesian product.

### 7.4 Linear Strength

Linear masks remain continuous soft masks. Compute their raw raster alpha, then:

```text
raw_alpha_mean <= 0.50: amount = 1.0
raw_alpha_mean >  0.50: amount = 0.50 / raw_alpha_mean
effective_alpha = clamp(raw_alpha * amount, 0, 1)
```

Because `raw_alpha_mean <= 1`, `amount` is never below `0.50`. Persist
`raw_alpha_mean`, `amount`, and `effective_alpha_mean`. C_GT stores
`effective_alpha`, because it is the alpha actually used for supervision and blend.

### 7.5 SAM3 Relabel Queue

- Initial eligibility requires a ready mask, so this queue handles corrupt/invalid
  cache entries and masks that cannot support the fixed geometry contract.
- Complete the initial render pass first, then batch relabel queued sources with the
  retained instance-level SAM3 strategy.
- Allow at most two relabel attempts. Re-run integrity and geometry checks after each.
- On terminal failure, record it and draw an unused replacement source for the local
  target. Never synthesize a fallback mask.

## 8. Before/After Visibility Gate

Compute the gate from the in-memory pre-JPEG before/after pair. Let `de(x)` be
per-pixel CIEDE2000 and `w(x)` be effective alpha for local or one for global:

```text
visible_de = sum(w * de) / sum(w)
visible_fraction = sum(w * 1[de >= 2.3]) / sum(w)
```

Accept a candidate only when:

```text
visible_de >= 2.5
visible_fraction >= 0.50
```

Use a deterministic bounded working resolution (recommended short edge 512) for the
metric. Handle an empty/invalid weight map as a hard candidate failure. For local
renders, also assert the pre-encode `alpha == 0` endpoint invariant.

When a candidate fails, keep the mask and request the next preset according to the
selector replacement rules. Failed candidates do not advance successful coverage.

## 9. QA, Tiering, And SFT

- Retain OneAlign plus the existing deterministic extreme exposure/color veto.
- The visibility gate answers "did a meaningful edit occur?"; OneAlign answers
  "which accepted results are best?" Do not combine these into one score.
- Keep the existing OneAlign SFT threshold (`q >= 0.50`) unless an explicit future
  calibration changes the TOML/schema contract.
- Select at most top-2 accepted, reliable, non-veto candidates per group.
- Do not fill missing top-2 slots with low-quality candidates.
- Produce SFT only. Delete DPO generation, thresholds, files, new DB writes, viewer
  UI, and tests. Historical DPO rows/files remain untouched.

Global SFT uses the `style` task and must name the preset's style name. Local SFT uses
the `local` task and describes the edited subject/region without naming the preset.
Delete `auto`, `param`, and `degrade` annotation branches from canonical databuild.

## 10. Annotation System

### 10.1 One Protocol

Use the official OpenAI Python SDK and `/v1/responses` with streaming for external
relays and local vLLM. Remove raw `requests` SSE parsing and all Chat Completions
fallbacks. Accept a response only after a completed stream and successful strict
schema parse.

Relay-specific non-content telemetry may be ignored only by an explicit discriminator
allowlist. The current allowlist contains `codex.rate_limits`; all other unknown or
mistyped stream events remain retryable transport failures.

Declare `openai` as a direct production dependency and lock one exact SDK version in
the repository's dependency/lock files. Preflight must fail clearly if that version
lacks the required Responses streaming or Structured Outputs surface; tests must
exercise the official SDK request and stream-event shapes rather than a hand-written
wire-compatible client.

Official references:

- https://developers.openai.com/api/docs/guides/streaming-responses
- https://developers.openai.com/api/docs/guides/structured-outputs
- https://developers.openai.com/api/docs/guides/images-vision

### 10.2 Input And Output

- Send exactly two images in order: before, then after.
- Encode both as JPEG quality 90 data URIs with longest edge 768.
- Use strict Structured Outputs with eight required string fields and
  `additionalProperties=false`:
  `problem_lighting`, `plan_lighting`, `problem_global_color`,
  `plan_global_color`, `problem_specific_color`, `plan_specific_color`,
  `instruction_long`, and `instruction_short`.
- Put sensible `minLength` constraints in the schema. This is structural validation,
  not semantic review.
- Assemble the six reasoning fields into the existing VeraRetouch problem/plan tokens.
- Output is English; a Chinese preset style name is retained verbatim in global style
  instructions.

Delete leak regexes, quality guards, rewrite feedback, corrective generations,
template fallback, and the old conditional winner verifier. A schema-valid response is
accepted without secondary semantic policing.

For local prompts, provide only the SAM3 subject name and the coarse centroid region.
Do not expose mask mode, geometry, alpha values, or IDs. Compute objective
brightness/warmth/chroma/contrast hints from the true C_GT-weighted before/after pair,
not a rectangular crop. Global uses unit weights.

### 10.3 External Relay Pool

- Configure one or more key lanes with one shared `external_model`; an empty lane list
  is rejected before the build mutates state. Multiple keys for one provider are
  represented as separate, uniquely identified lanes that reuse the provider base URL.
- Select the least-inflight endpoint; break ties round-robin.
- Retry network errors, 429, and 5xx across the pool. Respect `Retry-After`; otherwise
  use jittered exponential backoff.
- Treat other 4xx responses as non-retryable request errors.
- Use at most four transport attempts per queue round across the whole pool, not per
  endpoint.
- A permanent quota code such as `insufficient_quota` or
  `billing_hard_limit_reached` removes that endpoint from the build's pool.
- Continue with the other relay while it has quota. When removal of the final relay
  empties the pool, atomically persist `external_pool_exhausted` before choosing the
  next route. The task whose request discovered final exhaustion remains pending and
  is switched to local vLLM too: retry it locally in the current queue round when an
  attempt remains, otherwise begin its next durable round locally. It must not fail
  merely because it discovered pool exhaustion or wait for a later build. All later
  tasks, including after resume, route directly to local. Ordinary rate limiting or
  5xx errors do not trigger local fallback.
- Record endpoint ID, returned model, token usage, attempts, and status; never record
  API keys.

External request settings: `reasoning.effort=low`, `max_output_tokens=6000`.

### 10.4 Local vLLM

- Retain `Qwen3.5-35B-A3B`, served as `qwen3_5-35b-a3b`.
- Disable thinking, use temperature 0.2, and set `max_output_tokens=2048`.
- Retain vGate and extend it to transparently stream `/v1/responses` while preserving
  discovery, least-outstanding routing, admission control, and headers.
- Databuild talks to one configured local broker URL, not individual replicas.

### 10.5 Durable Queue And Failure Semantics

- Rendering and top-2 selection enqueue annotation tasks; they do not call the model
  inline.
- Run annotation only after the render and SAM3-relabel phases finish.
- Each task gets up to three queue rounds, each with at most four transport attempts.
- Do not issue content-correction requests. Schema failure is terminal for that task.
- After all transport rounds fail, mark `transport_failed` terminal.
- Annotation failure does not delete the group, candidates, QA, or other downstream
  audit data. It only suppresses the corresponding SFT row.

## 11. Pipeline State Machine And Resume

Use explicit phase and task states. A recommended state graph is:

```text
preflight
  -> rendering
  -> sam3_relabel
  -> annotation
  -> projection
  -> complete | complete_with_failures
```

The authoritative persistent artifacts are:

```text
groups.jsonl
sft.jsonl
failures.jsonl
manifest.json
```

PostgreSQL is a rebuildable viewer projection, not a resume authority. Its outage must
not stop production.

Requirements:

- Derive stable IDs from `build_id`, source ID, render mode, group attempt, candidate
  slot, and winner rank as appropriate.
- Reconstruct pending SAM3 and annotation work from accepted groups/SFT/failure events;
  do not require a second manual queue database.
- Serialize append operations, fsync at bounded checkpoints, tolerate one torn tail
  record, and never duplicate completed records on resume.
- A task that was running at process death becomes pending unless its completed record
  is durable.
- Completed render, QA, SAM3, and annotation tasks are skipped on resume.
- A build with unresolved retryable work remains `running`.
- Use `complete` only when targets are met and there are no terminal failures.
- Use `complete_with_failures` when every queue has reached a terminal state but there
  are terminal failures or target shortfalls.

## 12. Output Contracts

### 12.1 `groups.jsonl`

Each record represents one complete source/mode group and contains:

- stable group/build/source IDs, source path, scene, render mode, preset filter;
- selected major category and coverage-cycle metadata;
- exactly eight accepted candidates in stable slot order;
- per candidate: preset ID/format/major/minor, after path, render engine, visibility
  metrics, QA/veto/rank, and attempt lineage;
- local-only: slot mode, mask ID, C_GT path, subject metadata, region, raw/effective
  alpha means, amount, and pairing index;
- selected winner IDs/ranks and stage timestamps.

Do not store in-memory alpha arrays in JSONL.

Append a group record only after all eight accepted candidates and QA/winner fields
are complete. Never append a partial group. If source exhaustion or preset shortfall
prevents completion, write terminal structured events to `failures.jsonl` and counts
to `manifest.json`, but no `groups.jsonl` record for that attempt. Provisional
candidates from an abandoned group do not commit successful coverage; on resume they
may be deterministically recomputed or cleaned as unreferenced staging assets.

### 12.2 `sft.jsonl`

Preserve the training-facing shape:

```text
I_in, I_tar, recipe, local, task_type,
instruction, instruction_short, reasoning, annot_src, qa
```

Only successfully annotated top-2 rows appear. `annot_src` identifies external relay
or local vLLM provenance without leaking credentials.

### 12.3 `failures.jsonl`

Append structured failure events with task/stage IDs, source/group/candidate IDs,
attempt/round, retryability, error code, sanitized message, endpoint ID where relevant,
and terminal status. Do not rely on free-form log parsing for completion accounting.

### 12.4 `manifest.json`

Write atomically and include:

- schema/build IDs, sanitized effective configuration, start/end/status;
- requested and completed local/global counts and actual ratio;
- source eligibility, replacement, and exhaustion counts;
- preset usage by format/major/minor and candidate failure reasons;
- SAM3 relabel expected/completed/terminal counts;
- annotation backend/endpoint/model/usage/failure counts;
- group, candidate, top-1/top-2, and SFT totals;
- hashes/counts of the four final artifacts where practical.

Never include API keys, credential-bearing DSNs, tokens, other credentials, or the
raw secret-bearing TOML.

## 13. Viewer And PostgreSQL Projection

`databuild_viewer` remains a first-class sample-inspection surface.

### Required viewer behavior

- Browse source, all eight candidates, and top-2 SFT outputs.
- For local samples, switch among the rendered result, C_GT, and mask overlay.
- Display preset ID, format, major/minor, visibility metrics, OneAlign score/rank,
  subject/region metadata, and final instruction/reasoning.
- Filter by build, local/global, format, major/minor, queue/failure state, and winner.
- Remove DPO UI/API/selfchecks and old registry-v2 assumptions.
- Preserve read-only access to compatible historical group/SFT records where feasible.

Project canonical JSONL records into PostgreSQL with idempotent upserts keyed by stable
IDs. Projection failure is reported but never changes build completion or resume state.
Do not physically drop historical DPO tables as part of this refactor.

Frontend acceptance requires Playwright screenshots at desktop and mobile widths,
checking candidate strips, mask overlays, filters, long text, and empty/error states for
overlap and clipping.

## 14. Deletion And Retention Matrix

Delete code only after its replacement is wired and tested.

### Delete

- `construct/local_pipeline.py`, its dedicated tests, and benchmark/CLI.
- `construct/degrade.py` and all Track-B degradation production code/tests/docs.
- DPO builders, thresholds, file writes, new provenance writes, viewer code, and tests.
- `construct/reannotate.py`, `tools/annotate_backfill.py`, and old region/backfill tools.
- r5/r6/r7 and other versioned production shell scripts.
- Legacy concept SAM3 producer/cache/regions APIs and `linear_bisect` fallback.
- Old registry-v2 producers, ingest scripts, snapshot assumptions, and r6 globs.
- Source QA scoring/gate/apply/pilot pipeline and its old web UI; retain only the SAM3
  input path described below.
- Chat Completions annotation transport, raw SSE parser, leak/quality guards,
  corrective retry prompts, template fallback, and winner verify path.
- `dataset_build/config.yaml` and production YAML loading after TOML migration.
- Stale runbooks and comments that describe degrade, DPO, farm-backed databuild,
  deferred/manual annotation, or the 1+1+2+4 mask plan.

### Keep And Rewire

- `construct.agent` as the only orchestrator.
- Preset bank/taxonomy parsing, scene mixing, OneAlign, and top-2 SFT tiering.
- Instance-level `sam3_subject_instances`, the ingest/caption/DB support it directly
  requires, and subject-selector evaluation helpers it imports.
- `subject_geom`, `mask_synth`, local preset GPU replay, LUT rendering, and C_GT tests,
  rewritten to the new contract.
- vGate discovery/admission/routing, extended for Responses streaming.
- `databuild_viewer`, rewritten around canonical group/SFT records.
- Shared render farm code for non-databuild consumers, with no canonical databuild edge.
- Existing historical files and PostgreSQL rows.

Before deleting an ambiguous helper, prove with `rg`/imports that it is neither in the
canonical dependency graph nor required by the retained SAM3, renderer, OneAlign, or
viewer path.

## 15. Implementation Sequence

1. **Baseline and worktree safety**
   - Inventory current dirty changes; never reset or overwrite user work.
   - Capture existing targeted test results and import graph.
2. **Contract and TOML foundation**
   - Add strict typed configuration, example file, secret ignore/permission checks,
     canonical IDs, manifest model, and config tests.
3. **Source and selector**
   - Build SAM3-ready source inventory, scene allocation, exact mode quotas, shuffled
     coverage bags, reservations, replacement semantics, and resume tests.
4. **Mask and GPU rendering**
   - Replace the plan with 2+2+2+2, shared semantic alpha, linear mass scaling, mixed
     preset/mask pairing, GPU-only routing, common 1024 input, and effective C_GT.
5. **Visibility and QA**
   - Add alpha-normalized CIEDE2000 gates, candidate refill, OneAlign top-2, and remove
     DPO generation.
6. **Responses annotation**
   - Add strict SDK streaming client, external relay pool/quota latch, durable queue,
     local Qwen/vGate Responses support, and simplified prompt/schema validation.
7. **Orchestration and resume**
   - Implement phase barriers, SAM3 relabel/replacement, annotation drain, JSONL
     idempotency, statuses, failure events, and PostgreSQL projection.
8. **Viewer**
   - Rewire backend and frontend for canonical sample inspection and remove DPO/old
     registry assumptions.
9. **Delete obsolete paths**
   - Remove the approved branches, scripts, configs, tests, and stale documentation.
10. **Full verification**
    - Run unit/integration tests, GPU smoke, queue fault injection, resume tests, viewer
      selfcheck/build/Playwright screenshots, and dead-reference searches.

Do not perform deletion first. Maintain a runnable canonical path at each major phase.

## 16. Verification And Definition Of Done

### Configuration

- Example TOML parses; real config is ignored; permission and all-credential
  redaction tests (including credential-bearing DSNs) pass.
- Missing/unknown/invalid fields and invalid ratios fail before any mutation.
- At least one external key lane is required and every lane ID is distinct; disabled/effective preset
  format validation follows section 5.1.
- No production behavior still depends on old YAML or environment overrides.

### Selection

- Same config/build state yields the same source assignments and candidate sequence.
- Resume produces no duplicate source, group, candidate, mask, or SFT row.
- Major and minor usage differs by at most one within a completed coverage cycle,
  subject to eligibility/exhaustion.
- Every accepted group has one major and eight group-wide distinct presets, including
  when the taxonomy exposes overlapping preset IDs across minors.
- `all` follows natural eligible inventory; explicit filters never leak other formats.

### Masks And Rendering

- Every local group reports exactly two slots per mode; semantic slots share one mask.
- No bisect, concept mask, background inversion, farm call, or CPU operator fallback is
  reachable from canonical databuild.
- Startup and render-call assertions reject a missing GPU or any backend redirection,
  including environment-driven farm/CPU selection.
- Linear amount/mass formula and `amount >= 0.5` are unit-tested.
- C_GT equals the effective alpha used by composition.
- sRGB endpoints, order preservation, common 1024 geometry, and GPU/CPU golden raster
  parity tests pass where the CPU path is test oracle only, not a production fallback.
- LUT identity/axis sentinel and CPU-reference interpolation tests pass after the axis
  fix.

### Visibility And QA

- Synthetic no-op, localized edit, broad weak gradient, and strong outlier cases prove
  both visibility thresholds.
- Failed candidates refill without advancing successful coverage.
- OneAlign produces at most two SFT winners and no new DPO artifact or row.

### Annotation

- Mock SDK streams cover completed, interrupted, failed, malformed-schema, 429, 5xx,
  non-retryable 4xx, Retry-After, quota exhaustion, endpoint removal, dual exhaustion,
  local latch, and resume.
- The official `openai` dependency is directly declared, exactly locked, and its
  mocked event types cover the pinned SDK surface. Dual exhaustion reroutes the task
  that discovered it as well as subsequent and resumed tasks.
- External requests use low effort/6000 tokens; local requests use Qwen3.5,
  thinking off, temperature 0.2/2048 tokens.
- vGate streams `/v1/responses` without buffering the whole response and releases
  inflight accounting only after stream termination.
- No Chat Completions, guard, rewrite, template, verify, or manual backfill path remains.

### End-To-End

- A GPU smoke build of ten groups at the default mix yields exactly seven local and
  three global targets. Every emitted group has eight accepted candidates; any unmet
  target appears only as an explicit terminal shortfall in failures and the manifest.
- One smoke group succeeds under each explicit preset filter when inventory exists.
- Kill/restart tests during render, SAM3 relabel, annotation, and projection resume
  without duplication.
- Final status is `complete` or `complete_with_failures` according to the manifest
  rules; no unresolved retryable task is hidden.
- Viewer backend selfcheck and frontend build pass. Playwright screenshots demonstrate
  source/candidate/top-2 inspection and local C_GT overlays on desktop and mobile.
- `rg` finds no live imports, CLI examples, or production references to deleted paths.
- Existing historical data remains intact.

## 17. Known Risks

- Some major categories may not contain eight source-effective presets after the
  visibility gate; the next-major rule must not corrupt coverage reservations.
- `all` may be heavily skewed toward one format by design. The manifest must make the
  observed distribution visible.
- CIEDE2000 across many candidates can become CPU-heavy; optimize the bounded metric
  resolution without changing thresholds or weighting semantics.
- Third-party relays and vLLM may differ at the edges of Responses Structured Outputs.
  Capability checks and mocked protocol tests must fail clearly, never fall back to
  Chat Completions.
- The worktree is already dirty. Implementation must preserve overlapping user changes
  and must not use destructive Git operations.

## 18. Final Completion Rule

The refactor is complete only when the canonical TOML command runs end-to-end, all
retained tests and new acceptance tests pass, the viewer can inspect canonical samples,
obsolete production paths are physically removed, documentation describes only the new
path, and no required work remains hidden behind a TODO, compatibility branch, manual
backfill, or unverified claim.
