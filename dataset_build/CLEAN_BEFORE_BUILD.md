# Clean-before-build (Phase 3)

Read-only, per-asset PG cleaning predicate that lets the build consume **only**
source_qa-cleared assets. Implemented in `dataset_build/cleaning.py`, wired into
`run.py:load_plan_inputs`, configured under `config.yaml` `cleaning:`.

## What it does

When `cleaning.enabled: true`, `load_plan_inputs` filters each stream's source
(and optionally recipe) pool to assets whose `assets` row in PG satisfies:

```
auto_verdict IN (keep, needs_local_render)   AND   dup_of IS NULL   (require_dup_head)
```

Join key is direct: `assets.asset_id == SourceItem.source_id == RecipeAsset.recipe_id`
(source_qa ingest assigns the build's stable id as the asset id).

- **Opt-in**: default `enabled: false` → zero behavior change.
- **Fail-closed**: any PG/connection/query error raises `SystemExit` (HALT). The
  build never falls open to consuming un-cleaned assets.
- **Read-only**: opens its own autocommit SELECT connection; never writes.
- **Conservative**: absent / `drop` / `review` / NULL verdicts are excluded.
- **Scope**: `apply_to: ["image"]` by default (gate sources). Add `"preset"` to
  gate recipes too — but note the preset pool currently has very few cleared
  rows (preset verdicts are `needs_local_render`/`review`/`drop`, no `keep`), so
  gating recipes would starve S2/S6 until preset QA clears more.

Verified against the live `vera_source_qa` DB (42,454 cleared images / 44 cleared
presets): with `enabled` + `apply_to: [image]`, S2's source pool filtered
119,060 → 36,736; recipes unchanged; fail-closed on a bad DSN.

## Go-live checklist (operator decision)

Activating this changes which assets the 1M build consumes — do it deliberately:

1. **Populate verdicts.** Run the full source_qa pipeline through `apply.py` so
   `assets.auto_verdict` / `dup_of` reflect the current corpus. (The apply
   closure that writes `source_index.cleaned.jsonl` / `recipe_index.qa.jsonl` is
   a separate, also user-gated switch at `config.yaml` `storage:` lines 305-311 —
   this PG predicate is an *alternative* row-level gate that does not require
   flipping the index files.)
2. **Sanity-check counts** at the chosen verdict set (e.g. the `load_plan_inputs`
   integration check above) so a stream is not unexpectedly starved.
3. Set `cleaning.enabled: true` (and adjust `apply_to` / `keep_verdicts` as
   needed).
4. Run a pilot shard; confirm `[clean-before-build] <stream> sources N -> M` logs
   the expected reduction and the build commits.

Rollback: set `cleaning.enabled: false`.

## Deferred (not in Phase 3)

- **source_qa IQA via `core.iqa` GPU lease.** The QA CLI runs IQA as its own
  sequential stage in a separate process with no co-tenant, so an in-process
  GPU lease is currently a no-op. It becomes meaningful only when IQA and the
  renderer share a card inside the unified image — wire it as part of that
  cutover, not here. (source_qa's `llm_qa`/`preset_qa` already route through the
  broker with `X-vgate-class: qa-judge` from Phase 1.)
