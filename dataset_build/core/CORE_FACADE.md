# core facade (Phase 2)

The decoupled `core.*` surface the build consumes. Phase 2 relocates ownership
of the heavy resources behind a uniform synchronous facade **without changing
any algorithm** — the build's thread-based overlap loop, drain, and ShardWriter
are untouched, and output is byte-identical to the pre-core path.

## Modules

| module | what |
|---|---|
| `core/__init__.py` | `Core` facade (`.vllm/.sam3/.iqa/.render/.lr`) + `build_core(config, *, renderer, masker, cleaner, gpu=None)` factory |
| `core/render_worker.py` | `RenderClient` — owns `render_lock` (threading.Lock, **non-reentrant**) + renderer; `render_batch()`/`render_one()` are the verbatim `streams._render_after_batch`/`_render_after` logic. Canonical `downscale_rgb()` (streams' `_downscale_rgb` now delegates here). |
| `core/client.py` | `VLLMClient` (wraps `QwenVLCleaner`, **transparent `__getattr__` delegation** of `gen_instruction`/`reason_params`/`verify`/`tag_*`, plus low-level prio-aware `submit(messages, images, json_mode, prio)` — encodes `images` to data-URIs and absorbs broker `429`/transient with bounded, `Retry-After`-aware retry), `Sam3Client` (precomputed `CachedMasker` read **+ live `Sam3Masker` fallback on cache miss under a GPU lease**), `IqaClient` (QA-internal pyiqa under GPU lease; `score` + lease-amortized `score_batch`), `LrClient` (thin LR submit handle) |
| `core/gpu_compute.py` | `GpuCompute` — exclusive **non-reentrant** per-device lease; strict order **lease → render_lock**; least-busy pick. **Engaged in the build render path** (`run.py` passes a shared `GpuCompute(["cuda:0"])`; uncontended → byte-identical), establishing the render/IQA/SAM3 co-tenancy contract. |
| `core/worker.py` | In-container core worker HTTP server (unified-image topology): `POST /iqa/score` (pyiqa dict) and `POST /sam3/masks` (live SAM3 → **FS-drop PNG paths**, not inline binary). One per-process GPU lease serializes IQA/SAM3; lazy model load (a model that can't load in-env → `503`, server stays up). The in-process `core.iqa`/`core.sam3` clients are the default; this is their cross-process twin. |

`source_qa` consumption: `llm_qa`/`preset_qa` route through the broker with
`X-vgate-class: qa-judge` (Phase 1); `iqa.run()` now scores through
`core.IqaClient` under a `GpuCompute` lease. `core.lr`'s rich typed signature is
the one piece intentionally left as a thin passthrough stub.

## How it wires in (additive, reversible)

`run.py` builds a `Core` from the constructed collaborators and injects the
delegating wrappers:

```python
core = build_core(config, renderer=models["renderer"],
                  masker=models["masker"], cleaner=models["cleaner"])
ctx = BuildContext(..., renderer=models["renderer"],  # raw, for the render_needed gate
                   masker=core.sam3, cleaner=core.vllm, core=core)
```

- `ctx.cleaner` / `ctx.masker` are the delegating wrappers → every
  `ctx.cleaner.gen_instruction(...)` / `ctx.masker.masks(...)` call in `streams.py`
  is **unchanged** and byte-identical.
- `streams._render_after_batch` / `_render_after` route to `ctx.core.render` when
  present (owning the relocated `render_lock`+renderer), else fall back to the
  inline legacy code (the byte-identical fallback).
- **`gpu=None` in the build path**: render is serialized by `render_lock` alone,
  exactly as before. The GPU lease only engages when render and IQA share a card
  (the QA path) — build never calls IQA.

**Rollback**: drop the `core=` kwarg in `run.py` (and revert `masker=`/`cleaner=`
to `models[...]`). The inline path is the byte-identical fallback; `core=None`
everywhere reproduces today's behavior exactly.

## Invariants preserved

- `render_lock` is a `threading.Lock` (NOT RLock), wraps **only**
  `renderer.render()`, non-reentrant — moved verbatim into `RenderClient`.
- render is per-shard single-worker on the main thread; the overlap pipeline
  (render batch K+1 ‖ clean batch K), `max_outstanding` backpressure, drain, and
  single-threaded ShardWriter are all unchanged.
- `core.iqa` is QA-internal; the build request path never touches it.

## Acceptance

### GPU-free byte-identical unit proof (done)

Drives the **real** `streams.BaseStream._render_after_batch` / `_render_after`
with a deterministic mock renderer through both the inline (`core=None`) and the
`core.render` path and asserts identical arrays — covering None-param scatter,
input ordering, 768 downscale, `render_kw` equality vs `BuildContext.render_kw`
on the real `config.yaml`, all-None batches, and wrapper delegation. Plus
`run.py --dry-run` regression across S1/S2/S6/S7 (inline fallback) commits
samples cleanly.

### Real S2 byte-identical build-diff (user-gated; needs GPU + broker + sources)

The final cutover gate from the v2 design. Run one S2 shard **before** and
**after** enabling core and diff the shard bytes:

```bash
# 0) broker up (Phase 0) + a replica live; sources/recipes indexed.
# 1) BEFORE — pin the inline path: temporarily pass core=None in run.py
#    (or `git stash` the Phase-2 wiring) and build a fixed S2 slice:
python -m dataset_build.run --config dataset_build/config.yaml \
    --stream S2 --pilot 64 --limit 32 --out-suffix _s2_pre
# 2) AFTER — restore the Phase-2 wiring (core on) and rebuild the SAME slice:
python -m dataset_build.run --config dataset_build/config.yaml \
    --stream S2 --pilot 64 --limit 32 --out-suffix _s2_post
# 3) diff the shard parquet/jsonl + image bytes between _s2_pre and _s2_post.
#    Expectation: identical (the teacher render is deterministic for fixed
#    params; the VLM annotate text is sampled at temperature 0.2 so pin the
#    seed / compare structure, not the free-text token stream).
```

Note: the VLM annotate fields are non-deterministic at `temperature>0`; the
byte-identical guarantee is about the **render/mask/orchestration relocation**,
not the model's sampled text. Diff the rendered `after`, masks, params, and
record structure; treat annotate prose as out-of-scope for byte equality.
