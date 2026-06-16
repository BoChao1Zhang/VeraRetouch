# Running the build under vGate (operator quickstart)

The unified concurrency rewrite (design:
`docs/concurrency/UNIFIED_CONCURRENCY_DESIGN_v2_2026-06-16.md`) decouples the
build into independently-managed pieces. This is the single canonical startup
path.

## Start order

```bash
# 1) vGate broker — transparent OpenAI /v1 proxy in front of the vLLM replicas.
bash dataset_build/core/broker/launch_broker.sh start          # :8003 (or vgate.service)

# 2) vLLM replica(s) — the 35B reasoning model, one per GPU.
bash dataset_build/docker/launch_reasoning.sh start            # reason_g0:8001 / reason_g1:8002
#    …or let the elastic supervisor bring replicas up on free cards:
#    python -m dataset_build.core.broker.supervisor --apply     # (review --once dry-run first)

# 3) Build shards — all point at the broker; shard count = #GPUs in VERA_GPUS.
bash dataset_build/launch_build.sh pilot 2700                  # or: full [STREAMS] | fast-degrade
```

Everything is config-driven by `dataset_build/config.yaml` (its `vllm.base_url`
already points at the broker `:8003`). No per-port configs, no `mk_cfg`, no
hardcoded `/2`. `launch_dual.sh` is **deprecated** (stale 8B model + obsolete
per-port sed); kept only for rollback.

## The decoupled pieces (each independently runnable / rollback-able)

| piece | start | rollback |
|---|---|---|
| broker `:8003` | `launch_broker.sh` / `vgate.service` | point `base_url` back at a replica |
| replicas | `launch_reasoning.sh` / `supervisor.py` | n/a (broker tolerates 0–N) |
| build shards | `launch_build.sh` | n/a |
| core facade (in-process) | automatic in `run.py` | `VERA_DISABLE_CORE=1` (inline path) |
| clean-before-build | `config.yaml` `cleaning.enabled` | set `false` |

## core API (`dataset_build/core`)

The business consumes one synchronous facade (`run.py` builds it via
`build_core(...)`, injected into `BuildContext.core`):

| handle | call | backend |
|---|---|---|
| `core.vllm` | `gen_instruction/reason_params/verify/tag_*` (delegated to `QwenVLCleaner`); low-level `submit(messages, images, json_mode, prio)` | HTTP → broker `:8003`, `X-vgate-class` priority |
| `core.sam3` | `masks(image, concepts, native_size=…)` | `CachedMasker` FS read + live `Sam3Masker` fallback on miss (GPU lease) |
| `core.render` | `render_batch(paths, params_list, downscale_longedge)` / `render_one(path, params)` | in-process teacher renderer under `render_lock` + GPU lease |
| `core.iqa` | `score(path, want_face=…)` / `score_batch(items)` | pyiqa under GPU lease — QA-internal (`source_qa.iqa` consumes it; build never calls it) |
| `core.lr` | `submit(...)` | source_qa Lightroom-farm durable queue (thin passthrough stub) |

Out-of-process twin for the unified image: `core/worker.py` serves
`POST /iqa/score` and `POST /sam3/masks` (FS-drop) over localhost —
`python -m dataset_build.core.worker --port 8010 --device cuda:0`.

Priority classes (broker weighted-fair queue): `build-annotate` (4) > `qa-judge`
(2) > `tag` (1). Build annotation is `build-annotate`; stage-0 tagging is `tag`;
source_qa is `qa-judge`.

## Phase docs

- Broker + supervisor + priority wiring: `dataset_build/core/broker/README.md`
- core facade (render relocation, byte-identical): `dataset_build/core/CORE_FACADE.md`
- clean-before-build PG predicate: `dataset_build/CLEAN_BEFORE_BUILD.md`

## Not done (intentionally; design §10 "only when worth it")

- gpu-compute as a standalone `:8004` HTTP lease service — the in-process
  `GpuCompute` semaphore suffices until render/IQA genuinely cross processes.
- live SAM3 (PG-subprocess) — stage-0 precompute + `CachedMasker` is the current
  path; live SAM3 only pays off for dynamic tag-driven concept expansion.
