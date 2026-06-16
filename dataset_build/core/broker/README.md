# vGate — vLLM broker (Phase 0)

Out-of-process, OpenAI-compatible reverse proxy in front of the 1–2 stock
`vllm serve` replicas (`reason_g0:8001` / `reason_g1:8002`). It is the **Phase 0**
deliverable of the unified-concurrency rewrite
(`docs/concurrency/UNIFIED_CONCURRENCY_DESIGN_v2_2026-06-16.md`): the
highest-leverage, lowest-risk, fully-reversible first step. It needs **no**
business-code changes and **no** unified image.

## What it does

- **Replica discovery** — polls candidate ports (default `8001,8002`) every
  ~10s; a replica is *live* iff `GET /v1/models` returns the expected served
  name (`qwen3_5-35b-a3b`). Tolerates only one replica being up (today only
  `reason_g1:8002`). A port serving a *different* model is **isolated** (kept
  out of routing) and logged loudly.
- **Least-outstanding routing** — `argmin(inflight)` across live replicas, ties
  broken round-robin. This is the direct fix for "one replica 84 GB @ 0% while
  the other is overloaded".
- **Global admission budget** — `per_replica_cap × live_replicas` (cap default
  32). The single number that absorbs the 1↔2-card question. Over budget,
  requests **block** (bounded) rather than fail — only a queue wait past
  `VGATE_MAX_QUEUE_WAIT` returns `429 + Retry-After` (preset_qa has no client
  retry, so we prefer blocking).
- **3-class weighted-fair queue** — keyed on the `X-vgate-class` request header:
  `build-annotate` (4) > `qa-judge` (2) > `tag` (1), with an anti-starvation
  max-wait boost. Phase 0 clients don't set the header yet (Phase 1 wires it via
  `core.vllm.submit(prio=...)`), so traffic defaults to one class == plain FIFO.

Transparent on `/v1/chat/completions`, `/v1/completions`, `/v1/models`. Ops
endpoints: `/status` (replicas, budget, inflight, queue depth, vtime) and
`/healthz`. Pure Python (`fastapi`/`uvicorn`/`httpx`); no torch, no GPU.

## Run

```bash
# foreground
bash dataset_build/core/broker/launch_broker.sh start
# stop
bash dataset_build/core/broker/launch_broker.sh stop

# or directly (from the repo root)
/home/bc/miniconda3/bin/python -m dataset_build.core.broker.app --port 8003
```

SPOF mitigation = systemd auto-restart; see `vgate.service` for install steps.

### Environment / flags

| flag | env | default | meaning |
|---|---|---|---|
| `--port` | `VGATE_PORT` | `8003` | broker listen port |
| `--replica-ports` | `VGATE_REPLICA_PORTS` | `8001,8002` | candidate vLLM host ports |
| `--served-name` | `VGATE_SERVED_NAME` | `qwen3_5-35b-a3b` | expected model id; others isolated |
| `--cap` | `VGATE_REPLICA_CAP` | `32` | per-replica admission cap |
| — | `VGATE_MAX_QUEUE_WAIT` | `300` | seconds to block before `429` |
| — | `VGATE_POLL_INTERVAL` | `10` | replica discovery interval (s) |
| — | `VGATE_STARVE_BOOST` | `30` | queue age (s) that preempts WFQ |
| — | `VGATE_DEFAULT_CLASS` | `build-annotate` | class for unlabeled requests |

## Wiring the business onto the broker (the drop-in)

Point every vLLM consumer's `base_url` at the broker; **nothing else changes**:

- build: `config.yaml` → `vllm.base_url: "http://localhost:8003/v1"`
- source_qa: `SOURCE_QA_VLLM=http://localhost:8003/v1` (or the default in
  `source_qa/config.py`)

The legacy `launch_dual.sh` `mk_cfg` sed (`:8001` → `:8002`) no longer matches a
`:8003` base_url, so both shards naturally route through the broker.

## Acceptance (Phase 0)

- today's build + source_qa run unchanged against `:8003`;
- `reason_g1` no longer sits at 0% (least-outstanding spreads load);
- the "build `--shard` hits a dead `:8001`" failure disappears (broker routes to
  whatever is live).

## Rollback

Stop the broker and point `base_url` back at a replica (`:8001`/`:8002` for
build, `:8002` for source_qa). No state to clean up — replicas are untouched.

---

# Phase 1 — priority classes + elastic supervisor

## `X-vgate-class` priority wiring (now live)

The weighted-fair queue only bites once clients label their traffic. They now do:

| consumer | class | where |
|---|---|---|
| build `QwenVLCleaner` (gen_instruction / reason_params / verify) | `build-annotate` (P0) | `vlm_clean.py` instance attr `vgate_class`, sent per-request via `extra_headers` |
| stage-0 `tag_precompute` | `tag` (P2) | `tag_precompute.py` sets `cleaner.vgate_class = "tag"` |
| source_qa `llm_qa` + `preset_qa` (questionnaire C) | `qa-judge` (P1) | `X-vgate-class` header on the `requests.post` |

The header is harmless against a plain vLLM server (ignored). Verify the broker
sees the classes via `/status` → `vtime` (per-class virtual clock increments by
`1/weight` per admission: build +0.25, qa +0.5, tag +1.0).

`preset_qa` questionnaire C also gained a **bounded retry** (3 attempts, backoff,
honors `Retry-After`) so a transient blip or a broker `429` no longer silently
drops the result.

## `supervisor.py` — elastic replica lifecycle

Replaces `launch_reasoning.sh`'s all-or-nothing start with per-card management:
ensures one replica per *eligible* (free) GPU, restarts crashed ones, and leaves
healthy/warming ones alone. The broker discovers the result and rescales its
budget — the two are decoupled (supervisor actuates, broker observes).

```bash
# safe: observe what it WOULD do (no docker mutation)
/home/bc/miniconda3/bin/python -m dataset_build.core.broker.supervisor --once
# actuate (starts replicas on free cards via launch_reasoning.sh start-one):
/home/bc/miniconda3/bin/python -m dataset_build.core.broker.supervisor --apply
```

Safety: **dry-run by default**; **up-scale only** (starts on free cards, restarts
exited containers, never kills a healthy/compiling one); **free-card gate**
(`--min-free-mb`, default 70000 — a 35B-fp8 replica needs ~83 GB of a 97 GB
H100); **min-dwell** hysteresis. Reuses `launch_reasoning.sh start-one <name>
<gpu> <port>` (non-blocking `docker run -d` + the persistent compile cache), so
the docker template stays a single source of truth. systemd unit (runs with
`--apply`): `vgate-supervisor.service`.

| flag | env | default | meaning |
|---|---|---|---|
| `--apply` | `VGATE_SUP_APPLY=1` | off (dry-run) | actually start/stop containers |
| `--once` | — | — | single tick then exit (for inspection) |
| `--interval` | `VGATE_SUP_INTERVAL` | `15` | reconciliation period (s) |
| `--min-free-mb` | `VGATE_SUP_MIN_FREE_MB` | `70000` | free-mem gate to host a new replica |
| `--min-dwell` | `VGATE_SUP_MIN_DWELL` | `60` | min seconds between actions on one container |
| `--gpu-map` | `VGATE_SUP_GPU_MAP` | `0:reason_g0:8001,1:reason_g1:8002` | GPU→(name,port) |

Down-scaling (freeing a card for a heavy IQA pass) is intentionally **not**
automatic in Phase 1 — it's a future explicit signal.
