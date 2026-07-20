# vGate Responses broker

vGate is the local OpenAI-compatible reverse proxy used by canonical databuild
annotation. It discovers one or more `vllm serve` replicas, applies admission
control, routes each request to the least-busy live replica, and forwards the
Responses stream without buffering it.

The public model API is intentionally limited to:

- `POST /v1/responses`
- `GET /v1/models`

Operational endpoints are `GET /status` and `GET /healthz`. Chat Completions
and legacy Completions are not exposed.

## Routing contract

- Candidate replica ports are probed through `GET /v1/models`.
- A replica is routable only when its served model matches
  `qwen3_5-35b-a3b` (or the explicitly configured service value).
- Requests are admitted against `per_replica_cap * live_replicas`.
- Live replicas are selected by least outstanding requests, with round-robin
  tie breaking.
- The `X-vgate-class` header selects a weighted queue class. Canonical
  annotation uses `build-annotate`.
- A Responses request retains its admission slot until the upstream stream
  closes, fails, or is cancelled.
- When no replica is live, the broker returns `503` with `Retry-After`.

The broker does not load models or touch CUDA. Replica lifecycle remains owned
by the vLLM launch/supervisor service.

## Run

From the repository root:

```bash
bash dataset_build/core/broker/launch_broker.sh start
bash dataset_build/core/broker/launch_broker.sh stop
```

Or run it directly:

```bash
/home/bc/miniconda3/bin/python \
  -m dataset_build.core.broker.app \
  --host 127.0.0.1 \
  --port 8003
```

Configure canonical databuild with the broker URL in TOML:

```toml
[annotation.local]
base_url = "http://127.0.0.1:8003/v1"
api_key = "EMPTY"
model = "qwen3_5-35b-a3b"
temperature = 0.2
enable_thinking = false
max_output_tokens = 2048
```

The producer itself is invoked only through:

```bash
python -m construct.agent run --config /absolute/path/to/databuild.toml
```

## Service options

| Flag | Environment | Default | Meaning |
| --- | --- | --- | --- |
| `--host` | `VGATE_HOST` | `0.0.0.0` | broker listen address |
| `--port` | `VGATE_PORT` | `8003` | broker listen port |
| `--replica-ports` | `VGATE_REPLICA_PORTS` | `8001,8002` | candidate local vLLM ports |
| `--served-name` | `VGATE_SERVED_NAME` | `qwen3_5-35b-a3b` | required upstream model ID |
| `--cap` | `VGATE_REPLICA_CAP` | `32` | admission slots per live replica |
| - | `VGATE_MAX_QUEUE_WAIT` | `300` | maximum admission wait in seconds |
| - | `VGATE_POLL_INTERVAL` | `10` | discovery interval in seconds |
| - | `VGATE_STARVE_BOOST` | `30` | queue-age priority boost in seconds |
| - | `VGATE_DEFAULT_CLASS` | `build-annotate` | class for unlabeled requests |

## Verification

`GET /healthz` is healthy only when at least one matching replica is live.
`GET /status` reports replica health, inflight counts, queue depths, and the
current admission budget. A canonical preflight still verifies the pinned
OpenAI SDK and renderer separately.

Stopping vGate does not mutate replica or build state. After a restart,
canonical annotation resume reconstructs pending work from the JSONL journals.
