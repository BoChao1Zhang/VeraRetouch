# Lightroom Task Distribution Server

FastAPI server for distributing Lightroom editing tasks to client machines.

## 🚀 Quick Start

```bash
./start_reverse_server.sh                    # Default settings
./start_reverse_server.sh --port 9000        # Custom port
./start_reverse_server.sh --help             # View all options
```

## ⚙️ Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--host` | `-h` | Listen address | `0.0.0.0` |
| `--port` | `-p` | Listen port | `8081` |
| `--script` | `-s` | Python script path | `./lrc_task_server.py` |
| `--upload-dir` | `-u` | Upload directory | See script |
| `--results-dir` | `-r` | Results directory | See script |
| `--max-retries` | `-m` | Maximum retries | `5` |
| `--wait-timeout` | `-w` | File wait timeout (s) | `180.0` |

## 📊 State Machine

```
PENDING → READING → PROCESSING → COMPLETED/FAILED
```

## 🔌 API Endpoints

| Category | Endpoint | Description |
|----------|----------|-------------|
| Core | `POST /api/register_client` | Register client |
| Core | `GET /api/get_task/{client_id}` | Fetch task |
| Core | `POST /api/start_processing/{task_id}` | Confirm processing |
| Core | `POST /api/report_result` | Report result |
| Core | `POST /api/submit_task` | Submit task |
| Core | `POST /api/submit_task_with_files` | Submit task with server-side file transfer |
| File | `GET /api/download_file/{task_id}/{file_type}` | Download file |
| File | `POST /api/upload_result` | Upload result |
| File | `POST /api/upload_mask_result` | Upload one exported mask raster |
| File | `GET /api/download_task_result/{task_id}` | Download processed image |
| File | `GET /api/list_task_masks/{task_id}` | List exported mask rasters |
| File | `GET /api/download_task_mask/{task_id}/{mask_id}` | Download one mask raster |
| Monitor | `GET /api/health` | Health check |
| Monitor | `GET /api/stats` | Statistics |

`POST /api/submit_task` and `POST /api/submit_task_with_files` accept the
optional query/form parameter `export_masks=true`. New clients then export mask
rasters for configs containing `MaskGroupBasedCorrections`; the server stores
them as `LIGHTROOM_RESULTS_DIR/<task_id>/masks/mask_###.png` and includes mask
metadata in `task_status` results.

New clients also include `result_data.mask_features` in task results. This is a
lightweight G3 marker for `has_masks`, `mask_count`, `mask_types`, and
best-effort `has_ai_masks`.

## ⏳ Long-Polling (optional, backward compatible)

Both `GET /api/get_task/{client_id}` and `GET /api/task_status/{task_id}` accept
an **optional** `wait` query parameter (float seconds), e.g.
`GET /api/get_task/{client_id}?wait=20`. When set, the server holds the request
open until a task is available / the task reaches a terminal state, or `wait`
elapses, then returns the SAME JSON shape as a short poll (`get_task` returns
`{}` on timeout). Omitting `wait` (default `0.0`) preserves the original
immediate-return behavior, so old and new clients/servers interoperate. The
server caps `wait` at `LIGHTROOM_LONG_POLL_MAX`; callers must keep their HTTP read
timeout above the `wait` they send. Internally the server wakes waiting requests
when tasks/results change, so idle clients do not force periodic full-queue
rescans.

## 🧪 Environment Knobs

| Variable | Default | Purpose |
|----------|---------|---------|
| `LIGHTROOM_LONG_POLL_MAX` | `25.0` | Max seconds the server holds a long-poll request |
| `LIGHTROOM_CLEANUP_INTERVAL` | `10.0` | Background task-cleanup loop interval |
| `LIGHTROOM_COMPLETED_TTL` | `3600.0` | TTL for evicting completed task metadata from memory; result files remain durable |
| `LIGHTROOM_TASK_MAX_ATTEMPTS` | `3` | Max attempts for retryable failures before terminal failed |
| `LIGHTROOM_TASK_RETRY_BASE_DELAY` | `5.0` | Initial retry backoff in seconds for retryable failures |
| `LIGHTROOM_MIN_CLIENT_DISK_FREE_BYTES` | `5368709120` | Minimum reported client free disk before dispatching more tasks |
| `LIGHTROOM_MAX_CLIENT_CATALOG_COUNT` | `120` | Backpressure threshold for reported client catalog/cache size |

`GET /api/stats` returns `render_metrics` with terminal task counts, success
rate, p50/p95 render latency, and failure counts grouped by structured
`error_code`.

`GET /api/download_task_result/{task_id}` first uses in-memory task metadata,
then falls back to `LIGHTROOM_RESULTS_DIR/<task_id>/processed.jpg`, so completed
results remain downloadable after `LIGHTROOM_COMPLETED_TTL` evicts metadata.
Mask downloads use the same durable-store fallback through
`GET /api/list_task_masks/{task_id}` and
`GET /api/download_task_mask/{task_id}/{mask_id}`.

## 🔍 Verification

- API docs: `http://localhost:PORT/docs`
- Health check: `http://localhost:PORT/api/health`

## 📚 Related

- [Client README](../clients/README.md)
