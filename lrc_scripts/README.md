# Lightroom Reverse Connection System

A distributed task processing system that enables Linux training servers to distribute Lightroom image processing tasks to multiple client machines.

## 📖 System Overview

This system uses a **reverse connection architecture** where clients actively connect to Linux servers to fetch tasks, solving the traditional challenge of servers accessing client machines.

### Architecture

```
┌─────────────────┐         ┌──────────────────┐
│  Linux Server   │         │   Client 1       │
│                 │         │                  │
│  lrc_task_      │◄────────┤ lr_task_client   │
│  server.py      │  Polling│  + Lightroom     │
│                 │         │                  │
│  Task Queue     │         └──────────────────┘
│  State Machine  │
│                 │         ┌──────────────────┐
│                 │◄────────┤   Client 2       │
│                 │  Polling│                  │
│                 │         │ lr_task_client   │
│                 │         │  + Lightroom     │
└─────────────────┘         └──────────────────┘
```

### Core Features

- ✅ **Reverse Connection** - Clients actively connect to servers, no port forwarding needed
- ✅ **Multi-Server Support** - Clients can connect to multiple Linux servers simultaneously
- ✅ **State Machine** - Complete state management for atomic task processing
- ✅ **Auto-Reconnect** - Automatic reconnection during network interruptions
- ✅ **Load Balancing** - Fair polling of multiple servers for load distribution
- ✅ **Smart Timeout** - Dynamic timeout adjustment based on task complexity
- ✅ **File Transfer** - Support for task file download and result upload

## 📁 Directory Structure

```
lrc_scripts/
├── README.md                    # This document
├── servers/                     # Server components
│   ├── README.md                # Server documentation
│   ├── lrc_task_server.py       # FastAPI task server
│   └── start_reverse_server.sh  # Server startup script
└── clients/                     # Client components
    ├── README.md                # Client documentation
    ├── lr_task_client.py        # Reverse connection client
    ├── start_mac_client.sh      # Client startup script
    └── agent_to_lightroom/      # Lightroom API integration
        └── ...
```

## 🚀 Quick Start

### Prerequisites

#### Server Side (Linux)
- Python 3.7+
- FastAPI, uvicorn, aiofiles
- Sufficient disk space for task files and results

#### Client Side (Mac)
- Python 3.7+
- Adobe Lightroom (must be running)
- aiohttp, requests, Pillow
- Network connection to Linux servers

### Installation

#### Server
```bash
cd servers
pip install fastapi uvicorn aiofiles pydantic
```

#### Client
```bash
cd clients
pip install aiohttp requests Pillow
```

### Startup Process

#### 1. Start Linux Server
```bash
cd lrc_scripts/servers
chmod +x start_reverse_server.sh
./start_reverse_server.sh
```

Server options:
- `--host` - Listen address (default: `0.0.0.0`)
- `--port` - Listen port (default: `8081`)
- `--upload-dir` - Upload directory
- `--results-dir` - Results directory
- See [Server Documentation](servers/README.md) for more options

#### 2. Start Mac Client
```bash
cd lrc_scripts/clients
chmod +x start_mac_client.sh
./start_mac_client.sh
```

Client options:
- `--servers` - Server addresses (format: `IP1:PORT1,IP2:PORT2`)
- `--api-port` - Local Lightroom API port (default: `7777`)
- `--api-path` - API_Lightroom project path
- See [Client Documentation](clients/README.md) for more options

## 📊 State Machine Flow

The system uses a state machine to manage task processing:

```
PENDING → READING → PROCESSING → COMPLETED/FAILED
```

- **PENDING**: Task waiting in queue
- **READING**: Task read by client, awaiting processing confirmation
- **PROCESSING**: Task being processed by client
- **COMPLETED/FAILED**: Task processing finished

## 📋 Task Processing Flow

```
1. Submit task (Linux server)
   ↓
2. Task enters queue (state: PENDING)
   ↓
3. Client polls and fetches task (state: PENDING → READING)
   ↓
4. Client confirms processing start (state: READING → PROCESSING)
   ↓
5. Client downloads task files (if needed)
   ↓
6. Client processes image with Lightroom
   ↓
7. Client uploads processed result
   ↓
8. Client reports result (state: PROCESSING → COMPLETED/FAILED)
   ↓
9. Task completed, result stored on server
```

## 🔌 API Endpoints

### Server Endpoints

#### Core Endpoints
- `POST /api/submit_task` - Submit task (files already accessible)
- `POST /api/submit_task_with_files` - Submit task (with file transfer)
- `GET /api/task_status/{task_id}` - Query task status
- `GET /api/download_task_result/{task_id}` - Download task result
- `GET /api/list_task_masks/{task_id}` - List exported mask rasters for a task
- `GET /api/download_task_mask/{task_id}/{mask_id}` - Download one mask raster

#### Client Communication
- `POST /api/register_client` - Register client
- `GET /api/get_task/{client_id}` - Client fetches task
- `POST /api/start_processing/{task_id}` - Client confirms processing
- `POST /api/report_result` - Client reports result

#### Long-Polling (optional, backward compatible)

Both `GET /api/get_task/{client_id}` and `GET /api/task_status/{task_id}` accept
an **optional** `wait` query parameter (float seconds), e.g.
`GET /api/get_task/{client_id}?wait=20`. When supplied, the server holds the
request open until a task/terminal state is available or `wait` elapses, then
returns the same response shape as a short poll (`get_task` returns `{}` on
timeout). Omitting `wait` (default `0.0`) reproduces the original immediate-return
behavior, so old clients and old servers interoperate unchanged.

The server caps `wait` at `LIGHTROOM_LONG_POLL_MAX` (default `25.0`); callers must
keep their HTTP read timeout above the `wait` they send. Internally the server
wakes waiting requests on task/result changes instead of periodically rescanning
the whole queue.

#### Environment Knobs

New optional environment variables (all backward compatible — defaults preserve
prior behavior):

| Variable | Component | Default | Purpose |
|----------|-----------|---------|---------|
| `LIGHTROOM_LONG_POLL_MAX` | server | `25.0` | Max seconds the server will hold a long-poll request |
| `LIGHTROOM_CLEANUP_INTERVAL` | server | `10.0` | Background task-cleanup loop interval |
| `LIGHTROOM_COMPLETED_TTL` | server | `3600.0` | TTL for evicting completed task metadata from memory; result files remain durable |
| `LIGHTROOM_TASK_MAX_ATTEMPTS` | server | `3` | Max attempts for retryable failures before terminal failed |
| `LIGHTROOM_TASK_RETRY_BASE_DELAY` | server | `5.0` | Initial retry backoff in seconds for retryable failures |
| `LIGHTROOM_MIN_CLIENT_DISK_FREE_BYTES` | server | `5368709120` | Minimum reported client free disk before dispatching more tasks |
| `LIGHTROOM_MAX_CLIENT_CATALOG_COUNT` | server | `120` | Backpressure threshold for reported client catalog/cache size |
| `LIGHTROOM_STATUS_WAIT` | manager | `15.0` | `wait` value the manager sends on `task_status` |
| `LIGHTROOM_RESULT_POLL_INTERVAL` | manager | `0.2` | Result-file grace-poll interval |
| `LIGHTROOM_BRIDGE_POLL_INTERVAL` | bridge | `0.15` | Bridge output-file poll interval |
| `LIGHTROOM_BRIDGE_WAIT_TIMEOUT` | bridge | `120.0` | Bridge default wait when the client does not pass an adaptive timeout |
| `LIGHTROOM_BRIDGE_HEALTH_FILE` | bridge/plugin | `/tmp/lightroom_bridge_health.txt` on macOS/Linux | Plugin heartbeat file read by bridge `/health` for catalog LRU status |
| `LIGHTROOM_BRIDGE_HEALTH_FILE_MAX_AGE` | bridge | `300.0` | Max heartbeat age in seconds accepted by bridge `/health` |
| `LIGHTROOM_BRIDGE_RECOVERY_COMMAND` | bridge | unset | Optional command launched after export timeout; mac launcher can set this with `--enable-lightroom-recovery` |

### Client Features

- Multi-server polling with fair distribution
- Automatic reconnection on network failure
- Dynamic timeout adjustment based on task complexity and recent p95 render latency
- Health checks and statistics reporting with catalog/cache/disk backpressure fields
- File download and upload support
- Optional `export_masks=true` task option. For Lua configs with
  `MaskGroupBasedCorrections`, the client runs Lightroom diff-probe renders,
  uploads `mask_###.png` files, and the server persists them under
  `LIGHTROOM_RESULTS_DIR/<task_id>/masks/`. Configs without masks complete with
  `mask_export_status=skipped_no_masks`.
- Each Lightroom render request writes to a unique per-task/per-variant
  `render_outputs/<task-or-mask-id>_<timestamp>/` directory on the client, and
  uses a matching unique export basename so main, baseline, and mask-probe
  renders cannot overwrite each other.
- The client marks every completed/failed task with `result_data.mask_features`
  (`has_masks`, `mask_count`, `mask_types`, `has_ai_masks`) so consumers can
  distinguish ordinary local edits from mask / likely AI-mask renders even when
  `export_masks=false`.
- The Lightroom plugin writes per-render `.lightroom_render_status` files under
  each unique output directory. The bridge reads them to surface plugin-side
  parse/import/apply failures as structured, non-timeout errors.
- Client local task cache is LRU-bounded (`--local-cache-limit`, default 100) under a configurable `--workdir-base`; server result files under `LIGHTROOM_RESULTS_DIR` are the durable store and remain downloadable after metadata TTL eviction.

## 📚 Documentation

- [Server Documentation](servers/README.md) - Detailed server configuration and API
- [Client Documentation](clients/README.md) - Client configuration and usage

## 🛠️ Troubleshooting

Common issues:
- **Client can't connect**: Check server status, network, firewall settings
- **Lightroom API not starting**: Ensure Lightroom is running, check port 7777
- **Task processing timeout**: Increase processing timeout, check task complexity
- **File upload failure**: Check network connection, server disk space

## 📝 Notes

- The system is designed for JarvisEvo's distributed Lightroom image processing
- Requires Lightroom and corresponding API plugin
- Server handles task queuing, state management, and file storage
- Client handles connection, task processing, and result reporting
