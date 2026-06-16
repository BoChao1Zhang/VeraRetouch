# JarvisArt Lightroom Client

Client that connects to Linux training servers and automatically processes Lightroom image editing tasks.

## Quick Start

### 1. Install Dependencies
```bash
pip install aiohttp fastapi uvicorn requests pillow pyyaml
```

### 2. Install Lightroom Plugin
1. Open Lightroom Classic
2. `File` → `Plug-in Manager`
3. Click `Add`, select `agent_to_lightroom/XMPlayer.lrplugin/` directory

### 3. Start Client

#### macOS
```bash
chmod +x start_mac_client.sh
./start_mac_client.sh
```

Or specify server:
```bash
./start_mac_client.sh --servers "SERVER_IP:PORT"
```

#### Windows

Plugin installation is identical to macOS: open Lightroom Classic → `File` →
`Plug-in Manager` → `Add`, then select the
`agent_to_lightroom/XMPlayer.lrplugin/` directory.

Requirements: Python 3.9+ available via the Windows `py` launcher (or `python`
on PATH).

Install the package-local Python environment from the extracted client folder:
```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements_windows.txt
```

The Windows launcher automatically uses `.venv\Scripts\python.exe` when the
local virtual environment exists. To use a global Python instead:
```powershell
py -3 -m pip install -r requirements_windows.txt
```

Run the launcher with PowerShell:
```powershell
powershell -ExecutionPolicy Bypass -File start_windows_client.ps1
```

Or simply double-click `start_windows_client.bat` (it forwards all arguments to
the PowerShell script). The Windows launcher mirrors the Mac script: it resolves
Python, starts the local Lightroom bridge if needed, probes server reachability,
and runs the poller with auto-restart.

The packaged Windows client defaults to the Linux server's Tailscale endpoint
`100.65.247.100:8081`. Keep Tailscale connected on Windows before starting the
client. To override the target:
```powershell
powershell -ExecutionPolicy Bypass -File start_windows_client.ps1 --servers "TAILSCALE_IP:PORT"
```

## Main Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--servers` | Server addresses (comma-separated) | `100.65.247.100:8081` |
| `--api-port` | Local API port | `7777` |
| `--poll-interval` | Polling interval (seconds) | `1.0` |
| `--long-poll-wait` | Long-poll wait hint sent to the server on `/api/get_task` (seconds). The client holds the request open up to this long, returning as soon as a task is available; old servers ignore it and reply immediately. | `20.0` |
| `--local-cache-limit` | Number of local `lightroom_task_*` directories to keep on the client | `100` |
| `--workdir-base` | Directory used for downloaded local `lightroom_task_*` work dirs | mac: `~/lrc_client_workdir`, win: `%LOCALAPPDATA%\LightroomTaskClient\workdir` |
| `--enable-lightroom-recovery` | macOS only: enable the bundled Lightroom restart hook after bridge export timeout | disabled |
| `--lightroom-recovery-command` | Windows only: command assigned to `LIGHTROOM_BRIDGE_RECOVERY_COMMAND` after bridge export timeout | unset |
| `--skip-tailscale-check` | Skip local Tailscale CLI/IP validation | disabled |

The Lightroom plugin also keeps its imported task-photo catalog entries
LRU-bounded at 100 and reports the current LRU size through the local bridge
`/health` endpoint. The plugin writes a small heartbeat file
(`/tmp/lightroom_bridge_health.txt` on macOS) so bridge health does not need to
occupy the Lightroom socket. Reinstall or repackage
`agent_to_lightroom/XMPlayer.lrplugin` on each Lightroom client after updating
this directory.

Rendered files are written under the task workdir in unique
`render_outputs/<task-or-mask-id>_<timestamp>/` directories with unique export
basenames. This prevents the main render and `export_masks=true` baseline/probe
renders from sharing the same Lightroom export filename.

For each render, the plugin also writes `.lightroom_render_status` in that
unique output directory. The local bridge reads this file while waiting for the
JPEG so plugin-side failures such as preset parse/import/apply errors are
reported immediately as structured errors instead of generic export timeouts.

Task results include `result_data.mask_features` with mask count/type markers
and a best-effort `has_ai_masks` flag.

## Troubleshooting

**Connection Failed**:
- Ensure Lightroom is running
- Check server IP and port
- Confirm plugin is installed

**Port Occupied**:
```bash
netstat -an | grep 7878  # Check port
```

**Test Connection**:
```bash
python lr_task_client.py --servers "IP:PORT" --test
```
