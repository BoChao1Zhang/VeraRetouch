# Windows Lightroom Client Package

This package connects to the JarvisEvo Lightroom task server through Tailscale.

## Requirements

- Windows with Tailscale installed and logged in
- Python 3.9+ available as `py -3` or `python`
- Adobe Lightroom Classic

## Install Python Environment

Run these commands from the extracted package directory:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements_windows.txt
```

The launcher automatically uses `.venv\Scripts\python.exe` when `.venv` exists.
If you prefer a global Python environment, install the same requirements with:

```powershell
py -3 -m pip install -r requirements_windows.txt
```

## Server Endpoint

Default server: `100.65.247.100:8081`

This is the Linux server's Tailscale IPv4 address. Keep Tailscale connected
before starting the client.

## Install Lightroom Plugin

In Lightroom Classic:

1. Open `File` -> `Plug-in Manager`
2. Click `Add`
3. Select `agent_to_lightroom\XMPlayer.lrplugin`

## Start

Double-click:

```text
start_windows_client.bat
```

Or run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\start_windows_client.ps1
```

Override server if needed:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\start_windows_client.ps1 --servers "TAILSCALE_IP:8081"
```
