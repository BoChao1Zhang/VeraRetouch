@echo off
REM Thin wrapper that launches the PowerShell client launcher with all args forwarded.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_windows_client.ps1" %*
