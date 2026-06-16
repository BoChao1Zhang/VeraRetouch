#!/usr/bin/env bash
set -u

reason="${LIGHTROOM_STALL_REASON:-unknown}"
log_file="${LIGHTROOM_RECOVERY_LOG:-$HOME/lrc_client/lightroom_recovery.log}"

mkdir -p "$(dirname "$log_file")"
{
  echo "$(date '+%Y-%m-%d %H:%M:%S') recovery requested: $reason"

  osascript <<'APPLESCRIPT' >/dev/null 2>&1 || true
tell application "System Events"
  if exists process "Adobe Lightroom Classic" then
    tell process "Adobe Lightroom Classic"
      key code 53
      delay 1
      key code 36
    end tell
  end if
end tell
APPLESCRIPT

  sleep "${LIGHTROOM_RECOVERY_GRACE_SECONDS:-5}"

  if pgrep -f "/Applications/Adobe Lightroom Classic/Adobe Lightroom Classic.app/Contents/MacOS/Adobe Lightroom Classic" >/dev/null 2>&1; then
    osascript -e 'tell application "Adobe Lightroom Classic" to quit' >/dev/null 2>&1 || true
    sleep 8
  fi

  pkill -f "/Applications/Adobe Lightroom Classic/Adobe Lightroom Classic.app/Contents/MacOS/Adobe Lightroom Classic" >/dev/null 2>&1 || true
  sleep 2
  open -a "Adobe Lightroom Classic" >/dev/null 2>&1 || \
    nohup "/Applications/Adobe Lightroom Classic/Adobe Lightroom Classic.app/Contents/MacOS/Adobe Lightroom Classic" >/tmp/lightroom_recovery_start.log 2>&1 &

  echo "$(date '+%Y-%m-%d %H:%M:%S') recovery command finished"
} >>"$log_file" 2>&1
