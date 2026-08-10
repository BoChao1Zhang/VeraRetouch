#!/usr/bin/env bash
# =============================================================================
# Make the GPU queue daemon survive a reboot.
#
# The campaign runs unattended for days; a kernel update or a power blip must
# not be the thing that stops both cards.  pueue's state file already survives
# (queued tasks, groups and parallelism all come back -- verified), so the only
# piece that needs reviving is the daemon process.
#
# WHY NOT `@reboot` CRONTAB (the originally requested mechanism):
#   /var/spool/cron/crontabs on this box is `drwx-wx--T root:crontab` and the
#   existing `crontabs/bc` is not owned by `bc`.  `crontab -` writes a temp file
#   and renames it over that one; the directory's sticky bit forbids renaming
#   over a file you do not own, so it fails with
#       crontab: crontabs/bc: rename: Operation not permitted
#   and `sudo` on this box wants a password.  Installing the cron line therefore
#   cannot be done unattended.  See --cron-fallback below for the one root
#   command that would enable it, if cron is ever preferred.
#
# WHAT THIS DOES INSTEAD -- a systemd *user* service, which needs no root:
#   * `loginctl enable-linger bc` is permitted for one's own user by polkit
#     (verified: rc=0, Linger=yes), and linger is exactly what makes user units
#     start at boot with nobody logged in;
#   * the unit is upstream's own `systemd.pueued.service` from the v4.0.4
#     release, with two deliberate changes: the binary path points at
#     ~/.local/bin/pueued (we install per-user, not to /usr/bin), and
#     Restart=on-failure replaces upstream's Restart=no -- a daemon that dies at
#     3am should come back, and restart is safe because the state file is
#     authoritative.
#
# This is strictly stronger than the cron line it replaces: cron would only
# have covered reboots, this also covers the daemon dying mid-campaign.
#
#   install:  tools/queue/install_autostart.sh
#   check:    tools/queue/install_autostart.sh --check
#   remove:   tools/queue/install_autostart.sh --remove
# =============================================================================
set -uo pipefail

UNIT_DIR="${HOME}/.config/systemd/user"
UNIT="${UNIT_DIR}/pueued.service"
PUEUED="${PUEUED_BIN:-${HOME}/.local/bin/pueued}"
Q="${Q_PATH:-/home/bc/VeraRetouch/tools/queue/q}"

case "${1:-install}" in
  --check)
    echo "linger:  $(loginctl show-user "$(id -un)" 2>/dev/null | grep -i '^Linger=' || echo 'Linger=?')"
    echo "enabled: $(systemctl --user is-enabled pueued.service 2>&1)"
    echo "active:  $(systemctl --user is-active pueued.service 2>&1)"
    exit 0
    ;;

  --cron-fallback)
    cat <<'EOF'
Cron is only reachable through root on this box.  If you ever prefer it to the
user unit, one root command hands the crontab back to its owner:

    sudo chown bc:crontab /var/spool/cron/crontabs/bc

after which this line installs (and `q daemon` is itself a no-op when the
daemon is already up, so re-running it is harmless):

    (crontab -l 2>/dev/null; \
     echo '# VeraRetouch-QUEUE-1: keep the GPU queue daemon alive across reboots'; \
     echo '@reboot /home/bc/VeraRetouch/tools/queue/q daemon >> /home/bc/data/queue/reboot.log 2>&1' \
    ) | crontab -
EOF
    exit 0
    ;;

  --remove)
    systemctl --user disable --now pueued.service 2>&1 | sed 's/^/  /'
    rm -f "${UNIT}"
    systemctl --user daemon-reload
    echo "removed (linger left enabled; disable with: loginctl disable-linger $(id -un))"
    exit 0
    ;;

  install) ;;
  *) echo "usage: install_autostart.sh [install|--check|--remove|--cron-fallback]" >&2; exit 2 ;;
esac

[ -x "${PUEUED}" ] || { echo "refusing: ${PUEUED} is not executable" >&2; exit 1; }

# 1. linger -- without it a user unit only starts at login, which is precisely
#    the session-bound failure mode this whole tool exists to remove.
loginctl enable-linger "$(id -un)" 2>/dev/null
linger="$(loginctl show-user "$(id -un)" 2>/dev/null | sed -n 's/^Linger=//p')"
if [ "${linger}" != "yes" ]; then
  echo "FAILED: could not enable linger (got '${linger:-unknown}')." >&2
  echo "Without it the daemon will not start until someone logs in." >&2
  echo "Run as root:  loginctl enable-linger $(id -un)" >&2
  exit 1
fi

# 2. the unit itself.  Writing it unconditionally keeps this idempotent and
#    also repairs a hand-edited or stale copy.
mkdir -p "${UNIT_DIR}"
cat > "${UNIT}" <<EOF
# Installed by tools/queue/install_autostart.sh (VeraRetouch QUEUE-1).
# Based on upstream's systemd.pueued.service, pueue v4.0.4, with a per-user
# binary path and Restart=on-failure (see the script header).
[Unit]
Description=Pueue Daemon - GPU task queue for the VeraRetouch campaign
After=default.target

[Service]
Type=exec
ExecStart=${PUEUED} -vv
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload

# 3. Hand the running daemon over to systemd.  A pueued started earlier by
#    `q daemon` is not a child of the user manager, so systemd cannot supervise
#    it and `start` would fail on the socket.  Shut it down first; the state
#    file makes this lossless for queued work.  Anything RUNNING would be
#    killed, so refuse in that case rather than silently murdering a training
#    job -- installing autostart must never cost an arm.
if "${HOME}/.local/bin/pueue" status --json >/dev/null 2>&1; then
  running="$("${HOME}/.local/bin/pueue" status --json \
             | jq '[.tasks[] | select(.status|type=="object" and has("Running"))] | length' 2>/dev/null || echo 0)"
  if [ "${running:-0}" -gt 0 ] && [ "${FORCE_TAKEOVER:-0}" != "1" ]; then
    echo "NOT taking over: ${running} task(s) are running under the current daemon."
    echo "The unit is installed and enabled, so it will take effect at the next reboot."
    systemctl --user enable pueued.service 2>&1 | sed 's/^/  /'
    echo
    echo "To hand over right now (kills those running tasks):"
    echo "    FORCE_TAKEOVER=1 $0"
    exit 0
  fi
  echo "stopping the manually started daemon (queued work survives in state.json)"
  "${HOME}/.local/bin/pueue" shutdown >/dev/null 2>&1
  sleep 3
fi

systemctl --user enable --now pueued.service 2>&1 | sed 's/^/  /'
sleep 3

# 4. Verify.  Never report success on the strength of having tried -- the first
#    draft of this script printed "installed:" after crontab had already failed.
ok=1
[ "$(systemctl --user is-enabled pueued.service 2>&1)" = "enabled" ] || { echo "FAILED: unit is not enabled" >&2; ok=0; }
[ "$(systemctl --user is-active  pueued.service 2>&1)" = "active"  ] || { echo "FAILED: unit is not active"  >&2; ok=0; }
"${HOME}/.local/bin/pueue" status --json >/dev/null 2>&1 || { echo "FAILED: daemon does not answer" >&2; ok=0; }
[ "${ok}" -eq 1 ] || { systemctl --user status pueued.service --no-pager 2>&1 | head -20; exit 1; }

bash "${Q}" daemon >/dev/null 2>&1     # re-assert the two card groups
echo "installed and verified:"
echo "  linger : yes"
echo "  unit   : ${UNIT}"
echo "  enabled: $(systemctl --user is-enabled pueued.service)"
echo "  active : $(systemctl --user is-active pueued.service)"
bash "${Q}" status --json | jq -c '{daemon_ok:.daemon_ok, groups:(.groups|keys), tasks:(.tasks|length)}'
