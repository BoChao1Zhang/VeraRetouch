#!/usr/bin/env python3
"""qstatus -- the whole GPU queue in one command, for a Claude session that
has just woken up and knows nothing.

This is the status *contract*: one invocation, under two seconds, no follow-up
questions needed.  It joins three sources that nobody should have to join by
hand at 3am:

  * ``pueue status --json``  -- what the scheduler thinks (queue order, groups,
    exit codes, dependencies).  Its ``envs`` blob is dropped on the floor: it
    is a copy of the submitting shell's environment, it is enormous, and it
    contains API keys.
  * ``<QUEUE_HOME>/status/<label>.json`` -- what ``qjob.sh`` observed (gate
    state, real payload pid, the payload's own log path, failure signatures,
    last 30 log lines).  pueue only knows about the wrapper.
  * ``nvidia-smi`` -- whether the cards are actually busy, which is the one
    claim neither of the other two can make.  A queue that says "running"
    while a card sits at 0% is exactly the failure this tool exists to expose.

Markdown by default (for a human or an agent reading the transcript),
``--json`` for programmatic use.

Usage:
    tools/queue/qstatus.py            # markdown
    tools/queue/qstatus.py --json     # machine readable
    tools/queue/qstatus.py --tail 5   # more log context per job
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

QUEUE_HOME = Path(os.environ.get("QUEUE_HOME", "/home/bc/data/queue"))
STATUS_DIR = QUEUE_HOME / "status"
GPU_GROUPS = ("gpu0", "gpu1")
PUEUE = os.environ.get("PUEUE_BIN", str(Path.home() / ".local/bin/pueue"))


def run(cmd, timeout=10):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        return 127, "", str(exc)


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------
def pueue_state():
    rc, out, err = run([PUEUE, "status", "--json"])
    if rc != 0:
        return None, (err.strip().splitlines() or ["pueue unreachable"])[0]
    try:
        return json.loads(out), None
    except json.JSONDecodeError as exc:
        return None, f"unparseable pueue status: {exc}"


def gpu_state():
    if not shutil.which("nvidia-smi"):
        return []
    rc, out, _ = run(["nvidia-smi",
                      "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                      "--format=csv,noheader,nounits"])
    if rc != 0:
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            gpus.append({"index": int(parts[0]), "util_pct": int(parts[1]),
                         "mem_used_mib": int(parts[2]), "mem_total_mib": int(parts[3])})
        except ValueError:
            continue
    return gpus


def declared_gates(command):
    """Gates a task *declares*, recovered from the queued qjob command line.

    qjob.sh only writes its status file once it starts, so a job still sitting
    in the queue would otherwise report "no gate" when it in fact has several.
    Reading them back off the command keeps the queued rows honest.
    """
    if not command:
        return []
    return [{"path": p, "ok": Path(p).exists()}
            for p in re.findall(r"--gate\s+(\S+)", command)]


def job_records():
    """The qjob-written side of the story, keyed by job name (== pueue label)."""
    recs = {}
    if not STATUS_DIR.is_dir():
        return recs
    for path in sorted(STATUS_DIR.glob("*.json")):
        try:
            recs[path.stem] = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
    return recs


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------
def decode_status(status):
    """pueue's status enum -> (state, result, rc, times).

    It is either a bare string ("Queued", "Stashed") or a single-key object
    ({"Done": {...}}, {"Running": {...}}).
    """
    if isinstance(status, str):
        return status, None, None, {}
    if not isinstance(status, dict) or not status:
        return "Unknown", None, None, {}
    state = next(iter(status))
    body = status[state] if isinstance(status[state], dict) else {}
    result, rc = None, None
    raw = body.get("result")
    if isinstance(raw, str):
        result = raw
        rc = 0 if raw == "Success" else None
    elif isinstance(raw, dict) and raw:
        result = next(iter(raw))
        val = raw[result]
        if isinstance(val, int):
            rc = val
    return state, result, rc, body


def parse_ts(value):
    """pueue stamps nanoseconds; datetime.fromisoformat before 3.11 accepts at
    most microseconds, so every timestamp has to be trimmed to 6 digits or the
    whole elapsed column silently reads '-'."""
    if not value:
        return None
    for candidate in (value, re.sub(r"(\.\d{6})\d+", r"\1", value)):
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            continue
    return None


def human_dt(seconds):
    if seconds is None:
        return "-"
    seconds = int(seconds)
    h, rem = divmod(max(seconds, 0), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def elapsed_of(times, state):
    start = parse_ts(times.get("start"))
    if not start:
        return None
    end = parse_ts(times.get("end")) if state == "Done" else datetime.now(timezone.utc).astimezone()
    if not end:
        return None
    return (end - start).total_seconds()


def collect():
    state, err = pueue_state()
    gpus = gpu_state()
    jobs = job_records()

    tasks = []
    for raw in (state or {}).get("tasks", {}).values():
        st, result, rc, times = decode_status(raw.get("status"))
        label = raw.get("label") or ""
        rec = jobs.get(label, {})
        tasks.append({
            "id": raw.get("id"),
            "label": label,
            "group": raw.get("group"),
            "state": st,
            "result": result,
            "rc": rc if rc is not None else rec.get("rc"),
            "dependencies": raw.get("dependencies") or [],
            "elapsed_s": elapsed_of(times, st),
            "start": times.get("start"),
            "end": times.get("end"),
            # deliberately not raw["command"]: it is the qjob invocation, which
            # is long and mostly boilerplate.  The payload is what matters.
            "payload": rec.get("cmd") or raw.get("command"),
            "phase": rec.get("phase"),
            "pid": rec.get("pid"),
            "log": rec.get("log"),
            "note": rec.get("note"),
            "gates": rec.get("gates") or declared_gates(raw.get("command")),
            "failure_signatures": rec.get("failure_signatures") or [],
            "log_tail": rec.get("log_tail") or [],
        })
    tasks.sort(key=lambda t: (t["id"] if t["id"] is not None else 1 << 30))

    groups = {name: {"parallel": g.get("parallel_tasks"), "status": g.get("status")}
              for name, g in (state or {}).get("groups", {}).items()}

    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "daemon_ok": state is not None,
        "daemon_error": err,
        "groups": groups,
        "gpus": gpus,
        "tasks": tasks,
    }


# ---------------------------------------------------------------------------
# warnings -- the things a waking agent must not have to deduce
# ---------------------------------------------------------------------------
def warnings_for(snap):
    out = []
    if not snap["daemon_ok"]:
        out.append(f"pueued is NOT running ({snap['daemon_error']}). "
                   f"Nothing will start. Fix: tools/queue/q daemon")
        return out
    for g in GPU_GROUPS:
        info = snap["groups"].get(g)
        if info is None:
            out.append(f"group '{g}' does not exist -- run: tools/queue/q daemon")
        elif info.get("status") == "Paused":
            out.append(f"group '{g}' is PAUSED; queued jobs there will not start "
                       f"(resume: tools/queue/q resume {g})")

    running = [t for t in snap["tasks"] if t["state"] == "Running"]
    queued = [t for t in snap["tasks"] if t["state"] in ("Queued", "Stashed")]
    gpu_by_idx = {g["index"]: g for g in snap["gpus"]}

    for t in running:
        idx = {"gpu0": 0, "gpu1": 1}.get(t["group"])
        gpu = gpu_by_idx.get(idx) if idx is not None else None
        if t["phase"] == "gate-wait":
            out.append(f"[{t['group']}] {t['label']} is WAITING ON A GATE: {t['note']}. "
                       f"The card is idle until it opens.")
        elif gpu and gpu["util_pct"] == 0 and gpu["mem_used_mib"] < 2000 and (t["elapsed_s"] or 0) > 900:
            out.append(f"[{t['group']}] {t['label']} has been 'running' for "
                       f"{human_dt(t['elapsed_s'])} but GPU{idx} is at 0% / "
                       f"{gpu['mem_used_mib']}MiB. Check {t.get('log')}")

    for g in GPU_GROUPS:
        busy = any(t["group"] == g for t in running)
        waiting = [t for t in queued if t["group"] == g]
        if not busy and waiting:
            out.append(f"[{g}] IDLE with {len(waiting)} job(s) queued -- "
                       f"group paused, or a dependency is unmet.")
        if not busy and not waiting:
            out.append(f"[{g}] IDLE and the queue is EMPTY. Enqueue more work.")

    for t in snap["tasks"]:
        if t["state"] == "Done" and t["result"] not in (None, "Success"):
            sig = ", ".join(s["signature"] for s in t["failure_signatures"]) or "no known signature"
            out.append(f"FAILED: {t['label']} (id {t['id']}, rc={t['rc']}, "
                       f"{t['result']}) -- {sig}")
    return out


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------
def render(snap, tail_lines):
    L = []
    A = L.append
    A(f"# GPU queue @ {snap['generated_at']}")
    A("")

    warns = warnings_for(snap)
    if warns:
        A("## Attention")
        for w in warns:
            A(f"- {w}")
        A("")

    A("## Cards")
    A("")
    A("| card | GPU util | GPU mem | running | pueue id | payload pid | elapsed | phase |")
    A("|---|---|---|---|---|---|---|---|")
    gpu_by_idx = {g["index"]: g for g in snap["gpus"]}
    for idx, group in enumerate(GPU_GROUPS):
        gpu = gpu_by_idx.get(idx, {})
        util = f"{gpu.get('util_pct', '?')}%"
        mem = (f"{gpu.get('mem_used_mib', 0) // 1024}/"
               f"{gpu.get('mem_total_mib', 0) // 1024} GiB" if gpu else "?")
        run_t = next((t for t in snap["tasks"]
                      if t["group"] == group and t["state"] == "Running"), None)
        if run_t:
            A(f"| {group} | {util} | {mem} | **{run_t['label']}** | {run_t['id']} | "
              f"{run_t['pid'] or '-'} | {human_dt(run_t['elapsed_s'])} | "
              f"{run_t['phase'] or '-'} |")
        else:
            A(f"| {group} | {util} | {mem} | _(idle)_ | - | - | - | - |")
    A("")

    queued = [t for t in snap["tasks"] if t["state"] in ("Queued", "Stashed")]
    A(f"## Queued ({len(queued)})")
    A("")
    if not queued:
        A("_nothing queued_")
    else:
        A("| pos | id | label | group | state | waits for | gate |")
        A("|---|---|---|---|---|---|---|")
        for pos, t in enumerate(queued, 1):
            deps = ",".join(str(d) for d in t["dependencies"]) or "-"
            gates = t["gates"]
            gate = "-" if not gates else \
                f"{sum(1 for g in gates if g['ok'])}/{len(gates)} present"
            A(f"| {pos} | {t['id']} | {t['label']} | {t['group']} | {t['state']} | "
              f"{deps} | {gate} |")
    A("")

    done = [t for t in snap["tasks"] if t["state"] == "Done"]
    A(f"## Finished ({len(done)})")
    A("")
    if not done:
        A("_nothing finished yet_")
    else:
        A("| id | label | group | result | rc | duration | signatures |")
        A("|---|---|---|---|---|---|---|")
        for t in done:
            sigs = ", ".join(s["signature"] for s in t["failure_signatures"]) or "-"
            A(f"| {t['id']} | {t['label']} | {t['group']} | {t['result']} | "
              f"{t['rc'] if t['rc'] is not None else '-'} | "
              f"{human_dt(t['elapsed_s'])} | {sigs} |")
    A("")

    live = [t for t in snap["tasks"] if t["state"] == "Running"]
    interesting = live + [t for t in done
                          if t["result"] not in (None, "Success")][-3:]
    if interesting:
        A("## Log tails")
        A("")
        for t in interesting:
            A(f"### {t['label']} ({t['state']}/{t['result'] or t['phase']}) "
              f"-- `{t['log'] or 'no log'}`")
            A("")
            if t["note"]:
                A(f"note: {t['note']}")
                A("")
            body = t["log_tail"][-tail_lines:] if t["log_tail"] else ["(no output captured)"]
            A("```")
            for line in body:
                A(line[:300])
            A("```")
            A("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--tail", type=int, default=3,
                    help="log lines to show per running/failed job (default 3)")
    args = ap.parse_args()

    snap = collect()
    if args.json:
        snap["warnings"] = warnings_for(snap)
        json.dump(snap, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
    else:
        print(render(snap, args.tail))
    return 0 if snap["daemon_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
