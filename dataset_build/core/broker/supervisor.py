"""vGate supervisor — elastic vLLM replica lifecycle (Phase 1).

A small daemon (NOT on the request path) that replaces ``launch_reasoning.sh``'s
all-or-nothing start with adaptive, per-card management. Every ~15s it polls
``nvidia-smi`` + ``docker ps`` and ensures one vLLM replica per *eligible* GPU:

    GPU0 -> reason_g0 : 8001     GPU1 -> reason_g1 : 8002   (configurable)

This is the actuation half of the 1<->2-card elasticity: the broker discovers
whatever replicas exist and scales its admission budget; the supervisor brings a
replica *up* on a card that is free (no foreign load) and has no replica yet. It
reuses ``launch_reasoning.sh start-one`` verbatim, so the docker run template and
the persistent torch.compile / CUDA-graph cache mount stay a single source of
truth (cache-backed restarts skip the ~30min recompile).

Safety is deliberate and conservative:
  * **Dry-run by default.** Nothing is started/stopped unless ``--apply`` (or
    ``VGATE_SUP_APPLY=1``). Dry-run logs exactly what it *would* do.
  * **Up-scale only.** It starts replicas on free cards and restarts *crashed*
    (exited) ones. It never tears down a healthy or still-warming replica
    (a container that is Up but not yet serving /v1/models is assumed to be
    compiling and is left alone). Down-scaling for a heavy IQA pass is an
    explicit future signal, not an automatic behavior.
  * **Min-dwell hysteresis** between actions on the same container prevents
    flapping.
  * **Free-card gate**: a card is eligible to host a *new* replica only if its
    free memory exceeds ``--min-free-mb`` (no other tenant is using it).

Pure stdlib + ``urllib`` for the readiness probe; no torch, no GPU compute.

Run (from the repo root)::

    # observe only (safe): prints intended actions
    /home/bc/miniconda3/bin/python -m dataset_build.core.broker.supervisor --once
    # actuate:
    /home/bc/miniconda3/bin/python -m dataset_build.core.broker.supervisor --apply
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("vgate.supervisor")

# Default GPU -> (container name, host port). Mirrors launch_reasoning.sh.
DEFAULT_GPU_MAP: Dict[int, Tuple[str, int]] = {
    0: ("reason_g0", 8001),
    1: ("reason_g1", 8002),
}
LAUNCH_SCRIPT = "dataset_build/docker/launch_reasoning.sh"
REPO_ROOT = "/home/bc/VeraRetouch"


@dataclass
class GpuInfo:
    index: int
    mem_total_mb: float
    mem_used_mb: float
    util_pct: float

    @property
    def mem_free_mb(self) -> float:
        return max(0.0, self.mem_total_mb - self.mem_used_mb)


@dataclass
class SupervisorConfig:
    gpu_map: Dict[int, Tuple[str, int]] = field(default_factory=lambda: dict(DEFAULT_GPU_MAP))
    served_name: str = "qwen3_5-35b-a3b"
    interval: float = 15.0
    min_free_mb: float = 70_000.0      # a 35B fp8 replica needs ~0.85 of an H100
    min_dwell: float = 60.0            # seconds between actions on one container
    apply: bool = False                # False => dry-run (log only)
    probe_timeout: float = 3.0
    launch_script: str = LAUNCH_SCRIPT
    repo_root: str = REPO_ROOT


# --- system probes ----------------------------------------------------------
def query_gpus() -> Dict[int, GpuInfo]:
    """Parse ``nvidia-smi`` into ``{index: GpuInfo}``; empty on any failure."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,memory.total,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except Exception as exc:
        logger.warning("nvidia-smi unavailable: %s", exc)
        return {}
    gpus: Dict[int, GpuInfo] = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            idx = int(parts[0])
            gpus[idx] = GpuInfo(idx, float(parts[1]), float(parts[2]), float(parts[3]))
        except ValueError:
            continue
    return gpus


def running_containers() -> Dict[str, str]:
    """Map container name -> status string for currently *running* containers."""
    try:
        out = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}\t{{.Status}}"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except Exception as exc:
        logger.warning("docker ps failed: %s", exc)
        return {}
    result: Dict[str, str] = {}
    for line in out.strip().splitlines():
        if "\t" in line:
            name, status = line.split("\t", 1)
            result[name.strip()] = status.strip()
    return result


def replica_healthy(port: int, served_name: str, timeout: float) -> bool:
    """True iff the replica on ``port`` serves the expected model id."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=timeout) as r:
            payload = json.loads(r.read().decode("utf-8"))
    except Exception:
        return False
    data = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0].get("id") == served_name
    return False


# --- supervisor -------------------------------------------------------------
class Supervisor:
    def __init__(self, cfg: SupervisorConfig) -> None:
        self.cfg = cfg
        self._last_action: Dict[str, float] = {}  # container -> monotonic time

    def _dwell_ok(self, name: str) -> bool:
        last = self._last_action.get(name, 0.0)
        return (time.monotonic() - last) >= self.cfg.min_dwell

    def _run_launch(self, *args: str) -> None:
        cmd = ["bash", self.cfg.launch_script, *args]
        if not self.cfg.apply:
            logger.info("[dry-run] would run: %s", " ".join(cmd))
            return
        logger.info("running: %s", " ".join(cmd))
        try:
            subprocess.run(cmd, cwd=self.cfg.repo_root, check=True, timeout=120)
        except Exception as exc:
            logger.error("launch command failed (%s): %s", " ".join(cmd), exc)

    def start_replica(self, name: str, gpu: int, port: int) -> None:
        self._last_action[name] = time.monotonic()
        self._run_launch("start-one", name, str(gpu), str(port))

    def tick(self) -> dict:
        """One reconciliation pass. Returns a compact status dict for logging."""
        gpus = query_gpus()
        running = running_containers()
        cfg = self.cfg
        report: List[dict] = []

        for gpu_idx, (name, port) in sorted(cfg.gpu_map.items()):
            gpu = gpus.get(gpu_idx)
            is_running = name in running
            healthy = is_running and replica_healthy(port, cfg.served_name, cfg.probe_timeout)
            state = "absent-gpu" if gpu is None else (
                "healthy" if healthy else ("warming" if is_running else "down"))
            action = "none"

            if gpu is None:
                # Card not present on this host (e.g. 1-card machine): skip.
                action = "skip-no-gpu"
            elif healthy:
                action = "none"
            elif is_running:
                # Up but not serving yet -> assume compiling/warming; never kill.
                action = "leave-warming"
            else:
                # Not running (never started or crashed/exited). Start iff the
                # card is genuinely free and dwell has elapsed.
                if gpu.mem_free_mb < cfg.min_free_mb:
                    action = f"hold-card-busy(free={gpu.mem_free_mb:.0f}MB<{cfg.min_free_mb:.0f})"
                elif not self._dwell_ok(name):
                    action = "hold-dwell"
                else:
                    action = "start"
                    self.start_replica(name, gpu_idx, port)

            report.append({
                "gpu": gpu_idx, "replica": name, "port": port, "state": state,
                "free_mb": (round(gpu.mem_free_mb) if gpu else None),
                "util": (gpu.util_pct if gpu else None), "action": action,
            })

        live = sum(1 for r in report if r["state"] == "healthy")
        logger.info("tick: healthy=%d/%d %s", live, len(cfg.gpu_map),
                    "; ".join(f"{r['replica']}@{r['gpu']}:{r['state']}->{r['action']}" for r in report))
        return {"healthy": live, "replicas": report, "apply": cfg.apply}

    def run_forever(self) -> None:
        logger.info("supervisor up: interval=%.0fs apply=%s gpu_map=%s min_free=%.0fMB",
                    self.cfg.interval, self.cfg.apply, self.cfg.gpu_map, self.cfg.min_free_mb)
        while True:
            try:
                self.tick()
            except Exception as exc:
                logger.error("tick error: %s", exc)
            time.sleep(self.cfg.interval)


def _parse_gpu_map(spec: str) -> Dict[int, Tuple[str, int]]:
    """Parse ``0:reason_g0:8001,1:reason_g1:8002`` -> map."""
    out: Dict[int, Tuple[str, int]] = {}
    for entry in spec.replace(" ", "").split(","):
        if not entry:
            continue
        gpu, name, port = entry.split(":")
        out[int(gpu)] = (name, int(port))
    return out


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(prog="dataset_build.core.broker.supervisor",
                                description="Elastic vLLM replica supervisor for vGate")
    p.add_argument("--apply", action="store_true", default=os.environ.get("VGATE_SUP_APPLY") == "1",
                   help="actually start/stop containers (default: dry-run, log only)")
    p.add_argument("--once", action="store_true", help="run a single reconciliation tick and exit")
    p.add_argument("--interval", type=float, default=float(os.environ.get("VGATE_SUP_INTERVAL", 15.0)))
    p.add_argument("--min-free-mb", type=float, default=float(os.environ.get("VGATE_SUP_MIN_FREE_MB", 70000.0)))
    p.add_argument("--min-dwell", type=float, default=float(os.environ.get("VGATE_SUP_MIN_DWELL", 60.0)))
    p.add_argument("--served-name", default=os.environ.get("VGATE_SERVED_NAME", "qwen3_5-35b-a3b"))
    p.add_argument("--gpu-map", default=os.environ.get("VGATE_SUP_GPU_MAP", ""),
                   help="override GPU map, e.g. '0:reason_g0:8001,1:reason_g1:8002'")
    p.add_argument("--log-level", default=os.environ.get("VGATE_LOG_LEVEL", "info"))
    args = p.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = SupervisorConfig(
        gpu_map=(_parse_gpu_map(args.gpu_map) if args.gpu_map else dict(DEFAULT_GPU_MAP)),
        served_name=args.served_name,
        interval=args.interval,
        min_free_mb=args.min_free_mb,
        min_dwell=args.min_dwell,
        apply=args.apply,
    )
    sup = Supervisor(cfg)
    if not cfg.apply:
        logger.info("DRY-RUN mode (pass --apply or VGATE_SUP_APPLY=1 to actuate)")
    if args.once:
        sup.tick()
    else:
        sup.run_forever()


if __name__ == "__main__":
    main()
