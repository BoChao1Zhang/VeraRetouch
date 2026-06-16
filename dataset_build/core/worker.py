"""In-container core worker HTTP server (unified-image topology).

Exposes the same-card GPU-compute resources of one unified container over
localhost HTTP so a co-located business shard (or any consumer in the container)
can call them without importing torch/SAM3/pyiqa itself:

  * ``POST /iqa/score``   -> NR-IQA scores dict for one image (pyiqa).
  * ``POST /sam3/masks``  -> live SAM3 masks for an image, written to an FS drop
    dir; returns ``{concept: png_path}`` (FS-drop, NOT inline binary — passing
    raw mask bytes over stdout/HTTP is the fragile path the v2 design §8 rejects).

A single per-process GPU lease (``GpuCompute``) serializes IQA and SAM3 so they
never run concurrently on the card. Models load lazily on first use; if a model
cannot load in this env (e.g. SAM3 needs transformers 5.x), its endpoint returns
503 while the rest of the server stays up. Pure facade — the actual scoring /
masking reuses ``source_qa.iqa.IQARunner`` and ``masking.Sam3Masker`` unchanged.

This server is only needed when business and core cross *process* boundaries
inside the unified image. The in-process ``core.iqa`` / ``core.sam3`` clients
remain the default path; this is their out-of-process twin.

Run (inside the unified container)::

    python -m dataset_build.core.worker --port 8010 --device cuda:0 \
        --sam3-model-dir /home/bc/data/models
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
from typing import Any, Dict, List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .gpu_compute import GpuCompute

logger = logging.getLogger("vgate.worker")


class IqaScoreRequest(BaseModel):
    path: str
    want_face: bool = False


class Sam3MaskRequest(BaseModel):
    image_path: str
    concepts: List[str]
    drop_dir: Optional[str] = None          # where to write mask PNGs (FS-drop)
    native_size: Optional[List[int]] = None  # [H, W] hint


def build_app(device: str = "cuda:0", sam3_model_dir: str = "/home/bc/data/models",
              default_drop_dir: str = "/home/bc/data/datasets/_core_worker_masks",
              iqa_metrics: Optional[List[str]] = None) -> FastAPI:
    app = FastAPI(title="vGate core worker", version="0.1.0")
    gpu = GpuCompute([device])
    state: Dict[str, Any] = {"iqa": None, "sam3": None, "iqa_err": None, "sam3_err": None}
    load_lock = threading.Lock()

    def _iqa():
        if state["iqa"] is None and state["iqa_err"] is None:
            with load_lock:
                if state["iqa"] is None and state["iqa_err"] is None:
                    try:
                        from dataset_build.source_qa.iqa import IQARunner
                        state["iqa"] = IQARunner(device=device, metrics=iqa_metrics)
                        logger.info("IQA (pyiqa) loaded on %s", device)
                    except Exception as e:  # pragma: no cover - env dependent
                        state["iqa_err"] = f"{type(e).__name__}: {e}"
                        logger.error("IQA load failed: %s", state["iqa_err"])
        return state["iqa"]

    def _sam3():
        if state["sam3"] is None and state["sam3_err"] is None:
            with load_lock:
                if state["sam3"] is None and state["sam3_err"] is None:
                    try:
                        from dataset_build.masking import Sam3Masker
                        m = Sam3Masker(model_dir=sam3_model_dir, device=device)
                        # Force the model load now so an env that can't load SAM3
                        # (e.g. transformers <5) reports 503 here, not 500 per call.
                        m._ensure_loaded()
                        state["sam3"] = m
                        logger.info("SAM3 masker loaded on %s", device)
                    except Exception as e:  # pragma: no cover - env dependent
                        state["sam3_err"] = f"{type(e).__name__}: {e}"
                        logger.error("SAM3 load failed: %s", state["sam3_err"])
        return state["sam3"]

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "device": device,
                "iqa_loaded": state["iqa"] is not None,
                "sam3_loaded": state["sam3"] is not None}

    @app.post("/iqa/score")
    async def iqa_score(req: IqaScoreRequest):
        runner = _iqa()
        if runner is None:
            return JSONResponse({"error": "iqa unavailable", "detail": state["iqa_err"]},
                                status_code=503)
        try:
            with gpu.lease(device):
                scores = runner.score_path(req.path, want_face=req.want_face)
            return {"path": req.path, "scores": scores}
        except Exception as e:
            return JSONResponse({"error": "iqa score failed", "detail": f"{type(e).__name__}: {e}"},
                                status_code=500)

    @app.post("/sam3/masks")
    async def sam3_masks(req: Sam3MaskRequest):
        masker = _sam3()
        if masker is None:
            return JSONResponse({"error": "sam3 unavailable", "detail": state["sam3_err"]},
                                status_code=503)
        drop = req.drop_dir or default_drop_dir
        os.makedirs(drop, exist_ok=True)
        native = tuple(req.native_size) if req.native_size else None
        try:
            with gpu.lease(device):
                masks = masker.masks(req.image_path, list(req.concepts), native_size=native)
        except Exception as e:
            return JSONResponse({"error": "sam3 masks failed", "detail": f"{type(e).__name__}: {e}"},
                                status_code=500)
        # FS-drop: write each mask PNG, return paths (never inline binary).
        import hashlib
        base = hashlib.sha1(req.image_path.encode("utf-8", "surrogatepass")).hexdigest()[:16]
        out: Dict[str, Optional[str]] = {}
        for concept, arr in (masks or {}).items():
            if arr is None:
                out[concept] = None
                continue
            try:
                import cv2
                a = np.asarray(arr)
                if a.dtype != np.uint8:
                    a = (np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8)
                safe = "".join(ch if ch.isalnum() else "_" for ch in concept)[:40]
                p = os.path.join(drop, f"{base}__{safe}.png")
                cv2.imwrite(p, a)
                out[concept] = p
            except Exception as e:
                logger.error("mask write failed for %s: %s", concept, e)
                out[concept] = None
        return {"image_path": req.image_path, "masks": out, "drop_dir": drop}

    return app


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(prog="dataset_build.core.worker",
                                description="vGate in-container core worker (iqa/sam3 over HTTP)")
    p.add_argument("--host", default=os.environ.get("VGATE_WORKER_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("VGATE_WORKER_PORT", "8010")))
    p.add_argument("--device", default=os.environ.get("VGATE_WORKER_DEVICE", "cuda:0"))
    p.add_argument("--sam3-model-dir", default=os.environ.get("VGATE_SAM3_MODEL_DIR", "/home/bc/data/models"))
    p.add_argument("--drop-dir", default=os.environ.get("VGATE_WORKER_DROP_DIR",
                                                        "/home/bc/data/datasets/_core_worker_masks"))
    p.add_argument("--metrics", default=os.environ.get("VGATE_WORKER_IQA_METRICS", ""),
                   help="comma list of IQA metrics (default: source_qa config metrics)")
    p.add_argument("--log-level", default=os.environ.get("VGATE_LOG_LEVEL", "info"))
    args = p.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    metrics = [m for m in args.metrics.replace(",", " ").split() if m] or None
    app = build_app(device=args.device, sam3_model_dir=args.sam3_model_dir,
                    default_drop_dir=args.drop_dir, iqa_metrics=metrics)
    logger.info("core worker on %s:%d device=%s", args.host, args.port, args.device)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
