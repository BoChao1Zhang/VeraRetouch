"""NR-IQA runner (GPU0): MUSIQ + CLIP-IQA+ + NIQE + BRISQUE (pyiqa) + cv2 Laplacian
sharpness, PLUS cheap deterministic detectors the gate was missing (closes QA-1/
QA-6), all from established libraries:
  * native width/height/megapixels  -- from the decode (the index has them NULL)
  * noise_sigma                       -- skimage.restoration.estimate_sigma
  * max_face_frac (portrait pool)     -- cv2 Haar cascade, largest face / frame

Scores every `image` asset lacking a 'musiq' score, writes long-format iqa_scores
+ denormalized headline columns on `assets`, logs an `iqa` event.

Run: python -m dataset_build.source_qa.iqa [--limit N] [--corpus C] [--device cuda:0]
"""
from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional

from . import config, db

# metric name -> denormalized assets column
_COL = {"musiq": "musiq", "clipiqa+": "clipiqa", "niqe": "niqe",
        "brisque": "brisque", "laplacian": "sharpness", "noise_sigma": "noise_sigma",
        "max_face_frac": "max_face_frac", "megapixels": "megapixels",
        "width": "width", "height": "height"}

_FACE_CASCADE = None


def _face_cascade():
    """cv2's bundled frontal-face Haar cascade (no model download needed)."""
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        import cv2
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _FACE_CASCADE = cv2.CascadeClassifier(path)
    return _FACE_CASCADE


class IQARunner:
    def __init__(self, device: str = None, metrics: List[str] = None):
        import torch
        import pyiqa
        self.torch = torch
        self.device = device or config.IQA_DEVICE
        self.metrics = metrics or config.IQA_METRICS
        self.models = {}
        for m in self.metrics:
            if m in ("laplacian", "noise_sigma", "max_face_frac", "megapixels", "longedge"):
                continue  # not pyiqa metrics; computed directly in score_path
            try:
                self.models[m] = pyiqa.create_metric(m, device=self.device)
                self.models[m].eval()
            except Exception as e:
                print(f"[iqa] failed to create metric {m}: {e}", file=sys.stderr)

    def _tensor(self, path: str):
        import numpy as np
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        im = Image.open(path)
        im.load()
        im = im.convert("RGB")
        native_w, native_h = im.size            # native size BEFORE downscale (QA-1)
        le = config.IQA_LONGEDGE
        w, h = im.size
        if max(w, h) > le:
            s = le / max(w, h)
            im = im.resize((max(1, int(w * s)), max(1, int(h * s))))
        arr = np.asarray(im).astype("float32") / 255.0      # HWC
        t = self.torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return t, im, (native_w, native_h)

    def score_path(self, path: str, want_face: bool = False) -> Dict[str, Optional[float]]:
        import numpy as np
        out: Dict[str, Optional[float]] = {}
        t, im, (nw, nh) = self._tensor(path)
        out["width"] = float(nw)
        out["height"] = float(nh)
        out["megapixels"] = round(nw * nh / 1e6, 4)
        out["longedge"] = float(max(nw, nh))
        with self.torch.no_grad():
            for m, model in self.models.items():
                try:
                    v = model(t)
                    out[m] = float(v.flatten()[0].item())
                except Exception as e:
                    out[m] = None
                    print(f"[iqa] {m} failed on {path}: {e}", file=sys.stderr)
        gray = None
        # cv2 Laplacian sharpness on the (downscaled) image
        if "laplacian" in self.metrics:
            try:
                import cv2
                gray = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2GRAY)
                out["laplacian"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            except Exception:
                out["laplacian"] = None
        # noise sigma via skimage (library; not hand-rolled)
        if "noise_sigma" in self.metrics:
            try:
                from skimage.restoration import estimate_sigma
                out["noise_sigma"] = float(np.mean(
                    estimate_sigma(np.asarray(im), channel_axis=-1)) * 255.0)
            except Exception:
                out["noise_sigma"] = None
        # largest-face fraction for the portrait pool (cv2 Haar)
        if want_face:
            try:
                import cv2
                if gray is None:
                    gray = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2GRAY)
                faces = _face_cascade().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                H, W = gray.shape[:2]
                out["max_face_frac"] = (max((fw * fh for (_, _, fw, fh) in faces),
                                            default=0.0) / float(W * H)) if len(faces) else 0.0
            except Exception:
                out["max_face_frac"] = None
        return out


def run(limit: Optional[int] = None, corpus: Optional[str] = None,
        device: Optional[str] = None, metrics: Optional[List[str]] = None) -> dict:
    metrics = metrics or (config.IQA_METRICS + config.IQA_EXTRA)
    runner = IQARunner(device=device, metrics=metrics)
    # Consume IQA through the core facade: a GPU lease serializes scoring against
    # any co-tenant on the same card (renderer / live SAM3) per the
    # lease->render_lock contract. Uncontended in the standalone QA CLI.
    from dataset_build.core.client import IqaClient
    from dataset_build.core.gpu_compute import GpuCompute
    iqa = IqaClient(runner, gpu=GpuCompute([runner.device]), device=runner.device)

    conn = db.connect()
    run_id = db.start_run(conn, "iqa", {"metrics": metrics, "device": device or config.IQA_DEVICE, "corpus": corpus})
    where = ["a.asset_type='image'", "a.dup_of IS NULL",   # heads only; siblings inherit at apply
             "NOT EXISTS (SELECT 1 FROM iqa_scores s WHERE s.asset_id=a.asset_id AND s.metric='musiq')"]
    params: list = []
    if corpus:
        where.append("a.corpus=?"); params.append(corpus)
    sql = f"SELECT a.asset_id, a.path, a.is_portrait_pool FROM assets a WHERE {' AND '.join(where)}"
    if limit:
        sql += f" LIMIT {int(limit)}"
    todo = conn.execute(sql, params).fetchall()
    print(f"[iqa] {len(todo)} images to score with {metrics}", file=sys.stderr)

    n_ok = n_err = 0
    for i, r in enumerate(todo):
        aid, path = r["asset_id"], r["path"]
        want_face = bool(config.FACE_DETECT and r["is_portrait_pool"])
        try:
            scores = iqa.score(path, want_face=want_face)
            db.add_scores(conn, aid, scores, run_id=run_id, model_version="pyiqa0.1.15")
            # native dimensions (the index had them NULL) -> enable the resolution gate
            denorm = {_COL[m]: v for m, v in scores.items() if m in _COL and v is not None}
            denorm["status"] = "iqa_done"
            db.update_asset_fields(conn, aid, **denorm)
            db.log_event(conn, aid, "iqa", "ok", scores, run_id)
            n_ok += 1
        except Exception as e:
            db.log_event(conn, aid, "iqa", "error", {"err": str(e)[:300]}, run_id)
            n_err += 1
        if (i + 1) % 200 == 0:
            conn.commit()
            print(f"[iqa] {i+1}/{len(todo)} ok={n_ok} err={n_err}", file=sys.stderr)
    conn.commit()
    db.finish_run(conn, run_id, {"ok": n_ok, "err": n_err})
    conn.close()
    return {"ok": n_ok, "err": n_err, "run_id": run_id}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    import json
    print(json.dumps(run(limit=args.limit, corpus=args.corpus, device=args.device)))


if __name__ == "__main__":
    main()
