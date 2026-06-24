"""SiliconFlow VL embedding + rerank client for the 50k construct agent.

Hard facts established by live API probing (2026-06-24, see goal prompt):
  - /v1/embeddings  model Qwen/Qwen3-VL-Embedding-8B, dimensions=4096.
      * TEXT  input: a str or list[str]  (batched, order preserved).
      * IMAGE input: {"image": "data:image/...;base64,..."} (ONE image per item).
        image+text in one item -> the image is silently dropped (image_tokens=0),
        so source photos are embedded image-ONLY.
      * The `instruction`/`prompt` field is ACCEPTED (HTTP 200) but SILENTLY
        IGNORED — identical vector with/without it. To steer a TEXT embedding the
        instruction must be PREPENDED into the text; the image side cannot carry one.
  - /v1/rerank  model Qwen/Qwen3-VL-Reranker-8B: query={"image": durl}, documents=list[str].

ponytail: thin requests wrapper, thread-pool for per-image calls, no async/SDK.
"""
from __future__ import annotations

import base64
import io
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Sequence, Tuple

import numpy as np
import requests
from PIL import Image, ImageOps

BASE = "https://api.siliconflow.cn/v1"
EMBED_MODEL = "Qwen/Qwen3-VL-Embedding-8B"
RERANK_MODEL = "Qwen/Qwen3-VL-Reranker-8B"
DIMS = 4096
_TIMEOUT = 90

# Concurrency: a process-wide token bucket caps the REQUEST-START rate just under
# SiliconFlow's 1000 RPM, so we can throw a big thread pool at it and saturate the
# budget without triggering 429 storms. Tune via env SF_RPM / SF_CONCURRENCY.
_RPM = int(os.environ.get("SF_RPM", "950"))
CONCURRENCY = int(os.environ.get("SF_CONCURRENCY", "32"))
_MIN_INTERVAL = 60.0 / max(_RPM, 1)
_rl_lock = threading.Lock()
_next_slot = [0.0]


def _throttle() -> None:
    """Space request starts by 60/RPM across all threads (compute slot under lock,
    sleep outside it so threads don't serialize)."""
    with _rl_lock:
        slot = max(time.monotonic(), _next_slot[0])
        _next_slot[0] = slot + _MIN_INTERVAL
    d = slot - time.monotonic()
    if d > 0:
        time.sleep(d)


def _map(fn, items: Sequence, workers: Optional[int] = None) -> list:
    with ThreadPoolExecutor(max_workers=workers or CONCURRENCY) as ex:
        return list(ex.map(fn, items))


def _key() -> str:
    k = os.environ.get("SILICONFLOW_API_KEY")
    if not k:
        raise RuntimeError("SILICONFLOW_API_KEY not set (source the repo .env)")
    return k


def _post(path: str, body: dict, retries: int = 5) -> dict:
    """POST with backoff on 429 / 5xx / transport errors. Raises on hard failures."""
    url = f"{BASE}/{path}"
    headers = {"Authorization": f"Bearer {_key()}", "Content-Type": "application/json"}
    last = None
    for attempt in range(retries):
        _throttle()
        try:
            r = requests.post(url, headers=headers, json=body, timeout=_TIMEOUT)
            if r.status_code == 200:
                return r.json()
            last = f"{r.status_code} {r.text[:200]}"
            if r.status_code not in (429, 500, 502, 503, 504):
                raise RuntimeError(f"siliconflow {path}: {last}")  # client error, no retry
        except requests.RequestException as e:  # noqa: BLE001 - retry transport errors
            last = str(e)[:200]
        time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"siliconflow {path} failed after {retries}: {last}")


def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)


def to_data_url(path: str, longedge: int = 512, quality: int = 85) -> str:
    """Downscaled JPEG data-URL — keeps image_tokens (and payload) modest."""
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(path)
    im.load()
    im = ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail((longedge, longedge))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# --------------------------------------------------------------------------- #
# embeddings
# --------------------------------------------------------------------------- #
def embed_texts(texts: Sequence[str], instruction: Optional[str] = None,
                dims: int = DIMS, batch: int = 32) -> np.ndarray:
    """L2-normalized [N, dims] text embeddings. `instruction` (if given) is
    PREPENDED to every text (the dedicated API field is a no-op)."""
    items = [f"{instruction}\n{t}" if instruction else t for t in texts]
    chunks = [items[i:i + batch] for i in range(0, len(items), batch)]

    def one(chunk: List[str]) -> List[np.ndarray]:
        d = _post("embeddings", {"model": EMBED_MODEL, "input": chunk, "dimensions": dims})
        rows = sorted(d["data"], key=lambda x: x["index"])
        return [np.asarray(r["embedding"], dtype="float32") for r in rows]

    out: List[np.ndarray] = []
    for part in _map(one, chunks):  # chunks complete concurrently, reassembled in order
        out.extend(part)
    return _norm(np.stack(out))


def embed_images(paths: Sequence[str], dims: int = DIMS, workers: Optional[int] = None,
                 longedge: int = 512) -> np.ndarray:
    """L2-normalized [N, dims] image embeddings (one API call per image, concurrent)."""
    def one(p: str) -> np.ndarray:
        d = _post("embeddings", {"model": EMBED_MODEL,
                                 "input": {"image": to_data_url(p, longedge)},
                                 "dimensions": dims})
        return np.asarray(d["data"][0]["embedding"], dtype="float32")
    return _norm(np.stack(_map(one, list(paths), workers)))


def rerank(image_path: str, documents: Sequence[str],
           instruction: Optional[str] = None, longedge: int = 512) -> List[Tuple[int, float]]:
    """Rerank `documents` (preset texts) against a source IMAGE query.
    Returns [(orig_index, relevance_score), ...] sorted by score desc.
    `instruction` is prepended to each document (rerank's own field effect is
    unverified; prepend is the safe channel — revisit in R2)."""
    docs = [f"{instruction}\n{t}" if instruction else t for t in documents]
    body = {"model": RERANK_MODEL, "query": {"image": to_data_url(image_path, longedge)},
            "documents": list(docs)}
    d = _post("rerank", body)
    res = [(r["index"], float(r["relevance_score"])) for r in d["results"]]
    res.sort(key=lambda x: -x[1])
    return res


def _selfcheck() -> None:
    """assert-based smoke test: instruction is a no-op; batch order preserved."""
    a = embed_texts(["warm vibrant grade"])[0]
    b = embed_texts(["warm vibrant grade"], instruction="Represent the style.")[0]
    # prepending instruction MUST change the vector (proves we steer via text, not field)
    assert float(a @ b) < 0.999, "instruction prepend had no effect?"
    m = embed_texts(["bw high contrast", "warm vibrant grade"])
    assert m.shape == (2, DIMS) and float(m[1] @ a) > float(m[0] @ a), "batch/order broken"
    print("sf_client selfcheck OK")


if __name__ == "__main__":
    _selfcheck()
