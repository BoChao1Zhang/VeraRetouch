"""Pass 2: read only each ``I_in`` member's header to get its true geometry.

Decoding 172k full images (~200 GiB over a 93 MB/s NFS link) just to learn
their width/height would cost an hour of link time; a positional read of the
first 128 KiB is enough for the size of every container in this corpus and for
the EXIF orientation of every JPEG, so the whole pass costs ~11 GiB.

EXIF orientation is *not* optional here: values 5-8 transpose the image, so a
sample's short side -- and therefore its visual token count and sequence
length -- is wrong if orientation is ignored.  For non-JPEG containers the EXIF
block is only consulted when the ``eXIf`` chunk marker is actually present in
the buffer, which keeps PNGs on the fast path.

The geometry itself is computed by ``q3vl.train.imageproc.plan_geometry`` -- the
same function the training dataloader uses -- so the two sides cannot drift.
"""

from __future__ import annotations

import io
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from PIL import Image

from q3vl.train.imageproc import ImageRejected, plan_geometry

HEADER_BYTES = 128 * 1024
ORIENTATION_TAG = 274
# EXIF orientations 5-8 swap the two axes.
_TRANSPOSED = frozenset({5, 6, 7, 8})
# DNG decodes as TIFF in Pillow but the payload is a raw CFA frame, not an RGB
# image; it is rejected rather than silently demosaiced by the wrong code path.
_REJECTED_SUFFIXES = frozenset({".in.dng"})


class _FdCache:
    """One file descriptor per shard, opened lazily, shared across threads."""

    def __init__(self) -> None:
        self._fds: dict[str, int] = {}
        self._lock = threading.Lock()

    def get(self, root: str, shard: str) -> int:
        key = f"{root}/{shard}"
        fd = self._fds.get(key)
        if fd is not None:
            return fd
        with self._lock:
            fd = self._fds.get(key)
            if fd is None:
                fd = os.open(os.path.join(root, "shards", shard + ".tar"), os.O_RDONLY)
                self._fds[key] = fd
        return fd

    def close(self) -> None:
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()


def probe_header(fd: int, offset: int, length: int) -> dict[str, Any]:
    """Return ``{w, h, format, exif_orientation}`` from the member's first bytes.

    The read escalates 128 KiB -> 1 MiB -> whole member.  A JPEG whose EXIF/ICC
    block pushes the SOF marker past the first window is *not* corrupt, and
    reporting it as corrupt would silently delete real samples: the first pass
    over this corpus mislabelled 42 perfectly good images that way.
    """
    if length == 0:
        raise ImageRejected("image_corrupt", "empty member")
    windows = [w for w in (HEADER_BYTES, 1024 * 1024) if w < length] + [length]
    last: Exception | None = None
    for window in windows:
        data = os.pread(fd, window, offset)
        if len(data) != window:
            last = OSError(f"short read {len(data)} of {window}")
            continue
        try:
            image = Image.open(io.BytesIO(data))
            width, height = image.size
            fmt = image.format
        except Exception as exc:  # noqa: BLE001 - try a bigger window first
            last = exc
            continue
        orientation = 1
        exif_read = "skipped"
        if fmt == "JPEG" or b"eXIf" in data or b"Exif" in data:
            try:
                orientation = int(image.getexif().get(ORIENTATION_TAG) or 1)
                exif_read = "ok"
            except Exception:  # noqa: BLE001 - truncated EXIF: try a bigger window
                if window != length:
                    last = OSError("truncated EXIF")
                    continue
                orientation = 1
                exif_read = "partial"
        return {
            "raw_w": int(width),
            "raw_h": int(height),
            "format": fmt,
            "exif_orientation": orientation,
            "exif_read": exif_read,
            "header_window": window,
        }
    raise ImageRejected("image_corrupt", f"{type(last).__name__}: {last}")


def geometry_for(row: dict[str, Any], cache: _FdCache) -> dict[str, Any]:
    """Header probe + spec-5 geometry plan for one scanned row."""
    image = row["image"]
    out: dict[str, Any] = {"sft_id": row["sft_id"]}
    if image["suffix"] in _REJECTED_SUFFIXES:
        return {**out, "reason": "image_format_unsupported", "detail": image["suffix"]}
    try:
        fd = cache.get(image["root"], image["shard"])
        header = probe_header(fd, image["offset"], image["length"])
    except ImageRejected as exc:
        return {**out, "reason": exc.reason, "detail": exc.detail}
    except OSError as exc:
        return {**out, "reason": "image_unreadable", "detail": str(exc)}

    width, height = header["raw_w"], header["raw_h"]
    if header["exif_orientation"] in _TRANSPOSED:
        width, height = height, width
    try:
        geom = plan_geometry(height, width)
    except ImageRejected as exc:
        return {**out, **header, "oriented_w": width, "oriented_h": height,
                "reason": exc.reason, "detail": exc.detail}
    return {
        **out,
        **header,
        "oriented_w": width,
        "oriented_h": height,
        "out_w": geom.out_w,
        "out_h": geom.out_h,
        "grid_w": geom.grid_w,
        "grid_h": geom.grid_h,
        "vision_tokens": geom.n_visual_tokens,
        "aspect_in": geom.aspect_in,
        "aspect_out": geom.aspect_out,
        "upscaled": min(width, height) < 512,
        "reason": None,
    }


def run(rows: list[dict[str, Any]], workers: int = 32,
        progress_every: int = 20000, log=print) -> list[dict[str, Any]]:
    """Probe every row, in shard/offset order so NFS reads stay near-sequential."""
    ordered = sorted(rows, key=lambda r: (r["image"]["root"], r["image"]["shard"],
                                          r["image"]["offset"]))
    cache = _FdCache()
    results: list[dict[str, Any]] = []
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, res in enumerate(pool.map(lambda r: geometry_for(r, cache), ordered), 1):
                results.append(res)
                if progress_every and i % progress_every == 0:
                    log(f"[headers] {i}/{len(ordered)}")
    finally:
        cache.close()
    return results
