"""Materialise the spec-5 image for every surviving sample.

Why the images are re-encoded at the contract size instead of being copied
verbatim (see NOTES.md decision D-1):

* the spec-5 transform is deterministic and there is *no* train-time image
  augmentation anywhere in this campaign, so pre-applying it removes nothing;
* the originals total ~250 GiB and average ~8 MP; the contract-sized copies are
  ~25 GiB.  Sixteen-plus training runs (Base SFT + 8 Where arms + 8 What arms)
  each read the whole set, so the difference is measured in TB of NFS traffic
  and in per-step CPU spent decoding 8 MP JPEGs;
* the original member (path, offset, length, SHA-256) is recorded in every
  record, so the raw bytes remain reachable and re-baking is one command.

Chroma subsampling is disabled (``subsampling=0``).  This is a colour-grading
dataset; 4:2:0 would throw away exactly the signal the LUT stage has to predict.

The transform is byte-for-byte the one the trainer would apply online -- decode,
``ImageOps.exif_transpose``, RGB, ``plan_geometry`` size, BICUBIC -- with the
geometry taken from the plan (which got it from the same ``plan_geometry``).
Deliberately *no* ``Image.draft()``: DCT-domain pre-scaling would make the baked
pixels differ from what ``q3vl.train.imageproc.prepare_image`` produces.

I/O shape matters more than CPU here.  Twenty-four worker processes each issuing
their own 1.5 MB positional read into a 2 GiB tar measured **9.8 MB/s** on this
NFS mount -- concurrent random reads defeat the client's readahead -- while a
single sequential stream of the same tar measures **93 MB/s**.  So one reader
thread per shard streams the tar in 8 MiB chunks and slices members out of it,
and the process pool only decodes.  The queue between them is bounded, because
``Pool.imap`` would otherwise drain the whole generator into memory.
"""

from __future__ import annotations

import io
import os
import queue
import threading
from multiprocessing import Pool
from typing import Any, Iterator

from PIL import Image, ImageOps

from ..train.imageproc import _RESAMPLE
from .config import JPEG_QUALITY, JPEG_SUBSAMPLING

READ_CHUNK = 8 * 1024 * 1024
QUEUE_DEPTH = 512
BLOCK = 256


class BakeError(RuntimeError):
    pass


class _SequentialReader:
    """Forward-only reader over one tar, serving members by (offset, length)."""

    def __init__(self, path: str, chunk: int = READ_CHUNK) -> None:
        self.handle = open(path, "rb", buffering=0)
        self.chunk = chunk
        self.pos = 0                 # file offset of buf[0]
        self.buf = bytearray()
        try:
            os.posix_fadvise(self.handle.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
        except (AttributeError, OSError):
            pass

    def read_at(self, offset: int, length: int) -> bytes:
        if offset < self.pos:
            self.handle.seek(offset)
            self.pos, self.buf = offset, bytearray()
        elif offset > self.pos:
            drop = offset - self.pos
            if drop <= len(self.buf):
                del self.buf[:drop]
                self.pos = offset
            else:
                self.handle.seek(offset)
                self.pos, self.buf = offset, bytearray()
        while len(self.buf) < length:
            data = self.handle.read(max(self.chunk, length - len(self.buf)))
            if not data:
                raise BakeError(f"unexpected EOF at {offset}+{length}")
            self.buf.extend(data)
        return bytes(self.buf[:length])

    def close(self) -> None:
        self.handle.close()


def _stream(rows: list[dict[str, Any]], sink: queue.Queue) -> None:
    """Producer: yield ``(sft_id, out_w, out_h, raw_bytes)`` in plan order."""
    reader: _SequentialReader | None = None
    current: tuple[str, str] | None = None
    try:
        for row in rows:
            src = row["image_src"]
            key = (src["root"], src["shard"])
            if key != current:
                if reader is not None:
                    reader.close()
                reader = _SequentialReader(
                    os.path.join(src["root"], "shards", src["shard"] + ".tar"))
                current = key
            data = reader.read_at(src["offset"], src["length"])
            sink.put((row["sft_id"], row["out_w"], row["out_h"], data))
    except BaseException as exc:  # noqa: BLE001 - forwarded to the consumer
        sink.put(exc)
    finally:
        if reader is not None:
            reader.close()
        sink.put(None)


def bake_one(task: tuple[str, int, int, bytes]) -> tuple[str, bytes]:
    sft_id, out_w, out_h, data = task
    image = Image.open(io.BytesIO(data))
    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    if image.size != (out_w, out_h):
        image = image.resize((out_w, out_h), _RESAMPLE)
    if image.size != (out_w, out_h):
        raise BakeError(f"{sft_id}: resize produced {image.size}, expected {(out_w, out_h)}")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=JPEG_QUALITY,
               subsampling=JPEG_SUBSAMPLING, optimize=True)
    return sft_id, buffer.getvalue()


def bake_payloads(rows: list[dict[str, Any]], workers: int = 24,
                  log=print) -> Iterator[tuple[str, bytes]]:
    """Yield ``(logical_path, jpeg_bytes)`` in plan order."""
    sink: queue.Queue = queue.Queue(maxsize=QUEUE_DEPTH)
    producer = threading.Thread(target=_stream, args=(rows, sink), daemon=True)
    producer.start()
    done = 0
    finished = False
    with Pool(processes=workers) as pool:
        while not finished:
            block: list[tuple[str, int, int, bytes]] = []
            while len(block) < BLOCK:
                item = sink.get()
                if item is None:
                    finished = True
                    break
                if isinstance(item, BaseException):
                    raise item
                block.append(item)
            if not block:
                break
            for sft_id, blob in pool.map(bake_one, block, chunksize=4):
                done += 1
                if done % 10000 == 0:
                    log(f"[bake] {done}/{len(rows)}")
                yield f"sft2seg/images/{sft_id}.jpg", blob
    producer.join(timeout=30)
