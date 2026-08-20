"""Atomic SHA-256 content-addressed artifact storage."""
from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageOps
import numpy as np


def _open_image(path: str | os.PathLike[str]) -> Image.Image:
    try:
        from dataset_build.tools.archive_reader import open_image

        return open_image(path)
    except (ImportError, FileNotFoundError, ValueError):
        return Image.open(path)


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    sha256: str
    uri: str
    media_type: str
    size: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ArtifactRef":
        return cls(
            sha256=str(value["sha256"]), uri=str(value["uri"]),
            media_type=str(value["media_type"]), size=int(value["size"]),
        )


class ArtifactStore:
    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        recorder: Callable[[ArtifactRef, str], None] | None = None,
        catalog_db: str | os.PathLike[str] | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.blobs = self.root / "blobs"
        self.blobs.mkdir(parents=True, exist_ok=True)
        self._recorder = recorder
        self.catalog_db = Path(catalog_db).expanduser().resolve() if catalog_db else None

    def _path(self, sha256: str) -> Path:
        if len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256):
            raise ValueError("invalid artifact SHA-256")
        return self.blobs / sha256[:2] / sha256[2:4] / sha256

    def local_path(self, sha256: str) -> Path:
        return self._path(sha256)

    @staticmethod
    def _write_payload(target: Path, payload: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            return
        fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError:
                pass
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def put_bytes(
        self, payload: bytes, *, media_type: str, retention: str = "accepted"
    ) -> ArtifactRef:
        digest = hashlib.sha256(payload).hexdigest()
        target = self._path(digest)
        self._write_payload(target, payload)
        ref = ArtifactRef(
            sha256=digest,
            uri=f"sha256://{digest}",
            media_type=media_type,
            size=len(payload),
        )
        if self._recorder is not None:
            self._recorder(ref, retention)
        return ref

    def put_json(self, value: Any, *, retention: str = "audit") -> ArtifactRef:
        payload = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return self.put_bytes(payload, media_type="application/json", retention=retention)

    def put_file(
        self, path: str | os.PathLike[str], *, media_type: str, retention: str = "accepted"
    ) -> ArtifactRef:
        return self.put_bytes(Path(path).read_bytes(), media_type=media_type, retention=retention)

    def normalize_image(
        self,
        path: str | os.PathLike[str],
        *,
        longest_edge: int = 512,
        quality: int = 85,
        retention: str = "accepted",
    ) -> ArtifactRef:
        with _open_image(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            if max(image.size) > longest_edge:
                scale = longest_edge / max(image.size)
                size = tuple(max(1, round(value * scale)) for value in image.size)
                image = image.resize(size, getattr(Image, "Resampling", Image).LANCZOS)
            output = io.BytesIO()
            image.save(
                output, "JPEG", quality=quality, optimize=False, progressive=False,
                subsampling=0,
            )
        return self.put_bytes(
            output.getvalue(), media_type="image/jpeg", retention=retention
        )

    def normalize_source_images(
        self, path: str | os.PathLike[str], *, preview_longest_edge: int = 512,
        render_short_edge: int = 1024, preview_quality: int = 85,
        render_quality: int = 95, retention: str = "accepted",
    ) -> tuple[ArtifactRef, ArtifactRef]:
        with _open_image(path) as source:
            original = ImageOps.exif_transpose(source).convert("RGB")
            preview = original.copy()
            if max(preview.size) > preview_longest_edge:
                scale = preview_longest_edge / max(preview.size)
                preview = preview.resize(
                    tuple(max(1, round(value * scale)) for value in preview.size),
                    getattr(Image, "Resampling", Image).LANCZOS,
                )
            render_scale = render_short_edge / min(original.size)
            render_size = tuple(
                max(1, round(value * render_scale)) for value in original.size
            )
            render = original if original.size == render_size else original.resize(
                render_size, getattr(Image, "Resampling", Image).LANCZOS
            )
            preview_output = io.BytesIO()
            preview.save(
                preview_output, "JPEG", quality=preview_quality, optimize=False,
                progressive=False, subsampling=0,
            )
            render_output = io.BytesIO()
            render.save(
                render_output, "JPEG", quality=render_quality, optimize=False,
                progressive=False, subsampling=0,
            )
        return (
            self.put_bytes(
                preview_output.getvalue(), media_type="image/jpeg", retention=retention
            ),
            self.put_bytes(
                render_output.getvalue(), media_type="image/jpeg", retention=retention
            ),
        )

    def normalize_render_image(
        self,
        path: str | os.PathLike[str],
        *,
        short_edge: int = 1024,
        quality: int = 95,
        retention: str = "accepted",
    ) -> ArtifactRef:
        with _open_image(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            scale = short_edge / min(image.size)
            size = tuple(max(1, round(value * scale)) for value in image.size)
            if image.size != size:
                image = image.resize(size, getattr(Image, "Resampling", Image).LANCZOS)
            output = io.BytesIO()
            image.save(
                output, "JPEG", quality=quality, optimize=False, progressive=False,
                subsampling=0,
            )
        return self.put_bytes(
            output.getvalue(), media_type="image/jpeg", retention=retention
        )

    def put_image_array(
        self, pixels: np.ndarray, *, quality: int = 95, retention: str = "accepted"
    ) -> ArtifactRef:
        array = np.asarray(pixels, dtype=np.float32)
        if array.ndim != 3 or array.shape[2] != 3 or not np.isfinite(array).all():
            raise ValueError("image array must be finite HWC RGB")
        image = Image.fromarray(
            np.clip(array * 255.0 + 0.5, 0, 255).astype(np.uint8), "RGB"
        )
        output = io.BytesIO()
        image.save(
            output, "JPEG", quality=quality, optimize=False, progressive=False,
            subsampling=0,
        )
        return self.put_bytes(
            output.getvalue(), media_type="image/jpeg", retention=retention
        )

    def put_alpha(
        self, alpha: np.ndarray, *, retention: str = "accepted"
    ) -> ArtifactRef:
        array = np.asarray(alpha, dtype=np.float32)
        if array.ndim != 2 or not np.isfinite(array).all():
            raise ValueError("alpha must be a finite HW array")
        image = Image.fromarray(
            np.clip(array * 255.0 + 0.5, 0, 255).astype(np.uint8), "L"
        )
        output = io.BytesIO()
        image.save(output, "PNG", compress_level=6)
        return self.put_bytes(
            output.getvalue(), media_type="image/png", retention=retention
        )

    def path_for(self, ref: ArtifactRef | dict[str, Any] | str) -> Path:
        if isinstance(ref, str):
            digest = ref.removeprefix("sha256://")
        elif isinstance(ref, ArtifactRef):
            digest = ref.sha256
        else:
            digest = str(ref["sha256"])
        path = self._path(digest)
        if not path.is_file() and self.catalog_db is not None:
            from dataset_build.tools.archive_reader import read_bytes

            if not self.catalog_db.is_file():
                raise FileNotFoundError(f"artifact is missing: sha256://{digest}")
            try:
                payload = read_bytes(f"sha256://{digest}", db_path=self.catalog_db)
            except (FileNotFoundError, KeyError) as exc:
                raise FileNotFoundError(
                    f"artifact is missing: sha256://{digest}"
                ) from exc
            if hashlib.sha256(payload).hexdigest() != digest:
                raise ValueError(f"archived artifact checksum mismatch: sha256://{digest}")
            self._write_payload(path, payload)
        if not path.is_file():
            raise FileNotFoundError(f"artifact is missing: sha256://{digest}")
        return path

    def read_bytes(self, ref: ArtifactRef | dict[str, Any] | str) -> bytes:
        return self.path_for(ref).read_bytes()

    def promote(self, ref: ArtifactRef | dict[str, Any]) -> ArtifactRef:
        artifact = ref if isinstance(ref, ArtifactRef) else ArtifactRef.from_dict(dict(ref))
        self.path_for(artifact)
        if self._recorder is not None:
            self._recorder(artifact, "accepted")
        return artifact

    def discard(self, ref: ArtifactRef | dict[str, Any] | str) -> bool:
        digest = ref.sha256 if isinstance(ref, ArtifactRef) else \
            str(ref["sha256"]) if isinstance(ref, dict) else ref.removeprefix("sha256://")
        path = self._path(digest)
        if not path.is_file():
            return False
        path.unlink()
        return True


__all__ = ["ArtifactRef", "ArtifactStore"]
