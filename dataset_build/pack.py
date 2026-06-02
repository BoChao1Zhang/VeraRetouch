"""
dataset_build/pack.py
=====================
Atomic, resumable shard writer + Sample (de)serialization + C_GT sidecar
writer for the VeraRetouch Direction-A, RECIPE-BASED 1,000,000 dataset.

Responsibilities (DATASET_BUILD_PLAN.md §6 storage / resumability):
  - Serialize a :class:`~dataset_build.contracts.Sample` to one canonical JSONL
    line (``sample_to_jsonl``) and back (``jsonl_to_sample``).
  - Write accepted Samples into per-stream shard JSONL files
    ``shards/<stream>/shard_NNNNN.jsonl``, rotating at ``storage.shard_size``.
  - Persist the single-channel C_GT as ``cgt/<stream>/<shard>/<id>.png`` (8-bit,
    native HxW, value=round(255*C_GT)) + a tiny ``<id>.npy`` patch-grid map
    (float16, 16x16) where the model's C(x) actually lives.
  - Atomic writes: ``.jsonl.tmp`` -> fsync -> rename; ``manifest_index.jsonl``
    gets one row per committed shard.
  - Resumability: ``done_ids()`` returns every committed sample_id (scans
    existing shards once) so streams skip already-built work.
  - ``stats()`` reports per-stream counts, region-local fraction, reject rate,
    and bytes-on-disk.

IMPORT-LIGHT at top level: stdlib + ``contracts`` only. ``numpy`` / ``PIL`` are
imported lazily inside the C_GT writer (the only place pixels touch disk); the
global all-ones C_GT uses a single shared sentinel PNG, never 400k copies.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dataset_build.contracts import (
    AfterSource,
    CgtRef,
    DegradeSpec,
    MaskSource,
    Provenance,
    QualityScores,
    RawDecode,
    Recipe,
    RecipeKind,
    Sample,
    SceneMeta,
    StreamId,
)
from dataset_build.contracts import ShardWriter as ShardWriterABC


# ===========================================================================
# (De)serialization.
# ===========================================================================


def _enumify(value: Any) -> Any:
    """Recursively convert Enums -> .value and tuples -> lists for JSON."""
    from enum import Enum

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {k: _enumify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_enumify(v) for v in value]
    return value


def sample_to_jsonl(sample: Sample) -> str:
    """Canonical one-line JSON serialization of a Sample.

    Uses ``dataclasses.asdict`` (deep) then maps Enums -> their string values and
    tuples -> lists. ``native_size`` (H, W) becomes a 2-list. Stable key order so
    diffs/dedup are reproducible.
    """
    d = dataclasses.asdict(sample)
    d = _enumify(d)
    return json.dumps(d, ensure_ascii=False, sort_keys=True)


def _coerce_enum(enum_cls, value, default):
    if value is None:
        return default
    try:
        return enum_cls(value)
    except ValueError:
        return default


def _to_recipe(d: Optional[Dict[str, Any]]) -> Recipe:
    if not d:
        return Recipe(kind=RecipeKind.PARAM, params=None)
    kind = _coerce_enum(RecipeKind, d.get("kind"), RecipeKind.PARAM)
    degrade = None
    if d.get("degrade"):
        dg = d["degrade"]
        degrade = DegradeSpec(
            mode=dg.get("mode", "gaussian_op"),
            op_params=dg.get("op_params", {}) or {},
            sigma_profile=dg.get("sigma_profile"),
            aspects=list(dg.get("aspects", []) or []),
            forward=bool(dg.get("forward", False)),  # canonical v2 default: a record missing
                                                      # the field must not flip to +op_params
            seed=dg.get("seed"),
        )
    return Recipe(
        kind=kind,
        params=d.get("params"),
        lut_recipe_id=d.get("lut_recipe_id"),
        degrade=degrade,
        provenance=_coerce_enum(Provenance, d.get("provenance"), Provenance.PARAM),
        source_recipe_id=d.get("source_recipe_id"),
        meta=d.get("meta", {}) or {},
    )


def _to_cgt(d: Optional[Dict[str, Any]]) -> CgtRef:
    if not d:
        return CgtRef()
    return CgtRef(
        cgt_path=d.get("cgt_path"),
        cgt_aspect_paths=d.get("cgt_aspect_paths"),
        cgt_patchgrid_path=d.get("cgt_patchgrid_path"),
        raw_mask_path=d.get("raw_mask_path"),
        rle=d.get("rle"),
        mask_source=_coerce_enum(MaskSource, d.get("mask_source"), MaskSource.GLOBAL),
        concepts=list(d.get("concepts", []) or []),
        mask_score=d.get("mask_score"),
        aspect_magnitude=d.get("aspect_magnitude", {}) or {},
        coverage=d.get("coverage"),
        soft_blur_sigma_px=d.get("soft_blur_sigma_px"),
        magnitude_tau=d.get("magnitude_tau"),
    )


def jsonl_to_sample(line: str) -> Sample:
    """Inverse of ``sample_to_jsonl`` (training-time loader + eval)."""
    d = json.loads(line)
    sm = d.get("scene_meta", {}) or {}
    q = d.get("quality", {}) or {}
    ns = d.get("native_size")
    native_size: Optional[Tuple[int, int]] = tuple(ns) if ns else None  # type: ignore[assignment]
    return Sample(
        sample_id=d["sample_id"],
        stream=_coerce_enum(StreamId, d.get("stream"), StreamId.S6_RECIPE_GLOBAL),
        shard=d.get("shard", ""),
        source_path=d["source_path"],
        raw_decode=_coerce_enum(RawDecode, d.get("raw_decode"), RawDecode.NONE),
        recipe=_to_recipe(d.get("recipe")),
        region_local=bool(d.get("region_local", False)),
        c_gt=_to_cgt(d.get("c_gt")),
        instruction=d.get("instruction"),
        instruction_short=d.get("instruction_short"),
        think=d.get("think"),
        answer=d.get("answer"),
        scene_meta=SceneMeta(
            scene=sm.get("scene"),
            style=sm.get("style"),
            lang=sm.get("lang", "en"),
            masksubtype_hint=int(sm.get("masksubtype_hint", 0) or 0),
            tags=list(sm.get("tags", []) or []),
        ),
        quality=QualityScores(
            look_match=q.get("look_match"),
            param_sane=q.get("param_sane"),
            processed_ok=q.get("processed_ok"),
            mllm_score=q.get("mllm_score"),
            aesthetic=q.get("aesthetic"),
            mask_quality=q.get("mask_quality"),
            histsim=q.get("histsim"),
            er_recon_psnr=q.get("er_recon_psnr"),
            rejected_reason=q.get("rejected_reason"),
        ),
        source_id=d.get("source_id"),
        recipe_asset_id=d.get("recipe_asset_id"),
        native_size=native_size,
        build_version=d.get("build_version", "v2"),
        schema_version=d.get("schema_version", "datagen_v2"),
        after_source=_coerce_enum(AfterSource, d.get("after_source"), AfterSource.TEACHER),
        meta=d.get("meta", {}) or {},
    )


# ===========================================================================
# Low-level atomic file helpers.
# ===========================================================================


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (.tmp -> fsync -> rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ===========================================================================
# ShardWriter: per-stream sharded JSONL + C_GT sidecars, atomic & resumable.
# ===========================================================================


class ShardWriter(ShardWriterABC):
    """Writes accepted Samples to ``shards/<stream>/shard_NNNNN.jsonl`` and the
    C_GT sidecars, atomically and resumably.

    Buffers a shard in memory (``shard_size`` rows), then commits the whole shard
    atomically and appends a row to ``manifest_index.jsonl``. ``done_ids()`` scans
    committed shards once to support resumption.

    Thread-safe (a lock guards the per-stream buffers) so streams may run on a
    threadpool. C_GT PNGs are written immediately (before the JSONL row that
    references them) so a crash never references a missing sidecar.
    """

    def __init__(self, out_root: str, config: Dict[str, Any]):
        self.out_root = Path(out_root)
        self.config = config or {}
        storage = (self.config.get("storage", {}) or {})
        self.shard_size = int(storage.get("shard_size", 5000))
        self.shard_prefix = str(storage.get("shard_prefix", "shard"))
        self.manifest_index = storage.get("manifest_index", "manifest_index.jsonl")
        self.cgt_subdir = storage.get("cgt_subdir", "cgt")
        self.shards_dir = self.out_root / "shards"
        self.cgt_dir = self.out_root / self.cgt_subdir
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        self.cgt_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        # per-stream in-memory buffers + next shard index
        self._buffers: Dict[str, List[Sample]] = {}
        self._next_shard: Dict[str, int] = {}
        # stats
        self._committed = 0
        self._region_local = 0
        self._by_stream: Dict[str, int] = {}
        self._bytes = 0
        self._rejects = 0
        # the shared all-ones sentinel for global C_GT
        self._global_sentinel: Optional[str] = None
        self._teacher_manifest_cache: Optional[Dict[str, Any]] = None

        self._init_shard_indices()

    # ----------------------------------------------------------- internals
    def _stream_dir(self, stream: StreamId) -> Path:
        return self.shards_dir / stream.value

    def _shard_name(self, idx: int) -> str:
        return f"{self.shard_prefix}_{idx:05d}"

    def _shard_path(self, stream: StreamId, idx: int) -> Path:
        return self._stream_dir(stream) / f"{self._shard_name(idx)}.jsonl"

    def _init_shard_indices(self) -> None:
        """Resume: find the highest committed shard per stream and continue after it."""
        for sd in self.shards_dir.glob("*"):
            if not sd.is_dir():
                continue
            committed = [
                p for p in sd.glob(f"{self.shard_prefix}_*.jsonl")
                if p.stem.rsplit("_", 1)[-1].isdigit()
                and p.stem == f"{self.shard_prefix}_{p.stem.rsplit('_', 1)[-1]}"
            ]
            committed = sorted(committed)
            self._next_shard[sd.name] = (
                int(committed[-1].stem.rsplit("_", 1)[1]) + 1 if committed else 0
            )

    def _global_sentinel_path(self) -> str:
        """Lazily create one shared all-ones 1x1 PNG for global C_GT (DATASET §6)."""
        if self._global_sentinel:
            return self._global_sentinel
        p = self.cgt_dir / "_global_ones.png"
        if not p.exists():
            # Atomic create: render the 1x1 value-255 L PNG into a per-process
            # temp file then os.replace into place. os.replace is atomic, so a
            # dual-GPU TOCTOU race only ever clobbers identical bytes (the sha256
            # manifest pin stays stable regardless of who wins).
            try:
                import numpy as np
                from PIL import Image  # type: ignore
            except Exception as exc:
                # PIL/Pillow missing. A real (non-dry-run) build MUST fail loud:
                # an empty marker would have a different sha256 than the real PNG
                # and silently break the manifest sentinel pin.
                if not self.config.get("dry_run"):
                    raise RuntimeError(
                        "Pillow (PIL) is required to write the global C_GT "
                        "sentinel PNG; refusing to write an empty placeholder in a "
                        "real build (it would not match the manifest sha256 pin). "
                        "Install Pillow or run with dry_run."
                    ) from exc
                # dry-run only: write a marker file; trainers treat missing/sentinel
                # C_GT as all-ones.
                tmp = p.parent / (p.name + f".tmp.{os.getpid()}")
                tmp.write_bytes(b"")
                os.replace(tmp, p)
            else:
                tmp = p.parent / (p.name + f".tmp.{os.getpid()}")
                Image.fromarray(np.full((1, 1), 255, dtype="uint8"), mode="L").save(
                    tmp, format="PNG"
                )
                os.replace(tmp, p)
        self._global_sentinel = str(p)
        return self._global_sentinel

    def _global_sentinel_sha256(self) -> Optional[str]:
        p = Path(self._global_sentinel_path())
        try:
            return hashlib.sha256(p.read_bytes()).hexdigest()
        except OSError:
            return None

    # ------------------------------------------------------------ C_GT sink
    def write_cgt(
        self,
        stream: StreamId,
        shard: str,
        sample_id: str,
        cgt01: Any,
        patch_grid: int = 16,
        raw_mask01: Any = None,
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Persist a region-local single-channel C_GT in [0,1] HxW.

        Returns ``(png_path, patchgrid_npy_path, raw_mask_path)`` (absolute paths) or
        ``(None, None)`` on failure. Atomic per file. The ``shard`` argument may
        be ``"pending"`` (stream-time placeholder); the writer re-homes the file
        into the real shard dir at commit time via ``_rehome_cgt``. To keep paths
        stable we lay C_GT under ``cgt/<stream>/<sample_id>.{png,npy}`` (flat per
        stream), which is shard-independent and avoids a rename on rotation.
        """
        import numpy as np

        a = np.clip(np.asarray(cgt01, dtype="float32"), 0.0, 1.0)
        out_dir = self.cgt_dir / stream.value
        out_dir.mkdir(parents=True, exist_ok=True)
        png_path = out_dir / f"{sample_id}.png"
        npy_path = out_dir / f"{sample_id}.npy"
        raw_mask_path = out_dir / f"{sample_id}.raw_mask.png" if raw_mask01 is not None else None

        # --- native PNG (8-bit) ---
        u8 = np.round(a * 255.0).astype("uint8")
        try:
            from PIL import Image  # type: ignore

            # NB: PIL infers format from extension, so a ".png.tmp" suffix raises;
            # use an explicit format and a sibling .tmp path then atomic-rename.
            tmp = png_path.parent / (png_path.name + ".tmp")
            Image.fromarray(u8, mode="L").save(tmp, format="PNG", compress_level=1)
            os.replace(tmp, png_path)
        except Exception:
            return None, None

        # --- tiny patch-grid npy (float16, patch_grid x patch_grid) ---
        try:
            grid = self._downsample_grid(a, patch_grid)
            tmp = npy_path.parent / (npy_path.name + ".tmp")
            # np.save appends .npy if missing, so save to an explicit handle.
            with open(tmp, "wb") as fh:
                np.save(fh, grid.astype("float16"))
            os.replace(tmp, npy_path)
            patch_out: Optional[str] = str(npy_path)
        except Exception:
            patch_out = None

        if raw_mask_path is not None:
            try:
                raw = np.clip(np.asarray(raw_mask01, dtype="float32"), 0.0, 1.0)
                raw_u8 = np.round(raw * 255.0).astype("uint8")
                from PIL import Image  # type: ignore

                tmp = raw_mask_path.parent / (raw_mask_path.name + ".tmp")
                Image.fromarray(raw_u8, mode="L").save(tmp, format="PNG", compress_level=1)
                os.replace(tmp, raw_mask_path)
            except Exception:
                raw_mask_path = None

        return str(png_path), patch_out, (str(raw_mask_path) if raw_mask_path else None)

    @staticmethod
    def _downsample_grid(a: Any, n: int) -> Any:
        """Average-pool an HxW map to an n x n patch grid (where C(x) lives)."""
        import numpy as np

        H, W = a.shape[:2]
        if H == 0 or W == 0:
            return np.zeros((n, n), dtype="float32")
        try:
            import cv2

            return cv2.resize(
                a.astype("float32"), (n, n), interpolation=cv2.INTER_AREA
            ).astype("float32")
        except Exception:
            ys = np.linspace(0, H, n + 1).astype(int)
            xs = np.linspace(0, W, n + 1).astype(int)
            out = np.zeros((n, n), dtype="float32")
            for i in range(n):
                for j in range(n):
                    y0, y1 = ys[i], max(ys[i] + 1, ys[i + 1])
                    x0, x1 = xs[j], max(xs[j] + 1, xs[j + 1])
                    out[i, j] = float(a[y0:y1, x0:x1].mean())
            return out

    # ------------------------------------------------------------- write
    def write(self, sample: Sample) -> None:
        """Buffer ``sample`` into its stream's shard; commit when the buffer fills."""
        with self._lock:
            # global C_GT -> point at the shared sentinel (no per-sample PNG)
            if not sample.region_local and not sample.c_gt.cgt_path:
                sample.c_gt.cgt_path = self._global_sentinel_path()
            key = sample.stream.value
            buf = self._buffers.setdefault(key, [])
            # assign the concrete shard id now so the JSONL row is self-consistent
            idx = self._next_shard.setdefault(key, 0)
            sample.shard = self._shard_name(idx)
            buf.append(sample)
            if len(buf) >= self.shard_size:
                self._commit_stream(sample.stream)

    def _commit_stream(self, stream: StreamId) -> None:
        key = stream.value
        buf = self._buffers.get(key)
        if not buf:
            return
        idx = self._next_shard.get(key, 0)
        path = self._shard_path(stream, idx)
        lines = [sample_to_jsonl(s) for s in buf]
        text = "\n".join(lines) + "\n"
        _atomic_write_text(path, text)
        nbytes = path.stat().st_size

        # update stats
        self._committed += len(buf)
        self._by_stream[key] = self._by_stream.get(key, 0) + len(buf)
        self._region_local += sum(1 for s in buf if s.region_local)
        self._bytes += nbytes

        # append a manifest row (atomic append via temp-rewrite-free O_APPEND)
        self._append_manifest(
            {
                "stream": key,
                "shard": self._shard_name(idx),
                "path": str(path),
                "n": len(buf),
                "bytes": nbytes,
                "region_local": sum(1 for s in buf if s.region_local),
                "build_version": self.config.get("build_version", "v2"),
                "schema_version": self.config.get("schema_version", "datagen_v2"),
                "teacher": self._teacher_manifest(),
                "global_sentinel_sha256": self._global_sentinel_sha256(),
                "soft_blur_sigma_px": (self.config.get("cgt", {}) or {}).get("soft_blur_sigma_px"),
                "magnitude_tau": (self.config.get("cgt", {}) or {}).get("magnitude_tau"),
            }
        )

        self._next_shard[key] = idx + 1
        self._buffers[key] = []

    def _append_manifest(self, row: Dict[str, Any]) -> None:
        mpath = self.out_root / self.manifest_index
        line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        with open(mpath, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    def _teacher_manifest(self) -> Dict[str, Any]:
        if self._teacher_manifest_cache is None:
            self._teacher_manifest_cache = teacher_manifest_from_config(self.config)
        return dict(self._teacher_manifest_cache)

    def log_reject(self, sample: Sample) -> None:
        """Append a reject row to rejects.jsonl (never silent-pass; DATASET §7)."""
        with self._lock:
            self._rejects += 1
            rpath = self.out_root / (self.config.get("qa", {}) or {}).get(
                "reject_log", "rejects.jsonl"
            )
            row = {
                "sample_id": sample.sample_id,
                "stream": sample.stream.value,
                "source_path": sample.source_path,
                "region_local": sample.region_local,
                "reason": sample.quality.rejected_reason,
            }
            with open(rpath, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()

    def flush(self) -> None:
        """Commit every non-empty buffer (end-of-run / checkpoint)."""
        with self._lock:
            for key in list(self._buffers.keys()):
                if self._buffers[key]:
                    self._commit_stream(StreamId(key))

    def done_ids(self) -> Iterable[str]:
        """Sample_ids already committed to shards (for --resume).

        Under a multi-worker stride partition each worker writes only its own
        ``shard_w{i}of{n}`` prefix and (pos%n==i) emits a disjoint sid space, so
        this worker's done-set is fully self-contained in its own prefix files.
        Scope the scan to that prefix (skip the other worker's shards entirely).
        With the default ``shard`` prefix (single worker / no partition) keep
        scanning every committed shard for back-compat.
        """
        ids: List[str] = []
        worker_scoped = self.shard_prefix != "shard"
        glob_pat = f"{self.shard_prefix}_*.jsonl" if worker_scoped else "*.jsonl"
        for sd in self.shards_dir.glob("*"):
            if not sd.is_dir():
                continue
            for shard in sd.glob(glob_pat):
                # exact ``<prefix>_<digits>`` so a worker prefix never over-matches
                # a longer sibling prefix's files (mirrors _init_shard_indices).
                if worker_scoped:
                    tail = shard.stem.rsplit("_", 1)
                    if not (
                        len(tail) == 2
                        and tail[-1].isdigit()
                        and shard.stem == f"{self.shard_prefix}_{tail[-1]}"
                    ):
                        continue
                try:
                    with open(shard, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                ids.append(json.loads(line)["sample_id"])
                            except (json.JSONDecodeError, KeyError):
                                continue
                except OSError:
                    continue
        return ids

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._committed
            rl_frac = (self._region_local / total) if total else 0.0
            attempted = total + self._rejects
            reject_rate = (self._rejects / attempted) if attempted else 0.0
            return {
                "committed": total,
                "by_stream": dict(self._by_stream),
                "region_local": self._region_local,
                "region_local_fraction": rl_frac,
                "rejects": self._rejects,
                "reject_rate": reject_rate,
                "bytes_on_disk": self._bytes,
                "shards": {k: v for k, v in self._next_shard.items()},
            }


# ===========================================================================
# Manifest invariant helpers.
# ===========================================================================


def file_sha256(path: str, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Streaming SHA-256 for explicit invariant pins."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def teacher_manifest_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return the teacher invariant block written to every shard manifest row.

    By default this uses ``models.veraretouch.teacher_sha256`` as a literal pin.
    To compute a hash explicitly, set ``teacher_sha256_path`` to a checkpoint file
    or set ``teacher_sha256: auto`` and provide ``model_path`` containing
    ``model.safetensors``. ``ShardWriter`` caches the result per process.
    """
    models = (config.get("models", {}) or {})
    vr = (models.get("veraretouch", {}) or {})
    return {
        "model_path": vr.get("model_path"),
        "teacher_sha256": _resolve_teacher_sha256(vr),
        "greedy": vr.get("greedy", True),
        "dtype": vr.get("dtype", "bfloat16"),
        "max_new_tokens": vr.get("max_new_tokens"),
        "chunk": vr.get("chunk"),
    }


def validate_manifest_index(
    out_root: str,
    config: Dict[str, Any],
    *,
    manifest_index: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate committed shard manifest invariants against the current config.

    Returns a report instead of raising so callers can warn, reject, or branch on
    ``schema_version``. It verifies the plan-A pins: build/schema version,
    teacher block, global-sentinel sha256, shard bytes, and shard row counts.
    """
    root = Path(out_root)
    storage = (config.get("storage", {}) or {})
    index_name = manifest_index or storage.get("manifest_index", "manifest_index.jsonl")
    index_path = root / index_name
    expected_teacher = teacher_manifest_from_config(config)
    expected_build = config.get("build_version", "v2")
    expected_schema = config.get("schema_version", "datagen_v2")
    expected_sentinel = _sentinel_sha256_for_root(root, storage.get("cgt_subdir", "cgt"))
    cgt_cfg = (config.get("cgt", {}) or {})
    expected_soft_blur_sigma_px = cgt_cfg.get("soft_blur_sigma_px")
    expected_magnitude_tau = cgt_cfg.get("magnitude_tau")

    report: Dict[str, Any] = {
        "manifest_index": str(index_path),
        "rows": 0,
        "ok": True,
        "issues": [],
    }
    if not index_path.exists():
        report["ok"] = False
        report["issues"].append({"row": None, "kind": "missing_manifest", "detail": str(index_path)})
        return report

    with index_path.open("r", encoding="utf-8") as f:
        for row_index, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            report["rows"] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                _add_manifest_issue(report, row_index, "invalid_json", str(exc))
                continue
            _validate_manifest_row(
                report,
                row_index,
                row,
                expected_build=expected_build,
                expected_schema=expected_schema,
                expected_teacher=expected_teacher,
                expected_sentinel=expected_sentinel,
                expected_soft_blur_sigma_px=expected_soft_blur_sigma_px,
                expected_magnitude_tau=expected_magnitude_tau,
            )

    report["ok"] = not report["issues"]
    return report


def _resolve_teacher_sha256(vr: Dict[str, Any]) -> Optional[str]:
    literal = vr.get("teacher_sha256")
    if literal and str(literal).lower() != "auto":
        return str(literal)
    path = vr.get("teacher_sha256_path")
    if path:
        return file_sha256(str(path))
    if literal and str(literal).lower() == "auto":
        model_path = vr.get("model_path")
        if model_path:
            candidate = Path(str(model_path)) / "model.safetensors"
            if candidate.exists():
                return file_sha256(str(candidate))
    return None


def _sentinel_sha256_for_root(root: Path, cgt_subdir: str) -> Optional[str]:
    p = root / cgt_subdir / "_global_ones.png"
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except OSError:
        return None


def _validate_manifest_row(
    report: Dict[str, Any],
    row_index: int,
    row: Dict[str, Any],
    *,
    expected_build: Any,
    expected_schema: Any,
    expected_teacher: Dict[str, Any],
    expected_sentinel: Optional[str],
    expected_soft_blur_sigma_px: Any = None,
    expected_magnitude_tau: Any = None,
) -> None:
    if row.get("build_version") != expected_build:
        _add_manifest_issue(
            report,
            row_index,
            "build_version_mismatch",
            {"expected": expected_build, "actual": row.get("build_version")},
        )
    if row.get("schema_version") != expected_schema:
        _add_manifest_issue(
            report,
            row_index,
            "schema_version_mismatch",
            {"expected": expected_schema, "actual": row.get("schema_version")},
        )
    if row.get("teacher") != expected_teacher:
        _add_manifest_issue(
            report,
            row_index,
            "teacher_mismatch",
            {"expected": expected_teacher, "actual": row.get("teacher")},
        )
    if row.get("global_sentinel_sha256") != expected_sentinel:
        _add_manifest_issue(
            report,
            row_index,
            "global_sentinel_sha256_mismatch",
            {"expected": expected_sentinel, "actual": row.get("global_sentinel_sha256")},
        )
    if row.get("soft_blur_sigma_px") != expected_soft_blur_sigma_px:
        _add_manifest_issue(
            report,
            row_index,
            "soft_blur_sigma_px_mismatch",
            {"expected": expected_soft_blur_sigma_px, "actual": row.get("soft_blur_sigma_px")},
        )
    if row.get("magnitude_tau") != expected_magnitude_tau:
        _add_manifest_issue(
            report,
            row_index,
            "magnitude_tau_mismatch",
            {"expected": expected_magnitude_tau, "actual": row.get("magnitude_tau")},
        )

    shard_path = row.get("path")
    if not shard_path:
        _add_manifest_issue(report, row_index, "missing_shard_path", None)
        return
    shard = Path(str(shard_path))
    if not shard.exists():
        _add_manifest_issue(report, row_index, "missing_shard_file", str(shard))
        return
    try:
        actual_bytes = shard.stat().st_size
        if row.get("bytes") != actual_bytes:
            _add_manifest_issue(
                report,
                row_index,
                "shard_bytes_mismatch",
                {"expected": row.get("bytes"), "actual": actual_bytes},
            )
        with shard.open("r", encoding="utf-8") as f:
            actual_rows = sum(1 for line in f if line.strip())
        if row.get("n") != actual_rows:
            _add_manifest_issue(
                report,
                row_index,
                "shard_count_mismatch",
                {"expected": row.get("n"), "actual": actual_rows},
            )
    except OSError as exc:
        _add_manifest_issue(report, row_index, "shard_stat_error", str(exc))


def _add_manifest_issue(
    report: Dict[str, Any],
    row_index: Optional[int],
    kind: str,
    detail: Any,
) -> None:
    report["issues"].append({"row": row_index, "kind": kind, "detail": detail})
