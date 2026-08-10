"""Read-only FastAPI backend for canonical databuild inspection."""
from __future__ import annotations

import argparse
import io
import mimetypes
import os
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from dataset_build.tools.archive_reader import default_db

from .materialize import MaterializationManager, collect_asset_paths
from .repository import (
    FAILURE_STATES,
    QUEUE_STATES,
    WINNER_FILTERS,
    GroupFilters,
    ViewerRepository,
    canonical_group_error,
    canonical_sft_error,
)


DEFAULT_DATA_ROOT = Path("/home/bc/data/datasets/vera_directionA_1M")
DEFAULT_NFS_LEDGER_ROOT = Path("/mnt/nfs/bc/data/builds")
DEFAULT_BUILD_ROOT = DEFAULT_NFS_LEDGER_ROOT
DEFAULT_LOGICAL_ROOTS = (
    Path("/home/bc/data/datasets"),
    Path("/home/bc/datasets"),
    Path("/mnt/ramstage"),
    Path("/dev/shm/veradata"),
)
DEFAULT_CACHE_BYTES = 32 * 1024**3
DEFAULT_CACHE_ROOT = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "databuild_viewer"
DIST_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"})
MAX_PAGE_SIZE = 200
MAX_THUMB_WIDTH = 2048


def _path_list(value: str | None, default: Sequence[Path]) -> tuple[Path, ...]:
    if not value:
        return tuple(path.expanduser().resolve() for path in default)
    return tuple(
        Path(part).expanduser().resolve()
        for part in value.split(os.pathsep)
        if part.strip()
    )


def _load_databuild_config(path: str) -> tuple[Path, str]:
    try:
        from construct.config import load_config
    except ImportError:
        from dataset_build.src.construct.config import load_config
    config = load_config(path, require_private=True, validate_paths=False)
    return config.output_root, config.viewer.postgres_dsn


def _initial_settings() -> tuple[tuple[Path, ...], str | None, tuple[Path, ...]]:
    config_path = os.environ.get("DBV_CONFIG")
    postgres_dsn = _load_databuild_config(config_path)[1] if config_path else None
    # The build output_root is tmpfs by contract. JSONL fallback always comes
    # from the durable NFS ledger unless the operator explicitly overrides it.
    build_roots = _path_list(os.environ.get("DBV_BUILD_ROOTS"), (DEFAULT_NFS_LEDGER_ROOT,))
    allowed = _path_list(
        os.environ.get("DBV_ALLOWED_ROOTS"),
        (DEFAULT_DATA_ROOT, *DEFAULT_LOGICAL_ROOTS, *build_roots),
    )
    return build_roots, postgres_dsn, allowed


def _cache_settings() -> tuple[Path, int, Path]:
    cache_root = Path(os.environ.get("DBV_CACHE_ROOT") or DEFAULT_CACHE_ROOT).expanduser().resolve()
    cache_bytes = int(os.environ.get("DBV_CACHE_BYTES") or DEFAULT_CACHE_BYTES)
    catalog = Path(os.environ.get("DBV_CATALOG") or default_db()).expanduser().resolve()
    return cache_root, cache_bytes, catalog


_build_roots, _postgres_dsn, _allowed_roots = _initial_settings()
_cache_root, _cache_bytes, _catalog_path = _cache_settings()
repository = ViewerRepository(_build_roots, _postgres_dsn)
allowed_roots = _allowed_roots
materializer = MaterializationManager(_cache_root, _cache_bytes, _catalog_path)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    try:
        yield
    finally:
        materializer.close()


app = FastAPI(title="Canonical Databuild Viewer", version="1", lifespan=_lifespan)


def configure(
    *,
    build_roots: Sequence[str | os.PathLike[str]],
    postgres_dsn: str | None = None,
    image_roots: Sequence[str | os.PathLike[str]] | None = None,
    cache_root: str | os.PathLike[str] | None = None,
    cache_bytes: int | None = None,
    catalog_path: str | os.PathLike[str] | None = None,
    materialization_manager: MaterializationManager | None = None,
) -> None:
    """Replace process-local viewer settings; primarily used by CLI and tests."""
    global repository, allowed_roots, materializer
    roots = tuple(Path(path).expanduser().resolve() for path in build_roots)
    repository = ViewerRepository(roots, postgres_dsn)
    allowed_roots = tuple(
        Path(path).expanduser().resolve()
        for path in (
            image_roots
            if image_roots is not None
            else (DEFAULT_DATA_ROOT, *DEFAULT_LOGICAL_ROOTS, *roots)
        )
    )
    old_materializer = materializer
    if old_materializer is not materialization_manager:
        # Release the cross-process cache owner only after its active prefetch
        # has cooperatively stopped and the worker thread has exited.
        old_materializer.close()
    if materialization_manager is None:
        default_cache_root, default_cache_bytes, default_catalog = _cache_settings()
        materialization_manager = MaterializationManager(
            cache_root or default_cache_root,
            cache_bytes if cache_bytes is not None else default_cache_bytes,
            catalog_path or default_catalog,
        )
    materializer = materialization_manager
    _thumb.cache_clear()


def _validate_filter(value: str | None, allowed: Sequence[str], name: str) -> str | None:
    if value in (None, "", "any"):
        return None
    if value not in allowed:
        raise HTTPException(422, f"invalid {name}")
    return value


@app.get("/api/health")
def api_health():
    health = repository.health()
    health["materialization_cache"] = materializer.cache_status()
    return health


@app.get("/api/builds")
def api_builds():
    return repository.builds()


@app.get("/api/facets")
def api_facets(build_id: str | None = None):
    if not build_id:
        raise HTTPException(422, "build_id is required")
    return repository.facets(build_id)


@app.get("/api/groups")
def api_groups(
    build_id: str | None = None,
    mode: str | None = None,
    preset_format: str | None = Query(None, alias="format"),
    major: str | None = None,
    minor: str | None = None,
    queue_state: str | None = Query(None, alias="queue"),
    failure_state: str | None = Query(None, alias="failure"),
    winner: str = "any",
    page: int = 1,
    page_size: int = 50,
):
    mode = _validate_filter(mode, ("local", "global"), "mode")
    preset_format = _validate_filter(preset_format, ("xmp", "lrtemplate", "lut"), "format")
    queue_state = _validate_filter(queue_state, QUEUE_STATES, "queue")
    failure_state = _validate_filter(failure_state, FAILURE_STATES, "failure")
    if winner not in WINNER_FILTERS:
        raise HTTPException(422, "invalid winner")
    page = max(1, page)
    page_size = max(1, min(page_size, MAX_PAGE_SIZE))
    if not build_id:
        raise HTTPException(422, "build_id is required")
    filters = GroupFilters(
        build_id=build_id,
        mode=mode,
        preset_format=preset_format,
        major=major or None,
        minor=minor or None,
        queue_state=queue_state,
        failure_state=failure_state,
        winner=winner,
    )
    return repository.groups(filters, page, page_size)


@app.get("/api/groups/{group_id}")
def api_group(group_id: str, build_id: str | None = None):
    if not build_id:
        raise HTTPException(422, "build_id is required")
    detail = repository.group(group_id, build_id)
    if detail is None:
        raise HTTPException(404, "group not found")
    return detail


def _authorized_image_path(value: str) -> Path:
    try:
        candidate = Path(value)
        if not candidate.is_absolute() or value.startswith("//"):
            raise HTTPException(403, "image path not allowed")
        if value != os.path.normpath(value):
            raise HTTPException(403, "image path not allowed")
        path = candidate.resolve(strict=False)
    except ValueError:
        raise HTTPException(400, "invalid image path") from None
    except (OSError, RuntimeError):
        raise HTTPException(404, "image not found") from None
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(415, "unsupported image type")
    for root in allowed_roots:
        try:
            if os.path.commonpath((str(path), str(root))) == str(root):
                return path
        except ValueError:
            continue
    raise HTTPException(403, "image path not allowed")


def _safe_path(value: str) -> Path:
    path = _authorized_image_path(value)
    if not path.is_file():
        raise HTTPException(404, "image not found")
    return path


def _prepare_detail(group_id: str, build_id: str) -> tuple[dict, tuple[str, ...]]:
    detail = repository.group(group_id, build_id)
    if detail is None:
        raise HTTPException(404, "group not found")
    canonical_paths: list[str] = []
    for path in collect_asset_paths(detail):
        canonical = str(_authorized_image_path(path))
        if canonical not in canonical_paths:
            canonical_paths.append(canonical)
    return detail, tuple(canonical_paths)


@app.post("/api/groups/{group_id}/prepare")
def api_prepare_group(group_id: str, build_id: str | None = None, retry: bool = False):
    if not build_id:
        raise HTTPException(422, "build_id is required")
    _detail, paths = _prepare_detail(group_id, build_id)
    try:
        return materializer.prepare(group_id, paths, build_id=build_id, retry=retry)
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(503, f"asset preparation unavailable: {exc}") from None


@app.get("/api/groups/{group_id}/prepare")
def api_prepare_status(group_id: str, build_id: str | None = None):
    if not build_id:
        raise HTTPException(422, "build_id is required")
    snapshot = materializer.status(group_id, build_id=build_id)
    if snapshot is None:
        raise HTTPException(404, "asset preparation not started")
    return snapshot


def _thumbnail(source, width: int) -> bytes:
    source.thumbnail((width, width * 4))
    image = source.convert("RGB")
    output = io.BytesIO()
    image.save(output, "JPEG", quality=88, optimize=True)
    return output.getvalue()


@lru_cache(maxsize=2048)
def _thumb(path: str, mtime_ns: int, width: int) -> bytes:
    from PIL import Image

    with Image.open(path) as source:
        return _thumbnail(source, width)


@app.get("/img")
def image(path: str, w: int = 512, full: bool = False):
    from PIL import Image

    resolved = _authorized_image_path(path)
    if resolved.is_file():
        if full:
            return FileResponse(resolved)
        width = max(32, min(int(w), MAX_THUMB_WIDTH))
        try:
            content = _thumb(str(resolved), resolved.stat().st_mtime_ns, width)
        except Exception:
            raise HTTPException(422, "image could not be decoded") from None
        return Response(content, media_type="image/jpeg")

    logical_path = str(resolved)
    payload = materializer.read_cached(logical_path)
    if payload is None:
        raise HTTPException(409, "archived image is not prepared")
    if full:
        media_type = mimetypes.guess_type(logical_path)[0] or "application/octet-stream"
        return Response(payload, media_type=media_type)
    width = max(32, min(int(w), MAX_THUMB_WIDTH))
    try:
        with Image.open(io.BytesIO(payload)) as source:
            content = _thumbnail(source, width)
    except Exception:
        raise HTTPException(422, "image could not be decoded") from None
    return Response(content, media_type="image/jpeg")


if DIST_DIR.is_dir():
    app.mount("/", StaticFiles(directory=DIST_DIR, html=True), name="spa")
else:
    @app.get("/", response_class=HTMLResponse)
    def development_hint():
        return (
            "<body style='font-family:monospace;background:#14161a;color:#e6e3dc;padding:40px'>"
            "<h2>Canonical Databuild Viewer</h2>"
            "<p>Build or start the frontend in <code>databuild_viewer/frontend</code>.</p>"
            "</body>"
        )


def selfcheck() -> dict:
    health = repository.health()
    if not health.get("ok"):
        raise RuntimeError("viewer store unavailable")
    builds = repository.builds()
    for build in builds:
        page_number = 1
        expected_total: int | None = None
        seen_group_ids: set[str] = set()
        while True:
            page = repository.groups(
                GroupFilters(build_id=build["build_id"]),
                page_number,
                MAX_PAGE_SIZE,
            )
            total = page.get("total")
            items = page.get("items")
            if isinstance(total, bool) or not isinstance(total, int) or total < 0:
                raise RuntimeError("viewer group pagination returned an invalid total")
            if not isinstance(items, list):
                raise RuntimeError("viewer group pagination returned invalid items")
            if expected_total is None:
                expected_total = total
            elif total != expected_total:
                raise RuntimeError("viewer group pagination changed during selfcheck")
            if not items and len(seen_group_ids) < expected_total:
                raise RuntimeError("viewer group pagination ended before its reported total")
            for item in items:
                group_id = item.get("group_id") if isinstance(item, Mapping) else None
                if not isinstance(group_id, str) or not group_id:
                    raise RuntimeError("viewer group pagination returned an invalid group ID")
                if group_id in seen_group_ids:
                    raise RuntimeError("viewer group pagination returned a duplicate group")
                seen_group_ids.add(group_id)
                detail = repository.group(group_id, build["build_id"])
                if detail is None:
                    raise RuntimeError("canonical group detail unavailable")
                error = canonical_group_error(detail["group"], detail["candidates"])
                if error is not None:
                    raise RuntimeError(f"canonical group invariant failed: {error}")
                sft_error = canonical_sft_error(detail["group"], detail["sft"])
                if sft_error is not None:
                    raise RuntimeError(f"canonical SFT invariant failed: {sft_error}")
            if len(seen_group_ids) >= expected_total:
                if len(seen_group_ids) != expected_total:
                    raise RuntimeError("viewer group pagination exceeded its reported total")
                break
            page_number += 1
    return {
        "ok": True,
        "source": health.get("source"),
        "builds": len(builds),
        "groups": health.get("groups"),
        "malformed_records": health.get("malformed_records", 0),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the canonical databuild viewer")
    parser.add_argument("--config", help="absolute 0600 canonical databuild TOML")
    parser.add_argument("--build-root", action="append", help="JSONL build or builds directory")
    parser.add_argument(
        "--nfs-ledger",
        nargs="?",
        const=str(DEFAULT_NFS_LEDGER_ROOT),
        help="canonical NFS ledger root (optionally pass one build directory)",
    )
    parser.add_argument("--allow-root", action="append", help="additional image logical root")
    parser.add_argument("--catalog", default=os.environ.get("DBV_CATALOG"), help="global archive catalog")
    parser.add_argument("--cache-root", default=os.environ.get("DBV_CACHE_ROOT"), help="local disposable member cache")
    parser.add_argument(
        "--cache-bytes",
        type=int,
        default=int(os.environ.get("DBV_CACHE_BYTES") or DEFAULT_CACHE_BYTES),
        help="maximum disposable cache bytes",
    )
    parser.add_argument("--host", default=os.environ.get("DBV_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("DBV_PORT", "8077")))
    parser.add_argument("--selfcheck", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    postgres_dsn: str | None = None
    roots: tuple[Path, ...]
    if args.config:
        _config_output_root, postgres_dsn = _load_databuild_config(args.config)
    if args.build_root:
        roots = tuple(Path(path).expanduser().resolve() for path in args.build_root)
    elif args.nfs_ledger:
        roots = (Path(args.nfs_ledger).expanduser().resolve(),)
    else:
        # Config supplies PostgreSQL only; its output_root is volatile tmpfs and
        # is never the durable JSONL fallback. _build_roots already includes an
        # explicit DBV_BUILD_ROOTS override when one is configured.
        roots = _build_roots
    image_roots = tuple(Path(path).expanduser().resolve() for path in args.allow_root or ())
    configure(
        build_roots=roots,
        postgres_dsn=postgres_dsn,
        image_roots=(DEFAULT_DATA_ROOT, *DEFAULT_LOGICAL_ROOTS, *roots, *image_roots),
        cache_root=args.cache_root,
        cache_bytes=args.cache_bytes,
        catalog_path=args.catalog,
    )
    if args.selfcheck:
        result = selfcheck()
        print(
            "selfcheck OK: "
            f"source={result['source']} builds={result['builds']} "
            f"groups={result['groups']} malformed={result['malformed_records']}"
        )
        return 0
    import uvicorn

    print(f"databuild-viewer: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
