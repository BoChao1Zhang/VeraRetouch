"""Read-only FastAPI backend for canonical databuild inspection."""
from __future__ import annotations

import argparse
import io
import os
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from .repository import (
    FAILURE_STATES,
    QUEUE_STATES,
    WINNER_FILTERS,
    GroupFilters,
    ViewerRepository,
    canonical_group_error,
)


DEFAULT_DATA_ROOT = Path("/home/bc/data/datasets/vera_directionA_1M")
DEFAULT_BUILD_ROOT = DEFAULT_DATA_ROOT / "builds"
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
    if config_path:
        output_root, postgres_dsn = _load_databuild_config(config_path)
        build_roots = (output_root,)
    else:
        build_roots = _path_list(os.environ.get("DBV_BUILD_ROOTS"), (DEFAULT_BUILD_ROOT,))
        postgres_dsn = None
    allowed = _path_list(os.environ.get("DBV_ALLOWED_ROOTS"), (DEFAULT_DATA_ROOT, *build_roots))
    return build_roots, postgres_dsn, allowed


_build_roots, _postgres_dsn, _allowed_roots = _initial_settings()
repository = ViewerRepository(_build_roots, _postgres_dsn)
allowed_roots = _allowed_roots

app = FastAPI(title="Canonical Databuild Viewer", version="1")


def configure(
    *,
    build_roots: Sequence[str | os.PathLike[str]],
    postgres_dsn: str | None = None,
    image_roots: Sequence[str | os.PathLike[str]] | None = None,
) -> None:
    """Replace process-local viewer settings; primarily used by CLI and tests."""
    global repository, allowed_roots
    roots = tuple(Path(path).expanduser().resolve() for path in build_roots)
    repository = ViewerRepository(roots, postgres_dsn)
    allowed_roots = tuple(
        Path(path).expanduser().resolve()
        for path in (image_roots if image_roots is not None else (DEFAULT_DATA_ROOT, *roots))
    )
    _thumb.cache_clear()


def _validate_filter(value: str | None, allowed: Sequence[str], name: str) -> str | None:
    if value in (None, "", "any"):
        return None
    if value not in allowed:
        raise HTTPException(422, f"invalid {name}")
    return value


@app.get("/api/health")
def api_health():
    return repository.health()


@app.get("/api/builds")
def api_builds():
    return repository.builds()


@app.get("/api/facets")
def api_facets(build_id: str | None = None):
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
    filters = GroupFilters(
        build_id=build_id or None,
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
def api_group(group_id: str):
    detail = repository.group(group_id)
    if detail is None:
        raise HTTPException(404, "group not found")
    return detail


def _safe_path(value: str) -> Path:
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        raise HTTPException(404, "image not found") from None
    if not path.is_file():
        raise HTTPException(404, "image not found")
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(415, "unsupported image type")
    allowed = False
    for root in allowed_roots:
        try:
            allowed = os.path.commonpath((str(path), str(root))) == str(root)
        except ValueError:
            allowed = False
        if allowed:
            break
    if not allowed:
        raise HTTPException(403, "image path not allowed")
    return path


@lru_cache(maxsize=2048)
def _thumb(path: str, mtime_ns: int, width: int) -> bytes:
    from PIL import Image

    with Image.open(path) as source:
        source.thumbnail((width, width * 4))
        image = source.convert("RGB")
        output = io.BytesIO()
        image.save(output, "JPEG", quality=88, optimize=True)
    return output.getvalue()


@app.get("/img")
def image(path: str, w: int = 512, full: bool = False):
    resolved = _safe_path(path)
    if full:
        return FileResponse(resolved)
    width = max(32, min(int(w), MAX_THUMB_WIDTH))
    try:
        content = _thumb(str(resolved), resolved.stat().st_mtime_ns, width)
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
        while True:
            page = repository.groups(
                GroupFilters(build_id=build["build_id"]),
                page_number,
                MAX_PAGE_SIZE,
            )
            for item in page["items"]:
                detail = repository.group(item["group_id"])
                if detail is None:
                    raise RuntimeError("canonical group detail unavailable")
                error = canonical_group_error(detail["group"], detail["candidates"])
                if error is not None:
                    raise RuntimeError(f"canonical group invariant failed: {error}")
                sft = detail["sft"]
                winner_ids = set(detail["group"]["winner_ids"])
                if len(sft) > 2 or any(
                    row.get("candidate_id") not in winner_ids for row in sft
                ):
                    raise RuntimeError("canonical SFT invariant failed")
            if page_number * MAX_PAGE_SIZE >= page["total"]:
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
    parser.add_argument("--allow-root", action="append", help="additional image root")
    parser.add_argument("--host", default=os.environ.get("DBV_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("DBV_PORT", "8077")))
    parser.add_argument("--selfcheck", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    postgres_dsn: str | None = None
    roots: tuple[Path, ...]
    if args.config:
        output_root, postgres_dsn = _load_databuild_config(args.config)
        roots = (output_root,)
    elif args.build_root:
        roots = tuple(Path(path).expanduser().resolve() for path in args.build_root)
    else:
        roots = _build_roots
    image_roots = tuple(Path(path).expanduser().resolve() for path in args.allow_root or ())
    configure(
        build_roots=roots,
        postgres_dsn=postgres_dsn,
        image_roots=(DEFAULT_DATA_ROOT, *roots, *image_roots),
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
