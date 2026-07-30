# Canonical Databuild Viewer

Read-only inspection of canonical databuild artifacts. The backend prefers the
`canonical_*` PostgreSQL projection and falls back to `groups.jsonl`, `sft.jsonl`,
`failures.jsonl`, and `manifest.json` when PostgreSQL is unavailable.

The inspector exposes every complete eight-candidate group, source and rendered
images, local `C_GT` and mask overlay views, visibility and OneAlign metrics,
subject/region metadata, top-2 SFT annotations, and structured failure events.
Filters cover build, render mode, preset format and taxonomy, annotation queue,
failure state, and winner rank.

## Backend

Use the same owner-only canonical TOML as the build. Its `output_root` supplies the
JSONL fallback and `[viewer].postgres_dsn` supplies the projection connection.

```bash
cd /home/bc/VeraRetouch
python -m pip install -e '.[viewer]'
python -m databuild_viewer.backend.app \
  --config /absolute/path/to/databuild.toml
```

The server listens on `127.0.0.1:8077` by default. `--host` and `--port` are
explicit overrides. The selfcheck is read-only:

```bash
python -m databuild_viewer.backend.app \
  --config /absolute/path/to/databuild.toml --selfcheck
```

For a local JSONL-only build, omit `--config` and pass `--build-root`. Add image
roots outside the dataset/output roots explicitly:

```bash
python -m databuild_viewer.backend.app \
  --build-root /absolute/path/to/builds \
  --allow-root /absolute/path/to/source/images
```

`DBV_CONFIG`, `DBV_BUILD_ROOTS`, and `DBV_ALLOWED_ROOTS` provide equivalent
process-start configuration. `DBV_CONFIG` contains only a path; credentials remain
inside the ignored `0600` TOML.

## Frontend

```bash
cd databuild_viewer/frontend
npm install
npm run dev
```

Vite listens on `127.0.0.1:5173` and proxies `/api` and `/img` to
`http://127.0.0.1:8077`. For a single-port production-style run, build first and
then start the backend:

```bash
npm run build
```

The compiled SPA is included in the Python wheel; rebuild it only after frontend
source changes.

## Verification

```bash
uv run --no-progress --with fastapi==0.136.1 --with Pillow==12.2.0 \
  python -m unittest databuild_viewer.backend.test_app -v
cd databuild_viewer/frontend && npm run build
cd databuild_viewer/frontend && npm run test:e2e   # Playwright desktop/mobile screenshots
uv build --wheel --out-dir /tmp/veraretouch-wheel
unzip -l /tmp/veraretouch-wheel/*.whl | rg 'databuild_viewer/frontend/dist/'
```
