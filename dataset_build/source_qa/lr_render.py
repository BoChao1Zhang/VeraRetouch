"""Real-Lightroom render client for preset QA.

Drives the already-running JarvisEvo Lightroom task server (FastAPI, reverse
connection: Win/Mac LrC clients poll it). Flow:
    POST /api/submit_task_with_files?photo_path=&xmp_path=   (xmp_path is a config.lua)
    GET  /api/task_status/{id}?wait=N   until completed/failed
    GET  /api/download_task_result/{id} -> processed.jpg

The server downloads the submitted preset file to the client as `config.lua`, so
the LrC plugin applies a Lua develop-settings table. We therefore convert each
preset to that lua using JarvisEvo's OWN converters (so masks survive):
    .xmp        -> xmp2lua.parse_xmp  (preserves MaskGroupBasedCorrections etc.)
    .lrtemplate -> LuaConverter.from_lua -> value.settings -> to_lua
    .cube/.3dl  -> NOT sent to LR (no develop-preset form); rendered by numpy
                   trilinear in preset_qa (tier-3) instead.

A real LR develop preset includes local-mask corrections, so the 290 mask
presets render faithfully here — exactly the class the global teacher cannot do.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import threading
import time
import uuid
from typing import Optional

from . import config

# Global farm admission: ONE gate for every LR submission in this process
# (construct render threads, preset_qa via LrClient, ...). Sized by
# LR_MAX_CONCURRENCY (~2x online LrC clients): the server enforces one
# in-flight render per client, so extra width never causes write-lock
# collisions — it just keeps a small pending backlog on the server so no
# machine idles for a submit round-trip between renders. Held across the
# whole submit+poll cycle.
_FARM_GATE = threading.BoundedSemaphore(int(getattr(config, "LR_MAX_CONCURRENCY", 6)))

# lrc_scripts (LR task server + client + plugin + converters) is now part of THIS
# repo (moved out of JarvisEvo). Resolve it relative to the repo root.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LRC_ROOT = os.environ.get("SOURCE_QA_LRC_ROOT", os.path.join(_REPO, "lrc_scripts"))
_XMP2LUA = _LUACONV = None


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _xmp2lua():
    global _XMP2LUA
    if _XMP2LUA is None:
        _XMP2LUA = _load_module(
            os.path.join(LRC_ROOT, "clients/agent_to_lightroom/utils/xmp2lua.py"),
            "_lrc_xmp2lua")
    return _XMP2LUA


def _luaconv():
    global _LUACONV
    if _LUACONV is None:
        mod = _load_module(os.path.join(LRC_ROOT, "utils/lua_converter.py"), "_lrc_luaconv")
        _LUACONV = mod.LuaConverter
    return _LUACONV


# --------------------------------------------------------------------------- #
# preset file -> config.lua (develop-settings table the LrC plugin applies)
# --------------------------------------------------------------------------- #
def preset_to_configlua(recipe_path: str, fmt: str, out_path: str) -> bool:
    fmt = (fmt or "").lower()
    ext = os.path.splitext(recipe_path)[1].lower()
    try:
        if fmt == "xmp" or ext == ".xmp":
            lua_table = _xmp2lua().parse_xmp(recipe_path)        # masks preserved
            text = "return " + lua_table
        elif fmt == "lrtemplate" or ext == ".lrtemplate":
            with open(recipe_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            i = content.find("{")
            if i < 0:
                return False
            obj = _luaconv().from_lua(content[i:])
            settings = (obj.get("value") or {}).get("settings") if isinstance(obj, dict) else None
            if not settings:
                return False
            text = "return " + _luaconv().to_lua(settings)
        else:
            return False                                          # LUTs do not go to LR
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        return True
    except Exception as e:
        print(f"[lr_render] preset->lua failed for {recipe_path}: {e}", file=sys.stderr)
        return False


# --------------------------------------------------------------------------- #
# LR task server client
# --------------------------------------------------------------------------- #
def lr_health() -> bool:
    import requests
    try:
        r = requests.get(config.LR_SERVER_URL + "/api/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def submit_and_wait(photo_path: str, lua_path: str) -> dict:
    """Submit one render, long-poll to terminal state, download the after.
    Returns a dict with ok=True and after_path, or ok=False with structured error.
    Bounded by the global farm admission gate."""
    with _FARM_GATE:
        return _submit_and_wait(photo_path, lua_path)


def _submit_and_wait(photo_path: str, lua_path: str) -> dict:
    import requests
    base = config.LR_SERVER_URL
    task_id = None
    started = time.time()
    try:
        r = requests.post(f"{base}/api/submit_task_with_files",
                          params={"photo_path": photo_path, "xmp_path": lua_path},
                          timeout=config.LR_HTTP_TIMEOUT)
        r.raise_for_status()
        task_id = r.json()["task_id"]
    except Exception as e:
        print(f"[lr_render] submit failed: {e}", file=sys.stderr)
        return {"ok": False, "error_code": "submit_failed", "error": str(e), "retryable": True}

    deadline = time.time() + config.LR_JOB_TIMEOUT
    status = None
    js = {}
    while time.time() < deadline:
        try:
            s = requests.get(f"{base}/api/task_status/{task_id}",
                             params={"wait": config.LR_POLL_WAIT},
                             timeout=config.LR_POLL_WAIT + config.LR_HTTP_TIMEOUT)
            s.raise_for_status()
            js = s.json()
            status = js.get("status")
        except Exception:
            time.sleep(2)
            continue
        if status in ("completed", "failed"):
            break
    if status != "completed":
        print(f"[lr_render] task {task_id} ended status={status}", file=sys.stderr)
        result = js.get("result") if isinstance(js, dict) else None
        if isinstance(result, dict):
            data = result.get("result_data") or {}
            if not isinstance(data, dict):
                data = {}
            return {
                "ok": False,
                "task_id": task_id,
                "status": status or "timeout",
                "error_code": data.get("error_code") or f"task_{status or 'timeout'}",
                "error": result.get("error") or data.get("error") or f"LR task ended status={status}",
                "retryable": data.get("retryable", status != "failed"),
                "result": result,
                "elapsed": time.time() - started,
            }
        return {
            "ok": False,
            "task_id": task_id,
            "status": status or "timeout",
            "error_code": f"task_{status or 'timeout'}",
            "error": f"LR task ended status={status}",
            "retryable": status != "failed",
            "elapsed": time.time() - started,
        }

    # download the rendered after
    out_path = os.path.join(config.RENDER_STAGE, f"{task_id}.jpg")
    try:
        d = requests.get(f"{base}/api/download_task_result/{task_id}",
                         timeout=config.LR_HTTP_TIMEOUT, stream=True)
        d.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in d.iter_content(1024 * 256):
                f.write(chunk)
    except Exception as e:
        print(f"[lr_render] download failed for {task_id}: {e}", file=sys.stderr)
        return {
            "ok": False,
            "task_id": task_id,
            "error_code": "download_failed",
            "error": str(e),
            "retryable": True,
            "elapsed": time.time() - started,
        }
    return {"ok": True, "after_path": out_path, "task_id": task_id, "engine_version": "lrc",
            "elapsed": time.time() - started}


_RETRYABLE_SIGNS = (
    "write access", "withwriteaccessdo", "blocked by another", "could not execute action",
    "import and process", "write_access_failed", "timeout", "submit_failed", "download_failed",
)


def _is_retryable(res: dict) -> bool:
    """Transient LR failures worth re-submitting. The LrC catalog allows only one
    `withWriteAccessDo` action at a time; when a client gets overlapping renders the
    blocked one throws a `plugin_exception` ("could not execute action 'Import and
    Process Photo' … blocked by another write access call") that the plugin does NOT
    flag retryable -> it would otherwise drop the probe permanently. Re-submit after
    backoff so it lands on a now-free client."""
    if res.get("retryable"):
        return True
    blob = f"{res.get('error_code', '')} {res.get('error', '')}".lower()
    return any(s in blob for s in _RETRYABLE_SIGNS)


def render_via_lr(recipe_path: str, fmt: str, photo_path: str) -> dict:
    """High-level: convert preset -> config.lua, submit to LR, return the after.
    Bounded retry (config.RENDER_MAX_ATTEMPTS) on transient LrC write-lock / timeout."""
    os.makedirs(config.RENDER_STAGE, exist_ok=True)
    lua_path = os.path.join(config.RENDER_STAGE, f"preset_{uuid.uuid4().hex[:10]}.lua")
    if not preset_to_configlua(recipe_path, fmt, lua_path):
        return {"ok": False, "error_code": "preset_to_lua_failed",
                "error": "Failed to convert preset to Lightroom config.lua", "retryable": False}
    attempts = max(1, int(getattr(config, "RENDER_MAX_ATTEMPTS", 3)))
    res: dict = {}
    try:
        for i in range(attempts):
            res = submit_and_wait(photo_path, lua_path)
            if res.get("ok") or not _is_retryable(res) or i == attempts - 1:
                break
            backoff = min(2 ** i * 2, 12)        # 2s, 4s, 8s ...
            print(f"[lr_render] transient render fail ({res.get('error_code')}); "
                  f"retry {i + 1}/{attempts - 1} after {backoff}s", file=sys.stderr)
            time.sleep(backoff)
    finally:
        try:
            os.remove(lua_path)
        except OSError:
            pass
    if not res.get("ok"):
        res["attempts"] = attempts
    return res
