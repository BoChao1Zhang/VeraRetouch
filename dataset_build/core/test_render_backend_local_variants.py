from __future__ import annotations

import json
import os

import numpy as np
import pytest
from PIL import Image

from dataset_build.core import render_backend as rb


def _write_fixture(tmp_path):
    preset = tmp_path / "base.xmp"
    preset.write_text(
        '<rdf:Description xmlns:rdf="x" xmlns:crs="y" '
        'crs:Exposure2012="+0.5"/>', encoding="utf-8")
    yy, xx = np.mgrid[:18, :24]
    source = np.stack((xx * 7, yy * 9, (xx + yy) * 4), axis=-1).astype(np.uint8)
    source_path = tmp_path / "source.png"
    Image.fromarray(source, "RGB").save(source_path)
    return preset, source_path, source


def _gradient(amount=1.0):
    return {
        "mask_type": "gradient",
        "geom": {"ZeroX": 0.0, "ZeroY": 0.0,
                 "FullX": 1.0, "FullY": 0.0, "Flipped": "false"},
        "amount": amount,
    }


def test_embedded_local_base_is_rejected_as_batch_error(tmp_path, monkeypatch):
    preset, source_path, _source = _write_fixture(tmp_path)
    backend = rb.RenderBackend(policy="throughput")

    monkeypatch.setattr(backend, "_base_has_embedded_locals", lambda _path: True)
    monkeypatch.setattr(
        backend, "_local_capable",
        lambda *_args: (_ for _ in ()).throw(AssertionError("route must not run")))
    monkeypatch.setattr(
        backend, "_render_local_preset_gpu",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("GPU must not render embedded locals")))
    monkeypatch.setattr(
        backend, "_render_local_preset_farm",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("farm must not render embedded locals")))

    out_zero = tmp_path / "zero.png"
    out_masked = tmp_path / "masked.png"
    result = backend.render_local_variants(
        str(preset), "xmp", str(source_path),
        [{"out_path": str(out_zero), "spec": _gradient(amount=0.0)},
        {"out_path": str(out_masked), "spec": _gradient(amount=1.0)}],
        preset_id="embedded")

    assert result["ok"] is False
    assert result["route"] == "none"
    assert result["n_local"] == 0
    assert result["n_farm"] == 0
    assert [r["error_code"] for r in result["results"]] == [
        "base_preset_has_embedded_locals", "base_preset_has_embedded_locals"]
    assert all("embedded Lightroom local corrections" in r["error"]
               for r in result["results"])
    assert not out_zero.exists()
    assert not out_masked.exists()
    assert backend.stats_snapshot()["failed"] == 2


def test_fidelity_policy_farms_heuristically_capable_base(tmp_path, monkeypatch):
    preset, source_path, _source = _write_fixture(tmp_path)
    backend = rb.RenderBackend(policy="fidelity")
    farm_calls = []

    monkeypatch.setattr(rb, "has_residual", lambda _pid: False)
    monkeypatch.setattr(backend, "_base_has_embedded_locals", lambda _path: False)
    monkeypatch.setattr(
        backend, "_local_capable",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("fidelity must not use heuristic routing")))
    monkeypatch.setattr(
        rb, "_gpu1_free_mb",
        lambda: (_ for _ in ()).throw(
            AssertionError("fidelity must not enter the GPU route")))

    def fake_farm(_preset, _fmt, _source, specs, out_paths, _pid, cgt_paths=None):
        farm_calls.append((list(specs), list(out_paths)))
        return [
            {"ok": True, "out_path": dst, "after_path": dst,
             "engine": "farm_local_composite"}
            for dst in out_paths
        ]

    monkeypatch.setattr(backend, "_render_local_preset_farm", fake_farm)
    result = backend.render_local_variants(
        str(preset), "xmp", str(source_path),
        [{"out_path": str(tmp_path / "fidelity.png"), "spec": _gradient()}],
        preset_id="uncalibrated")

    assert result["ok"] is True
    assert result["route"] == "farm"
    assert result["route_reason"] == "fidelity policy requires a dedicated residual"
    assert result["n_local"] == 0
    assert result["n_farm"] == 1
    assert len(farm_calls) == 1


def test_farm_composite_requests_smoothstep_raster(tmp_path, monkeypatch):
    preset, source_path, source = _write_fixture(tmp_path)
    backend = rb.RenderBackend(policy="fidelity")
    raster_modes = []

    def fake_farm(_preset, _fmt, _source, dst):
        Image.fromarray(np.full_like(source, 220), "RGB").save(dst, "JPEG")
        return {"ok": True, "after_path": dst, "engine": "fake_lr"}

    from gpu_render import local_replay

    real_raster_alpha = local_replay.raster_alpha

    def tracked_raster_alpha(*args, **kwargs):
        raster_modes.append(kwargs.get("smoothstep"))
        return real_raster_alpha(*args, **kwargs)

    monkeypatch.setattr(backend, "_render_farm_one", fake_farm)
    monkeypatch.setattr(local_replay, "raster_alpha", tracked_raster_alpha)

    out_zero = tmp_path / "farm_zero.png"
    out_masked = tmp_path / "farm_smooth.png"
    results = backend._render_local_preset_farm(
        str(preset), "xmp", str(source_path),
        [_gradient(amount=0.0), _gradient(amount=1.0)],
        [str(out_zero), str(out_masked)], "base")

    assert all(row["ok"] for row in results)
    assert raster_modes == [True, True]
    np.testing.assert_array_equal(np.asarray(Image.open(out_zero)), source)
    assert not np.array_equal(np.asarray(Image.open(out_masked)), source)


def test_throughput_policy_uses_global_residual_for_capable_base(
        tmp_path, monkeypatch):
    preset, source_path, source = _write_fixture(tmp_path)
    backend = rb.RenderBackend(policy="throughput")
    gpu_calls = []

    monkeypatch.setattr(rb, "has_residual", lambda _pid: False)
    monkeypatch.setattr(rb, "_gpu1_free_mb", lambda: rb.LOCAL_MIN_FREE_MB + 1)
    monkeypatch.setattr(backend, "_base_has_embedded_locals", lambda _path: False)
    monkeypatch.setattr(backend, "_local_capable", lambda _path, _fmt: True)

    def fake_gpu(_preset, _fmt, _source, specs, out_paths, residual_id,
                 cgt_paths=None):
        gpu_calls.append((list(specs), list(out_paths), residual_id))
        for dst in out_paths:
            Image.fromarray(source, "RGB").save(dst)
        return {"residual_id": residual_id}

    monkeypatch.setattr(backend, "_render_local_preset_gpu", fake_gpu)
    result = backend.render_local_variants(
        str(preset), "xmp", str(source_path),
        [{"out_path": str(tmp_path / "throughput.png"), "spec": _gradient()}],
        preset_id="uncalibrated")

    assert result["ok"] is True
    assert result["route"] == "local"
    assert result["route_reason"] == "well-covered base with global residual"
    assert result["n_local"] == 1
    assert result["n_farm"] == 0
    assert len(gpu_calls) == 1
    assert gpu_calls[0][2] == "_global"


def test_dedicated_residual_routes_all_variants_through_one_gpu_call(
        tmp_path, monkeypatch):
    preset, source_path, source = _write_fixture(tmp_path)
    backend = rb.RenderBackend()
    calls = []

    monkeypatch.setattr(rb, "has_residual", lambda pid: pid == "calibrated")
    monkeypatch.setattr(rb, "_gpu1_free_mb", lambda: rb.LOCAL_MIN_FREE_MB + 1)
    monkeypatch.setattr(backend, "_base_has_embedded_locals", lambda _path: False)

    def fake_gpu(_preset, _fmt, _source, specs, out_paths, residual_id,
                 cgt_paths=None):
        calls.append((list(specs), list(out_paths), residual_id))
        for dst in out_paths:
            Image.fromarray(source, "RGB").save(dst)
        return {"variants": len(specs), "residual_id": residual_id,
                "consumed": {"Exposure2012"}, "alpha": "must-not-escape"}

    monkeypatch.setattr(backend, "_render_local_preset_gpu", fake_gpu)
    variants = [
        {"out_path": str(tmp_path / "a.png"), "spec": _gradient()},
        {"out_path": str(tmp_path / "b.png"), "spec": _gradient(0.5)},
    ]
    result = backend.render_local_variants(
        str(preset), "xmp", str(source_path), variants, preset_id="calibrated")

    assert result["ok"] is True
    assert result["route"] == "local"
    assert result["n_local"] == 2
    assert result["n_farm"] == 0
    assert len(calls) == 1
    assert calls[0][2] == "calibrated"
    assert all(r["engine"] == "gpu_local_preset" for r in result["results"])
    assert "alpha" not in result["local_info"]
    assert result["local_info"]["consumed"] == ["Exposure2012"]
    json.dumps(result)


@pytest.mark.parametrize("bad_alpha", [
    None,                                # 缺失
    [[0.0, 1.0], [0.5, 0.5]],            # list 不是 ndarray
    np.zeros((4, 6), np.uint8),          # 非 float dtype
    np.zeros((2, 4, 6), np.float32),     # 非 2D
])
def test_semantic_spec_rejects_bad_alpha(tmp_path, monkeypatch, bad_alpha):
    preset, source_path, _source = _write_fixture(tmp_path)
    backend = rb.RenderBackend()
    monkeypatch.setattr(backend, "_base_has_embedded_locals", lambda _path: False)

    spec = {"mask_type": "semantic", "amount": 1.0}
    if bad_alpha is not None:
        spec["alpha"] = bad_alpha
    result = backend.render_local_variants(
        str(preset), "xmp", str(source_path),
        [{"out_path": str(tmp_path / "sem.png"), "spec": spec}],
        preset_id="calibrated")

    assert result["ok"] is False
    assert result["results"][0]["error_code"] == "invalid_variant"
    assert "alpha" in result["results"][0]["error"]


def test_semantic_spec_is_clipped_and_cgt_paths_forwarded(tmp_path, monkeypatch):
    preset, source_path, source = _write_fixture(tmp_path)
    backend = rb.RenderBackend()
    calls = []

    monkeypatch.setattr(rb, "has_residual", lambda pid: pid == "calibrated")
    monkeypatch.setattr(rb, "_gpu1_free_mb", lambda: rb.LOCAL_MIN_FREE_MB + 1)
    monkeypatch.setattr(backend, "_base_has_embedded_locals", lambda _path: False)

    def fake_gpu(_preset, _fmt, _source, specs, out_paths, residual_id,
                 cgt_paths=None):
        calls.append((list(specs), list(out_paths), list(cgt_paths)))
        for dst in out_paths:
            Image.fromarray(source, "RGB").save(dst)
        return {"residual_id": residual_id}

    monkeypatch.setattr(backend, "_render_local_preset_gpu", fake_gpu)
    alpha = np.linspace(-0.5, 1.5, 9 * 12, dtype=np.float64).reshape(9, 12)
    cgt_sem = str(tmp_path / "sem_cgt.png")
    variants = [
        {"out_path": str(tmp_path / "sem.png"), "cgt_path": cgt_sem,
         "spec": {"mask_type": "semantic", "alpha": alpha, "amount": 0.8}},
        {"out_path": str(tmp_path / "geo.png"), "spec": _gradient()},
    ]
    result = backend.render_local_variants(
        str(preset), "xmp", str(source_path), variants, preset_id="calibrated")

    assert result["ok"] is True
    specs, _out_paths, cgt_paths = calls[0]
    assert specs[0]["mask_type"] == "semantic"
    assert specs[0]["alpha"].dtype == np.float32
    assert float(specs[0]["alpha"].min()) == 0.0    # 值域裁剪 [0,1]
    assert float(specs[0]["alpha"].max()) == 1.0
    assert specs[0]["amount"] == 0.8
    assert cgt_paths == [cgt_sem, None]
    # 结果 row 顺序不变量 + 回填 cgt_path
    assert [r["out_path"] for r in result["results"]] == [
        v["out_path"] for v in variants]
    assert [r.get("cgt_path") for r in result["results"]] == [cgt_sem, None]


def test_gpu_route_writes_cgt_png_and_encodes_outside_lock(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import gpu_render.gpu.gpu_replay as gpu_replay
    import gpu_render.gpu.local_preset as local_preset_mod

    preset, source_path, source = _write_fixture(tmp_path)
    backend = rb.RenderBackend()
    h, w = source.shape[:2]

    alpha = torch.zeros(2, 1, h, w)
    alpha[1, 0, :, : w // 2] = 0.6
    out_t = torch.rand(2, 3, h, w)

    def fake_render_tensor(source_t, _preset, specs, _fits, residual_id=None,
                           fallback="cpu"):
        assert tuple(source_t.shape) == (1, 3, h, w)
        assert len(specs) == 2
        return out_t, {"alpha": alpha, "consumed": set()}

    lock_states = {"rgb": [], "cgt": []}
    real_rgb = rb.RenderBackend._save_rgb_u8
    real_alpha = rb.RenderBackend._save_alpha_png

    def gpu_slots_exhausted():
        locked = getattr(backend._gpu_lock, "locked", None)
        if callable(locked):
            return locked()
        acquired = backend._gpu_lock.acquire(blocking=False)
        if acquired:
            backend._gpu_lock.release()
        return not acquired

    def spy_rgb(arr, dst, quality=92):
        lock_states["rgb"].append(gpu_slots_exhausted())
        real_rgb(arr, dst, quality)

    def spy_alpha(arr, dst):
        lock_states["cgt"].append(gpu_slots_exhausted())
        real_alpha(arr, dst)

    monkeypatch.setattr(gpu_replay, "DEVICE", "cpu")
    monkeypatch.setattr(
        local_preset_mod, "render_local_preset_tensor", fake_render_tensor)
    monkeypatch.setattr(
        backend, "_parse_cached", lambda _p, _f: {"attrs": {}, "curves": {}})
    monkeypatch.setattr(backend, "_save_rgb_u8", spy_rgb)
    monkeypatch.setattr(backend, "_save_alpha_png", spy_alpha)

    out_a, out_b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    cgt_b = tmp_path / "cgt_b.png"
    info = backend._render_local_preset_gpu(
        str(preset), "xmp", str(source_path),
        [_gradient(), _gradient(0.5)], [str(out_a), str(out_b)],
        residual_id="calibrated", cgt_paths=[None, str(cgt_b)])

    assert "alpha" not in info
    assert out_a.exists() and out_b.exists()
    with Image.open(cgt_b) as im:
        assert im.mode == "L"
        saved = np.asarray(im)
    # 实际合成用 alpha 的 u8 截断（(α*255) 向零取整），仅请求的 variant 产出
    np.testing.assert_array_equal(
        saved, (alpha[1, 0].numpy() * 255).astype(np.uint8))
    # JPEG/PNG 编码写盘发生在 GPU 锁外（性能项4+5）
    assert lock_states["rgb"] == [False, False]
    assert lock_states["cgt"] == [False]


def test_farm_semantic_composite_resizes_alpha_and_writes_cgt(
        tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    preset, source_path, source = _write_fixture(tmp_path)
    backend = rb.RenderBackend(policy="fidelity")

    def fake_farm(_preset, _fmt, _source, dst):
        Image.fromarray(np.full_like(source, 220), "RGB").save(dst, "JPEG")
        return {"ok": True, "after_path": dst, "engine": "fake_lr"}

    monkeypatch.setattr(backend, "_render_farm_one", fake_farm)

    h, w = source.shape[:2]
    small = np.zeros((h // 2, w // 2), np.float32)   # 降采样语义 alpha
    small[:, (w // 2) // 2:] = 1.0                    # 右半选中，左半 α=0
    spec = {"mask_type": "semantic", "alpha": small, "amount": 1.0}

    out_sem, out_geo = tmp_path / "sem.png", tmp_path / "geo.png"
    cgt_sem = tmp_path / "sem_cgt.png"
    results = backend._render_local_preset_farm(
        str(preset), "xmp", str(source_path),
        [spec, _gradient(amount=0.0)],
        [str(out_sem), str(out_geo)], "base",
        cgt_paths=[str(cgt_sem), None])

    assert [row["ok"] for row in results] == [True, True]
    # 结果 row 顺序不变量 + cgt_path 回填
    assert [row["out_path"] for row in results] == [str(out_sem), str(out_geo)]
    assert results[0]["cgt_path"] == str(cgt_sem)
    assert results[1]["cgt_path"] is None

    alpha_full = np.clip(
        cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR), 0.0, 1.0)
    out_arr = np.asarray(Image.open(out_sem))
    zero, one = alpha_full <= 0.0, alpha_full >= 1.0
    assert zero.any() and one.any()
    # α=0 处逐字节等于源图（PNG 无损）；α=1 区域被编辑
    np.testing.assert_array_equal(out_arr[zero], source[zero])
    assert not np.array_equal(out_arr[one], source[one])
    with Image.open(cgt_sem) as im:
        assert im.mode == "L"
        saved = np.asarray(im)
    np.testing.assert_array_equal(saved, (alpha_full * 255.0).astype(np.uint8))
    # amount=0 的几何 variant：输出仍等于源图
    np.testing.assert_array_equal(np.asarray(Image.open(out_geo)), source)


def test_route_cache_key_includes_mtime_and_fmt(tmp_path, monkeypatch):
    preset = tmp_path / "cache.xmp"
    preset.write_text("first", encoding="utf-8")
    backend = rb.RenderBackend()
    calls = []

    import gpu_render.route as route

    def fake_route(path, fmt):
        calls.append((path, fmt))
        return {"route": "local"}

    monkeypatch.setattr(route, "route_preset", fake_route)
    assert backend._local_capable(str(preset), "xmp") is True
    assert backend._local_capable(str(preset), "xmp") is True
    assert backend._local_capable(str(preset), "lrtemplate") is True

    stat = os.stat(preset)
    os.utime(preset, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert backend._local_capable(str(preset), "xmp") is True
    assert [fmt for _, fmt in calls] == ["xmp", "lrtemplate", "xmp"]
