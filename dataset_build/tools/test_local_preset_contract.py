from __future__ import annotations

import threading

import numpy as np
import pytest
from PIL import Image

from dataset_build.src.construct import agent, mask_synth, subject_geom, tier
from gpu_render.gpu.local_preset import raster_cgt_batch


@pytest.mark.parametrize("spec", [
    {
        "mask_type": "circulargradient",
        "geom": {
            "Top": 0.17, "Bottom": 0.83, "Left": 0.21, "Right": 0.79,
            "Angle": 17.0, "Feather": 65.0, "Flipped": "true",
        },
    },
    {
        "mask_type": "gradient",
        "geom": {
            "ZeroX": 0.18, "ZeroY": 0.42, "FullX": 0.81, "FullY": 0.61,
            "Flipped": "false",
        },
    },
])
def test_construct_cgt_matches_gpu_composite_alpha(spec):
    expected = mask_synth.cgt_raster(
        spec["mask_type"], spec["geom"], 123, 177)
    actual = raster_cgt_batch(spec, 123, 177, "cpu")[0, 0].numpy()

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=5e-7)
    np.testing.assert_array_equal(
        (actual * 255).astype(np.uint8),
        (expected * 255).astype(np.uint8),
    )


def test_make_local_samples_passes_plan_specs_and_cgt_paths(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    Image.fromarray(np.full((12, 16, 3), 96, dtype=np.uint8)).save(source)
    captured = {}

    def fake_render(base_path, fmt, source_path, specs, preset_id=None, store=True):
        captured.update(base_path=base_path, fmt=fmt, source_path=source_path,
                        specs=specs, preset_id=preset_id, store=store)
        results = []
        for i, spec in enumerate(specs):
            out = tmp_path / f"rendered_{i}.png"
            Image.open(source).save(out)
            # 契约 §2：后端产出 C_GT PNG 并回填 cgt_path
            cgt = spec.get("cgt_path")
            if cgt:
                Image.fromarray(np.zeros((12, 16), np.uint8)).save(cgt)
            results.append({"ok": True, "after_path": str(out),
                            "cgt_path": cgt, "engine": "gpu_local_preset"})
        return {"ok": True, "route": "local", "results": results}

    monkeypatch.setattr(
        mask_synth.render, "render_local_preset_variants", fake_render)
    plan = [
        {"mask_type": "circulargradient", "what": "Mask/CircularGradient",
         "geom": {"Top": 0.2, "Bottom": 0.8, "Left": 0.25, "Right": 0.75,
                  "Angle": 10.0, "Feather": 70.0, "Flipped": "true"},
         "amount": 1.0, "_mode": "radial", "_apply": "inside"},
        {"mask_type": "semantic",
         "alpha": np.pad(np.ones((6, 8), np.float32), ((3, 3), (4, 4))),
         "amount": 1.0, "_mode": "semantic", "_apply": "inside"},
    ]
    rows = mask_synth.make_local_samples(
        str(source),
        {"path": "/presets/base.xmp", "fmt": "xmp", "preset_id": "base-id"},
        plan, str(tmp_path))

    assert captured["base_path"] == "/presets/base.xmp"
    assert captured["preset_id"] == "base-id"
    assert len(captured["specs"]) == 2
    geo_spec, sem_spec = captured["specs"]
    assert geo_spec["mask_type"] == "circulargradient"
    assert geo_spec["geom"] == plan[0]["geom"]
    assert geo_spec["amount"] == 1.0
    assert geo_spec["cgt_path"].endswith(".png")
    assert sem_spec["mask_type"] == "semantic"
    assert sem_spec["alpha"] is plan[1]["alpha"]
    # 私有 _ 前缀元数据不进渲染 spec
    assert not any(k.startswith("_") for k in geo_spec)

    assert len(rows) == 2
    assert rows[0]["blend_mode"] == "exact"
    assert rows[0]["engine"] == "gpu_local_preset"
    assert rows[0]["cgt_path"] == geo_spec["cgt_path"]
    assert rows[1]["spec"]["mask_type"] == "semantic"
    assert rows[1]["region"] == "中心"
    assert "local_params" not in rows[0]


def test_recipe_distinguishes_full_local_preset_from_legacy_local_params():
    common = {
        "mask_unit_id": "mask-id",
        "mask_type": "gradient",
        "geom": {"ZeroX": 0.2, "ZeroY": 0.5, "FullX": 0.8, "FullY": 0.5},
        "base_preset_id": "base-id",
        "base_preset_path": "/presets/base.xmp",
        "base_preset_content_hash": "base-hash",
    }
    full = tier._recipe_of({"local": {**common, "blend_mode": "exact"}})
    legacy = tier._recipe_of({
        "local": {**common, "local_params": {"LocalExposure2012": 0.8}}
    })

    assert full == {
        "kind": "local_preset",
        "mask_type": "gradient",
        "geom": common["geom"],
        "blend_mode": "exact",
        "base_preset_id": "base-id",
        "base_preset_path": "/presets/base.xmp",
        "base_preset_content_hash": "base-hash",
    }
    assert legacy["kind"] == "local_param"
    assert legacy["local_params"] == {"LocalExposure2012": 0.8}


def test_process_source_local_emits_local_preset_candidates(monkeypatch):
    base = {
        "preset_id": "base-id", "kind": "param", "fmt": "xmp",
        "path": "/presets/base.xmp", "has_local_mask": 0,
        "preset_content_hash": "base-hash",
        "axes": {"grade_family": "clean_natural"},
    }
    farm = {
        "preset_id": "farm-id", "kind": "param", "fmt": "xmp",
        "path": "/presets/farm.xmp", "has_local_mask": 0,
        "axes": {"grade_family": "stylized"},
    }
    pool = [farm, base] + [
        {"preset_id": f"extra-{i}", "kind": "param", "fmt": "xmp",
         "path": f"/presets/extra-{i}.xmp", "has_local_mask": 0,
         "axes": {"grade_family": "stylized"}}
        for i in range(38)
    ]

    class FakeSelector(agent.Selector):
        def __init__(self):
            self._quota = agent.mixing.FamilyQuota()
            self._quota_lock = threading.Lock()
            self.pool_k = None
            self.selected_k = None

        def candidate_pool(self, _path, k):
            self.pool_k = k
            return pool

        def select_candidates(self, candidates, k):
            self.selected_k = k
            return super().select_candidates(candidates, k)

        def _is_farm(self, candidate):
            return candidate["preset_id"] == "farm-id"

    plan = [
        {"mask_type": "gradient", "what": "Mask/Gradient",
         "geom": {"ZeroX": 0.2, "ZeroY": 0.5, "FullX": 0.8, "FullY": 0.5},
         "amount": 1.0, "_mode": "linear", "_apply": "subject_side",
         "_subject": {"concept": "person"}},
        {"mask_type": "semantic", "alpha": np.ones((4, 4), np.float32),
         "amount": 1.0, "_mode": "semantic", "_apply": "inside",
         "_feather": {"f_in": 0.02}, "_subject": {"concept": "person"}},
    ]
    monkeypatch.setattr(
        subject_geom, "sample_plan",
        lambda _path, _rng, n=8, cache_dir=None: [dict(p) for p in plan[:n]])

    def fake_make(_source, selected, got_plan, _cgt_dir):
        assert selected is base
        return [
            {"mask_unit_id": f"mask-{i}", "after_path": f"/renders/{i}.jpg",
             "cgt_path": f"/cgt/{i}.png", "spec": spec, "geom": spec.get("geom"),
             "region": "中心", "blend_mode": "exact",
             "engine": "gpu_local_preset", "variant_index": i}
            for i, spec in enumerate(got_plan)
        ]

    monkeypatch.setattr(mask_synth, "make_local_samples", fake_make)
    monkeypatch.setattr(
        agent.qa, "qa_rank",
        lambda _source, variants, is_portrait=False: {
            "scores": {mask_id: {"q": 0.8} for mask_id, _ in variants}
        })

    selector = FakeSelector()
    group = agent.process_source_local(
        selector, {"path": "/images/source.jpg", "asset_id": "source-id"},
        2, "/cgt")

    assert group is not None
    assert selector.pool_k == 40
    assert selector.selected_k == 1
    assert selector._quota.total == 1
    assert dict(selector._quota.counts) == {"clean_natural": 1}
    assert len(group["candidates"]) == 2
    for candidate in group["candidates"]:
        assert candidate["kind"] == "local_preset"
        assert candidate["preset_path"] == base["path"]
        assert "local_params" not in candidate["local"]
        assert candidate["local"]["blend_mode"] == "exact"
        assert candidate["local"]["base_preset_content_hash"] == "base-hash"
    geo_c, sem_c = group["candidates"]
    assert geo_c["local"]["mask_type"] == "gradient"
    assert geo_c["local"]["geom"] == plan[0]["geom"]
    assert sem_c["local"]["mask_type"] == "semantic"
    assert sem_c["local"]["geom"] == {"semantic": True}
    assert sem_c["local"]["feather"] == {"f_in": 0.02}
