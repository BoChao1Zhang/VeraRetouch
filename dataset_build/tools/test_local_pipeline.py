from __future__ import annotations

import numpy as np

from dataset_build.src.construct import local_pipeline, mask_synth, qa, subject_geom


def _plan():
    geo = {"ZeroX": 0.2, "ZeroY": 0.5, "FullX": 0.8, "FullY": 0.5}
    return [
        {"mask_type": "circulargradient", "geom": {"Top": 0.2, "Bottom": 0.8,
         "Left": 0.2, "Right": 0.8, "Feather": 60.0, "Flipped": "true"},
         "amount": 1.0, "_mode": "radial"},
        {"mask_type": "semantic", "alpha": np.ones((4, 4), np.float32),
         "amount": 1.0, "_mode": "semantic", "_feather": {"f_in": 0.02}},
        {"mask_type": "circulargradient", "geom": dict(geo), "amount": 1.0, "_mode": "band"},
        {"mask_type": "circulargradient", "geom": dict(geo), "amount": 1.0, "_mode": "band"},
        {"mask_type": "gradient", "geom": dict(geo), "amount": 1.0, "_mode": "linear"},
        {"mask_type": "gradient", "geom": dict(geo), "amount": 1.0, "_mode": "linear"},
        {"mask_type": "gradient", "geom": dict(geo), "amount": 1.0, "_mode": "linear"},
        {"mask_type": "gradient", "geom": dict(geo), "amount": 1.0, "_mode": "linear_bisect"},
    ]


def _rows(plan):
    return [{"mask_unit_id": f"m{i}", "after_path": f"/r/{i}.jpg", "cgt_path": None,
             "spec": spec, "geom": spec.get("geom"), "region": "中心",
             "blend_mode": "exact", "engine": "gpu_local_preset", "variant_index": i}
            for i, spec in enumerate(plan)]


def test_shortlist_conservative_policy():
    plan = _plan()
    rendered = _rows(plan)
    # q: band m3 强于 m2；linear m5 最佳；其余高分是 m6
    qres = {"scores": {f"m{i}": {"q": q} for i, q in
                       enumerate([0.5, 0.4, 0.55, 0.62, 0.58, 0.70, 0.66, 0.30])}}
    keep = local_pipeline.shortlist(plan, rendered, qres)
    assert len(keep) == 5
    modes = [plan[i]["_mode"] for i in keep]
    # 语义 + 径向必进；最佳束状 m3；最佳线性 m5；剩余最高 m6
    assert "semantic" in modes and "radial" in modes
    assert 3 in keep and 5 in keep and 6 in keep


def test_shortlist_handles_all_bisect_group():
    plan = [{"mask_type": "gradient", "geom": {"ZeroX": 0.1, "ZeroY": 0.5,
             "FullX": 0.9, "FullY": 0.5}, "amount": 1.0, "_mode": "linear_bisect"}
            for _ in range(8)]
    rendered = _rows(plan)
    qres = {"scores": {f"m{i}": {"q": i / 10} for i in range(8)}}
    keep = local_pipeline.shortlist(plan, rendered, qres)
    assert len(keep) == 5
    assert keep[0] == 7 or 7 in keep  # 最高分入选


def test_process_one_two_level_flow(monkeypatch, tmp_path):
    plan = _plan()
    calls = {"render": [], "qa": []}

    monkeypatch.setattr(subject_geom, "sample_plan",
                        lambda _p, _rng, n=8, cache_dir=None: plan)
    monkeypatch.setattr(local_pipeline, "_preview_source",
                        lambda p, d: "/pv/src.jpg")

    def fake_make(source, base, got_plan, out_dir, save_cgt=True, store=True):
        calls["render"].append((source, len(got_plan), save_cgt))
        assert store == save_cgt  # 预览级同时关闭 CGT 与永久存储
        rows = _rows(got_plan)
        if save_cgt:
            for r in rows:
                r["cgt_path"] = f"/cgt/{r['mask_unit_id']}.png"
        return rows

    monkeypatch.setattr(mask_synth, "make_local_samples", fake_make)

    def fake_qa(source, variants, is_portrait=False):
        calls["qa"].append((source, len(variants)))
        return {"scores": {mid: {"q": 0.5 + i * 0.01}
                           for i, (mid, _) in enumerate(variants)}}

    monkeypatch.setattr(qa, "qa_rank", fake_qa)

    base = {"preset_id": "b1", "path": "/p/b.xmp", "fmt": "xmp",
            "preset_content_hash": "h"}
    g = local_pipeline.process_one({"path": "/img/s.jpg", "asset_id": "a1"},
                                   base, str(tmp_path))
    assert g is not None
    # 预览 8（不存 CGT）→ native 5（存 CGT）
    assert calls["render"] == [("/pv/src.jpg", 8, False), ("/img/s.jpg", 5, True)]
    assert calls["qa"] == [("/pv/src.jpg", 8), ("/img/s.jpg", 5)]
    assert g["two_level"] == {"preview_n": 8, "shortlist_n": 5, "final_n": 2}
    assert len(g["candidates"]) == 2
    for c in g["candidates"]:
        assert c["kind"] == "local_preset"
        assert c["local"]["cgt_path"].startswith("/cgt/")
        assert c["local"]["base_preset_id"] == "b1"
