from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataset_build.contracts import (
    AfterSource,
    CgtRef,
    DegradeSpec,
    MaskSource,
    Provenance,
    RawDecode,
    Recipe,
    RecipeAsset,
    RecipeKind,
    Sample,
    SourceItem,
    StreamId,
)
from dataset_build import run as run_mod
from dataset_build.audit import _main as audit_main, audit_dataset
from dataset_build.pack import (
    ShardWriter,
    sample_to_jsonl,
    teacher_manifest_from_config,
    validate_manifest_index,
)
from dataset_build.sam3_precompute import DEFAULT_CORPORA, _iter_sources, _parse_corpora_arg
from dataset_build.reproduce import (
    ArrayRenderAdapter,
    inverse_search_z_star,
    reproduce_pair,
    region_composite_from_pair,
)
from dataset_build.run import MockParser, build_mock_context
from dataset_build.stage0_probe import (
    DryRunParamRenderer,
    _choose_scratch_dir,
    _main as stage0_probe_main,
    run_stage0_probe,
)
from dataset_build.streams import (
    InverseDegradeStream,
    MMArtTextStream,
    QAGate,
    RecipeXSourceStream,
    Tier1ExpertStream,
    _PlanInputs,
)
from dataset_build.vlm_clean import QwenVLCleaner


class _SpyCleaner:
    def __init__(self):
        self.calls = []

    def gen_instruction(self, *args, **kwargs):
        self.calls.append(("gen_instruction", args, kwargs))
        return {"instruction_long": "vlm", "instruction_short": "vlm", "lang": "en"}

    def reason_params(self, *args, **kwargs):
        self.calls.append(("reason_params", args, kwargs))
        return {"think": "vlm", "answer": {}}

    def verify(self, *args, **kwargs):
        self.calls.append(("verify", args, kwargs))
        return {"look_match": True, "param_sane": True, "processed_ok": True, "score": 1.0}

    def tag_scene_region(self, *args, **kwargs):
        self.calls.append(("tag_scene_region", args, kwargs))
        return {
            "scene": "any",
            "style": "",
            "sam3_concepts": ["sky", "face"],
            "masksubtype_hint": 0,
        }


class _SpyMasker:
    def __init__(self):
        self.calls = []

    def masks(self, image, concepts, **kwargs):
        import numpy as np

        self.calls.append((image, list(concepts), kwargs))
        mask = np.zeros((8, 8), dtype="float32")
        mask[:4, :] = 1.0
        return {c: mask.copy() for c in concepts}


class _FakeRenderer:
    def __init__(self, value=180):
        self.value = value
        self.calls = []

    def render(self, image_paths, param_dicts, **kwargs):
        import numpy as np

        self.calls.append((list(image_paths), list(param_dicts), kwargs))
        return [np.full((4, 4, 3), self.value, dtype="uint8") for _ in image_paths]


class _PathEchoRenderer:
    def __init__(self):
        self.calls = []

    def render(self, image_paths, param_dicts, **kwargs):
        import cv2

        self.calls.append((list(image_paths), list(param_dicts), kwargs))
        out = []
        for path in image_paths:
            bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return out


class _NullWriter:
    def __init__(self):
        self.samples = []
        self.rejects = []

    def write_cgt(self, *args, **kwargs):
        return None, None

    def done_ids(self):
        return []

    def write(self, sample):
        self.samples.append(sample)

    def log_reject(self, sample):
        self.rejects.append(sample)

    def flush(self):
        pass


def _ctx(tmp_path: Path):
    config = {
        "seed": 1234,
        "build_version": "test",
        "scratch_dir": str(tmp_path),
        "cgt": {"min_coverage": 0.01, "max_coverage": 0.95, "patch_grid": 16},
        "qa": {},
        "degrade": {"region_shapes": ["random_blob"]},
    }
    ctx = build_mock_context(config, _NullWriter())
    ctx.cleaner = _SpyCleaner()
    ctx.parser = MockParser()
    return ctx


@pytest.mark.parametrize(
    ("stream_id", "region_local"),
    [(StreamId.S1_DEGRADE_LOCAL, True), (StreamId.S7_DEGRADE_GLOBAL, False)],
)
def test_inverse_degrade_streams_do_not_call_vlm_cleaner(tmp_path, stream_id, region_local):
    ctx = _ctx(tmp_path)
    stream = InverseDegradeStream(
        ctx,
        stream_id=stream_id,
        inputs=_PlanInputs([]),
        region_local=region_local,
        gate=QAGate(ctx.config),
    )
    source = SourceItem(
        source_id="giant-quandian",
        path="/not/read/by/s1_or_s7.jpg",
        corpus="quandian",
        width=12000,
        height=9000,
        scene="any",
    )
    recipe = Recipe(
        kind=RecipeKind.PARAM,
        params={"Exposure2012": {"value": -25.0}},
        provenance=Provenance.DEGRADE,
        degrade=DegradeSpec(
            mode="gaussian_op",
            op_params={"Exposure2012": 25.0},
            aspects=["L"],
            forward=False,
            seed=1,
        ),
        meta={"aspects": ["L"]},
    )

    sample = stream.build_one(source, recipe, region_local, "sample_id", "pending", precomputed_after=None)

    assert sample is not None
    assert sample.instruction
    assert sample.think
    assert sample.answer == {"Exposure2012": {"value": 25.0}}
    assert sample.recipe.kind == RecipeKind.PARAM
    assert sample.recipe.provenance == Provenance.DEGRADE
    assert sample.recipe.params == {"Exposure2012": {"value": -25.0}}
    assert ctx.cleaner.calls == []


def test_inverse_degrade_plan_emits_teacher_param_recipe(tmp_path):
    ctx = _ctx(tmp_path)
    stream = InverseDegradeStream(
        ctx,
        stream_id=StreamId.S1_DEGRADE_LOCAL,
        inputs=_PlanInputs(
            [
                SourceItem(
                    source_id="src",
                    path="/not/read.jpg",
                    corpus="quandian",
                    width=64,
                    height=64,
                )
            ]
        ),
        region_local=True,
        gate=QAGate(ctx.config),
    )

    _source, recipe, region_local = next(stream.plan(1, 1234))

    assert region_local is True
    assert recipe.kind == RecipeKind.PARAM
    assert recipe.provenance == Provenance.DEGRADE
    assert recipe.degrade is not None
    assert recipe.params


def test_vlm_image_encoder_rejects_images_above_pixel_limit(tmp_path):
    from PIL import Image

    p = tmp_path / "too_large.jpg"
    Image.new("RGB", (20, 20), color=(128, 128, 128)).save(p)
    cleaner = QwenVLCleaner(
        image_longedge=8,
        max_image_pixels=100,
        image_encode_concurrency=1,
    )

    with pytest.raises(RuntimeError, match="image too large"):
        cleaner._img_to_data_uri(str(p))


def test_vlm_image_encoder_caches_thumbnail_data_uri(tmp_path, monkeypatch):
    import PIL.Image
    from PIL import Image

    p = tmp_path / "source.jpg"
    Image.new("RGB", (20, 20), color=(128, 128, 128)).save(p)
    cleaner = QwenVLCleaner(
        image_longedge=8,
        max_image_pixels=1000,
        image_encode_concurrency=1,
        image_cache_entries=1,
    )
    real_open = PIL.Image.open
    opened = []

    def spy_open(*args, **kwargs):
        opened.append(args[0])
        return real_open(*args, **kwargs)

    monkeypatch.setattr(PIL.Image, "open", spy_open)

    first = cleaner._img_to_data_uri(str(p))
    second = cleaner._img_to_data_uri(str(p))

    assert first == second
    assert len(opened) == 1
    assert len(cleaner._image_uri_cache) == 1


def test_run_stream_respects_max_outstanding_samples(monkeypatch, tmp_path):
    class TrackingFuture:
        def __init__(self, executor, fn, args, kwargs):
            self.executor = executor
            self.fn = fn
            self.args = args
            self.kwargs = kwargs
            self._done = False
            self._result = None

        def done(self):
            return self._done

        def result(self):
            if not self._done:
                self._result = self.fn(*self.args, **self.kwargs)
                self._done = True
                self.executor.active -= 1
            return self._result

    class TrackingExecutor:
        last = None

        def __init__(self, max_workers):
            self.max_workers = max_workers
            self.active = 0
            self.max_active = 0
            TrackingExecutor.last = self

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            return TrackingFuture(self, fn, args, kwargs)

    monkeypatch.setattr(run_mod, "ThreadPoolExecutor", TrackingExecutor)

    writer = _NullWriter()
    config = {
        "seed": 1234,
        "build_version": "test",
        "scratch_dir": str(tmp_path),
        "models": {"veraretouch": {"batch_size": 8}},
        "vllm": {"concurrency": 8, "max_outstanding_samples": 2},
        "qa": {"enable_verify": False},
        "cgt": {"min_coverage": 0.01, "max_coverage": 0.95, "patch_grid": 16},
        "degrade": {"region_shapes": ["random_blob"]},
    }
    ctx = build_mock_context(config, writer)
    inputs = _PlanInputs(
        [
            SourceItem(
                source_id=f"src{i}",
                path=f"/not/read/{i}.jpg",
                corpus="quandian",
                width=64,
                height=64,
            )
            for i in range(8)
        ]
    )

    result = run_mod.run_stream(
        StreamId.S1_DEGRADE_LOCAL,
        budget=8,
        config=config,
        ctx=ctx,
        inputs=inputs,
        writer=writer,
        gate=QAGate(config),
    )

    assert result["accepted"] == 8
    assert TrackingExecutor.last is not None
    assert TrackingExecutor.last.max_active <= 2


def test_run_stream_degrade_verify_flag_renders_preview_without_vlm(tmp_path):
    writer = _NullWriter()
    config = {
        "seed": 1234,
        "build_version": "test",
        "scratch_dir": str(tmp_path),
        "models": {"veraretouch": {"batch_size": 4}},
        "vllm": {"concurrency": 1, "max_outstanding_samples": 1},
        "qa": {
            "annotation_mode": "template",
            "enable_verify": True,
            "verify_degrade_streams": True,
            "verify_sample_rate": 1.0,
        },
        "cgt": {"min_coverage": 0.01, "max_coverage": 0.95, "patch_grid": 16},
        "degrade": {"region_shapes": ["random_blob"]},
    }
    ctx = build_mock_context(config, writer)
    ctx.renderer = _FakeRenderer(value=64)
    ctx.cleaner = None
    inputs = _PlanInputs(
        [
            SourceItem(
                source_id="src",
                path="/not/read/src.jpg",
                corpus="quandian",
                width=64,
                height=64,
            )
        ]
    )

    result = run_mod.run_stream(
        StreamId.S1_DEGRADE_LOCAL,
        budget=1,
        config=config,
        ctx=ctx,
        inputs=inputs,
        writer=writer,
        gate=QAGate(config),
    )

    assert result["accepted"] == 1
    assert writer.samples[0].meta["teacher_degraded_preview"] is True
    assert len(ctx.renderer.calls) == 1
    assert ctx.renderer.calls[0][0] == ["/not/read/src.jpg"]
    assert ctx.renderer.calls[0][1] == [writer.samples[0].recipe.params]


def test_patch_grid_downsample_matches_area_average_for_even_cells():
    import numpy as np

    a = np.arange(16, dtype="float32").reshape(4, 4)
    grid = ShardWriter._downsample_grid(a, 2)

    assert np.allclose(grid, [[2.5, 4.5], [10.5, 12.5]])


def test_cgt_coverage_counts_mask_pixels_above_half(tmp_path):
    import numpy as np

    ctx = _ctx(tmp_path)
    stream = RecipeXSourceStream(
        ctx,
        StreamId.S2_RECIPE_LOCAL,
        inputs=_PlanInputs([]),
        region_local=True,
        gate=QAGate(ctx.config),
    )
    mask = np.array([[0.0, 0.51], [0.5, 1.0]], dtype="float32")

    assert stream._coverage(mask) == pytest.approx(0.5)
    assert stream._coverage(np.zeros((0, 0), dtype="float32")) == 0.0


def test_run_stream_rejects_oversized_sources_before_worker_decode(tmp_path):
    writer = _NullWriter()
    config = {
        "seed": 1234,
        "build_version": "test",
        "scratch_dir": str(tmp_path),
        "models": {"veraretouch": {"batch_size": 4}},
        "vllm": {"concurrency": 4, "max_outstanding_samples": 4, "max_image_pixels": 100},
        "sources": {"max_source_pixels": 100},
        "qa": {"enable_verify": False},
        "cgt": {"min_coverage": 0.01, "max_coverage": 0.95, "patch_grid": 16},
        "degrade": {"region_shapes": ["random_blob"]},
    }
    ctx = build_mock_context(config, writer)
    inputs = _PlanInputs(
        [
            SourceItem(
                source_id="huge-frame",
                path="/not/read/huge.jpg",
                corpus="quandian",
                width=20,
                height=20,
            )
        ]
    )

    result = run_mod.run_stream(
        StreamId.S1_DEGRADE_LOCAL,
        budget=1,
        config=config,
        ctx=ctx,
        inputs=inputs,
        writer=writer,
        gate=QAGate(config),
    )

    assert result["planned"] == 1
    assert result["accepted"] == 0
    assert result["rejected"] == 1
    assert writer.samples == []
    assert writer.rejects[0].quality.rejected_reason == "source_pixels>100(400)"


def test_run_stream_probes_image_header_when_dimensions_missing(tmp_path):
    from PIL import Image

    p = tmp_path / "oversized_header.jpg"
    Image.new("RGB", (20, 20), color=(128, 128, 128)).save(p)
    writer = _NullWriter()
    config = {
        "seed": 1234,
        "build_version": "test",
        "scratch_dir": str(tmp_path),
        "models": {"veraretouch": {"batch_size": 1}},
        "vllm": {"concurrency": 1, "max_outstanding_samples": 1, "max_image_pixels": 100},
        "sources": {"max_source_pixels": 100},
        "qa": {"enable_verify": False},
        "cgt": {"min_coverage": 0.01, "max_coverage": 0.95, "patch_grid": 16},
        "degrade": {"region_shapes": ["random_blob"]},
    }
    ctx = build_mock_context(config, writer)
    inputs = _PlanInputs(
        [
            SourceItem(
                source_id="missing-dims",
                path=str(p),
                corpus="quandian",
                width=None,
                height=None,
            )
        ]
    )

    result = run_mod.run_stream(
        StreamId.S1_DEGRADE_LOCAL,
        budget=1,
        config=config,
        ctx=ctx,
        inputs=inputs,
        writer=writer,
        gate=QAGate(config),
    )

    assert result["accepted"] == 0
    assert result["rejected"] == 1
    assert writer.rejects[0].quality.rejected_reason == "source_pixels>100(400)"


def test_template_annotation_skips_vlm_instruction_and_reasoning(tmp_path):
    writer = _NullWriter()
    config = {
        "seed": 1234,
        "build_version": "test",
        "scratch_dir": str(tmp_path),
        "qa": {"annotation_mode": "template", "enable_verify": False},
        "cgt": {"min_coverage": 0.01, "max_coverage": 0.95, "patch_grid": 16},
        "degrade": {"region_shapes": ["random_blob"]},
    }
    ctx = build_mock_context(config, writer)
    ctx.cleaner = _SpyCleaner()
    stream = RecipeXSourceStream(
        ctx,
        stream_id=StreamId.S6_RECIPE_GLOBAL,
        inputs=_PlanInputs([]),
        region_local=False,
        gate=QAGate(config),
    )
    source = SourceItem(
        source_id="src",
        path="/not/read/template.jpg",
        corpus="quandian",
        width=64,
        height=64,
        scene="portrait",
    )
    recipe = Recipe(
        kind=RecipeKind.PARAM,
        params={"Exposure2012": {"value": 10.0}},
        meta={"style": "warm film"},
    )

    sample = stream.build_one(source, recipe, False, "sample_id", "pending", precomputed_after=None)

    assert sample is not None
    assert sample.instruction == "Apply the stored warm film retouching parameters to this portrait photo."
    assert sample.answer == {"Exposure2012": {"value": 10.0}}
    assert ctx.cleaner.calls == []


def test_vlm_region_tags_drive_s2_mask_concepts_without_overwriting_answer(tmp_path):
    writer = _NullWriter()
    config = {
        "seed": 1234,
        "build_version": "test",
        "scratch_dir": str(tmp_path),
        "qa": {
            "annotation_mode": "vlm",
            "enable_verify": False,
            "vlm_override_answer": False,
        },
        "cgt": {
            "min_coverage": 0.01,
            "max_coverage": 0.95,
            "patch_grid": 16,
            "soft_blur_sigma_px": 0,
        },
        "degrade": {"region_shapes": ["random_blob"]},
    }
    ctx = build_mock_context(config, writer)
    ctx.cleaner = _SpyCleaner()
    ctx.masker = _SpyMasker()
    stream = RecipeXSourceStream(
        ctx,
        stream_id=StreamId.S2_RECIPE_LOCAL,
        inputs=_PlanInputs([]),
        region_local=True,
        gate=QAGate(config),
    )
    source = SourceItem(
        source_id="src",
        path="/not/read/region.jpg",
        corpus="quandian",
        width=8,
        height=8,
        scene="landscape",
    )
    recipe = Recipe(
        kind=RecipeKind.PARAM,
        params={"Exposure2012": {"value": 10.0}},
        meta={"style": "cinematic"},
    )

    sample = stream.build_one(source, recipe, True, "sample_id", "pending", precomputed_after=None)

    assert sample is not None
    assert ctx.masker.calls
    assert ctx.masker.calls[0][1] == ["sky", "face"]
    call_names = [c[0] for c in ctx.cleaner.calls]
    assert "gen_instruction" in call_names
    assert "tag_scene_region" in call_names
    assert "reason_params" in call_names
    assert sample.answer == {"Exposure2012": {"value": 10.0}}
    assert sample.meta["vlm_region_tag"]["sam3_concepts"] == ["sky", "face"]


def test_lut_recipe_validates_and_records_domain_sha(tmp_path):
    lut_path = tmp_path / "identity.cube"
    lut_path.write_text(
        "\n".join(
            [
                "LUT_3D_SIZE 2",
                "DOMAIN_MIN 0 0 0",
                "DOMAIN_MAX 1 1 1",
                "0 0 0",
                "1 0 0",
                "0 1 0",
                "1 1 0",
                "0 0 1",
                "1 0 1",
                "0 1 1",
                "1 1 1",
            ]
        )
        + "\n"
    )
    writer = _NullWriter()
    config = {
        "seed": 1234,
        "build_version": "test",
        "scratch_dir": str(tmp_path),
        "qa": {"annotation_mode": "template", "enable_verify": False},
        "cgt": {"min_coverage": 0.01, "max_coverage": 0.95, "patch_grid": 16},
        "degrade": {"region_shapes": ["random_blob"]},
    }
    ctx = build_mock_context(config, writer)
    stream = RecipeXSourceStream(
        ctx,
        stream_id=StreamId.S6_RECIPE_GLOBAL,
        inputs=_PlanInputs([]),
        region_local=False,
        gate=QAGate(config),
    )
    asset = RecipeAsset(
        recipe_id="lut1",
        path=str(lut_path),
        kind=RecipeKind.LUT,
        fmt="cube",
    )

    recipe = stream._recipe_to_obj(asset)

    assert recipe.kind == RecipeKind.LUT
    assert recipe.provenance == Provenance.LUT
    assert recipe.meta["lut_size"] == 2
    assert recipe.meta["domain_min"] == [0.0, 0.0, 0.0]
    assert recipe.meta["domain_max"] == [1.0, 1.0, 1.0]
    assert recipe.meta["lut_sha256"]
    assert "lut1" in ctx.lut_cache


def test_s5_vlm_annotation_uses_expert_jpg_for_raw_source(tmp_path):
    writer = _NullWriter()
    config = {
        "seed": 1234,
        "build_version": "test",
        "scratch_dir": str(tmp_path),
        "qa": {
            "annotation_mode": "vlm",
            "enable_verify": False,
            "tag_global": True,
            "vlm_override_answer": False,
        },
        "cgt": {"min_coverage": 0.01, "max_coverage": 0.95, "patch_grid": 16},
        "degrade": {"region_shapes": ["random_blob"]},
    }
    ctx = build_mock_context(config, writer)
    ctx.cleaner = _SpyCleaner()
    stream = Tier1ExpertStream(ctx, inputs=_PlanInputs([]), gate=QAGate(config))
    source = SourceItem(
        source_id="greysky",
        path="/not/read/source.dng",
        corpus="greysky",
        width=8,
        height=8,
        meta={
            "expert_after_jpg": "/read/this/expert.jpg",
            "expert_xmp": "/not/read/expert.xmp",
        },
    )
    recipe = Recipe(
        kind=RecipeKind.PARAM,
        params={"Exposure2012": {"value": 10.0}},
        meta={"expert_after_jpg": "/read/this/expert.jpg", "use_real_jpg": True},
    )

    sample = stream.build_one(source, recipe, False, "sample_id", "pending", precomputed_after=None)

    assert sample is not None
    image_args = [c[1][0] for c in ctx.cleaner.calls if c[0] in {"gen_instruction", "reason_params", "tag_scene_region"}]
    assert image_args
    assert set(image_args) == {"/read/this/expert.jpg"}


def test_global_source_tags_are_cached_per_source_image(tmp_path):
    writer = _NullWriter()
    config = {
        "seed": 1234,
        "build_version": "test",
        "scratch_dir": str(tmp_path),
        "qa": {
            "annotation_mode": "vlm",
            "enable_verify": False,
            "tag_global": True,
            "vlm_override_answer": False,
        },
        "cgt": {"min_coverage": 0.01, "max_coverage": 0.95, "patch_grid": 16},
        "degrade": {"region_shapes": ["random_blob"]},
    }
    ctx = build_mock_context(config, writer)
    ctx.cleaner = _SpyCleaner()
    stream = RecipeXSourceStream(
        ctx,
        stream_id=StreamId.S6_RECIPE_GLOBAL,
        inputs=_PlanInputs([]),
        region_local=False,
        gate=QAGate(config),
    )
    source = SourceItem(
        source_id="shared-src",
        path="/not/read/shared.jpg",
        corpus="quandian",
        width=8,
        height=8,
        scene="landscape",
    )
    recipe_a = Recipe(
        kind=RecipeKind.PARAM,
        params={"Exposure2012": {"value": 10.0}},
        meta={"style": "cinematic"},
    )
    recipe_b = Recipe(
        kind=RecipeKind.PARAM,
        params={"Exposure2012": {"value": -5.0}},
        meta={"style": "clean"},
    )

    sample_a = stream.build_one(source, recipe_a, False, "sample_a", "pending", precomputed_after=None)
    sample_b = stream.build_one(source, recipe_b, False, "sample_b", "pending", precomputed_after=None)

    tag_calls = [c for c in ctx.cleaner.calls if c[0] == "tag_scene_region"]
    assert sample_a is not None
    assert sample_b is not None
    assert len(tag_calls) == 1
    assert tag_calls[0][1] == ("/not/read/shared.jpg", "")
    assert sample_a.meta["vlm_source_tag"]["sam3_concepts"] == ["sky", "face"]
    assert sample_b.meta["vlm_source_tag"]["sam3_concepts"] == ["sky", "face"]


def test_region_tags_are_cached_by_source_and_instruction(tmp_path):
    writer = _NullWriter()
    config = {
        "seed": 1234,
        "build_version": "test",
        "scratch_dir": str(tmp_path),
        "qa": {
            "annotation_mode": "vlm",
            "enable_verify": False,
            "vlm_override_answer": False,
        },
        "cgt": {
            "min_coverage": 0.01,
            "max_coverage": 0.95,
            "patch_grid": 16,
            "soft_blur_sigma_px": 0,
        },
        "degrade": {"region_shapes": ["random_blob"]},
    }
    ctx = build_mock_context(config, writer)
    ctx.cleaner = _SpyCleaner()
    ctx.masker = _SpyMasker()
    stream = MMArtTextStream(ctx, inputs=_PlanInputs([]), gate=QAGate(config))
    source = SourceItem(
        source_id="shared-src",
        path="/not/read/mmart.jpg",
        corpus="mmart",
        width=8,
        height=8,
        scene="portrait",
    )

    def recipe(instruction):
        return Recipe(
            kind=RecipeKind.PARAM,
            params={"Exposure2012": {"value": 10.0}},
            meta={"instruction": instruction, "think": "given", "masksubtype_hint": 0},
        )

    sample_a = stream.build_one(
        source, recipe("brighten the sky"), True, "sample_a", "pending", precomputed_after=None
    )
    sample_b = stream.build_one(
        source, recipe("brighten the sky"), True, "sample_b", "pending", precomputed_after=None
    )
    sample_c = stream.build_one(
        source, recipe("brighten the face"), True, "sample_c", "pending", precomputed_after=None
    )

    tag_calls = [c for c in ctx.cleaner.calls if c[0] == "tag_scene_region"]
    assert sample_a is not None
    assert sample_b is not None
    assert sample_c is not None
    assert len(tag_calls) == 2
    assert [c[1] for c in tag_calls] == [
        ("/not/read/mmart.jpg", "brighten the sky"),
        ("/not/read/mmart.jpg", "brighten the face"),
    ]
    assert sample_a.meta["vlm_region_tag"]["sam3_concepts"] == ["sky", "face"]
    assert sample_b.meta["vlm_region_tag"]["sam3_concepts"] == ["sky", "face"]
    assert sample_c.meta["vlm_region_tag"]["sam3_concepts"] == ["sky", "face"]


def test_real_model_loader_skips_renderer_for_degrade_only_verify_off(monkeypatch):
    called = {"renderer": False, "masker": False, "cleaner": False, "parser": False}

    def import_guard(name, *args, **kwargs):
        if name in {
            "dataset_build.render",
            "dataset_build.masking",
            "dataset_build.vlm_clean",
        }:
            key = name.rsplit(".", 1)[-1]
            if key == "render":
                called["renderer"] = True
            elif key == "masking":
                called["masker"] = True
            elif key == "vlm_clean":
                called["cleaner"] = True
        return real_import(name, *args, **kwargs)

    real_import = __import__
    monkeypatch.setattr("builtins.__import__", import_guard)

    models = run_mod._build_real_models(
        {"qa": {"annotation_mode": "vlm", "enable_verify": False}},
        [StreamId.S1_DEGRADE_LOCAL, StreamId.S7_DEGRADE_GLOBAL],
    )

    # DATAGEN v2 S1/S7 are recipe-only at build time by default. Stage-0 owns the
    # z* PSNR gate, so fast-degrade mode must not pull in renderer or VLM.
    assert models["renderer"] is None
    assert models["masker"] is None
    assert models["cleaner"] is None
    assert called["renderer"] is False
    assert called["masker"] is False
    assert called["cleaner"] is False


def test_real_model_loader_skips_cleaner_in_template_degrade_mode(monkeypatch):
    called = {"renderer": False, "masker": False, "cleaner": False}

    def import_guard(name, *args, **kwargs):
        if name in {"dataset_build.render", "dataset_build.masking", "dataset_build.vlm_clean"}:
            key = name.rsplit(".", 1)[-1]
            if key == "render":
                called["renderer"] = True
            elif key == "masking":
                called["masker"] = True
            elif key == "vlm_clean":
                called["cleaner"] = True
        return real_import(name, *args, **kwargs)

    real_import = __import__
    monkeypatch.setattr("builtins.__import__", import_guard)

    models = run_mod._build_real_models(
        {"qa": {"annotation_mode": "template", "enable_verify": False}},
        [StreamId.S1_DEGRADE_LOCAL, StreamId.S7_DEGRADE_GLOBAL],
    )

    assert models["renderer"] is None
    assert models["masker"] is None
    assert models["cleaner"] is None
    assert called == {"renderer": False, "masker": False, "cleaner": False}


def test_real_model_loader_loads_renderer_for_degrade_verify_enabled(monkeypatch):
    called = {"renderer": False, "masker": False, "cleaner": False}

    def import_guard(name, *args, **kwargs):
        if name in {"dataset_build.render", "dataset_build.masking", "dataset_build.vlm_clean"}:
            key = name.rsplit(".", 1)[-1]
            if key == "render":
                called["renderer"] = True
            elif key == "masking":
                called["masker"] = True
            elif key == "vlm_clean":
                called["cleaner"] = True
        return real_import(name, *args, **kwargs)

    real_import = __import__
    monkeypatch.setattr("builtins.__import__", import_guard)

    models = run_mod._build_real_models(
        {
            "qa": {
                "annotation_mode": "template",
                "enable_verify": True,
                "verify_degrade_streams": True,
                "verify_sample_rate": 1.0,
            }
        },
        [StreamId.S1_DEGRADE_LOCAL, StreamId.S7_DEGRADE_GLOBAL],
    )

    assert models["renderer"] is not None
    assert models["masker"] is None
    assert models["cleaner"] is None
    assert called == {"renderer": True, "masker": False, "cleaner": False}


def test_real_model_loader_uses_cached_masker_without_importing_sam3(monkeypatch, tmp_path):
    called = {"masking": False}

    def import_guard(name, *args, **kwargs):
        if name == "dataset_build.masking":
            called["masking"] = True
        return real_import(name, *args, **kwargs)

    real_import = __import__
    monkeypatch.setattr("builtins.__import__", import_guard)

    models = run_mod._build_real_models(
        {
            "models": {
                "sam3": {"use_cache": True, "cache_dir": str(tmp_path)},
                "veraretouch": {},
            },
            "qa": {"annotation_mode": "template", "enable_verify": False},
        },
        [StreamId.S2_RECIPE_LOCAL],
    )

    assert models["masker"].__class__.__name__ == "CachedMasker"
    assert called["masking"] is False


def test_sam3_precompute_defaults_to_wave2_source_pools_and_dedups(tmp_path):
    index = tmp_path / "source_index.jsonl"
    rows = [
        {"source_id": "s1", "path": "/data/a.jpg", "corpus": "tad66k"},
        {"source_id": "s1_dup", "path": "/data/a.jpg", "corpus": "tad66k"},
        {"source_id": "s2", "path": "/data/b.jpg", "corpus": "mmart"},
        {"source_id": "s3", "path": "/data/c.jpg", "corpus": "fivek"},
        {"source_id": "s4", "path": "/data/d.jpg", "corpus": "greysky"},
    ]
    index.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    default_corpora = _parse_corpora_arg("")
    kept = list(_iter_sources(str(index), default_corpora, shard_i=0, shard_n=1, limit=0))
    all_kept = list(_iter_sources(str(index), _parse_corpora_arg("ALL"), shard_i=0, shard_n=1, limit=0))

    assert default_corpora == DEFAULT_CORPORA
    assert [r["source_id"] for r in kept] == ["s1", "s2"]
    assert [r["source_id"] for r in all_kept] == ["s1", "s2", "s3", "s4"]

    sharded = []
    for i in range(2):
        sharded.extend(_iter_sources(str(index), _parse_corpora_arg("ALL"), shard_i=i, shard_n=2, limit=0))
    assert sorted(r["path"] for r in sharded) == ["/data/a.jpg", "/data/b.jpg", "/data/c.jpg", "/data/d.jpg"]


def _write_rgb(path: Path, value: int):
    import numpy as np
    from PIL import Image

    Image.fromarray(np.full((4, 4, 3), value, dtype="uint8"), mode="RGB").save(path)


def _sample(source_path: str, recipe: Recipe, **kwargs) -> Sample:
    return Sample(
        sample_id=kwargs.get("sample_id", "sample"),
        stream=kwargs.get("stream", StreamId.S6_RECIPE_GLOBAL),
        shard="pending",
        source_path=source_path,
        raw_decode=RawDecode.NONE,
        recipe=recipe,
        region_local=kwargs.get("region_local", False),
        c_gt=kwargs.get("c_gt", CgtRef(mask_source=MaskSource.GLOBAL)),
        after_source=kwargs.get("after_source", AfterSource.TEACHER),
        meta=kwargs.get("meta", {}),
    )


def test_reproduce_pair_param_uses_teacher_renderer(tmp_path):
    src = tmp_path / "src.png"
    _write_rgb(src, 20)
    renderer = _FakeRenderer(value=190)
    sample = _sample(
        str(src),
        Recipe(
            kind=RecipeKind.PARAM,
            params={"Exposure2012": {"value": 10.0}},
            provenance=Provenance.PARAM,
        ),
    )

    pair = reproduce_pair(sample, renderer=renderer)

    assert pair.needs_z_search is False
    assert pair.input_rgb.mean() == 20
    assert pair.target_rgb.mean() == 190
    assert renderer.calls[0][1] == [{"Exposure2012": {"value": 10.0}}]


def test_reproduce_pair_degrade_materializes_teacher_input_and_warm_start(tmp_path):
    src = tmp_path / "src.png"
    _write_rgb(src, 40)
    renderer = _FakeRenderer(value=80)
    recipe = Recipe(
        kind=RecipeKind.PARAM,
        params={"Exposure2012": {"value": -25.0}},
        provenance=Provenance.DEGRADE,
        degrade=DegradeSpec(
            mode="gaussian_op",
            op_params={"Exposure2012": 25.0},
            aspects=["L"],
            forward=False,
        ),
    )
    sample = _sample(
        str(src),
        recipe,
        stream=StreamId.S1_DEGRADE_LOCAL,
        region_local=True,
        c_gt=CgtRef(mask_source=MaskSource.DEGRADE),
    )

    pair = reproduce_pair(sample, renderer=renderer)

    assert pair.needs_z_search is True
    assert pair.input_rgb.mean() == 80
    assert pair.target_rgb.mean() == 40
    assert pair.meta["warm_start"] == {"Exposure2012": {"value": 25.0}}
    with pytest.raises(ValueError, match="z_search_render_fn"):
        reproduce_pair(sample, renderer=renderer, run_z_search=True)


def test_reproduce_pair_degrade_runs_reference_z_search(tmp_path):
    src = tmp_path / "src.png"
    _write_rgb(src, 40)
    renderer = _FakeRenderer(value=80)
    recipe = Recipe(
        kind=RecipeKind.PARAM,
        params={"Exposure2012": {"value": -25.0}},
        provenance=Provenance.DEGRADE,
        degrade=DegradeSpec(
            mode="gaussian_op",
            op_params={"Exposure2012": 25.0},
            aspects=["L"],
            forward=False,
        ),
    )
    sample = _sample(str(src), recipe, stream=StreamId.S7_DEGRADE_GLOBAL)

    def render_from_array(input_rgb, params):
        import numpy as np

        v = float(params["Exposure2012"]["value"])
        # Optimum for target value 40 when input value is 80 is v=-40.
        return np.clip(input_rgb.astype("float32") + v, 0, 255).astype("uint8")

    pair = reproduce_pair(
        sample,
        renderer=renderer,
        run_z_search=True,
        z_search_render_fn=render_from_array,
        z_search_keys=["Exposure2012"],
        z_search_max_iters=4,
        er_recon_psnr_min=30.0,
    )

    assert pair.z_star == {"Exposure2012": {"value": -40.0}}
    assert pair.er_recon_psnr == float("inf")
    assert pair.needs_z_search is False


def test_inverse_search_z_star_improves_psnr():
    import numpy as np

    input_rgb = np.full((2, 2, 3), 100, dtype="uint8")
    target_rgb = np.full((2, 2, 3), 70, dtype="uint8")

    def render_fn(inp, params):
        v = float(params["Exposure2012"]["value"])
        return np.clip(inp.astype("float32") + v, 0, 255).astype("uint8")

    z_star, psnr = inverse_search_z_star(
        input_rgb,
        target_rgb,
        {"Exposure2012": {"value": 0.0}},
        render_fn,
        keys=["Exposure2012"],
        max_iters=2,
        step_schedule=(20.0, 10.0, 5.0),
    )

    assert z_star == {"Exposure2012": {"value": -30.0}}
    assert psnr == float("inf")


def test_array_render_adapter_bridges_array_to_path_renderer(tmp_path):
    import numpy as np

    renderer = _PathEchoRenderer()
    adapter = ArrayRenderAdapter(renderer, scratch_dir=str(tmp_path), prefix="unit")
    arr = np.full((4, 4, 3), 77, dtype="uint8")
    params = {"Exposure2012": {"value": 3.0}}

    out = adapter(arr, params)

    assert out.mean() == 77
    assert renderer.calls
    temp_path = Path(renderer.calls[0][0][0])
    assert temp_path.name.startswith("unit_")
    assert not temp_path.exists()


def test_reproduce_pair_degrade_z_search_with_array_adapter(tmp_path):
    import numpy as np

    src = tmp_path / "src.png"
    _write_rgb(src, 50)

    class ShiftPathRenderer:
        def __init__(self):
            self.calls = []

        def render(self, image_paths, param_dicts, **kwargs):
            import cv2

            outs = []
            for path, params in zip(image_paths, param_dicts):
                self.calls.append((path, params))
                bgr = cv2.imread(path, cv2.IMREAD_COLOR)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype("float32")
                v = float((params.get("Exposure2012") or {}).get("value", 0.0))
                outs.append(np.clip(rgb + v, 0, 255).astype("uint8"))
            return outs

    renderer = ShiftPathRenderer()
    recipe = Recipe(
        kind=RecipeKind.PARAM,
        params={"Exposure2012": {"value": 30.0}},
        provenance=Provenance.DEGRADE,
        degrade=DegradeSpec(
            mode="gaussian_op",
            op_params={"Exposure2012": -30.0},
            aspects=["L"],
            forward=False,
        ),
    )
    sample = _sample(str(src), recipe, stream=StreamId.S7_DEGRADE_GLOBAL)
    adapter = ArrayRenderAdapter(renderer, scratch_dir=str(tmp_path), prefix="stage0")

    pair = reproduce_pair(
        sample,
        renderer=renderer,
        run_z_search=True,
        z_search_render_fn=adapter,
        z_search_keys=["Exposure2012"],
        z_search_max_iters=4,
    )

    assert pair.input_rgb.mean() == 80
    assert pair.target_rgb.mean() == 50
    assert pair.z_star == {"Exposure2012": {"value": -30.0}}
    assert pair.er_recon_psnr == float("inf")
    assert not list((tmp_path / "datagen_zsearch").glob("stage0_*.png"))


def test_reproduce_pair_degrade_z_search_builds_adapter_from_scratch_dir(tmp_path):
    import numpy as np

    src = tmp_path / "src.png"
    _write_rgb(src, 50)

    class ShiftPathRenderer:
        def render(self, image_paths, param_dicts, **kwargs):
            import cv2

            outs = []
            for path, params in zip(image_paths, param_dicts):
                bgr = cv2.imread(path, cv2.IMREAD_COLOR)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype("float32")
                v = float((params.get("Exposure2012") or {}).get("value", 0.0))
                outs.append(np.clip(rgb + v, 0, 255).astype("uint8"))
            return outs

    renderer = ShiftPathRenderer()
    recipe = Recipe(
        kind=RecipeKind.PARAM,
        params={"Exposure2012": {"value": 30.0}},
        provenance=Provenance.DEGRADE,
        degrade=DegradeSpec(
            mode="gaussian_op",
            op_params={"Exposure2012": -30.0},
            aspects=["L"],
            forward=False,
        ),
    )

    pair = reproduce_pair(
        _sample(str(src), recipe, stream=StreamId.S7_DEGRADE_GLOBAL, sample_id="sid"),
        renderer=renderer,
        run_z_search=True,
        z_search_scratch_dir=str(tmp_path),
        z_search_keys=["Exposure2012"],
        z_search_max_iters=1,
    )

    assert pair.z_star == {"Exposure2012": {"value": -30.0}}
    assert pair.er_recon_psnr == float("inf")
    assert not list((tmp_path / "datagen_zsearch").glob("sid_*.png"))


def test_reproduce_pair_real_jpg_uses_enforced_after_source(tmp_path):
    src = tmp_path / "src.png"
    after = tmp_path / "after.png"
    _write_rgb(src, 10)
    _write_rgb(after, 210)
    sample = _sample(
        str(src),
        Recipe(kind=RecipeKind.PARAM, params={}, provenance=Provenance.PARAM),
        stream=StreamId.S5_GREYSKY_GLOBAL,
        after_source=AfterSource.REAL_JPG,
        meta={"expert_after_path": str(after)},
    )

    pair = reproduce_pair(sample)

    assert pair.input_rgb.mean() == 10
    assert pair.target_rgb.mean() == 210
    assert pair.meta["contract"] == "real_jpg"


def test_reproduce_pair_lut_applies_lut_and_keeps_metadata(tmp_path):
    src = tmp_path / "src.png"
    lut_path = tmp_path / "identity.cube"
    _write_rgb(src, 64)
    lut_path.write_text(
        "\n".join(
            [
                "LUT_3D_SIZE 2",
                "0 0 0",
                "1 0 0",
                "0 1 0",
                "1 1 0",
                "0 0 1",
                "1 0 1",
                "0 1 1",
                "1 1 1",
            ]
        )
        + "\n"
    )
    parser = MockParser()
    from dataset_build.run import MockLutApplier

    sample = _sample(
        str(src),
        Recipe(
            kind=RecipeKind.LUT,
            lut_recipe_id="lut1",
            provenance=Provenance.LUT,
            meta={"path": str(lut_path), "lut_sha256": "sha"},
        ),
    )

    pair = reproduce_pair(sample, parser=parser, lut_applier=MockLutApplier())

    assert pair.input_rgb.mean() == 64
    assert pair.target_rgb.mean() == 64
    assert pair.meta["contract"] == "lut"
    assert pair.meta["lut_sha256"] == "sha"


def test_region_composite_from_pair_uses_raw_mask(tmp_path):
    import numpy as np

    pair = type(
        "Pair",
        (),
        {
            "input_rgb": np.zeros((2, 2, 3), dtype="uint8"),
            "target_rgb": np.full((2, 2, 3), 255, dtype="uint8"),
            "raw_mask01": np.array([[1, 0], [0, 1]], dtype="float32"),
        },
    )()

    comp = region_composite_from_pair(pair)

    assert comp[0, 0, 0] == 255
    assert comp[0, 1, 0] == 0


def test_stage0_probe_reports_accept_drop_and_skips_non_degrade(tmp_path):
    import numpy as np
    from PIL import Image

    src = tmp_path / "src.png"
    _write_rgb(src, 50)
    raw_mask = tmp_path / "raw_mask.png"
    Image.fromarray(np.full((4, 4), 255, dtype="uint8"), mode="L").save(raw_mask)

    ok = _sample(
        str(src),
        Recipe(
            kind=RecipeKind.PARAM,
            params={"Exposure2012": {"value": 30.0}},
            provenance=Provenance.DEGRADE,
            degrade=DegradeSpec(
                mode="gaussian_op",
                op_params={"Exposure2012": -30.0},
                aspects=["L"],
                forward=False,
            ),
        ),
        stream=StreamId.S7_DEGRADE_GLOBAL,
        sample_id="ok",
    )
    drop = _sample(
        str(src),
        Recipe(
            kind=RecipeKind.PARAM,
            params={"Exposure2012": {"value": 30.0}},
            provenance=Provenance.DEGRADE,
            degrade=DegradeSpec(
                mode="gaussian_op",
                op_params={"Exposure2012": 0.0},
                aspects=["L"],
                forward=False,
            ),
        ),
        stream=StreamId.S1_DEGRADE_LOCAL,
        sample_id="drop",
        region_local=True,
        c_gt=CgtRef(mask_source=MaskSource.DEGRADE, raw_mask_path=str(raw_mask)),
    )
    skip = _sample(
        str(src),
        Recipe(
            kind=RecipeKind.PARAM,
            params={"Exposure2012": {"value": 1.0}},
            provenance=Provenance.PARAM,
        ),
        sample_id="skip",
    )
    shard = tmp_path / "shard.jsonl"
    shard.write_text(
        "\n".join([sample_to_jsonl(ok), sample_to_jsonl(skip), sample_to_jsonl(drop)]) + "\n",
        encoding="utf-8",
    )
    report = tmp_path / "stage0_report.jsonl"

    summary = run_stage0_probe(
        shard,
        renderer=DryRunParamRenderer(),
        report_path=report,
        scratch_dir=str(tmp_path),
        er_recon_psnr_min=35.0,
        z_search_max_iters=0,
        collect=True,
    )

    assert summary["lines"] == 3
    assert summary["skipped_non_degrade"] == 1
    assert summary["degrade"] == 2
    assert summary["accepted"] == 1
    assert summary["dropped"] == 1
    assert summary["errors"] == 0

    records = [json.loads(line) for line in report.read_text(encoding="utf-8").splitlines()]
    assert [r["sample_id"] for r in records] == ["ok", "drop"]
    assert records[0]["accepted"] is True
    assert records[0]["er_recon_psnr"] == "inf"
    assert records[0]["z_star"] == {"Exposure2012": {"value": -30.0}}
    assert records[1]["accepted"] is False
    assert records[1]["drop_reason"] == "er_recon_psnr<35"
    assert records[1]["region_composite_ready"] is True
    assert records[1]["raw_mask_path"] == str(raw_mask)
    assert [r["sample_id"] for r in summary["records"]] == ["ok", "drop"]


def test_stage0_probe_rejects_region_degrade_without_raw_mask(tmp_path):
    src = tmp_path / "src.png"
    _write_rgb(src, 50)
    sample = _sample(
        str(src),
        Recipe(
            kind=RecipeKind.PARAM,
            params={"Exposure2012": {"value": 30.0}},
            provenance=Provenance.DEGRADE,
            degrade=DegradeSpec(
                mode="gaussian_op",
                op_params={"Exposure2012": -30.0},
                aspects=["L"],
                forward=False,
            ),
        ),
        stream=StreamId.S1_DEGRADE_LOCAL,
        sample_id="missing-mask",
        region_local=True,
        c_gt=CgtRef(mask_source=MaskSource.DEGRADE),
    )
    shard = tmp_path / "shard.jsonl"
    shard.write_text(sample_to_jsonl(sample) + "\n", encoding="utf-8")

    summary = run_stage0_probe(
        shard,
        renderer=DryRunParamRenderer(),
        scratch_dir=str(tmp_path),
        er_recon_psnr_min=35.0,
        collect=True,
    )

    record = summary["records"][0]
    assert summary["errors"] == 1
    assert record["decision"] == "error"
    assert record["drop_reason"] == "missing_raw_mask"
    assert record["region_composite_ready"] is False


def test_stage0_probe_drops_over_pixel_budget_before_render(tmp_path):
    src = tmp_path / "src.png"
    _write_rgb(src, 50)
    sample = _sample(
        str(src),
        Recipe(
            kind=RecipeKind.PARAM,
            params={"Exposure2012": {"value": 30.0}},
            provenance=Provenance.DEGRADE,
            degrade=DegradeSpec(
                mode="gaussian_op",
                op_params={"Exposure2012": -30.0},
                aspects=["L"],
                forward=False,
            ),
        ),
        stream=StreamId.S7_DEGRADE_GLOBAL,
        sample_id="too-large",
    )
    sample.native_size = (20, 20)
    shard = tmp_path / "shard.jsonl"
    shard.write_text(sample_to_jsonl(sample) + "\n", encoding="utf-8")
    renderer = DryRunParamRenderer()

    summary = run_stage0_probe(
        shard,
        renderer=renderer,
        scratch_dir=str(tmp_path),
        er_recon_psnr_min=35.0,
        max_stage0_pixels=100,
        collect=True,
    )

    record = summary["records"][0]
    assert summary["dropped"] == 1
    assert summary["errors"] == 0
    assert record["drop_reason"] == "stage0_pixels>100(400)"
    assert record["native_pixels"] == 400


def test_stage0_probe_scratch_dir_falls_back_for_config_default(tmp_path):
    bad = tmp_path / "not_a_dir"
    bad.write_text("file blocks mkdir", encoding="utf-8")

    chosen = _choose_scratch_dir(str(bad), explicit=False)

    assert chosen == "/tmp/datagen_stage0_probe"


def test_stage0_probe_cli_dry_run_writes_report(tmp_path, capsys):
    src = tmp_path / "src.png"
    _write_rgb(src, 50)
    sample = _sample(
        str(src),
        Recipe(
            kind=RecipeKind.PARAM,
            params={"Exposure2012": {"value": 30.0}},
            provenance=Provenance.DEGRADE,
            degrade=DegradeSpec(
                mode="gaussian_op",
                op_params={"Exposure2012": -30.0},
                aspects=["L"],
                forward=False,
            ),
        ),
        stream=StreamId.S7_DEGRADE_GLOBAL,
        sample_id="cli-ok",
    )
    shard = tmp_path / "shard.jsonl"
    shard.write_text(sample_to_jsonl(sample) + "\n", encoding="utf-8")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "scratch_dir: " + str(tmp_path) + "\nqa:\n  er_recon_psnr_min: 35.0\n",
        encoding="utf-8",
    )
    report = tmp_path / "report.jsonl"

    rc = stage0_probe_main(
        [
            str(shard),
            "--config",
            str(cfg),
            "--out",
            str(report),
            "--dry-run",
            "--max-iters",
            "0",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    summary = json.loads(captured.err.strip().splitlines()[-1])
    assert summary["accepted"] == 1
    assert summary["dropped"] == 0
    assert json.loads(report.read_text(encoding="utf-8"))["sample_id"] == "cli-ok"


def test_teacher_manifest_can_hash_explicit_checkpoint_file(tmp_path):
    ckpt = tmp_path / "teacher.bin"
    ckpt.write_bytes(b"teacher-weights")
    cfg = {
        "models": {
            "veraretouch": {
                "model_path": str(tmp_path),
                "teacher_sha256_path": str(ckpt),
                "greedy": True,
                "dtype": "float32",
                "max_new_tokens": 32,
                "chunk": 128,
            }
        }
    }

    teacher = teacher_manifest_from_config(cfg)

    assert teacher["teacher_sha256"] == (
        "30c86875b5a4e5f16c50289742b551a6d3248d13f922890be2df1cfe1f0319c1"
    )
    assert teacher["model_path"] == str(tmp_path)


def test_manifest_index_validator_checks_teacher_sentinel_and_shard_stats(tmp_path):
    ckpt = tmp_path / "teacher.bin"
    ckpt.write_bytes(b"teacher-weights")
    cfg = {
        "build_version": "test-v2",
        "schema_version": "datagen_v2",
        "models": {
            "veraretouch": {
                "model_path": str(tmp_path),
                "teacher_sha256_path": str(ckpt),
                "greedy": True,
                "dtype": "float32",
                "max_new_tokens": 32,
                "chunk": 128,
            }
        },
        "storage": {"shard_size": 1},
    }
    writer = ShardWriter(str(tmp_path), cfg)
    sample = _sample(
        "/not/read.jpg",
        Recipe(kind=RecipeKind.PARAM, params={}, provenance=Provenance.PARAM),
        sample_id="manifest-ok",
    )
    writer.write(sample)
    writer.flush()

    report = validate_manifest_index(str(tmp_path), cfg)

    assert report["ok"] is True
    assert report["rows"] == 1
    manifest_row = json.loads((tmp_path / "manifest_index.jsonl").read_text(encoding="utf-8"))
    assert manifest_row["teacher"]["teacher_sha256"] == (
        "30c86875b5a4e5f16c50289742b551a6d3248d13f922890be2df1cfe1f0319c1"
    )
    assert manifest_row["global_sentinel_sha256"]

    bad_cfg = {
        **cfg,
        "models": {
            "veraretouch": {
                **cfg["models"]["veraretouch"],
                "teacher_sha256_path": None,
                "teacher_sha256": "wrong",
            }
        },
    }
    bad = validate_manifest_index(str(tmp_path), bad_cfg)

    assert bad["ok"] is False
    assert any(i["kind"] == "teacher_mismatch" for i in bad["issues"])


def test_run_resume_rejects_manifest_invariant_mismatch_before_loading_models(tmp_path, monkeypatch):
    ckpt = tmp_path / "teacher.bin"
    ckpt.write_bytes(b"teacher-weights")
    cfg = {
        "build_version": "test-v2",
        "schema_version": "datagen_v2",
        "out_root": str(tmp_path / "out"),
        "models": {
            "veraretouch": {
                "model_path": str(tmp_path),
                "teacher_sha256_path": str(ckpt),
                "greedy": True,
                "dtype": "float32",
                "max_new_tokens": 32,
                "chunk": 128,
            },
            "ppr10k_masks": {"available": True},
        },
        "pilot": {"streams": {}},
        "budget": {"streams": {}},
        "storage": {"shard_size": 1},
    }
    writer = ShardWriter(cfg["out_root"], cfg)
    writer.write(
        _sample(
            "/not/read.jpg",
            Recipe(kind=RecipeKind.PARAM, params={}, provenance=Provenance.PARAM),
            sample_id="resume-manifest",
        )
    )
    writer.flush()
    bad_cfg = {
        **cfg,
        "models": {
            **cfg["models"],
            "veraretouch": {
                **cfg["models"]["veraretouch"],
                "teacher_sha256_path": None,
                "teacher_sha256": "wrong",
            },
        },
    }
    cfg_path = tmp_path / "bad_resume.yaml"
    import yaml

    cfg_path.write_text(yaml.safe_dump(bad_cfg), encoding="utf-8")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("real models should not be built after manifest mismatch")

    monkeypatch.setattr(run_mod, "_build_real_models", fail_if_called)

    with pytest.raises(SystemExit, match="manifest invariant check failed"):
        run_mod.main(["--config", str(cfg_path), "--resume", "--pilot"])


def test_audit_dataset_accepts_valid_datagen_contract_rows(tmp_path, capsys):
    import numpy as np
    from PIL import Image
    import yaml

    src = tmp_path / "src.png"
    raw_mask = tmp_path / "raw_mask.png"
    lut_path = tmp_path / "look.cube"
    after = tmp_path / "after.jpg"
    _write_rgb(src, 50)
    _write_rgb(after, 80)
    Image.fromarray(np.full((4, 4), 255, dtype="uint8"), mode="L").save(raw_mask)
    lut_path.write_text(
        "\n".join(
            [
                "LUT_3D_SIZE 2",
                "0 0 0",
                "1 0 0",
                "0 1 0",
                "1 1 0",
                "0 0 1",
                "1 0 1",
                "0 1 1",
                "1 1 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = {
        "build_version": "audit-v2",
        "schema_version": "datagen_v2",
        "storage": {"shard_size": 10},
        "models": {"veraretouch": {"teacher_sha256": "teacher"}},
    }
    writer = ShardWriter(str(tmp_path), cfg)
    s1 = _sample(
        str(src),
        Recipe(
            kind=RecipeKind.PARAM,
            params={"Exposure2012": {"value": 30.0}},
            provenance=Provenance.DEGRADE,
            degrade=DegradeSpec(
                mode="gaussian_op",
                op_params={"Exposure2012": -30.0},
                aspects=["L"],
                forward=False,
            ),
        ),
        stream=StreamId.S1_DEGRADE_LOCAL,
        sample_id="audit-s1",
        region_local=True,
        c_gt=CgtRef(mask_source=MaskSource.DEGRADE, raw_mask_path=str(raw_mask)),
    )
    cgt, grid, raw = writer.write_cgt(
        StreamId.S1_DEGRADE_LOCAL,
        "pending",
        s1.sample_id,
        np.ones((4, 4), dtype="float32"),
        raw_mask01=np.ones((4, 4), dtype="float32"),
    )
    s1.c_gt = CgtRef(
        cgt_path=cgt,
        cgt_patchgrid_path=grid,
        raw_mask_path=raw,
        mask_source=MaskSource.DEGRADE,
    )
    lut = _sample(
        str(src),
        Recipe(
            kind=RecipeKind.LUT,
            lut_recipe_id="lut",
            provenance=Provenance.LUT,
            meta={
                "path": str(lut_path),
                "domain_min": [0.0, 0.0, 0.0],
                "domain_max": [1.0, 1.0, 1.0],
                "lut_size": 2,
                "lut_sha256": "sha",
            },
        ),
        stream=StreamId.S6_RECIPE_GLOBAL,
        sample_id="audit-lut",
    )
    s5 = _sample(
        str(src),
        Recipe(kind=RecipeKind.PARAM, params={}, provenance=Provenance.PARAM),
        stream=StreamId.S5_GREYSKY_GLOBAL,
        sample_id="audit-s5",
        after_source=AfterSource.REAL_JPG,
        meta={"expert_after_path": str(after)},
    )
    for sample in (s1, lut, s5):
        sample.build_version = "audit-v2"
        sample.schema_version = "datagen_v2"
        writer.write(sample)
    writer.flush()

    report = audit_dataset(str(tmp_path), cfg)

    assert report["ok"] is True
    assert report["counts"]["rows"] == 3
    assert report["counts"]["degrade"] == 1
    assert report["counts"]["lut"] == 1
    assert report["counts"]["real_jpg"] == 1

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    rc = audit_main(["--config", str(cfg_path), "--out-root", str(tmp_path)])
    assert rc == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["ok"] is True


def test_audit_dataset_flags_contract_violations(tmp_path):
    src = tmp_path / "src.png"
    _write_rgb(src, 50)
    cfg = {
        "build_version": "audit-v2",
        "schema_version": "datagen_v2",
        "storage": {"shard_size": 10},
        "models": {"veraretouch": {"teacher_sha256": "teacher"}},
    }
    writer = ShardWriter(str(tmp_path), cfg)
    bad_degrade = _sample(
        str(src),
        Recipe(
            kind=RecipeKind.DEGRADE,
            params=None,
            provenance=Provenance.DEGRADE,
            degrade=None,
        ),
        stream=StreamId.S1_DEGRADE_LOCAL,
        sample_id="bad-degrade",
        region_local=True,
        c_gt=CgtRef(mask_source=MaskSource.GLOBAL),
    )
    bad_lut = _sample(
        str(src),
        Recipe(kind=RecipeKind.LUT, lut_recipe_id="lut", provenance=Provenance.PARAM, meta={}),
        stream=StreamId.S6_RECIPE_GLOBAL,
        sample_id="bad-lut",
    )
    bad_s5 = _sample(
        str(src),
        Recipe(kind=RecipeKind.PARAM, params={}, provenance=Provenance.PARAM),
        stream=StreamId.S5_GREYSKY_GLOBAL,
        sample_id="bad-s5",
        meta={"expert_after_path": str(tmp_path / "after.jpg")},
    )
    for sample in (bad_degrade, bad_lut, bad_s5):
        sample.build_version = "audit-v2"
        sample.schema_version = "datagen_v2"
        writer.write(sample)
    writer.flush()

    report = audit_dataset(str(tmp_path), cfg, strict_paths=False)
    kinds = {issue["kind"] for issue in report["issues"]}

    assert report["ok"] is False
    assert "legacy_degrade_kind" in kinds
    assert "degrade_not_param_kind" in kinds
    assert "degrade_missing_neg_params" in kinds
    assert "s1_missing_raw_mask" in kinds
    assert "lut_missing_lut_sha256" in kinds
    assert "lut_provenance_mismatch" in kinds
    assert "s5_expert_path_without_real_jpg_after_source" in kinds


def test_done_ids_respects_custom_shard_prefix_for_sharded_resume(tmp_path):
    writer = ShardWriter(
        str(tmp_path),
        {
            "storage": {"shard_size": 1, "shard_prefix": "shard_w0of2"},
            "build_version": "test",
            "schema_version": "datagen_v2",
        },
    )
    sample = _sample(
        "/not/read.jpg",
        Recipe(kind=RecipeKind.PARAM, params={}, provenance=Provenance.PARAM),
    )
    sample.sample_id = "sid-custom-prefix"
    writer.write(sample)
    writer.flush()

    resumed = ShardWriter(
        str(tmp_path),
        {
            "storage": {"shard_size": 1, "shard_prefix": "shard_w0of2"},
            "build_version": "test",
            "schema_version": "datagen_v2",
        },
    )

    assert "sid-custom-prefix" in set(resumed.done_ids())
