"""Unit tests for the model-side SFT pipeline (no GPU, no model weights).

    /home/bc/envs/q3vl_sft/bin/python -m pytest q3vl/train/tests -q
"""

from __future__ import annotations

import json
import os
import sys
import tarfile
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from q3vl.train.collator import SequenceTooLong, Sft2SegCollator  # noqa: E402
from q3vl.train.constants import (  # noqa: E402
    IGNORE_INDEX, SEG_COLOR, SEG_EOS, SEG_IGNORE, SEG_WHERE, SPECIAL_TOKENS,
)
from q3vl.train.dataset import RecordSchemaError, Sft2SegDataset  # noqa: E402
from q3vl.train.diagnostics import aggregate_generation_diagnostics, parse_two_segment  # noqa: E402
from q3vl.train.freeze import is_trainable_param  # noqa: E402
from q3vl.train.imageproc import ImageRejected, plan_geometry, prepare_image  # noqa: E402
from q3vl.train.mock_shards import build_mock_shard  # noqa: E402
from q3vl.train.shards import (  # noqa: E402
    ShardIndex, ShardIntegrityError, ShardStore, TerminalManifest,
)
from q3vl.train.tokens import register_special_tokens, verify_single_token  # noqa: E402

MODEL_PATH = os.environ.get("Q3VL_MODEL_PATH", "/home/bc/data/models/Qwen3-VL-4B-Instruct")


# --------------------------------------------------------------- image spec 5
class TestImageGeometry:
    @pytest.mark.parametrize(
        "h,w,eh,ew,ntok",
        [
            (512, 512, 512, 512, 256),
            (1024, 1024, 512, 512, 256),
            (600, 900, 512, 768, 384),
            (900, 600, 768, 512, 384),
            (500, 2000, 512, 2048, 1024),   # exactly 4:1
            (4000, 3000, 683 // 1 * 0 + 672, 512, 336),  # 4:3 portrait -> 672x512
        ],
    )
    def test_target_sizes(self, h, w, eh, ew, ntok):
        g = plan_geometry(h, w)
        assert (g.out_h, g.out_w) == (eh, ew)
        assert g.n_visual_tokens == ntok
        assert min(g.out_h, g.out_w) == 512
        assert g.out_h % 32 == 0 and g.out_w % 32 == 0
        assert max(g.out_h, g.out_w) <= 2048

    def test_rejects_extreme_aspect(self):
        with pytest.raises(ImageRejected) as e:
            plan_geometry(400, 2000)  # 5:1
        assert e.value.reason == "aspect_ratio"

    def test_aspect_ratio_bounds_long_side(self):
        # ar<=4 with short side 512 implies long side <=2048 by construction
        for w in range(512, 2049, 7):
            g = plan_geometry(512, w) if w / 512 <= 4 else None
            if g:
                assert max(g.out_h, g.out_w) <= 2048

    def test_exif_and_rgb(self, tmp_path):
        from PIL import Image

        p = tmp_path / "a.png"
        Image.new("L", (800, 600)).save(p)
        img, geom = prepare_image(str(p))
        assert img.mode == "RGB"
        assert img.size == (geom.out_w, geom.out_h)
        assert (geom.out_h, geom.out_w) == (512, 683 // 32 * 32 + 0) or geom.out_h == 512

    def test_corrupt_image_rejected(self, tmp_path):
        p = tmp_path / "bad.png"
        p.write_bytes(b"not an image")
        with pytest.raises(ImageRejected) as e:
            prepare_image(str(p))
        assert e.value.reason == "image_corrupt"


# --------------------------------------------------------- indexed tar shards
@pytest.fixture(scope="module")
def mock(tmp_path_factory):
    root = tmp_path_factory.mktemp("sft2seg")
    info = build_mock_shard(root, n_samples=6, split="train")
    build_mock_shard(root, n_samples=3, split="eval", shard_name="shard-eval.tar", seed=99)
    return info


class TestShards:
    def test_index_layout_and_random_read(self, mock):
        index = ShardIndex.load(mock["index"], split="train")
        assert index.layout == "nested"
        assert len(index) == 6
        store = ShardStore(mock["shard_root"], verify="checksum")
        # read out of order to exercise random positioning
        for i in (4, 0, 5, 2):
            ref = index[i]
            blob = store.read(ref.members["record"])
            rec = json.loads(blob)
            assert rec["sample_id"] == ref.sample_id
        assert store.n_checksum_verified == 4

    def test_offsets_agree_with_tarfile(self, mock):
        index = ShardIndex.load(mock["index"], split="train")
        with tarfile.open(mock["shard"]) as tar:
            by_name = {m.name: m for m in tar.getmembers()}
        for ref in index.samples:
            for m in ref.members.values():
                assert by_name[m.member].offset_data == m.offset
                assert by_name[m.member].size == m.length

    def test_checksum_mismatch_detected(self, mock, tmp_path):
        index = ShardIndex.load(mock["index"], split="train")
        store = ShardStore(mock["shard_root"], verify="checksum")
        ref = index[0].members["record"]
        bad = type(ref)(ref.shard, ref.member, ref.offset, ref.length, ref.size, "sha256:" + "0" * 64)
        with pytest.raises(ShardIntegrityError, match="mismatch"):
            store.read(bad)

    def test_truncated_read_detected(self, mock):
        index = ShardIndex.load(mock["index"], split="train")
        store = ShardStore(mock["shard_root"], verify="none")
        ref = index[0].members["record"]
        bad = type(ref)(ref.shard, ref.member, 10**9, ref.length, None, None)
        with pytest.raises(ShardIntegrityError, match="short read"):
            store.read(bad)

    def test_flat_layout_supported(self, mock, tmp_path):
        index = ShardIndex.load(mock["index"], split="train")
        flat = tmp_path / "flat.jsonl"
        with flat.open("w") as fh:
            for s in index.samples:
                for role, m in s.members.items():
                    fh.write(json.dumps({"sample_id": s.sample_id, "role": role, **m.to_dict()}) + "\n")
        loaded = ShardIndex.load(flat)
        assert loaded.layout == "per_member"
        assert len(loaded) == len(index)
        assert set(loaded[0].members) == {"image", "record"}

    def test_unknown_layout_raises_with_keys(self, tmp_path):
        p = tmp_path / "weird.jsonl"
        p.write_text(json.dumps({"sample_id": "x", "blah": 1}) + "\n")
        with pytest.raises(Exception, match="observed keys"):
            ShardIndex.load(p)

    def test_manifest_n_effective(self, mock):
        man = TerminalManifest.load(mock["manifest"])
        assert man.n_effective("train") == 6
        assert man.n_effective("eval") == 3
        assert man.digest()

    def test_manifest_refuses_to_guess(self, tmp_path):
        p = tmp_path / "m.json"
        p.write_text(json.dumps({"schema_version": "x"}))
        with pytest.raises(Exception, match="N_effective"):
            TerminalManifest.load(p).n_effective("train")


# ------------------------------------------------------------------- dataset
class TestDataset:
    def test_reads_samples(self, mock):
        index = ShardIndex.load(mock["index"], split="train")
        ds = Sft2SegDataset(index, ShardStore(mock["shard_root"]))
        s = ds[0]
        assert s.instruction and s.where_text and s.color_text
        assert s.image.mode == "RGB"
        assert min(s.geometry.out_h, s.geometry.out_w) == 512

    def test_refuses_missing_segments(self, mock, tmp_path):
        index = ShardIndex.load(mock["index"], split="train")
        ds = Sft2SegDataset(index, ShardStore(mock["shard_root"]))
        with pytest.raises(RecordSchemaError, match="where segment"):
            ds.extract_segments("x", {"instruction": "i", "color": "c"})

    def test_rejects_legacy_tag_leak(self, mock):
        index = ShardIndex.load(mock["index"], split="train")
        ds = Sft2SegDataset(index, ShardStore(mock["shard_root"]))
        with pytest.raises(RecordSchemaError, match="legacy"):
            ds.extract_segments("x", {"where": "w", "color": "<plan_light_start>c<plan_light_end>"})

    def test_local_assembly_is_opt_in(self, mock):
        index = ShardIndex.load(mock["index"], split="train")
        rec = {
            "region_scope": "the sky",
            "problem_lighting": "a", "problem_global_color": "b", "problem_specific_color": "c",
            "plan_lighting": "d", "plan_global_color": "e", "plan_specific_color": "f",
        }
        strict = Sft2SegDataset(index, ShardStore(mock["shard_root"]), allow_local_assembly=False)
        with pytest.raises(RecordSchemaError):
            strict.extract_segments("x", rec)
        lenient = Sft2SegDataset(index, ShardStore(mock["shard_root"]), allow_local_assembly=True)
        where, color = lenient.extract_segments("x", rec)
        assert where == "the sky"
        assert color.splitlines() == ["a", "b", "c", "d", "e", "f"]  # spec 4.2 order


# ------------------------------------------------------ tokens + loss masking
@pytest.fixture(scope="module")
def processor():
    pytest.importorskip("transformers")
    if not Path(MODEL_PATH).exists():
        pytest.skip(f"model not available at {MODEL_PATH}")
    from transformers import AutoProcessor

    p = AutoProcessor.from_pretrained(MODEL_PATH)
    register_special_tokens(p.tokenizer)
    return p


class TestSpecialTokens:
    def test_single_token_and_distinct(self, processor):
        ids = verify_single_token(processor.tokenizer)
        # v2seg: 6 tokens (the original 4 + <seg_where>/<seg_color>)
        assert len(set(ids.values())) == len(SPECIAL_TOKENS)
        assert list(ids) == list(SPECIAL_TOKENS)

    def test_registration_is_idempotent(self, processor):
        before = len(processor.tokenizer)
        register_special_tokens(processor.tokenizer)
        assert len(processor.tokenizer) == before

    def test_reload_keeps_ids(self, processor, tmp_path):
        from q3vl.train.tokens import verify_reload

        ids = verify_single_token(processor.tokenizer)
        processor.save_pretrained(str(tmp_path / "tok"))
        info = verify_reload(str(tmp_path / "tok"), ids)
        assert info["ids"] == ids


class TestCollator:
    def test_loss_mask_and_segments(self, processor, mock):
        index = ShardIndex.load(mock["index"], split="train")
        ds = Sft2SegDataset(index, ShardStore(mock["shard_root"]))
        col = Sft2SegCollator(processor)
        batch = col([ds[i] for i in range(4)])

        labels, segs, ids = batch["labels"], batch["segment_ids"], batch["input_ids"]
        supervised = labels != IGNORE_INDEX
        # nothing before the assistant turn is supervised
        assert (segs[~supervised] == SEG_IGNORE).all()
        # image placeholders are never supervised
        img_id = processor.image_token_id
        assert int(((ids == img_id) & supervised).sum()) == 0
        # supervised labels equal the inputs at the same positions
        assert (labels[supervised] == ids[supervised]).all()
        # where strictly precedes color in every row
        for r in range(ids.shape[0]):
            w = (segs[r] == SEG_WHERE).nonzero().flatten()
            c = (segs[r] == SEG_COLOR).nonzero().flatten()
            e = (segs[r] == SEG_EOS).nonzero().flatten()
            assert len(w) and len(c) and len(e)
            assert w[-1] < c[0] < e[0]

    def test_four_tokens_supervised_once(self, processor, mock):
        index = ShardIndex.load(mock["index"], split="train")
        ds = Sft2SegDataset(index, ShardStore(mock["shard_root"]))
        col = Sft2SegCollator(processor)
        batch = col([ds[0]])
        ids = verify_single_token(processor.tokenizer)
        row, sup = batch["input_ids"][0], batch["labels"][0] != IGNORE_INDEX
        for tok, tid in ids.items():
            pos = (row == tid).nonzero().flatten()
            assert len(pos) == 1, f"{tok} appears {len(pos)} times"
            assert bool(sup[pos[0]]), f"{tok} is not supervised"

    def test_image_token_count_matches_geometry(self, processor, mock):
        index = ShardIndex.load(mock["index"], split="train")
        ds = Sft2SegDataset(index, ShardStore(mock["shard_root"]))
        col = Sft2SegCollator(processor)
        for i in range(4):
            s = ds[i]
            enc = col.encode_one(s)
            n = sum(1 for t in enc["input_ids"] if t == processor.image_token_id)
            assert n == s.geometry.n_visual_tokens == (s.geometry.out_h // 32) * (s.geometry.out_w // 32)

    def test_piecewise_equals_whole_string(self, processor, mock):
        index = ShardIndex.load(mock["index"], split="train")
        ds = Sft2SegDataset(index, ShardStore(mock["shard_root"]))
        col = Sft2SegCollator(processor)
        for i in range(4):
            assert col.check_concat_equivalence(ds[i])["equal"]

    def test_overlong_raises_not_truncates(self, processor, mock):
        index = ShardIndex.load(mock["index"], split="train")
        ds = Sft2SegDataset(index, ShardStore(mock["shard_root"]))
        col = Sft2SegCollator(processor, max_length=64)
        with pytest.raises(SequenceTooLong):
            col([ds[0]])

    def test_eos_supervision_toggle(self, processor, mock):
        index = ShardIndex.load(mock["index"], split="train")
        ds = Sft2SegDataset(index, ShardStore(mock["shard_root"]))
        off = Sft2SegCollator(processor, supervise_eos=False)([ds[0]])
        assert int((off["segment_ids"] == SEG_EOS).sum()) == 0

    def test_padding_is_ignored(self, processor, mock):
        index = ShardIndex.load(mock["index"], split="train")
        ds = Sft2SegDataset(index, ShardStore(mock["shard_root"]))
        col = Sft2SegCollator(processor)
        batch = col([ds[i] for i in range(4)])
        pad = batch["attention_mask"] == 0
        assert (batch["labels"][pad] == IGNORE_INDEX).all()
        assert pad.any(), "expected ragged lengths in the mock batch"

    def test_pixel_values_match_grid(self, processor, mock):
        index = ShardIndex.load(mock["index"], split="train")
        ds = Sft2SegDataset(index, ShardStore(mock["shard_root"]))
        batch = Sft2SegCollator(processor)([ds[i] for i in range(4)])
        grid = batch["image_grid_thw"]
        assert int(grid.prod(dim=1).sum()) == batch["pixel_values"].shape[0]
        merge = processor.image_processor.merge_size ** 2
        n_llm = int(grid.prod(dim=1).sum()) // merge
        assert n_llm == int((batch["input_ids"] == processor.image_token_id).sum())


# ------------------------------------------------------------------- freezing
class TestFreezeRule:
    @pytest.mark.parametrize(
        "name,trainable",
        [
            ("model.visual.blocks.0.attn.qkv.weight", False),
            ("model.visual.blocks.23.mlp.linear_fc2.bias", False),
            ("model.visual.patch_embed.proj.weight", False),
            ("model.visual.pos_embed.weight", False),
            ("model.visual.merger.linear_fc1.weight", True),
            ("model.visual.merger.norm.bias", True),
            ("model.visual.deepstack_merger_list.0.linear_fc1.weight", True),
            ("model.visual.deepstack_merger_list.2.norm.weight", True),
            ("model.language_model.embed_tokens.weight", True),
            ("model.language_model.layers.35.mlp.down_proj.weight", True),
            ("model.language_model.norm.weight", True),
            ("lm_head.weight", True),
        ],
    )
    def test_rule(self, name, trainable):
        assert is_trainable_param(name) is trainable


# ------------------------------------------------------- ZeRO-3 partitioning
class _FakeDSParam(torch.nn.Parameter):
    """A parameter shaped like one that ``deepspeed.zero.Init`` has freed.

    Reproduces the observed state on 2xH100 (probe transcript in NOTES.md):
    ``shape == torch.Size([0])`` and ``numel() == 0`` while ``ds_shape`` /
    ``ds_numel`` carry the real geometry.
    """

    def __new__(cls, ds_shape, dtype=torch.bfloat16):
        obj = super().__new__(cls, torch.empty(0, dtype=dtype), requires_grad=True)
        obj.ds_shape = torch.Size(ds_shape)
        obj.ds_numel = int(torch.Size(ds_shape).numel())
        obj.ds_id = 0
        return obj


class TestZero3ParameterAccounting:
    """Regression tests for the two ZeRO-3 defects the joint smoke exposed."""

    def test_full_numel_uses_ds_numel(self):
        from q3vl.train.freeze import full_numel, full_shape

        p = _FakeDSParam((151936, 2560))
        assert p.numel() == 0 and tuple(p.shape) == (0,)
        assert full_numel(p) == 151936 * 2560
        assert full_shape(p) == (151936, 2560)

    def test_full_numel_plain_parameter(self):
        from q3vl.train.freeze import full_numel, full_shape

        p = torch.nn.Parameter(torch.zeros(7, 3))
        assert full_numel(p) == 21
        assert full_shape(p) == (7, 3)

    def test_vacuous_report_is_rejected(self):
        """All-zero counts satisfy every 'must be 0' rule; that must not pass."""
        from q3vl.train.freeze import FreezeBoundaryError, FreezeReport, assert_freeze_boundary

        class _Tree(torch.nn.Module):
            def named_parameters(self, *a, **kw):
                names = (
                    [f"model.visual.blocks.{i}.attn.qkv.weight" for i in range(24)]
                    + [f"model.visual.deepstack_merger_list.{i}.linear_fc1.weight" for i in range(3)]
                    + ["model.visual.merger.linear_fc1.weight",
                       "model.visual.patch_embed.proj.weight",
                       "model.language_model.embed_tokens.weight",
                       "model.language_model.layers.0.mlp.down_proj.weight"]
                )
                for n in names:
                    p = _FakeDSParam((4, 4))
                    p.requires_grad_(is_trainable_param(n))
                    yield n, p

        report = FreezeReport()
        report.subtree_counts = {
            k: {"params": 0, "tensors": 0, "trainable_params": 0, "frozen_params": 0}
            for k in ("vision_blocks", "vision_patch_embed", "vision_pos_embed", "vision_merger",
                      "vision_deepstack_mergers", "language_embed_tokens", "language_layers",
                      "language_norm", "lm_head")
        }
        with pytest.raises(FreezeBoundaryError, match="0 parameters"):
            assert_freeze_boundary(_Tree(), report)

    def test_apply_freeze_counts_partitioned_params(self):
        from q3vl.train.freeze import apply_arm_b_freeze

        names = (
            [f"model.visual.blocks.{i}.attn.qkv.weight" for i in range(24)]
            + [f"model.visual.deepstack_merger_list.{i}.linear_fc1.weight" for i in range(3)]
            + ["model.visual.merger.linear_fc1.weight",
               "model.visual.patch_embed.proj.weight",
               "model.language_model.embed_tokens.weight",
               "model.language_model.layers.0.mlp.down_proj.weight"]
        )
        params = [(n, _FakeDSParam((8, 8))) for n in names]

        class _Tree(torch.nn.Module):
            def named_parameters(self, *a, **kw):
                return iter(params)

        report = apply_arm_b_freeze(_Tree())
        assert report.total_params == len(params) * 64
        assert report.trainable_params == 6 * 64
        assert report.frozen_params == 25 * 64
        assert report.subtree_counts["vision_blocks"]["frozen_params"] == 24 * 64
        assert report.subtree_counts["vision_merger"]["trainable_params"] == 64
        assert report.subtree_counts["vision_deepstack_mergers"]["trainable_params"] == 3 * 64

    def test_prepare_embeddings_does_not_resize_a_partitioned_matrix(self):
        """`emb.weight.shape[0]` reads 0 under ZeRO-3 -- the never-shrink policy
        (NOTES.md D-4) must survive that."""
        from q3vl.train.tokens import prepare_embeddings

        rows, dim = 151936, 8
        real = torch.randn(rows, dim, dtype=torch.float32)

        class _Emb(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = _FakeDSParam((rows, dim), dtype=torch.float32)

        class _Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = _Emb()
                self.config = type("C", (), {"tie_word_embeddings": False})()
                self.resize_calls = []

            def get_input_embeddings(self):
                return self.emb

            def get_output_embeddings(self):
                return None

            def resize_token_embeddings(self, *a, **kw):
                self.resize_calls.append((a, kw))

        model = _Model()
        tok = type("T", (), {"__len__": lambda self: 151673})()
        ids = {"<where>": 151669, "</where>": 151670, "<color>": 151671, "</color>": 151672}

        # gather window: swap in the real matrix, exactly like GatheredParameters
        import q3vl.train.tokens as tokens_mod
        import contextlib

        @contextlib.contextmanager
        def _fake_gather(params, modifier_rank=0):
            model.emb.weight.data = real
            try:
                yield
            finally:
                real.copy_(model.emb.weight.data)
                model.emb.weight.data = torch.empty(0, dtype=torch.float32)

        original = tokens_mod.gathered
        tokens_mod.gathered = _fake_gather
        try:
            info = prepare_embeddings(model, tok, ids, seed=0, reinit_new_rows=True)
        finally:
            tokens_mod.gathered = original

        assert model.resize_calls == [], "resize_token_embeddings must not be called"
        assert info["embedding_rows_before"] == rows
        assert info["resize_action"] == "kept"
        assert info["zero3_partitioned"] is True
        assert info["reinit"]["min_pairwise_distance"] > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
    def test_mean_init_rows_on_cuda_weight(self):
        """The noise came from a CPU generator and was multiplied by a CUDA std."""
        from q3vl.train.tokens import _mean_init_rows

        w = torch.randn(64, 16, dtype=torch.bfloat16, device="cuda")
        gen = torch.Generator(device="cpu").manual_seed(0)
        stats = _mean_init_rows(w, [60, 61, 62, 63], 60, gen)
        assert stats["ref_rows"] == 60
        assert torch.isfinite(w[60:].float()).all()
        assert not torch.equal(w[60], w[61])

    def test_mean_init_rows_is_device_independent(self):
        from q3vl.train.tokens import _mean_init_rows

        base = torch.randn(64, 16, dtype=torch.float32)
        a, b = base.clone(), base.clone()
        _mean_init_rows(a, [60, 61, 62, 63], 60, torch.Generator(device="cpu").manual_seed(0))
        _mean_init_rows(b, [60, 61, 62, 63], 60, torch.Generator(device="cpu").manual_seed(0))
        assert torch.equal(a, b)


# ---------------------------------------------------------------- diagnostics
class TestDiagnostics:
    def test_parse_good(self):
        p = parse_two_segment("<where>the sky</where><color>warm it</color>")
        assert p["tags_complete"] and p["order_ok"]
        assert p["where_nonempty"] and p["color_nonempty"]
        assert not p["legacy_tag_leak"]

    def test_parse_wrong_order(self):
        p = parse_two_segment("<color>warm it</color><where>the sky</where>")
        assert p["tags_complete"] and not p["order_ok"]

    def test_parse_missing_close(self):
        assert not parse_two_segment("<where>sky<color>c</color>")["tags_complete"]

    def test_legacy_leak_detected(self):
        p = parse_two_segment("<where>s</where><color><plan_light_start>x</plan_light_end></color>")
        assert p["legacy_tag_leak"]

    def test_aggregate(self):
        parsed = [
            parse_two_segment("<where>a</where><color>b</color>"),
            parse_two_segment("<color>b</color><where>a</where>"),
        ]
        agg = aggregate_generation_diagnostics(parsed)
        assert agg["eval_gen_n"] == 2
        assert agg["eval_gen_tag_completeness"] == 1.0
        assert agg["eval_gen_order_accuracy"] == 0.5

    def test_segment_token_stats(self):
        from q3vl.train.diagnostics import segment_token_stats

        V = 7
        logits = torch.zeros(1, 5, V)
        labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 3, 4, 5]])
        segs = torch.tensor([[SEG_IGNORE, SEG_IGNORE, SEG_WHERE, SEG_COLOR, SEG_EOS]])
        per_token, correct, s = segment_token_stats(logits, labels, segs)
        assert per_token.numel() == 3  # positions 2,3,4 after the causal shift
        assert set(s.tolist()) == {SEG_WHERE, SEG_COLOR, SEG_EOS}
        assert torch.allclose(per_token, torch.full((3,), float(torch.log(torch.tensor(float(V))))))


# ------------------------------------------------------------ frozen hparams
class TestFrozenHyperparameters:
    def _args(self, tmp_path, **over):
        from q3vl.train.args import SFTTrainingArguments

        return SFTTrainingArguments(output_dir=str(tmp_path), **over)

    def test_spec_defaults_pass(self, tmp_path):
        from q3vl.train.args import validate_frozen_hyperparameters

        info = validate_frozen_hyperparameters(self._args(tmp_path), world_size=2)
        assert info["effective_global_batch"] == 32

    def test_oom_fallback_combo_allowed(self, tmp_path):
        from q3vl.train.args import validate_frozen_hyperparameters

        args = self._args(tmp_path, per_device_train_batch_size=2, gradient_accumulation_steps=8)
        assert validate_frozen_hyperparameters(args, world_size=2)["effective_global_batch"] == 32

    def test_lr_drift_rejected(self, tmp_path):
        from q3vl.train.args import validate_frozen_hyperparameters

        with pytest.raises(ValueError, match="learning_rate"):
            validate_frozen_hyperparameters(self._args(tmp_path, learning_rate=2e-5), world_size=2)

    def test_bad_batch_combo_rejected(self, tmp_path):
        from q3vl.train.args import validate_frozen_hyperparameters

        args = self._args(tmp_path, per_device_train_batch_size=8, gradient_accumulation_steps=2)
        with pytest.raises(ValueError, match="global batch|not one of"):
            validate_frozen_hyperparameters(args, world_size=2)

    def test_single_gpu_rejected(self, tmp_path):
        from q3vl.train.args import validate_frozen_hyperparameters

        with pytest.raises(ValueError, match="world_size"):
            validate_frozen_hyperparameters(self._args(tmp_path), world_size=1)


# ------------------------------------------------- protected checkpoint logic
class TestProtectedSteps:
    def test_derived_from_max_steps(self):
        from q3vl.train.trainer import Qwen3VLSFTTrainer

        t = Qwen3VLSFTTrainer.__new__(Qwen3VLSFTTrainer)
        t.protected_steps = set()
        t.expected_steps_per_epoch = None
        assert t.resolve_protected_steps(5290) == {2645, 5290}
        assert t.resolve_protected_steps(4001) == {2000, 4001}
        # never the spec's estimate unless the data says so
        assert 2645 not in t.resolve_protected_steps(1234)


# ------------------------------- checkpoint deletion paths (no model weights)
class TestCheckpointProtection:
    """Both transformers deletion paths must spare the 0.5/1.0 epoch milestones.

    Path A: Trainer._rotate_checkpoints (save_total_limit rolling cleanup).
    Path B: the end-of-_inner_training_loop block that fires when
            save_total_limit == 1 and best_model_checkpoint is set. It does NOT
            call _rotate_checkpoints, so protecting only that method is not
            enough -- both source their list from _sorted_checkpoints.
    """

    def _trainer(self, out_dir, protected, limit=3, best=None):
        from q3vl.train.trainer import Qwen3VLSFTTrainer

        t = Qwen3VLSFTTrainer.__new__(Qwen3VLSFTTrainer)
        t.protected_steps = set(protected)

        class _Args:
            save_total_limit = limit
            output_dir = str(out_dir)
            should_save = True

        class _State:
            best_model_checkpoint = best

        t.args, t.state = _Args(), _State()
        return t

    def _mk(self, root, steps):
        for s in steps:
            (root / f"checkpoint-{s}").mkdir(parents=True)

    @staticmethod
    def _left(root):
        return sorted(int(p.name.split("-")[1]) for p in root.glob("checkpoint-*"))

    @staticmethod
    def _simulate_end_of_training_cleanup(trainer, run_dir):
        """Verbatim copy of transformers 4.57.1 trainer.py:2841-2845."""
        import os
        import shutil

        checkpoints_sorted = trainer._sorted_checkpoints(use_mtime=False, output_dir=str(run_dir))
        if (
            trainer.args.should_save
            and trainer.state.best_model_checkpoint is not None
            and trainer.args.save_total_limit == 1
        ):
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, trainer.state.best_model_checkpoint):
                    shutil.rmtree(checkpoint, ignore_errors=True)

    # -- path A ------------------------------------------------------------
    def test_rotation_spares_protected(self, tmp_path):
        self._mk(tmp_path, [500, 1000, 1500, 2000, 2645, 3000])
        self._trainer(tmp_path, protected={2645, 5290}, limit=3)._rotate_checkpoints(
            output_dir=str(tmp_path)
        )
        left = self._left(tmp_path)
        assert 2645 in left, "0.5-epoch milestone was deleted by save_total_limit"
        assert left == [1500, 2000, 2645, 3000]

    def test_rotation_still_prunes_unprotected(self, tmp_path):
        self._mk(tmp_path, [100, 200, 300, 400, 500])
        self._trainer(tmp_path, protected=set(), limit=2)._rotate_checkpoints(
            output_dir=str(tmp_path)
        )
        assert self._left(tmp_path) == [400, 500]

    def test_no_limit_is_noop(self, tmp_path):
        self._mk(tmp_path, [100, 200])
        self._trainer(tmp_path, protected=set(), limit=None)._rotate_checkpoints(
            output_dir=str(tmp_path)
        )
        assert self._left(tmp_path) == [100, 200]

    # -- path B ------------------------------------------------------------
    def test_end_of_training_cleanup_spares_protected(self, tmp_path):
        self._mk(tmp_path, [2645, 5000, 5290])
        t = self._trainer(
            tmp_path, protected={2645, 5290}, limit=1,
            best=str(tmp_path / "checkpoint-5290"),
        )
        self._simulate_end_of_training_cleanup(t, tmp_path)
        left = self._left(tmp_path)
        assert 2645 in left, "path B deleted the 0.5-epoch milestone"
        assert left == [2645, 5290]

    def test_end_of_training_cleanup_still_prunes_unprotected(self, tmp_path):
        self._mk(tmp_path, [100, 200, 300])
        t = self._trainer(
            tmp_path, protected=set(), limit=1, best=str(tmp_path / "checkpoint-300")
        )
        self._simulate_end_of_training_cleanup(t, tmp_path)
        assert self._left(tmp_path) == [300]

    def test_protected_best_checkpoint_does_not_crash_sorting(self, tmp_path):
        # base _sorted_checkpoints indexes best_model_checkpoint in the list;
        # filtering must happen after that, or it raises ValueError.
        self._mk(tmp_path, [1000, 2645])
        t = self._trainer(
            tmp_path, protected={2645}, limit=1, best=str(tmp_path / "checkpoint-2645")
        )
        assert t._sorted_checkpoints(output_dir=str(tmp_path)) == [str(tmp_path / "checkpoint-1000")]


class TestProtectedCallback:
    def test_forces_save_and_eval_on_milestone(self):
        from transformers import TrainerControl

        from q3vl.train.trainer import ProtectedStepCallback, Qwen3VLSFTTrainer

        t = Qwen3VLSFTTrainer.__new__(Qwen3VLSFTTrainer)
        t.protected_steps = {2645, 5290}
        cb = ProtectedStepCallback(t)

        class _State:
            global_step = 2645

        ctrl = TrainerControl()
        ctrl.should_save = False
        ctrl.should_evaluate = False
        cb.on_step_end(None, _State(), ctrl)
        assert ctrl.should_save and ctrl.should_evaluate

        _State.global_step = 2644
        ctrl2 = TrainerControl()
        ctrl2.should_save = False
        cb.on_step_end(None, _State(), ctrl2)
        assert not ctrl2.should_save


# ------------------------------------------------ interop with S0-DATA output
class TestS0DataInterop:
    """Reader must accept the exact index/manifest shape q3vl.data.pipeline emits.

    Row shape copied from q3vl/data/pipeline.py::stage_manifest (2026-08-05):
    nested ``members`` with absolute ``shard`` paths and bare-hex ``sha256``;
    manifest carries both ``counts.<split>.n_effective`` and top-level
    ``n_effective``. Split names are train / V_where / V_what / T_final /
    T_lut_unseen -- there is no split literally called "eval".
    """

    def _rows(self, mock, split):
        import hashlib

        index = ShardIndex.load(mock["index"], split="train")
        out = []
        for s in index.samples:
            members = {}
            for role, m in s.members.items():
                blob = ShardStore(mock["shard_root"], verify="none").read(m)
                members[role] = {
                    "shard": str(Path(mock["shard_root"]) / m.shard),  # absolute
                    "member": m.member, "offset": m.offset,
                    "length": m.length, "size": m.size,
                    "sha256": hashlib.sha256(blob).hexdigest(),        # bare hex
                }
            out.append({
                "sample_id": s.sample_id, "split": split, "members": members,
                "build": "g1", "task_type": "global", "source_image_id": "src-0",
                "lut_id": "lut-0", "winner_confidence": "high",
                "n_visual_tokens": 256, "total_tokens": 400,
            })
        return out

    def test_reads_producer_index_shape(self, mock, tmp_path):
        p = tmp_path / "V_where.index.jsonl"
        rows = self._rows(mock, "V_where")
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))

        index = ShardIndex.load(p, split="V_where")
        assert index.layout == "nested"
        assert len(index) == len(rows)
        # absolute shard paths must resolve even with an unrelated shard_root
        store = ShardStore(tmp_path / "unused", verify="checksum")
        rec = json.loads(store.read(index[0].members["record"]))
        assert rec["sample_id"] == index[0].sample_id
        assert store.n_checksum_verified == 1

    def test_bare_hex_sha256_is_verified(self, mock, tmp_path):
        p = tmp_path / "x.index.jsonl"
        rows = self._rows(mock, "train")
        rows[0]["members"]["record"]["sha256"] = "0" * 64
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
        index = ShardIndex.load(p)
        with pytest.raises(ShardIntegrityError, match="sha256 mismatch"):
            ShardStore(tmp_path, verify="checksum").read(index[0].members["record"])

    def test_producer_manifest_keys(self, tmp_path):
        p = tmp_path / "terminal_manifest.json"
        p.write_text(json.dumps({
            "schema_version": "q3vl.sft2seg.splits/1",
            "status": "complete",
            "counts": {
                "train": {"n_effective": 160123, "sample_count": 160123},
                "V_where": {"n_effective": 1100, "sample_count": 1100},
            },
            "n_effective": 160123,
            "digest": "ab" * 32,
        }))
        man = TerminalManifest.load(p)
        assert man.n_effective("train") == 160123
        assert man.n_effective("V_where") == 1100
        assert man.digest() == "ab" * 32
        assert man.schema_version() == "q3vl.sft2seg.splits/1"

    def test_producer_target_string_matches_collator(self, processor):
        # producer NOTES: "<where>" + where + "</where>" + "<color>" + color +
        # "</color>", no newline between tag and body.
        # v2seg (2026-08-14) appends "<seg_where><seg_color>" after "</color>".
        from q3vl.train.collator import Sft2SegCollator

        where, color = "the standing subject", "line a\nline b"
        assert (
            Sft2SegCollator.build_target_text(where, color)
            == f"<where>{where}</where><color>{color}</color><seg_where><seg_color>"
        )
