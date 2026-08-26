"""EPR-033: the adapter surface, the frozen area, and the two z parities.

Every test here runs on a toy VLM with the *shape* of the real one
(``model.language_model.layers[i].self_attn.{q,k,v,o}_proj`` and a separate
``model.visual.blocks``), because what is being tested is wiring, not weights:
which modules got an adapter, which parameters can move, and whether the
adapter is the zero map before the first step.  The three assertions this arm
puts in front of its first training step are exercised in both directions --
they pass on a correct wiring and **raise** on a broken one.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from torch import nn

from q3vl.whatb import loraspan as L
from q3vl.whatb.readout import WhatReadoutBuilder, WhatReadoutSpec

pytest.importorskip("peft")

D = L.Z_DIM          # the readout contract's 2560; the toy honours it
HEAD = 16            # toy head dim, so q/k/v/o stay small


# --------------------------------------------------------------------------- #
# a toy VLM with the real attribute paths
# --------------------------------------------------------------------------- #
class ToyAttn(nn.Module):
    def __init__(self, d: int = D, head: int = HEAD) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d, head, bias=False)
        self.k_proj = nn.Linear(d, head, bias=False)
        self.v_proj = nn.Linear(d, head, bias=False)
        self.o_proj = nn.Linear(head, d, bias=False)


class ToyBlock(nn.Module):
    def __init__(self, d: int = D) -> None:
        super().__init__()
        self.self_attn = ToyAttn(d)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        a = self.self_attn
        q, k, v = a.q_proj(h), a.k_proj(h), a.v_proj(h)
        s = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]) + mask
        return h + a.o_proj(torch.softmax(s, dim=-1) @ v)


class ToyLM(nn.Module):
    def __init__(self, n_layers: int = 4, d: int = D) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(64, d)
        self.layers = nn.ModuleList(ToyBlock(d) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d)


class ToyVisualBlock(nn.Module):
    """Carries the SAME projection names as a language block, on purpose.

    peft matches ``target_modules`` by suffix; a bare ``"q_proj"`` would adapt
    this too.  ``test_bare_suffix_would_reach_the_vision_tower`` is the proof
    that the arm's fully-qualified names do not.
    """

    def __init__(self, d: int = 8) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)


class ToyVisual(nn.Module):
    def __init__(self, n: int = 2) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(ToyVisualBlock() for _ in range(n))


class ToyInner(nn.Module):
    def __init__(self, n_layers: int = 4) -> None:
        super().__init__()
        self.language_model = ToyLM(n_layers)
        self.visual = ToyVisual()


class ToyVLM(nn.Module):
    """``model.language_model`` / ``model.visual``, the paths hiddens.py resolves.

    ``language_model`` is exposed on the top level too: the real
    ``Qwen3VLForConditionalGeneration`` does (verified on checkpoint-4976), and
    peft's wrapper forwards attribute access to the *base* model, so without it
    ``resolve_language_model`` would find the tower before the adapter and not
    after -- which is precisely the difference this suite has to see.

    Attention is causal **and** honours ``attention_mask``, like the real
    decoder.  Without both, right padding would move earlier positions and the
    micro-batch invariance this arm relies on would be a property of the toy
    rather than of the contract.
    """

    def __init__(self, n_layers: int = 4) -> None:
        super().__init__()
        self.model = ToyInner(n_layers)

    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    def forward(self, *, input_ids, attention_mask=None, pixel_values=None,
                image_grid_thw=None):
        lm = self.model.language_model
        h = lm.embed_tokens(input_ids)
        if pixel_values is not None:
            # the image contributes a constant per-sample offset; enough that a
            # missing image would change z, which is what the parity checks care
            # about
            h = h + pixel_values.reshape(h.shape[0], 1, -1)[:, :, :h.shape[-1]]
        t = h.shape[1]
        mask = torch.zeros(h.shape[0], t, t, dtype=h.dtype)
        mask.masked_fill_(torch.triu(torch.ones(t, t, dtype=torch.bool), 1), -1e9)
        if attention_mask is not None:
            mask = mask.masked_fill(attention_mask[:, None, :] == 0, -1e9)
        for blk in lm.layers:
            h = blk(h, mask)
        return h


class ToyTokenizer:
    pad_token_id = 0
    eos_token_id = 1


class ToyProcessor:
    def __init__(self) -> None:
        self.tokenizer = ToyTokenizer()

    def image_processor(self, *, images, do_resize=False, return_tensors="pt"):
        n = len(images)
        return {"pixel_values": torch.stack([torch.full((D,), float(im))
                                             for im in images]),
                "image_grid_thw": torch.ones((n, 3), dtype=torch.long)}


@pytest.fixture()
def toy():
    torch.manual_seed(0)
    return ToyVLM()


def _items(builder, n: int = 3, *, seed: int = 0) -> list[L.SpanItem]:
    rng = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        w = [11, 20 + i, 12]                      # <where> ... </where>
        c = [13, 30 + i, 31 + i, 14]              # <color> ... </color>
        plan = _plan(builder, w, c, sid=f"s{i}")
        n_p = 2 + int(torch.randint(0, 3, (1,), generator=rng))
        out.append(L.SpanItem(sample_id=f"s{i}", image=0.1 * (i + 1),
                              prompt_ids=list(range(2, 2 + n_p)), plan=plan))
    return out


def _plan(builder, w, c, *, sid="s0"):
    return builder.plan_for(sample_id=sid, where_ids=w, color_ids=c,
                            source="generated", control_tag="none")


@pytest.fixture()
def builder(tokenizer):
    return WhatReadoutBuilder(tokenizer, WhatReadoutSpec(kind="color_span_pool"))


# --------------------------------------------------------------------------- #
# A_lora: the adapter surface is the declared one
# --------------------------------------------------------------------------- #
def test_targets_are_the_last_n_language_blocks_only(toy):
    names, layers = L.lora_target_names(toy, last_n=2)
    assert layers == [2, 3]
    assert names == [f"model.language_model.layers.{i}.self_attn.{s}"
                     for i in (2, 3) for s in L.LORA_TARGET_SUFFIXES]
    assert not any("visual" in n for n in names)


def test_last_n_out_of_range_is_refused(toy):
    with pytest.raises(ValueError, match="--lora-last-n"):
        L.lora_target_names(toy, last_n=99)
    with pytest.raises(ValueError, match="--lora-last-n"):
        L.lora_target_names(toy, last_n=0)


def test_attach_adapts_exactly_the_declared_modules(toy):
    pm, facts = L.attach_lora(toy, last_n=2, r=4, alpha=4)
    d = facts.to_dict()
    assert d["n_target_modules"] == 8 == len(d["target_modules"])
    assert d["layer_indices"] == [2, 3]
    assert d["language_blocks_frozen"] == 2
    assert d["scaling"] == 1.0
    from peft.tuners.lora import LoraLayer

    adapted = {n.split("base_model.model.", 1)[-1]
               for n, m in pm.named_modules() if isinstance(m, LoraLayer)}
    assert adapted == set(d["target_modules"])


def test_attach_carries_the_two_assertions_onto_the_artefact(toy):
    """A check whose record never reaches run_setup.json is a check nobody can audit."""
    _, facts = L.attach_lora(toy, last_n=2, r=4, alpha=4)
    a = facts.to_dict()["assertions"]
    assert set(a) == {"A_lora", "A_freeze"}
    assert a["A_lora"] == {"n_adapted": 8, "matches_declared": True}
    assert a["A_freeze"]["outside_declared"] == 0
    assert a["A_freeze"]["n_trainable_tensors"] == 16      # 8 modules x (A, B)


def test_declared_surface_assertion_catches_a_missing_module(toy):
    pm, facts = L.attach_lora(toy, last_n=2, r=4, alpha=4)
    with pytest.raises(AssertionError, match="not the declared one"):
        L.assert_declared_surface(pm, list(facts.target_modules)[:-1])


def test_bare_suffix_would_reach_the_vision_tower(toy):
    """The hazard the fully-qualified names exist to avoid, measured."""
    from peft import LoraConfig, get_peft_model
    from peft.tuners.lora import LoraLayer

    declared, _ = L.lora_target_names(toy, last_n=2)
    bare = get_peft_model(toy, LoraConfig(r=4, lora_alpha=4, bias="none",
                                          target_modules=["q_proj"],
                                          task_type=None))
    adapted = {n.split("base_model.model.", 1)[-1]
               for n, m in bare.named_modules() if isinstance(m, LoraLayer)}
    assert any("visual" in n for n in adapted)          # the bare suffix does
    assert not any("visual" in n for n in declared)     # this arm's names do not
    with pytest.raises(AssertionError, match="vision tower"):
        L.assert_declared_surface(bare, declared)


# --------------------------------------------------------------------------- #
# A_freeze: the frozen area
# --------------------------------------------------------------------------- #
def test_only_lora_parameters_of_declared_modules_train(toy):
    pm, facts = L.attach_lora(toy, last_n=2, r=4, alpha=4)
    train = [n for n, p in pm.named_parameters() if p.requires_grad]
    assert train and all("lora_" in n for n in train)
    for n in train:
        owner = n.split("base_model.model.", 1)[-1].split(".lora_", 1)[0]
        assert owner in set(facts.target_modules)
    # the frozen area, positively: nothing in the vision tower, the embedding
    # table or the first two blocks may move
    frozen = [n for n, p in pm.named_parameters() if not p.requires_grad]
    assert any("visual" in n for n in frozen)
    assert any("embed_tokens" in n for n in frozen)
    assert any("layers.0." in n for n in frozen)
    assert any("layers.1." in n for n in frozen)
    n_lora = sum(p.numel() for n, p in pm.named_parameters() if p.requires_grad)
    assert facts.n_lora_params == n_lora == facts.n_trainable_vlm_params


def test_frozen_area_assertion_catches_an_unfrozen_base_weight(toy):
    pm, facts = L.attach_lora(toy, last_n=2, r=4, alpha=4)
    # unfreeze the embedding table -- the exact failure the assertion names
    for n, p in pm.named_parameters():
        if "embed_tokens" in n:
            p.requires_grad_(True)
            break
    with pytest.raises(AssertionError, match="outside the declared LoRA surface"):
        L.assert_frozen_area(pm, facts.target_modules)


def test_frozen_area_assertion_catches_a_wired_but_dead_adapter(toy):
    pm, facts = L.attach_lora(toy, last_n=2, r=4, alpha=4)
    for p in pm.parameters():
        p.requires_grad_(False)
    with pytest.raises(AssertionError, match="wired but nothing would ever be"):
        L.assert_frozen_area(pm, facts.target_modules)


def test_lora_b_is_zero_initialised(toy):
    pm, _ = L.attach_lora(toy, last_n=2, r=4, alpha=4)
    b = [p for n, p in pm.named_parameters() if "lora_B" in n]
    a = [p for n, p in pm.named_parameters() if "lora_A" in n]
    assert b and all(float(p.detach().abs().sum()) == 0.0 for p in b)
    assert a and any(float(p.detach().abs().sum()) > 0.0 for p in a)


# --------------------------------------------------------------------------- #
# A_reply: the reply is the cached one, token for token
# --------------------------------------------------------------------------- #
def _cache_row(plan, sid="s0"):
    return {"sample_id": sid, "reply_token_ids": list(plan.token_ids),
            "readout_index": [plan.start, plan.end]}


def test_plan_from_cache_row_round_trips(builder):
    ref = _plan(builder, [11, 21, 12], [13, 31, 32, 14])
    row = _cache_row(ref)
    got = L.plan_from_cache_row(builder, row)
    assert got.token_ids == ref.token_ids
    assert (got.start, got.end, got.pool) == (ref.start, ref.end, True)


def test_a_reply_catches_a_span_rebuilt_one_token_off(builder):
    """The plan the forward would be fed vs. the reply the cache recorded."""
    ref = _plan(builder, [11, 21, 12], [13, 31, 32, 14])
    row = _cache_row(ref)
    row["reply_token_ids"] = list(ref.token_ids[:-1]) + [999]
    with pytest.raises(AssertionError, match="differs from the cached"):
        L.assert_reply_parity(ref, row)


def test_a_reply_catches_a_moved_pooling_slice(builder):
    ref = _plan(builder, [11, 21, 12], [13, 31, 32, 14])
    row = _cache_row(ref)
    row["readout_index"] = [ref.start - 1, ref.end]
    with pytest.raises(AssertionError, match="rebuilt pooling slice"):
        L.assert_reply_parity(ref, row)


def test_a_reply_catches_an_end_past_the_reply(builder):
    """Reached through the real entry point, not by hand-editing a plan."""
    ref = _plan(builder, [11, 21, 12], [13, 31, 32, 14])
    row = _cache_row(ref)
    row["readout_index"] = [ref.start, ref.end + 5]
    with pytest.raises(AssertionError, match="rebuilt pooling slice"):
        L.plan_from_cache_row(builder, row)


def test_a_scalar_readout_index_is_refused(builder):
    """``color_span_pool`` is pooled; a bare ``int`` index is the wrong kind."""
    ref = _plan(builder, [11, 21, 12], [13, 31, 32, 14])
    row = _cache_row(ref)
    row["readout_index"] = int(ref.start)
    with pytest.raises(AssertionError, match=r"\[start, end\]"):
        L.plan_from_cache_row(builder, row)


# --------------------------------------------------------------------------- #
# the encoder + A_step0 + A_cache
# --------------------------------------------------------------------------- #
def _encoder(model, *, micro=2):
    return L.SpanPoolEncoder(model, ToyProcessor(), device="cpu",
                             final_norm=True, micro_batch=micro)


def test_z_is_the_mean_over_the_colour_span(toy, builder):
    enc = _encoder(toy, micro=8)
    items = _items(builder, 3)
    z = enc.z(items, grad=False)
    assert z.shape == (3, D) and z.dtype == torch.float32
    hs = enc.h_reply(items)
    for i, (h, it) in enumerate(zip(hs, items)):
        assert h.shape[0] == len(it.plan.token_ids)
        assert torch.equal(z[i], h[it.plan.start:it.plan.end].mean(0))


def test_micro_batching_does_not_change_the_slice_bookkeeping(toy, builder):
    """Padding shape may move fp arithmetic; the *rows* read must not move."""
    enc_a, enc_b = _encoder(toy, micro=1), _encoder(toy, micro=8)
    items = _items(builder, 4)
    za, zb = enc_a.z(items, grad=False), enc_b.z(items, grad=False)
    assert torch.allclose(za, zb, atol=1e-5)


def test_step0_identity_holds_and_is_exact(toy, builder):
    pm, _ = L.attach_lora(toy, last_n=2, r=4, alpha=4)
    enc = _encoder(pm)
    rec = L.assert_lora_step0_identity(enc, _items(builder, 3))
    assert rec["max_abs_delta"] == 0.0 and rec["exact"] is True


def test_step0_identity_raises_once_the_adapter_has_moved(toy, builder):
    pm, _ = L.attach_lora(toy, last_n=2, r=4, alpha=4)
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "lora_B" in n:
                p.add_(torch.randn_like(p))
    enc = _encoder(pm)
    with pytest.raises(AssertionError, match="not the identity at step 0"):
        L.assert_lora_step0_identity(enc, _items(builder, 2))


def test_cache_parity_passes_against_its_own_adapter_off_z(toy, builder):
    pm, _ = L.attach_lora(toy, last_n=2, r=4, alpha=4)
    enc = _encoder(pm)
    items = _items(builder, 3)
    with torch.no_grad(), L.adapter_disabled(pm):
        z_ref = enc.z(items, grad=False)
    rec = L.assert_cache_parity(enc, items, z_ref)
    assert rec["passed"] and rec["max_abs_delta_vs_cache"] <= rec["threshold"]


def test_cache_parity_raises_on_a_prompt_that_is_not_the_cached_one(toy, builder):
    pm, _ = L.attach_lora(toy, last_n=2, r=4, alpha=4)
    enc = _encoder(pm)
    items = _items(builder, 3)
    with torch.no_grad(), L.adapter_disabled(pm):
        z_ref = enc.z(items, grad=False)
    for it in items:                      # a different image = a different prompt
        it.image = it.image + 5.0
    with pytest.raises(AssertionError, match="cache parity FAILED"):
        L.assert_cache_parity(enc, items, z_ref)


def test_gradient_reaches_the_adapters_and_nothing_else(toy, builder):
    pm, facts = L.attach_lora(toy, last_n=2, r=4, alpha=4)
    with torch.no_grad():                 # B=0 gives a zero gradient to lora_A
        for n, p in pm.named_parameters():
            if "lora_B" in n:
                p.add_(0.01 * torch.randn_like(p))
    enc = _encoder(pm)
    z = enc.z(_items(builder, 3), grad=True)
    assert z.requires_grad
    z.sum().backward()
    lora = L.trainable_lora_parameters(pm)
    assert lora and any(p.grad is not None and float(p.grad.abs().sum()) > 0
                        for p in lora)
    assert L.lora_grad_norm(lora) > 0.0
    frozen = [p for n, p in pm.named_parameters() if not p.requires_grad]
    assert all(p.grad is None for p in frozen)


def test_no_grad_path_builds_no_graph(toy, builder):
    pm, _ = L.attach_lora(toy, last_n=2, r=4, alpha=4)
    enc = _encoder(pm)
    z = enc.z(_items(builder, 2), grad=False)
    assert not z.requires_grad


def test_z_drift_is_one_when_nothing_moved(toy, builder):
    pm, _ = L.attach_lora(toy, last_n=2, r=4, alpha=4)
    enc = _encoder(pm)
    items = _items(builder, 3)
    z_on = enc.z(items, grad=False)
    with torch.no_grad(), L.adapter_disabled(pm):
        z_off = enc.z(items, grad=False)
    assert L.z_cos_drift(z_on, z_off) == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# the optimiser groups
# --------------------------------------------------------------------------- #
def test_optimizer_keeps_the_carrier_group_order_and_appends_lora():
    from q3vl.whatb.arms import carrier as A

    cfg = A.CarrierConfig(readout="color_span_pool", n_gauss=32, loss_level=1,
                          epochs=4, train_n=93934)
    model = A.CarrierModel(cfg)
    lora = [torch.zeros(2, 2, requires_grad=True)]
    opt = L.build_optimizer(model, cfg, lora_params=lora, lora_lr=1e-4)
    assert [g["name"] for g in opt.param_groups] == ["generator", "pi", "lora"]
    assert opt.param_groups[0]["lr"] == cfg.base_lr
    assert opt.param_groups[1]["lr"] == cfg.base_lr * cfg.proj_lr_scale
    assert opt.param_groups[2]["lr"] == 1e-4
    # the control row has no third group, so its step columns are unchanged
    ctl = L.build_optimizer(model, cfg, lora_params=(), lora_lr=1e-4)
    assert [g["name"] for g in ctl.param_groups] == ["generator", "pi"]
    ref = A.build_optimizer(model, cfg)
    assert [g["lr"] for g in ctl.param_groups] == [g["lr"] for g in ref.param_groups]


# --------------------------------------------------------------------------- #
# --carrier affonly_frozen: the JetLUT-p1 carrier
#
# Same four questions the LoRA side asks, asked of the carrier: what got
# mounted, what can move, does step 0 reproduce stage 1, and does the frozen
# claim hold.  The stage-1 checkpoint is built here from the real
# ``AffineOnlyHead`` under the arm's own flags (``--share geo_opacity
# --global-affine affine --num-gauss 64 --mu-init grid``) rather than mocked,
# because what is being tested is precisely that the two constructions agree.
# --------------------------------------------------------------------------- #
AFF_N = 64


@pytest.fixture
def stage1():
    """``(blob, carrier_cfg)`` -- a perturbed AFFONLY head saved the way
    ``run_affonly_arm.py:621`` saves one (``{"state_dict": ..., "config": ...}``)."""
    from q3vl.whatb.arms import affonly as AF
    from q3vl.whatb.arms import carrier as A

    torch.manual_seed(20260810)
    acfg = AF.AffineOnlyConfig(
        share="geo_opacity", global_affine="affine", n_gauss=AFF_N, cond_dim=64,
        gen_width=128, mu_init=L.AFFONLY_MU_INIT, shared_lr_scale=0.0,
        clamp="two", loss_level=1, zero_init_heads=True)
    head = AF.AffineOnlyHead(acfg)
    with torch.no_grad():                      # move off the zero-init fixed point
        for p in head.parameters():
            p.add_(0.01 * torch.randn_like(p))
    blob = {"state_dict": head.state_dict(), "step": 5872,
            "headline_normal_only": 4.20, "config": head.config}
    ccfg = A.CarrierConfig(readout="color_span_pool", cond_dim=64, n_gauss=AFF_N,
                           gen_width=128, loss_level=1, clamp="two",
                           epochs=L.T2_EPOCHS, train_n=93934)
    return blob, ccfg


def _loaded(stage1):
    blob, ccfg = stage1
    model = L.build_model(ccfg, carrier="affonly_frozen")
    L.load_stage1(model, blob, carrier="affonly_frozen", path="<fixture>")
    L.freeze_shared_geometry(model)
    return model, blob, ccfg


# -- 1. the mounted surface --------------------------------------------------
def test_affonly_frozen_mounts_the_affine_only_generator(stage1):
    _, ccfg = stage1
    model = L.build_model(ccfg, carrier="affonly_frozen")
    gen = model.generator
    assert gen.mode == "affine_only"
    assert gen.m_residual is True and gen.zero_init_last is True
    # 12N + 12 = 780 at N = 64, and the two heads are the only generated ones
    assert gen.theta_dim == 12 * AFF_N + 12 == 780
    assert gen.head_mu is None and gen.head_cov is None and gen.head_opacity is None
    sg = L.shared_geometry_of(model)
    assert sg is not None and sg.n_gauss == AFF_N
    assert sg.init_sigma == L.AFFONLY_SIGMA_INIT == 0.15
    assert sg.init_opacity_logit == L.AFFONLY_OPACITY_LOGIT_INIT == 4.0
    assert sum(p.numel() for p in sg.parameters_list()) == 10 * AFF_N == 640
    # the default 档 is untouched: Full Generation, no shared table
    assert L.shared_geometry_of(L.build_model(ccfg, carrier="carrier")) is None
    assert L.build_model(ccfg, carrier="carrier").generator.theta_dim == 22 * AFF_N + 12


def test_stage1_load_remaps_proj_to_pi_with_nothing_left_over(stage1):
    blob, ccfg = stage1
    model = L.build_model(ccfg, carrier="affonly_frozen")
    facts = L.load_stage1(model, blob, carrier="affonly_frozen", path="<fixture>")
    assert facts["n_tensors_loaded"] == len(blob["state_dict"])
    assert facts["key_remap"] == [["proj.", "pi."]]
    # ``run_affonly_arm`` writes "state_dict", this runner writes "model";
    # both are accepted, anything else is refused rather than warm-started from
    assert set(L.stage1_tensors(blob)) == set(blob["state_dict"])
    assert set(L.stage1_tensors({"model": blob["state_dict"]})) == set(blob["state_dict"])
    with pytest.raises(AssertionError, match="state_dict"):
        L.stage1_tensors({"step": 3})
    # without the remap the same file is a total miss on both sides
    with pytest.raises(AssertionError, match="does not fit"):
        L.load_stage1(L.build_model(ccfg, carrier="affonly_frozen"), blob,
                      carrier="carrier", path="<fixture>")


# -- 2. the frozen area ------------------------------------------------------
def test_only_the_shared_geometry_is_frozen_and_adam_never_sees_it(stage1):
    model, _, ccfg = _loaded(stage1)
    sg = L.shared_geometry_of(model)
    frozen = [p for p in model.parameters() if not p.requires_grad]
    assert {id(p) for p in frozen} == {id(p) for p in sg.parameters_list()}
    assert sum(p.numel() for p in frozen) == 10 * AFF_N == 640
    opt = L.build_optimizer(model, ccfg, lora_params=(), lora_lr=1e-4)
    assert [g["name"] for g in opt.param_groups] == ["generator", "pi"]
    in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
    assert all(id(p) not in in_opt for p in sg.parameters_list())
    assert all(p.requires_grad for g in opt.param_groups for p in g["params"])
    # everything else in the generator is still trained
    facts = L.carrier_facts(model, "affonly_frozen")
    assert facts["n_params_generator"] - facts["n_params_generator_trainable"] == 640


def test_a_gradient_step_moves_the_heads_and_not_the_geometry(stage1):
    model, _, ccfg = _loaded(stage1)
    sg = L.shared_geometry_of(model)
    before = {n: getattr(sg, n).detach().clone() for n in L.SHARED_GEOMETRY_TENSORS}
    head_before = model.generator.head_color[-1].weight.detach().clone()
    opt = L.build_optimizer(model, ccfg, lora_params=(), lora_lr=1e-4)
    z = torch.randn(4, L.Z_DIM)
    y = model.transform_grid(z, model.query_grid9)
    opt.zero_grad(set_to_none=True)
    y.pow(2).mean().backward()
    opt.step()
    for n, v in before.items():
        assert torch.equal(getattr(sg, n).detach(), v), n
        assert getattr(sg, n).grad is None
    assert not torch.equal(model.generator.head_color[-1].weight.detach(), head_before)


# -- 3. step 0 reproduces stage 1, bitwise -----------------------------------
def test_step0_reproduces_the_stage1_affonly_head_bitwise(stage1):
    model, blob, _ = _loaded(stage1)
    z = torch.randn(8, L.Z_DIM)
    rec = L.assert_stage1_forward_identity(model, blob, z)
    assert rec["exact"] is True and rec["max_abs_delta"] == 0.0
    assert rec["n"] == 8 and rec["grid_n"] == 17 and rec["dtype"] == "torch.float32"


def test_step0_identity_raises_once_the_generator_has_moved(stage1):
    model, blob, _ = _loaded(stage1)
    with torch.no_grad():
        model.generator.head_global[-1].bias.add_(1e-3)
    with pytest.raises(AssertionError, match="A_stage1 FAILED"):
        L.assert_stage1_forward_identity(model, blob, torch.randn(4, L.Z_DIM))


def test_step0_identity_refuses_a_stage1_from_another_construction(stage1):
    model, blob, _ = _loaded(stage1)
    z = torch.randn(4, L.Z_DIM)
    shared = dict(blob, config=dict(blob["config"], share="geo"))
    with pytest.raises(AssertionError, match="share 'geo'"):
        L.assert_stage1_forward_identity(model, shared, z)
    narrow = dict(blob, config=dict(blob["config"], gen_width=64))
    with pytest.raises(AssertionError, match="gen_width"):
        L.assert_stage1_forward_identity(model, narrow, z)
    with pytest.raises(AssertionError, match="no 'config' block"):
        L.assert_stage1_forward_identity(model, {"state_dict": blob["state_dict"]}, z)


# -- 4. A_geom_frozen, both directions ---------------------------------------
def test_a_geom_frozen_passes_on_the_wired_model(stage1):
    model, blob, _ = _loaded(stage1)
    rec = L.assert_geometry_frozen(model, blob)
    assert rec["all_frozen"] and rec["all_equal_stage1"] and rec["bitwise"]
    assert rec["n_tensors"] == 4 and rec["n_params"] == 640
    assert set(rec["per_tensor"]) == set(L.SHARED_GEOMETRY_TENSORS)
    assert all(t["max_abs_dev"] == 0.0 for t in rec["per_tensor"].values())


def test_a_geom_frozen_catches_a_table_that_can_still_move(stage1):
    model, blob, _ = _loaded(stage1)
    L.shared_geometry_of(model).mu.requires_grad_(True)
    with pytest.raises(AssertionError, match=r"\['mu'\] still require grad"):
        L.assert_geometry_frozen(model, blob)


def test_a_geom_frozen_catches_a_table_that_is_not_stage_ones(stage1):
    model, blob, _ = _loaded(stage1)
    with torch.no_grad():                     # one ulp is enough: torch.equal
        L.shared_geometry_of(model).chol_off.add_(1e-7)
    with pytest.raises(AssertionError, match=r"\['chol_off'\] differ"):
        L.assert_geometry_frozen(model, blob)


def test_a_geom_frozen_needs_a_checkpoint_that_has_the_tables(stage1):
    model, blob, _ = _loaded(stage1)
    stripped = {k: v for k, v in blob["state_dict"].items()
                if "shared_geometry" not in k}
    with pytest.raises(AssertionError, match="no 'generator.shared_geometry"):
        L.assert_geometry_frozen(model, {"state_dict": stripped})
    # ... and a Full-Generation model has no table to assert about at all
    _, ccfg = stage1
    with pytest.raises(AssertionError, match="no SharedGeometry"):
        L.freeze_shared_geometry(L.build_model(ccfg, carrier="carrier"))


# -- the runner's flag surface -----------------------------------------------
def test_runner_resolves_the_stage1_default_per_carrier():
    from q3vl.whatb.scripts import run_lora_span_arm as RL

    base = ["--zcache-root", "/x", "--readout", "color_span_pool"]
    ap = RL.build_parser()
    a = ap.parse_args(base)
    assert a.carrier == "carrier" == L.CARRIER_DEFAULT
    assert RL.resolve_stage1(a) == str(Path(L.STAGE1_RUN_DIR) / "best.pt")
    b = ap.parse_args(base + ["--carrier", "affonly_frozen"])
    assert RL.resolve_stage1(b) == str(Path(L.AFFONLY_STAGE1_RUN_DIR) / "best.pt")
    c = ap.parse_args(base + ["--carrier", "affonly_frozen", "--stage1", "/tmp/x.pt"])
    assert RL.resolve_stage1(c) == "/tmp/x.pt"
    with pytest.raises(SystemExit):
        ap.parse_args(base + ["--carrier", "jetlut"])


def test_t2_horizon_is_four_epochs():
    from q3vl.whatb.arms import carrier as A

    # 93,934 is the v20260804 normal-only train n the stage-1 board was
    # measured on; the runner asserts the same number against the live index.
    cfg = A.CarrierConfig(readout="color_span_pool", n_gauss=32, loss_level=1,
                          epochs=L.T2_EPOCHS, train_n=93934)
    assert cfg.steps_per_epoch == 2936
    assert cfg.total_steps == 11744 == 2936 * L.T2_EPOCHS


# --------------------------------------------------------------------------- #
# the runner does not modify what it borrows
# --------------------------------------------------------------------------- #
def test_the_arm_imports_the_live_modules_and_declares_it():
    import inspect

    from q3vl.whatb.scripts import run_lora_span_arm as RL

    src = inspect.getsource(RL)
    assert "from q3vl.whatb.arms import carrier as A" in src
    assert "from q3vl.whatb.scripts import run_carrier_arm as R" in src
    # the runner reuses the carrier board end to end: no criterion of its own
    assert "def build_board" not in src and "def image_delta_e00" not in src


def test_runner_refuses_a_readout_other_than_color_span_pool():
    from q3vl.whatb.scripts import run_lora_span_arm as RL

    with pytest.raises(SystemExit, match="color_span_pool"):
        RL.main(["--zcache-root", "/nonexistent", "--readout", "seg_color"])


def test_smoke_defaults_shrink_every_eval_fixture():
    from q3vl.whatb.scripts import run_lora_span_arm as RL

    args = RL.build_parser().parse_args(
        ["--zcache-root", "/x", "--readout", "color_span_pool", "--smoke", "20"])
    RL.apply_smoke(args)
    assert args.max_steps == 20 and args.eval_every == 10
    assert args.select_samples == 8 and args.eval_n == 64
    assert args.live_z_select >= args.select_samples
