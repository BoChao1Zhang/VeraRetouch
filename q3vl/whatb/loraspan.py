"""EPR-033: the LoRA-unfrozen conditional path behind the ``color_span_pool`` readout.

    z_b = mean_{t in colour span} norm(h[-1])[t]        (as in readout.py)

with the difference that ``h`` is now produced by a language tower whose **last
eight blocks carry LoRA on q/k/v/o**, and that the forward happens *inside* the
training step so the task loss reaches those adapters.  Everything else -- the
vision tower, the embedding table, the other twenty-eight blocks, ``pi``, the
generator, the carrier, the loss, the board -- is byte-identical to the arm that
produced ``whatb_CARRIER_ro_color_span_pool`` (headline 4.0656).

Why this module and not a flag on ``arms/carrier.py``
----------------------------------------------------
``q3vl/whatb/arms/carrier.py``, ``q3vl/whatb/scripts/run_carrier_arm.py`` and
``q3vl/whatb/glut.py`` are imported **right now** by three live jobs (PIDs
1884539 / 1884689 / 1884898 on 2026-08-24), and the campaign's operating rule is
that a process's source may not change once it is running.  This module and
:mod:`q3vl.whatb.scripts.run_lora_span_arm` therefore *import* those three and
modify none of them.

What is asserted at start-up, and why each one exists
-----------------------------------------------------
``A_lora`` (declared surface)
    The set of modules that actually received an adapter is compared, as a set,
    with the set this module declared.  peft matches ``target_modules`` by
    suffix; a suffix that matches one module too many is invisible in every
    metric.
``A_freeze`` (frozen area)
    Every parameter with ``requires_grad`` must be a LoRA parameter of a
    declared module.  Nothing else in the VLM may be trainable -- the arm's
    claim is "LoRA on eight blocks", not "the language tower drifted".
``A_reply`` (bitwise reply parity)
    The teacher-forced reply this arm feeds the forward is rebuilt through
    :func:`q3vl.whatb.readout.build_reply` and compared **token by token** with
    the ``reply_token_ids`` the z cache recorded.  Integers, so this one is
    exact.
``A_step0`` (LoRA-B zero init)
    ``B = 0`` means the adapter is the zero map at step 0, so z with the adapter
    enabled must equal z with it disabled **bitwise**, in the same forward
    shape.  This is the exact form of "step 0 reproduces stage 1"; see the note
    on bf16 below for why the *cache* comparison cannot be bitwise.
``A_cache`` (cache parity, tolerance measured not guessed)
    z recomputed here with the adapter disabled is compared with the cached z.
    The cache was written in **bf16** and its own build report records a
    batch-shape control of ``max|dz| = 5.75`` against ``max|z| = 67.5``
    (``build_report.json``, ``causality_gate``): re-encoding the *same* tokens
    in a different padding shape already moves z by that much.  A bitwise
    comparison here would therefore fail on arithmetic, not on wiring, so the
    threshold is taken from a batch-shape control **measured in this process**
    through :func:`q3vl.whatb.scripts.build_zcache.causality_gate_verdict` --
    the same rule the cache builder used on itself.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch
from torch import Tensor

from q3vl.whatb.generator import CGLUTGenerator
from q3vl.whatb.glut import EPS as GLUT_EPS
from q3vl.whatb.readout import (
    ReplyPlan,
    WhatReadoutBuilder,
    readout_vector,
    SEGMENT_HIDDEN_FINAL_NORM,
    SEGMENT_HIDDEN_LAYER,
)
from q3vl.whereb.hiddens import LastLayerHook, resolve_language_model, resolve_visual

__all__ = [
    "ARM",
    "ARM_NAME",
    "CARRIER_CHOICES",
    "CARRIER_DEFAULT",
    "AFFONLY_N_GAUSS",
    "AFFONLY_SIGMA_INIT",
    "AFFONLY_OPACITY_LOGIT_INIT",
    "AFFONLY_MU_INIT",
    "AFFONLY_GENERATOR_FORM",
    "AFFONLY_STAGE1_RUN_DIR",
    "SHARED_GEOMETRY_TENSORS",
    "AFFONLY_KEY_REMAP",
    "build_model",
    "carrier_facts",
    "shared_geometry_of",
    "stage1_tensors",
    "remap_stage1_keys",
    "load_stage1",
    "freeze_shared_geometry",
    "affonly_config_from_blob",
    "assert_geometry_frozen",
    "assert_stage1_forward_identity",
    "LORA_R",
    "LORA_ALPHA",
    "LORA_DROPOUT",
    "LORA_LAST_N_BLOCKS",
    "LORA_TARGET_SUFFIXES",
    "LORA_LR",
    "STAGE1_RUN_DIR",
    "T2_EPOCHS",
    "LoraFacts",
    "lora_target_names",
    "attach_lora",
    "adapter_disabled",
    "trainable_lora_parameters",
    "assert_declared_surface",
    "assert_frozen_area",
    "SpanItem",
    "SpanPoolEncoder",
    "plan_from_cache_row",
    "assert_reply_parity",
    "assert_lora_step0_identity",
    "assert_cache_parity",
    "lora_grad_norm",
    "z_cos_drift",
    "build_optimizer",
]

# --------------------------------------------------------------------------- #
# 0. identity + the frozen numbers
# --------------------------------------------------------------------------- #
ARM = "EPR-033"
ARM_NAME = "LORASPAN"

#: LoRA rank / scaling / dropout.  ``alpha == r`` so the adapter scale is 1.0.
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
#: how many *language* decoder blocks, counted from the end, carry an adapter
LORA_LAST_N_BLOCKS = 8
#: the four projections inside each of those blocks' self-attention
LORA_TARGET_SUFFIXES: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
#: the attribute the four projections hang off inside a decoder block
#: (``Qwen3VLTextDecoderLayer.self_attn``, verified on the loaded checkpoint)
LORA_ATTN_ATTR = "self_attn"
#: the adapters' learning rate -- 0.1x the generator's 1e-3 base
LORA_LR = 1e-4

#: stage 1 = the frozen board this arm continues from (headline 4.0656, n=567)
STAGE1_RUN_DIR = "/home/bc/data/runs/what_b/whatb_CARRIER_ro_color_span_pool"
#: pre-registered stage-2 horizon: 4 epochs x 2936 steps/epoch = 11,744 steps
T2_EPOCHS = 4

#: cos(z_lora, z_frozen) below this at a diagnostic point is *recorded*, never
#: acted on; the guard that can kill the run is the degeneracy check in
#: ``arms/carrier.quick_eval``.
Z_DIM = 2560


# --------------------------------------------------------------------------- #
# 0b. the two carriers this arm can run the same ablation on
# --------------------------------------------------------------------------- #
#: ``carrier``          Full Generation (22N+12), the EPR-024/033 stage-1 form;
#: ``affonly_frozen``   the JetLUT-p1 image-domain form: AFFONLY's shared
#:                      analytic geometry loaded from a stage-1 AFFONLY
#:                      checkpoint and **frozen**, only ``{M_i, b_i, G, g}``
#:                      (12N+12) generated from the condition.
CARRIER_CHOICES: tuple[str, ...] = ("carrier", "affonly_frozen")
CARRIER_DEFAULT = "carrier"

#: the JetLUT-p1 geometry: N = 64 -> ``uniform_grid_positions(64)`` is the 4^3
#: cube-centre lattice, sigma = 0.15 isotropic, opacity logit +4.0 constant.
#: 12N + 12 = 780 generated dimensions at N = 64.
AFFONLY_N_GAUSS = 64
AFFONLY_SIGMA_INIT = 0.15
AFFONLY_OPACITY_LOGIT_INIT = 4.0
AFFONLY_MU_INIT = "grid"

#: the generator form of ``q3vl/whatb/arms/affonly.py``'s ``AffineOnlyHead``
#: under ``--share geo_opacity --global-affine affine --zero-init-heads``:
#: ``mode="affine_only"`` (two heads: local 12N + global 12),
#: ``m_residual=True`` (``M_i = I + dM_i``), ``zero_init_last=True``.
#: Read off ``arms/affonly.py:AffineOnlyHead.__init__`` -- that file is on a
#: live job's import chain, so it is imported and never edited.
AFFONLY_GENERATOR_FORM: dict[str, Any] = {
    "mode": "affine_only",
    "m_residual": True,
    "zero_init_last": True,
    "shared_sigma": AFFONLY_SIGMA_INIT,
    "shared_opacity_logit": AFFONLY_OPACITY_LOGIT_INIT,
}

#: stage 1 for the ``affonly_frozen`` carrier on the span-pool readout
AFFONLY_STAGE1_RUN_DIR = "/home/bc/data/runs/what_b/whatb_AFFONLY_jetgeo_ro_spanpool"

#: the four tensors ``A_geom_frozen`` binds on (``generator.SharedGeometry``)
SHARED_GEOMETRY_TENSORS: tuple[str, ...] = ("mu", "chol_diag", "chol_off",
                                            "opacity_logit")

#: ``run_affonly_arm`` saves ``AffineOnlyHead.state_dict()``, whose projection is
#: named ``proj``; ``CarrierModel`` names the same module ``pi``.  Every other
#: key (``generator.*``, including ``generator.shared_geometry.*``) is identical,
#: and ``GlutCarrier`` has no parameters at all -- measured on a constructed
#: checkpoint, both sides 24 tensors, ``strict=False`` load with an empty
#: missing/unexpected pair.
AFFONLY_KEY_REMAP: tuple[tuple[str, str], ...] = (("proj.", "pi."),)


def build_model(cfg, *, carrier: str = CARRIER_DEFAULT):
    """``CarrierModel`` with the ``--carrier`` 档's generator installed.

    ``affonly_frozen`` swaps ``CarrierModel.generator`` for the ``affine_only``
    :class:`~q3vl.whatb.generator.CGLUTGenerator` and nothing else.  The swap is
    enough because ``CGLUTGenerator`` returns the shared tables **broadcast to
    the batch** inside :class:`~q3vl.whatb.glut.GlutParams` (generator.py:357),
    so ``train_step`` / ``quick_eval`` / ``interpolation_columns`` /
    ``params_slice`` in ``arms/carrier.py`` need no mode switch -- and none of
    them may be edited, three of the campaign's live jobs import that file.

    Measured equivalence: with the same weights loaded on both sides,
    ``AffineOnlyHead.transform(z, grid17)`` and
    ``CarrierModel.transform_grid(z, grid17)`` are ``torch.equal`` (the run-time
    form of this is :func:`assert_stage1_forward_identity`).
    """
    from q3vl.whatb.arms.carrier import CarrierModel

    if carrier not in CARRIER_CHOICES:
        raise ValueError(f"--carrier must be one of {CARRIER_CHOICES}, got {carrier!r}")
    model = CarrierModel(cfg)
    if carrier == "affonly_frozen":
        model.generator = CGLUTGenerator(
            cond_dim=int(cfg.cond_dim), hidden=int(cfg.gen_width),
            n_gauss=int(cfg.n_gauss), eps=GLUT_EPS, **AFFONLY_GENERATOR_FORM)
    return model


def shared_geometry_of(model) -> Any:
    """The :class:`~q3vl.whatb.generator.SharedGeometry`, or ``None``."""
    return getattr(getattr(model, "generator", None), "shared_geometry", None)


def carrier_facts(model, carrier: str) -> dict[str, Any]:
    """What ``run_setup.json`` records about the 档 -- shapes, not adjectives."""
    sg = shared_geometry_of(model)
    out: dict[str, Any] = {
        "carrier": carrier,
        "generator": dict(model.generator.config),
        "theta_gen_dim": int(model.generator.theta_dim),
        "n_params_generator": int(sum(p.numel() for p in model.generator.parameters())),
        "n_params_generator_trainable": int(
            sum(p.numel() for p in model.generator.parameters() if p.requires_grad)),
    }
    if sg is None:
        out["shared_geometry"] = None
        return out
    tensors = {n: getattr(sg, n) for n in SHARED_GEOMETRY_TENSORS}
    out["shared_geometry"] = {
        "n_gauss": int(sg.n_gauss),
        "mu_init": AFFONLY_MU_INIT,
        "sigma_init": float(sg.init_sigma),
        "opacity_logit_init": float(sg.init_opacity_logit),
        "n_tensors": len(tensors),
        "n_params": int(sum(t.numel() for t in tensors.values())),
        "shapes": {n: list(t.shape) for n, t in tensors.items()},
        "requires_grad": {n: bool(t.requires_grad) for n, t in tensors.items()},
    }
    return out


def stage1_tensors(blob: Any) -> dict[str, Tensor]:
    """The tensor dict inside a stage-1 checkpoint, whichever runner wrote it.

    ``run_affonly_arm.py:621`` writes ``{"state_dict": head.state_dict(), ...}``;
    ``run_carrier_arm`` / this runner write ``{"model": ..., ...}``.  Both keys
    are accepted and a bare state dict is accepted too; anything else raises
    rather than silently warm-starting from nothing.
    """
    if isinstance(blob, Mapping):
        for key in ("state_dict", "model"):
            sd = blob.get(key)
            if isinstance(sd, Mapping):
                return dict(sd)
        if blob and all(isinstance(v, Tensor) for v in blob.values()):
            return dict(blob)
    raise AssertionError(
        "the stage-1 checkpoint carries neither a 'state_dict' key "
        "(run_affonly_arm.py:621) nor a 'model' key (run_carrier_arm / this "
        f"runner) nor a bare tensor dict; top-level keys were "
        f"{sorted(blob)[:8] if isinstance(blob, Mapping) else type(blob).__name__}")


def remap_stage1_keys(sd: Mapping[str, Tensor], *,
                      carrier: str = CARRIER_DEFAULT) -> dict[str, Tensor]:
    """``AffineOnlyHead`` key names -> ``CarrierModel`` key names."""
    if carrier != "affonly_frozen":
        return dict(sd)
    out: dict[str, Tensor] = {}
    for k, v in sd.items():
        for src, dst in AFFONLY_KEY_REMAP:
            if k.startswith(src):
                k = dst + k[len(src):]
                break
        out[k] = v
    return out


def affonly_config_from_blob(blob: Mapping[str, Any]):
    """Rebuild the stage-1 ``AffineOnlyConfig`` from the checkpoint's own config.

    ``run_affonly_arm`` saves ``head.config``, which is ``AffineOnlyConfig.to_dict()``
    plus derived keys; only the dataclass fields are read back, so a derived key
    (``steps_per_epoch``, ``theta_gen_dim``, ...) can never be fed in as a knob.
    """
    from q3vl.whatb.arms.affonly import AffineOnlyConfig

    raw = blob.get("config") if isinstance(blob, Mapping) else None
    if not isinstance(raw, Mapping):
        raise AssertionError(
            "the stage-1 checkpoint has no 'config' block, so the AFFONLY head it "
            "was written from cannot be rebuilt and the forward-parity assertion "
            "would have nothing to compare against")
    fields = {k for k in AffineOnlyConfig.__dataclass_fields__ if k != "degeneracy"}
    return AffineOnlyConfig(**{k: v for k, v in raw.items() if k in fields})


def load_stage1(model, blob: Any, *, carrier: str = CARRIER_DEFAULT,
                path: str = "") -> dict[str, Any]:
    """Warm-start ``pi`` + generator from stage 1.  A partial load raises."""
    sd = remap_stage1_keys(stage1_tensors(blob), carrier=carrier)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise AssertionError(
            f"--stage1 {path} does not fit the --carrier {carrier} model: missing "
            f"{list(missing)[:6]}, unexpected {list(unexpected)[:6]}.  Stage 2 "
            "continues stage 1; a partial load would silently reinitialise "
            "whichever half did not match.")
    cfg1 = blob.get("config") if isinstance(blob, Mapping) else None
    return {
        "path": str(path), "carrier": carrier,
        "step": int(blob.get("step", -1)) if isinstance(blob, Mapping) else -1,
        "headline": float(blob.get("headline", blob.get("headline_normal_only",
                                                        float("nan"))))
        if isinstance(blob, Mapping) else float("nan"),
        "n_tensors_loaded": len(sd),
        "key_remap": [list(p) for p in AFFONLY_KEY_REMAP]
        if carrier == "affonly_frozen" else [],
        "config": cfg1,
    }


def freeze_shared_geometry(model) -> dict[str, Any]:
    """``requires_grad_(False)`` on the four shared tensors.

    Returns the record; the *assertion* that it took is
    :func:`assert_geometry_frozen`, which also compares against stage 1.
    """
    sg = shared_geometry_of(model)
    if sg is None:
        raise AssertionError(
            "--carrier affonly_frozen has no SharedGeometry to freeze; the "
            "generator was built in Full Generation mode")
    for name in SHARED_GEOMETRY_TENSORS:
        getattr(sg, name).requires_grad_(False)
    return {"n_tensors": len(SHARED_GEOMETRY_TENSORS),
            "n_params": int(sum(getattr(sg, n).numel()
                                for n in SHARED_GEOMETRY_TENSORS))}


def assert_geometry_frozen(model, blob: Any, *,
                           carrier: str = "affonly_frozen") -> dict[str, Any]:
    """``A_geom_frozen``: the four shared tensors are dead **and** are stage 1's.

    Two separate failures, both invisible in every metric:

    * a tensor that still carries ``requires_grad`` -- the arm's claim is
      "frozen analytic geometry", and Adam would move it at ``base_lr``;
    * a tensor that is not bit-for-bit the stage-1 one -- a key that did not
      land (``proj.`` / ``pi.`` remap, a renamed submodule) leaves the *fresh*
      initialisation in place, which for ``mu``/``sigma``/``o`` is numerically
      close enough to the stage-1 table to look plausible on a loss curve.

    ``torch.equal``, not ``allclose``: both sides are the same fp32 tensor.
    """
    sg = shared_geometry_of(model)
    if sg is None:
        raise AssertionError(
            "A_geom_frozen ran on a model with no SharedGeometry; --carrier "
            f"{carrier!r} did not install the affine-only generator")
    ref = remap_stage1_keys(stage1_tensors(blob), carrier=carrier)
    unfrozen: list[str] = []
    differing: list[str] = []
    per_tensor: dict[str, Any] = {}
    for name in SHARED_GEOMETRY_TENSORS:
        key = f"generator.shared_geometry.{name}"
        live = getattr(sg, name)
        if key not in ref:
            raise AssertionError(
                f"A_geom_frozen: the stage-1 checkpoint has no {key!r}; it was "
                "not written by an AFFONLY run under --share geo_opacity, so "
                "there is no shared geometry to continue from")
        want = ref[key].to(device=live.device, dtype=live.dtype)
        same = bool(torch.equal(live.detach(), want))
        if live.requires_grad:
            unfrozen.append(name)
        if not same:
            differing.append(name)
        per_tensor[name] = {
            "requires_grad": bool(live.requires_grad),
            "equals_stage1": same,
            "max_abs_dev": float((live.detach() - want).abs().max()),
            "shape": list(live.shape),
        }
    if unfrozen or differing:
        raise AssertionError(
            f"A_geom_frozen FAILED: {unfrozen or 'none'} still require grad and "
            f"{differing or 'none'} differ from the stage-1 tensors.  Details: "
            + repr(per_tensor))
    return {"n_tensors": len(SHARED_GEOMETRY_TENSORS),
            "n_params": int(sum(getattr(sg, n).numel()
                                for n in SHARED_GEOMETRY_TENSORS)),
            "all_frozen": True, "all_equal_stage1": True, "bitwise": True,
            "per_tensor": per_tensor}


def assert_stage1_forward_identity(model, blob: Any, z: Tensor, *,
                                   grid: Tensor | None = None) -> dict[str, Any]:
    """``A_stage1``: step 0 reproduces the stage-1 generator, bitwise.

    The same statement ``A_step0`` makes about the adapter, made about the
    carrier port: a fresh :class:`~q3vl.whatb.arms.affonly.AffineOnlyHead` is
    rebuilt from the checkpoint's own config, loaded with the checkpoint's own
    weights, and its ``f(x)`` on the 17^3 grid is compared with this model's
    ``f(x)`` for the same conditions.  Both sides are fp32
    (``CarrierModel``/``AffineOnlyHead`` parameters are fp32 regardless of the
    VLM's bf16), so ``torch.equal`` is reachable and a non-zero delta means the
    port -- not arithmetic -- moved.  ``z`` is the cached stage-1 condition, so
    with ``--lora`` this is the adapter-off condition by ``A_step0``.
    """
    from q3vl.whatb.arms.affonly import AffineOnlyHead

    cfg1 = affonly_config_from_blob(blob)
    if cfg1.share != "geo_opacity":
        raise AssertionError(
            f"--carrier affonly_frozen continues an AFFONLY run under --share "
            f"geo_opacity (mu, Sigma AND o shared); the stage-1 checkpoint says "
            f"--share {cfg1.share!r}")
    for name, want, got in (("n_gauss", cfg1.n_gauss, model.cfg.n_gauss),
                            ("cond_dim", cfg1.cond_dim, model.cfg.cond_dim),
                            ("gen_width", cfg1.gen_width, model.cfg.gen_width),
                            ("clamp", cfg1.clamp, model.cfg.clamp)):
        if want != got:
            raise AssertionError(
                f"stage 1 was trained with {name} = {want!r} and this run declares "
                f"{got!r}; the two rows would not be the same construction")
    head = AffineOnlyHead(cfg1)
    ref = model.pi.proj.weight
    head.load_state_dict(stage1_tensors(blob))
    head = head.to(device=ref.device, dtype=ref.dtype).eval()
    x = model.query_grid17 if grid is None else grid.to(device=ref.device,
                                                        dtype=ref.dtype)
    zz = z.to(device=ref.device, dtype=ref.dtype)
    with torch.no_grad():
        y_ref = head.transform(zz, x)
        y_got = model.transform_grid(zz, x)
    assert isinstance(y_ref, Tensor) and isinstance(y_got, Tensor)
    delta = float((y_got - y_ref).abs().max())
    exact = bool(torch.equal(y_got, y_ref))
    rec = {"n": int(zz.shape[0]), "grid_n": int(round(float(x.shape[0]) ** (1 / 3))),
           "max_abs_delta": delta, "exact": exact, "dtype": str(ref.dtype),
           "quantity": "max_x |f_carrier(x) - f_affonly_head(x)| on the 17^3 grid"}
    if not exact:
        raise AssertionError(
            f"A_stage1 FAILED: this arm's affonly_frozen carrier and the stage-1 "
            f"AffineOnlyHead disagree by max|df| = {delta:.6e} over "
            f"{zz.shape[0]} conditions, expected exactly 0.  Stage 2 would not "
            "start from the stage-1 function.  Details: " + repr(rec))
    return rec


# --------------------------------------------------------------------------- #
# 1. where the adapters go
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LoraFacts:
    """Everything ``run_setup.json`` has to carry about the adapter surface."""

    r: int
    alpha: int
    dropout: float
    last_n_blocks: int
    target_suffixes: tuple[str, ...]
    n_text_layers: int
    layer_indices: tuple[int, ...]
    target_modules: tuple[str, ...]
    n_lora_params: int
    n_base_params: int
    n_trainable_vlm_params: int
    #: the measured records of ``A_lora`` / ``A_freeze``.  They run inside
    #: :func:`attach_lora`, so without carrying them out the artefact would only
    #: show that the run did not die -- not that the two checks were performed.
    assertions: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "assertions": dict(self.assertions),
            "r": self.r, "alpha": self.alpha, "dropout": self.dropout,
            "scaling": float(self.alpha) / float(self.r),
            "last_n_blocks": self.last_n_blocks,
            "target_suffixes": list(self.target_suffixes),
            "n_text_layers": self.n_text_layers,
            "layer_indices": list(self.layer_indices),
            "n_target_modules": len(self.target_modules),
            "target_modules": list(self.target_modules),
            "n_lora_params": self.n_lora_params,
            "n_base_params": self.n_base_params,
            "n_trainable_vlm_params": self.n_trainable_vlm_params,
            "trainable_fraction": (self.n_trainable_vlm_params
                                   / max(1, self.n_base_params)),
            "vision_tower": "frozen",
            "embedding": "frozen",
            "language_blocks_frozen": self.n_text_layers - self.last_n_blocks,
        }


def lora_target_names(model: torch.nn.Module, *,
                      last_n: int = LORA_LAST_N_BLOCKS,
                      suffixes: Sequence[str] = LORA_TARGET_SUFFIXES,
                      ) -> tuple[list[str], list[int]]:
    """Fully-qualified names of the modules that get an adapter, and their blocks.

    Fully qualified and not suffixes: peft's ``check_target_module_exists``
    accepts a bare ``"q_proj"`` and would then adapt the vision tower's
    attention too (``model.visual.blocks.*``), which is exactly the "one module
    too many" failure this arm cannot afford.  A full name hits peft's
    ``key in config.target_modules`` branch, i.e. an exact match.
    """
    lm = resolve_language_model(model)
    n_layers = len(lm.layers)
    if not 1 <= int(last_n) <= n_layers:
        raise ValueError(
            f"--lora-last-n must be 1..{n_layers} (the tower has {n_layers} "
            f"blocks), got {last_n}")
    by_module = {id(mod): name for name, mod in model.named_modules()}
    if id(lm) not in by_module:                            # pragma: no cover
        raise AssertionError("the language tower is not a submodule of the model")
    layer_indices = list(range(n_layers - int(last_n), n_layers))
    names: list[str] = []
    for li in layer_indices:
        block_name = by_module.get(id(lm.layers[li]))
        if block_name is None:                             # pragma: no cover
            raise AssertionError(f"block {li} is not reachable from the model")
        for suf in suffixes:
            full = f"{block_name}.{LORA_ATTN_ATTR}.{suf}"
            if not isinstance(model.get_submodule(full), torch.nn.Linear):
                raise AssertionError(
                    f"{full} is not an nn.Linear; the adapter surface this arm "
                    "declares is q/k/v/o of the self-attention")
            names.append(full)
    return names, layer_indices


def attach_lora(model: torch.nn.Module, *, r: int = LORA_R,
                alpha: int = LORA_ALPHA, dropout: float = LORA_DROPOUT,
                last_n: int = LORA_LAST_N_BLOCKS,
                suffixes: Sequence[str] = LORA_TARGET_SUFFIXES,
                ) -> tuple[torch.nn.Module, LoraFacts]:
    """Wrap ``model`` in a peft ``LoraModel`` over exactly the declared modules.

    Returns ``(peft_model, facts)``.  Both start-up assertions (``A_lora`` and
    ``A_freeze``) have already run when this returns, so a caller cannot reach
    a training step with an adapter surface it did not declare.
    """
    from peft import LoraConfig, get_peft_model

    n_base = sum(p.numel() for p in model.parameters())
    targets, layer_indices = lora_target_names(model, last_n=last_n,
                                               suffixes=suffixes)
    cfg = LoraConfig(
        r=int(r), lora_alpha=int(alpha), lora_dropout=float(dropout),
        bias="none", target_modules=list(targets), init_lora_weights=True,
        # not a task peft knows about: the head is this repo's pi + generator,
        # so no task-specific wrapper may be added on top of the tower.
        task_type=None,
    )
    peft_model = get_peft_model(model, cfg)
    checks = {"A_lora": assert_declared_surface(peft_model, targets),
              "A_freeze": assert_frozen_area(peft_model, targets)}
    lm = resolve_language_model(peft_model)
    n_lora = sum(p.numel() for n, p in peft_model.named_parameters() if "lora_" in n)
    facts = LoraFacts(
        r=int(r), alpha=int(alpha), dropout=float(dropout),
        last_n_blocks=int(last_n), target_suffixes=tuple(suffixes),
        n_text_layers=len(lm.layers), layer_indices=tuple(layer_indices),
        target_modules=tuple(targets), n_lora_params=int(n_lora),
        n_base_params=int(n_base),
        n_trainable_vlm_params=int(sum(p.numel() for p in peft_model.parameters()
                                       if p.requires_grad)),
        assertions=checks,
    )
    return peft_model, facts


@contextlib.contextmanager
def adapter_disabled(model: torch.nn.Module) -> Iterator[None]:
    """Run the base tower with the adapters switched off.

    Used by three things that all need the *same* forward without LoRA: the
    step-0 identity assertion, the cache-parity assertion, and the ``z_drift``
    diagnostic column.
    """
    disable = getattr(model, "disable_adapter", None)
    if disable is None:                                    # pragma: no cover
        yield
        return
    with disable():
        yield


def trainable_lora_parameters(model: torch.nn.Module) -> list[Tensor]:
    """The adapter parameters, in ``named_parameters`` order."""
    return [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]


def assert_declared_surface(model: torch.nn.Module,
                            declared: Sequence[str]) -> dict[str, Any]:
    """``A_lora``: the adapted set equals the declared set, as a set."""
    from peft.tuners.lora import LoraLayer

    got = set()
    for name, mod in model.named_modules():
        if isinstance(mod, LoraLayer):
            # peft renames ``a.b.q_proj`` to ``base_model.model.a.b.q_proj``
            got.add(name.split("base_model.model.", 1)[-1])
    want = set(str(d) for d in declared)
    if got != want:
        extra, missing = sorted(got - want), sorted(want - got)
        raise AssertionError(
            f"the adapter surface is not the declared one: {len(extra)} module(s) "
            f"were adapted that this arm did not declare ({extra[:6]}) and "
            f"{len(missing)} declared module(s) were not adapted "
            f"({missing[:6]}).  peft matches target_modules by suffix, so a "
            "suffix that reaches the vision tower is invisible in every metric.")
    return {"n_adapted": len(got), "matches_declared": True}


def assert_frozen_area(model: torch.nn.Module,
                       declared: Sequence[str]) -> dict[str, Any]:
    """``A_freeze``: only LoRA parameters of declared modules may train."""
    want = set(str(d) for d in declared)
    bad: list[str] = []
    n_trainable = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n_trainable += 1
        short = name.split("base_model.model.", 1)[-1]
        if "lora_" not in short:
            bad.append(name)
            continue
        owner = short.split(".lora_", 1)[0]
        if owner not in want:
            bad.append(name)
    if bad:
        raise AssertionError(
            f"{len(bad)} trainable VLM parameter(s) are outside the declared LoRA "
            f"surface, first few: {bad[:6]}.  This arm's claim is 'LoRA on the "
            "last eight language blocks' -- the vision tower, the embedding "
            "table and the other blocks stay frozen.")
    if n_trainable == 0:
        raise AssertionError(
            "no VLM parameter is trainable after attaching LoRA; the adapter is "
            "wired but nothing would ever be optimised (the campaign's "
            "'defined but not wired' failure, fourth occurrence)")
    return {"n_trainable_tensors": n_trainable, "outside_declared": 0}


# --------------------------------------------------------------------------- #
# 2. the grad-carrying forward
# --------------------------------------------------------------------------- #
@dataclass
class SpanItem:
    """One sample's teacher-forced forward: prompt + the whole reply, no sampling."""

    sample_id: str
    image: Any                     # PIL image, already spec-5 sized
    prompt_ids: Sequence[int]
    plan: ReplyPlan                # token_ids = the reply; start/end = the span


class SpanPoolEncoder:
    """``SpanItem`` -> ``z`` with the gradient intact.

    :class:`q3vl.whereb.hiddens.FrozenVLM` cannot serve here: its ``encode`` is
    ``@torch.no_grad()`` and its ``__init__`` calls ``requires_grad_(False)`` on
    every parameter, both of which are exactly right for the six frozen arms and
    exactly wrong for this one.  Rather than add a flag to a module three live
    jobs import, the forward is re-stated here **with the same contract**: same
    hidden layer (:data:`SEGMENT_HIDDEN_LAYER`), same final norm
    (:data:`SEGMENT_HIDDEN_FINAL_NORM`), same right padding, same
    ``hidden[i, n_p:n_p+n_w]`` slice, same ``.float()`` before
    :func:`~q3vl.whereb.readout.readout_vector` -- so with the adapter disabled
    it reproduces the cache builder's arithmetic, which ``A_cache`` measures.

    ``F_pre`` is not captured: the what side never reads it, and the hook costs
    a full ``(grid, 1024)`` tensor per sample.
    """

    def __init__(self, model: torch.nn.Module, processor, *,
                 device: str | torch.device = "cuda",
                 layer: int = SEGMENT_HIDDEN_LAYER,
                 final_norm: bool = SEGMENT_HIDDEN_FINAL_NORM,
                 micro_batch: int = 8):
        self.model = model
        self.processor = processor
        self.device = torch.device(device)
        self.layer = int(layer)
        self.final_norm = bool(final_norm)
        self.micro_batch = max(1, int(micro_batch))
        self.visual = resolve_visual(model)
        self.lm = resolve_language_model(model)
        self.pad_id = processor.tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = processor.tokenizer.eos_token_id

    # -- one padded forward --------------------------------------------------
    def _forward_chunk(self, items: Sequence[SpanItem]) -> Tensor:
        """``(B, T, 2560)`` post-norm hidden states for one padded chunk."""
        tok_lists = [list(it.prompt_ids) + [int(t) for t in it.plan.token_ids]
                     for it in items]
        max_len = max(len(t) for t in tok_lists)
        input_ids = torch.full((len(items), max_len), self.pad_id, dtype=torch.long)
        attn = torch.zeros((len(items), max_len), dtype=torch.long)
        for i, t in enumerate(tok_lists):
            input_ids[i, : len(t)] = torch.tensor(t, dtype=torch.long)
            attn[i, : len(t)] = 1
        img = self.processor.image_processor(
            images=[it.image for it in items], do_resize=False, return_tensors="pt")
        dtype = next(self.model.parameters()).dtype
        hook = LastLayerHook(self.lm, self.layer)
        with hook.attached():
            self.model(
                input_ids=input_ids.to(self.device),
                attention_mask=attn.to(self.device),
                pixel_values=img["pixel_values"].to(self.device, dtype),
                image_grid_thw=img["image_grid_thw"].to(self.device),
            )
            hidden = hook.captured
        if hidden is None:                                 # pragma: no cover
            raise RuntimeError("no decoder hidden states captured")
        if self.final_norm:
            hidden = self.lm.norm(hidden)
        if hidden.shape[-1] != Z_DIM:
            raise AssertionError(
                f"h is {hidden.shape[-1]}-dim, the readout contract says {Z_DIM}")
        return hidden

    def h_reply(self, items: Sequence[SpanItem]) -> list[Tensor]:
        """``(T_reply, 2560)`` fp32 reply hiddens per item, gradient intact."""
        out: list[Tensor] = []
        for i in range(0, len(items), self.micro_batch):
            chunk = items[i:i + self.micro_batch]
            hidden = self._forward_chunk(chunk)
            for j, it in enumerate(chunk):
                n_p, n_w = len(it.prompt_ids), len(it.plan.token_ids)
                out.append(hidden[j, n_p:n_p + n_w].float())
        return out

    def z(self, items: Sequence[SpanItem], *, grad: bool = True) -> Tensor:
        """``(B, 2560)`` fp32 conditions.  ``grad=False`` runs under ``no_grad``."""
        ctx = contextlib.nullcontext() if grad else torch.no_grad()
        with ctx:
            hs = self.h_reply(items)
            return torch.stack([readout_vector(h, it.plan)
                                for h, it in zip(hs, items)], dim=0)

    def facts(self) -> dict[str, Any]:
        return {
            "layer": self.layer, "final_norm": self.final_norm,
            "micro_batch": self.micro_batch, "text_hidden": Z_DIM,
            "n_text_layers": len(self.lm.layers),
            "n_vision_blocks": len(self.visual.blocks),
            "dtype": str(next(self.model.parameters()).dtype),
            "device": str(self.device),
            "n_trainable_params": int(sum(p.numel() for p in self.model.parameters()
                                          if p.requires_grad)),
            "padding": "right (same as q3vl/whereb/hiddens.py:172-176)",
        }


# --------------------------------------------------------------------------- #
# 3. the reply plan, replayed from the cache
# --------------------------------------------------------------------------- #
def plan_from_cache_row(builder: WhatReadoutBuilder, row: Mapping[str, Any], *,
                        source: str = "generated",
                        control_tag: str = "none") -> ReplyPlan:
    """Rebuild the teacher-forced plan of one cached row, then check it bitwise.

    The cached row carries the reply the frozen VLM actually generated
    (``reply_token_ids``) and the slice it pooled (``readout_index`` =
    ``[start, end]``).  For ``color_span_pool`` the reply is ``w + c`` and
    ``start`` is exactly ``len(w)``, so the two halves are recoverable, fed back
    through :func:`q3vl.whatb.readout.build_reply`, and the result compared with
    what the cache recorded.  The comparison is over integers, so it is exact:
    it catches a span rebuilt one token off, which no metric would show.
    """
    ids = [int(t) for t in row["reply_token_ids"]]
    idx = row["readout_index"]
    if not isinstance(idx, (list, tuple)) or len(idx) != 2:
        raise AssertionError(
            f"{row.get('sample_id')}: readout_index is {idx!r}; "
            "color_span_pool is a pooled kind and its cached index must be "
            "[start, end] (build_zcache.readout_index_field)")
    start, end = int(idx[0]), int(idx[1])
    plan = builder.plan_for(sample_id=str(row.get("sample_id")),
                            where_ids=ids[:start], color_ids=ids[start:end],
                            source=source, control_tag=control_tag)
    assert_reply_parity(plan, row)
    return plan


def assert_reply_parity(plan: ReplyPlan, row: Mapping[str, Any]) -> None:
    """``A_reply``: token-for-token identity with the cached reply."""
    sid = row.get("sample_id")
    cached = [int(t) for t in row["reply_token_ids"]]
    got = [int(t) for t in plan.token_ids]
    if got != cached:
        n = next((i for i, (a, b) in enumerate(zip(got, cached)) if a != b),
                 min(len(got), len(cached)))
        raise AssertionError(
            f"{sid}: the teacher-forced reply this arm would feed the forward "
            f"differs from the cached reply_token_ids at token {n} "
            f"(rebuilt {got[n:n + 1]}, cached {cached[n:n + 1]}; lengths "
            f"{len(got)} vs {len(cached)}).  The two must agree token for token "
            "or stage 2 starts from a different condition than stage 1 ended on.")
    idx = [int(plan.start), int(plan.end)]
    if idx != [int(row["readout_index"][0]), int(row["readout_index"][1])]:
        raise AssertionError(
            f"{sid}: rebuilt pooling slice {idx} != cached "
            f"{list(row['readout_index'])}")
    if not plan.pool:
        raise AssertionError(
            f"{sid}: color_span_pool must pool; the rebuilt plan does not")


# --------------------------------------------------------------------------- #
# 4. the two numerical start-up assertions
# --------------------------------------------------------------------------- #
def assert_lora_step0_identity(encoder: "SpanPoolEncoder",
                               items: Sequence[SpanItem]) -> dict[str, Any]:
    """``A_step0``: with ``B = 0`` the adapter is the zero map, bitwise.

    Same items, same padding shape, same forward -- only the adapter switched
    off -- so unlike the cache comparison this one **is** exact, in bf16 or
    anywhere else.  A non-zero delta here means either the adapter was not
    zero-initialised or something other than LoRA moved.
    """
    with torch.no_grad():
        z_on = encoder.z(items, grad=False)
        with adapter_disabled(encoder.model):
            z_off = encoder.z(items, grad=False)
    delta = float((z_on - z_off).abs().max())
    if delta != 0.0:
        raise AssertionError(
            f"LoRA is not the identity at step 0: max|z_adapter_on - "
            f"z_adapter_off| = {delta:.6e} over {len(items)} samples, expected "
            "exactly 0 (lora_B is zero-initialised, so B A x = 0).  Stage 2 "
            "would not start from the stage-1 condition.")
    return {"n": len(items), "max_abs_delta": delta, "exact": True}


def assert_cache_parity(encoder: "SpanPoolEncoder", items: Sequence[SpanItem],
                        z_cached: Tensor, *, tol: float = 1e-3,
                        factor: float = 4.0) -> dict[str, Any]:
    """``A_cache``: adapter-off z reproduces the cache, to a *measured* threshold.

    The control is the same tokens re-encoded one at a time: everything about
    the sequence is identical, only the padding shape of the batch changes.  In
    bf16 that alone is O(1) on this model (the cache's own build report records
    5.75 against max|z| = 67.5), so it -- and not a guessed epsilon -- is what
    sets the bar.  The verdict function is the cache builder's own.
    """
    from q3vl.whatb.scripts.build_zcache import causality_gate_verdict

    with torch.no_grad(), adapter_disabled(encoder.model):
        z_batch = encoder.z(items, grad=False)
        z_solo = torch.cat([encoder.z([it], grad=False) for it in items], dim=0)
    ref = z_cached.to(device=z_batch.device, dtype=z_batch.dtype)
    delta = float((z_batch - ref).abs().max())
    control = float((z_batch - z_solo).abs().max())
    thr, ok = causality_gate_verdict(delta=delta, control=control, tol=tol,
                                     factor=factor)
    rec = {"n": len(items), "max_abs_delta_vs_cache": delta,
           "max_abs_delta_batch_control": control,
           "max_abs_z": float(z_batch.abs().max()),
           "threshold": thr, "tol": float(tol), "factor": float(factor),
           "passed": bool(ok)}
    if not ok:
        raise AssertionError(
            f"cache parity FAILED on {len(items)} samples: this arm's "
            f"adapter-off z differs from the cached z by max|dz| = {delta:.4e}, "
            f"over the threshold {thr:.4e} (= max(tol {tol:.1e}, {factor} x the "
            f"batch-shape control {control:.4e})); max|z| = {rec['max_abs_z']:.4e}.  "
            "Either the prompt this arm rebuilds is not the one the cache was "
            "written under, or the readout contract moved.  Details: " + repr(rec))
    return rec


# --------------------------------------------------------------------------- #
# 5. diagnostics
# --------------------------------------------------------------------------- #
def lora_grad_norm(params: Iterable[Tensor]) -> float:
    """``||g||_2`` over the adapter parameters, before clipping."""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().float().pow(2).sum())
    return math.sqrt(total)


def z_cos_drift(z_lora: Tensor, z_frozen: Tensor) -> float:
    """Mean per-sample ``cos(z_lora, z_frozen)`` -- 1.0 at step 0 by ``A_step0``."""
    a = z_lora.detach().float()
    b = z_frozen.detach().float()
    return float(torch.nn.functional.cosine_similarity(a, b, dim=-1).mean())


# --------------------------------------------------------------------------- #
# 6. the optimiser
# --------------------------------------------------------------------------- #
def build_optimizer(model, cfg, *, lora_params: Sequence[Tensor] = (),
                    lora_lr: float = LORA_LR) -> torch.optim.Adam:
    """The carrier's two groups, verbatim, plus the adapters at ``lora_lr``.

    Group order matters: ``arms/carrier.train_step`` writes ``lr_generator`` from
    ``param_groups[0]`` and ``lr_pi`` from ``param_groups[1]``, so the adapters
    are appended as a **third** group and every pre-existing step column keeps
    its meaning.  ``clip_grad_norm_`` in that same function walks all groups, so
    the pre-registered clip of 1.0 covers ``pi``, the generator and the adapters
    as one vector -- which is what "clip 1.0" has meant on every board of this
    campaign.

    ``requires_grad=False`` tensors are filtered out of the generator group.
    Under ``--carrier carrier`` nothing in the generator is frozen and the group
    is unchanged tensor for tensor; under ``--carrier affonly_frozen`` this is
    what keeps the four shared-geometry tables out of Adam.  Handing a frozen
    leaf to Adam is *silent* (``p.grad is None`` -> the step skips it), so the
    tensor count of the group is written onto ``run_setup.json`` rather than
    assumed.
    """
    from q3vl.whatb.arms.carrier import ADAM_BETAS

    groups: list[dict[str, Any]] = [
        {"params": [p for p in model.generator.parameters() if p.requires_grad],
         "lr": float(cfg.base_lr), "name": "generator"},
        {"params": list(model.pi.parameters()),
         "lr": float(cfg.base_lr) * float(cfg.proj_lr_scale), "name": "pi"},
    ]
    lora = [p for p in lora_params]
    if lora:
        groups.append({"params": lora, "lr": float(lora_lr), "name": "lora"})
    return torch.optim.Adam(groups, lr=float(cfg.base_lr), betas=ADAM_BETAS)
