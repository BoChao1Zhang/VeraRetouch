"""Protocol 14 -- Stage-What's share of the mandatory preflight.

    8.  prove that ``Q_where`` does not read ``H_color`` and ``Q_color`` does not
        read ``H_where``  (the second half; the first is Where-B's);
    9.  prove that no main arm's input contains ``I_tar``, a GT mask, a GT LUT or
        an oracle latent;
    12. 48-Gaussian SPD, weight normalisation, identity initialisation and finite
        gradients;
    7.  (transposed from protocol 5.4 by amendment A-4) the GT / generated
        ``<color>`` context flows do not cross, and there is no GT fallback;
    13. Lab loss units and per-term gradient norms;
    14. analytic -> 33^3 bake -> tetrahedral readback numerical unit test.

Plus four repo-specific rows the arms cannot start without:

    W1. every arm's ``T_gt`` resolves to a real ``.cube``/``.3dl`` on disk;
    W2. ``T_lut_unseen`` is genuinely LUT-disjoint from train / V_where / V_what /
        T_final, without which protocol 2.2 says the report may only say
        "held-out sample", not "unseen LUT generalisation";
    W3. the SRHT ``z_gt`` encoder is deterministic and distance preserving;
    W4. FG48 and SB48 differ by <= 2% in trainable parameters (protocol 7.5).

Every check returns ``id / status / detail``; :func:`run_what_preflight` writes
the lot to JSON and exits non-zero on any failure.  A *missing* required row is a
failure, not a silent pass -- the same rule Where-B's review forced (blocker B1).
"""

from __future__ import annotations

import argparse
import inspect
import json
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch

from . import attention as _attention_mod
from . import color as _color_mod
from .config import (
    ARM_IDS,
    BAKE_SIZE,
    LOSS_WEIGHTS,
    PARAM_MATCH_TOLERANCE,
    RUN_ROOT,
    SPLIT_DIR,
    ArmConfig,
    arm_config,
)
from .gaussians import bake, decode_global, decode_primitives, mixture_weights, render
from .generator import PRIM_LAYOUT_FG, match_report
from .lut import lattice_points, tetra_lookup
from .model import MODEL_INPUT_KEYS, WhatModel
from .srht import SRHT, default_srht, encode_z_gt
from .wc import WhereSignals

__all__ = ["Check", "PreflightReport", "run_what_preflight", "write_report"]

#: modules that must not contain the identifier ``where`` (protocol 14.8)
COLOR_PATH_MODULES = (_color_mod, _attention_mod)

REQUIRED_CHECKS: tuple[str, ...] = (
    "WT-P7-color-context-flows",
    "WT-P8-no-h-where",
    "WT-P9-no-target-leak",
    "WT-P12-gaussian-constraints",
    "WT-P13-lab-units-and-grad-norms",
    "WT-P14-bake-readback",
    "WT-W1-gt-lut-resolves",
    "WT-W2-lut-unseen-disjoint",
    "WT-W3-srht-deterministic",
    "WT-W4-param-match",
    "WT-P-arm-matrix",
    "WT-P-zero-init-identity",
)
#: rows only a real dataset / real VLM can satisfy
DATA_CHECKS: tuple[str, ...] = ("WT-W1-gt-lut-resolves", "WT-W2-lut-unseen-disjoint")


@dataclass
class Check:
    id: str
    status: str                       # pass | fail | skip
    detail: dict[str, Any] = field(default_factory=dict)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "status": self.status, "message": self.message,
                "detail": self.detail}


@dataclass
class PreflightReport:
    checks: list[Check] = field(default_factory=list)
    env: dict[str, Any] = field(default_factory=dict)
    required: tuple[str, ...] = REQUIRED_CHECKS

    def add(self, c: Check) -> Check:
        self.checks.append(c)
        return c

    @property
    def ids(self) -> set[str]:
        return {c.id for c in self.checks}

    @property
    def missing(self) -> list[str]:
        return [c for c in self.required if c not in self.ids]

    @property
    def ok(self) -> bool:
        return not self.missing and all(c.status != "fail" for c in self.checks)

    @property
    def complete(self) -> bool:
        return not self.missing and all(c.status == "pass" for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        skipped = [c.id for c in self.checks if c.status == "skip"]
        return {
            "ok": self.ok, "complete": self.complete,
            "n_pass": sum(c.status == "pass" for c in self.checks),
            "n_fail": sum(c.status == "fail" for c in self.checks),
            "n_skip": len(skipped), "skipped": skipped,
            "missing_required": self.missing, "required": list(self.required),
            "env": self.env, "checks": [c.to_dict() for c in self.checks],
        }


def _env() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parents[2]).stdout.strip()
    except Exception:                                            # noqa: BLE001
        commit = ""
    return {"git_commit": commit, "python": platform.python_version(),
            "torch": torch.__version__, "cuda_available": torch.cuda.is_available()}


#: library calls whose *name* collides with the forbidden substring but which
#: carry no stage information.  ``torch.where`` is the only one in practice; it is
#: allowlisted by its qualified form, so a bare local called ``where`` still trips.
IDENTIFIER_ALLOWLIST = frozenset({"torch.where"})


def _code_identifiers(path: Path) -> set[str]:
    """Identifiers a source file actually *uses*, qualified where possible.

    An AST walk rather than a token scan: ``torch.where`` is a legitimate tensor
    op and a raw token scan flags it, which would make the protocol 14.8 proof
    fire on library calls instead of on stage leakage.  Attributes of a simple
    name are recorded as ``base.attr`` so the allowlist can be specific.
    """
    import ast

    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            names.add(f"{node.value.id}.{node.attr}")
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.alias):
            names.add(node.asname or node.name)
    return {n for n in names if n not in IDENTIFIER_ALLOWLIST}


# --- amendment A-4: the two <color> context flows ---------------------------

def check_color_context_flows() -> Check:
    """Amendment A-4 -- protocol 5.4's discipline, transposed to ``<color>``.

    Four properties, each of which was false before A-4:

    1. the teacher and the generated builder are different functions, and the
       generated one **cannot** reach the GT text -- checked on its signature,
       not by reading its body;
    2. a generation with no closing tag is cut and marked, never replaced;
    3. the micro-batch is exactly 50/50 for any even micro-batch size;
    4. ``C01``/``C02`` map to the forced-``<color>``-prefix generation and every
       other arm to the with-``<where>``-prefix one.
    """
    from .config import (
        CONTEXT_GENERATED,
        CONTEXT_GT,
        GENCTX_MODE_FORCED_COLOR,
        GENCTX_MODE_WITH_WHERE,
        genctx_mode_of,
    )
    from .context import (
        BalancedContextSampler,
        generated_color_context,
        gt_color_context,
    )

    detail: dict[str, Any] = {}
    bad: list[str] = []

    # 1. no GT is reachable from the generated builder
    gen_params = set(inspect.signature(generated_color_context).parameters)
    detail["generated_color_context_params"] = sorted(gen_params)
    for forbidden in ("color_text", "sample", "record", "tokenizer"):
        if forbidden in gen_params:
            bad.append(f"generated_color_context accepts {forbidden!r}")

    # 2. a missing close tag is a recorded failure, not a fallback
    ctx = generated_color_context("s0", [10, 11, 12], close_id=99)
    detail["no_close_tag"] = ctx.to_dict()
    if not (ctx.format_failure and ctx.stop_reason == "no_close_tag"
            and ctx.token_ids == [10, 11, 12]):
        bad.append("a generation without </color> is not reported as a failure")
    closed = generated_color_context("s0", [10, 99, 12], close_id=99)
    detail["closed"] = closed.to_dict()
    if closed.token_ids != [10, 99] or closed.format_failure:
        bad.append("the span is not cut at the first </color>")
    empty = generated_color_context("s0", [], close_id=99)
    if not (empty.format_failure and empty.stop_reason == "empty"):
        bad.append("an empty generation is not reported")

    # 3. the 50/50 ratio, for every even micro-batch
    ratios = {}
    for mb in (2, 4, 8):
        sampler = BalancedContextSampler(64, mb, seed=0)
        counts = [sum(1 for _, m in batch if m == CONTEXT_GT) for batch in sampler]
        ratios[mb] = sorted(set(counts))
        if counts and set(counts) != {mb // 2}:
            bad.append(f"micro-batch {mb} is not 50/50: {sorted(set(counts))}")
    detail["teacher_per_micro_batch"] = ratios
    try:
        BalancedContextSampler(64, 3, seed=0)
        bad.append("an odd micro-batch was accepted; 50/50 needs an even one")
    except ValueError:
        detail["odd_micro_batch_rejected"] = True

    # 4. arm -> generation mode
    modes = {a: genctx_mode_of(a) for a in ARM_IDS}
    detail["genctx_mode_by_arm"] = modes
    forced = {a for a, m in modes.items() if m == GENCTX_MODE_FORCED_COLOR}
    if forced != {"C01", "C02"}:
        bad.append(f"forced-<color>-prefix arms are {sorted(forced)}, expected C01/C02")
    if any(modes[a] != GENCTX_MODE_WITH_WHERE for a in ARM_IDS if a not in forced):
        bad.append("a with-<where> arm is not on the with_where_prefix generation")

    # the teacher path does exist and is separate
    tok = _StubTokenizer()
    gt = gt_color_context(tok, "s0", "a body")
    detail["gt"] = gt.to_dict()
    if gt.mode != CONTEXT_GT or gt.genctx_mode:
        bad.append("the teacher context is not labelled as such")
    if ctx.mode != CONTEXT_GENERATED:
        bad.append("the generated context is not labelled as such")

    return Check("WT-P7-color-context-flows", "fail" if bad else "pass",
                 detail | {"offenders": bad})


class _StubTokenizer:
    """Just enough tokeniser for the context-flow check (no model needed)."""

    def __call__(self, text: str, add_special_tokens: bool = False):
        return {"input_ids": [hash(t) % 1000 for t in text.split()]}


# --- item 8 (the Stage-What half) -------------------------------------------

def check_no_h_where(model: WhatModel | None = None) -> Check:
    """``Q_color`` cannot read ``H_where`` -- structurally, not by convention."""
    detail: dict[str, Any] = {}
    bad: list[str] = []
    for mod in COLOR_PATH_MODULES:
        path = Path(inspect.getfile(mod))
        offenders = sorted(n for n in _code_identifiers(path) if "where" in n.lower())
        detail[mod.__name__] = {"file": str(path), "offending_identifiers": offenders}
        bad.extend(f"{mod.__name__}:{n}" for n in offenders)

    from .color import ColorConnector, ColorStack

    for fn, allowed in ((ColorConnector.forward, {"self", "queries", "h_color",
                                                  "h_color_mask"}),
                        (ColorStack.forward, {"self", "h_color", "h_color_mask"})):
        got = set(inspect.signature(fn).parameters)
        detail[fn.__qualname__] = sorted(got)
        if got - allowed:
            bad.append(f"{fn.__qualname__} accepts {sorted(got - allowed)}")

    if model is not None:
        names = [n for n, _ in model.color.named_parameters()]
        detail["color_parameter_names"] = names
        bad.extend(n for n in names if "where" in n.lower())
    return Check("WT-P8-no-h-where", "fail" if bad else "pass",
                 detail | {"offenders": bad},
                 "Q_color reads only H_color" if not bad else "H_where is reachable")


# --- item 9 ------------------------------------------------------------------

def check_no_target_leak(model: WhatModel | None = None, batch=None) -> Check:
    """No arm input carries ``I_tar``, a GT mask, a GT LUT or an oracle latent."""
    detail: dict[str, Any] = {"model_input_keys": list(MODEL_INPUT_KEYS)}
    bad: list[str] = []

    sig = set(inspect.signature(WhatModel.forward).parameters) - {"self"}
    detail["forward_signature"] = sorted(sig)
    if sig != set(MODEL_INPUT_KEYS):
        bad.append(f"forward takes {sorted(sig)}, whitelist is {sorted(MODEL_INPUT_KEYS)}")

    fields = set(WhereSignals.__dataclass_fields__)
    detail["where_signal_fields"] = sorted(fields)
    forbidden = [f for f in fields
                 if any(k in f.lower() for k in ("i_tar", "baked", "target", "lut"))]
    if forbidden:
        bad.append(f"WhereSignals carries {forbidden}")

    from .data import META_KEYS

    detail["meta_keys"] = list(META_KEYS)
    if any("baked" in k for k in META_KEYS):
        bad.append("META_KEYS keeps the baked-image locator")

    # the oracle arms are the protocol's own carve-out and must be tagged as such
    ceiling = [a for a in ARM_IDS if arm_config(a).where_source == "oracle"]
    detail["oracle_arms"] = ceiling
    if set(ceiling) != {"C03", "C04"}:
        bad.append(f"oracle inputs reach {ceiling}, protocol 8.2 allows C03/C04 only")

    if batch is not None:
        try:
            batch.check_inputs()
            detail["batch_check"] = "pass"
        except AssertionError as exc:
            bad.append(f"batch.check_inputs: {exc}")
    if model is not None:
        detail["arm"] = model.arm
        detail["is_ceiling"] = model.cfg.is_ceiling
    return Check("WT-P9-no-target-leak", "fail" if bad else "pass",
                 detail | {"offenders": bad},
                 "no target reachable from a model input" if not bad else "leak")


# --- item 12 -----------------------------------------------------------------

def check_gaussian_constraints(cfg: ArmConfig | None = None, n: int = 4) -> Check:
    """SPD, normalised weights, identity initialisation and finite gradients."""
    cfg = cfg or arm_config("T01")
    torch.manual_seed(0)
    z_prim = torch.randn(n, cfg.lut.n_slots, 23, requires_grad=True)
    z_glob = torch.randn(n, 12, requires_grad=True)
    from .gaussians import anchor_points

    anchors = anchor_points(cfg.lut.anchor_grid)
    p = decode_primitives(z_prim, cfg.lut, PRIM_LAYOUT_FG, anchors=anchors)
    p.update(decode_global(z_glob, cfg.lut))
    x = torch.rand(n, 512, 3)
    q = mixture_weights(p, x, cfg.lut.mixture_eps)
    y = render(p, x, cfg.lut)
    y.sum().backward()

    detail = {
        "mu_in_cube": bool(((p["mu"] > 0) & (p["mu"] < 1)).all()),
        "sigma_positive": bool((p["sigma"] > 0).all()),
        "sigma_min": float(p["sigma"].min()), "sigma_max": float(p["sigma"].max()),
        "weight_sum_min": float(q.sum(1).min()), "weight_sum_max": float(q.sum(1).max()),
        "opacity_in_01": bool(((p["opacity"] > 0) & (p["opacity"] < 1)).all()),
        "existence_in_01": bool(((p["existence"] > 0) & (p["existence"] < 1)).all()),
        "grad_prim_finite": bool(torch.isfinite(z_prim.grad).all()),
        "grad_glob_finite": bool(torch.isfinite(z_glob.grad).all()),
        "output_finite": bool(torch.isfinite(y).all()),
        "sigma_param": cfg.lut.sigma_param,
    }
    ok = (detail["mu_in_cube"] and detail["sigma_positive"]
          and detail["opacity_in_01"] and detail["existence_in_01"]
          and detail["grad_prim_finite"] and detail["grad_glob_finite"]
          and detail["output_finite"] and detail["weight_sum_max"] <= 1.0 + 1e-5)
    return Check("WT-P12-gaussian-constraints", "pass" if ok else "fail", detail)


def check_zero_init_identity(cfg: ArmConfig | None = None, tol: float = 1e-4) -> Check:
    """A zero-initialised head must give ``T(x) = x`` exactly (campaign red line).

    Also *measures* what the protocol's literal formula would do, so the
    ``2x`` claim behind ``GLOBAL_AFFINE_MODE`` is a number in the record rather
    than an argument in a comment.
    """
    cfg = cfg or arm_config("T01")
    from dataclasses import replace

    from .gaussians import anchor_points

    anchors = anchor_points(cfg.lut.anchor_grid)
    z_prim = torch.zeros(1, cfg.lut.n_slots, 23)
    z_glob = torch.zeros(1, 12)
    x = lattice_points(9).unsqueeze(0)

    def _t(lut) -> torch.Tensor:
        p = decode_primitives(z_prim, lut, PRIM_LAYOUT_FG, anchors=anchors)
        p.update(decode_global(z_glob, lut))
        return render(p, x, lut)

    err = float((_t(cfg.lut) - x).abs().max())
    literal = replace(cfg.lut, global_affine_mode="identity_centered")
    literal_raw = replace(literal, clamp_output=False)
    detail = {
        "mode": cfg.lut.global_affine_mode,
        "max_abs_identity_error": err,
        "tolerance": tol,
        # clamped: the 2x is capped at the top of the cube, so the worst error is
        # 0.5 (at x = 0.5).  Unclamped it is the full 1.0 (at x = 1).
        "literal_formula_max_abs_error": float((_t(literal) - x).abs().max()),
        "literal_formula_unclamped_max_abs_error":
            float((_t(literal_raw) - x).abs().max()),
        "literal_formula_unclamped_ratio":
            float((_t(literal_raw)[x > 0] / x[x > 0]).mean()),
        "literal_formula_note": (
            "protocol 7.6's literal (I + dG) x + b_g + sum_i q_i (M_i x + b_i) with "
            "identity-centred M_i gives T(x) = 2x at zero init; measured here"),
    }
    return Check("WT-P-zero-init-identity", "pass" if err <= tol else "fail", detail)


# --- item 13 -----------------------------------------------------------------

def check_lab_units_and_grad_norms(n: int = 8) -> Check:
    """Lab is dimensionless in the loss, and the two headline gradients are logged."""
    from .colorspace import srgb_to_lab, srgb_to_lab_norm
    from .losses import grad_norm_ratio, loss_func, loss_hue_chroma

    torch.manual_seed(0)
    rgb = torch.rand(n, 256, 3)
    raw, norm = srgb_to_lab(rgb), srgb_to_lab_norm(rgb)
    detail: dict[str, Any] = {
        "lab_raw_abs_max": float(raw.abs().max()),
        "lab_norm_abs_max": float(norm.abs().max()),
        "amplification_L": float(raw[..., 0].abs().max() / norm[..., 0].abs().max().clamp_min(1e-9)),
        "amplification_ab": float(raw[..., 1:].abs().max() / norm[..., 1:].abs().max().clamp_min(1e-9)),
    }
    p = torch.nn.Parameter(torch.zeros(1, 1, 3))
    t_pred = (rgb + p).clamp(0, 1)
    t_gt = torch.rand(n, 256, 3)
    lf = loss_func(t_pred, t_gt)["L_func"]
    lh = loss_hue_chroma(t_pred, t_gt)
    detail.update(grad_norm_ratio([p], lf, lh))
    detail["w_hc"] = LOSS_WEIGHTS["L_hc"]
    ok = (detail["lab_norm_abs_max"] <= 1.5
          and detail["amplification_ab"] > 100.0        # the unit error is real
          and all(torch.isfinite(torch.tensor(v)) for v in
                  (detail.get("grad_norm_L_func", 0.0),
                   detail.get("grad_norm_w_L_hc", 0.0))))
    return Check("WT-P13-lab-units-and-grad-norms", "pass" if ok else "fail", detail)


# --- item 14 -----------------------------------------------------------------

def check_bake_readback(size: int = BAKE_SIZE) -> Check:
    """analytic -> 33^3 bake -> tetrahedral readback, three independent assertions.

    1. reading the lattice points themselves returns the table exactly;
    2. an affine ``T`` is reproduced exactly, because tetrahedral interpolation is
       piecewise linear and reproduces affine functions on any lattice -- this is
       the assertion that isolates *the readback* from the Gaussian resampling.
       The clamp has to be off for it: ``clamp(.,0,1)`` is not affine, and a
       lattice cell straddling the clip boundary would make an exact readback
       impossible for reasons that have nothing to do with the interpolator;
    3. a random Gaussian mixture's resampling error is measured and reported.  It
       is an *arm result*, not a preflight gate -- protocol 12.1's 1e-4/5e-4 bake
       gate is what ``L_bake`` (weight 0.10) exists to drive it down to.
    """
    from dataclasses import replace

    cfg = arm_config("T01").lut
    torch.manual_seed(0)

    tbl = torch.rand(1, size, size, size, 3)
    pts = lattice_points(size).unsqueeze(0)
    exact = float((tetra_lookup(tbl, pts) - tbl.reshape(1, -1, 3)).abs().max())

    # an affine-only T: zero primitives (M = I, b = 0) plus a non-trivial global
    from .gaussians import anchor_points

    unclamped = replace(cfg, clamp_output=False)
    anchors = anchor_points(cfg.anchor_grid)
    z_prim = torch.zeros(1, cfg.n_slots, 23)
    z_glob = torch.randn(1, 12) * 0.5
    p = decode_primitives(z_prim, unclamped, PRIM_LAYOUT_FG, anchors=anchors)
    p.update(decode_global(z_glob, unclamped))
    x = torch.rand(1, 4096, 3)
    affine_err = float((tetra_lookup(bake(p, unclamped, size), x)
                        - render(p, x, unclamped)).abs().max())

    p2 = decode_primitives(torch.randn(1, cfg.n_slots, 23) * 0.5, cfg,
                           PRIM_LAYOUT_FG, anchors=anchors)
    p2.update(decode_global(torch.randn(1, 12) * 0.2, cfg))
    err = (tetra_lookup(bake(p2, cfg, size), x) - render(p2, x, cfg)).abs()
    detail = {
        "size": size,
        "lattice_readback_max_abs": exact,
        "affine_readback_max_abs": affine_err,
        "gaussian_mixture_mae": float(err.mean()),
        "gaussian_mixture_p99": float(torch.quantile(err.flatten(), 0.99)),
        "gaussian_mixture_max": float(err.max()),
        "non_finite": int((~torch.isfinite(err)).sum()),
        "note": ("rows 1-2 are the gate; row 3 is a random-parameter measurement "
                 "of the resampling cost the protocol 12.1 bake gate must be met "
                 "against after training"),
    }
    ok = exact < 1e-5 and affine_err < 1e-4 and detail["non_finite"] == 0
    return Check("WT-P14-bake-readback", "pass" if ok else "fail", detail)


# --- repo-specific rows ------------------------------------------------------

def check_gt_lut_resolves(splits: Sequence[str] = ("train", "V_what", "T_final",
                                                   "T_lut_unseen", "V_where"),
                          limit: int | None = None) -> Check:
    """Every ``lut_id`` used by a split resolves to a file that exists."""
    from .data import WhatDataset

    detail: dict[str, Any] = {}
    missing: list[str] = []
    try:
        for split in splits:
            ds = WhatDataset(split, need_mask=False, limit=limit, verify="none")
            m = ds.lut_path_map()
            absent = sorted(k for k, v in m.items() if not Path(v).exists())
            detail[split] = {"n_samples": len(ds), "n_lut": len(m),
                             "n_missing": len(absent), "missing": absent[:8]}
            missing.extend(absent)
    except (FileNotFoundError, OSError) as exc:
        return Check("WT-W1-gt-lut-resolves", "skip",
                     {"error": f"{type(exc).__name__}: {exc}"},
                     "split indexes or shards not reachable from here")
    return Check("WT-W1-gt-lut-resolves", "fail" if missing else "pass", detail)


def check_lut_unseen_disjoint() -> Check:
    """``T_lut_unseen`` shares no ``lut_id`` with any other split (protocol 2.2)."""
    sets: dict[str, set[str]] = {}
    try:
        for split in ("train", "V_where", "V_what", "T_final", "T_lut_unseen"):
            path = SPLIT_DIR / f"{split}.index.jsonl"
            ids = set()
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        ids.add(json.loads(line)["lut_id"])
            sets[split] = ids
    except (FileNotFoundError, OSError) as exc:
        return Check("WT-W2-lut-unseen-disjoint", "skip",
                     {"error": f"{type(exc).__name__}: {exc}"})
    unseen = sets.pop("T_lut_unseen")
    overlaps = {k: len(unseen & v) for k, v in sets.items()}
    detail = {"n_lut_unseen": len(unseen),
              "n_lut": {k: len(v) for k, v in sets.items()},
              "overlaps": overlaps}
    ok = all(v == 0 for v in overlaps.values())
    return Check("WT-W2-lut-unseen-disjoint", "pass" if ok else "fail", detail,
                 "" if ok else "report may only say 'held-out sample'")


def check_srht_deterministic(n_pairs: int = 64) -> Check:
    """The ``z_gt`` projection is reproducible and preserves relative distance."""
    a, b = default_srht(), SRHT(default_srht().n_in)
    torch.manual_seed(0)
    v = torch.randn(8, a.n_in)
    same = float((a(v) - b(v)).abs().max())

    u = torch.randn(n_pairs, a.n_in) * 0.05
    pu = a(u)
    iu = torch.triu_indices(n_pairs, n_pairs, offset=1)
    d0 = (u[iu[0]] - u[iu[1]]).norm(dim=-1)
    d1 = (pu[iu[0]] - pu[iu[1]]).norm(dim=-1)
    ratio = (d1 / d0.clamp_min(1e-12))
    z = encode_z_gt(torch.rand(3, 17 ** 3, 3))
    detail = {
        "digest": a.digest(), "seed": a.seed, "k": a.k, "pad": a.pad,
        "reconstruction_max_abs_diff": same,
        "distance_ratio_mean": float(ratio.mean()),
        "distance_ratio_std": float(ratio.std()),
        "distance_ratio_min": float(ratio.min()),
        "distance_ratio_max": float(ratio.max()),
        "z_gt_norms": [float(t) for t in z.norm(dim=-1)],
    }
    ok = (same == 0.0 and abs(detail["distance_ratio_mean"] - 1.0) < 0.05
          and detail["distance_ratio_min"] > 0.7 and detail["distance_ratio_max"] < 1.3
          and all(abs(nn - 1.0) < 1e-5 for nn in detail["z_gt_norms"]))
    return Check("WT-W3-srht-deterministic", "pass" if ok else "fail", detail)


def check_param_match(cfg: ArmConfig | None = None) -> Check:
    """Protocol 7.5: FG48 and SB48 within 2% of each other."""
    cfg = cfg or arm_config("T01")
    rep = match_report(cfg.backend, cfg.lut, cfg.fg_bottleneck, cfg.sb_bottleneck)
    return Check("WT-W4-param-match", "pass" if rep["within_tolerance"] else "fail",
                 rep | {"tolerance": PARAM_MATCH_TOLERANCE})


def check_arm_matrix() -> Check:
    """All twelve arms of protocol 8 construct, and the matrix is 4 WC x 2 gen + 4."""
    from .generator import parameter_matrix

    rows = parameter_matrix()
    main = [r for r in rows if r["arm"].startswith("T")]
    wcs = sorted({r["wc"] for r in main})
    gens = sorted({r["generator"] for r in main})
    detail = {
        "n_arms": len(rows), "main": len(main),
        "wc_ids": wcs, "generators": gens,
        "rows": [{k: r[k] for k in ("arm", "wc", "generator", "where_source",
                                    "n_trainable_params")} for r in rows],
    }
    ok = (len(rows) == 12 and len(main) == 8 and len(wcs) == 4 and len(gens) == 2)
    return Check("WT-P-arm-matrix", "pass" if ok else "fail", detail)


# --- driver ------------------------------------------------------------------

def run_what_preflight(out_dir: Path | None = None, *, arm: str = "T01",
                       with_data: bool = True, limit: int | None = None,
                       force: bool = False) -> PreflightReport:
    rep = PreflightReport(env=_env())
    cfg = arm_config(arm)
    model = WhatModel(cfg)
    rep.add(check_color_context_flows())
    rep.add(check_no_h_where(model))
    rep.add(check_no_target_leak(model))
    rep.add(check_gaussian_constraints(cfg))
    rep.add(check_zero_init_identity(cfg))
    rep.add(check_lab_units_and_grad_norms())
    rep.add(check_bake_readback())
    rep.add(check_srht_deterministic())
    rep.add(check_param_match(cfg))
    rep.add(check_arm_matrix())
    if with_data:
        rep.add(check_gt_lut_resolves(limit=limit))
        rep.add(check_lut_unseen_disjoint())
    else:
        for cid in DATA_CHECKS:
            rep.add(Check(cid, "skip", {}, "--no-data"))
    if out_dir is not None:
        write_report(rep, Path(out_dir), force=force)
    return rep


def write_report(rep: PreflightReport, out_dir: Path, *, force: bool = False) -> Path:
    """Write ``preflight_what.json``, refusing a silent completeness downgrade.

    Review N-16 / the reviewer's own accident (`REVIEW-impl-What` section 3-bis):
    the previous ``--out`` default was the *deliverable* directory, so anyone who
    ran ``python -m q3vl.what.preflight --no-data`` overwrote the delivered
    evidence with a 9-pass/2-skip version -- and the replacement still said
    ``ok: true``.  Only ``complete`` and ``skipped`` changed, which is exactly the
    campaign's "second failure mode is the silent one".

    Two guards: the default output is now a scratch directory, and overwriting a
    ``complete: true`` report with a ``complete: false`` one is refused unless the
    caller says ``--force`` (which first backs the old one up).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "preflight_what.json"
    payload = rep.to_dict()
    if path.exists() and not payload["complete"]:
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            old = {}
        if old.get("complete"):
            if not force:
                raise RuntimeError(
                    f"refusing to overwrite {path}: the existing report is "
                    f"complete (11/11) and the new one is not "
                    f"(skipped={payload['skipped']}).  This is how delivered "
                    "preflight evidence gets silently downgraded.  Use --force to "
                    "overwrite (the old report is backed up), or point --out "
                    "somewhere else."
                )
            backup = path.with_suffix(".json.superseded")
            backup.write_text(json.dumps(old, indent=1, ensure_ascii=False),
                              encoding="utf-8")
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage-What preflight (protocol 14)")
    ap.add_argument("--arm", default="T01", choices=ARM_IDS)
    # N-16: NOT the deliverable directory.  A default that overwrites delivered
    # evidence is a trap, and the reviewer fell into it before this was changed.
    ap.add_argument("--out", default=str(RUN_ROOT / "preflight"),
                    help="output directory (default: a scratch dir under RUN_ROOT; "
                         "pass the deliverable path explicitly to publish)")
    ap.add_argument("--no-data", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true",
                    help="allow overwriting a complete report with an incomplete one")
    args = ap.parse_args()
    rep = run_what_preflight(Path(args.out), arm=args.arm,
                             with_data=not args.no_data, limit=args.limit,
                             force=args.force)
    print(json.dumps(rep.to_dict(), indent=1, ensure_ascii=False))
    return 0 if rep.ok else 1


if __name__ == "__main__":                                       # pragma: no cover
    raise SystemExit(main())
