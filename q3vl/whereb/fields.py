"""From a predicted latent to ``s(p)`` and ``m(p)`` -- the Where-A analytic path.

Protocol 4.2 fixes the whole chain and Where-B only *predicts* its parameters:

    phi_dir(p) = [geo5, L, S, semantic_1..64]        (71-dim, per image)
    s_low(p)   = 3 tanh((w0 + alpha <phi_dir(p), w_dir>) / 3)
    s(p)       = one edge-aware guided upsample of the scalar s_low
    m(p)       = R(s(p); rho)

Two invariants this module exists to protect:

1. **The basis ``B`` is frozen.**  Protocol 3 says Where-B trains ``Q_where``,
   the connector and the heads -- not ``B``.  :class:`FrozenBasis` stores it as
   a *buffer*, so it cannot end up in an optimiser parameter group by accident,
   and it is loaded with a digest check against the ``BA-3-Joint`` artifact.
2. **Combine first, upsample once.**  ``guided_upsample`` in
   :mod:`q3vl.where.upsample` refuses multi-channel input; nothing here ever
   upsamples ``phi_dir``.

:func:`phi_dir_fast` is a diagnostics-free copy of
:func:`q3vl.where.phi.build_phi_dir`'s numeric path (the two 71x71 SVDs it runs
per call for the condition number are not free inside a training loop).
``tests/test_fields.py::test_phi_fast_matches_where_a`` asserts the two agree
bit-for-bit, so the fast path cannot drift from Where-A.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from q3vl.where.config import PhiConfig, RESID_BLOCK_DIM, UpsampleConfig
from q3vl.where.phi import (
    design_block,
    geo5_grid,
    range_channels,
    residualize,
    standardize_live,
)
from q3vl.where.readout import apply_readout
from q3vl.where.upsample import area_resize, guided_upsample, luma_guide

from .config import FPRE_DIM, PHI_DIR_DIM, S_SCALE, SEM_DIM, WHERE_A_BASIS_DIR

__all__ = ["FrozenBasis", "load_basis", "phi_dir_fast", "s_from_params",
           "predict_fields", "oracle_fields", "luma_guide", "area_resize"]


# --- the frozen projector ---------------------------------------------------

class FrozenBasis(nn.Module):
    """``B: 1024 -> 64`` held as a buffer, never trained (protocol 3)."""

    def __init__(self, weight: torch.Tensor, meta: dict[str, Any] | None = None):
        super().__init__()
        if tuple(weight.shape) != (SEM_DIM, FPRE_DIM):
            raise ValueError(
                f"B must be ({SEM_DIM}, {FPRE_DIM}), got {tuple(weight.shape)}"
            )
        self.register_buffer("weight", weight.detach().clone().float(), persistent=True)
        self.meta = dict(meta or {})

    def forward(self, fpre: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(fpre.to(self.weight.dtype), self.weight)

    def digest(self) -> str:
        w = self.weight.detach().cpu().contiguous().numpy()
        return hashlib.sha256(w.tobytes()).hexdigest()

    def facts(self) -> dict[str, Any]:
        return {
            "shape": list(self.weight.shape),
            "digest": self.digest(),
            "source": self.meta.get("source"),
            "arm": self.meta.get("arm"),
            "basis_json_sha256": self.meta.get("sha256"),
        }


def load_basis(arm: str = "BA-3-Joint", root: Path | None = None) -> FrozenBasis:
    """Load the calibrated Where-A basis and verify the published digest."""
    d = Path(root or WHERE_A_BASIS_DIR) / arm
    npy, js = d / "B.npy", d / "basis.json"
    if not npy.exists():
        raise FileNotFoundError(
            f"{npy} does not exist -- Where-A arm {arm} has not been calibrated yet. "
            "Where-B must not start on an uncalibrated basis (protocol 4.4)."
        )
    meta = json.loads(js.read_text()) if js.exists() else {}
    raw = npy.read_bytes()
    if meta.get("sha256") and hashlib.sha256(raw).hexdigest() != meta["sha256"]:
        raise RuntimeError(f"{npy} does not match the sha256 recorded in {js}")
    w = torch.from_numpy(np.load(npy, allow_pickle=False)).float()
    meta["source"] = str(npy)
    meta.setdefault("arm", arm)
    return FrozenBasis(w, meta)


# --- phi_dir ----------------------------------------------------------------

def phi_dir_fast(
    semantic_low: torch.Tensor,
    img_low: torch.Tensor,
    grid_h: int,
    grid_w: int,
    cfg: PhiConfig | None = None,
) -> torch.Tensor:
    """``(P, 71)``; numerically identical to ``build_phi_dir(...).phi_dir``."""
    cfg = cfg or PhiConfig()
    if semantic_low.dim() != 2 or semantic_low.shape[1] != cfg.sem_dim:
        raise ValueError(
            f"semantic_low must be (P,{cfg.sem_dim}), got {tuple(semantic_low.shape)}"
        )
    if semantic_low.shape[0] != grid_h * grid_w:
        raise ValueError(
            f"semantic_low has {semantic_low.shape[0]} rows, grid says {grid_h * grid_w}"
        )
    if tuple(img_low.shape) != (3, grid_h, grid_w):
        raise ValueError(f"img_low must be (3,{grid_h},{grid_w}), got {tuple(img_low.shape)}")

    dtype, device = semantic_low.dtype, semantic_low.device
    geo5 = geo5_grid(grid_h, grid_w, device=device, dtype=dtype, coord_mode=cfg.coord_mode)
    L, S = range_channels(img_low.to(dtype), luma=cfg.luma, saturation=cfg.saturation)
    if cfg.standardize_range:
        # Mirrors q3vl.where.phi.build_phi_dir exactly (Where-A changed this on
        # 2026-08-05): a range channel with no variation -- a greyscale photo
        # makes HSV saturation identically 0, and 3.3-3.8% of the local pool is
        # greyscale -- is zeroed rather than having its round-off amplified by
        # 1/(0 + eps) into a unit-variance "feature".
        # test_phi_fast_matches_where_a_bit_for_bit is what caught the drift.
        rng = torch.stack([L, S], dim=1)
        rng_z, _live = standardize_live(rng, rng.detach().std(dim=0, unbiased=False),
                                        cfg.std_eps)
        L, S = rng_z[:, 0], rng_z[:, 1]
    A = design_block(geo5, L, S)
    if A.shape[1] != RESID_BLOCK_DIM:
        raise AssertionError(f"design block is {A.shape[1]} wide, expected {RESID_BLOCK_DIM}")
    std_before = semantic_low.detach().std(dim=0, unbiased=False)
    sem = residualize(semantic_low, A, cfg.std_eps)[0] if cfg.residualize else semantic_low
    sem, _live = standardize_live(sem, std_before, cfg.std_eps)
    phi = torch.cat([geo5, L.unsqueeze(1), S.unsqueeze(1), sem], dim=1)
    if phi.shape[1] != PHI_DIR_DIM:
        raise AssertionError(f"phi_dir is {phi.shape[1]}-dim, protocol 4.2 says {PHI_DIR_DIM}")
    return phi


# --- the scalar field and the mask ------------------------------------------

def no_autocast(device_type: str):
    """Disable autocast for a block.  ``s_low`` must never be computed in bf16.

    Review blocker B4: ``phi_dir @ w_dir`` is a matmul, i.e. the first entry on
    autocast's bf16 allow-list, so wrapping the loss computation in
    ``torch.autocast`` silently demoted the one quantity this whole stage is
    about.  Measured at realistic scale (71-dim phi ~ N(0,1), ||w_dir||=1,
    alpha=2): ``max|ds| = 2.1e-2``, which for CBand12 near its sigma lower bound
    (0.025) moves ``exp(-0.5((z-mu)/sigma)^2)`` by up to ``|dm| = 4.2e-1`` --
    a systematic penalty on the four CBand12 arms and not on the four Band arms,
    i.e. straight into the §5.3 controlled comparison.  Evaluation ran without
    autocast, so training and the gate were not even measuring the same function.

    The guard lives *inside* :func:`s_from_params`, not at the call site, so no
    future caller can reintroduce the bug by adding an outer autocast.
    """
    return torch.autocast(device_type=device_type, enabled=False)


def s_from_params(
    phi_dir: torch.Tensor, w0: torch.Tensor, alpha: torch.Tensor, w_dir: torch.Tensor
) -> torch.Tensor:
    """``(P,)`` -- protocol 4.2, with the parameters *predicted* rather than fitted.

    The result is always exactly ``phi_dir.dtype``; see :func:`no_autocast`.
    """
    with no_autocast(phi_dir.device.type):
        q = w0.to(phi_dir.dtype) + alpha.to(phi_dir.dtype) * (
            phi_dir @ w_dir.to(phi_dir.dtype)
        )
        s = S_SCALE * torch.tanh(q / S_SCALE)
    if s.dtype != phi_dir.dtype:
        raise AssertionError(
            f"s_low came out as {s.dtype} but phi_dir is {phi_dir.dtype}; an "
            "autocast context leaked into the analytic path (review blocker B4)"
        )
    return s


def predict_fields(
    phi_dir: torch.Tensor,
    params: dict[str, torch.Tensor],
    readout: str,
    grid_h: int,
    grid_w: int,
    guide_hi: torch.Tensor | None = None,
    up_cfg: UpsampleConfig | None = None,
    require_dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    """One sample: raw predicted parameters -> ``s`` and ``m`` at both scales.

    ``params`` carries ``w0``, ``w_raw``, ``alpha_raw`` and the readout's raw
    entries, all unbatched.  **Everything is computed in the dtype of
    ``phi_dir``**, with autocast disabled for the whole body: the guided filter's
    box means and its ``1/(var + 1e-3)`` division, the readouts' exponentials and
    ``s_low`` itself are all unsafe in bf16, and ``s_low``'s matmul would
    otherwise be demoted by an outer autocast (review blocker B4).

    ``require_dtype`` turns the intended precision into a runtime assertion; the
    training and evaluation paths both pass ``torch.float32``, which is how the
    two are kept measuring the same function.
    """
    from q3vl.where.basis import alpha_of, w_dir_of

    dtype = phi_dir.dtype
    if require_dtype is not None and dtype != require_dtype:
        raise AssertionError(
            f"phi_dir is {dtype}, the caller requires {require_dtype} "
            "(review blocker B4: train and eval must share one precision)"
        )
    with no_autocast(phi_dir.device.type):
        w0 = params["w0"].to(dtype)
        w_dir = w_dir_of(params["w_raw"].to(dtype))
        alpha = alpha_of(params["alpha_raw"].to(dtype))
        rho = {k: v.to(dtype) for k, v in params.items()
               if k not in ("w0", "w_raw", "alpha_raw")}

        s_low = s_from_params(phi_dir, w0, alpha, w_dir)
        out: dict[str, torch.Tensor] = {
            "s_low": s_low,
            "m_low": apply_readout(readout, s_low, rho),
            "w_dir": w_dir,
            "alpha": alpha,
            "w0": w0,
        }
        if guide_hi is not None:
            s_map = s_low.reshape(1, 1, grid_h, grid_w)
            s_hi = guided_upsample(s_map, guide_hi.to(dtype), up_cfg)
            out["s_hi"] = s_hi
            out["m_hi"] = apply_readout(readout, s_hi.reshape(-1), rho).reshape(s_hi.shape)
    bad = {k: v.dtype for k, v in out.items() if v.dtype != dtype}
    if bad:
        raise AssertionError(f"analytic path left {dtype}: {bad} (review blocker B4)")
    return out


def oracle_fields(
    phi_dir: torch.Tensor,
    latent,
    readout: str,
    z: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """``s*`` on the ``F_pre`` grid and ``r*(z)`` on the protocol 5.5 z-grid.

    Recomputing ``s*`` from the published latent (instead of storing it) is what
    keeps the supervision consistent with whatever ``B`` the run is using: the
    same ``phi_dir`` feeds ``s_pred`` and ``s*``.
    """
    from q3vl.where.basis import alpha_of, w_dir_of

    dtype = phi_dir.dtype
    with no_autocast(phi_dir.device.type):
        w0 = latent.w0.to(dtype)
        w_dir = w_dir_of(latent.w_raw.to(dtype))
        alpha = alpha_of(latent.alpha_raw.to(dtype))
        rho = {k: v.to(dtype) for k, v in latent.rho.items()}
        return {
            "s_star": s_from_params(phi_dir, w0, alpha, w_dir),
            "r_star": apply_readout(readout, z.to(dtype), rho),
            "w_dir_star": w_dir,
        }
