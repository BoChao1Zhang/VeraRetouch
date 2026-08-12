"""``AmortModel`` -- one geometry arm (P1 or P3') plus the shared semantic path.

The two arms are the same object with ``arm="P1"`` or ``arm="P3prime"``; every
other input, loss and criterion is identical, which is what makes the comparison
an A/B on the Phi-71 intermediate layer rather than on two separate systems.

Routing is the zero-training type-word rule, and only the **binary** decision
(semantic vs geometric) is ever taken from it: NOTES §3 measured that split at
396/396 = 100.0% on *generated* text, while the four-way split is only 74.2% --
and all 26% of that error is internal to the three geometric families, which
share one path and therefore cannot be harmed by it.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from q3vl.where.config import UpsampleConfig
from q3vl.where.upsample import guided_upsample
from q3vl.whereb.fields import no_autocast, predict_fields

from .heads import CondEncoder, P1Head, P3PrimeHead, SemanticHead, ShapeDistHead

__all__ = ["AmortModel", "ARMS"]

ARMS = ("P1", "P3prime", "SHAPE3")


class AmortModel(nn.Module):
    def __init__(
        self,
        arm: str = "P1",
        *,
        readout: str = "band",
        in_dim: int = 1024,
        ch: int = 128,
        n_blocks: int = 6,
        sem_ch: int = 96,
        cond_text_dim: int = 2560,
        cond_out: int = 256,
        n_words: int = 32,
        word_dim: int = 32,
        use_sim_field: bool = True,
        use_center_prior_channel: bool = False,
        with_semantic: bool = True,
        use_film: bool = True,
        pooled_w: bool = False,
        geom_inject: bool = False,
        geom_mode: str = "broadcast",
        pch_size: str = "full",
        seed: int = 0,
        upsample: UpsampleConfig | None = None,
        gate_upsample: bool = True,
    ):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
        self.arm = arm
        self.readout = readout
        self.use_sim_field = use_sim_field
        self.use_center_prior_channel = use_center_prior_channel
        self.upsample = upsample or UpsampleConfig()
        #: ADOPTED 2026-08-11 into the P3' inference recipe.  The guided
        #: upsampler is family-conditional: semantic keeps it, analytic families
        #: go to low-pass.  Both pre-registered criteria passed on V_where 400:
        #:   analytic  kappa~ 3.398 -> 0.080 (-97.6%, paired p=1e-4) at
        #:             IoU delta -0.0000 (p=0.99)   -- costs nothing
        #:   semantic  turning guidance OFF costs bF1 -0.107 (p=1e-4) and
        #:             IoU -0.0118 (p=8e-4)          -- so it is earning its keep
        #: Routing is the zero-training type-word rule (400/400 vs GT family).
        #: `gate_upsample=False` reproduces the pre-adoption behaviour and is
        #: kept so every board can carry the A column alongside.
        self.gate_upsample = gate_upsample
        # B2: the parsed geometry code enters as BROADCAST SPATIAL CHANNELS,
        # not through the pooled <where> hidden.  That is the whole point --
        # reading 1 showed perfect GT reasoning text buys <0.01 IoU through the
        # pooled/FiLM path, while reading 2 showed the text is right about shape
        # 82% of the time.  The information exists; the pathway loses it.
        #
        # B2-PCH (2026-08-12): the broadcast form above is the *lower bound*,
        # and it also cannot be resumed -- widening `stem.weight` by 21 input
        # channels is precisely what killed the first three B2 jobs.  PCH
        # instead adds a zero-initialised residual to the tower's penultimate
        # features, so the checkpoint's own tensors keep their shapes and the
        # arm starts bit-identical to the model it resumes from.
        from .geomparse import GEOM_DIM

        if geom_mode not in ("broadcast", "pch"):
            raise ValueError(f"unknown geom_mode {geom_mode!r}")
        self.geom_inject = geom_inject
        self.geom_mode = geom_mode if geom_inject else "broadcast"
        self.geom_code_dim = GEOM_DIM if geom_inject else 0
        #: channels concatenated onto the stem input -- zero in PCH mode, which
        #: is the whole reason PCH can be resumed
        self.geom_dim = GEOM_DIM if (geom_inject and self.geom_mode == "broadcast") else 0
        extra_ch = (int(use_sim_field) + int(use_center_prior_channel)
                    + self.geom_dim)

        self.cond = CondEncoder(cond_text_dim, n_words, word_dim, cond_out)
        cond_dim = self.cond.out_dim
        # review U2: the ablation switch has to reach the towers, or the
        # "-FiLM" row silently runs the main-arm configuration and gets
        # published as an ablation.  `film_dim=None` removes every FiLM layer.
        self.use_film = use_film
        film_dim = cond_dim if use_film else None
        if arm == "P1":
            self.geo = P1Head(readout, ch, in_dim, extra_ch, n_blocks, film_dim,
                              seed, pooled_w=pooled_w)
        elif arm == "SHAPE3":
            self.geo = ShapeDistHead(ch, in_dim, extra_ch, n_blocks, film_dim)
        else:
            self.geo = P3PrimeHead(ch, in_dim, extra_ch, n_blocks, film_dim)
        self.sem = (SemanticHead(in_dim, extra_ch, sem_ch, film_dim)
                    if with_semantic else None)

        # The injector lives on the GEOMETRY head only.  The semantic head is
        # supervised directly from `.cgt` and already sits at 0.820, and the
        # code's slots (shape / direction / extent) describe the analytic
        # families; injecting there would add a second module's worth of
        # parameters to a path the hypothesis is not about.  The cost is that
        # semantic-routed samples (17.5%) are untouched, so the headline Delta
        # is a *dilution-conservative* reading of the mechanism -- the boards
        # therefore carry the geometric-family subset alongside it.
        self.pch = None
        if geom_inject and self.geom_mode == "pch":
            from .pch import PCH, PCHConfig

            if pch_size not in ("full", "lite"):
                raise ValueError(f"unknown pch_size {pch_size!r}")
            maker = PCHConfig.full if pch_size == "full" else PCHConfig.lite
            self.pch_size = pch_size
            self.pch = PCH(maker(code_dim=GEOM_DIM, feat_dim=ch))

    # -- conditioning -------------------------------------------------------
    def cond_of(self, h_where, h_mask, word_ids, word_offsets) -> torch.Tensor:
        return self.cond(h_where, h_mask, word_ids, word_offsets)

    def _extra(self, sim: torch.Tensor | None, center: torch.Tensor | None,
               geom: torch.Tensor | None = None):
        chans = []
        if self.use_sim_field:
            if sim is None:
                raise ValueError("arm was built with use_sim_field=True but got none")
            chans.append(sim)
        if self.use_center_prior_channel:
            if center is None:
                raise ValueError("centre-prior channel requested but not supplied")
            chans.append(center)
        if self.geom_dim:
            if geom is None:
                raise ValueError("geom_inject=True but no geometry code supplied")
            ref = chans[0] if chans else None
            h, w = (ref.shape[-2:] if ref is not None else (1, 1))
            chans.append(geom.reshape(1, self.geom_dim, 1, 1).expand(1, -1, h, w))
        return torch.cat(chans, dim=1) if chans else None

    def _inject_of(self, geom: torch.Tensor | None):
        """Injection hook for the geometry head, or None.

        Returns None -- i.e. an exact no-op, not a zero tensor added -- whenever
        the arm is not in PCH mode or the sample carried no code at all, so the
        "no geometry named" fallback is structural rather than numerical.
        """
        if self.pch is None or geom is None:
            return None
        return lambda codes: self.pch(codes, geom)

    # -- one sample ---------------------------------------------------------
    def forward_geo(
        self,
        feat: torch.Tensor,                 # (1, 1024, gh, gw)
        cond: torch.Tensor,                 # (1, cond_dim)
        phi_dir: torch.Tensor,              # (P, 71) -- P1 only
        *,
        sim: torch.Tensor | None = None,
        center: torch.Tensor | None = None,
        geom: torch.Tensor | None = None,
        guide_hi: torch.Tensor | None = None,
        grid_h: int = 0,
        grid_w: int = 0,
    ) -> dict[str, Any]:
        extra = self._extra(sim, center, geom)
        inject = self._inject_of(geom)
        if self.arm == "P1":
            params = self.geo(feat, extra, cond, phi_dir, inject)
            # Everything from phi_dir onward is float32 with autocast disabled --
            # `phi_dir @ w_dir` is a matmul and an outer autocast would silently
            # demote the one quantity this stage is about (review blocker B4).
            fields = predict_fields(
                phi_dir.float(), {k: v.float() for k, v in params.items()},
                self.readout, grid_h, grid_w,
                guide_hi=guide_hi.float() if guide_hi is not None else None,
                up_cfg=self.upsample, require_dtype=torch.float32,
            )
            out = {"m_low": fields["m_low"].reshape(grid_h, grid_w),
                   "s_low": fields["s_low"].reshape(grid_h, grid_w),
                   "params": params}
            if "m_hi" in fields:
                out["m_hi"] = fields["m_hi"]
                out["s_hi"] = fields["s_hi"]
            if guide_hi is not None and self.gate_upsample:
                # analytic path -> low-pass (the adopted gate)
                s_bl = F.interpolate(fields["s_low"].reshape(1, 1, grid_h, grid_w),
                                     size=guide_hi.shape[-2:], mode="bilinear",
                                     align_corners=False)
                rho = {k: v.float() for k, v in params.items()
                       if k not in ("w0", "w_raw", "alpha_raw")}
                from q3vl.where.readout import apply_readout

                out["m_hi"] = apply_readout(self.readout, s_bl.reshape(-1), rho
                                            ).reshape(s_bl.shape)
                out["s_hi"] = s_bl
            return out

        if self.arm == "SHAPE3":
            # distance-field reparameterisation: the field is free, the mask is
            # a thresholded profile of it, so iso-contour regularity is a
            # property of the parameterisation rather than of the loss.
            with no_autocast(feat.device.type):
                s_low, t, width = self.geo(
                    feat.float(), None if extra is None else extra.float(),
                    cond.float(), inject)
                out = {"m_low": self.geo.mask_of(s_low, t, width)[0, 0],
                       "s_low": s_low[0, 0], "params": {},
                       "s_raw": s_low, "level": t, "width": width}
                if guide_hi is not None:
                    s_hi = F.interpolate(s_low, size=guide_hi.shape[-2:],
                                         mode="bilinear", align_corners=False)
                    out["s_hi"] = s_hi
                    out["m_hi"] = self.geo.mask_of(s_hi, t, width)
            return out

        # P3': no Phi anywhere on this path.
        with no_autocast(feat.device.type):
            s_low = self.geo(feat.float(), None if extra is None else extra.float(),
                             cond.float(), inject)              # (1,1,gh,gw) in (-3,3)
            out = {"m_low": self.geo.mask_of(s_low)[0, 0],
                   "s_low": s_low[0, 0], "params": {}}
            if guide_hi is not None:
                if self.gate_upsample:
                    # analytic path -> low-pass (adopted gate); see gate_upsample
                    s_hi = F.interpolate(s_low, size=guide_hi.shape[-2:],
                                         mode="bilinear", align_corners=False)
                else:
                    s_hi = guided_upsample(s_low, guide_hi.float(), self.upsample)
                out["s_hi"] = s_hi
                out["m_hi"] = self.geo.mask_of(s_hi)
        return out

    def forward_sem(
        self, feat: torch.Tensor, cond: torch.Tensor, *,
        sim: torch.Tensor | None = None, center: torch.Tensor | None = None,
        geom: torch.Tensor | None = None,
        guide_hi: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if self.sem is None:
            raise RuntimeError("this arm was built without a semantic head")
        extra = self._extra(sim, center, geom)
        with no_autocast(feat.device.type):
            s = self.sem(feat.float(), None if extra is None else extra.float(),
                         cond.float())
            out = {"m_low": self.sem.mask_of(s)[0, 0], "s_low": s[0, 0]}
            if guide_hi is not None:
                s_hi = guided_upsample(s, guide_hi.float(), self.upsample)
                out["m_hi"] = self.sem.mask_of(s_hi)
                out["s_hi"] = s_hi
        return out

    # -- reporting ----------------------------------------------------------
    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def facts(self) -> dict[str, Any]:
        groups: dict[str, int] = {}
        for name, p in self.named_parameters():
            groups[name.split(".")[0]] = groups.get(name.split(".")[0], 0) + p.numel()
        return {
            "arm": self.arm,
            "readout": self.readout,
            "n_trainable_params": self.n_trainable(),
            "params_by_group": groups,
            "use_sim_field": self.use_sim_field,
            "use_film": self.use_film,
            "gate_upsample": self.gate_upsample,
            "geom_inject": self.geom_inject,
            "geom_mode": self.geom_mode,
            "geom_dim": self.geom_dim,
            "geom_code_dim": self.geom_code_dim,
            "pch": (self.pch.facts() if self.pch is not None else None),
            "pooled_w": bool(getattr(self.geo, "pooled_w", False)),
            "use_center_prior_channel": self.use_center_prior_channel,
            "has_semantic_head": self.sem is not None,
            "cond_dim": self.cond.out_dim,
            "upsample": {"radius_low": self.upsample.radius_low,
                         "eps": self.upsample.eps,
                         "domain": list(self.upsample.domain)},
        }


def upsample_sim_to_grid(sim32: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    """H/32 similarity field -> the H/16 grid the tower runs on.

    Bilinear, because this is a *conditioning* channel and not a mask being
    scored: the visualisation red line ("overlaying a grid field on the image
    must use the exact inverse map, never a resize") governs fields that enter
    criteria or figures, and this one does neither.
    """
    return F.interpolate(sim32.reshape(1, 1, *sim32.shape[-2:]),
                         size=(grid_h, grid_w), mode="bilinear", align_corners=False)
