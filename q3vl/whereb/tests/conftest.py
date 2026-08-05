"""Shared fixtures: a fake tokenizer, synthetic samples and a toy Qwen3-VL.

The synthetic sample is not noise: its GT mask is *generated from a known
latent* through the same analytic chain the model has to invert, so "the loss
goes down" in the mock closed loop means the parameter path actually carries
signal, not that the model learned a constant.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pytest
import torch

from q3vl.where.basis import Latent, canonicalize
from q3vl.where.readout import apply_readout, param_shapes
from q3vl.whereb.config import CBAND_M, arm_config
from q3vl.whereb.data import Batch
from q3vl.whereb.fields import phi_dir_fast, predict_fields
from q3vl.whereb.losses import curve_grid
from q3vl.whereb.qwhere import fpre_grid_positions

SPECIALS = ("<where>", "</where>", "<color>", "</color>")


class FakeTokenizer:
    """Whitespace + tag tokeniser.  Deterministic ids, stable across calls."""

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {t: i + 10 for i, t in enumerate(SPECIALS)}
        self._inv: dict[int, str] = {v: k for k, v in self._vocab.items()}
        self._next = 100
        self.eos_token_id = 2
        self.pad_token_id = 0

    def _id(self, tok: str) -> int:
        if tok not in self._vocab:
            self._vocab[tok] = self._next
            self._inv[self._next] = tok
            self._next += 1
        return self._vocab[tok]

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        # `[^<\s]+` rather than `\S+` so a tag glued to a word ("jellyfish</where>")
        # still splits into two tokens, the way a real BPE special token does.
        toks = re.findall(r"</?\w+>|[^<\s]+", text)
        return {"input_ids": [self._id(t) for t in toks]}

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        out = [self._inv.get(int(i), "?") for i in ids]
        if skip_special_tokens:
            out = [t for t in out if t not in SPECIALS]
        return " ".join(out)


@pytest.fixture()
def tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


# --- synthetic samples ------------------------------------------------------

@dataclass
class MockSample:
    sample_id: str
    phi_dir: torch.Tensor
    guide_hi: torch.Tensor
    mask_hi: torch.Tensor
    grid_h: int
    grid_w: int
    latent: Latent
    f_pre: torch.Tensor
    h_where: torch.Tensor
    is_global: bool = False


def random_latent(readout: str, seed: int, dtype=torch.float32) -> Latent:
    g = torch.Generator().manual_seed(seed)
    rho: dict[str, torch.Tensor] = {}
    for name, shape in param_shapes(readout).items():
        rho[name] = (torch.randn((), generator=g, dtype=dtype) if shape == ()
                     else torch.randn(CBAND_M, generator=g, dtype=dtype))
    if readout == "band":
        rho["pi_raw"] = torch.tensor(2.0, dtype=dtype)
        rho["k_raw"] = torch.tensor(0.0, dtype=dtype)
    return canonicalize(Latent(
        readout,
        torch.randn((), generator=g, dtype=dtype) * 0.3,
        torch.tensor(0.6, dtype=dtype),
        torch.randn(71, generator=g, dtype=dtype),
        rho,
    ))


def make_mock_sample(
    sample_id: str, readout: str, seed: int, grid_h: int = 6, grid_w: int = 8,
    upscale: int = 16, is_global: bool = False, text_len: int = 5,
) -> MockSample:
    g = torch.Generator().manual_seed(seed)
    f_pre = torch.randn(grid_h * grid_w, 1024, generator=g)
    img_low = torch.rand(3, grid_h, grid_w, generator=g)
    B = torch.randn(64, 1024, generator=g) / 32.0
    phi = phi_dir_fast(f_pre @ B.T, img_low, grid_h, grid_w)

    H, W = grid_h * upscale, grid_w * upscale
    guide = torch.rand(1, 1, H, W, generator=g)
    lat = random_latent(readout, seed + 1000)
    params = {"w0": lat.w0, "w_raw": lat.w_raw, "alpha_raw": lat.alpha_raw, **lat.rho}
    with torch.no_grad():
        f = predict_fields(phi, params, readout, grid_h, grid_w, guide_hi=guide)
    mask = torch.ones(H, W) if is_global else f["m_hi"].reshape(H, W).clone()
    return MockSample(
        sample_id=sample_id, phi_dir=phi, guide_hi=guide, mask_hi=mask,
        grid_h=grid_h, grid_w=grid_w, latent=lat, f_pre=f_pre,
        h_where=torch.randn(text_len, 2560, generator=g), is_global=is_global,
    )


def mock_batch(samples: list[MockSample], readout: str, modes: list[str] | None = None,
               with_oracle: bool = True) -> Batch:
    from q3vl.whereb.context import WhereContext
    from q3vl.whereb.fields import oracle_fields

    modes = modes or ["gt"] * len(samples)
    z = curve_grid()
    n_p = max(s.phi_dir.shape[0] for s in samples)
    n_t = max(s.h_where.shape[0] for s in samples)
    f_pre = torch.zeros(len(samples), n_p, 1024)
    f_pos = torch.zeros(len(samples), n_p, 2)
    f_mask = torch.zeros(len(samples), n_p, dtype=torch.bool)
    h = torch.zeros(len(samples), n_t, 2560)
    h_mask = torch.zeros(len(samples), n_t, dtype=torch.bool)
    targets, contexts = [], []
    for i, s in enumerate(samples):
        p = s.phi_dir.shape[0]
        f_pre[i, :p] = s.f_pre
        f_pos[i, :p] = fpre_grid_positions(s.grid_h, s.grid_w)
        f_mask[i, :p] = True
        t = s.h_where.shape[0]
        h[i, :t] = s.h_where
        h_mask[i, :t] = True
        tgt: dict[str, Any] = {
            "sample_id": s.sample_id, "phi_dir": s.phi_dir, "guide_hi": s.guide_hi,
            "grid_h": s.grid_h, "grid_w": s.grid_w, "mask_hi": s.mask_hi,
            # amendment A-5: the grid-level / centre-prior columns need the GT
            # projected onto the F_pre grid
            "mask_low": torch.nn.functional.interpolate(
                s.mask_hi[None, None], size=(s.grid_h, s.grid_w),
                mode="area")[0, 0],
            "is_global": s.is_global, "meta": {"render_mode":
                                               "global" if s.is_global else "local"},
            "has_oracle": False,
        }
        if with_oracle and not s.is_global:
            o = oracle_fields(s.phi_dir, s.latent, readout, z)
            tgt.update({"s_star": o["s_star"], "r_star": o["r_star"],
                        "w_dir_star": o["w_dir_star"], "has_oracle": True})
        targets.append(tgt)
        contexts.append(WhereContext(mode=modes[i], token_ids=[1] if modes[i] != "null" else [],
                                     provenance=s.sample_id))
    return Batch(
        inputs={"f_pre": f_pre, "f_pre_pos": f_pos, "f_pre_mask": f_mask,
                "h_where": h, "h_where_mask": h_mask},
        targets=targets, contexts=contexts,
        sample_ids=[s.sample_id for s in samples],
        meta=[t["meta"] for t in targets],
    )


@pytest.fixture()
def mock_samples_band() -> list[MockSample]:
    return [make_mock_sample(f"s{i}", "band", seed=i) for i in range(4)]


@pytest.fixture()
def mock_samples_cband() -> list[MockSample]:
    return [make_mock_sample(f"s{i}", "cband12", seed=100 + i) for i in range(4)]


# --- a toy Qwen3-VL (real class, tiny config) -------------------------------

@pytest.fixture(scope="session")
def toy_qwen3vl():
    """The real ``Qwen3VLForConditionalGeneration`` at ~1M params.

    Used to assert facts about transformers' behaviour (hidden-state layout,
    causal independence) on the actual class rather than on a stand-in.
    """
    transformers = pytest.importorskip("transformers")
    from pathlib import Path

    model_dir = Path("/home/bc/data/models/Qwen3-VL-4B-Instruct")
    if not (model_dir / "config.json").exists():
        pytest.skip("Qwen3-VL config not available")
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLForConditionalGeneration,
    )

    cfg = transformers.AutoConfig.from_pretrained(model_dir)
    t = cfg.text_config
    t.num_hidden_layers, t.hidden_size, t.intermediate_size = 2, 64, 128
    t.num_attention_heads, t.num_key_value_heads, t.head_dim = 4, 2, 16
    v = cfg.vision_config
    v.depth, v.hidden_size, v.intermediate_size, v.num_heads = 2, 32, 64, 2
    v.out_hidden_size, v.deepstack_visual_indexes = 64, [0]
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    return Qwen3VLForConditionalGeneration(cfg).eval()
