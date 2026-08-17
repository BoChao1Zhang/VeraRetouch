"""Colour-query sampling.  Frozen: ``B = 32`` samples x ``Q = 256`` colours.

Frozen block item 2: a step carries **8192 colours**, split ``32 x 256``.  The
8192 is CGLUT's own colour batch (GLUT App A.1: "trained for 40 epochs with a
batch size of 8192"); the split into (samples, colours-per-sample) is the
campaign's, and it is frozen so the six arms are step-matched (U4).

Training colours are the ``128^3`` uniform subset of the 8-bit RGB cube
(App A.1: "uniformly sample the full 8-bit RGB space to construct a 128^3
training set, reserving the remaining colours for evaluation"), implemented as
the even 8-bit levels ``{0, 2, ..., 254} / 255`` on each axis (EPR-024 §3.3).
:meth:`QuerySampler.sample_heldout` draws from the complement, which is the
"unseen colours" evaluation column of §4.B.

**RNG discipline** (the where-side N1 lesson): every draw comes from a private
:class:`torch.Generator`; the global stream is never touched, so adding or
removing a query draw cannot shift any other random decision in the run.  The
generator is a CPU one and the colours are moved to the device afterwards, so a
run is bit-identical on CPU and GPU.
"""

from __future__ import annotations

from typing import Any

import torch

__all__ = [
    "BATCH_SAMPLES",
    "QUERIES_PER_SAMPLE",
    "COLORS_PER_STEP",
    "TRAIN_LEVELS",
    "DEFAULT_SEED",
    "QuerySampler",
    "train_color_levels",
    "uniform_grid",
    "image_histogram_colors",
    "mining_ratio",
    "select_hard",
]

#: frozen block: B = 32 samples, Q = 256 colours, B * Q = 8192 colours per step
BATCH_SAMPLES = 32
QUERIES_PER_SAMPLE = 256
COLORS_PER_STEP = BATCH_SAMPLES * QUERIES_PER_SAMPLE
assert COLORS_PER_STEP == 8192

#: 128 uniform levels per axis = the training colour set (GLUT App A.1)
TRAIN_LEVELS = 128
#: repository-wide seed (EPR-024 §3.4)
DEFAULT_SEED = 20260810

#: the two implemented query distributions.  ``uniform128`` is the frozen main
#: arm; ``image_hist`` is EPR-029's ablation row (``:575``: colours drawn from
#: the alpha-weighted 5-bit histogram of the image) and needs per-sample colours
#: handed in, so the sampler only provides the draw, not the histogram.
QUERY_MODES: tuple[str, ...] = ("uniform128", "image_hist")


def train_color_levels(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """The 128 training levels: ``{0, 2, ..., 254} / 255``."""
    return torch.arange(0, 256, 2, dtype=dtype) / 255.0


def heldout_color_levels(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """The 128 evaluation levels: ``{1, 3, ..., 255} / 255`` (the complement)."""
    return torch.arange(1, 256, 2, dtype=dtype) / 255.0


def uniform_grid(n: int = 17, dtype: torch.dtype = torch.float32,
                 device: Any = "cpu") -> torch.Tensor:
    """``(n^3, 3)`` uniform sRGB grid, levels ``i / (n - 1)``.

    ``n = 17`` is ``X_grid`` of the function-value metric (§4.B); ``n = 9`` is
    the grid the pre-registered baseline floors were measured on (§4.C).
    """
    if n < 2:
        raise ValueError(f"grid needs n >= 2, got {n}")
    ax = torch.linspace(0.0, 1.0, n, dtype=dtype, device=device)
    r, g, b = torch.meshgrid(ax, ax, ax, indexing="ij")
    return torch.stack((r.reshape(-1), g.reshape(-1), b.reshape(-1)), dim=-1)


class QuerySampler:
    """Q colours per sample, from a private generator.

    ``sampler.sample(b)`` -> ``(b, Q, 3)`` on ``device``; with the defaults that
    is ``(32, 256, 3)`` = the frozen 8192 colours of one step.
    """

    def __init__(self, *, seed: int = DEFAULT_SEED, q: int = QUERIES_PER_SAMPLE,
                 mode: str = "uniform128", levels: int = TRAIN_LEVELS):
        if mode not in QUERY_MODES:
            raise ValueError(f"unknown query mode {mode!r}; expected {QUERY_MODES}")
        self.seed = int(seed)
        self.q = int(q)
        self.mode = mode
        self.levels = int(levels)
        #: private stream -- never ``torch.manual_seed``, never the global RNG
        self.generator = torch.Generator().manual_seed(int(seed))
        self.n_draws = 0
        self.n_colors = 0

    # -- draws --------------------------------------------------------------
    def _levels_tensor(self, heldout: bool, dtype: torch.dtype) -> torch.Tensor:
        return (heldout_color_levels(dtype) if heldout
                else train_color_levels(dtype))

    def sample(self, b: int = BATCH_SAMPLES, q: int | None = None, *,
               device: Any = "cpu", dtype: torch.dtype = torch.float32,
               heldout: bool = False) -> torch.Tensor:
        """``(b, q, 3)`` colours drawn uniformly from the 128^3 grid."""
        q = self.q if q is None else int(q)
        if self.mode != "uniform128":
            raise ValueError(
                f"mode={self.mode!r} does not draw from the uniform grid; use "
                "sample_from_pool() with the per-sample colours")
        lv = self._levels_tensor(heldout, dtype)
        idx = torch.randint(0, lv.numel(), (b, q, 3), generator=self.generator)
        out = lv[idx]
        self.n_draws += 1
        self.n_colors += b * q
        return out.to(device=device)

    def sample_heldout(self, b: int = BATCH_SAMPLES, q: int | None = None,
                       **kw) -> torch.Tensor:
        """Colours from the complement of the training set (§4.B unseen-colour
        column: "训练采 128³ 均匀色，评测在其补集上算")."""
        return self.sample(b, q, heldout=True, **kw)

    def sample_from_pool(self, pool: torch.Tensor, q: int | None = None, *,
                         weights: torch.Tensor | None = None) -> torch.Tensor:
        """``(q, 3)`` drawn from a per-sample colour pool (``image_hist`` mode).

        ``pool`` is ``(K, 3)`` and ``weights`` the matching frequencies; drawing
        is with replacement, from this sampler's private generator.
        """
        if pool.dim() != 2 or pool.shape[-1] != 3:
            raise ValueError(f"pool must be (K, 3), got {tuple(pool.shape)}")
        q = self.q if q is None else int(q)
        if weights is None:
            idx = torch.randint(0, pool.shape[0], (q,), generator=self.generator)
        else:
            w = weights.detach().to(dtype=torch.float32).clamp_min(0).cpu()
            idx = torch.multinomial(w, q, replacement=True,
                                    generator=self.generator)
        self.n_draws += 1
        self.n_colors += q
        return pool[idx.to(pool.device)]

    # -- bookkeeping --------------------------------------------------------
    def state(self) -> torch.Tensor:
        return self.generator.get_state()

    def load_state(self, state: torch.Tensor) -> None:
        self.generator.set_state(state)

    def facts(self) -> dict[str, Any]:
        return {"seed": self.seed, "q": self.q, "mode": self.mode,
                "levels": self.levels, "batch_samples": BATCH_SAMPLES,
                "colors_per_step": COLORS_PER_STEP,
                "n_draws": self.n_draws, "n_colors": self.n_colors,
                "rng": "private torch.Generator (global stream untouched)"}


def image_histogram_colors(img: torch.Tensor, *, bits: int = 5,
                           top_k: int = 4096,
                           alpha: torch.Tensor | None = None
                           ) -> tuple[torch.Tensor, torch.Tensor]:
    """``(colors, weights)`` -- the ``X_img`` measure of §4.B.

    ``img`` is ``(3, H, W)`` in [0,1].  Colours are quantised to ``bits`` per
    channel (5 bits per the criterion), counted, and the ``top_k`` most frequent
    are returned with their (normalised) frequencies.  ``alpha`` weights the
    count by the edit mask, which is EPR-029's ablation form.

    Everything happens on the image's device.
    """
    if img.dim() != 3 or img.shape[0] != 3:
        raise ValueError(f"img must be (3, H, W), got {tuple(img.shape)}")
    n = 1 << int(bits)
    flat = img.reshape(3, -1).transpose(0, 1)                      # (P, 3)
    qi = (flat.clamp(0, 1) * (n - 1)).round().to(torch.int64)      # (P, 3)
    key = (qi[:, 0] * n + qi[:, 1]) * n + qi[:, 2]
    if alpha is None:
        w = torch.ones(key.shape[0], device=img.device, dtype=torch.float32)
    else:
        w = alpha.reshape(-1).to(device=img.device, dtype=torch.float32)
        if w.numel() != key.numel():
            raise ValueError("alpha does not match the image's pixel count")
    counts = torch.zeros(n ** 3, device=img.device, dtype=torch.float32)
    counts.scatter_add_(0, key, w)
    k = min(int(top_k), int((counts > 0).sum().item()))
    vals, idx = torch.topk(counts, k)
    b = idx % n
    g = (idx // n) % n
    r = idx // (n * n)
    colors = torch.stack((r, g, b), dim=-1).to(img.dtype) / (n - 1)
    return colors, vals / vals.sum().clamp_min(1e-12)


def mining_ratio(epoch: int | float, *, start_epoch: int = 5, end_epoch: int = 20,
                 r_start: float = 0.10, r_end: float = 0.40) -> float:
    """Hard-example mining ratio ``r`` (GLUT App A.1: epoch 5 -> 20, 10% -> 40%).

    Ruling 11.1-4 fixes the granularity: within-batch top-r resampling of colour
    query points, no cross-step state.  Below ``start_epoch`` it is ``r_start``,
    above ``end_epoch`` it is ``r_end``, linear in between.
    """
    e = float(epoch)
    if e <= start_epoch:
        return float(r_start)
    if e >= end_epoch:
        return float(r_end)
    t = (e - start_epoch) / float(end_epoch - start_epoch)
    return float(r_start + t * (r_end - r_start))


def select_hard(err: torch.Tensor, ratio: float) -> torch.Tensor:
    """Indices of the ``ratio`` fraction of colours with the largest error.

    ``err`` is ``(N,)`` per-colour L1, already computed on the device -- and the
    ``topk`` runs there too.  (CPU and CUDA break ties differently; moving the
    tensor first is how the where side lost 0.296 of an IoU.)
    """
    if err.dim() != 1:
        raise ValueError(f"err must be (N,), got {tuple(err.shape)}")
    k = int(round(float(ratio) * err.numel()))
    k = max(0, min(k, err.numel()))
    if k == 0:
        return err.new_empty((0,), dtype=torch.long)
    return torch.topk(err, k, largest=True, sorted=False).indices
