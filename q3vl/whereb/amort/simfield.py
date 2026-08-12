"""The spatial conditioning channel: ``merger_out . E[subject noun]``.

This is the only channel in the campaign with *demonstrated* word specificity
(P-W5: four readouts all PASS, Delta=+0.021, p=1.4e-7, "tree" lights up on the
tree).  It is also, on its own, **worse than a zero-parameter centre prior**
(-0.065, p=6.3e-11, E5 §3.4) and collapses below the random floor on small
targets.  Both facts are why it enters here as a *conditioning input* and never
as a target or a selection criterion (RESEARCH §4 general principle; the E5/P4
prior-field prohibition).

Two contracts this module exists to keep
----------------------------------------

**1. Arm-constant normalisation, never per image.**  The s-cache contract makes
per-image min-max/softmax a red line, and for a good reason here: the whole
point of the channel is that cell *A* is brighter than cell *B* in the same
image *and* across images.  A per-image squash destroys exactly that.  The
centre/scale below are robust statistics over every valid cell of the whole arm,
computed once by :func:`fit_norm` and then frozen into the run config.

**2. The producing attention kernel is part of the domain.**  Measured on
2026-08-10 while validating :class:`~q3vl.where.fpre.MergerHook`: the same image
through the same weights gives merger outputs that agree to 3e-9 under a
matching kernel but only to a relative max of **0.12** (corr 0.999) between
``eager`` and ``sdpa`` in bf16.  A normalisation fitted under one kernel is
therefore not valid under another, so :class:`SimFieldNorm` records the kernel
and :meth:`SimFieldNorm.assert_compatible` refuses a mismatch rather than
silently shifting every field in the run.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

__all__ = ["SimFieldNorm", "subject_nouns", "WordEmbedder", "similarity_field",
           "gaussian_soften", "fit_norm"]


# Re-exported so callers cannot drift from the parser P-W5 validated at 400/400.
from q3vl.whereb.scripts.run_pw5_fpresim import subject_nouns  # noqa: E402


def gaussian_soften(field: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian with **reflect** padding.

    SAMRefiner's lesson is that a raw/binary prior reads badly and a softened one
    reads well.  The padding mode is not a detail: zero padding darkens the frame
    edge, which manufactures a vignette -- i.e. hands the head a free centre
    prior, the exact confound this campaign is policed for (E3: the trained heads
    already correlate 0.64 with the centre prior).
    """
    if sigma <= 0:
        return field
    rad = max(1, int(round(3 * sigma)))
    t = torch.arange(-rad, rad + 1, dtype=field.dtype, device=field.device)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    x = field.reshape(1, 1, *field.shape[-2:])
    x = torch.nn.functional.pad(x, (rad, rad, 0, 0), mode="reflect")
    x = torch.nn.functional.conv2d(x, k.reshape(1, 1, 1, -1))
    x = torch.nn.functional.pad(x, (0, 0, rad, rad), mode="reflect")
    x = torch.nn.functional.conv2d(x, k.reshape(1, 1, -1, 1))
    return x.reshape(field.shape)


class WordEmbedder:
    """``word -> E[tok(" "+word)].mean(0)`` off the frozen input embedding.

    ``tie_word_embeddings=True`` is already verified for this checkpoint, and
    E5 reads the matrix straight out of the safetensors shard on CPU so that a
    pure-CPU job never has to occupy a card just to look up one tensor.
    """

    def __init__(self, embeddings: torch.Tensor, tokenizer):
        self.E = embeddings
        self.tok = tokenizer
        self._cache: dict[str, torch.Tensor] = {}

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path, tokenizer) -> "WordEmbedder":
        from safetensors import safe_open

        checkpoint = Path(checkpoint)
        key = "model.language_model.embed_tokens.weight"
        idx_path = checkpoint / "model.safetensors.index.json"
        if idx_path.exists():
            shard = json.loads(idx_path.read_text())["weight_map"][key]
        else:
            shard = "model.safetensors"
        with safe_open(checkpoint / shard, framework="pt", device="cpu") as f:
            return cls(f.get_tensor(key).float(), tokenizer)

    def __call__(self, word: str) -> torch.Tensor:
        if word not in self._cache:
            ids = self.tok(" " + word, add_special_tokens=False)["input_ids"]
            if not ids:
                raise ValueError(f"{word!r} tokenised to nothing")
            self._cache[word] = self.E[torch.tensor(ids)].mean(0)
        return self._cache[word]


def similarity_field(
    merger_out: torch.Tensor, emb: torch.Tensor, kind: str = "dot"
) -> torch.Tensor:
    """``(gh32, gw32)`` raw similarity -- **un-normalised, un-softened**.

    ``dot`` is the default because it is the readout that scored best in both
    P-W5 and E5 (0.4478 vs 0.3950 decoded); ``cosine`` stays available so the
    choice remains an experiment rather than an assumption.
    """
    m = merger_out.reshape(-1, merger_out.shape[-1]).float()
    e = emb.float()
    if kind == "dot":
        v = m @ e
    elif kind == "cosine":
        v = (torch.nn.functional.normalize(m, dim=-1)
             @ torch.nn.functional.normalize(e, dim=0))
    else:
        raise ValueError(f"unknown similarity kind {kind!r}")
    return v.reshape(merger_out.shape[0], merger_out.shape[1])


@dataclass
class SimFieldNorm:
    """The frozen, arm-wide squash + the domain declaration it is valid under.

    ``meta.norm`` in the s-cache contract's sense: a producer must state the
    measured domain of the whole arm and a consumer must assert that the raw
    data it is handed actually lives there.  Both halves are here because in
    this campaign producer and consumer are the same run -- which is precisely
    the situation in which the assertion is usually skipped and the failure is
    silent (contract: "the second failure mode is silent").
    """

    kind: str = "dot"
    center: float = 0.0
    scale: float = 1.0
    gain: float = 1.0
    sigma: float = 1.0
    raw_min: float = float("nan")
    raw_max: float = float("nan")
    raw_p001: float = float("nan")
    raw_p999: float = float("nan")
    n_cells: int = 0
    n_samples: int = 0
    attn_implementation: str = ""
    checkpoint: str = ""
    #: consumer-side expectation.  squash() output is a probability-like field.
    out_domain: tuple[float, float] = (0.0, 1.0)
    notes: str = (
        "arm-constant robust median/MAD squash; per-image normalisation is a "
        "red line.  Valid only under the recorded attention kernel: eager vs "
        "sdpa move the merger output by rel-max 0.12 in bf16 (measured "
        "2026-08-10, merger_hook_check.json)."
    )

    def squash(self, v: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gain * (v - self.center) / max(self.scale, 1e-9))

    def assert_compatible(self, attn_implementation: str, checkpoint: str = "") -> None:
        if self.attn_implementation and attn_implementation != self.attn_implementation:
            raise AssertionError(
                f"similarity-field norm was fitted under attention kernel "
                f"{self.attn_implementation!r} but this run uses "
                f"{attn_implementation!r}; the merger output differs by up to "
                "rel-max 0.12 between kernels, so the frozen centre/scale do not "
                "apply.  Refit the norm or switch the kernel."
            )
        if checkpoint and self.checkpoint and checkpoint != self.checkpoint:
            raise AssertionError(
                f"norm fitted on {self.checkpoint!r}, run uses {checkpoint!r}"
            )

    #: A consumer-side check has to catch the failure it exists for, and that
    #: failure is a **distribution shift** -- the norm being applied under a
    #: different attention kernel, a different checkpoint, or a mis-parsed noun.
    #: It is NOT tail thickness: the similarity field is genuinely heavy-tailed
    #: (P-W5's high-norm "register" cells), and ~2% of cells legitimately sit
    #: beyond 8 MAD, so a tail-fraction ceiling would fire on healthy data and
    #: teach everyone to raise it -- the classic way an assertion becomes noise.
    #: So: the centre must not move (shift detector), and only gross corruption
    #: trips the tail check.
    max_center_shift_mad: float = 4.0
    gross_tol_mad: float = 12.0
    max_frac_outside_gross: float = 0.05

    def assert_in_domain(self, v: torch.Tensor, *, tol: float = 8.0,
                         enforce: bool = True) -> dict[str, float]:
        """Consumer-side check that the RAW field lives where the producer said.

        Not a clamp: clamping is the silent failure the s-cache contract names
        ("the field is still inside the anchor domain, the orphan check stays
        quiet, but the s axis is gone").  It **raises**.

        Review B2: the first version computed ``frac_outside_tol`` and merely
        returned it, so the contract's "the consumer must assert the raw data
        really lives there" had the form of an assertion and none of the force --
        the number landed in a JSON field nobody reads, which is exactly the
        silent-failure mode the contract was written against.
        """
        scale = max(self.scale, 1e-9)
        lo, hi = self.center - tol * scale, self.center + tol * scale
        vmin, vmax = float(v.min()), float(v.max())
        if not np.isfinite([vmin, vmax]).all():
            raise AssertionError("similarity field contains non-finite cells")
        med = float(v.median())
        shift = abs(med - self.center) / scale
        frac_out = float(((v < lo) | (v > hi)).float().mean())
        g_lo = self.center - self.gross_tol_mad * scale
        g_hi = self.center + self.gross_tol_mad * scale
        frac_gross = float(((v < g_lo) | (v > g_hi)).float().mean())
        rep = {"raw_min": vmin, "raw_max": vmax, "raw_median": med,
               "center_shift_mad": shift, "frac_outside_tol": frac_out,
               "frac_outside_gross": frac_gross, "tol_lo": lo, "tol_hi": hi,
               "violated": bool(shift > self.max_center_shift_mad
                                or frac_gross > self.max_frac_outside_gross)}
        if enforce and rep["violated"]:
            raise AssertionError(
                f"similarity field is not where the producer declared: median "
                f"{med:.4f} is {shift:.2f} MAD from the fitted centre "
                f"{self.center:.4f} (ceiling {self.max_center_shift_mad}), and "
                f"{frac_gross:.3%} of cells lie beyond {self.gross_tol_mad} MAD "
                f"(ceiling {self.max_frac_outside_gross:.1%}). The arm-wide norm "
                f"was fitted under kernel {self.attn_implementation!r} on "
                f"{self.checkpoint!r} -- refit it or fix the producer."
            )
        return rep

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["out_domain"] = list(self.out_domain)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SimFieldNorm":
        d = dict(d)
        d["out_domain"] = tuple(d.get("out_domain", (0.0, 1.0)))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def fit_norm(
    values: Sequence[np.ndarray] | Iterable[np.ndarray],
    *,
    kind: str = "dot",
    gain: float = 1.0,
    sigma: float = 1.0,
    attn_implementation: str = "",
    checkpoint: str = "",
    n_samples: int = 0,
) -> SimFieldNorm:
    """Robust median/MAD over **every valid cell of the whole arm**."""
    allv = np.concatenate([np.asarray(v).ravel() for v in values])
    med = float(np.median(allv))
    mad = float(np.median(np.abs(allv - med))) * 1.4826
    return SimFieldNorm(
        kind=kind, center=med, scale=max(mad, 1e-9), gain=gain, sigma=sigma,
        raw_min=float(allv.min()), raw_max=float(allv.max()),
        raw_p001=float(np.percentile(allv, 0.1)),
        raw_p999=float(np.percentile(allv, 99.9)),
        n_cells=int(allv.size), n_samples=int(n_samples),
        attn_implementation=attn_implementation, checkpoint=str(checkpoint),
    )
