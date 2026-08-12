"""Eager per-head attention readout: ``<where>`` text rows -> image token columns.

This is the read side of probe **PR-ATT1-E1** (PROPOSAL 2026-08-10 section 2.2
steps 2/4/5, card P-W1 + P-W3 raw arm).  It answers one question and nothing
else: *given a teacher-forced ``prompt + <where>...</where>`` sequence, what does
each (layer, head) attention row over the image tokens look like?*

Four facts were re-verified on 2026-08-10 against the installed transformers
4.57.1 and the local checkpoint-4976 ``config.json`` -- none of them is quoted
from memory, all are asserted at runtime by :func:`assert_model_facts`:

1. ``Qwen3VLTextAttention.forward`` returns ``(attn_output, attn_weights)`` and
   picks ``eager_attention_forward`` **iff** ``config._attn_implementation ==
   "eager"``; SDPA/FA2 hand back ``None`` for the weights.  So the tap below
   registers a plain ``forward_hook`` on each attention module and needs no
   ``output_attentions=True`` -- but it *does* need eager, and it raises rather
   than falling back (red line).
2. Text stack: 36 layers, 32 query heads, 8 KV heads.  ``eager_attention_forward``
   calls ``repeat_kv`` **before** the matmul, so ``attn_weights`` already carries
   32 rows per layer; there is no GQA folding left for us to undo.
3. Image tokens are the ``151655`` (``<|image_pad|>``) positions, laid out
   row-major on the **merged** grid ``(out_h/32, out_w/32)`` -- ``patch_size=16``
   times ``spatial_merge_size=2``.  ``F_pre`` lives on the finer ``out/16`` grid;
   the two must never be confused, which is why :class:`AttnFields` carries the
   merged grid explicitly.
4. deepstack injects its three vision features at **language layers 0, 1, 2**
   (``layer_idx in range(len(deepstack_visual_embeds))`` in
   ``Qwen3VLTextModel.forward``).  The ``[5, 11, 17]`` in ``config.vision_config``
   are the *vision tower* block indices the features are taken **from**.  The
   PROPOSAL's "deepstack injects at LLM layers 5/11/17" is therefore wrong, and
   the preregistered "drop the first 2 layers" filter sits inside the injection
   zone rather than after it.  Recorded here because a layer-scan conclusion that
   assumed the wrong injection depth would be uninterpretable.

Nothing in this module thresholds, normalises per image, or computes a
criterion: it returns raw fields plus the masks needed to read them.
:mod:`q3vl.whereb.metrics` owns the numbers, :mod:`q3vl.whereb.viz` owns the
colours.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import numpy as np
import torch

__all__ = [
    "IMAGE_TOKEN_ID", "WHERE_OPEN_ID", "WHERE_CLOSE_ID", "VISION_END_ID", "IM_END_ID",
    "POOL_NAMES", "PREREGISTERED_POOLS", "SINK_MAD_K",
    "QueryPools", "AttnFields", "AttentionTap",
    "assert_model_facts", "locate_image_columns", "locate_query_pools",
    "sink_mask_from_profile", "merged_grid",
]

# --- token ids: read from checkpoint-4976 added_tokens.json / config.json ----
IMAGE_TOKEN_ID = 151655        # <|image_pad|>
VISION_START_ID = 151652       # <|vision_start|>
VISION_END_ID = 151653         # <|vision_end|>
IM_END_ID = 151645             # <|im_end|>
WHERE_OPEN_ID = 151669         # <where>
WHERE_CLOSE_ID = 151670        # </where>

#: patch_size(16) * spatial_merge_size(2); image tokens live on ``out/32``.
IMAGE_TOKEN_STRIDE = 32

#: The three preregistered query pools of PR-ATT1-E1.  ``where_close`` is carried
#: as an exploratory fourth column and is deliberately **not** in the FWER
#: family -- adding it after the fact would be the "enumerate until something
#: survives" move that DELTA ruling 4 forbids.
PREREGISTERED_POOLS = ("where_special", "where_content", "instr_text")
POOL_NAMES = PREREGISTERED_POOLS + ("where_close",)

#: RO9b's D-0 outlier rule (``tools/readout/ro9_gl_attention.py`` MAD_K = 3.0).
#: Reused verbatim so the sink definition does not silently drift between arms.
SINK_MAD_K = 3.0


def merged_grid(out_h: int, out_w: int) -> tuple[int, int]:
    """``(out_h, out_w)`` in pixels -> the image-token grid ``(out/32, out/32)``."""
    if out_h % IMAGE_TOKEN_STRIDE or out_w % IMAGE_TOKEN_STRIDE:
        raise ValueError(
            f"image size {out_h}x{out_w} is not aligned to {IMAGE_TOKEN_STRIDE}; "
            "plan_geometry should have guaranteed that"
        )
    return out_h // IMAGE_TOKEN_STRIDE, out_w // IMAGE_TOKEN_STRIDE


def assert_model_facts(model) -> dict[str, Any]:
    """Fail loudly if the loaded model is not the one this module was written for.

    The eager check is the red line: FA2/SDPA return ``None`` attention weights
    and the tap would silently record nothing, so this refuses to run at all
    rather than degrade.
    """
    cfg = model.config
    text = getattr(cfg, "text_config", cfg)
    impl = getattr(cfg, "_attn_implementation", None)
    if impl != "eager":
        raise RuntimeError(
            f"attn_implementation is {impl!r}; attention export requires 'eager' "
            "(FA2/SDPA return None weights and must NOT be fallen back to -- red line)"
        )
    facts = {
        "attn_implementation": impl,
        "num_hidden_layers": int(text.num_hidden_layers),
        "num_attention_heads": int(text.num_attention_heads),
        "num_key_value_heads": int(text.num_key_value_heads),
        "hidden_size": int(text.hidden_size),
        "image_token_id": int(getattr(cfg, "image_token_id", -1)),
        "deepstack_visual_indexes": list(
            getattr(getattr(cfg, "vision_config", None), "deepstack_visual_indexes", [])
        ),
        "patch_size": int(getattr(getattr(cfg, "vision_config", None), "patch_size", -1)),
        "spatial_merge_size": int(
            getattr(getattr(cfg, "vision_config", None), "spatial_merge_size", -1)
        ),
    }
    if facts["image_token_id"] != IMAGE_TOKEN_ID:
        raise RuntimeError(
            f"config.image_token_id={facts['image_token_id']} != {IMAGE_TOKEN_ID}"
        )
    stride = facts["patch_size"] * facts["spatial_merge_size"]
    if stride != IMAGE_TOKEN_STRIDE:
        raise RuntimeError(
            f"patch {facts['patch_size']} x merge {facts['spatial_merge_size']} = "
            f"{stride}, but this module assumes an image-token stride of "
            f"{IMAGE_TOKEN_STRIDE}"
        )
    return facts


# --- where the rows and the columns are -------------------------------------

def locate_image_columns(input_ids: Sequence[int]) -> np.ndarray:
    """Positions of the image tokens, in model order (row-major on the merged grid).

    Qwen expands one ``<|image_pad|>`` into ``n_visual_tokens`` consecutive
    copies, so the block is contiguous; it is still located by value rather than
    by arithmetic, and asserted contiguous, because an off-by-one here silently
    rotates every field.
    """
    ids = np.asarray(input_ids)
    cols = np.flatnonzero(ids == IMAGE_TOKEN_ID)
    if cols.size == 0:
        raise ValueError("no image tokens in the sequence")
    if cols.size > 1 and not np.all(np.diff(cols) == 1):
        raise ValueError("image token block is not contiguous")
    return cols


@dataclass
class QueryPools:
    """Row indices per query pool, plus the bookkeeping a report needs."""

    rows: dict[str, np.ndarray]
    #: rows used for the query-agnostic column profile that defines sinks
    profile_rows: np.ndarray
    n_prompt_tokens: int
    seq_len: int
    detail: dict[str, Any] = field(default_factory=dict)

    def union(self) -> np.ndarray:
        """Every row the tap has to slice, sorted and de-duplicated."""
        allr = np.concatenate([r for r in self.rows.values() if r.size] or [np.zeros(0, int)])
        return np.unique(allr.astype(np.int64))

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": {k: int(v.size) for k, v in self.rows.items()},
            "n_profile_rows": int(self.profile_rows.size),
            "n_prompt_tokens": self.n_prompt_tokens,
            "seq_len": self.seq_len,
            **self.detail,
        }


def _is_content_token(text: str) -> bool:
    """A token carries region semantics only if it has an alphanumeric character.

    Pure punctuation / whitespace pieces ("``,``", "`` ``") are dropped from the
    content pool: RO-3's decisive variable was "the text tokens that carry region
    semantics", and a comma carries none.  The filter is deliberately this dumb
    -- a curated stop-word list would be a tunable knob, and tunable knobs on a
    query pool are how one enumerates one's way to a positive result.
    """
    return any(ch.isalnum() for ch in text)


def locate_query_pools(
    input_ids: Sequence[int],
    n_prompt_tokens: int,
    tokenizer=None,
) -> QueryPools:
    """Split a teacher-forced ``prompt + <where>...</where>`` sequence into pools.

    * ``where_special``  -- the single ``<where>`` open-tag row (RO-9's channel);
    * ``where_content``  -- the content tokens strictly inside the span (RO-3's);
    * ``instr_text``     -- the instruction, i.e. everything between
      ``<|vision_end|>`` and the next ``<|im_end|>``.  Located structurally from
      the chat template rather than by re-tokenising the instruction string,
      because a subsequence search would mis-align on any tokenizer merge across
      the boundary;
    * ``where_close``    -- the ``</where>`` row, exploratory.

    ``profile_rows`` is every text row **after** the image block.  Image rows are
    excluded on purpose: attention is causal, so an early image token is visible
    to more queries than a late one, and averaging image rows in would stamp a
    monotone positional ramp onto the column profile and make the first cells
    look like sinks.
    """
    ids = np.asarray(input_ids)
    cols = locate_image_columns(ids)
    img_end = int(cols[-1])

    def _rows(mask: np.ndarray) -> np.ndarray:
        return np.flatnonzero(mask).astype(np.int64)

    open_rows = _rows(ids == WHERE_OPEN_ID)
    close_rows = _rows(ids == WHERE_CLOSE_ID)
    if open_rows.size != 1:
        raise ValueError(f"expected exactly one <where>, found {open_rows.size}")

    # content span: strictly between <where> and </where>
    start = int(open_rows[0]) + 1
    stop = int(close_rows[0]) if close_rows.size else len(ids)
    inner = np.arange(start, stop, dtype=np.int64)
    n_inner = int(inner.size)
    if tokenizer is not None and n_inner:
        keep = [
            i for i in inner
            if _is_content_token(tokenizer.decode([int(ids[i])], skip_special_tokens=False))
        ]
        content = np.asarray(keep, dtype=np.int64) if keep else inner
    else:
        content = inner

    # instruction span: <|vision_end|> .. next <|im_end|>
    ve = _rows(ids == VISION_END_ID)
    if ve.size == 0:
        raise ValueError("no <|vision_end|> in the prompt")
    i0 = int(ve[0]) + 1
    ime = _rows(ids == IM_END_ID)
    after = ime[ime > i0]
    if after.size == 0:
        raise ValueError("no <|im_end|> after <|vision_end|>")
    instr = np.arange(i0, int(after[0]), dtype=np.int64)

    rows = {
        "where_special": open_rows,
        "where_content": content,
        "instr_text": instr,
        "where_close": close_rows,
    }
    for name, r in rows.items():
        if r.size and (r.min() < 0 or r.max() >= len(ids)):
            raise ValueError(f"pool {name} has out-of-range rows")
    if instr.size == 0:
        raise ValueError("instruction pool is empty; the prompt template changed")

    profile_rows = np.arange(img_end + 1, len(ids), dtype=np.int64)
    return QueryPools(
        rows=rows,
        profile_rows=profile_rows,
        n_prompt_tokens=int(n_prompt_tokens),
        seq_len=int(len(ids)),
        detail={
            "n_inner_where_tokens": n_inner,
            "n_content_after_filter": int(content.size),
            "image_block": [int(cols[0]), int(cols[-1])],
        },
    )


# --- the tap ----------------------------------------------------------------

@dataclass
class AttnFields:
    """One sample, one context arm: raw per-head fields over the image grid."""

    sample_id: str
    arm: str
    grid_h: int                       # merged grid, = out_h / 32
    grid_w: int
    #: pool -> (n_layers, n_heads, grid_h, grid_w), float32, **un-normalised**
    fields: dict[str, np.ndarray]
    #: (n_layers, n_heads, grid_h, grid_w): mean over post-image text rows, the
    #: query-agnostic profile the sink rule is computed from
    col_profile: np.ndarray
    pools: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def n_img(self) -> int:
        return self.grid_h * self.grid_w


class AttentionTap:
    """Per-layer forward hooks that slice ``rows x image cols`` *inside* the hook.

    The full weight tensor is ``(1, 32, T, T)``; at T=1500 that is 288 MiB of
    fp32 softmax per layer, and ``output_attentions=True`` would keep all 36 of
    them alive at once.  Slicing in the hook and moving to CPU immediately keeps
    the peak at one layer's worth, which is PROPOSAL 2.2 step 2's requirement.

    The hook asserts the weights are not ``None`` on every single layer, so an
    accidental SDPA load cannot produce a silently empty cache.
    """

    def __init__(self, lm: torch.nn.Module, *, store_profile: bool = True):
        self.lm = lm
        self.n_layers = len(lm.layers)
        self.store_profile = store_profile
        self._handles: list[Any] = []
        self._rows: torch.Tensor | None = None
        self._cols: torch.Tensor | None = None
        self._profile_rows: torch.Tensor | None = None
        self.slices: list[torch.Tensor | None] = [None] * self.n_layers
        self.profiles: list[torch.Tensor | None] = [None] * self.n_layers

    def set_selection(self, rows: np.ndarray, cols: np.ndarray,
                      profile_rows: np.ndarray) -> None:
        self._rows = torch.as_tensor(np.asarray(rows), dtype=torch.long)
        self._cols = torch.as_tensor(np.asarray(cols), dtype=torch.long)
        self._profile_rows = torch.as_tensor(np.asarray(profile_rows), dtype=torch.long)
        self.slices = [None] * self.n_layers
        self.profiles = [None] * self.n_layers

    def _make_hook(self, layer_idx: int):
        def fn(module, inputs, output):  # noqa: ANN001
            weights = output[1] if isinstance(output, tuple) and len(output) > 1 else None
            if weights is None:
                raise RuntimeError(
                    f"layer {layer_idx} returned attn_weights=None -- the model is not "
                    "running eager attention.  Falling back is forbidden (red line)."
                )
            w = weights[0]                       # (heads, T, T); batch is always 1
            dev = w.device
            rows = self._rows.to(dev)
            cols = self._cols.to(dev)
            sel = w.index_select(1, rows).index_select(2, cols)
            self.slices[layer_idx] = sel.to(torch.float32).cpu()
            if self.store_profile:
                prows = self._profile_rows.to(dev)
                prof = w.index_select(1, prows).index_select(2, cols)
                self.profiles[layer_idx] = prof.to(torch.float32).mean(dim=1).cpu()
            del w, sel
        return fn

    @contextmanager
    def attached(self) -> Iterator["AttentionTap"]:
        for i, layer in enumerate(self.lm.layers):
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                raise AttributeError(f"layer {i} has no .self_attn")
            self._handles.append(attn.register_forward_hook(self._make_hook(i)))
        try:
            yield self
        finally:
            for h in self._handles:
                h.remove()
            self._handles = []

    def stack(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``(n_layers, n_heads, n_rows, n_img)`` and the profile stack."""
        if any(s is None for s in self.slices):
            missing = [i for i, s in enumerate(self.slices) if s is None]
            raise RuntimeError(f"layers {missing} produced no attention slice")
        rows = torch.stack(self.slices, dim=0)
        prof = None
        if self.store_profile:
            if any(p is None for p in self.profiles):
                raise RuntimeError("profile stack incomplete")
            prof = torch.stack(self.profiles, dim=0)
        return rows, prof


# --- sink rule --------------------------------------------------------------

def sink_mask_from_profile(profile: np.ndarray, k: float = SINK_MAD_K) -> np.ndarray:
    """``median + k*MAD`` outlier rule on the layer/head-mean column profile.

    Returns a boolean array over image cells, ``True`` = sink = **excluded**.

    The rule is an arm constant (same statistic, same ``k``, every sample); the
    threshold it yields is per image, which is unavoidable because the cells
    themselves are per image.  That is a different object from the banned
    per-image min-max *normalisation*: no field value is rescaled here, cells are
    only dropped from the support, and every number downstream comes from the
    raw field on the surviving cells.

    ``median + k*MAD`` rather than ``mean + k*sigma`` because a couple of extreme
    sinks inflate sigma enough to hide themselves -- this is RO9b's D-0 rule
    (``tools/readout/ro9_gl_attention.py``), reused so the two arms cannot drift.

    ``MAD == 0`` means more than half the cells sit on exactly the median, so
    the bulk has no spread at all.  RO9b's version returns "no outliers" there,
    which fails **open**: a profile that is flat except for two enormous spikes
    would keep the spikes.  Here that case falls back to "anything strictly above
    the median is an outlier", which is the correct reading of a degenerate bulk
    and fails closed.  It cannot mis-fire on a real profile, where MAD > 0.
    """
    p = np.asarray(profile, dtype=np.float64).reshape(-1)
    med = np.median(p)
    mad = np.median(np.abs(p - med))
    if mad <= 0:
        return p > med
    return p > med + k * mad


def assert_domain(
    field: np.ndarray,
    domain: tuple[float, float],
    *,
    name: str = "field",
    tol: float = 1e-6,
) -> dict[str, Any]:
    """Consumer-side domain assertion -- the s-cache contract's second half.

    The contract's two failure modes are asymmetric and the second one is silent:
    a field that falls *outside* the declared anchors collapses loudly, but a
    field that gets **clamped** into range keeps living inside the anchors while
    its axis is gone, and nothing downstream notices except a mysteriously weak
    result.  A readout arm that reports "the real field lost to the oracle field"
    without showing this assertion is not to be believed (CLAUDE.md).

    So this refuses to be decorative: it raises if the data leaves the declared
    box, and it returns the measured extremes plus a saturation count, which is
    what actually catches a clamp (mass piled exactly on an endpoint).
    """
    f = np.asarray(field, dtype=np.float64).reshape(-1)
    lo, hi = float(domain[0]), float(domain[1])
    fmin, fmax = float(f.min()), float(f.max())
    if fmin < lo - tol or fmax > hi + tol:
        raise ValueError(
            f"{name}: measured domain [{fmin:.6g}, {fmax:.6g}] escapes the declared "
            f"[{lo:.6g}, {hi:.6g}] -- either the declaration is stale or the field "
            "is not the one it claims to be"
        )
    n_lo = int(np.isclose(f, lo, atol=tol).sum())
    n_hi = int(np.isclose(f, hi, atol=tol).sum())
    return {
        "name": name, "declared": [lo, hi], "measured": [fmin, fmax],
        "n_at_lower": n_lo, "n_at_upper": n_hi,
        "frac_saturated": (n_lo + n_hi) / max(f.size, 1),
        "crosses_zero": bool(fmin < 0 < fmax),
        "passed": True,
    }


def sink_mask_conjunctive(
    profile_gt: np.ndarray,
    profile_shuffled: np.ndarray,
    k: float = SINK_MAD_K,
) -> np.ndarray:
    """The card's full sink criterion: high mass **and** instruction-invariant.

    ``sink = above-threshold under the real instruction AND above-threshold under
    a shuffled one``.  The second clause is what separates a sink from a hit: a
    cell that is only bright when the instruction points at it is *signal*, and
    excluding it would delete the very thing the probe is looking for.  Taking
    the intersection spares those cells automatically.

    The mask is a property of the image, not of the arm -- both arms are scored
    on the same support -- which is what makes the P3 gt-vs-shuffled comparison a
    paired test rather than a comparison of two different supports.
    """
    a = sink_mask_from_profile(profile_gt, k=k)
    b = sink_mask_from_profile(profile_shuffled, k=k)
    return a & b
