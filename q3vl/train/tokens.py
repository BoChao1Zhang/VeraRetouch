"""Special-token registration and embedding handling (spec 4.3).

Verified facts about ``/home/bc/data/models/Qwen3-VL-4B-Instruct`` (see
``NOTES.md`` for the probe transcript):

* ``len(tokenizer) == 151669`` before registration, ``151673`` after;
* the input embedding matrix has ``151936`` rows (``config.text_config.vocab_size``),
  i.e. the checkpoint ships **263 unused padded rows**;
* ``tie_word_embeddings == True`` -- ``lm_head.weight`` *is* ``embed_tokens.weight``;
* the padded rows are near-degenerate (every row above 151668 has L2 ~= 0.358 and
  several are bit-identical), so the four new tokens would start out
  indistinguishable from one another. We therefore always re-initialise them.

Resize policy: **never shrink**. ``resize_token_embeddings(151673)`` would drop
the padded rows and leave a non-aligned vocab; since ``151673 <= 151936`` the
new ids already have rows, so we keep the matrix and only re-init the 4 rows.
A resize is still performed if a future tokenizer grows past the matrix.
"""

from __future__ import annotations

import contextlib
from typing import Any

import torch
import torch.nn as nn

from .constants import SPECIAL_TOKENS
from .freeze import full_shape


class SpecialTokenError(RuntimeError):
    pass


def is_zero3_partitioned(param) -> bool:
    """True when ``deepspeed.zero.Init`` owns this parameter."""
    return hasattr(param, "ds_id")


def gathered(params, modifier_rank: int | None = 0):
    """``deepspeed.zero.GatheredParameters`` if needed, else a no-op context.

    Under ZeRO-3 the real two-GPU launch reaches this module with every
    parameter partitioned and freed, so touching ``emb.weight`` directly reads a
    zero-length tensor. Row surgery on the embedding matrix has to happen inside
    a gather window, and the modified rows have to be re-partitioned from a
    single rank on exit.
    """
    params = [p for p in params]
    if any(is_zero3_partitioned(p) for p in params):
        import deepspeed  # local import: single-GPU paths must not need it

        return deepspeed.zero.GatheredParameters(params, modifier_rank=modifier_rank)
    return contextlib.nullcontext()


def register_special_tokens(tokenizer) -> dict[str, int]:
    """Append the four tokens as ``additional_special_tokens`` (idempotent)."""
    existing = list(tokenizer.additional_special_tokens or [])
    missing = [t for t in SPECIAL_TOKENS if t not in existing]
    if missing:
        n_added = tokenizer.add_special_tokens(
            {"additional_special_tokens": missing},
            replace_additional_special_tokens=False,
        )
        if n_added != len(missing):
            raise SpecialTokenError(
                f"tokenizer added {n_added} tokens, expected {len(missing)} ({missing})"
            )
    return verify_single_token(tokenizer)


def verify_single_token(tokenizer) -> dict[str, int]:
    """Assert every special token encodes to exactly one id; return the map."""
    ids: dict[str, int] = {}
    for tok in SPECIAL_TOKENS:
        enc = tokenizer(tok, add_special_tokens=False)["input_ids"]
        if len(enc) != 1:
            raise SpecialTokenError(
                f"{tok!r} does not encode to a single token: {enc} "
                f"(decoded {[tokenizer.decode([i]) for i in enc]})"
            )
        ids[tok] = int(enc[0])
        if tok not in (tokenizer.additional_special_tokens or []):
            raise SpecialTokenError(f"{tok!r} missing from additional_special_tokens")
        # round-trip: the id must decode back to the exact literal
        back = tokenizer.decode([ids[tok]])
        if back != tok:
            raise SpecialTokenError(f"id {ids[tok]} decodes to {back!r}, expected {tok!r}")
    if len(set(ids.values())) != len(SPECIAL_TOKENS):
        raise SpecialTokenError(f"special token ids are not distinct: {ids}")
    return ids


def _mean_init_rows(
    weight: torch.Tensor,
    row_ids: list[int],
    ref_upper: int,
    generator: torch.Generator,
    noise_scale: float = 1e-3,
) -> dict[str, float]:
    """Re-init ``row_ids`` to (mean of rows [0, ref_upper)) + tiny per-dim noise.

    Mirrors the intent of ``transformers`` ``mean_resizing=True``: sit at the
    centre of the existing embedding cloud, but break the symmetry between the
    new tokens so their logits are not identical at step 0.
    """
    with torch.no_grad():
        ref = weight[:ref_upper].to(torch.float32)
        mean = ref.mean(dim=0)
        std = ref.std(dim=0)
        stats = {
            "ref_rows": int(ref_upper),
            "mean_row_l2": float(mean.norm()),
            "ref_mean_row_l2": float(ref.norm(dim=-1).mean()),
            "per_dim_std_mean": float(std.mean()),
            "noise_scale": noise_scale,
        }
        for rid in row_ids:
            # The generator is a CPU generator on purpose: the drawn values must
            # not depend on device or world size, so the four rows are
            # bit-identical to the ones the single-GPU preflight validated.
            noise = torch.randn(mean.shape, generator=generator, dtype=torch.float32)
            noise = noise.to(mean.device) * std * noise_scale
            weight[rid] = (mean + noise).to(weight.dtype)
    return stats


def prepare_embeddings(
    model: nn.Module,
    tokenizer,
    special_ids: dict[str, int],
    seed: int = 0,
    reinit_new_rows: bool = True,
) -> dict[str, Any]:
    """Resize (never shrink) and initialise the rows of the four new tokens."""
    emb = model.get_input_embeddings()
    # NEVER read `emb.weight.shape[0]` directly: under ZeRO-3 it is 0 (the
    # parameter is partitioned and freed), which would make `need > rows_before`
    # true and trigger a resize_token_embeddings(151673, pad_to_multiple_of=64)
    # -> a matrix of 151680 rows, i.e. silently *shrinking* the vendor's 151936
    # and violating the never-shrink policy this function exists to enforce.
    rows_before = int(full_shape(emb.weight)[0])
    need = len(tokenizer)
    info: dict[str, Any] = {
        "tokenizer_len": need,
        "embedding_rows_before": rows_before,
        "zero3_partitioned": is_zero3_partitioned(emb.weight),
        "tie_word_embeddings": bool(getattr(model.config, "tie_word_embeddings", False)),
    }

    if need > rows_before:
        model.resize_token_embeddings(need, pad_to_multiple_of=64, mean_resizing=True)
        info["resize_action"] = "grown"
    else:
        # 151673 <= 151936: the ids already have rows. Shrinking would delete
        # the vendor's vocab padding and de-align the matrix, so we do not.
        info["resize_action"] = "kept"

    emb = model.get_input_embeddings()
    rows_after = int(full_shape(emb.weight)[0])
    info["embedding_rows_after"] = rows_after
    if rows_after < need:
        raise SpecialTokenError(
            f"embedding matrix has {rows_after} rows but tokenizer needs {need}"
        )
    max_id = max(special_ids.values())
    if max_id >= rows_after:
        raise SpecialTokenError(f"special token id {max_id} out of embedding range {rows_after}")

    if reinit_new_rows:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        target_rows = sorted(special_ids.values())
        # reference = the rows that actually carry pretrained semantics, i.e.
        # everything strictly below the first newly-assigned id.
        ref_upper = min(target_rows)
        weight = emb.weight
        was_meta = weight.device.type == "meta"
        if was_meta:
            raise SpecialTokenError("cannot initialise embeddings on a meta-device model")
        with gathered([weight], modifier_rank=0):
            if int(weight.shape[0]) < need:
                raise SpecialTokenError(
                    f"embedding weight exposes {tuple(weight.shape)} rows inside the gather "
                    f"window but the tokenizer needs {need}"
                )
            info["reinit"] = _mean_init_rows(weight.data, target_rows, ref_upper, gen)
            info["reinit"]["rows"] = target_rows
            with torch.no_grad():
                new_rows = weight.data[target_rows].to(torch.float32)
                pair = torch.cdist(new_rows, new_rows)
                info["reinit"]["min_pairwise_distance"] = float(
                    pair[~torch.eye(len(target_rows), dtype=torch.bool, device=pair.device)].min()
                )
    else:
        info["reinit"] = None

    info["tied_ok"] = assert_tied(model)
    return info


def assert_tied(model: nn.Module) -> bool:
    """Verify the input embedding and output head still share storage."""
    if not getattr(model.config, "tie_word_embeddings", False):
        return False
    inp = model.get_input_embeddings()
    out = model.get_output_embeddings()
    if out is None:
        raise SpecialTokenError("model reports tied weights but has no output embedding")
    if inp.weight is out.weight:
        # Under ZeRO-3 both are zero-length views and `data_ptr()` can be 0 for
        # each, so object identity is the stronger check when it is available.
        return True
    if inp.weight.data_ptr() != out.weight.data_ptr():
        raise SpecialTokenError(
            "tied-weight contract broken: input and output embeddings no longer share storage"
        )
    return True


def verify_reload(save_dir: str, expected_ids: dict[str, int]) -> dict[str, Any]:
    """Reload a saved tokenizer/processor dir and re-check ids + single-tokenness."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(save_dir)
    ids = verify_single_token(tok)
    mismatched = {k: (expected_ids.get(k), v) for k, v in ids.items() if expected_ids.get(k) != v}
    if mismatched:
        raise SpecialTokenError(f"special token ids changed after reload: {mismatched}")
    return {"reload_dir": save_dir, "ids": ids, "tokenizer_len": len(tok)}
