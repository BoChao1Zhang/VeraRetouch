"""Shape- and key-aware checkpoint resume for ``AmortModel``.

Why this file exists.  Three B2 jobs died two minutes after launch on

    RuntimeError: size mismatch for geo.tower.stem.weight: copying a param with
    shape [128, 1025, 1, 1] ... the shape in current model is [128, 1046, 1, 1]

because ``--geom-inject`` widens the stem by 21 code channels while the
checkpoint being resumed was trained without them.  ``strict=False`` alone is
*not* the fix: it would leave the widened stem at its random initialisation, so
the resumed arm would start somewhere other than the checkpoint it claims to
continue, and every Delta measured against that checkpoint would be measuring
the perturbation as much as the injection.

The rule this module enforces instead:

* a tensor that exists in both, same shape        -> copied;
* a tensor the model grew **input channels** for  -> old weights copied into the
  leading slice, the new channels **zeroed**, so the layer computes exactly what
  it computed before until training moves it;
* a tensor missing from the checkpoint             -> allowed only for an
  explicitly named new module (the injector), which is zero-initialised at its
  output and therefore also a no-op at step 0;
* a tensor in the checkpoint the model has no slot for -> **hard error**.  That
  direction means trained weights are being silently dropped, which is how a
  "resumed" run quietly becomes a fresh one.

Everything the loader did is returned as a report and printed into the run log,
because "the code ran and did not raise" is not evidence that a resume resumed.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

__all__ = ["load_resumable", "ResumeError"]


class ResumeError(RuntimeError):
    """Raised when a checkpoint cannot be loaded without losing information."""


def load_resumable(
    model: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    *,
    new_module_prefixes: Sequence[str] = ("pch.",),
) -> dict[str, Any]:
    """Load ``state`` into ``model``, tolerating only benign differences.

    Returns a report ``{copied, zero_padded, from_init, ...}`` naming every
    tensor in each category -- the observable that proves which branch ran.
    """
    own = model.state_dict()
    copied: list[str] = []
    zero_padded: list[dict[str, Any]] = []
    from_init: list[str] = []

    unexpected = [k for k in state if k not in own]
    if unexpected:
        raise ResumeError(
            f"checkpoint has {len(unexpected)} tensor(s) this model cannot "
            f"place, e.g. {unexpected[:5]}; refusing to silently drop trained "
            "weights -- check --arm/--readout/ablation flags match the run "
            "being resumed")

    merged: dict[str, torch.Tensor] = {}
    for key, cur in own.items():
        if key not in state:
            if not any(key.startswith(p) for p in new_module_prefixes):
                raise ResumeError(
                    f"{key} is absent from the checkpoint and is not part of a "
                    f"declared new module {tuple(new_module_prefixes)}; a "
                    "randomly initialised tensor in a resumed arm invalidates "
                    "every Delta measured against that checkpoint")
            merged[key] = cur
            from_init.append(key)
            continue

        old = state[key]
        if old.shape == cur.shape:
            merged[key] = old
            copied.append(key)
            continue

        # the only tolerated growth: more INPUT channels on a conv/linear
        # weight, every other dimension unchanged
        ok = (old.dim() == cur.dim() and old.dim() >= 2
              and old.shape[0] == cur.shape[0]
              and old.shape[1] < cur.shape[1]
              and tuple(old.shape[2:]) == tuple(cur.shape[2:]))
        if not ok:
            raise ResumeError(
                f"{key}: checkpoint shape {tuple(old.shape)} vs model "
                f"{tuple(cur.shape)} is not an input-channel extension; this "
                "is a configuration mismatch, not a resumable difference")
        grown = torch.zeros_like(cur)
        grown[:, :old.shape[1]] = old.to(grown.dtype)
        merged[key] = grown
        zero_padded.append({"key": key, "old_in": int(old.shape[1]),
                            "new_in": int(cur.shape[1])})

    model.load_state_dict(merged, strict=True)
    return {
        "n_copied": len(copied),
        "zero_padded": zero_padded,
        "from_init": from_init,
        "n_from_init": len(from_init),
        "note": ("new input channels are zeroed and new modules are "
                 "zero-initialised, so step 0 reproduces the checkpoint"),
    }
