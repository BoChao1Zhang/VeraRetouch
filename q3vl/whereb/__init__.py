"""Stage-Where-B: MetaCanvas queries predicting the global basis parameters.

Protocol: ``docs/METACANVAS_WHERE_WHAT_FINAL_EXPERIMENT_PROTOCOL_2026-08-04.md``
sections 5 (all), 3, 10.3 and 14 items 7/8/9.

    frozen Qwen3-VL  ->  F_pre, H_where
    Q_where + connector + heads  ->  global (w0, w_dir, alpha, rho)
    s(p) = Phi(I, p) @ w   (Where-A basis, frozen)
    m(p) = R(s(p); rho)

Nothing in this package trains the VLM or the Where-A basis ``B``; nothing in it
reads ``H_color`` or ``I_tar``.

Nothing is imported eagerly here.  ``q3vl.whereb.model`` imports torch, and an
eager re-export meant that ``python -m q3vl.whereb.scripts.<job>`` loaded torch
while walking the package chain -- *before* the entry point's own body ran.  The
sqlite3-before-torch guard at the top of each script was therefore already too
late to help (campaign bug R6: torch shadows the libstdc++ that ``_sqlite3``'s
dependency chain needs, so a torch-first process can never open a published
shard).  ``q3vl.what`` has the same shape for the same reason.

The convenience re-exports still work -- PEP 562 resolves them on first
attribute access -- so ``from q3vl.whereb import WhereBModel`` is unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

_LAZY = {
    "ARMS": "config", "ARM_IDS": "config", "STRUCTURES": "config",
    "ArmConfig": "config", "TrainConfig": "config", "arm_config": "config",
    "WhereBModel": "model", "WhereBOutput": "model",
    "MODEL_INPUT_KEYS": "model", "parameter_table": "model",
}

__all__ = list(_LAZY)


def __getattr__(name: str):
    """PEP 562 lazy re-export: torch is imported on use, not on package import."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(f".{module}", __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:  # keep type checkers and IDEs seeing the real symbols
    from .config import (  # noqa: F401
        ARMS, ARM_IDS, STRUCTURES, ArmConfig, TrainConfig, arm_config,
    )
    from .model import (  # noqa: F401
        MODEL_INPUT_KEYS, WhereBModel, WhereBOutput, parameter_table,
    )
