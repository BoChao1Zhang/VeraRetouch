"""Stage-Where-B: MetaCanvas queries predicting the global basis parameters.

Protocol: ``docs/METACANVAS_WHERE_WHAT_FINAL_EXPERIMENT_PROTOCOL_2026-08-04.md``
sections 5 (all), 3, 10.3 and 14 items 7/8/9.

    frozen Qwen3-VL  ->  F_pre, H_where
    Q_where + connector + heads  ->  global (w0, w_dir, alpha, rho)
    s(p) = Phi(I, p) @ w   (Where-A basis, frozen)
    m(p) = R(s(p); rho)

Nothing in this package trains the VLM or the Where-A basis ``B``; nothing in it
reads ``H_color`` or ``I_tar``.
"""

from __future__ import annotations

from .config import ARMS, ARM_IDS, STRUCTURES, ArmConfig, TrainConfig, arm_config
from .model import MODEL_INPUT_KEYS, WhereBModel, WhereBOutput, parameter_table

__all__ = [
    "ARMS", "ARM_IDS", "STRUCTURES", "ArmConfig", "TrainConfig", "arm_config",
    "WhereBModel", "WhereBOutput", "MODEL_INPUT_KEYS", "parameter_table",
]
