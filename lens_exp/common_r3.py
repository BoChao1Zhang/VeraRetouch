# lens-exp round 3 common: R3 result paths + shared loaders for S1-S7.
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from common import TOKENS, TARGET_NAMES  # noqa: F401
from common_r2 import (RESULTS_R2, C0_DIR, load_manifest_r2, get_split_r2,  # noqa: F401
                       load_targets_r2, DIM_NAMES)

RESULTS_R3 = os.path.expanduser("~/VeraRetouch/lens_exp_results_r3")
os.makedirs(RESULTS_R3, exist_ok=True)

# 26-dim target slice per token (light 6 | colortemp 4 | colormixer 16)
TOKEN_SLICES = {"light": slice(0, 6), "colortemp": slice(6, 10), "colormixer": slice(10, 26)}


def captured_keys():
    return sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(C0_DIR, "*.npz")))


def load_latents_all_layers(keys):
    """token -> [N, 25, 896] fp32 from C0 dumps."""
    out = {t: [] for t in TOKENS}
    for k in keys:
        d = np.load(os.path.join(C0_DIR, k + ".npz"))
        for t in TOKENS:
            out[t].append(d[f"lat_{t}"].astype(np.float32))
        d.close()
    return {t: np.stack(v) for t, v in out.items()}


def load_latents_layer(keys, layer):
    """token -> [N, 896] fp32 at one hidden_states index."""
    out = {t: [] for t in TOKENS}
    for k in keys:
        d = np.load(os.path.join(C0_DIR, k + ".npz"))
        for t in TOKENS:
            out[t].append(d[f"lat_{t}"][layer].astype(np.float32))
        d.close()
    return {t: np.stack(v) for t, v in out.items()}


def targets_matrix(keys):
    tg = load_targets_r2()
    return np.stack([tg[k] for k in keys])
