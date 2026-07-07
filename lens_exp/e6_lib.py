# E6 (round 2): readout variants — peak-layer retrain, multi-layer fusion, affine calibration.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn as nn

from e2_lib import load_head_decoder, latents_at_layer  # noqa: F401

HID = 896
PEAK = (11, 14, 23)          # per-token peak layers from round-1 E1 (light, colortemp, colormixer)
FUSE_LAYERS = (11, 14, 23, 24)


def latents_fused(npz, layers=FUSE_LAYERS):
    """per token concat of `layers`, tokens concatenated -> [1, 3*len(layers)*896]"""
    v = np.concatenate([npz[f"lat_{t}"][int(l)].astype(np.float32)
                        for t in ("light", "colortemp", "colormixer") for l in layers])
    return torch.from_numpy(v).unsqueeze(0)


class FusedHead(nn.Module):
    """Original retouch head with the first Linear widened to n_layers*2688 inputs.
    Init: L24 block copies the original first-layer weights (function-preserving at start),
    other layer blocks zero."""

    def __init__(self, base_head, layers=FUSE_LAYERS):
        super().__init__()
        nl = len(layers)
        first = nn.Linear(2688 * nl, 1344)
        with torch.no_grad():
            first.weight.zero_()
            w0 = base_head[0].weight  # [1344, 2688]
            j24 = layers.index(24)
            for t in range(3):
                dst = (t * nl + j24) * HID
                first.weight[:, dst:dst + HID] = w0[:, t * HID:(t + 1) * HID]
            first.bias.copy_(base_head[0].bias)
        self.net = nn.Sequential(first, *list(base_head)[1:])

    def forward(self, x):
        return self.net(x)


class AffineHead(nn.Module):
    """Tuned-Lens style: per-token residual affine (zero-init) on the peak-layer latent,
    then the original head."""

    def __init__(self, base_head):
        super().__init__()
        self.affine = nn.ModuleList([nn.Linear(HID, HID) for _ in range(3)])
        for a in self.affine:
            nn.init.zeros_(a.weight)
            nn.init.zeros_(a.bias)
        self.head = base_head

    def forward(self, x):  # x [B, 2688]
        parts = []
        for t in range(3):
            xt = x[:, t * HID:(t + 1) * HID]
            parts.append(xt + self.affine[t](xt))
        return self.head(torch.cat(parts, dim=1))


VARIANTS = {
    # name: (latent_fn, head_builder, layer_spec)
    "e6a":     ("peak",  "orig",   PEAK),
    "e6b":     ("fused", "fused",  FUSE_LAYERS),
    "e6c":     ("peak",  "affine", PEAK),
    "e6_ctrl": ("l24",   "orig",   24),
}


def build_variant_model(variant, dtype=torch.float32):
    lat_kind, head_kind, layers = VARIANTS[variant]
    base_head, decoder = load_head_decoder(dtype=dtype)
    if head_kind == "orig":
        head = base_head
    elif head_kind == "fused":
        head = FusedHead(base_head, FUSE_LAYERS).to("cuda", dtype)
    elif head_kind == "affine":
        head = AffineHead(base_head).to("cuda", dtype)
    return head, decoder, lat_kind, layers


def load_latent(npz, lat_kind, layers):
    if lat_kind == "fused":
        return latents_fused(npz, layers)
    return latents_at_layer(npz, layers)
