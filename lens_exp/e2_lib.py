# E2 offline pipeline: standalone retouch_head + decoder (no VLM needed).
# The readout layer only affects the post-generation render path, so layer swaps
# and decoder retraining operate on dumped all-layer latents.
import os, sys
sys.path.insert(0, os.path.expanduser("~/VeraRetouch"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn as nn
import cv2
from safetensors import safe_open
from utils import get_model as get_retouch_decoder_model

CKPT = os.path.expanduser("~/data/models/VeraRetouch/model.safetensors")
HID = 896


def build_head():
    return nn.Sequential(
        nn.Linear(2688, 1344), nn.LayerNorm(1344), nn.GELU(),
        nn.Linear(1344, 1344), nn.LayerNorm(1344), nn.GELU(),
        nn.Linear(1344, 2688))


def load_head_decoder(device="cuda", dtype=torch.bfloat16):
    head = build_head()
    decoder = get_retouch_decoder_model("RetouchRenderer_Resnet18Encoder_InputCatMixedCBAM").decoder
    hs, ds = {}, {}
    with safe_open(CKPT, framework="pt") as f:
        for k in f.keys():
            if k.startswith("retouch_head."):
                hs[k[len("retouch_head."):]] = f.get_tensor(k)
            elif k.startswith("retouch_decoder."):
                ds[k[len("retouch_decoder."):]] = f.get_tensor(k)
    head.load_state_dict(hs)
    decoder.load_state_dict(ds)
    return head.to(device, dtype).eval(), decoder.to(device, dtype).eval()


def latents_at_layer(npz, layer):
    """concat [light, colortemp, colormixer] latents at tuple index `layer` -> [1, 2688]"""
    v = np.concatenate([npz[f"lat_{t}"][layer].astype(np.float32)
                        for t in ("light", "colortemp", "colormixer")])
    return torch.from_numpy(v).unsqueeze(0)


def load_input_tensor(path, max_side=None):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if max_side and max(img.shape[:2]) > max_side:
        s = max_side / max(img.shape[:2])
        img = cv2.resize(img, (int(img.shape[1]*s), int(img.shape[0]*s)), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = torch.from_numpy(rgb).permute(2, 0, 1)
    return (x - 0.5) * 2  # [-1, 1], matches Infer_Style_Dataset


@torch.no_grad()
def render(head, decoder, lat, input_tensor, device="cuda", dtype=torch.bfloat16):
    """lat [1,2688] fp32, input_tensor [3,H,W] in [-1,1] -> BGR uint8 image"""
    z = head(lat.to(device, dtype))
    x = input_tensor.unsqueeze(0).to(device, dtype)
    out = decoder(x, z).float().clamp(0, 1)[0]
    img = (out.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
