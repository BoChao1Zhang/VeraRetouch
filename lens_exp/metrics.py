# Image + text metrics: PSNR / SSIM / deltaE00 / CLIP consistency.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.color import rgb2lab, deltaE_ciede2000

CLIP_PATH = os.path.expanduser("~/data/models/clip-vit-large-patch14")


def _pair(pred_path, gt_path, max_side=768):
    p = cv2.imread(pred_path, cv2.IMREAD_COLOR)
    g = cv2.imread(gt_path, cv2.IMREAD_COLOR)
    if p is None or g is None:
        raise IOError(f"read fail {pred_path} {gt_path}")
    if p.shape[:2] != g.shape[:2]:
        g = cv2.resize(g, (p.shape[1], p.shape[0]), interpolation=cv2.INTER_AREA)
    s = max_side / max(p.shape[:2])
    if s < 1:
        d = (int(p.shape[1]*s), int(p.shape[0]*s))
        p = cv2.resize(p, d, interpolation=cv2.INTER_AREA)
        g = cv2.resize(g, d, interpolation=cv2.INTER_AREA)
    return p[..., ::-1], g[..., ::-1]  # RGB


def image_metrics(pred_path, gt_path):
    p, g = _pair(pred_path, gt_path)
    psnr = peak_signal_noise_ratio(g, p, data_range=255)
    ssim = structural_similarity(g, p, channel_axis=2, data_range=255)
    de = float(deltaE_ciede2000(rgb2lab(g/255.0), rgb2lab(p/255.0)).mean())
    return dict(psnr=float(psnr), ssim=float(ssim), de00=de)


class ClipScorer:
    def __init__(self, device="cuda"):
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(CLIP_PATH, torch_dtype=torch.float16).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(CLIP_PATH)
        self.device = device

    @torch.no_grad()
    def score(self, img_path, text):
        from PIL import Image
        img = Image.open(img_path).convert("RGB")
        inp = self.proc(text=[text], images=[img], return_tensors="pt",
                        padding=True, truncation=True, max_length=77).to(self.device)
        inp["pixel_values"] = inp["pixel_values"].half()
        out = self.model(**inp)
        a = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        b = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        return float((a * b).sum())
