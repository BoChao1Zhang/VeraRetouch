from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image


def load_image(path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    data = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    data = data.view(image_size, image_size, 3).to(torch.float32) / 255.0
    return data.permute(2, 0, 1).contiguous()
