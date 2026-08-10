"""RD-G data plumbing: the LUT bank, the colour sampling caliber, and the
memmapped D-RENDER image cache.

Two things are pinned here because every number in the experiment depends on
them being the same for every arm and for the ceiling:

1. GT colour transform.  The L_cube target is the preset's 33^3 resample from
   the T2 cube corpus (/var/cache/veradata/dcube/npy33), read with TRILINEAR
   interpolation -- the production renderer's own interpolator (F5 report 5.1,
   tri vs tetra differ by 0.009 dE00 median, i.e. below the JPEG floor of the
   archive).  Bake-consistency read-back is the one place that deliberately
   uses tetrahedral instead, because that is what a .cube host does.

2. Colour sampling caliber.  Headline numbers are on UNIFORM colours over the
   cube (E1's caliber, so E1's per-LUT direct fits and this experiment's
   ceiling are on the same axis); natural-image colours are reported as a
   second track because ENNELUT showed a model trained only on natural colours
   collapses on the full Hald (PLAN 3 level-1 item 2).  Training samples half
   and half so neither track is starved.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

NPY33 = "/var/cache/veradata/dcube/npy33"
CACHE = "/home/bc/VeraRetouch/experiments/RDG_transformer_20260803/cache"
CUBE_SIZE = 33


# ---------------------------------------------------------------------------
# LUT bank
# ---------------------------------------------------------------------------

def load_lut_bank(names, device="cuda") -> torch.Tensor:
    """-> (P, 33,33,33, 3) float32 on `device`; index order [r,g,b]."""
    out = np.empty((len(names), CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 3),
                   dtype=np.float32)
    for i, nm in enumerate(names):
        out[i] = np.load(os.path.join(NPY33, nm + ".npy"))
    return torch.from_numpy(out).to(device)


def tri_lookup(table: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Trilinear read of table[r,g,b].  table (B,K,K,K,3), x (B,P,3)."""
    B, K = table.shape[0], table.shape[1]
    return _tri_flat(table.reshape(B * K * K * K, 3), K,
                     torch.arange(B, device=x.device) * (K * K * K), x)


def tri_lookup_bank(bank: torch.Tensor, pid: torch.Tensor,
                    x: torch.Tensor) -> torch.Tensor:
    """Trilinear read of bank[pid[b]] without materialising bank[pid].

    `bank` is the whole corpus (n_presets, K,K,K, 3) resident on the device;
    gathering B tables out of it first would copy B*K^3*3 floats per step
    (256 x 35937 x 3 x 4 B = 110 MB) for no reason.  Folding the preset index
    into the flat lattice index removes that copy entirely.
    """
    K = bank.shape[1]
    return _tri_flat(bank.reshape(-1, 3), K, pid.long() * (K * K * K), x)


def _tri_flat(flat: torch.Tensor, K: int, base: torch.Tensor,
              x: torch.Tensor) -> torch.Tensor:
    """flat (M,3), base (B,) row offset per sample, x (B,P,3) -> (B,P,3)."""
    B, P, _ = x.shape
    xc = x.clamp(0, 1) * (K - 1)
    i0 = xc.floor().clamp(0, K - 2).long()
    f = xc - i0.to(x.dtype)
    ir, ig, ib = i0.unbind(-1)
    fr, fg, fb = f.unbind(-1)
    b0 = base.view(B, 1)
    out = None
    for dr in (0, 1):
        wr = fr if dr else 1 - fr
        for dg in (0, 1):
            wg = fg if dg else 1 - fg
            for db in (0, 1):
                wb = fb if db else 1 - fb
                idx = b0 + (((ir + dr) * K + (ig + dg)) * K + (ib + db))
                v = flat.index_select(0, idx.reshape(-1)).view(B, P, 3)
                t = (wr * wg * wb).unsqueeze(-1) * v
                out = t if out is None else out + t
    return out


# ---------------------------------------------------------------------------
# colour sampling
# ---------------------------------------------------------------------------

def uniform_colors(b: int, p: int, gen: torch.Generator,
                   device="cuda") -> torch.Tensor:
    return torch.rand(b, p, 3, generator=gen, device=device)


def image_colors(img: torch.Tensor, p: int, gen: torch.Generator
                 ) -> torch.Tensor:
    """Sample p pixel colours per image from the (B,3,R,R) source frame."""
    B, _, R, _ = img.shape
    flat = img.permute(0, 2, 3, 1).reshape(B, R * R, 3)
    idx = torch.randint(0, R * R, (B, p), generator=gen, device=img.device)
    return torch.gather(flat, 1, idx.unsqueeze(-1).expand(-1, -1, 3))


def eval_colors(b: int, p: int, seed: int, device="cuda") -> torch.Tensor:
    """Deterministic held-out colour set (never seen in a training step's RNG
    stream because it is drawn from its own generator/seed)."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return torch.rand(b, p, 3, generator=g, device=device)


# ---------------------------------------------------------------------------
# D-RENDER memmap cache
# ---------------------------------------------------------------------------

class RenderCache:
    """Read-only view of the build_cache.py output."""

    POOL = {"train": 0, "val_img": 1, "val_lut": 2}

    def __init__(self, root: str = CACHE, res: int = 128):
        self.root, self.R = root, res
        ix = np.load(os.path.join(root, "index.npz"), allow_pickle=False)
        self.group = ix["group"]
        self.preset = ix["preset"]
        self.pool = ix["pool"]
        self.conf = ix["conf"]
        self.p_split = ix["p_split"]
        self.presets = [str(s) for s in ix["presets"]]
        self.n_groups = int(ix["n_groups"])
        self.n = len(self.group)
        self.after = np.memmap(os.path.join(root, "imgs_after.u8"),
                               dtype=np.uint8, mode="r",
                               shape=(self.n, res, res, 3))
        self.imgin = np.memmap(os.path.join(root, "imgs_in.u8"),
                               dtype=np.uint8, mode="r",
                               shape=(self.n_groups, res, res, 3))

    def idx_of(self, pool: str) -> np.ndarray:
        return np.nonzero(self.pool == self.POOL[pool])[0]

    def batch(self, rows: np.ndarray):
        """-> (img (B,6,R,R) float32 in [0,1], preset_idx (B,))"""
        a = np.asarray(self.after[rows], dtype=np.float32) / 255.0
        i = np.asarray(self.imgin[self.group[rows]], dtype=np.float32) / 255.0
        x = np.concatenate([i, a], axis=-1).transpose(0, 3, 1, 2)
        return torch.from_numpy(np.ascontiguousarray(x)), \
            torch.from_numpy(self.preset[rows].astype(np.int64))


class Loader(torch.utils.data.Dataset):
    """Dataset wrapper so torch's worker pool does the memmap reads."""

    def __init__(self, cache: RenderCache, rows: np.ndarray):
        self.c, self.rows = cache, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = int(self.rows[i])
        a = torch.from_numpy(np.asarray(self.c.after[r], dtype=np.uint8))
        s = torch.from_numpy(
            np.asarray(self.c.imgin[self.c.group[r]], dtype=np.uint8))
        return s, a, int(self.c.preset[r])


def collate_to_gpu(batch, device, non_blocking=True):
    s = torch.stack([b[0] for b in batch]).to(device, non_blocking=non_blocking)
    a = torch.stack([b[1] for b in batch]).to(device, non_blocking=non_blocking)
    p = torch.tensor([b[2] for b in batch], device=device)
    s = s.permute(0, 3, 1, 2).float() / 255.0
    a = a.permute(0, 3, 1, 2).float() / 255.0
    return torch.cat([s, a], 1), s, p


def load_rows_meta(root: str = CACHE) -> list:
    with open(os.path.join(root, "rows.jsonl")) as f:
        return [json.loads(l) for l in f]
