"""C0 -- the canonical LUT function code (EPR-031 §3.1).

```
r_l    = vec{ L_l(x) - x }_{x in X}      X = 17^3 uniform sRGB grid, dim = 14,739
c*_l   = WhitenedPCA_k(r_l)              fitted on the TRAIN LUT pool only
```

Three properties this module exists to keep true.

**1. The fit set is the train LUT pool.**  :func:`train_lut_ids` collects the
unique ``lut_id`` of the ``train`` split index (3,149 of them).  The
LUT-disjoint test split ``T_lut_unseen`` shares **zero** ids with it, and
:func:`assert_fit_set_clean` is what says so at fit time rather than in a
docstring.  ``V_what`` (531 ids) and ``T_final`` (577 ids) are *sample*-level
splits whose ``lut_id`` sets are complete subsets of train's -- see
:func:`eval_only_lut_ids` and the C0+C1 NOTES; ``--exclude-eval-luts`` is the
flag that drops them, and it is **off** by default because turning it on
contradicts §1's "canonical code 拟合集 = train 的 3,149 条".

**2. One SVD, three ``d_LUT``.**  The pre-registered ladder 128 / 192 / 256 is
three *prefixes* of the same decomposition, never three fits: PCA components are
ordered, so ``c*[:, :128]`` is bit-identical to a 128-component fit's output.
:meth:`WhitenedPCA.transform` takes ``k`` for exactly this reason.

**3. ``code_recon_de00`` is geometry, not a training result.**  It is the dE00
between ``x + r_l`` (the LUT itself, on the same grid and the same operator the
carrier is trained against) and ``x + inverse(c*_l)``.  Nothing here is
learnable, so the number is a property of the code's dimension alone.

Whitening.  ``c* = (r - mu) V / sqrt(lambda)`` with ``lambda`` the per-component
variance (``S^2 / (n - 1)``); the inverse multiplies by ``sqrt(lambda)`` again.
Whitening is a per-axis rescaling, so it changes neither the explained-variance
ratios nor the reconstruction -- it only stops the VLM read-out from being asked
to regress coordinates whose scales differ by three orders of magnitude.

Everything numerical here runs in **float64** (``n = 3,149`` against
``d = 14,739`` is a rank-deficient SVD; float32 loses the tail of the spectrum
that the 99 %-variance column reads).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from q3vl.whatb import splits as S
from q3vl.whatb.colorimetry import delta_e00_srgb
from q3vl.whatb.lutdata import BANK_DIR, LutBank
from q3vl.whatb.queries import DEFAULT_SEED, uniform_grid

__all__ = [
    "CODE_DIR",
    "GRID_MAIN",
    "GRID_CONTROL",
    "D_LUT_CHOICES",
    "D_LUT_DEFAULT",
    "LEDGER_9CUBED_DIMS",
    "LEDGER_9CUBED_ROWS",
    "LEDGER_9CUBED_IDS",
    "VARIANCE_TARGETS",
    "WhitenedPCA",
    "assert_fit_set_clean",
    "code_recon_de00",
    "control_lut_ids",
    "cumulative_dims",
    "eval_only_lut_ids",
    "fit_whitened_pca",
    "load_code",
    "lut_residual_matrix",
    "open_bank",
    "pca_from_arrays",
    "sha256_file",
    "train_lut_ids",
]

#: where ``build_lut_code.py`` writes ``code.npz`` / ``manifest.json``
CODE_DIR = Path("/home/bc/data/runs/whatb/lutcode_17c_pca")

#: EPR-031 §3.1: the main grid (14,739-dim residual)
GRID_MAIN: int = 17
#: the control grid the repository's existing ledger was measured on (2,187-dim)
GRID_CONTROL: int = 9

#: §6.2's pre-registered ladder, and the report's default
D_LUT_CHOICES: tuple[int, ...] = (128, 192, 256)
D_LUT_DEFAULT: int = 192

#: the cumulative-variance columns every PCA table in this campaign reports
VARIANCE_TARGETS: tuple[float, ...] = (0.90, 0.95, 0.99)

#: the repository's existing ledger, verbatim (``EPR-024:659`` and five more
#: proposals, word for word): 9^3 grid, 2,187 dims, 1,137 lut_id drawn as the
#: unique ids of 2,500 random train index rows, cumulative variance 90/95/99 %.
#: Reproduced side by side, **never tuned to match** -- the draw's seed was not
#: recorded, so the id count of a re-draw is itself a measured number.
LEDGER_9CUBED_DIMS: tuple[int, int, int] = (15, 28, 99)
LEDGER_9CUBED_ROWS: int = 2500
LEDGER_9CUBED_IDS: int = 1137


# --------------------------------------------------------------------------- #
# 1. the LUT pools
# --------------------------------------------------------------------------- #
def train_lut_ids(split: str = "train", root: str | Path = S.DATASET_ROOT
                  ) -> list[str]:
    """Sorted unique ``lut_id`` of the train split index (measured, 3,149)."""
    return sorted({r.lut_id for r in S.load_index_cached(split, root) if r.lut_id})


def eval_only_lut_ids(root: str | Path = S.DATASET_ROOT,
                      splits_: Sequence[str] = ("V_what", "T_final",
                                                "T_lut_unseen"),
                      ) -> dict[str, list[str]]:
    """``split -> its lut_id`` for the three splits the fit set must respect.

    Returned whole (not just the train-disjoint part) so a caller can report both
    the set size and its intersection with train instead of only the difference.
    """
    return {sp: sorted({r.lut_id for r in S.load_index_cached(sp, root) if r.lut_id})
            for sp in splits_}


def control_lut_ids(rows: int = LEDGER_9CUBED_ROWS, seed: int = DEFAULT_SEED,
                    split: str = "train", root: str | Path = S.DATASET_ROOT
                    ) -> list[str]:
    """The ledger's ``Lib_tr`` protocol: unique ids of ``rows`` random index rows.

    ``random.Random(seed).sample(index_rows, rows)`` -- the same draw shape
    ``run_carrier_arm._library_ids`` uses.  The published ledger says the draw
    produced 1,137 ids; the seed behind it was never recorded, so the count this
    reproduction gets is reported as measured.
    """
    import random

    index = list(S.load_index_cached(split, root))
    picked = random.Random(int(seed)).sample(index, min(int(rows), len(index)))
    return sorted({r.lut_id for r in picked if r.lut_id})


def assert_fit_set_clean(fit_ids: Iterable[str], *,
                         root: str | Path = S.DATASET_ROOT,
                         forbidden: Sequence[str] = ("T_lut_unseen",),
                         ) -> dict[str, Any]:
    """The fit set carries no ``lut_id`` of a LUT-disjoint evaluation split.

    ``T_lut_unseen`` is the split whose LUT ids are disjoint from train by
    construction (measured: intersection 0 of 259).  ``V_what`` / ``T_final`` are
    sample-level splits drawn from the same LUT library, so their ids are a
    subset of train's and cannot be excluded without shrinking the fit set below
    §1's 3,149 -- that is a flag (``--exclude-eval-luts``), not a silent default.
    """
    ids = set(map(str, fit_ids))
    pools = eval_only_lut_ids(root)
    report: dict[str, Any] = {"n_fit": len(ids)}
    for sp, pool in pools.items():
        inter = ids & set(pool)
        report[sp] = {"n_split_ids": len(pool), "n_in_fit_set": len(inter)}
    for sp in forbidden:
        n = report[sp]["n_in_fit_set"]
        if n:
            raise AssertionError(
                f"the PCA fit set carries {n} lut_id of {sp}, which is a "
                "LUT-disjoint evaluation split; the canonical code would then be "
                "fitted on the very LUTs it is tested on (EPR-031 §1)")
    report["forbidden_splits"] = list(forbidden)
    return report


# --------------------------------------------------------------------------- #
# 2. the residual matrix
# --------------------------------------------------------------------------- #
def lut_residual_matrix(bank: LutBank, lut_ids: Sequence[str], *,
                        n_grid: int = GRID_MAIN,
                        dtype: np.dtype | type = np.float64,
                        ) -> tuple[np.ndarray, np.ndarray]:
    """``(R (n_lut, 3 n_grid^3), grid (n_grid^3, 3))``.

    ``R[i] = vec{L_i(x) - x}`` with ``L`` evaluated by the bank's own operator
    (``grid_sample`` bilinear / border / align_corners=True -- the generator's
    call, ``rendering.py:390-405``).  The row layout is the ``(P, 3)`` block's
    C order, so ``R[i].reshape(P, 3)`` is the residual per grid colour.
    """
    grid = uniform_grid(int(n_grid), dtype=torch.float32)
    p = grid.shape[0]
    out = np.empty((len(lut_ids), p * 3), dtype=dtype)
    for i, lid in enumerate(lut_ids):
        values = bank.apply(grid, lid)                       # (P, 3)
        out[i] = (values - grid).reshape(-1).to(torch.float64).numpy()
    return out, grid.to(torch.float64).numpy()


# --------------------------------------------------------------------------- #
# 3. whitened PCA
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WhitenedPCA:
    """A fitted whitened PCA.  ``components`` are ordered, so ``k`` is a prefix."""

    mean: np.ndarray                     # (d,)
    components: np.ndarray               # (k_max, d), rows orthonormal
    explained_variance: np.ndarray       # (k_max,)  = S^2 / (n - 1)
    explained_variance_ratio: np.ndarray  # (k_max,) sums to 1 over the full rank
    whiten_scale: np.ndarray             # (k_max,)  = 1 / sqrt(explained_variance)
    n_samples: int
    total_variance: float

    @property
    def n_components(self) -> int:
        return int(self.components.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.components.shape[1])

    def _k(self, k: int | None) -> int:
        k = self.n_components if k is None else int(k)
        if not (1 <= k <= self.n_components):
            raise ValueError(
                f"d_LUT must lie in [1, {self.n_components}]; got {k}")
        return k

    def transform(self, x: np.ndarray, k: int | None = None) -> np.ndarray:
        """``(n, d) -> (n, k)`` whitened canonical coordinates."""
        k = self._k(k)
        xc = np.asarray(x, dtype=np.float64) - self.mean
        return (xc @ self.components[:k].T) * self.whiten_scale[:k]

    def inverse_transform(self, c: np.ndarray, k: int | None = None) -> np.ndarray:
        """``(n, k) -> (n, d)``; the exact inverse of :meth:`transform` at rank k."""
        c = np.asarray(c, dtype=np.float64)
        k = self._k(k if k is not None else c.shape[-1])
        return (c[:, :k] / self.whiten_scale[:k]) @ self.components[:k] + self.mean

    def facts(self) -> dict[str, Any]:
        return {"n_samples": int(self.n_samples),
                "n_features": self.n_features,
                "n_components": self.n_components,
                "total_variance": float(self.total_variance),
                "whiten": "c = (r - mu) V / sqrt(lambda), lambda = S^2 / (n - 1)"}


def fit_whitened_pca(x: np.ndarray, n_components: int | None = None
                     ) -> WhitenedPCA:
    """Whitened PCA in float64.  ``n_components`` defaults to ``min(n - 1, d)``.

    The rank of a mean-centred ``(n, d)`` matrix is at most ``n - 1``; asking for
    more components than that returns directions whose variance is numerically
    zero and whose whitening scale is ``1 / 0``.  The cap is therefore a hard
    error, not a clamp.
    """
    xm = np.asarray(x, dtype=np.float64)
    if xm.ndim != 2:
        raise ValueError(f"x must be (n, d); got {xm.shape}")
    n, d = xm.shape
    if n < 2:
        raise ValueError(f"a PCA needs at least 2 samples; got {n}")
    k_max = int(min(n - 1, d))
    k = k_max if n_components is None else int(n_components)
    if not (1 <= k <= k_max):
        raise ValueError(
            f"n_components must lie in [1, min(n-1, d) = {k_max}]; got {k}")
    mean = xm.mean(axis=0)
    xc = xm - mean
    # full_matrices=False keeps U at (n, min(n,d)); the rank-deficient tail is
    # dropped explicitly below rather than left to produce 1/0 whitening scales.
    _u, s, vt = np.linalg.svd(xc, full_matrices=False)
    var = (s ** 2) / (n - 1)
    total = float(var.sum())
    if total <= 0:
        raise ValueError("the residual matrix has zero total variance")
    return WhitenedPCA(
        mean=mean,
        components=np.ascontiguousarray(vt[:k]),
        explained_variance=np.ascontiguousarray(var[:k]),
        # the ratio is reported over the FULL rank (it must sum to 1), so a
        # 90/95/99 % column is not silently computed against a truncated total
        explained_variance_ratio=np.ascontiguousarray(var[:k_max] / total),
        whiten_scale=np.ascontiguousarray(1.0 / np.sqrt(var[:k])),
        n_samples=int(n),
        total_variance=total)


def cumulative_dims(explained_variance_ratio: np.ndarray,
                    targets: Sequence[float] = VARIANCE_TARGETS
                    ) -> dict[str, int]:
    """``{"90": k, "95": k, "99": k}`` -- components to reach each cumulative %.

    ``k`` is the **smallest** number of leading components whose cumulative
    ratio is ``>= target``; a target the spectrum never reaches records the full
    rank (it cannot be exceeded) rather than a silent -1.
    """
    evr = np.asarray(explained_variance_ratio, dtype=np.float64)
    cum = np.cumsum(evr)
    out: dict[str, int] = {}
    for t in targets:
        hit = np.searchsorted(cum, float(t), side="left") + 1
        out[f"{round(float(t) * 100)}"] = int(min(hit, evr.size))
    return out


# --------------------------------------------------------------------------- #
# 4. the geometric read-out
# --------------------------------------------------------------------------- #
def code_recon_de00(pca: WhitenedPCA, residual: np.ndarray, grid: np.ndarray, *,
                    d_lut: int, chunk: int = 64, clamp: bool = False
                    ) -> dict[str, Any]:
    """``dE00( x + r , x + inverse(transform(r, k)) )`` on the fit grid.

    Purely geometric: no carrier, no training, no conditioning.  ``clamp`` gates
    the ``[0,1]`` clamp on the reconstruction -- the *unclamped* number is the
    primary one (it measures the code, not a display transform) and the clamped
    one is reported next to it because the carrier's own forward clamps.
    """
    r = np.asarray(residual, dtype=np.float64)
    g = torch.from_numpy(np.asarray(grid, dtype=np.float64))          # (P, 3)
    p = g.shape[0]
    total = 0.0
    count = 0
    per_lut: list[float] = []
    for start in range(0, r.shape[0], int(chunk)):
        block = r[start:start + int(chunk)]
        code = pca.transform(block, d_lut)
        rec = pca.inverse_transform(code, d_lut)
        y_hat = g.unsqueeze(0) + torch.from_numpy(rec).reshape(-1, p, 3)
        y = g.unsqueeze(0) + torch.from_numpy(block).reshape(-1, p, 3)
        if clamp:
            y_hat = y_hat.clamp(0.0, 1.0)
        de = delta_e00_srgb(y_hat, y)                                 # (n, P)
        per_lut.extend(de.mean(dim=-1).tolist())
        total += float(de.sum())
        count += de.numel()
    arr = np.asarray(per_lut, dtype=np.float64)
    return {"d_lut": int(d_lut), "clamped": bool(clamp),
            "mean": total / max(1, count),
            "per_lut_mean_median": float(np.median(arr)) if arr.size else 0.0,
            "per_lut_mean_p95": float(np.quantile(arr, 0.95)) if arr.size else 0.0,
            "per_lut_mean_max": float(arr.max()) if arr.size else 0.0,
            "n_lut": int(r.shape[0]), "n_points": int(p)}


# --------------------------------------------------------------------------- #
# 5. artefact I/O
# --------------------------------------------------------------------------- #
def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_code(code_dir: str | Path = CODE_DIR) -> tuple[dict[str, np.ndarray],
                                                        dict[str, Any]]:
    """``(arrays of code.npz, manifest.json)`` -- the frozen C0 artefact."""
    d = Path(code_dir)
    with np.load(d / "code.npz", allow_pickle=False) as z:
        arrays = {k: np.asarray(z[k]) for k in z.files}
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    return arrays, manifest


def pca_from_arrays(arrays: Mapping[str, np.ndarray]) -> WhitenedPCA:
    """Rebuild a :class:`WhitenedPCA` from a loaded ``code.npz``."""
    return WhitenedPCA(
        mean=np.asarray(arrays["mean"], dtype=np.float64),
        components=np.asarray(arrays["components"], dtype=np.float64),
        explained_variance=np.asarray(arrays["explained_variance"],
                                      dtype=np.float64),
        explained_variance_ratio=np.asarray(arrays["explained_variance_ratio"],
                                            dtype=np.float64),
        whiten_scale=np.asarray(arrays["whiten_scale"], dtype=np.float64),
        n_samples=int(arrays["c_star"].shape[0]),
        total_variance=float(arrays["total_variance"]))


def open_bank(bank_dir: str | Path = BANK_DIR, *, cache_size: int = 8) -> LutBank:
    """The preset bank, with a small LRU (C0 touches every LUT exactly once)."""
    return LutBank(bank_dir, cache_size=int(cache_size))
