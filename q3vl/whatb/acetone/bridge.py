"""The one place that touches the external AceTone clone.

Everything here is import-safe in **both** environments this experiment uses:

* ``/home/bc/envs/q3vl_sft/bin/python`` -- the campaign environment.  Scores the
  boards with the repository's own criteria (``q3vl.whatb.criteria`` /
  ``colorimetry`` / ``lutdata`` / ``publish``).  It does **not** have
  ``skimage`` / ``qwen_vl_utils`` and nothing is installed into it.
* ``/home/bc/data/external/acetone/venv/bin/python`` -- python 3.11 + the
  AceTone ``requirements.txt`` pins (``transformers==4.50.0``,
  ``qwen-vl-utils==0.0.8``, ``scikit-image``, torch cu128).  Runs AceTone's own
  code; writes predicted LUTs to disk and nothing else.

Facts about the external repo, all read out of the clone at
``/home/bc/data/external/acetone/repo`` (branch ``open-source-ready``, commit
``916393b3f26bdf89c3d939cc5f2a9a3c115ccbc5``):

1. ``useful_tools/convert_luts.py:read_cube_file`` reshapes the ``.cube`` file's
   line order straight into ``(size, size, size, 3)``.  ``.cube`` stores red
   fastest, so that array is indexed ``grid[b, g, r, :]`` -- the *same* storage
   order as this repository's ``q3vl/whatb/lutdata.py`` docstring declares.
   ``save_cube_file`` iterates the axes under the names ``r, g, b`` in the
   opposite direction, so its read/write round trip is consistent while the
   variable names are transposed.
2. ``useful_tools/convert_luts.py:resize_lut`` is the resampler the LUT
   tokenizer's training distribution went through: ``RegularGridInterpolator``
   over ``linspace(0, 1, orig_size)`` per axis with ``fill_value=None``
   (linear extrapolation), evaluated on ``linspace(0, 1, 32)``.
3. ``dataset/lut3d.py:LUTDataset`` feeds the VQ-VAE ``np.transpose(arr,
   (3,0,1,2))`` of that array, i.e. ``(3, D_b, D_g, D_r)``; ``model/vq.py``
   returns the same layout from ``decode_indices``.
4. ``dataset/lut3d.py:apply_lut`` -- their *image* applier -- reverses the
   colour channels (``lut = lut[..., ::-1]``) and then queries the interpolator
   with the pixel's ``(R, G, B)`` against array axes ``(0, 1, 2)``.  Whether
   that agrees with this repository's applier is the ``A_axis`` measurement
   (:func:`axis_parity`), not an assumption: ``tools/cube/NOTES.md`` §四·4
   flagged exactly this ``lut[..., ::-1]`` as a risk.
5. ``dataset/qwen_data/__init__.py`` of the released branch does **not** define
   ``get_path``, while ``dataset/lut3d.py:22`` and ``eval/predict_lut_ddp.py``
   import it.  :func:`ensure_get_path` installs the two-line shim that reads the
   repo's own ``acetone_paths.json``; the clone's files are never edited.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

__all__ = [
    "ACETONE_ROOT",
    "ACETONE_REPO",
    "ACETONE_VENV_PY",
    "VQ_CKPT",
    "PST_MODEL_DIR",
    "PST_PROMPT",
    "GENERATION_KWARGS",
    "repo_facts",
    "ensure_repo_on_path",
    "ensure_get_path",
    "load_lutdata",
    "resize_lut_acetone",
    "identity_grid",
    "grid_to_volume",
    "load_vq",
    "vq_roundtrip",
    "axis_parity",
]

#: where the external material lives (nothing outside this tree is written)
ACETONE_ROOT = Path("/home/bc/data/external/acetone")
ACETONE_REPO = ACETONE_ROOT / "repo"
ACETONE_VENV_PY = ACETONE_ROOT / "venv" / "bin" / "python"

#: the tokenizer weights that ship inside the clone (~4M params)
VQ_CKPT = ACETONE_REPO / "model" / "acetone-vqvae-d64.pt"

#: huggingface.co/Vivre/AceTone-3B-PST-Preview, downloaded 2026-08-26
PST_MODEL_DIR = ACETONE_ROOT / "AceTone-3B-PST-Preview"

#: ``eval/predict_lut_ddp.py`` verbatim -- the text field of the user turn.
#: Copied character for character, including the backslash-escaped quotes.
PST_PROMPT = (
    "The first image is an un-touched raw image, and the second is toned with "
    "stylish LUTs. These two images may have the same source or not, and your "
    "task is to mimic the toning method. You are a professional color grader. "
    "Please generate the 64-bit LUT in \\'Global toning: <SoT>...<EoT>\\'."
)

#: ``eval/predict_lut_ddp.py`` verbatim
GENERATION_KWARGS: dict[str, Any] = {"max_new_tokens": 128, "do_sample": True,
                                     "temperature": 0.01}

#: the token id block ``model/modeling_acetone.py`` reserves for the LUT codes
MM_TOKEN_BASE = 151667
MM_VOCAB_SIZE = 256


def repo_facts() -> dict[str, Any]:
    """Everything a board needs to say *which* external code produced a number."""
    import subprocess

    try:
        commit = subprocess.run(["git", "-C", str(ACETONE_REPO), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True
                                ).stdout.strip()
        branch = subprocess.run(["git", "-C", str(ACETONE_REPO), "rev-parse",
                                 "--abbrev-ref", "HEAD"],
                                capture_output=True, text=True, check=True
                                ).stdout.strip()
    except Exception as exc:                                # pragma: no cover
        commit, branch = f"unavailable: {exc}", "unavailable"
    return {
        "upstream": "https://github.com/martian422/AceTone",
        "branch": branch, "commit": commit,
        "clone": str(ACETONE_REPO),
        "paper": "arXiv:2604.00530",
        "vq_ckpt": str(VQ_CKPT),
        "vq_ckpt_sha256": _sha256(VQ_CKPT) if VQ_CKPT.is_file() else None,
        "vlm_weights": "huggingface.co/Vivre/AceTone-3B-PST-Preview",
        "vlm_dir": str(PST_MODEL_DIR),
        "resampler": ("useful_tools/convert_luts.py:resize_lut -- "
                      "RegularGridInterpolator(linspace(0,1,D)^3, fill_value=None) "
                      "on linspace(0,1,32)^3"),
        "applier": ("q3vl/whatb/lutdata.py:apply_lut_volume (this repository's "
                    "operator); AceTone's dataset/lut3d.py:apply_lut is measured "
                    "against it by A_axis and not used for any published number"),
        "igg_branch_released": False,
        "igg_note": ("only the PST (paired-style-transfer) head is public; the "
                     "instruction (IGG) branch has no released weights"),
    }


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# the external clone
# --------------------------------------------------------------------------- #
def ensure_repo_on_path(repo: Path | str = ACETONE_REPO) -> Path:
    repo = Path(repo)
    if not repo.is_dir():
        raise FileNotFoundError(
            f"{repo} does not exist; clone "
            "https://github.com/martian422/AceTone at branch open-source-ready")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo


def ensure_get_path(repo: Path | str = ACETONE_REPO) -> None:
    """Install the ``dataset.qwen_data.get_path`` the released branch is missing.

    ``dataset/lut3d.py:22`` and ``eval/predict_lut_ddp.py:36`` both do
    ``from dataset.qwen_data import get_path`` and
    ``dataset/qwen_data/__init__.py`` defines no such name, so *every* import of
    their LUT helpers raises ``ImportError`` on a fresh clone.  The shim reads
    the repo's own ``acetone_paths.json`` (the file those keys name) and falls
    back to the caller's default -- it adds no behaviour beyond what the missing
    function's call sites already declare, and the clone stays unmodified.
    """
    repo = ensure_repo_on_path(repo)
    import dataset.qwen_data as qd                        # type: ignore

    if hasattr(qd, "get_path"):
        return
    paths_file = repo / "acetone_paths.json"
    table: dict[str, Any] = {}
    if paths_file.is_file():
        table = json.loads(paths_file.read_text(encoding="utf-8"))

    def get_path(key: str, default: Any = None) -> Any:
        value = table.get(key, default)
        return default if isinstance(value, str) and value.startswith("PATH_TO_") \
            else value

    qd.get_path = get_path                                 # type: ignore[attr-defined]


def acetone_lut3d() -> ModuleType:
    """``dataset/lut3d.py`` of the clone (needs ``skimage`` -> AceTone venv)."""
    ensure_get_path()
    import dataset.lut3d as m                             # type: ignore

    return m


def acetone_convert_luts() -> ModuleType:
    """``useful_tools/convert_luts.py`` (numpy + scipy only, safe in both envs)."""
    ensure_repo_on_path()
    import useful_tools.convert_luts as m                 # type: ignore

    return m


# --------------------------------------------------------------------------- #
# this repository's LUT operator, importable from the AceTone venv too
# --------------------------------------------------------------------------- #
_LUTDATA: ModuleType | None = None


def load_lutdata() -> ModuleType:
    """``q3vl/whatb/lutdata.py``, by file, so the AceTone venv can use it.

    The AceTone venv has torch and numpy but not the campaign package's other
    dependencies, and importing ``q3vl.whatb.lutdata`` normally would execute
    ``q3vl/whatb/__init__.py``.  Loading the same *file* under a private module
    name keeps ``apply_lut_volume`` byte-identical in both environments -- which
    is the whole point of the ``A_axis`` measurement.
    """
    global _LUTDATA
    if _LUTDATA is not None:
        return _LUTDATA
    try:
        from q3vl.whatb import lutdata as m               # type: ignore

        _LUTDATA = m
        return m
    except Exception:
        pass
    path = Path(__file__).resolve().parents[1] / "lutdata.py"
    spec = importlib.util.spec_from_file_location("_whatb_lutdata_standalone", path)
    if spec is None or spec.loader is None:               # pragma: no cover
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _LUTDATA = mod
    return mod


def grid_to_volume(grid: np.ndarray):
    """``(D,D,D,3)`` stored ``grid[b,g,r]`` -> the ``(1,3,D,D,D)`` volume."""
    import torch

    arr = np.ascontiguousarray(np.asarray(grid, dtype=np.float32))
    if arr.ndim != 4 or arr.shape[-1] != 3 or len(set(arr.shape[:3])) != 1:
        raise ValueError(f"expected a cubic (D,D,D,3) grid, got {arr.shape}")
    return torch.from_numpy(arr).permute(3, 0, 1, 2)[None].contiguous()


def identity_grid(size: int = 32) -> np.ndarray:
    """``grid[b,g,r] = (r, g, b)`` -- the identity in this storage order."""
    lin = np.linspace(0.0, 1.0, int(size), dtype=np.float32)
    b, g, r = np.meshgrid(lin, lin, lin, indexing="ij")
    return np.stack([r, g, b], axis=-1).astype(np.float32)


# --------------------------------------------------------------------------- #
# the tokenizer's own resampler
# --------------------------------------------------------------------------- #
def resize_lut_acetone(grid: np.ndarray, target_size: int = 32) -> np.ndarray:
    """``useful_tools/convert_luts.py:resize_lut``, called on their code.

    Storage order is irrelevant to the resampler (it interpolates the three
    spatial axes symmetrically over ``[0, 1]``), so the ``grid[b,g,r]`` layout
    survives it unchanged.  ``fill_value=None`` means linear extrapolation --
    it never triggers here because ``linspace(0,1,32)`` is inside the domain.
    """
    mod = acetone_convert_luts()
    out = mod.resize_lut(np.asarray(grid, dtype=np.float64), target_size=int(target_size))
    return np.asarray(out, dtype=np.float32)


# --------------------------------------------------------------------------- #
# the tokenizer
# --------------------------------------------------------------------------- #
def load_vq(ckpt_path: Path | str = VQ_CKPT, *, device: Any = "cpu"):
    """``model/vq.py:VQVAE3DLUT`` with the in-repo weights.

    Loaded exactly the way ``eval/predict_lut_ddp.py:75-85`` loads it, including
    reading ``codebook_size`` / ``embedding_dim`` out of the checkpoint's own
    ``args``.  ``weights_only=False`` is required because the checkpoint stores
    that ``args`` dict next to the tensors.
    """
    import torch

    ensure_repo_on_path()
    from model.vq import VQVAE3DLUT                       # type: ignore

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    if not isinstance(args, dict):
        args = vars(args)
    model = VQVAE3DLUT(codebook_size=int(args.get("codebook_size", 256)),
                       embedding_dim=int(args.get("embedding_dim", 64)))
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    facts = {"ckpt": str(ckpt_path),
             "codebook_size": int(args.get("codebook_size", 256)),
             "embedding_dim": int(args.get("embedding_dim", 64)),
             "n_params": int(sum(p.numel() for p in model.parameters())),
             "latent_grid": [4, 4, 4], "n_tokens": 64}
    return model, facts


def vq_roundtrip(model, grids: np.ndarray, *, device: Any = "cpu"):
    """``(N,32,32,32,3)`` -> ``(indices (N,4,4,4), recon (N,32,32,32,3))``.

    The channel transpose is ``dataset/lut3d.py:LUTDataset.__getitem__``'s
    (``(3,0,1,2)`` on a ``(32,32,32,3)`` array, then ``clip(0,1)``); the inverse
    transpose is applied to the decoder output.
    """
    import torch

    x = np.asarray(grids, dtype=np.float32)
    if x.ndim == 4:
        x = x[None]
    x = np.clip(x, 0.0, 1.0)
    t = torch.from_numpy(np.transpose(x, (0, 4, 1, 2, 3)).copy()).to(device)
    with torch.no_grad():
        idx = model.encode_indices(t)
        rec = model.decode_indices(idx)
    rec_np = np.transpose(rec.detach().cpu().numpy(), (0, 2, 3, 4, 1))
    return idx.detach().cpu().numpy().astype(np.int64), rec_np.astype(np.float32)


def decode_indices(model, indices: np.ndarray, *, device: Any = "cpu") -> np.ndarray:
    """``(N,64)`` or ``(N,4,4,4)`` token ids -> ``(N,32,32,32,3)`` LUTs.

    The reshape is ``eval/predict_lut_ddp.py:158`` verbatim:
    ``prediction_ids_flatten[:64].reshape(4,4,4)``.
    """
    import torch

    idx = np.asarray(indices, dtype=np.int64)
    if idx.ndim == 2:
        idx = idx.reshape(idx.shape[0], 4, 4, 4)
    t = torch.from_numpy(idx).to(device)
    with torch.no_grad():
        rec = model.decode_indices(t)
    return np.transpose(rec.detach().cpu().numpy(), (0, 2, 3, 4, 1)).astype(np.float32)


# --------------------------------------------------------------------------- #
# A_axis
# --------------------------------------------------------------------------- #
def axis_parity(*, size: int = 32, image_size: int = 64, seed: int = 20260826
                ) -> dict[str, Any]:
    """The pre-registered ``A_axis`` measurement.  Needs the AceTone venv.

    Two LUTs on the same random 8-bit image: the identity, and a strongly
    asymmetric one that raises only the red channel (``r -> sqrt(r)``).  Each is
    applied twice -- once by this repository's ``apply_lut_volume`` and once by
    ``dataset/lut3d.py:apply_lut`` -- and the per-pixel ``max |Δ|`` of the two
    results is reported next to each applier's own distance from the analytic
    answer.  The identity LUT alone cannot see a transposed axis order; the
    asymmetric one is what makes the measurement able to fail.

    Nothing here decides anything: the numbers go on the board.
    """
    from PIL import Image

    lut3d = acetone_lut3d()
    lutdata = load_lutdata()

    lin = np.linspace(0.0, 1.0, int(size), dtype=np.float32)
    b, g, r = np.meshgrid(lin, lin, lin, indexing="ij")
    luts = {"identity": np.stack([r, g, b], -1).astype(np.float32),
            "asym_sqrt_r": np.stack([np.sqrt(r), g, b], -1).astype(np.float32)}

    rng = np.random.default_rng(seed)
    img_u8 = rng.integers(0, 256, size=(int(image_size), int(image_size), 3),
                          dtype=np.uint8)
    import torch

    img = torch.from_numpy(img_u8.astype(np.float32) / 255.0)

    out: dict[str, Any] = {
        "quantity": ("max |Δ| per pixel between q3vl apply_lut_volume and "
                     "AceTone dataset/lut3d.py:apply_lut, on the same "
                     "grid[b,g,r] array"),
        "image": [int(image_size), int(image_size)], "lut_size": int(size),
        "seed": int(seed), "tolerance": 1e-5,
        "quantisation_floor": 1.0 / 255.0,
        "note": ("AceTone's applier returns a uint8 PIL image, so 1/255 = "
                 "0.00392 is the floor of any comparison with a float applier"),
        "cases": {}}
    for name, grid in luts.items():
        ours = lutdata.apply_lut_volume(grid_to_volume(grid), img).numpy()
        theirs = np.asarray(
            lut3d.apply_lut(Image.fromarray(img_u8), grid.copy())
        ).astype(np.float32) / 255.0
        # the same array with the R<->B conjugation ``apply_lut`` performs
        # (``lut[..., ::-1]`` on the values + (R,G,B) queried against axes
        # (0,1,2) of a grid[b,g,r] array) undone in advance
        conj = np.ascontiguousarray(grid.transpose(2, 1, 0, 3)[..., ::-1])
        theirs_conj = np.asarray(
            lut3d.apply_lut(Image.fromarray(img_u8), conj)
        ).astype(np.float32) / 255.0
        analytic = img.numpy().copy()
        if name == "asym_sqrt_r":
            analytic[..., 0] = np.sqrt(analytic[..., 0])
        out["cases"][name] = {
            "max_abs_delta_ours_vs_acetone": float(np.abs(ours - theirs).max()),
            "max_abs_delta_ours_vs_analytic": float(np.abs(ours - analytic).max()),
            "max_abs_delta_acetone_vs_analytic": float(np.abs(theirs - analytic).max()),
            "max_abs_delta_ours_vs_acetone_rb_conjugated":
                float(np.abs(ours - theirs_conj).max()),
        }
    out["passed"] = all(c["max_abs_delta_ours_vs_acetone"] < out["tolerance"]
                        for c in out["cases"].values())
    out["passed_rb_conjugated"] = all(
        c["max_abs_delta_ours_vs_acetone_rb_conjugated"]
        <= out["quantisation_floor"] + 1e-6
        for c in out["cases"].values())
    return out


def cube_reader_parity(cube_path: Path | str) -> dict[str, Any]:
    """``read_cube_file`` (AceTone) vs ``dataset_build.lut_io.load_lut`` (here).

    Array-level, no interpolation and no 8-bit quantisation in the way, so a
    transposed axis order shows up as a large ``max |Δ|`` and nothing else can.
    """
    from dataset_build.lut_io import load_lut               # clean tree, read-only

    mod = acetone_convert_luts()
    _size, theirs, _hdr = mod.read_cube_file(str(cube_path))
    ours = np.asarray(load_lut(str(cube_path))[0], dtype=np.float32)
    theirs = np.asarray(theirs, dtype=np.float32)
    return {"path": str(cube_path), "shape": list(ours.shape),
            "max_abs_delta": float(np.abs(ours - theirs).max()),
            "quantity": ("useful_tools/convert_luts.py:read_cube_file vs "
                         "dataset_build/lut_io.py:load_lut, same file")}
