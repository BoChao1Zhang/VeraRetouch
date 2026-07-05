"""
dataset_build/recipes.py
========================
Recipe parsers + LUT applier + raw decode + archive extraction for the
VeraRetouch Direction-A, RECIPE-BASED dataset build.

This module implements the two abstract interfaces declared in
``dataset_build/contracts.py``:
  - ``RecipeParser`` : on-disk recipe files -> VeraRetouch param dicts /
    LUT tensors / Track-B degrade specs.
  - ``LutApplier``   : 3D-LUT trilinear application via torch ``grid_sample``.

Plus a handful of pure helper functions (raw decode, archive extraction)
that the streams need but that the architect grouped under this module.

Design constraints (read the contract header + the probes):
  - The contract is the SINGLE source of truth for the schema. We import
    ``PARAM_KEYS`` / ``DegradeSpec`` / ``RecipeKind`` from it and NEVER
    redefine them here.
  - VeraRetouch ``param.json`` stores RAW Lightroom units, shape
    ``{key: {"value": <raw_number>}}``; ``get_organized_dict`` divides by 100
    at load (data/infer_dataset.py L229-294). So we emit RAW units, default 0.
  - Heavy deps (torch, rawpy, cv2) are imported LAZILY inside the methods that
    need them, so this module imports on any conda env (base / fivek-cleaning /
    monetgpt_sam3 / vllm). Only stdlib + the dep-free contract at top level.
  - Reuse the existing parsers:
      * XMP  -> presets/scripts/build_preset_dataset.parse_xmp_file logic
        (re-implemented self-contained with stdlib ElementTree so we do not
        import that 1100-line script; the CRS namespace + value rule are the
        only load-bearing bits — see probe_recipe_parsers §1a/§1c).
      * lrtemplate -> pe_kg/scripts/_lua_table_parser.LuaTableParser
        (vendored fallback if the path is unavailable).

Grounding:
  - docs/plan/dataset/probe/probe_recipe_parsers.md   (the whole module spec)
  - data/infer_dataset.py get_organized_dict L229-294 (the /100 + default-0 rule)
  - data_samples/param.json                            (raw-units target shape)
  - config.yaml: recipes.filters / degrade.* / cgt.* (thresholds, never hard-code)
"""

from __future__ import annotations

import math
import os
import random
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Dep-free contract: schema + interfaces. Safe on every env.
from dataset_build.contracts import (
    COLORMIXER_KEYS,
    COLORTEMP_KEYS,
    DegradeSpec,
    LIGHT_KEYS,
    LutApplier,
    PARAM_KEYS,
    RecipeParser,
)

# ---------------------------------------------------------------------------
# Constants grounded in the on-disk formats.
# ---------------------------------------------------------------------------

#: Adobe Camera Raw settings XML namespace (build_preset_dataset.py L35).
CRS_NS = "http://ns.adobe.com/camera-raw-settings/1.0/"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
NS = {"crs": CRS_NS, "rdf": RDF_NS}

#: Empty VeraRetouch param dict: all 38 keys default to {"value": 0.0}.
#: Mirrors get_organized_dict's default-0 behaviour (infer_dataset.py L229-294).
def _zero_params() -> Dict[str, Dict[str, float]]:
    return {k: {"value": 0.0} for k in PARAM_KEYS}


#: Lightroom's *legacy* HSL slider names (used by older .lrtemplate / .xmp) map
#: onto the modern HueAdjustment*/SaturationAdjustment*/LuminanceAdjustment*
#: param keys. The 8 color bands; LR legacy uses these exact stems.
_HSL_LEGACY_BANDS = ("Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta")
_HSL_LEGACY_MAP: Dict[str, str] = {}
for _band in _HSL_LEGACY_BANDS:
    _HSL_LEGACY_MAP[f"{_band}Hue"] = f"HueAdjustment{_band}"
    _HSL_LEGACY_MAP[f"{_band}Saturation"] = f"SaturationAdjustment{_band}"
    _HSL_LEGACY_MAP[f"{_band}Luminance"] = f"LuminanceAdjustment{_band}"

#: CRS sub-keys that, if present, mean the preset carries a *local* edit (a
#: Lightroom mask / radial / gradient / brush). We record this as a region hint
#: on the RecipeAsset; param-mode rendering itself is GLOBAL and ignores them.
LOCAL_MASK_MARKERS: Tuple[str, ...] = (
    "MaskGroupBasedCorrections",
    "CircularGradientBasedCorrections",
    "GradientBasedCorrections",
    "PaintBasedCorrections",
    "RetouchAreas",
)

#: Per-operator Gaussian sigma profile for Track-B degrade sampling.
#: Keyed by the "aether_tab8" profile name in config.yaml (degrade.sigma_profile).
#: Sigmas are in RAW LR units（Exposure2012 为 EV，其余 -100..+100 滑杆族）。
#: 2026-07-05: 逐值对齐论文 App A Tab 8（P0_001 PDF p.10，人工修图数据集统计导出的
#: 真实 σ），替换此前手调圆整值——旧值既非统计导出也与 profile 名不符。
#: 旧手调档保留为 "modest_legacy"（历史 shards S1/S7 用它产出，复现审计用）。
SIGMA_PROFILES: Dict[str, Dict[str, float]] = {
    "aether_tab8": {
        # Light Adjustment
        "Exposure2012": 0.6543,
        "Contrast2012": 12.6789,
        "Highlights2012": 21.5888,
        "Shadows2012": 16.2265,
        "Whites2012": 16.4355,
        "Blacks2012": 15.5995,
        "ParametricShadows": 7.2495,
        "ParametricDarks": 15.8214,
        "ParametricLights": 7.6688,
        "ParametricHighlights": 9.1287,
        # Global Color Adjustment
        "IncrementalTemperature": 15.0,
        "IncrementalTint": 15.0,
        "Vibrance": 7.8137,
        "Saturation": 7.4315,
        # Specific Color Adjustment（24 通道逐值）
        "HueAdjustmentRed": 5.8140,
        "HueAdjustmentOrange": 8.3549,
        "HueAdjustmentYellow": 15.1914,
        "HueAdjustmentGreen": 8.4875,
        "HueAdjustmentAqua": 19.8922,
        "HueAdjustmentBlue": 11.8419,
        "HueAdjustmentPurple": 10.1451,
        "HueAdjustmentMagenta": 19.0949,
        "SaturationAdjustmentRed": 19.8318,
        "SaturationAdjustmentOrange": 9.6656,
        "SaturationAdjustmentYellow": 18.2479,
        "SaturationAdjustmentGreen": 17.7113,
        "SaturationAdjustmentAqua": 7.4975,
        "SaturationAdjustmentBlue": 15.6967,
        "SaturationAdjustmentPurple": 21.7025,
        "SaturationAdjustmentMagenta": 27.8002,
        "LuminanceAdjustmentRed": 10.0289,
        "LuminanceAdjustmentOrange": 13.4234,
        "LuminanceAdjustmentYellow": 16.2116,
        "LuminanceAdjustmentGreen": 28.3202,
        "LuminanceAdjustmentAqua": 17.1250,
        "LuminanceAdjustmentBlue": 22.4162,
        "LuminanceAdjustmentPurple": 18.2913,
        "LuminanceAdjustmentMagenta": 25.4936,
    },
    "modest_legacy": {
        "Exposure2012": 0.50,
        "Contrast2012": 18.0,
        "Highlights2012": 25.0,
        "Shadows2012": 25.0,
        "Whites2012": 18.0,
        "Blacks2012": 18.0,
        "ParametricShadows": 15.0,
        "ParametricDarks": 15.0,
        "ParametricLights": 15.0,
        "ParametricHighlights": 15.0,
        "IncrementalTemperature": 20.0,
        "IncrementalTint": 15.0,
        "Vibrance": 25.0,
        "Saturation": 20.0,
    },
}
# legacy 档 HSL 带共享 σ（与历史产物一致）
for _k in COLORMIXER_KEYS:
    SIGMA_PROFILES["modest_legacy"].setdefault(_k, 18.0)

#: Which PARAM_KEYS belong to each Direction-A aspect (L / GC / SC). Used to
#: select the perturbed operators for a given degrade combo.
ASPECT_KEY_GROUPS: Dict[str, Tuple[str, ...]] = {
    "L": LIGHT_KEYS,
    "GC": COLORTEMP_KEYS,
    "SC": COLORMIXER_KEYS,
}


# ===========================================================================
# Pure helpers (no heavy deps).
# ===========================================================================

def _localname(tag: str) -> str:
    """Strip the ``{namespace}`` prefix from an ElementTree tag/attr name."""
    return tag.rsplit("}", 1)[1] if "}" in tag else tag


def _parse_num(raw: Any) -> Optional[float]:
    """Parse a raw LR value string/number -> float, stripping a leading '+'.

    Returns None if it is not numeric (e.g. "Custom", "Adobe Color"). Mirrors
    probe_recipe_parsers §1c: ``value = float(str.lstrip('+'))``.
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip()
    if not s:
        return None
    s = s.lstrip("+")
    try:
        return float(s)
    except ValueError:
        return None


def _crs_attrs_to_params(attrs: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Project a flat {CRS-key: raw-value} dict onto the 38 VeraRetouch keys.

    - Identity on the modern key name (Exposure2012, HueAdjustmentRed, ...).
    - Maps the LR *legacy* HSL names (BlueHue -> HueAdjustmentBlue, ...).
    - Drops every CRS key not in PARAM_KEYS (Texture, Dehaze, Clarity2012,
      SplitToning*, ColorGrade*, absolute Temperature/Tint Kelvin, ...).
    - Missing keys default to {"value": 0.0}.
    Values are kept in RAW LR units (get_organized_dict divides by 100 at load).
    """
    out = _zero_params()
    for key, raw in attrs.items():
        canon = key
        if canon not in PARAM_KEYS:
            canon = _HSL_LEGACY_MAP.get(key, "")
            if canon not in PARAM_KEYS:
                continue
        v = _parse_num(raw)
        if v is None:
            continue
        out[canon] = {"value": v}
    return out


def detect_local_mask(attrs_or_settings: Dict[str, Any]) -> bool:
    """True if the preset settings carry a Lightroom local-mask correction.

    Recorded as a region hint on RecipeAsset.has_local_mask; never sent to the
    GLOBAL param renderer (MEMORY caveat b; config.yaml recipes.local_mask_recipes).
    """
    return any(m in attrs_or_settings for m in LOCAL_MASK_MARKERS)


#: The local-edit param attributes Lightroom writes on a correction's
#: rdf:Description (the modern 2012 set + the few non-versioned ones that still
#: matter). Used by parse_local_masks to capture the edit *direction* per mask.
_LOCAL_PARAM_KEYS: Tuple[str, ...] = (
    "LocalExposure2012", "LocalContrast2012", "LocalHighlights2012",
    "LocalShadows2012", "LocalWhites2012", "LocalBlacks2012",
    "LocalClarity2012", "LocalDehaze", "LocalTemperature", "LocalTint",
    "LocalSaturation", "LocalToningHue", "LocalToningSaturation",
)


def _crs_attr(el: Any, name: str) -> Optional[str]:
    """Read a crs-namespaced attribute (compact form) off an element."""
    v = el.attrib.get(f"{{{CRS_NS}}}{name}")
    return v if v is not None else el.attrib.get(name)


def parse_local_masks(xmp_path: str) -> List[Dict[str, Any]]:
    """Deep-parse the local-mask corrections of an XMP preset.

    Returns one record (plain dict) per mask; a correction may hold several
    masks. Each record:
      {container, mask_type, what, geom {Top,Left,...|ZeroX,...},
       local_params {LocalExposure2012: float, ...}, correction_amount,
       range_mask {Type,LumMin,LumMax,ColorAmount}|None, is_ai}

    Unlike xmp_to_params (which keeps only the 38 GLOBAL keys), this reads the
    CorrectionMasks geometry + the Local*2012 edit vector that the mask-template
    miner needs to extract anchor templates. xmp_to_params is left untouched —
    GLOBAL param rendering still ignores all of this.
    """
    try:
        tree = ET.parse(xmp_path)
    except (ET.ParseError, OSError):
        return []
    root = tree.getroot()
    out: List[Dict[str, Any]] = []
    for container in LOCAL_MASK_MARKERS:
        for cont_el in root.iter(f"{{{CRS_NS}}}{container}"):
            for corr in cont_el.iter(f"{{{RDF_NS}}}Description"):
                ca = _parse_num(_crs_attr(corr, "CorrectionAmount"))
                local_params = {
                    k: v for k in _LOCAL_PARAM_KEYS
                    if (v := _parse_num(_crs_attr(corr, k))) is not None and v != 0.0
                }
                rm = corr.find(f"{{{CRS_NS}}}CorrectionRangeMask")
                range_mask = None
                if rm is not None:
                    range_mask = {
                        n: _parse_num(_crs_attr(rm, n))
                        for n in ("Type", "LumMin", "LumMax", "ColorAmount")
                    }
                cmasks = corr.find(f"{{{CRS_NS}}}CorrectionMasks")
                mask_lis = list(cmasks.iter(f"{{{RDF_NS}}}li")) if cmasks is not None else []
                for li in mask_lis:
                    what = _crs_attr(li, "What") or ""
                    geom = {
                        _localname(k): v for k, v in li.attrib.items()
                        if k.startswith(f"{{{CRS_NS}}}") and _localname(k) != "What"
                    }
                    out.append({
                        "container": container,
                        "mask_type": what.rsplit("/", 1)[-1].lower(),
                        "what": what,
                        "geom": geom,
                        "local_params": local_params,
                        "correction_amount": ca,
                        "range_mask": range_mask,
                        "is_ai": container == "MaskGroupBasedCorrections"
                                 or "image" in what.lower(),
                    })
    return out


def is_grayscale_preset(attrs: Dict[str, Any]) -> bool:
    """True if ``ConvertToGrayscale="True"`` — dropped for the color C_GT pilot
    (config.yaml recipes.filters.drop_bw)."""
    v = attrs.get("ConvertToGrayscale")
    return str(v).strip().lower() == "true"


# ===========================================================================
# RecipeParser implementation.
# ===========================================================================

class DiskRecipeParser(RecipeParser):
    """Concrete RecipeParser over the on-disk corpora.

    All methods are stateless/pure except for the lazy import of torch inside
    ``load_cube`` (LUT tensors must be torch). XMP/lrtemplate parsing uses only
    stdlib, so this class instantiates on any env.

    Parameters
    ----------
    sigma_profile:
        Name of the Track-B per-op sigma table (config.yaml degrade.sigma_profile).
    """

    def __init__(self, sigma_profile: str = "aether_tab8") -> None:
        if sigma_profile not in SIGMA_PROFILES:
            raise KeyError(
                f"unknown sigma_profile {sigma_profile!r}; "
                f"known: {sorted(SIGMA_PROFILES)}"
            )
        self.sigma_profile = sigma_profile

    # ---- XMP --------------------------------------------------------------
    def xmp_to_params(self, xmp_path: str) -> Dict[str, Dict[str, float]]:
        """Parse an Adobe XMP preset -> {key: {"value": raw_LR_number}}.

        Re-implements the load-bearing core of build_preset_dataset.parse_xmp_file
        (probe_recipe_parsers §1a/§1c): read the ``rdf:Description`` element's CRS
        attributes (and any CRS child elements that carry a numeric value), keep
        only the 38 PARAM_KEYS, ``value = float(str.lstrip('+'))``, default 0.
        """
        attrs = self._read_xmp_crs_attrs(xmp_path)
        return _crs_attrs_to_params(attrs)

    @staticmethod
    def _read_xmp_crs_attrs(xmp_path: str) -> Dict[str, str]:
        """Return the flat {localname: value} CRS dict from an XMP file.

        Handles both forms LR emits:
          (a) attributes on rdf:Description (the common compact form), and
          (b) CRS values written as child elements ``<crs:Key>val</crs:Key>``.
        """
        tree = ET.parse(xmp_path)
        root = tree.getroot()
        desc = root.find(".//rdf:Description", NS)
        if desc is None:
            raise ValueError(f"{xmp_path}: missing rdf:Description")

        attrs: Dict[str, str] = {}
        # (a) compact attribute form
        for k, v in desc.attrib.items():
            attrs[_localname(k)] = str(v)
        # (b) expanded child-element form (only crs:* leaves with text)
        for child in list(desc):
            if not child.tag.startswith(f"{{{CRS_NS}}}"):
                continue
            name = _localname(child.tag)
            if name in attrs:
                continue
            text = (child.text or "").strip()
            if text:
                attrs[name] = text
        return attrs

    def xmp_metadata(self, xmp_path: str) -> Dict[str, Any]:
        """Lightweight metadata for the registry: flags used by the junk/scene
        filters without re-parsing twice. Returns
        {is_bw, has_local_mask, white_balance, has_temp_relative}.
        """
        attrs = self._read_xmp_crs_attrs(xmp_path)
        return {
            "is_bw": is_grayscale_preset(attrs),
            "has_local_mask": detect_local_mask(attrs),
            "white_balance": attrs.get("WhiteBalance"),
            "has_temp_relative": "IncrementalTemperature" in attrs,
        }

    # ---- lrtemplate (Lua) -------------------------------------------------
    def lrtemplate_to_params(self, path: str) -> Dict[str, Dict[str, float]]:
        """Parse a Lightroom ``.lrtemplate`` (Lua table) -> param dict.

        The develop settings live under ``s.value.settings`` and use the SAME
        CRS key names (Exposure2012, ...) plus the LR *legacy* HSL names
        (BlueHue, BlueSaturation, ...) which we remap. Reuses the repo's
        pe_kg ``LuaTableParser`` (vendored fallback below if unavailable).
        """
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        table = _parse_lua_return_table(text)
        settings = self._extract_lr_settings(table)
        return _crs_attrs_to_params(settings)

    @staticmethod
    def _extract_lr_settings(table: Any) -> Dict[str, Any]:
        """Pull the develop ``settings`` dict out of a parsed .lrtemplate table.

        Layout: ``{ id=..., value = { settings = { Exposure2012=.., ... } } }``.
        Falls back to the top-level table if the nesting differs.
        """
        if not isinstance(table, dict):
            return {}
        val = table.get("value")
        if isinstance(val, dict):
            settings = val.get("settings")
            if isinstance(settings, dict):
                return settings
        # Some templates store settings at the top level.
        settings = table.get("settings")
        if isinstance(settings, dict):
            return settings
        return table

    def lrtemplate_metadata(self, path: str) -> Dict[str, Any]:
        """Registry-side flags for a .lrtemplate (mirrors xmp_metadata)."""
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        table = _parse_lua_return_table(text)
        settings = self._extract_lr_settings(table)
        return {
            "is_bw": is_grayscale_preset(settings),
            "has_local_mask": detect_local_mask(settings),
            "internal_name": table.get("internalName") if isinstance(table, dict) else None,
            "title": table.get("title") if isinstance(table, dict) else None,
        }

    # ---- .cube / .3dl -----------------------------------------------------
    def load_cube(
        self, path: str
    ) -> Tuple[Any, Tuple[float, float, float], Tuple[float, float, float]]:
        """Load a 3D-LUT (.cube or .3dl) -> (lut, domain_min, domain_max).

        Returns
        -------
        lut : torch.FloatTensor, shape ``[N, N, N, 3]``, indexed ``lut[b, g, r]``
            (slowest axis = Blue, fastest = Red), values in [0, 1]. This index
            order is the canonical one expected by ``DiskLutApplier.apply_lut``.
        domain_min, domain_max : 3-tuples (per-channel input domain; (0,0,0)/
            (1,1,1) for the typical creative LUT).

        Format notes (probe_recipe_parsers §2):
          - .cube  : ``LUT_3D_SIZE N``, N^3 ``R G B`` float lines, RED-fastest.
          - .3dl   : ``Mesh`` header + node row, 17^3 ``R G B`` int lines,
            BLUE-fastest, output ints /1023 (10-bit).
        """
        import torch  # lazy

        ext = Path(path).suffix.lower()
        if ext == ".cube":
            grid, dmin, dmax = self._parse_cube(path)
        elif ext == ".3dl":
            grid, dmin, dmax = self._parse_3dl(path)
        else:
            raise ValueError(f"unsupported LUT extension {ext!r} for {path}")

        lut = torch.tensor(grid, dtype=torch.float32)  # [N,N,N,3] in [b,g,r]
        return lut, dmin, dmax

    @staticmethod
    def _parse_cube(
        path: str,
    ) -> Tuple[List[List[List[List[float]]]], Tuple[float, float, float], Tuple[float, float, float]]:
        """Parse an Adobe/Resolve .cube into a nested ``[b][g][r][3]`` list.

        RED-fastest on disk: line index = r + g*N + b*N*N. We therefore fill
        ``grid[b][g][r]`` directly.
        """
        size: Optional[int] = None
        dmin = [0.0, 0.0, 0.0]
        dmax = [1.0, 1.0, 1.0]
        rows: List[Tuple[float, float, float]] = []
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                head = s.split()[0].upper()
                if head == "TITLE":
                    continue
                if head == "LUT_3D_SIZE":
                    size = int(s.split()[1])
                    continue
                if head == "LUT_1D_SIZE":
                    raise ValueError(f"{path}: 1D LUT not supported (LUT_1D_SIZE)")
                if head == "DOMAIN_MIN":
                    dmin = [float(x) for x in s.split()[1:4]]
                    continue
                if head == "DOMAIN_MAX":
                    dmax = [float(x) for x in s.split()[1:4]]
                    continue
                parts = s.split()
                if len(parts) >= 3:
                    try:
                        rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
                    except ValueError:
                        continue
        if size is None:
            raise ValueError(f"{path}: missing LUT_3D_SIZE header")
        if len(rows) != size ** 3:
            raise ValueError(
                f"{path}: expected {size**3} entries, found {len(rows)}"
            )
        grid = DiskRecipeParser._rows_to_bgr_grid(rows, size, red_fastest=True)
        return grid, (dmin[0], dmin[1], dmin[2]), (dmax[0], dmax[1], dmax[2])

    @staticmethod
    def _parse_3dl(
        path: str,
    ) -> Tuple[List[List[List[List[float]]]], Tuple[float, float, float], Tuple[float, float, float]]:
        """Parse an Autodesk/Picture-Instruments .3dl into ``[b][g][r][3]``.

        BLUE-fastest on disk: line index = b + g*N + r*N*N. Output ints span the
        bit-depth implied by the max value (10-bit -> /1023). The leading node
        row is uniform so we treat the grid as a uniform 0..1 domain.
        """
        node_row: Optional[List[int]] = None
        rows: List[Tuple[float, float, float]] = []
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                parts = s.split()
                # The node-position row: many ints, monotonically increasing,
                # appears before the RGB triplets (e.g. "0 64 128 ... 1023").
                if node_row is None and len(parts) > 3 and all(p.lstrip("-").isdigit() for p in parts):
                    node_row = [int(p) for p in parts]
                    continue
                if s.upper().startswith(("3DMESH", "MESH", "3DL")):
                    continue
                if len(parts) == 3 and all(p.lstrip("-").isdigit() for p in parts):
                    rows.append((int(parts[0]), int(parts[1]), int(parts[2])))
        if node_row is None:
            raise ValueError(f"{path}: missing 3dl node-position row")
        size = len(node_row)
        if len(rows) != size ** 3:
            raise ValueError(
                f"{path}: expected {size**3} entries, found {len(rows)}"
            )
        # Output normalization: infer bit-depth from the max output value.
        max_out = max((max(r) for r in rows), default=1)
        denom = 1023.0 if max_out > 255 else (255.0 if max_out > 1 else 1.0)
        norm_rows = [(r[0] / denom, r[1] / denom, r[2] / denom) for r in rows]
        grid = DiskRecipeParser._rows_to_bgr_grid(norm_rows, size, red_fastest=False)
        return grid, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)

    @staticmethod
    def _rows_to_bgr_grid(
        rows: Sequence[Tuple[float, float, float]], size: int, red_fastest: bool
    ) -> List[List[List[List[float]]]]:
        """Reorder a flat list of RGB triplets into ``grid[b][g][r] = [R,G,B]``.

        ``red_fastest`` (cube): line index = r + g*N + b*N*N.
        else (.3dl, blue-fastest): line index = b + g*N + r*N*N.
        """
        N = size
        grid = [
            [[[0.0, 0.0, 0.0] for _ in range(N)] for _ in range(N)]
            for _ in range(N)
        ]
        idx = 0
        if red_fastest:
            for b in range(N):
                for g in range(N):
                    for r in range(N):
                        grid[b][g][r] = list(rows[idx])
                        idx += 1
        else:
            for r in range(N):
                for g in range(N):
                    for b in range(N):
                        grid[b][g][r] = list(rows[idx])
                        idx += 1
        return grid

    # ---- Track-B degrade --------------------------------------------------
    def sample_degrade_spec(
        self, aspects: Sequence[str], seed: int, region_local: bool
    ) -> DegradeSpec:
        """Gaussian-sample raw CRS operators for a Track-B degradation.

        For each requested aspect (subset of L/GC/SC), pick a small random
        subset of that aspect's PARAM_KEYS and draw each from N(0, sigma_op)
        using the configured sigma profile (RAW LR units). Deterministic in
        ``seed`` so a sample re-renders identically.
        """
        rng = random.Random(seed)
        sigmas = SIGMA_PROFILES[self.sigma_profile]
        op_params: Dict[str, float] = {}
        used_aspects: List[str] = []
        for asp in aspects:
            keys = ASPECT_KEY_GROUPS.get(asp)
            if not keys:
                continue
            used_aspects.append(asp)
            # Perturb 1..3 operators per aspect (HSL has many; light has fewer).
            n = rng.randint(1, min(3, len(keys)))
            chosen = rng.sample(list(keys), n)
            for key in chosen:
                sigma = sigmas.get(key, 18.0)
                val = rng.gauss(0.0, sigma)
                # Keep within the slider's natural envelope.
                if key == "Exposure2012":
                    val = max(-4.0, min(4.0, val))
                else:
                    val = max(-100.0, min(100.0, val))
                op_params[key] = round(val, 3)
        return DegradeSpec(
            mode="gaussian_op",
            op_params=op_params,
            sigma_profile=self.sigma_profile,
            aspects=used_aspects,
            forward=False,
            seed=seed,
        )

    @staticmethod
    def degrade_spec_to_params(spec: DegradeSpec) -> Dict[str, Dict[str, float]]:
        """Convert a Track-B DegradeSpec into a VeraRetouch param dict
        (the perturbation operators, raw units, rest default 0) so the renderer
        can apply it. ``forward=False`` negates the operators (invert direction).
        """
        params = _zero_params()
        sign = 1.0 if spec.forward else -1.0
        for key, val in spec.op_params.items():
            if key in params:
                params[key] = {"value": round(sign * float(val), 3)}
        return params


# ===========================================================================
# LutApplier implementation.
# ===========================================================================

class DiskLutApplier(LutApplier):
    """Trilinear 3D-LUT application via torch ``grid_sample`` (5-D).

    Imports torch lazily. Operates at NATIVE resolution (no resize) per USER
    DECISION 1.
    """

    def apply_lut(
        self,
        img: Any,
        lut: Any,
        domain_min: Tuple[float, float, float] = (0, 0, 0),
        domain_max: Tuple[float, float, float] = (1, 1, 1),
    ) -> Any:
        """Apply a 3D-LUT to a batch of images.

        Parameters
        ----------
        img : torch.FloatTensor ``[B, 3, H, W]`` in [0, 1], RGB.
        lut : torch.FloatTensor ``[N, N, N, 3]`` indexed ``lut[b, g, r]`` in
            [0, 1] (as returned by ``DiskRecipeParser.load_cube``).
        domain_min, domain_max : per-channel input domain.

        Returns
        -------
        torch.FloatTensor ``[B, 3, H, W]`` in [0, 1], RGB.

        Method (probe_recipe_parsers §2): normalise each channel into its domain
        then to grid coords in [-1, 1]; build a sampling grid whose last axis is
        (x=R, y=G, z=B) to match the volume axes (W=R, H=G, D=B after permute);
        ``grid_sample`` with mode='bilinear' on a 5-D volume == trilinear,
        align_corners=True, padding_mode='border'.
        """
        import torch
        import torch.nn.functional as F

        if img.dim() != 4 or img.shape[1] != 3:
            raise ValueError(f"img must be [B,3,H,W], got {tuple(img.shape)}")
        B, _, H, W = img.shape
        device = img.device
        dtype = torch.float32

        lut = lut.to(device=device, dtype=dtype)
        N = lut.shape[0]
        if lut.shape != (N, N, N, 3):
            raise ValueError(f"lut must be [N,N,N,3], got {tuple(lut.shape)}")

        dmin = torch.tensor(domain_min, device=device, dtype=dtype).view(1, 3, 1, 1)
        dmax = torch.tensor(domain_max, device=device, dtype=dtype).view(1, 3, 1, 1)

        x = img.to(dtype).clamp(0.0, 1.0)
        # Normalise into domain -> [0,1] -> grid coords [-1,1].
        span = (dmax - dmin).clamp(min=1e-6)
        x = ((x - dmin) / span).clamp(0.0, 1.0)
        coords = x * 2.0 - 1.0  # [B,3,H,W] channels = (R,G,B) in [-1,1]

        r = coords[:, 0]  # [B,H,W]
        g = coords[:, 1]
        b = coords[:, 2]
        # grid last-dim order = (x, y, z) maps to volume (W, H, D).
        # We permute the LUT to volume [1,3,D=B,H=G,W=R], so x<-R, y<-G, z<-B.
        grid = torch.stack([r, g, b], dim=-1)          # [B,H,W,3]
        grid = grid.unsqueeze(1)                        # [B,1,H,W,3] (D=1)

        # lut[b,g,r,c] -> volume[c, d=b, h=g, w=r] -> [1,3,N,N,N]
        vol = lut.permute(3, 0, 1, 2).unsqueeze(0)      # [1,3,N(b),N(g),N(r)]
        vol = vol.expand(B, -1, -1, -1, -1).contiguous()

        out = F.grid_sample(
            vol, grid, mode="bilinear", align_corners=True, padding_mode="border"
        )  # [B,3,1,H,W]
        out = out.squeeze(2).clamp(0.0, 1.0)            # [B,3,H,W]
        return out


def identity_lut(size: int = 33) -> Any:
    """Build an identity 3D-LUT ``[N,N,N,3]`` indexed ``lut[b,g,r]`` for testing
    ``apply_lut`` (out == in). Imports torch lazily."""
    import torch

    n = size
    ramp = torch.linspace(0.0, 1.0, n)
    # lut[b,g,r] = (r, g, b)
    rr = ramp.view(1, 1, n).expand(n, n, n)
    gg = ramp.view(1, n, 1).expand(n, n, n)
    bb = ramp.view(n, 1, 1).expand(n, n, n)
    return torch.stack([rr, gg, bb], dim=-1).contiguous()


# ===========================================================================
# Raw decode + archive extraction helpers (grouped here per the module spec).
# ===========================================================================

def decode_raw_to_srgb(path: str, half_size: bool = False) -> Any:
    """Decode a camera-raw file (.dng/.cr2/.arw) to an 8-bit sRGB RGB array.

    Returns ``np.uint8`` HxWx3 RGB at NATIVE resolution (USER DECISION 1).
    Requires ``rawpy`` (conda env ``fivek-cleaning``; config.yaml raw_decode.env).
    Imports rawpy + numpy lazily so this module loads on envs that lack them.

    Parameters
    ----------
    half_size : if True, rawpy's half-resolution debayer (faster previews only;
        the build path uses full native resolution).
    """
    import numpy as np  # noqa: F401  (kept for callers / type clarity)
    import rawpy

    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=True,
            output_bps=8,
            half_size=half_size,
            gamma=(2.222, 4.5),  # standard sRGB-ish tone curve
        )
    return rgb  # HxWx3 uint8 RGB


def extract_archive(
    archive_path: str,
    dest_dir: str,
    members_glob: Optional[str] = None,
    seven_zip: str = "/usr/bin/7z",
) -> List[str]:
    """Extract a .rar/.zip/.7z archive to ``dest_dir`` using the ``7z`` CLI.

    Used for the awards corpus (double-nested .rar) and any zipped pack. Returns
    the list of extracted file paths actually present under ``dest_dir`` after
    extraction. Idempotent enough for resumable builds: re-extracts on demand
    (7z overwrites). Does NOT recurse a second level — callers re-invoke for
    the inner archive (config.yaml sources.awards.extract == "7z_two_step").

    Raises CalledProcessError if 7z fails.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [seven_zip, "x", "-y", f"-o{dest}", archive_path]
    if members_glob:
        cmd.append(members_glob)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return [str(p) for p in dest.rglob("*") if p.is_file()]


def is_technical_lut(name: str, technical_re: Optional[str] = None) -> bool:
    """True if a LUT filename looks like a technical Slog/REC709 conversion
    (config.yaml recipes.filters.technical_name_re), not a creative 'look'."""
    pat = technical_re or (
        r"Slog|S-Log|SLog2|SLog3|REC709|Rec\.709|709toLog|LogtoRec|LinearTo"
        r"|_to_|Conversion|Technical|LUTCalc|Identity|Neutral"
    )
    return re.search(pat, name, flags=re.IGNORECASE) is not None


# ===========================================================================
# Vendored Lua parser fallback (so this module does not hard-depend on the
# pe_kg path being importable at build time).
# ===========================================================================

def _strip_lua_table_prefix(text: str) -> str:
    """Reduce a Lightroom ``.lrtemplate`` body to the leading ``{ ... }`` table.

    LR writes either ``return { ... }`` or an assignment ``s = { ... }``
    (verified: 全店素材/H010.../Cali 5.lrtemplate starts ``s = {``). Strip any
    ``return`` keyword and/or ``<name> =`` assignment so the parser sees the
    table literal. Also drops a trailing ``;``/newlines after the closing brace.
    """
    s = text.strip()
    if s.startswith("return"):
        s = s[len("return"):].strip()
    # Strip a leading "<identifier> =" assignment (e.g. "s =").
    m = re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=\s*", s)
    if m:
        s = s[m.end():].strip()
    # Truncate to the matched outermost braces if there is trailing junk.
    start = s.find("{")
    if start > 0:
        s = s[start:]
    return s.rstrip().rstrip(";").rstrip()


def _parse_lua_return_table(text: str) -> Any:
    """Parse a Lua develop-settings table (``return {..}`` OR ``s = {..}``).
    Prefers the repo's pe_kg LuaTableParser; falls back to the vendored copy."""
    body = _strip_lua_table_prefix(text)
    try:
        import sys

        pe_kg_scripts = "/home/bc/retouching/pe_kg/scripts"
        if pe_kg_scripts not in sys.path:
            sys.path.insert(0, pe_kg_scripts)
        from _lua_table_parser import LuaTableParser  # type: ignore

        return LuaTableParser.from_lua(body)
    except Exception:
        return _VendoredLua.from_lua(body)


class _VendoredLua:
    """Minimal self-contained Lua-literal parser (mirror of pe_kg
    ``LuaTableParser``) used only if the repo copy is unavailable."""

    @staticmethod
    def _close_find(content: str, key_char: str, start: int = 0) -> int:
        result = -1
        offset = start
        pair_map = {"'": "'", '"': '"', "[": "]", "{": "}"}
        pair_temp: List[str] = []
        char = ""
        while offset < len(content):
            char_prev = char
            char = content[offset]
            if not char.isspace():
                if pair_temp:
                    last = pair_temp[-1]
                    if char == last:
                        pair_temp.pop()
                    elif char_prev != "\\" and char == "{" and last == "}":
                        pair_temp.append(pair_map[char])
                else:
                    if char_prev != "\\" and char in pair_map:
                        pair_temp.append(pair_map[char])
                    elif char == key_char:
                        result = offset
                        break
            offset += 1
        return result

    @staticmethod
    def from_lua(content: str, *, level: int = 0) -> Any:
        content = content.strip()
        if level == 0:
            for note in re.findall(r"--\s*?\[\[[\s\S]*?\]\]", content):
                content = content.replace(note, "")
            for note in re.findall(r"--.*", content):
                content = content.replace(note, "")
        if not content:
            raise ValueError("Cannot parse blank Lua content")
        if (content.count("-") == 0 or (content.count("-") == 1 and content[0] == "-")) and \
                content.replace("-", "").isnumeric():
            return int(content)
        if (content.count("-") == 0 or (content.count("-") == 1 and content[0] == "-")) and \
                content.count(".") <= 1 and content.replace("-", "").replace(".", "").isnumeric():
            return float(content)
        if content[:2] == "0x" and content.replace("0x", "").isalnum():
            return int(content, 16)
        if content == "false":
            return False
        if content == "true":
            return True
        if content == "nil":
            return None
        if content[:2] == "[[" and content[-2:] == "]]":
            return content[2:-2]
        if (content[0] == '"' and content[-1] == '"') or (content[0] == "'" and content[-1] == "'"):
            return content[1:-1]
        if content[0] == "{" and content[-1] == "}":
            inner = content[1:-1]
            level += 1
            items: List[Tuple[Any, Any]] = []
            count = 0
            offset = 0
            is_list = True
            while offset != -1:
                count += 1
                offset_prev = offset
                offset = _VendoredLua._close_find(inner, ",", offset + 1)
                if offset_prev != 0:
                    offset_prev += 1
                item = inner[offset_prev:].strip() if offset == -1 else inner[offset_prev:offset].strip()
                if not item:
                    continue
                divider = _VendoredLua._close_find(item, "=")
                if divider == -1:
                    key: Any = count
                    value_str = item
                else:
                    is_list = False
                    key_str = item[0:divider].strip()
                    if key_str and key_str[0] == "[" and key_str[-1] == "]":
                        key = _VendoredLua.from_lua(key_str[1:-1], level=level)
                    else:
                        key = key_str
                    value_str = item[divider + 1:].strip()
                items.append((key, _VendoredLua.from_lua(value_str, level=level)))
            if is_list:
                return [v for _, v in items]
            out: Dict[Any, Any] = {}
            for k, v in items:
                if isinstance(k, list):
                    k = tuple(k)
                out[k] = v
            return out
        return None

    @staticmethod
    def parse_return_table(text: str) -> Any:
        s = text.strip()
        if s.startswith("return"):
            s = s[len("return"):].strip()
        return _VendoredLua.from_lua(s)


# ===========================================================================
# Tiny CLI for spot-checking a single recipe file (no heavy model loads).
# ===========================================================================

def _main() -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="Parse one recipe file (xmp/lrtemplate/cube/3dl) and print result."
    )
    ap.add_argument("path", help="ABS path to an .xmp / .lrtemplate / .cube / .3dl")
    ap.add_argument("--sigma-profile", default="aether_tab8")
    args = ap.parse_args()

    parser = DiskRecipeParser(sigma_profile=args.sigma_profile)
    ext = Path(args.path).suffix.lower()
    if ext == ".xmp":
        params = parser.xmp_to_params(args.path)
        nz = {k: v for k, v in params.items() if abs(v["value"]) > 1e-9}
        print(json.dumps({"format": "xmp", "nonzero_params": nz,
                          "meta": parser.xmp_metadata(args.path)}, ensure_ascii=False, indent=2))
    elif ext == ".lrtemplate":
        params = parser.lrtemplate_to_params(args.path)
        nz = {k: v for k, v in params.items() if abs(v["value"]) > 1e-9}
        print(json.dumps({"format": "lrtemplate", "nonzero_params": nz,
                          "meta": parser.lrtemplate_metadata(args.path)}, ensure_ascii=False, indent=2))
    elif ext in (".cube", ".3dl"):
        lut, dmin, dmax = parser.load_cube(args.path)
        print(json.dumps({
            "format": ext.lstrip("."),
            "lut_size": int(lut.shape[0]),
            "domain_min": list(dmin),
            "domain_max": list(dmax),
            "is_technical": is_technical_lut(Path(args.path).name),
        }, ensure_ascii=False, indent=2))
    else:
        raise SystemExit(f"unsupported extension {ext!r}")


if __name__ == "__main__":
    _main()
