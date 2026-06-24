"""Minimal checks for mask-template geometry normalization. Run: python -m pytest -q
or just `python dataset_build/tools/test_mine_mask_templates.py`."""
import math

from dataset_build.tools.mine_mask_templates import _norm_geom, _bucket_id


def test_radial_center_radius():
    rec = {"mask_type": "circulargradient", "is_ai": False, "what": "Mask/CircularGradient",
           "geom": {"Top": "0.3", "Left": "0.3", "Bottom": "0.7", "Right": "0.7",
                    "Angle": "0", "Feather": "50", "Flipped": "true"}}
    ng = _norm_geom(rec)
    assert ng["shape"] == "radial"
    assert all(abs(a - b) < 1e-9 for a, b in zip(ng["center"], [0.5, 0.5]))  # ((L+R)/2,(T+B)/2)
    assert all(abs(a - b) < 1e-9 for a, b in zip(ng["radius"], [0.2, 0.2]))  # ((R-L)/2,(B-T)/2)
    assert abs(ng["feather"] - 0.5) < 1e-9 and ng["flipped"] is True
    bid, concept = _bucket_id(ng, rec)
    assert bid == "circ_mid_ctr" and concept == "subject"


def test_degenerate_mask_dropped():
    rec = {"mask_type": "circulargradient", "is_ai": False, "what": "",
           "geom": {"Top": "0", "Left": "0", "Bottom": "0", "Right": "0"}}
    assert _norm_geom(rec) is None


def test_linear_angle():
    rec = {"mask_type": "gradient", "is_ai": False, "what": "Mask/Gradient",
           "geom": {"ZeroX": "0.5", "ZeroY": "0.0", "FullX": "0.5", "FullY": "1.0"}}
    ng = _norm_geom(rec)
    assert ng["shape"] == "linear"
    assert abs(ng["angle"] - math.pi / 2) < 1e-9   # straight down -> vertical
    bid, _ = _bucket_id(ng, rec)
    assert bid == "grad_v"


if __name__ == "__main__":
    test_radial_center_radius()
    test_degenerate_mask_dropped()
    test_linear_angle()
    print("ok")
