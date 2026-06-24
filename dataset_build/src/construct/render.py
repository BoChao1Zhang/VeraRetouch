"""Unified source-conditioned render: apply ONE preset to ONE source photo.

  param (xmp/lrtemplate) -> real Lightroom render via the LrC farm (render_via_lr)
  lut   (.cube/.3dl)     -> code 3D-LUT trilinear (LR has no .cube develop-preset form;
                            this is how the 6-probe previews were made — preset_qa.py:14)

Returns {ok, after_path, engine} or {ok:False, error_code}. ponytail: thin dispatch over
the two proven paths (lr_render.render_via_lr + pilot_preset._apply_cube/_parser).
"""
from __future__ import annotations

import os
import uuid

from PIL import Image

from dataset_build.source_qa import config, lr_render
from dataset_build.source_qa import preset_qa as PQ  # _parser().load_cube + _apply_cube


def render_preset(preset_path: str, kind: str, fmt: str, source_path: str,
                  lut_longedge: int = 1024) -> dict:
    if kind == "param":
        return lr_render.render_via_lr(preset_path, fmt, source_path)
    # lut
    try:
        cube = PQ._parser().load_cube(preset_path)
        im = Image.open(source_path).convert("RGB")
        im.thumbnail((lut_longedge, lut_longedge))
        after = PQ._apply_cube(im, cube)
        os.makedirs(config.RENDER_STAGE, exist_ok=True)
        out = os.path.join(config.RENDER_STAGE, f"lut_{uuid.uuid4().hex[:12]}.jpg")
        after.save(out, "JPEG", quality=95)
        return {"ok": True, "after_path": out, "engine": "lut_trilinear"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error_code": "lut_apply_failed", "error": str(e)[:200]}


def _smoke() -> None:
    import json
    from dataset_build.source_qa import db
    feats = {}
    for l in open("/home/bc/data/datasets/vera_directionA_1M/preset_bank_full/features.jsonl"):
        f = json.loads(l); feats.setdefault(f["kind"], f)
        if len(feats) == 2:
            break
    conn = db.connect()
    src = conn.execute("SELECT path FROM assets WHERE asset_type='image' AND b_quality=3 "
                       "AND dup_of IS NULL ORDER BY asset_id LIMIT 1").fetchall()[0]["path"]
    conn.close()
    for kind, f in feats.items():
        r = render_preset(f["path"], kind, f.get("fmt"), src)
        assert r.get("ok") and os.path.exists(r["after_path"]), f"{kind} render failed: {r}"
        print(f"  {kind}: ok -> {r['after_path']}")
    print("render._smoke OK")


if __name__ == "__main__":
    _smoke()
