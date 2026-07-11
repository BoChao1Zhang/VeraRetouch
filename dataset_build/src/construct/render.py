"""Unified source-conditioned render helpers.

  param (xmp/lrtemplate) -> core.render_backend 双路分流：有残差 LUT 的标定 preset
                            走本地 GPU（gpu_render, batch=16, cuda:1），其余走
                            LrC 农场（render_via_lr）；本地失败自动回退农场
  lut   (.cube/.3dl)     -> code 3D-LUT trilinear (LR has no .cube develop-preset form;
                            this is how the 6-probe previews were made — preset_qa.py:14)
  local preset           -> render the complete base preset once, then composite one or more
                            per-image CGT alpha variants back over the untouched source

Returns {ok, after_path, engine} or {ok:False, error_code}. ponytail: thin dispatch over
the two proven paths (core.render_backend + pilot_preset._apply_cube/_parser).
"""
from __future__ import annotations

import hashlib
import os
import shutil
import uuid

from PIL import Image

from dataset_build.core import render_backend
from dataset_build.source_qa import config
from dataset_build.source_qa import preset_qa as PQ  # _parser().load_cube + _apply_cube

RENDERS_ROOT = os.path.join(config.OUT_ROOT, "renders")   # content-addressed sharded render store


def shard_save(tmp_path: str) -> str:
    """Move a freshly-rendered jpg into content-addressed sharded storage and return the new path.

    Filename IS the sha256, sharded renders/<ab>/<cd>/<sha>.jpg — avoids a 400k-file flat dir and
    gives natural dedup (identical render bytes resolve to the same path, the duplicate is dropped).
    """
    h = hashlib.sha256()
    with open(tmp_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    sha = h.hexdigest()
    d = os.path.join(RENDERS_ROOT, sha[:2], sha[2:4])
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, sha + ".jpg")
    if os.path.exists(dst):
        os.remove(tmp_path)          # dedup: identical render already stored
    else:
        shutil.move(tmp_path, dst)
    return dst


def render_preset(preset_path: str, kind: str, fmt: str, source_path: str,
                  lut_longedge: int = 1024) -> dict:
    if kind == "param":
        # 双路后端：残差 LUT 已标定的 preset 本地 GPU 渲，其余 LR 农场（保真第一）。
        r = render_backend.render_one(preset_path, fmt, source_path)
        if r.get("ok") and r.get("after_path"):
            r["after_path"] = shard_save(r["after_path"])
        return r
    # lut
    try:
        cube = PQ._parser().load_cube(preset_path)
        im = Image.open(source_path).convert("RGB")
        im.thumbnail((lut_longedge, lut_longedge))
        after = PQ._apply_cube(im, cube)
        os.makedirs(config.RENDER_STAGE, exist_ok=True)
        out = os.path.join(config.RENDER_STAGE, f"lut_{uuid.uuid4().hex[:12]}.jpg")
        after.save(out, "JPEG", quality=95)
        return {"ok": True, "after_path": shard_save(out), "engine": "lut_trilinear"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error_code": "lut_apply_failed", "error": str(e)[:200]}


def render_local_preset_variants(base_preset_path: str, fmt: str, source_path: str,
                                 specs: list, preset_id: str | None = None,
                                 store: bool = True) -> dict:
    """Render a complete base preset once and localize it with one CGT spec per output.

    ``specs`` are flat ``{mask_type, geom, amount?}`` (geometric) or
    ``{mask_type: "semantic", alpha, amount?}`` dictionaries; either form may carry an
    optional ``cgt_path`` — it is popped here and forwarded so the backend writes the
    actually-composited alpha as a single-channel PNG (result rows carry ``cgt_path``
    back; missing/None means no CGT PNG, backward compatible). The backend applies any
    preset residual to the edited branch before exact alpha compositing, so alpha-zero
    pixels remain the source image rather than receiving a global residual correction.
    Successful temporary outputs are moved into the content-addressed render store
    (``store=False`` keeps them in RENDER_STAGE — for ephemeral preview passes);
    CGT PNGs stay at the caller-given ``cgt_path`` (not content-addressed).
    """
    os.makedirs(config.RENDER_STAGE, exist_ok=True)
    variants = []
    for spec in specs:
        spec = dict(spec)                       # 不改调用方对象
        cgt_path = spec.pop("cgt_path", None)   # CGT PNG 由后端产出（性能项1+3）
        variants.append(
            {"spec": spec, "cgt_path": cgt_path,
             "out_path": os.path.join(
                 config.RENDER_STAGE, f"localpreset_{uuid.uuid4().hex[:12]}.jpg")})
    res = render_backend.get_backend().render_local_variants(
        base_preset_path, fmt, source_path, variants, preset_id=preset_id)
    for row in res.get("results") or []:
        tmp = row.get("after_path") or row.get("out_path")
        if row.get("ok") and tmp and os.path.exists(tmp):
            if store:
                saved = shard_save(tmp)
                row["after_path"] = saved
                row["out_path"] = saved
        elif tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    return res


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
