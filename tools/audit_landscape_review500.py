"""Verify every landscape stage, support, residual and matched crop."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import matplotlib
import numpy as np
from PIL import Image
from tools.build_landscape_review_gallery import WEB


def verify(item):
    r,deep=item
    folder=WEB/r['id'];src=Path(r['folder'])
    with np.load(src/'float_states.npz') as f:
        masks=f['masks'];states=np.clip(f['recovery'],0,1) if deep else None
    assert np.isfinite(masks).all()
    if deep:assert np.isfinite(states).all()
    pixels={s:np.asarray(Image.open(folder/f'z{s}.png')) for s in range(1,6)}
    for s in range(2,6):
        info=r['steps'][str(s)];box=info['box']
        quantized=np.abs(pixels[s].astype(np.float32)-pixels[s-1].astype(np.float32)).mean(-1)/255*100
        assert abs(float(quantized.mean())-info['mean_change'])<=100/255+1e-5
        for when,slot in [('before',s-1),('after',s)]:
            x0,y0,x1,y1=box;expected=pixels[slot][y0:y1,x0:x1]
            np.testing.assert_array_equal(expected,np.asarray(Image.open(folder/f'{s}_{when}_crop.png')))
        np.testing.assert_array_equal(np.rint(masks[s-1]*255).astype(np.uint8),np.asarray(Image.open(folder/f'{s}_support.png')))
        if deep:
            delta=np.abs(states[s]-states[s-1]).mean(-1)*100
            assert abs(float(delta.mean())-info['mean_change'])<1e-5
            assert float(delta.max())<=r['residual_max']+1e-5
            expected=np.rint(matplotlib.colormaps['inferno'](delta/r['residual_max'])[...,:3]*255).astype(np.uint8)
            np.testing.assert_array_equal(expected,np.asarray(Image.open(folder/f'{s}_residual.png')))
        else:
            with Image.open(folder/f'{s}_residual.png') as im:im.verify()
    return hashlib.sha256((folder/'reference.png').read_bytes()).hexdigest()


def main():
    records=json.loads((WEB/'index.json').read_text())
    assert len(records)==500 and len({r['source_id'] for r in records})==500
    assert [r['id'] for r in records]==[f'L{i:03d}' for i in range(1,501)]
    strata={}
    for r in records:strata.setdefault((r['scene_bucket'],r['style_bucket']),r['id'])
    deep_ids=set(strata.values())
    hashes=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,digest in enumerate(pool.map(verify,[(r,r['id'] in deep_ids) for r in records]),1):
            hashes.append(digest)
            if i%100==0:print(json.dumps(dict(verified=i,total=500)),flush=True)
    report=dict(count=500,unique_source_ids=500,unique_reference_files=len(set(hashes)),stages_checked=2000,
                matched_crop_pairs=2000,scene_groups=dict(Counter(r['scene_bucket'] for r in records)),
                style_groups=dict(Counter(r['style_bucket'] for r in records)),
                exact_float_residual_samples=len(deep_ids),exact_float_residual_stages=4*len(deep_ids),deep_sample_ids=sorted(deep_ids),
                note='All 500 supports and 2,000 crop pairs checked; exact float residual recomputation stratified by every scene/style combination. Images are target-conditioned recoveries, not model predictions.')
    (WEB/'integrity_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
