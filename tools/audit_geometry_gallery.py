"""Validate expanded geometry gallery and face-oriented crops, without fitting."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from tools.build_local_review500 import WEB,ROOT
from tools.local_subject_crop import FACE_MODEL_SHA256


def verify(r):
    public=WEB/f'case_{r["number"]:03d}';src=Path(r['folder'])
    with np.load(src/'float_states.npz') as f:mask=f['masks'][-1]
    x0,y0,x1,y1=r['crop']['inside']['box']
    assert (mask[y0:y1,x0:x1]>0).mean()>=.5
    if r['geometry_kind']=='semantic':assert (mask[y0:y1,x0:x1]>0).mean()>=.8
    for when in ['before','after']:
        im=Image.open(public/f'{when}.png').convert('RGB')
        np.testing.assert_array_equal(np.asarray(im.crop((x0,y0,x1,y1))),np.asarray(Image.open(public/f'crop_{when}.png')))
    a,b,c,d=r['crop']['outside']['box']
    outside=mask[b:d,a:c];kind=r['crop'].get('outside_kind','zero')
    if kind=='zero':
        assert (outside==0).all()
        np.testing.assert_array_equal(np.asarray(Image.open(public/'before.png'))[b:d,a:c],
                                      np.asarray(Image.open(public/'after.png'))[b:d,a:c])
    else:assert kind=='low_support' and outside.mean()>0
    for name in ['input','gt','support','residual','z1','z2','z3','z4']:
        with Image.open(public/f'{name}.png') as im:im.verify()
    return r['geometry_kind'],r['crop']['inside'].get('focus'),kind


def main():
    path=WEB/'combined_results.json';records=json.loads(path.read_text())
    assert len(records)==800
    assert [r['number'] for r in records]==list(range(1,801))
    old=json.loads((ROOT/'results.json').read_text())
    assert [r['key'] for r in records[:500]]==[r['key'] for r in old]
    counts=Counter(r['geometry_kind'] for r in records)
    assert counts==dict(semantic=500,radial=100,band=100,linear=100)
    assert all(r.get('face_crop_version')==3 for r in records)
    for kind in ['radial','band','linear']:
        assert len({r['source_id'] for r in records if r['geometry_kind']==kind})==100
    checked=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,r in enumerate(pool.map(verify,records),1):
            checked.append(r)
            if i%100==0:print(json.dumps(dict(verified=i,total=800)),flush=True)
    report=dict(candidates=len(records),families=dict(counts),preserved_first_500=True,
                portrait=sum(r['portrait'] for r in records),focus=dict(Counter(r[1] for r in checked)),
                control_types=dict(Counter(r[2] for r in checked)),
                manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                crop_locator='OpenCV Zoo YuNet 2023mar, face-centered crop version 3',face_model_sha256=FACE_MODEL_SHA256,
                execution='Target-conditioned closed-form recovery; not model prediction.')
    (WEB/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
