"""Validate final candidate identities, crop semantics and saved endpoints."""
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from tools.build_local_review500 import ROOT,WEB


def verify(r):
    src=Path(r['folder']);public=WEB/f'case_{r["number"]:03d}'
    digest=hashlib.sha256((src/'reference.png').read_bytes()).hexdigest()
    with np.load(src/'float_states.npz') as data:
        masks=data['masks'];raw=data['recovery']
    x0,y0,x1,y1=r['crop']['inside']['box']
    support=masks[5,y0:y1,x0:x1]
    assert (support>0).mean()>=.8
    assert r['crop']['inside']['area_fraction']>=.07
    for when,slot in [('before',5),('after',6)]:
        expected=np.asarray(Image.open(src/f'recovery_{slot}.png').crop((x0,y0,x1,y1)))
        np.testing.assert_array_equal(expected,np.asarray(Image.open(public/f'crop_{when}.png')))
    x0,y0,x1,y1=r['crop']['outside']['box']
    assert (masks[5,y0:y1,x0:x1]==0).all()
    np.testing.assert_array_equal(raw[5,y0:y1,x0:x1],raw[6,y0:y1,x0:x1])
    for name in ['before','after','input','gt','support','residual','crop_before','crop_after','z1','z2','z3','z4']:
        with Image.open(public/f'{name}.png') as img:img.verify()
    return digest,r['crop']['inside']['area_fraction']


def main():
    records=json.loads((ROOT/'results.json').read_text())
    assert len(records)==500
    assert len({r['source_id'] for r in records})==500
    assert [r['number'] for r in records]==list(range(1,501))
    assert not {'ppr10k_8325_a','ppr10k_7581_a'}&{r['source_id'] for r in records}
    extrema=[];reference_hashes=set()
    # Only local saved artifacts, never concurrent reads from training shards.
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,(digest,area) in enumerate(pool.map(verify,records),1):
            reference_hashes.add(digest);extrema.append(area)
            if i%50==0:print(json.dumps(dict(verified=i,total=500)),flush=True)
    assert len(reference_hashes)==500
    stats=dict(count=len(records),unique_sources=500,unique_reference_files=len(reference_hashes),portrait=sum(r['portrait'] for r in records),
               pools=dict(Counter(r['pool'] for r in records)),large_crop_area_min=min(extrema),large_crop_area_max=max(extrema),
               exact_outside_controls=500,source_manifest_sha256=hashlib.sha256((ROOT/'results.json').read_bytes()).hexdigest(),
               execution='target-conditioned closed-form recovery; not model prediction',
               selection='author-facing shortlist, not a representative evaluation sample')
    (ROOT/'audit.json').write_text(json.dumps(stats,indent=2)+'\n')
    (WEB/'audit.json').write_text(json.dumps(stats,indent=2)+'\n')
    print(json.dumps(stats),flush=True)


if __name__=='__main__':main()
