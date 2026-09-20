"""Refresh only gallery ROI crops; preserve IDs, full images and controls."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import json
from pathlib import Path
import numpy as np
from PIL import Image
from tools.build_local_review500 import WEB
from tools.local_subject_crop import context_windows,detect_faces


def update(r):
    if r.get('face_crop_version')==3:return r
    public=WEB/f'case_{r["number"]:03d}'
    with np.load(Path(r['folder'])/'float_states.npz') as data:mask=data['masks'][-1]
    before=np.asarray(Image.open(public/'before.png').convert('RGB'),dtype=np.float32)/255
    after=np.asarray(Image.open(public/'after.png').convert('RGB'),dtype=np.float32)/255
    reference=np.asarray(Image.open(public/'gt.png').convert('RGB'),dtype=np.float32)/255
    faces=detect_faces(reference)
    crop=context_windows(np.stack([before,after]),mask[None],1,
                         subject_priority=r['geometry_kind']=='semantic',allow_low_support=True,face_boxes=faces)
    if not crop:raise ValueError(f'No valid context crop: {r["number"]}')
    # Control ROI remains the already-audited original. Only the edit ROI moves.
    box=crop['inside']['box']
    for when in ['before','after']:
        with Image.open(public/f'{when}.png') as im:im.crop(box).save(public/f'crop_{when}.png')
    result=dict(r,crop=dict(r['crop'],inside=crop['inside']),face_crop_version=3)
    return result


def main():
    path=WEB/'combined_results.json';records=json.loads(path.read_text());updated=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,r in enumerate(pool.map(update,records),1):
            updated.append(r)
            if i%100==0:print(json.dumps(dict(crops=i,total=len(records))),flush=True)
    path.write_text(json.dumps(updated,ensure_ascii=False,indent=2)+'\n')
    report=dict(count=len(updated),focus=dict(Counter(r['crop']['inside'].get('focus') for r in updated)),
                preserved_ids=[r['number'] for r in updated]==[r['number'] for r in records],
                note='Face detection guides cropping only; no identification. Full frames and control ROIs are unchanged.')
    (WEB/'face_crop_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
