"""Append recorded non-semantic mask families without renumbering the first 500."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
from PIL import Image
from tools.build_local_review500 import ROOT,WEB,dump,save_image
from tools.local_subject_crop import context_windows,detect_faces
from tools.update_gallery_face_crops import update as refresh_crop

GEOM=Path('/home/bc/data/runs/paper_geometry_portraits_20260920')


def render_one(item):
    i,r=item;src=Path(r['folder']);public=WEB/f'case_{i:03d}';public.mkdir(exist_ok=True)
    with np.load(src/'float_states.npz') as f:
        states=np.clip(f['recovery'],0,1);masks=f['masks']
    h,w=states.shape[1:3]
    reference=np.asarray(Image.open(src/'reference.png').convert('RGB'),dtype=np.float32)/255
    crop=context_windows(states,masks,6,subject_priority=False,allow_low_support=True,face_boxes=detect_faces(reference))
    if not crop:raise ValueError(f'No large edit window: {src}')
    for a,b in [('recovery_0','input'),('recovery_5','before'),('recovery_6','after'),('reference','gt')]:
        shutil.copyfile(src/f'{a}.png',public/f'{b}.png')
    for s in range(1,5):shutil.copyfile(src/f'recovery_{s}.png',public/f'z{s}.png')
    x0,y0,x1,y1=crop['inside']['box']
    for when,s in [('before',5),('after',6)]:save_image(public/f'crop_{when}.png',states[s,y0:y1,x0:x1])
    delta=np.abs(states[6]-states[5]).mean(-1)*100;maximum=max(1.,float(np.ceil(delta.max())))
    save_image(public/'support.png',masks[5]);save_image(public/'residual.png',matplotlib.colormaps['inferno'](delta/maximum)[...,:3])
    rgb=states[6];gray=rgb.mean(-1);gy,gx=np.gradient(gray)
    amplitude=float(delta[masks[5]>0].mean());a,b,c,d=crop['outside']['box']
    return dict(r,number=i,image_size=[w,h],portrait=h>w,crop=crop,residual_max=maximum,face_crop_version=3,
                local_amplitude=amplitude,background=dict(fg_chroma=float((rgb.max(-1)-rgb.min(-1))[masks[5]>0].mean()),
                bg_texture=float((np.abs(gx)+np.abs(gy))[b:d,a:c].mean())))


def prepare_task(task):
    kind,payload=task
    return render_one(payload) if kind=='new' else refresh_crop(payload)


def main():
    base=json.loads((ROOT/'results.json').read_text())
    groups=json.loads((GEOM/'all_candidates.json').read_text())
    classification={r['key']:r['geometry_kind'] for rows in groups.values() for r in rows}
    old=WEB/'combined_results.json'
    previous={r['key']:r for r in json.loads(old.read_text())} if old.exists() else {}
    merged=[dict(previous.get(r['key'],r),geometry_kind=classification[r['key']]) for r in base]
    geometries=json.loads((GEOM/'local_results.json').read_text())
    cached={k:r for k,r in previous.items() if r['number']>500}
    pending=[]
    for i,r in enumerate(geometries,501):
        src=Path(r['folder']);public=WEB/f'case_{i:03d}';public.mkdir(exist_ok=True)
        if r['key'] in cached and cached[r['key']]['number']==i:
            assert all((public/f'{n}.png').is_file() for n in ['input','before','after','gt','support','residual','crop_before','crop_after','z1','z2','z3','z4'])
            merged.append(cached[r['key']]);continue
        pending.append((i,r))
    tasks=[('refresh',r) for r in merged]+[('new',item) for item in pending]
    merged=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for done,r in enumerate(pool.map(prepare_task,tasks),1):
            merged.append(r)
            if done%100==0:print(json.dumps(dict(gallery_cases=done,total=len(tasks))),flush=True)
    merged.sort(key=lambda r:r['number'])
    dump(WEB/'combined_results.json',merged)
    print(json.dumps(dict(total=len(merged),families=dict(Counter(r['geometry_kind'] for r in merged)),
                          new_nonzero_controls=sum(r['crop'].get('outside_kind')=='low_support' for r in merged[500:]))),flush=True)


if __name__=='__main__':main()
