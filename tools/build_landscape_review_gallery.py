"""Landscape gallery assets: true hue/shadow/midtone/highlight stage changes."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
import matplotlib
from tools.local_subject_crop import context_windows
from tools.preview_three_part_local import window_sum

ROOT=Path('/home/bc/data/runs/paper_landscape_review500_20260920')
WEB=Path('/home/bc/VeraRetouch/outputs/landscape_review500_20260920')


def recommended(number,steps):
    # Rotate the stage across sampling rounds so a scene is not permanently
    # coupled to one of the four stages (12 scenes is divisible by four).
    focus=2+((number-1+(number-1)//12)%4)
    values={int(k):v for k,v in steps.items()}
    return max(values,key=lambda s:values[s]['active_change']) if values[focus]['active_change']<.15 else focus


def render(item):
    number,r=item;name=f'L{number:03d}';out=WEB/name;out.mkdir(exist_ok=True)
    cached=out/'meta.json'
    if cached.exists():
        old=json.loads(cached.read_text());assert old['key']==r['key']
        old['focus']=recommended(number,old['steps']);return old
    src=Path(r['folder'])
    with np.load(src/'float_states.npz') as d:states=np.clip(d['recovery'],0,1);masks=d['masks']
    h,w=states.shape[1:3]
    for s in range(7):shutil.copyfile(src/f'recovery_{s}.png',out/f'z{s}.png')
    shutil.copyfile(src/'reference.png',out/'reference.png')
    differences={s:np.abs(states[s]-states[s-1]).mean(-1)*100 for s in range(2,6)}
    vmax=max(1.,float(np.ceil(max(a.max() for a in differences.values()))));steps={}
    for s in range(2,6):
        delta=differences[s];mask=masks[s-1];active=mask>0
        crop=context_windows(states,masks,s,subject_priority=False,allow_low_support=True,face_boxes=[])
        if crop:box=crop['inside']['box']
        else:
            cw,ch=round(w*.42),round(h*.42);score=window_sum(delta,cw,ch)
            y,x=np.unravel_index(score.argmax(),score.shape);box=[int(x),int(y),int(x+cw),int(y+ch)]
        for when,slot in [('before',s-1),('after',s)]:
            with Image.open(src/f'recovery_{slot}.png') as im:im.crop(box).save(out/f'{s}_{when}_crop.png')
        Image.fromarray(np.rint(mask*255).astype(np.uint8)).save(out/f'{s}_support.png',compress_level=1)
        Image.fromarray(np.rint(matplotlib.colormaps['inferno'](delta/vmax)[...,:3]*255).astype(np.uint8)).save(out/f'{s}_residual.png',compress_level=1)
        steps[s]=dict(stage=s,box=box,mean_change=float(delta.mean()),active_fraction=float(active.mean()),
                      active_change=float(delta[active].mean()) if active.any() else 0,
                      active=bool(delta.max()>.05))
    focus=recommended(number,steps)
    result={k:r[k] for k in ['source_id','key','instruction','annotation','pool','scene_bucket','style_bucket','scene_terms','intent_terms']}
    result.update(id=name,number=number,folder=str(src),size=[w,h],focus=focus,steps=steps,residual_max=vmax)
    cached.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');return result


def main():
    WEB.mkdir(parents=True,exist_ok=True)
    rows=json.loads((ROOT/'local_results.json').read_text())
    with ThreadPoolExecutor(max_workers=4) as pool:
        records=[]
        for i,r in enumerate(pool.map(render,enumerate(rows,1)),1):
            records.append(r)
            if i%50==0:print(json.dumps(dict(exported=i,total=len(rows))),flush=True)
    payload=json.dumps(records,ensure_ascii=False).replace('<','\\u003c')
    template=Path(__file__).with_name('landscape_review_gallery.html').read_text()
    (WEB/'index.html').write_text(template.replace('__RECORDS__',payload))
    (WEB/'index.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    audit=dict(count=len(records),unique_sources=len({r['source_id'] for r in records}),
               scene_groups=dict(Counter(r['scene_bucket'] for r in records)),style_groups=dict(Counter(r['style_bucket'] for r in records)),
               focus_steps=dict(Counter(r['focus'] for r in records)),execution='Target-conditioned closed-form recovery, not model inference.')
    (WEB/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit),flush=True)


if __name__=='__main__':main()
