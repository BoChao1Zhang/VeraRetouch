"""Recover 500 distinct training portraits for author selection, not inference.

Single-reader, shard-ordered IO; bounded device memory; restartable outputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import time
import numpy as np
from PIL import Image

ROOT=Path('/home/bc/data/runs/paper_local_review500_20260920')
WEB=Path('/home/bc/VeraRetouch/outputs/local_review500_20260920')


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def save_image(path,a):
    Image.fromarray(np.rint(np.clip(a,0,1)*255).astype(np.uint8)).save(path)


def background_stats(rgb,mask):
    bg=rgb[mask==0]
    if len(bg)<rgb.shape[0]*rgb.shape[1]*.08:return None
    lum=bg.mean(-1);gray=rgb.mean(-1);gy,gx=np.gradient(gray)
    return dict(bg_dark=float((lum<.10).mean()),bg_white=float((lum>.88).mean()),
                bg_std=float(lum.std()),bg_texture=float((np.abs(gx)+np.abs(gy))[mask==0].mean()),
                fg_chroma=float((rgb.max(-1)-rgb.min(-1))[mask>0].mean()))


def recover(count):
    import torch
    import matplotlib
    from veraretouch_sprf.readout import mixed_data as MX,multistage_data as MD,mixed_codes as MC
    from tools.epr059_glutbasis.g2_capacity import GlutBasis
    from veraretouch_sprf.data.stage_targets import LutVolumes
    from tools.local_subject_crop import context_windows
    from tools.paper_appendix_qualitative import WORK as OLD
    torch.set_num_threads(3);device='cuda:0';torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.08,device)
    torch.backends.cuda.matmul.allow_tf32=False
    WEB.mkdir(parents=True,exist_ok=True)
    candidates=json.loads((ROOT/'local_candidates.json').read_text())
    candidates=[r for r in candidates if r['source_id'] not in {'ppr10k_8325_a','ppr10k_7581_a'}]
    seed=ROOT/'subject_paths.json'
    if not seed.exists():shutil.copyfile(OLD/'subject_paths.json',seed)
    print(json.dumps(dict(phase='load_subjects',candidates=len(candidates))),flush=True)
    payload,mapping,_=MX.MixedChainSource.collect_subjects([r['key'] for r in candidates],cache_path=seed)
    source=MX.MixedChainSource(subject_payloads=payload,subject_mapping=mapping);source.setup()
    candidates.sort(key=lambda r:(source.ds.index[r['key']]['dir'],hashlib.sha256(r['key'].encode()).hexdigest()))
    run=json.loads((MD.BK_RUN/'run_args.json').read_text())['config']
    bank=LutVolumes(run['data']['lut_bank_dir'],64);basis=GlutBasis(str(MD.GEOMETRY),device)
    results=json.loads((ROOT/'results.json').read_text()) if (ROOT/'results.json').exists() else []
    done={r['key'] for r in results};rejected=[];started=time.monotonic()
    for portrait_only in [True,False]:
        for row in candidates:
            if len(results)>=count:break
            if row['key'] in done:continue
            law=source.chain(row['key']);h,w=law['hw']
            if portrait_only!=(h>w):continue
            target=law['x0'].numpy().reshape(h,w,3);subject=law['beta'][0].numpy().reshape(h,w)
            stats=background_stats(target,subject)
            if stats is None or stats['bg_std']<.045 or (stats['bg_dark']>.65 and stats['bg_texture']<.015):
                rejected.append(dict(key=row['key'],reason='uniform/dark background or insufficient outside region',stats=stats));continue
            beta=law['beta'].to(device)
            with torch.no_grad():
                states=MD.chain_states(law['x0'],beta,law['luts'],bank,device)
                codes=[MC.solve_support_code(states[s+1],states[s],beta[s],basis.geometry) for s in range(6)]
                current=states[-1];replay=[current]
                for slot in range(5,-1,-1):
                    current=MC.apply_support(codes[slot],current,beta[slot],basis.geometry);replay.append(current)
                raw=torch.stack(replay).cpu().numpy().reshape(7,h,w,3)
                masks=beta.flip(0).cpu().numpy().reshape(6,h,w)
            clipped=np.clip(raw,0,1);window=context_windows(clipped,masks,6)
            if not window:
                rejected.append(dict(key=row['key'],reason='no sufficiently large subject crop with exact outside control'));continue
            local=np.abs(clipped[6]-clipped[5]).mean(-1)*100
            amplitude=float(local[masks[5]>0].mean())
            if amplitude<.5:
                rejected.append(dict(key=row['key'],reason='very small subject change',amplitude=amplitude));continue
            number=len(results)+1;name=f'{number:03d}';folder=ROOT/f'case_{name}';folder.mkdir(exist_ok=True)
            public=WEB/f'case_{name}';public.mkdir(exist_ok=True)
            np.savez_compressed(folder/'float_states.npz',recovery=raw,masks=masks)
            np.save(folder/'codes.npy',np.stack([c.detach().cpu().numpy() for c in codes]))
            for s in range(7):save_image(folder/f'recovery_{s}.png',clipped[s])
            save_image(folder/'reference.png',target)
            for src,dst in [('recovery_0','input'),('recovery_5','before'),('recovery_6','after'),('reference','gt')]:
                shutil.copyfile(folder/f'{src}.png',public/f'{dst}.png')
            for s in range(1,5):shutil.copyfile(folder/f'recovery_{s}.png',public/f'z{s}.png')
            for name2,s in [('crop_before',5),('crop_after',6)]:
                x0,y0,x1,y1=window['inside']['box'];save_image(public/f'{name2}.png',clipped[s,y0:y1,x0:x1])
            save_image(public/'support.png',masks[5])
            scale=max(1.,float(np.ceil(local.max())))
            save_image(public/'residual.png',matplotlib.colormaps['inferno'](local/scale)[...,:3])
            record=dict(**row,number=number,folder=str(folder),image_size=[w,h],portrait=h>w,
                        local_amplitude=amplitude,background=stats,crop=window,residual_max=scale,
                        execution='Target-conditioned adjacent-state code fitting and sequential replay; not model inference.')
            dump(folder/'provenance.json',record);results.append(record);done.add(row['key'])
            dump(ROOT/'results.json',results)
            if number%10==0:
                dump(ROOT/'progress.json',dict(completed=number,target=count,seconds=time.monotonic()-started,rejected=len(rejected)))
                print(json.dumps(dict(completed=number,target=count,seconds=round(time.monotonic()-started))),flush=True)
        if len(results)>=count:break
    dump(ROOT/'rejected.json',rejected)
    dump(ROOT/'progress.json',dict(completed=len(results),target=count,finished=True,seconds=time.monotonic()-started))
    if len(results)!=count:raise RuntimeError(f'Only {len(results)} eligible unique candidates; requested {count}')


def main():
    p=argparse.ArgumentParser();p.add_argument('--count',type=int,default=500);args=p.parse_args()
    recover(args.count)


if __name__=='__main__':main()
