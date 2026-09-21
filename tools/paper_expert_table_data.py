"""Add L1/L2 on the exact legacy evaluated image sets; preserve original scores."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sqlite3

import numpy as np

BASE=Path('/home/bc/data/runs/pubbench')
SPECS={
    'MonetGPT':(Path('/home/bc/nfsvfs/bc/data/runs/pubbench/monetgpt_fixed'),'instr_real'),
    'JarvisArt':(BASE/'jarvisart','instr_real'),
    'JarvisEvo':(BASE/'jarvisevo','instr_real'),
    'VeraRetouch':(BASE/'veraretouch','real')}
OUT=Path('/home/bc/nfsvfs/bc/data/runs/paper_expert_table_update_20260921')


def main():
    from q3vl.whatb.pubbench.epr035d_metrics import load_rgb_u8
    from veraretouch_sprf.e2e.bench import samples_for
    result={}
    for bench in ('fivek','ppr10k'):
        samples={r['sample_id']:r for r in samples_for(bench)[0]}
        target={sid:load_rgb_u8(r['gt_path']) for sid,r in samples.items()}
        rows=[]
        for name,(root,lane) in SPECS.items():
            prior=json.loads((root/f'metrics_{bench}.json').read_text())['headline'][lane]
            scored={r['sample_id']:r for r in (json.loads(s) for s in (root/f'rows_{bench}.jsonl').read_text().splitlines())
                    if r.get('lane')==lane and isinstance(r.get('psnr'),(int,float)) and np.isfinite(r['psnr'])}
            if len(scored)!=prior['n']:raise ValueError(f'{name} {bench}: legacy count mismatch')
            def measure(sid):
                path=root/bench/lane/(sid+'.png')
                pred=load_rgb_u8(path);gt=target[sid]
                if pred.shape!=gt.shape:raise ValueError(f'{name} {sid}: geometry mismatch')
                diff=(pred.astype(np.float64)-gt.astype(np.float64))/255
                l1=float(np.abs(diff).mean());l2=float(np.square(diff).mean())
                psnr=float(-10*np.log10(l2))
                if abs(psnr-scored[sid]['psnr'])>1e-6:raise ValueError(f'{name} {sid}: archived PSNR mismatch')
                return dict(sample_id=sid,l1_x100=l1*100,l2_x1000=l2*1000,psnr=psnr,
                            prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            with ThreadPoolExecutor(max_workers=4) as pool:
                measured=list(pool.map(measure,sorted(scored)))
            rows.append(dict(method=name,n=len(measured),L1=float(np.mean([r['l1_x100'] for r in measured])),
                             L2=float(np.mean([r['l2_x1000'] for r in measured])),
                             PSNR=prior['psnr'],SSIM=prior['ssim'],DE00=prior['de00'],
                             source=str(root),lane=lane,rows=measured))
            print(bench,name,len(measured),flush=True)
        summary=json.loads((Path('/home/bc/nfsvfs/bc/data/runs/epr072_sixstage_public_20260920/full')/bench/'summary.json').read_text())
        assert summary['complete'] and summary['n']==len(samples)
        m=summary['means']
        rows.append(dict(method='Ours (Stagewise)',n=summary['n'],L1=m['l1']*100,L2=m['l2']*1000,
                         PSNR=m['psnr'],SSIM=m['ssim'],DE00=m['de00'],checkpoint_sha256=summary['checkpoint_sha256'],
                         execution='six-stage',stage_text='none'))
        result[bench]=rows
        OUT.mkdir(parents=True,exist_ok=True)
        tmp=OUT/'table_data.partial';tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(OUT/'table_data.json')
    print(json.dumps({k:[{a:b for a,b in r.items() if a!='rows'} for r in v] for k,v in result.items()},indent=2))


if __name__=='__main__':main()
