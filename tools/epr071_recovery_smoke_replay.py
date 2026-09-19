"""Replay R@800 with the original micro=2 evaluation batch size."""
import argparse
import csv
import gc
import json
from pathlib import Path
import shutil
import sqlite3
import time

import numpy as np
import torch

from tools.epr071_recovery_smoke100 import write_json, emit
from tools.epr071_val50_diag import (PROTOSET,JOURNAL,HALVES,load_checkpoint,
                                    geometry_of,stats,sha256_file)
from veraretouch_sprf.readout import artedit_eval as AE,epr071_data as E71,select_train as ST


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True)
    args=ap.parse_args(); out=Path(args.out); device='cuda:0'
    torch.set_num_threads(4); torch.manual_seed(20260919)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.45)
    records=json.loads((out/'manifest.json').read_text())['records']
    rows=[json.loads(l) for l in (out/'rows.jsonl').read_text().splitlines()]
    summary=json.loads((out/'summary.json').read_text())
    bank=ST.Bank(device,path=PROTOSET); mean,std,_=E71.load_scaler()
    model,facts=load_checkpoint(out/'R_best800_snapshot.pt',bank,device)
    model.model.eval(); model.head.eval()
    evaluator=ST.SelectEval(model,bank,device,micro=2,num_workers=0,render_chunk=6,
                            halves_path=HALVES,arm='R',code_mean=mean,code_std=std,
                            journal=str(JOURNAL))
    _,normalized=evaluator.predict(records)
    raw=normalized*std+mean
    renderer=evaluator.renderer
    for i,(rec,row) in enumerate(zip(records,rows)):
        assert rec['sample_id']==row['sample_id']
        src,gt=AE.load_rgb_u8(rec['input_path']),AE.load_rgb_u8(rec['gt_path'])
        z=torch.as_tensor(src.astype(np.float32)/255,device=device).reshape(-1,3)
        target=torch.as_tensor(gt.astype(np.float32)/255,device=device).reshape(-1,3)
        pred=renderer.apply(raw[i].reshape(3,-1),z).clamp(0,1)
        pred=(pred*255+.5).to(torch.uint8).double()/255
        metrics=geometry_of(z.double(),target.double(),pred)
        for key,val in metrics.items(): row['model_'+key]=val
        row['model_l1']*=100
        row['model_amplitude_ratio']=metrics['m_pred']/max(float((target-z).abs().mean()),1e-12)
    smoke=float(np.mean([r['model_l1'] for r in rows[:50]]))
    difference=abs(smoke-summary['smoke_reference'])
    emit('replay_smoke',micro=2,reference=summary['smoke_reference'],here=smoke,difference=difference)
    write_json(out/'replay_smoke.json',dict(micro=2,reference=summary['smoke_reference'],
               here=smoke,abs_diff=difference,script_sha256=sha256_file(Path(__file__))))
    if difference>1e-4: raise RuntimeError('micro=2 smoke still differs from training log')
    for filename in ('rows.jsonl','rows.csv','summary.json','model_predictions.npz'):
        p=out/filename; copy=p.with_name(p.stem+'_micro1'+p.suffix)
        if not copy.exists(): shutil.copyfile(p,copy)
    np.savez(out/'model_predictions.npz',raw_codes=raw)
    (out/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with (out/'rows.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader();writer.writerows(rows)
    for group,block in summary['groups'].items():
        selected=[r for r in rows if group=='all' or r['group']==group]
        for key in block['metrics']:
            if key.startswith('model_'): block['metrics'][key]=stats([r[key] for r in selected])
        block['closed_beats_model']=sum(r['closed_l1']<r['model_l1'] for r in selected)
    summary.update(smoke_here=smoke,smoke_abs_diff=difference,model_micro=2,
                   replay='micro=1 first pass archived; micro=2 matches original evaluation')
    write_json(out/'summary.json',summary)


if __name__=='__main__': main()
