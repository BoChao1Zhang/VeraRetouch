"""Infer final@6250 on the fixed 100-pair diagnostic set for mechanism figures."""
import argparse
import json
from pathlib import Path
import shutil
import sqlite3
import time

import numpy as np
import torch

from tools.epr071_val50_diag import load_checkpoint,PROTOSET,JOURNAL,HALVES,geometry_of,sha256_file
from veraretouch_sprf.readout import select_train as ST,epr071_data as E71,artedit_eval as AE


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',default='/home/bc/data/runs/epr071_recovery_smoke100_20260919')
    ap.add_argument('--out',required=True)
    args=ap.parse_args();source=Path(args.source);out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'final_rows.json').exists():raise SystemExit('Completed output exists')
    torch.set_num_threads(4);torch.manual_seed(20260919)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.20)
    manifest=json.loads((source/'manifest.json').read_text())
    records=manifest['records'];assert len(records)==100
    checkpoint=out/'R_final6250_snapshot.pt'
    if not checkpoint.exists():shutil.copyfile(E71.STORE/'train/R/run/final.pt',checkpoint)
    began=time.monotonic();device='cuda:0'
    bank=ST.Bank(device,path=PROTOSET);mean,std,_=E71.load_scaler()
    model,facts=load_checkpoint(checkpoint,bank,device)
    if facts['step']!=6250 or facts['arm']!='R':raise ValueError(facts)
    model.model.eval();model.head.eval()
    evaluator=ST.SelectEval(model,bank,device,micro=2,num_workers=0,render_chunk=6,
                            halves_path=HALVES,arm='R',code_mean=mean,code_std=std,
                            journal=str(JOURNAL))
    _,q=evaluator.predict(records);raw=q*std+mean
    np.savez(out/'final_predictions.npz',sample_ids=np.array([r['sample_id'] for r in records]),
             raw_codes=raw)
    old=json.loads((E71.STORE/'train/R/run/evaluations/artedit_full_final.json').read_text())
    logged={r['sample_id']:r['top1__l1']*100 for r in old['rows']}
    val_record=json.loads((E71.STORE/'train/R/run/evaluations/artedit_val50_final.json').read_text())
    val_reference=float(np.mean([r['top1__l1']*100 for r in val_record['rows']]))
    rows=[]
    for i,rec in enumerate(records):
        src,gt=AE.load_rgb_u8(rec['input_path']),AE.load_rgb_u8(rec['gt_path'])
        if src.shape!=gt.shape:raise ValueError('geometry mismatch')
        z=torch.as_tensor(src.astype(np.float32)/255,device=device).reshape(-1,3)
        target=torch.as_tensor(gt.astype(np.float32)/255,device=device).reshape(-1,3)
        rendered=evaluator.renderer.apply(raw[i].reshape(3,-1),z).clamp(0,1)
        pred=(rendered*255+.5).to(torch.uint8).double()/255
        row=geometry_of(z.double(),target.double(),pred)
        row['l1']*=100
        row.update(sample_id=rec['sample_id'],group='val50' if i<50 else 'rest50',
                   magnitude_gt=float((target-z).abs().mean()),
                   logged_l1=logged[rec['sample_id']])
        row['amplitude_ratio']=row['m_pred']/max(row['magnitude_gt'],1e-12)
        rows.append(row)
    summary=dict(checkpoint=facts,source_manifest_sha256=sha256_file(source/'manifest.json'),
        source_script_sha256=sha256_file(Path(__file__)),micro=2,n=100,
        l1=float(np.mean([r['l1'] for r in rows])),
        val50_l1=float(np.mean([r['l1'] for r in rows[:50]])),
        max_logged_l1_delta=max(abs(r['l1']-r['logged_l1']) for r in rows),
        val50_reference=val_reference,
        val50_reference_abs_diff=abs(float(np.mean([r['l1'] for r in rows[:50]]))-val_reference),
        seconds=time.monotonic()-began,rows=rows)
    (out/'final_rows.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2),flush=True)
    # The selected rest images are re-batched, so BF16 batch-context drift is
    # reported per sample; val50 retains its original grouping exactly.
    if summary['val50_reference_abs_diff']>1e-4:
        raise RuntimeError('val50 replay mismatch')


if __name__=='__main__':main()
