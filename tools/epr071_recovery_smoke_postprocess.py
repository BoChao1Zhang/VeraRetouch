"""Functional bank-distance check and reproducible contact sheet for the smoke."""
import argparse
import json
from pathlib import Path
import sqlite3

import numpy as np
from PIL import Image, ImageDraw
import torch

from veraretouch_sprf.readout import artedit_eval as AE, select_train as ST
from tools.epr071_val50_diag import PROTOSET, stats


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',required=True)
    ap.add_argument('--device',default='cuda:0')
    args=ap.parse_args()
    torch.set_num_threads(4)
    out=Path(args.out)
    manifest=json.loads((out/'manifest.json').read_text())
    codes=np.load(out/'rebuilt_supervision_codes.npz')['restoration_codes']
    predicted=np.load(out/'model_predictions.npz')['raw_codes']
    rows=[json.loads(l) for l in (out/'rows.jsonl').read_text().splitlines()]
    assert len(rows)==len(codes)==100
    renderer=AE.GlutRenderer(AE.GEOMETRY,args.device)
    bank=ST.Bank(args.device,path=PROTOSET)
    phi=renderer.phi.double()
    gram=phi.T@phi/len(phi)
    eig,vec=torch.linalg.eigh(gram)
    root=vec*eig.clamp_min(0).sqrt()[None,:]
    bank_metric=bank.codes.double()@root
    distances=[]
    for code in codes:
        metric=torch.as_tensor(code,device=args.device).double()@root
        mse=(bank_metric-metric).square().sum((-1,-2))/3
        distances.append(float(mse.min().sqrt()))
    data=dict(grid='33^3 uniform RGB; unclipped frozen-basis outputs',
              formula='sqrt(trace(delta W * mean(phi^T phi) * delta W^T)/3)',
              n=100,nearest_bank_function_rmse=stats(distances),
              matches_below_1e_6=sum(v<1e-6 for v in distances),
              rows=[dict(sample_id=r['sample_id'],rmse=v) for r,v in zip(rows,distances)])
    (out/'functional_bank_distance.json').write_text(json.dumps(data,indent=2)+'\n')
    # Show error quantiles, rather than selecting only successful examples.
    order=np.argsort([r['closed_l1'] for r in rows])
    selected=[int(order[j]) for j in (0,49,89,99)]
    w,h=320,235
    canvas=Image.new('RGB',(4*w,4*h+35),'white')
    draw=ImageDraw.Draw(canvas)
    for j,title in enumerate(('Input','Target','R best@800','Fresh closed-form fit (uses target)')):
        draw.text((j*w+8,10),title,fill='black')
    records=manifest['records']
    for r,i in enumerate(selected):
        rec=records[i]
        src,gt=AE.load_rgb_u8(rec['input_path']),AE.load_rgb_u8(rec['gt_path'])
        rgb=torch.as_tensor(src.astype(np.float32)/255,device=args.device).reshape(-1,3)
        images=[Image.fromarray(src),Image.fromarray(gt)]
        for code in (predicted[i].reshape(3,-1),codes[i]):
            pixels=renderer.apply(code,rgb).clamp(0,1)
            u8=(pixels*255+.5).to(torch.uint8).cpu().numpy().reshape(src.shape)
            images.append(Image.fromarray(u8))
        for c,im in enumerate(images):
            im.thumbnail((w-12,h-42))
            canvas.paste(im,(c*w+(w-im.width)//2,35+r*h))
            label=rec['sample_id'] if c==0 else ''
            if c==2: label=f'L1x100={rows[i]["model_l1"]:.3f}'
            if c==3: label=f'L1x100={rows[i]["closed_l1"]:.3f}'
            draw.text((c*w+8,35+r*h+h-30),label,fill='black')
    canvas.save(out/'contact_sheet.png')
    (out/'contact_sheet_selection.json').write_text(json.dumps(
        dict(rule='closed-fit error sorted ranks 1,50,90,100',
             ids=[rows[i]['sample_id'] for i in selected]),indent=2)+'\n')
    print(json.dumps({k:v for k,v in data.items() if k!='rows'},indent=2))


if __name__=='__main__':
    main()
