"""Explain the code-prediction gap with paired photographs and one aggregate."""
import json
from pathlib import Path
import sqlite3

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from veraretouch_sprf.readout import artedit_eval as AE


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    source=Path('/home/bc/data/runs/epr071_recovery_smoke100_20260919')
    infer=Path('/home/bc/data/runs/epr071_mechanism_figures_20260919')
    out=Path('/home/bc/VeraRetouch/EPR/ICLR2027/figures/mechanism_candidates_20260919')
    rows=[json.loads(s) for s in (source/'rows.jsonl').read_text().splitlines()]
    records=json.loads((source/'manifest.json').read_text())['records']
    final=json.loads((infer/'final_rows.json').read_text())
    late=np.load(infer/'final_predictions.npz')
    codes=np.load(source/'rebuilt_supervision_codes.npz')
    assert list(late['sample_ids'])==[r['sample_id'] for r in records]
    assert list(codes['sample_ids'])==list(late['sample_ids'])
    # Fixed error quantiles, not selected for the largest model-vs-fit gap.
    order=np.argsort([r['closed_l1'] for r in rows])
    selected=[int(order[49]),int(order[99])]
    renderer=AE.GlutRenderer(AE.GEOMETRY,'cpu')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
                         'pdf.fonttype':42,'axes.spines.top':False,
                         'axes.spines.right':False})
    fig=plt.figure(figsize=(7.2,5.8))
    grid=fig.add_gridspec(3,4,height_ratios=[1,1,.85],hspace=.50,wspace=.08,
                          left=.04,right=.98,bottom=.13,top=.90)
    fig.text(.5,.97,'Can the color executor reproduce the target?',ha='center',fontsize=12,weight='bold')
    headings=['Input','Target','Predicted code','Fitted code\n(target used)']
    for ri,index in enumerate(selected):
        record=records[index]
        src=AE.load_rgb_u8(record['input_path']);gt=AE.load_rgb_u8(record['gt_path'])
        x=torch.as_tensor(src.astype(np.float32)/255).reshape(-1,3)
        frames=[src,gt]
        for w in (late['raw_codes'][index].reshape(3,-1),codes['restoration_codes'][index]):
            frame=renderer.apply(w,x).clamp(0,1)
            frames.append((frame*255+.5).to(torch.uint8).numpy().reshape(src.shape))
        for col,frame in enumerate(frames):
            ax=fig.add_subplot(grid[ri,col]);ax.imshow(frame);ax.axis('off')
            if ri==0:ax.set_title(headings[col],fontsize=9,pad=7)
            if col==0:
                label='Median-fit case' if ri==0 else 'Largest-fit-error case'
                ax.text(0,-.12,label,transform=ax.transAxes,fontsize=8)
            if col==2:
                ax.text(.5,-.12,f'Error: {final["rows"][index]["l1"]:.2f}',ha='center',transform=ax.transAxes,fontsize=9,color='#0072B2')
            if col==3:
                ax.text(.5,-.12,f'Error: {rows[index]["closed_l1"]:.2f}',ha='center',transform=ax.transAxes,fontsize=9,color='#007653')
    textax=fig.add_subplot(grid[2,:2]);textax.axis('off')
    textax.text(0,1,'Same input. Same color executor.\nOnly the code changes.',va='top',fontsize=10,weight='bold',linespacing=1.5)
    textax.text(0,.49,'Fitting uses the target image.\nIt diagnoses what the executor can fit;\nit is not target-free inference.',va='top',fontsize=8.5,linespacing=1.5)
    ax=fig.add_subplot(grid[2,2:])
    values=[final['l1'],np.mean([r['bank_l1'] for r in rows]),np.mean([r['closed_l1'] for r in rows])]
    labels=['Predicted','Bank*','Fitted*']
    ax.barh(labels,values,color=['#0072B2','#888888','#009E73'],height=.57)
    ax.invert_yaxis();ax.set_xlim(0,max(values)*1.22)
    for i,v in enumerate(values):ax.text(v+.12,i,f'{v:.2f}',va='center',fontsize=9)
    ax.set_xlabel(r'Mean error on 100 pairs (MAE $\times100$) $\downarrow$',fontsize=8)
    ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
    fig.text(.04,.014,'R final@6250; fixed 50 validation + 50 remaining images. *Target-dependent references.',fontsize=8)
    fig.savefig(out/'recovery_explained.pdf')
    fig.savefig(out/'recovery_explained.png',dpi=220)
    (out/'recovery_explained_selection.json').write_text(json.dumps(dict(
        sample_ids=[rows[i]['sample_id'] for i in selected],
        rule='Ranks 50 and 100 of target-conditioned fit MAE among the fixed 100 pairs',
        checkpoint=final['checkpoint'],caption='The fixed color executor fits the target more closely with target-conditioned codes than with predicted codes on the diagnostic set. The examples also show remaining fitting error. Bank selection uses a reduced-resolution screening followed by full-resolution rescoring of 20 candidates.'),indent=2)+'\n')


if __name__=='__main__':main()
