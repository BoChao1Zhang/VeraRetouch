"""Publication-sized mechanism figures from fixed, auditable inference records."""
import argparse
import csv
import json
from pathlib import Path
import sqlite3

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS={'best800':'#0072B2','final6250':'#D55E00','bank':'#777777','fit':'#009E73'}
STYLES={'best800':'-','final6250':'--','bank':':','fit':'-.'}
LABELS={'best800':'R best@800','final6250':'R final@6250','bank':'Bank comparison',
        'fit':'Fit (target available)'}


def ecdf(ax,values,name):
    values=np.sort(np.asarray(values))
    ax.step(values,np.arange(1,len(values)+1)/len(values),where='post',
            color=COLORS[name],ls=STYLES[name],lw=1.5,label=LABELS[name])


def save(fig,out,name):
    fig.savefig(out/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(out/(name+'.png'),dpi=200,bbox_inches='tight')
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',default='/home/bc/data/runs/epr071_recovery_smoke100_20260919')
    ap.add_argument('--inference',default='/home/bc/data/runs/epr071_mechanism_figures_20260919')
    ap.add_argument('--out',required=True)
    args=ap.parse_args();src=Path(args.source);infer=Path(args.inference);out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],
        'font.size':9,'axes.labelsize':9,'axes.titlesize':9,'xtick.labelsize':8,
        'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False})
    rows=[json.loads(l) for l in (src/'rows.jsonl').read_text().splitlines()]
    final=json.loads((infer/'final_rows.json').read_text())
    finals={r['sample_id']:r for r in final['rows']}
    assert len(rows)==len(finals)==100
    error={'best800':[r['model_l1'] for r in rows],
           'final6250':[finals[r['sample_id']]['l1'] for r in rows],
           'bank':[r['bank_l1'] for r in rows],'fit':[r['closed_l1'] for r in rows]}
    fig,axes=plt.subplots(1,2,figsize=(6.6,2.65),layout='constrained')
    for name in error:ecdf(axes[0],error[name],name)
    axes[0].set(xlabel=r'Image MAE $\times 100$',ylabel='Fraction of images',
                title='(a) Reconstruction-error distribution',xlim=(0,max(map(max,error.values()))*1.03),ylim=(0,1.02))
    axes[0].legend(loc='lower right',frameon=False)
    for group,marker,color in [('val50','o','#0072B2'),('rest50','^','#777777')]:
        rr=[r for r in rows if r['group']==group]
        axes[1].scatter([r['bank_l1'] for r in rr],[r['closed_l1'] for r in rr],
                        s=15,marker=marker,c=color,alpha=.8,label=group)
    limit=max(max(error['bank']),max(error['fit']))*1.08
    axes[1].plot([0,limit],[0,limit],color='black',lw=.8,ls='--')
    wins=sum(a<b for a,b in zip(error['fit'],error['bank']))
    axes[1].text(.04,.96,f'{wins}/100 below diagonal',transform=axes[1].transAxes,va='top',fontsize=8)
    axes[1].set(xlabel=r'Bank comparison MAE $\times 100$',ylabel=r'Fitted MAE $\times 100$',
                title='(b) Pair-conditioned fit vs. bank',xlim=(0,limit),ylim=(0,limit))
    axes[1].legend(loc='lower right',frameon=False)
    save(fig,out,'recovery_capacity')

    magnitude={'best800':[r['model_amplitude_ratio'] for r in rows],
               'final6250':[finals[r['sample_id']]['amplitude_ratio'] for r in rows],
               'fit':[r['closed_amplitude_ratio'] for r in rows]}
    direction={'best800':[r['model_cos'] for r in rows],
               'final6250':[finals[r['sample_id']]['cos'] for r in rows],
               'fit':[r['closed_cos'] for r in rows]}
    fig,axes=plt.subplots(1,2,figsize=(6.6,2.65),layout='constrained')
    for name in magnitude:
        ecdf(axes[0],magnitude[name],name);ecdf(axes[1],direction[name],name)
    axes[0].axvline(1,color='black',lw=.8,ls=':',label='Target magnitude')
    axes[0].set(xlabel='Edit-magnitude ratio (target = 1)',ylabel='Fraction of images',
                title='(a) Action magnitude',xlim=(0,max(map(max,magnitude.values()))*1.03),ylim=(0,1.02))
    axes[1].set(xlabel='Cosine with target pixel change',ylabel='Fraction of images',
                title='(b) Action direction',xlim=(-1.02,1.02),ylim=(0,1.02))
    axes[0].legend(loc='lower right',frameon=False);axes[1].legend(loc='upper left',frameon=False)
    save(fig,out,'action_magnitude_direction')

    # Probe the code as a color function; no t-SNE or bank-boundary inference.
    import torch
    from veraretouch_sprf.readout import artedit_eval as AE
    from tools.epr059_glutbasis.common import glut_features
    torch.set_num_threads(4)
    renderer=AE.GlutRenderer(AE.GEOMETRY,'cpu')
    t=torch.linspace(0,1,256);query=t[:,None].expand(-1,3)
    phi=glut_features(query,renderer.basis.geometry)
    best=np.load(src/'model_predictions.npz')['raw_codes'].reshape(100,3,-1)
    fitted=np.load(src/'rebuilt_supervision_codes.npz')['restoration_codes']
    late=np.load(infer/'final_predictions.npz')['raw_codes'].reshape(100,3,-1)
    selected=['artedit_en_0270','artedit_en_0254']
    code_sets=[best,late,fitted];names=['best800','final6250','fit']
    probe=[]
    for sample in selected:
        i=[r['sample_id'] for r in rows].index(sample)
        probe.append([(phi@torch.tensor(a[i]).T).detach().numpy() for a in code_sets])
    low=min(0,min(float(v.min()) for row in probe for v in row))
    high=max(1,max(float(v.max()) for row in probe for v in row))
    fig,axes=plt.subplots(2,3,figsize=(6.6,3.9),layout='constrained',sharex=True,sharey=True)
    for r,sample in enumerate(selected):
        for c,name in enumerate(names):
            ax=axes[r,c];ax.plot(t,t,color='black',ls=':',lw=.7)
            for ch,color,ls in [(0,'#D55E00','-'),(1,'#009E73','--'),(2,'#0072B2','-.')]:
                ax.plot(t,probe[r][c][:,ch],color=color,ls=ls,lw=1.2,label='RGB'[ch])
            ax.set(xlim=(0,1),ylim=(low-.03,high+.03))
            if r==0:ax.set_title(LABELS[name])
            if c==0:ax.set_ylabel(sample.replace('artedit_en_','Case ')+'\nOutput channel value')
            if r==1:ax.set_xlabel('Gray input level')
    axes[0,0].legend(frameon=False,loc='upper left',ncol=3,handlelength=1.2,columnspacing=.8)
    save(fig,out,'gray_ramp_responses')

    compact=[]
    for i,row in enumerate(rows):
        compact.append(dict(sample_id=row['sample_id'],group=row['group'],
          **{name+'_l1':v[i] for name,v in error.items()},
          **{name+'_magnitude_ratio':v[i] for name,v in magnitude.items()},
          **{name+'_direction_cosine':v[i] for name,v in direction.items()}))
    with (out/'figure_data.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(compact[0]));writer.writeheader();writer.writerows(compact)
    summary={name:dict(l1_mean=float(np.mean(error[name])),
                     magnitude_median=float(np.median(magnitude[name])),
                     direction_mean=float(np.mean(direction[name]))) for name in magnitude}
    (out/'figure_summary.json').write_text(json.dumps(dict(n=100,summary=summary,
        source=str(src),inference=str(infer),probe_sample_ids=selected,
        probe='unclipped continuous Gaussian-basis response on neutral gray; black line is identity'),indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
