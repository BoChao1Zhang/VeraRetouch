"""Three-part author-directed layout with verified mask-in/mask-out controls."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,colors,cm

from tools.build_appendix_paper_samples import PAPER,tex
from tools.build_appendix_local_detail import compile_pdf

OUT=PAPER/'drafts/three_part_local_candidate'
SOURCE=PAPER/'figures/appendix_local_large'
NAMES=['Global','Hue','Shadows','Midtones','Highlights','Spatial']


def window_sum(a,cw,ch):
    integral=np.pad(a.astype(np.float64),((1,0),(1,0))).cumsum(0).cumsum(1)
    return integral[ch:,cw:]-integral[:-ch,cw:]-integral[ch:,:-cw]+integral[:-ch,:-cw]


def select_window(score,valid,cw,ch):
    if not valid.any():return None
    values=np.where(valid,window_sum(score,cw,ch)/(cw*ch),-np.inf)
    y,x=np.unravel_index(values.argmax(),values.shape)
    return dict(box=[int(x),int(y),int(x+cw),int(y+ch)],score=float(values[y,x]))


def prepare(index, meta=None, output=None, require_spatial=False, subject_context=False,
            subject_priority=None, allow_low_support=False):
    meta=meta or json.loads((SOURCE/f'case_{index:02d}/provenance.json').read_text())
    src=Path(meta['folder']);folder=output or OUT/f'case_{index:02d}';folder.mkdir(parents=True,exist_ok=True)
    with np.load(src/'float_states.npz') as data:
        states=np.clip(data['recovery'],0,1);masks=data['masks']
    h,w=states.shape[1:3]
    if subject_priority is None:subject_priority=require_spatial
    if subject_context:
        from tools.local_subject_crop import context_windows
        candidates=[]
        for step in range(2,7):
            r=context_windows(states,masks,step,subject_priority=subject_priority,allow_low_support=allow_low_support)
            candidates.append(dict(r,stage=NAMES[step-1]) if r else dict(step=step,eligible=False))
        eligible=sorted([r for r in candidates if r['eligible']],key=lambda r:r['inside']['score'],reverse=True)
        if require_spatial:
            picked=[r for r in eligible if r['step']==6]+[r for r in eligible if r['step']!=6][:1]
        else:picked=eligible[:2]
    else:
        picked=[]
    # Use an equal-sized window inside and outside, and across candidate steps.
    aspect=1.4
    for fraction in [.04,.02,.01,.005]:
        if subject_context:break
        cw=round(np.sqrt(fraction*h*w*aspect));ch=round(cw/aspect);candidates=[]
        for step in range(2,7):
            mask=masks[step-1];delta=np.abs(states[step]-states[step-1]).mean(-1)*100
            gy,gx=np.gradient(states[step-1].mean(-1));g=np.abs(gy)+np.abs(gx)
            textured=delta*(.25+np.minimum(g/max(float(g.mean()),1e-6),4))
            positive=window_sum(mask>0,cw,ch)==cw*ch
            zero=window_sum(mask!=0,cw,ch)==0
            inside=select_window(textured,positive,cw,ch)
            outside=select_window(g,zero,cw,ch)
            if inside is None or outside is None:
                candidates.append(dict(step=step,eligible=False,reason='No full inside/outside window at this common area'))
                continue
            x0,y0,x1,y1=outside['box']
            outside_change=float(delta[y0:y1,x0:x1].max())
            assert outside_change<1e-5
            candidates.append(dict(step=step,stage=NAMES[step-1],eligible=True,inside=inside,outside=outside,
                                   outside_max_change=outside_change))
        eligible=sorted([r for r in candidates if r['eligible']],key=lambda r:r['inside']['score'],reverse=True)
        if require_spatial:
            spatial=[r for r in eligible if r['step']==6]
            picked=spatial+[r for r in eligible if r['step']!=6][:1] if spatial else []
        else:
            picked=eligible[:2]
        if len(picked)==2:break
    if len(picked)!=2:raise ValueError(f'Not enough verifiable outside controls: {index}')
    picked.sort(key=lambda r:r['step'])
    for s in range(7):
        frame=Image.open(src/f'recovery_{s}.png').convert('RGB')
        frame.thumbnail((768,768),Image.Resampling.LANCZOS);frame.save(folder/f'z{s}.png')
    gt=Image.open(src/'reference.png').convert('RGB');gt.thumbnail((768,768),Image.Resampling.LANCZOS)
    gt.save(folder/'gt.png')
    maximum=max(1.,float(np.ceil(max(np.abs(states[r['step']]-states[r['step']-1]).mean(-1).max()*100 for r in picked))))
    for r in picked:
        step=r['step'];delta=np.abs(states[step]-states[step-1]).mean(-1)*100
        for area in ['inside','outside']:
            for state,name in [(step-1,'before'),(step,'after')]:
                img=Image.open(src/f'recovery_{state}.png').convert('RGB').crop(r[area]['box'])
                img.save(folder/f'{step}_{area}_{name}.png')
        if r.get('outside_kind','zero')=='zero':
            assert np.array_equal(np.asarray(Image.open(folder/f'{step}_outside_before.png')),
                                  np.asarray(Image.open(folder/f'{step}_outside_after.png')))
        Image.fromarray(np.rint(masks[step-1]*255).astype(np.uint8)).save(folder/f'{step}_support.png')
        Image.fromarray((matplotlib.colormaps['inferno'](delta/maximum)[...,:3]*255).astype(np.uint8)).save(folder/f'{step}_residual.png')
    fig,ax=plt.subplots(figsize=(3.8,.35));fig.subplots_adjust(left=.02,right=.98,bottom=.6,top=.9)
    bar=fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(0,maximum),cmap='inferno'),cax=ax,orientation='horizontal')
    bar.set_ticks([0,maximum/2,maximum]);bar.ax.tick_params(labelsize=8,length=2)
    bar.set_label('Actual step change (mean absolute RGB difference x100)',fontsize=8,labelpad=1)
    fig.savefig(folder/'colorbar.pdf',bbox_inches='tight',pad_inches=.01);plt.close(fig)
    record=dict(source_id=meta['source_id'],source_key=meta['key'],source_folder=str(src),image_size=[w,h],
                instruction=meta['annotation']['instruction_medium'],picked=picked,candidates=candidates,
                region_area_fraction=(None if subject_context else cw*ch/(w*h)),region_definition='inside: all weights >0; outside: all weights exactly zero',
                selection='top two texture-weighted regional changes among steps 2--6 with both verified control windows',
                residual_scale=[0,maximum],gap_css_px=5,gap_pdf_bp=3.75)
    record['annotation']=meta['annotation']
    record['pool']=meta.get('pool','unsplash')
    record['execution']=meta.get('execution','Target-conditioned recovery, not model inference.')
    if require_spatial:
        record['selection']='spatial subject move plus strongest texture-weighted regional move among steps 2--5; both require exact-zero outside controls'
    if subject_context:
        record.update(region_definition='edit ROI: >=80% subject coverage when applicable, >=50% active stage support; control ROI: exact zero support',
                      selection=('large subject-oriented spatial detail plus strongest eligible local detail' if subject_priority else
                                 'large edit-region context; spatial step plus strongest eligible local detail'),
                      subject_context=True,subject_priority=subject_priority,allow_low_support=allow_low_support)
        if allow_low_support:record['region_definition']+='; low_support controls are labeled explicitly and may change'
    (folder/'audit.json').write_text(json.dumps(record,indent=2)+'\n')
    return folder,record


def picture(folder,name,boxes=None,size=None):
    img=r'\includegraphics[width=\linewidth]{'+str((folder/(name+'.png')).relative_to(PAPER))+'}'
    if not boxes:return img
    w,h=size
    lines=[r'\begin{tikzpicture}\node[anchor=south west,inner sep=0] (im) at (0,0) {'+img+r'};',
           r'\begin{scope}[x={(im.south east)},y={(im.north west)}]']
    for area,color in [('inside','orange!90!black'),('outside','cyan!70!black')]:
        x0,y0,x1,y1=boxes[area]['box']
        lines.append(rf'\draw[{color},line width=.65pt] ({x0/w:.6f},{1-y1/h:.6f}) rectangle ({x1/w:.6f},{1-y0/h:.6f});')
    return ''.join(lines)+r'\end{scope}\pgfresetboundingbox\path[use as bounding box] (im.south west) rectangle (im.north east);\end{tikzpicture}'


def row(folder,names,labels):
    n=len(names)
    lines=[rf'\setlength{{\CellWidth}}{{\dimexpr(\linewidth-{n-1}\PhotoGap)/{n}\relax}}',r'\noindent']
    cells=[]
    for name,label in zip(names,labels):
        cells.append(r'\begin{minipage}[t]{\CellWidth}\vspace{0pt}\centering '+label+r'\par'+picture(folder,name)+r'\end{minipage}')
    lines += [(r'\hspace{\PhotoGap}').join(cells)+r'\par']
    return lines


def render(folder,r,mode):
    lines=[r'\documentclass{article}',r'\usepackage{iclr2027/iclr2027_conference,times}',
           r'\usepackage{graphicx,tikz,array,amsmath}',r'\newlength{\PhotoGap}\setlength{\PhotoGap}{3.75bp}',
           r'\newlength{\CellWidth}',r'\begin{document}\fontsize{9}{10.5}\selectfont\setlength{\parskip}{0pt}\setlength{\parindent}{0pt}',
           r'\textbf{1. Before / After / GT}\par']
    lines+=row(folder,['z0','z6','gt'],['Before','After','GT'])
    lines += [r'\smallskip\textbf{2. Iterative recovery}\par']
    lines+=row(folder,['z0','z1','z2','z3'],[r'Input ($z_0$)',r'$z_1$: Global',r'$z_2$: Hue',r'$z_3$: Shadows'])
    lines += [r'\vspace{\PhotoGap}']
    lines+=row(folder,['z4','z5','z6','gt'],[r'$z_4$: Midtones',r'$z_5$: Highlights',r'$z_6$: Spatial','GT'])
    # A portrait's width-fitted overview and 2x4 sequence already occupy most
    # of a page. Do not shrink either to fit the local section below them.
    portrait=r['image_size'][1]>r['image_size'][0]
    if portrait or mode=='region_rows':
        lines += [r'\par\smallskip\textit{Selected target-conditioned recovery example. Original aspect ratios are retained.}',r'\clearpage']
    lines += [r'\smallskip\textbf{3. Two selected local updates}\par',
              r'Orange: mask-in ($\beta>0$). Blue: mask-out ($\beta=0$).\par',
              r'\setlength{\CellWidth}{\dimexpr\linewidth-2\PhotoGap\relax}']
    for chosen in r['picked']:
        step=chosen['step']
        lines += [r'\smallskip\textbf{Step '+str(step)+' / '+chosen['stage']+r'}\par']
        def colwidth(f):return str(f)+r'\CellWidth'
        cells=[]
        if mode=='step_rows':
            for name,label in [(f'z{step-1}','Before'),(f'z{step}','After')]:
                cells.append(r'\begin{minipage}[c]{'+colwidth(.4)+r'}\centering '+label+r'\par'+picture(folder,name,chosen,r['image_size'])+r'\end{minipage}')
        else:
            for when,label in [('before','Before'),('after','After')]:
                body=r'\begin{minipage}[c]{'+colwidth(.4)+r'}\centering '+label+r'\par '
                body+=r'Mask-in\par'+picture(folder,f'{step}_inside_{when}')
                body+=r'\par\vspace{\PhotoGap}Mask-out\par'+picture(folder,f'{step}_outside_{when}')
                cells.append(body+r'\end{minipage}')
        auxiliary=r'\begin{minipage}[c]{'+colwidth(.2)+r'}\centering Support\par'
        auxiliary+=picture(folder,f'{step}_support',chosen,r['image_size'])
        auxiliary+=r'\par\vspace{\PhotoGap}Residual\par'+picture(folder,f'{step}_residual',chosen,r['image_size'])
        cells.append(auxiliary+r'\end{minipage}')
        lines += [r'\noindent'+r'\hspace{\PhotoGap}'.join(cells)+r'\par']
    lines += [r'\smallskip\noindent\hfill\includegraphics[width=.7\linewidth]{'+str((folder/'colorbar.pdf').relative_to(PAPER))+r'}\hfill\null\par',
              r'\smallskip\textit{The two steps are selected by regional change among moves with both verified mask-in and mask-out controls. The same crop coordinates are used before and after each step.}',
              r'\end{document}']
    source=folder/(mode+'.tex');source.write_text('\n'.join(lines)+'\n')
    compile_pdf(source,folder/(mode+'.pdf'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--cases',default='1,4');args=p.parse_args()
    for index in map(int,args.cases.split(',')):
        folder,record=prepare(index)
        for mode in ['step_rows','region_rows']:render(folder,record,mode)
        print(json.dumps(dict(case=index,selected_steps=[r['step'] for r in record['picked']],
                              rejected=[r['step'] for r in record['candidates'] if not r['eligible']])),flush=True)


if __name__=='__main__':main()
