"""Dense cross-example stage grids, with explicit emphasis on the hue update."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import rgb_to_hsv, Normalize
from matplotlib import pyplot as plt, cm

from tools.build_appendix_paper_samples import PAPER, tex
from tools.build_appendix_local_detail import compile_pdf

WORK=Path('/home/bc/data/runs/paper_appendix_landscapes_20260920')
DEST=PAPER/'figures/appendix_landscape_stages'


def rank():
    records=json.loads((WORK/'local_results.json').read_text())
    scored=[]
    for r in records:
        folder=Path(r['folder'])
        with np.load(folder/'float_states.npz') as data:
            before=np.clip(data['recovery'][1],0,1);after=np.clip(data['recovery'][2],0,1)
        h,w=before.shape[:2]
        hsv=rgb_to_hsv(before);hsv_after=rgb_to_hsv(after)
        blue=(hsv[...,0]>.48)&(hsv[...,0]<.72)&(hsv[...,1]>.12)&(hsv[...,2]>.25)
        blue[int(h*.65):]=False
        white=(hsv[...,1]<.15)&(hsv[...,2]>.72);white[int(h*.65):]=False
        diff=np.abs(after-before).mean(-1)*100
        angle=np.abs(hsv[...,0]-hsv_after[...,0]);angle=np.minimum(angle,1-angle)*360
        area=float(blue.mean())
        amount=float(diff[blue].mean()) if blue.any() else 0.
        hue=float(angle[blue].mean()) if blue.any() else 0.
        row=dict(**r,aspect=w/h,blue_area=area,white_area=float(white.mean()),
                 blue_action=amount,blue_hue_degrees=hue,
                 score=amount*np.sqrt(area)*(.5+min(hue/5,1)))
        scored.append(row)
    scored.sort(key=lambda r:r['score'],reverse=True)
    (WORK/'hue_ranking.json').write_text(json.dumps(scored,indent=2)+'\n')
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
    # Internal selection aid: actual full frames before/after only the hue step.
    for part in range(4):
        canvas=Image.new('RGB',(1200,1000),'white');draw=ImageDraw.Draw(canvas)
        for i,r in enumerate(scored[part*10:part*10+10]):
            x=(i%2)*600;y=(i//2)*200;folder=Path(r['folder'])
            draw.text((x+2,y+2),f'{folder.name} blue {r["blue_area"]:.2f} change {r["blue_action"]:.2f} hue {r["blue_hue_degrees"]:.1f}',font=font,fill='black')
            for j,name in enumerate(['recovery_1','recovery_2']):
                frame=Image.open(folder/(name+'.png'));frame.thumbnail((293,173))
                canvas.paste(frame,(x+j*300+(293-frame.width)//2,y+24))
        canvas.save(WORK/f'hue_selection_internal_{part}.png')
    print(json.dumps([dict(case=Path(r['folder']).name,aspect=r['aspect'],blue=r['blue_area'],action=r['blue_action'],hue=r['blue_hue_degrees']) for r in scored[:15]]),flush=True)


def window(energy, cw, ch):
    h,w=energy.shape;cw=min(cw,w);ch=min(ch,h)
    integ=np.pad(energy.astype(np.float64),((1,0),(1,0))).cumsum(0).cumsum(1)
    total=integ[ch:,cw:]-integ[:-ch,cw:]-integ[ch:,:-cw]+integ[:-ch,:-cw]
    y,x=np.unravel_index(total.argmax(),total.shape)
    return [int(x),int(y),int(x+cw),int(y+ch)]


def prepare(r,index):
    if r['stage_order']!=['global','hue','lum_shadow','lum_mid','lum_high','geom']:
        raise ValueError('Unexpected execution order')
    src=Path(r['folder']);folder=DEST/f'case_{index:02d}';folder.mkdir(parents=True,exist_ok=True)
    with np.load(src/'float_states.npz') as data:
        states=np.clip(data['recovery'],0,1);masks=data['masks']
    h,w=states.shape[1:3]
    hsv=rgb_to_hsv(states[1]);blue=(hsv[...,0]>.47)&(hsv[...,0]<.74)&(hsv[...,1]>.09)
    blue[int(h*.7):]=False
    delta=np.abs(np.diff(states,axis=0)).mean(-1)*100
    gray=states[1].mean(-1)
    gy,gx=np.gradient(gray);texture=np.abs(gx)+np.abs(gy)
    energy=delta[1]*(.15+blue)*(.35+np.minimum(texture/max(float(texture.mean()),1e-6),3))
    # One fixed portrait-shaped detail window for all seven execution states.
    # Full context is retained in the source manifest; a wide sky crop is shown
    # separately in the hue before/after page.
    cw=int(w*.35);ch=min(h,round(cw/.82))
    box=window(energy,cw,ch)
    wide=window(energy,w,min(h,round(w/2.4)))
    x0,y0,x1,y1=box
    for step in range(7):
        frame=Image.open(src/f'recovery_{step}.png').convert('RGB')
        frame.crop(box).save(folder/f'state_{step}.png')
        if step in (1,2):
            frame.crop(wide).save(folder/f'hue_{step}.png')
    for step in range(2,6):
        mask=masks[step-1,y0:y1,x0:x1]
        Image.fromarray(np.rint(mask*255).astype(np.uint8)).save(folder/f'support_{step}.png')
    result=dict(**r,crop_xyxy=box,wide_hue_crop_xyxy=wide,
                display='matched native-pixel crops; no aspect distortion or contrast modification',
                crop_selection='texture-weighted hue-stage action energy, emphasizing blue/cyan pixels in the upper image')
    (folder/'provenance.json').write_text(json.dumps(result,indent=2)+'\n')
    return result,delta[1:5,y0:y1,x0:x1]


def header():
    return [r'\documentclass[10pt,border=0pt]{standalone}',
            r'\usepackage{times,graphicx,array,xcolor}',
            r'\begin{document}\begin{minipage}{5.5in}\fontsize{8.5}{10}\selectfont\setlength{\parindent}{0pt}']


def grid(rows,height):
    lines=[r'\setlength{\tabcolsep}{.7pt}\begin{tabular}{@{}>{\raggedright\arraybackslash}m{.105\linewidth}*{6}{>{\centering\arraybackslash}m{.144\linewidth}}@{}}',
           ' & '+' & '.join(r'\textbf{'+chr(65+i)+'}' for i in range(6))+r'\\[2pt]']
    for label,names in rows:
        panels=[]
        for i,name in enumerate(names,1):
            path=f'figures/appendix_landscape_stages/case_{i:02d}/{name}.png'
            panels.append(r'\includegraphics[width=\linewidth,height='+str(height)+r'in,keepaspectratio]{'+path+'}')
        lines += [label+' & '+' & '.join(panels)+r'\\[1.5pt]']
    lines += [r'\end{tabular}\par']
    return lines


def finish(name,lines):
    lines += [r'\end{minipage}\end{document}']
    path=DEST/(name+'.tex');path.write_text('\n'.join(lines)+'\n');compile_pdf(path,DEST/(name+'.pdf'))


def build(names):
    all_records=json.loads((WORK/'hue_ranking.json').read_text())
    records=[next(r for r in all_records if Path(r['folder']).name==name) for name in names]
    if len(records)!=6:
        raise ValueError('This layout requires six distinct source images')
    DEST.mkdir(exist_ok=True)
    prepared=[prepare(r,i) for i,r in enumerate(records,1)]
    maximum=max(1.,float(np.ceil(max(d.max() for r,d in prepared))))
    for i,(r,delta) in enumerate(prepared,1):
        for j,step in enumerate(range(2,6)):
            image=(matplotlib.colormaps['inferno'](delta[j]/maximum)[...,:3]*255).astype(np.uint8)
            Image.fromarray(image).save(DEST/f'case_{i:02d}/residual_{step}.png')
    labels=[r'Input',r'\shortstack[l]{1\\Global}',r'\textbf{\shortstack[l]{2\\Hue}}',
            r'\textbf{\shortstack[l]{3\\Shadows}}',r'\textbf{\shortstack[l]{4\\Midtones}}',
            r'\textbf{\shortstack[l]{5\\Highlights}}',r'\shortstack[l]{6\\Spatial}']
    lines=header()+[r'\textbf{Continuous recovery: six landscape examples}\par\smallskip']
    lines+=grid([(label,[f'state_{s}']*6) for s,label in enumerate(labels)],.96)
    lines += [r'\smallskip\textbf{Editing intents}\par',
              r'\setlength{\tabcolsep}{2pt}\begin{tabular}{@{}p{.49\linewidth}p{.49\linewidth}@{}}']
    for left,right in [(0,1),(2,3),(4,5)]:
        lines += [' & '.join(r'\textbf{'+chr(65+i)+r'.} '+tex(prepared[i][0]['annotation']['instruction_short']) for i in (left,right))+r'\\']
    lines += [r'\end{tabular}']
    finish('states_grid',lines)

    lines=header()+[r'\setlength{\tabcolsep}{2pt}\begin{tabular}{@{}*{2}{>{\centering\arraybackslash}m{.49\linewidth}}@{}}',
             r'Before hue ($z_1$) & After hue ($z_2$)\\[3pt]']
    for i,(r,delta) in enumerate(prepared,1):
        lines += [r'\multicolumn{2}{@{}l@{}}{\textbf{'+chr(64+i)+r'}}\\[1pt]',
             ' & '.join(r'\includegraphics[width=\linewidth]{figures/appendix_landscape_stages/case_'+f'{i:02d}'+f'/hue_{step}.png'+'}' for step in (1,2))+r'\\[3pt]']
    lines += [r'\end{tabular}']
    finish('hue_pairs',lines)

    fig,ax=plt.subplots(figsize=(3.8,.38));fig.subplots_adjust(left=.025,right=.97,top=.9,bottom=.6)
    bar=fig.colorbar(cm.ScalarMappable(norm=Normalize(0,maximum),cmap='inferno'),cax=ax,orientation='horizontal')
    bar.set_ticks([0,maximum/2,maximum]);bar.ax.tick_params(labelsize=8,length=2)
    bar.set_label('Actual change (mean absolute RGB difference x100)',fontsize=8,labelpad=1)
    fig.savefig(DEST/'colorbar.pdf',bbox_inches='tight',pad_inches=.02);plt.close(fig)
    lines=header()+[r'\textbf{Intermediate color control: steps 2--5}\par\smallskip',
                    r'\textbf{Recorded soft supports}\par\smallskip']
    stage_labels=[r'\shortstack[l]{2\\Hue}',r'\shortstack[l]{3\\Shadows}',
                  r'\shortstack[l]{4\\Midtones}',r'\shortstack[l]{5\\Highlights}']
    lines+=grid([(label,[f'support_{s}']*6) for s,label in zip(range(2,6),stage_labels)],.80)
    lines += [r'\smallskip\textbf{Actual adjacent-output residuals}\par\smallskip']
    lines+=grid([(label,[f'residual_{s}']*6) for s,label in zip(range(2,6),stage_labels)],.80)
    lines += [r'\smallskip\centering\includegraphics[width=.75\linewidth]{figures/appendix_landscape_stages/colorbar.pdf}']
    finish('supports_residuals',lines)
    (DEST/'selection.json').write_text(json.dumps(dict(candidates=len(all_records),cases=[r for r,d in prepared],
                residual_scale=[0,maximum],residual_definition='100*mean_RGB(abs(clip(z_k)-clip(z_(k-1))))',
                scope='target-conditioned closed-form recovery on training trajectories, not model prediction'),indent=2)+'\n')
    print(json.dumps(dict(n=6,residual_scale=[0,maximum],files=['states_grid.pdf','hue_pairs.pdf','supports_residuals.pdf'])),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['rank','build','audit']);p.add_argument('--cases')
    args=p.parse_args()
    if args.mode=='rank':rank()
    elif args.mode=='build':build(args.cases.split(','))
    else:audit()


def audit():
    manifest=json.loads((DEST/'selection.json').read_text())
    assert len({r['source_id'] for r in manifest['cases']})==6
    maximum=manifest['residual_scale'][1]; checks=[]
    for i,r in enumerate(manifest['cases'],1):
        src=Path(r['folder']);folder=DEST/f'case_{i:02d}'
        with np.load(src/'float_states.npz') as data:
            states=np.clip(data['recovery'],0,1);masks=data['masks']
        for step in range(7):
            actual=np.asarray(Image.open(folder/f'state_{step}.png'))
            expected=np.asarray(Image.open(src/f'recovery_{step}.png').crop(r['crop_xyxy']))
            assert np.array_equal(actual,expected)
        for step in (1,2):
            assert np.array_equal(np.asarray(Image.open(folder/f'hue_{step}.png')),
                                  np.asarray(Image.open(src/f'recovery_{step}.png').crop(r['wide_hue_crop_xyxy'])))
        x0,y0,x1,y1=r['crop_xyxy']; outside=[]
        for step in range(2,6):
            change=np.abs(states[step]-states[step-1]).mean(-1)*100
            expected=(matplotlib.colormaps['inferno'](change[y0:y1,x0:x1]/maximum)[...,:3]*255).astype(np.uint8)
            assert np.array_equal(expected,np.asarray(Image.open(folder/f'residual_{step}.png')))
            zero=masks[step-1]==0
            outside.append(float(change[zero].max()) if zero.any() else 0.)
            assert outside[-1]<1e-5
        checks.append(dict(source_id=r['source_id'],states=7,hue_pair=True,residuals=4,outside_zero=outside))
    (WORK/'grid_audit.json').write_text(json.dumps(dict(passed=True,checks=checks),indent=2)+'\n')
    print(json.dumps(dict(passed=True,state_crops=42,hue_images=12,residual_maps=24)),flush=True)


if __name__=='__main__':main()
