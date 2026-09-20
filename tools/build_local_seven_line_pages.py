"""Native LaTeX local-detail pages with exact 84pt (=7 review lines) images.

Only height is specified, never a width cap or a page-level resize. Each page
contains two moves. Every move retains full before/after states, support,
matched before/after zooms, and its real residual field.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, cm, colors

from tools.build_appendix_paper_samples import PAPER,tex
from tools.build_appendix_local_detail import crop_box,detail_energy,compile_pdf

OUT=PAPER/'figures/appendix_local_large'
OLD=PAPER/'figures/appendix_local_detail'
SKY=Path('/home/bc/data/runs/paper_appendix_landscapes_20260920/local_09/provenance.json')
HEIGHT_PT=84


def prepare(record,index):
    folder=OUT/f'case_{index:02d}';folder.mkdir(parents=True,exist_ok=True)
    src=Path(record['folder'])
    with np.load(src/'float_states.npz') as data:
        states=np.clip(data['recovery'],0,1);masks=data['masks']
    residual=np.abs(np.diff(states,axis=0)).mean(-1)*100
    maximum=max(1.,float(np.ceil(residual.max())))
    boxes=[]
    for i in range(7):
        frame=Image.open(src/f'recovery_{i}.png').convert('RGB')
        frame.thumbnail((512,512),Image.Resampling.LANCZOS)
        frame.save(folder/f'state_{i}.png')
    reference=Image.open(src/'reference.png').convert('RGB')
    reference.thumbnail((512,512),Image.Resampling.LANCZOS);reference.save(folder/'reference.png')
    for i in range(1,7):
        energy=detail_energy(residual[i-1],states[i-1])
        if index==1 and i==2:
            energy[energy.shape[0]//2:]=0
        box=crop_box(energy);boxes.append(box)
        for step,name in [(i-1,'before'),(i,'after')]:
            Image.open(src/f'recovery_{step}.png').crop(box).save(folder/f'{name}_zoom_{i}.png')
        mask=Image.fromarray(np.rint(masks[i-1]*255).astype(np.uint8))
        mask.thumbnail((512,512),Image.Resampling.LANCZOS);mask.save(folder/f'support_{i}.png')
        heat=Image.fromarray((matplotlib.colormaps['inferno'](residual[i-1]/maximum)[...,:3]*255).astype(np.uint8))
        heat.thumbnail((512,512),Image.Resampling.LANCZOS);heat.save(folder/f'residual_{i}.png')
    fig,ax=plt.subplots(figsize=(3.8,.35));fig.subplots_adjust(left=.02,right=.98,bottom=.6,top=.9)
    bar=fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(0,maximum),cmap='inferno'),cax=ax,orientation='horizontal')
    bar.set_ticks([0,maximum/2,maximum]);bar.ax.tick_params(labelsize=8,length=2)
    bar.set_label('Actual change (mean absolute RGB difference x100)',fontsize=8,labelpad=1)
    fig.savefig(folder/'colorbar.pdf',bbox_inches='tight',pad_inches=.01);plt.close(fig)
    meta=dict(**record,crops_xyxy=boxes,image_height_tex_pt=84,review_line_pitch_tex_pt=12,
              image_height_review_lines=7,full_frame_display_max_side=512,
              residual_scale=[0,maximum],crop_policy='matched texture-weighted change crops; first case hue crop restricted to upper half')
    (folder/'provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
    return folder,meta


def image_cell(folder,name,box=None,size=None):
    path=folder/(name+'.png')
    w,h=Image.open(path).size
    if w/h*HEIGHT_PT>127:
        raise ValueError(f'Panel does not fit at seven-line height: {path}')
    image=r'\includegraphics[height=84pt]{'+str(path.relative_to(PAPER))+'}'
    if box is None:
        return image
    w,h=size;x0,y0,x1,y1=box
    return (r'\begin{tikzpicture}\node[anchor=south west,inner sep=0] (im) at (0,0) {'+image+r'};'
            +r'\begin{scope}[x={(im.south east)},y={(im.north west)}]'
            +rf'\draw[orange!90!black,line width=.5pt] ({x0/w:.6f},{1-y1/h:.6f}) rectangle ({x1/w:.6f},{1-y0/h:.6f});'
            +r'\end{scope}\end{tikzpicture}')


def panel_row(folder,names,labels,boxes=None,size=None):
    # Zero array strut and a one-point label gap avoid empty text rows.
    lines=[r'\noindent\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}*{3}{>{\centering\arraybackslash}p{.32\linewidth}}@{}}',
           ' & '.join(labels)+r'\\[1pt]']
    lines += [' & '.join(image_cell(folder,name,(boxes or [None]*3)[i],size) for i,name in enumerate(names))+r'\\',r'\end{tabular*}\par']
    return lines


def build_page(folder,record,index,part):
    start=part*2+1;end=start+1
    size=Image.open(Path(record['folder'])/'recovery_0.png').size
    lines=[r'\begingroup',r'\fontsize{9}{10.5}\selectfont\raggedright\setlength{\parindent}{0pt}\setlength{\parskip}{0pt}',
           r'\setlength{\tabcolsep}{1pt}\renewcommand{\arraystretch}{0}',
           r'\noindent\textbf{Local recovery '+str(index)+f': moves {start}--{end}'+r'}\par',
           r'\noindent\textbf{Request.} '+tex(record['annotation']['instruction_short'])+r'\par']
    lines += panel_row(folder,[f'state_{start-1}',f'state_{end}','reference'],
                       [rf'Pair input ($z_{start-1}$)',rf'Pair output ($z_{end}$)','Reference'])
    stage_names=['Global','Hue','Shadows','Midtones','Highlights','Spatial']
    for step in range(start,end+1):
        sentence=record['annotation']['cot'][step-1]['observation'].split('. ')[0].rstrip('.')+'.'
        if len(sentence.split())>22:
            sentence=' '.join(sentence.split()[:22]).rstrip('.,;:')+'...'
        lines += [r'\noindent\textbf{'+str(step)+'. '+stage_names[step-1]+r'.} '+tex(sentence)+r'\par']
        box=record['crops_xyxy'][step-1]
        lines += panel_row(folder,[f'state_{step-1}',f'state_{step}',f'support_{step}'],
                           ['Before','After','Support'],[box,box,None],size)
        lines += panel_row(folder,[f'before_zoom_{step}',f'after_zoom_{step}',f'residual_{step}'],
                           ['Before (zoom)','After (zoom)','Actual residual'])
    lines += [r'\noindent\hfill\includegraphics[width=.67\linewidth]{'+str((folder/'colorbar.pdf').relative_to(PAPER))+r'}\hfill\null\par',r'\endgroup']
    body=folder/f'part_{part+1}.tex';body.write_text('\n'.join(lines)+'\n')
    standalone=folder/f'preview_{part+1}.tex'
    standalone.write_text(r'\documentclass[10pt,border=0pt]{standalone}'+'\n'+
        r'\usepackage{times,graphicx,tikz,array,xcolor}'+'\n'+
        r'\begin{document}\begin{minipage}{5.5in}\input{'+str(body.relative_to(PAPER))+r'}\end{minipage}\end{document}'+'\n')
    compile_pdf(standalone,folder/f'preview_{part+1}.pdf')
    return str(body.relative_to(PAPER))


def main():
    p=argparse.ArgumentParser();p.add_argument('--first-only',action='store_true');args=p.parse_args()
    records=json.loads((OLD/'selection.json').read_text())['cases']
    records[0]=json.loads(SKY.read_text())
    if args.first_only:records=records[:1]
    outputs=[]
    for index,record in enumerate(records,1):
        folder,meta=prepare(record,index)
        for part in range(3):
            outputs.append(dict(case=index,part=part+1,tex=build_page(folder,meta,index,part)))
    (OUT/'manifest.json').write_text(json.dumps(dict(image_height_tex_pt=84,review_line_pitch_tex_pt=12,panels_per_page=15,pages=outputs),indent=2)+'\n')
    if not args.first_only:
        section=[r'% Image height = 7 x 12pt review-ruler intervals. No outer scaling.',
                 r'\clearpage',r'\section{Local Editing Trajectories}',r'\label{app:maskgrade_examples}']
        anchors=['sky','deer','industrial','flowers','dog']
        for i,item in enumerate(outputs):
            if i:section.append(r'\clearpage')
            start=(item['part']-1)*2+1;end=start+1
            label='fig:local_recovery_'+anchors[item['case']-1]
            if item['part']>1:label+='_part_'+str(item['part'])
            section += [r'\begin{figure}[!ht]',r'\centering',r'\input{'+item['tex']+'}',
                        r'\caption{Target-conditioned recovery, example '+str(item['case'])+f', moves {start}--{end}.'+
                        ' Each move shows the executed before/after states, its recorded support, matched crops, and the actual residual. The residual scale is shared across all six moves of this example.}',
                        r'\label{'+label+'}',r'\end{figure}',r'\FloatBarrier']
        (PAPER/'sections/appendix/local_seven_line_pages.tex').write_text('\n'.join(section)+'\n')
    print(json.dumps(outputs),flush=True)


if __name__=='__main__':main()
