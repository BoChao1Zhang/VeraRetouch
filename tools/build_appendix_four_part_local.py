"""Author-directed four-part local pages; native LaTeX and audited real crops.

Inputs are archived target-conditioned recoveries, never model predictions.
The excerpted reasoning remains verbatim observation/support annotation.
"""
import argparse
import json
import re
from pathlib import Path
import subprocess

from tools.preview_three_part_local import prepare, picture, row, NAMES
from tools.build_appendix_paper_samples import PAPER, tex
from tools.build_appendix_local_detail import compile_pdf

ASSETS=PAPER/'figures/appendix_local_equal'
DRAFTS=PAPER/'drafts/local_equal_20260920'


def heading(text):
    return r'\par\smallskip\noindent\textbf{'+text+r'}\par\nobreak'


def local_panel(folder, record, chosen):
    """Two independently aligned rows: support/before/after, residual/before/after."""
    step=chosen['step']
    lines=[r'\begin{minipage}{\linewidth}',heading('Step '+str(step)+' / '+chosen['stage']),
           r'\setlength{\CellWidth}{\dimexpr(\linewidth-2\PhotoGap)/3\relax}']
    for area,aux,label in [('inside','support','Edit ROI'),('outside','residual','Mask-out')]:
        if area=='outside' and chosen.get('outside_kind')=='low_support':label='Low-support ROI'
        # Keep each auxiliary on the SAME row as its paired crops. Equal-height
        # boxes align centers while preserving the auxiliary's aspect ratio.
        w,h=record['image_size']
        x0,y0,x1,y1=chosen[area]['box']
        lines += [rf'\setlength{{\CropHeight}}{{{h/w:.8f}\CellWidth}}']
        auximg=picture(folder,f'{step}_{aux}',chosen,record['image_size']).replace(
            r'width=\linewidth',r'width=\linewidth,height=\CropHeight,keepaspectratio')
        cells=[r'\begin{minipage}[t]{\CellWidth}\vspace{0pt}\centering '+aux.capitalize()+
               r'\par\begin{minipage}[c][\CropHeight][c]{\linewidth}\centering '+auximg+r'\end{minipage}\end{minipage}']
        for when,title in [('before','Before'),('after','After')]:
            crop=picture(folder,f'{step}_{area}_{when}').replace(r'width=\linewidth',r'width=\linewidth,height=\CropHeight,keepaspectratio')
            cells.append(r'\begin{minipage}[t]{\CellWidth}\vspace{0pt}\centering '+title+' / '+label+
                         r'\par\begin{minipage}[c][\CropHeight][c]{\linewidth}\centering '+crop+r'\end{minipage}\end{minipage}')
        lines += [r'\noindent'+r'\hspace{\PhotoGap}'.join(cells)+r'\par']
        if area=='inside':lines += [r'\vspace{\PhotoGap}']
    lines += [r'\end{minipage}\par']
    return lines


def reasoning(folder,record,compact=False):
    selected={r['step'] for r in record['picked']}
    lines=([r'\fontsize{10.5}{12.3}\selectfont'] if not compact else [])+[heading('4. Stagewise reasoning'),
           r'\textit{Verbatim excerpts from the archived trajectory annotation.}\par\smallskip']
    for step in record['annotation']['cot']:
        number=step['step']
        title=f"Step {number} / {NAMES[number-1]}"+(' (detail shown above)' if number in selected else '')
        observation=step['observation'];support=step['mask']
        if compact:
            observation=re.split(r'(?<=[.!?])\s+(?=[A-Z])',observation)[0]
            support=re.split(r'(?<=[.!?])\s+(?=[A-Z])',support)[0]
            lines += [r'\begin{minipage}{\linewidth}\textbf{'+title+r'.} '+tex(observation)+
                      r' \textit{Support:} '+tex(support)+r'\end{minipage}\par\smallskip']
            continue
        lines += [r'\begin{minipage}{\linewidth}',r'\textbf{'+title+r'}\par',
                  r'\textit{Observation.} '+tex(observation)+r'\par',
                  r'\textit{Support.} '+tex(support)+r'\par']
        if number in selected:
            actions=[s for s in re.split(r'(?<=[.!?])\s+(?=[A-Z])',step['adjustment'])
                     if not re.search(r'\d',s)]
            if actions:
                lines += [r'\textit{Action.} '+tex(actions[-1])+r'\par']
        lines += [r'\end{minipage}\par\smallskip']
    return lines


def body(folder,record,title):
    # Scope all typography locally; official page geometry and ruler unchanged.
    pool={'unsplash':'Unsplash','ppr10k':'PPR10K','fivek_gold':'MIT-Adobe FiveK'}.get(record['pool'],record['pool'])
    lines=[r'\begingroup\fontsize{9}{10.5}\selectfont\setlength{\parskip}{0pt}\setlength{\parindent}{0pt}',
           r'\noindent\textbf{'+tex(title)+r'}\par',
           r'\textit{'+tex(pool)+r' training source; target-conditioned closed-form recovery.}\par',
           heading('1. Before / After / GT')]
    lines+=row(folder,['z0','z6','gt'],['Before','After','GT'])
    lines+=[heading('2. Iterative recovery')]
    lines+=row(folder,['z0','z1','z2','z3'],[r'Input ($z_0$)',r'$z_1$: Global',r'$z_2$: Hue',r'$z_3$: Shadows'])
    lines += [r'\vspace{\PhotoGap}']
    lines+=row(folder,['z4','z5','z6','gt'],[r'$z_4$: Midtones',r'$z_5$: Highlights',r'$z_6$: Spatial','GT'])
    lines += [r'\smallskip\textit{Instruction.} '+tex(record['instruction'])+r'\par']
    portrait=record['image_size'][1]>record['image_size'][0]
    if portrait:lines += [r'\clearpage']
    low_support=any(s.get('outside_kind')=='low_support' for s in record['picked'])
    lines += [heading('3. Local changes and spatial controls' if low_support else '3. Local changes and unchanged controls'),
              (r'Orange box: edit detail. Blue box: control ROI (see panel labels).\par' if low_support else
               r'Orange box: edit detail. Blue box: unchanged control ($\beta=0$).\par')]
    for i,chosen in enumerate(record['picked']):
        if i==1:lines += [r'\clearpage']
        lines+=local_panel(folder,record,chosen)
    lines += [r'\smallskip\noindent\hfill\includegraphics[width=.7\linewidth]{'+
              str((folder/'colorbar.pdf').relative_to(PAPER))+r'}\hfill\null\par',
              (r'\smallskip Low-support controls can change; exact preservation applies where $\beta=0$. Residuals show the actual adjacent-state RGB change on a shared scale.\par' if low_support else
               r'\smallskip The matched mask-out crops are unchanged. Residuals show the actual adjacent-state RGB change; the two steps share one color scale.\par')]
    if portrait:lines += [r'\clearpage']
    lines+=reasoning(folder,record,compact=not portrait)
    lines += [r'\endgroup']
    return '\n'.join(lines)+'\n'


PREAMBLE=r'''\documentclass{article}
\usepackage{iclr2027/iclr2027_conference,times}
\usepackage{graphicx,tikz,array,amsmath}
\newlength{\PhotoGap}\setlength{\PhotoGap}{3.75bp}
\newlength{\CellWidth}\newlength{\CropHeight}
\begin{document}
'''


def main():
    global ASSETS,DRAFTS
    parser=argparse.ArgumentParser()
    parser.add_argument('manifest',type=Path)
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--tag',help='Separate candidate output folder; does not replace the manuscript without --apply.')
    args=parser.parse_args()
    if args.tag:
        if not re.fullmatch(r'[a-z0-9_]+',args.tag):raise ValueError('Use a simple lowercase output tag')
        ASSETS=PAPER/'figures'/('appendix_'+args.tag);DRAFTS=PAPER/'drafts'/args.tag
    DRAFTS.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(args.manifest.read_text())
    if args.apply:
        kinds=[r.get('geometry_kind') for r in manifest]
        if len(kinds)!=4 or set(kinds)!={'semantic','radial','band','linear'}:
            raise ValueError('The author now requires one verified example per geometry family before manuscript insertion.')
    all_bodies=[]; audit=[]
    for i,item in enumerate(manifest,1):
        meta=json.loads(Path(item['provenance']).read_text())
        if meta.get('geometry_kind') and item.get('geometry_kind')!=meta['geometry_kind']:
            raise ValueError('Manifest geometry differs from the construction record')
        folder,record=prepare(i,meta=meta,output=ASSETS/f'case_{i:02d}',
                              require_spatial=item.get('require_spatial',True),subject_context=True,
                              subject_priority=item.get('subject_priority'),
                              allow_low_support=item.get('geometry_kind') in {'radial','band','linear'})
        record['geometry_kind']=item.get('geometry_kind')
        record['display_title']=item['title']
        (folder/'audit.json').write_text(json.dumps(record,indent=2)+'\n')
        content=body(folder,record,item['title'])
        (folder/'panels.tex').write_text(content)
        source=DRAFTS/f'case_{i:02d}.tex'
        source.write_text(PREAMBLE+content+r'\end{document}'+'\n')
        compile_pdf(source,DRAFTS/f'case_{i:02d}.pdf')
        pdf=DRAFTS/f'case_{i:02d}.pdf'
        info=subprocess.check_output(['pdfinfo',str(pdf)],text=True)
        pages=int(next(line.split(':')[1] for line in info.splitlines() if line.startswith('Pages:')))
        subprocess.run(['pdftoppm','-r','100','-png',str(pdf),str(DRAFTS/f'case_{i:02d}_page')],check=True)
        audit.append(dict(case=i,pages=pages,source=record['source_id'],steps=[x['step'] for x in record['picked']],
                          outside_max_change=max(x['outside_max_change'] for x in record['picked'])))
        all_bodies.append(content)
        print(json.dumps(audit[-1]),flush=True)
    combined=DRAFTS/'local_four_part_preview.tex'
    combined.write_text(PREAMBLE+'\n\\clearpage\n'.join(all_bodies)+r'\end{document}'+'\n')
    compile_pdf(combined,DRAFTS/'local_four_part_preview.pdf')
    (DRAFTS/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    if args.apply:
        lines=[r'\clearpage\section{Local Editing Trajectories}',r'\label{app:maskgrade_examples}',
               r'\newlength{\PhotoGap}\setlength{\PhotoGap}{3.75bp}',
               r'\newlength{\CellWidth}\newlength{\CropHeight}']
        for i in range(1,len(manifest)+1):
            if i>1:lines += [r'\clearpage']
            lines += [r'\input{'+str(ASSETS.relative_to(PAPER))+r'/case_'+f'{i:02d}'+r'/panels}']
        (PAPER/'sections/appendix/local_four_part_pages.tex').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
