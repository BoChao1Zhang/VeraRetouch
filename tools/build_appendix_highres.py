"""Native 4032x2268 result with unchanged-grid crops and measured change."""
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, colors, cm

from tools.build_appendix_paper_samples import PAPER, tex
from tools.build_appendix_local_detail import compile_pdf, photo

WORK=Path('/home/bc/data/runs/paper_appendix_expansion_20260919/highres')
DEST=PAPER/'figures/appendix_highres'


def main():
    DEST.mkdir(exist_ok=True)
    formal=WORK/'output/results.json'
    results=json.loads((formal if formal.exists() else WORK/'preview_output/results.json').read_text())
    r=results['records'][0]
    source=Image.open(r['image']).convert('RGB')
    output_root=WORK/('output' if formal.exists() else 'preview_output')
    output=Image.open(output_root/'blanca_native.png').convert('RGB')
    if source.size!=output.size or source.size!=(4032,2268):
        raise ValueError('Native geometry mismatch')
    source.save(DEST/'input.png'); output.save(DEST/'output.png')
    # A shared 768x432 native-pixel rectangle spanning the distant valley,
    # rock detail and treeline, selected geometrically, not from a baseline.
    box=[900,850,1668,1282]
    source.crop(box).save(DEST/'input_zoom.png'); output.crop(box).save(DEST/'output_zoom.png')
    delta=np.abs(np.asarray(output,dtype=np.float32)-np.asarray(source,dtype=np.float32)).mean(-1)/255*100
    maximum=max(1.,float(np.ceil(delta.max())))
    Image.fromarray((matplotlib.colormaps['inferno'](delta/maximum)[...,:3]*255).astype(np.uint8)).save(DEST/'residual.png')
    for name, frame in [('input',source),('output',output)]:
        display=frame.copy();display.thumbnail((1800,1800),Image.Resampling.LANCZOS)
        display.save(DEST/(name+'_display.png'))
    fig,ax=plt.subplots(figsize=(3.7,.38));fig.subplots_adjust(left=.025,right=.97,top=.9,bottom=.6)
    bar=fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(0,maximum),cmap='inferno'),cax=ax,orientation='horizontal')
    bar.set_ticks([0,maximum/2,maximum]);bar.ax.tick_params(labelsize=8,length=2)
    bar.set_label('Actual change (mean absolute RGB difference x100)',fontsize=8,labelpad=1)
    fig.savefig(DEST/'colorbar.pdf',bbox_inches='tight',pad_inches=.02);plt.close(fig)
    source_facts=json.loads((WORK/'source.json').read_text())
    manifest=dict(source=source_facts, checkpoint=results['config']['checkpoint'],
                  native_size=list(source.size),crop_xyxy=box,request=r['instruction'],
                  rationale=r['rationale'],generation='fresh, not cached',
                  formal_timing_available=formal.exists(),residual_scale=[0,maximum],
                  display='full-frame panels reduced to 1800 pixels for print; native 768x432 crops unchanged')
    (DEST/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
    lines=[r'\documentclass[10pt,border=0pt]{standalone}',r'\usepackage{times,graphicx,tikz,array}',
       r'\begin{document}\begin{minipage}{5.5in}\small\setlength{\parindent}{0pt}',
       r'\textbf{Native-resolution retouching: $4032\times2268$}\par\smallskip',
       r'\textbf{Request.} '+tex(r['instruction'])+r'\par\smallskip\centering',
       photo(DEST,'input',1,2.32,box,source.size),r'\par Input ($4032\times2268$)\par\smallskip',
       photo(DEST,'output',1,2.32,box,source.size),r'\par Model output ($4032\times2268$)\par\medskip',
       r'\setlength{\tabcolsep}{2pt}\begin{tabular}{@{}*{3}{>{\centering\arraybackslash}p{.32\linewidth}}@{}}',
       r'Input (native crop) & Output (same crop) & Actual residual\\',
       ' & '.join(photo(DEST,name,1,.97) for name in ['input_zoom','output_zoom','residual'])+r'\\',
       r'\end{tabular}\par\smallskip',r'\includegraphics[width=.72\linewidth]{figures/appendix_highres/colorbar.pdf}']
    if formal.exists():
        median=float(np.median([x['seconds'] for x in r['repetitions']]))
        render=float(np.median([x['render_seconds'] for x in r['repetitions']]))
        lines += [r'\par\smallskip End-to-end: '+f'{median:.2f} s'+r'\quad Native-grid execution: '+f'{render:.3f} s'+'.']
    lines += [r'\end{minipage}\end{document}']
    source=DEST/'highres.tex';source.write_text('\n'.join(lines)+'\n');compile_pdf(source,DEST/'highres.pdf')


if __name__=='__main__':
    main()
