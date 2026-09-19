"""Dense, text-width in-domain photo galleries with matched zooms and changes."""
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, colors, cm

from tools.build_appendix_paper_samples import PAPER, tex
from tools.build_appendix_local_detail import compile_pdf, crop_box, detail_energy, photo

WORK = Path('/home/bc/data/runs/paper_appendix_expansion_20260919')
OLD = Path('/home/bc/data/runs/paper_appendix_qualitative_20260919')
DEST = PAPER/'figures/appendix_domain_gallery'
GROUPS = [
    ('unsplash', 'Unsplash', OLD, ['infer_18','infer_44','infer_34','infer_23','infer_35','infer_20']),
    ('fivek', 'MIT-Adobe FiveK', WORK, ['fivek_gold_02','fivek_gold_18','fivek_gold_16','fivek_gold_19','fivek_gold_17','fivek_gold_10']),
    ('ppr10k', 'PPR10K', WORK, ['ppr10k_27','ppr10k_39','ppr10k_30','ppr10k_25','ppr10k_37','ppr10k_33'])]


def prepare_case(source, folder):
    folder.mkdir(parents=True, exist_ok=True)
    r = json.loads((source/'provenance.json').read_text())
    for name in ['input','prediction','target']:
        shutil.copyfile(source/(name+'.png'), folder/(name+'.png'))
    before = np.asarray(Image.open(source/'input.png'), dtype=np.float32)/255
    after = np.asarray(Image.open(source/'prediction.png'), dtype=np.float32)/255
    change = np.abs(after-before).mean(-1)*100
    box = crop_box(detail_energy(change, before))
    for name, out in [('input','before_zoom'), ('prediction','after_zoom')]:
        Image.open(source/(name+'.png')).crop(box).save(folder/(out+'.png'))
    r.update(crop_xyxy=box, source_size=[before.shape[1],before.shape[0]],
             residual_definition='100*mean_RGB(abs(output-input)) on decoded archived uint8 model renders',
             crop_policy='texture-weighted action energy, identical coordinates in before/after')
    (folder/'provenance.json').write_text(json.dumps(r, indent=2)+'\n')
    return r, change


def main():
    manifest=[]
    for ident, title, root, names in GROUPS:
        group=DEST/ident; group.mkdir(parents=True, exist_ok=True)
        cases=[prepare_case(root/name, group/f'case_{i:02d}') for i,name in enumerate(names,1)]
        maximum=max(1.,float(np.ceil(max(delta.max() for r,delta in cases))))
        cmap=matplotlib.colormaps['inferno']
        for i,(r,delta) in enumerate(cases,1):
            Image.fromarray((cmap(delta/maximum)[...,:3]*255).astype(np.uint8)).save(group/f'case_{i:02d}/residual.png')
        fig,ax=plt.subplots(figsize=(3.7,.38));fig.subplots_adjust(left=.025,right=.97,top=.9,bottom=.6)
        bar=fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(0,maximum),cmap=cmap),cax=ax,orientation='horizontal')
        bar.set_ticks([0,maximum/2,maximum]);bar.ax.tick_params(labelsize=8,length=2)
        bar.set_label('Actual change (mean absolute RGB difference x100)',fontsize=8,labelpad=1)
        fig.savefig(group/'colorbar.pdf',bbox_inches='tight',pad_inches=.02);plt.close(fig)
        lines=[r'\documentclass[10pt,border=0pt]{standalone}',r'\usepackage{times,graphicx,tikz,array,xcolor}',
               r'\begin{document}\begin{minipage}{5.5in}\fontsize{8.5}{10}\selectfont\setlength{\parindent}{0pt}',
               r'\textbf{In-domain style editing / '+title+r' source photographs}\par\medskip',
               r'\setlength{\tabcolsep}{1pt}\begin{tabular}{@{}*{5}{>{\centering\arraybackslash}p{.192\linewidth}}@{}}',
               r'Input & Model output & Input (zoom) & Output (zoom) & Actual residual\\']
        for i,(r,delta) in enumerate(cases,1):
            folder=group/f'case_{i:02d}'
            request=r['instruction']
            if len(request.split())>36:
                request=' '.join(request.split()[:36]).rstrip('.,;:')+'...'
            lines += [r'\multicolumn{5}{@{}p{\linewidth}@{}}{\vspace{4pt}\textbf{'+str(i)+'.} '+tex(request)+r'}\\[2pt]']
            panels=[photo(folder,'input',1,.80,r['crop_xyxy'],r['source_size']),
                    photo(folder,'prediction',1,.80,r['crop_xyxy'],r['source_size']),
                    photo(folder,'before_zoom',1,.80),photo(folder,'after_zoom',1,.80),photo(folder,'residual',1,.80)]
            lines += [' & '.join(panels)+r'\\']
        lines += [r'\end{tabular}\par\smallskip\centering',r'\includegraphics[width=.72\linewidth]{'+str((group/'colorbar.pdf').relative_to(PAPER))+'}',r'\end{minipage}\end{document}']
        source=group/'gallery.tex'; source.write_text('\n'.join(lines)+'\n')
        compile_pdf(source,group/'gallery.pdf')
        manifest.append(dict(group=ident,source_dataset=title,n=6,cases=[r for r,d in cases],
                             scale=[0,maximum],figure=str((group/'gallery.pdf').relative_to(PAPER)),
                             target='synthetic LUT style; not expert benchmark evaluation'))
    (DEST/'selection.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps([dict(group=r['group'],n=r['n']) for r in manifest]))


if __name__=='__main__':
    main()
