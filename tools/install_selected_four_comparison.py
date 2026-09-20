"""Compose the four fresh model comparisons; never fill missing methods."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from PIL import Image
from tools.build_appendix_paper_samples import PAPER,tex
from tools.build_appendix_local_detail import compile_pdf

WORK=Path('/home/bc/data/runs/paper_selected_four_pix3200_20260920')
OUT=PAPER/'figures/selected_four_pix3200'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--apply',action='store_true');a=p.parse_args()
    rows=json.loads((WORK/'inputs.json').read_text())['records']
    ours=json.loads((WORK/'ours.json').read_text());assert ours['checkpoint']['step']==3200 and 'PIX_step3200' in ours['checkpoint']['path']
    assert ours['target_access'] is False and len(ours['rows'])==4
    for row in ours['rows']:assert row['ok']
    vera=json.loads((WORK/'veraretouch.json').read_text());assert len(vera['rows'])==4 and all(r['ok'] for r in vera['rows'])
    has_jarvis=all((Path(r['folder'])/'jarvisevo.png').is_file() for r in rows)
    if a.apply and not has_jarvis:raise RuntimeError('JarvisEvo official rendering is incomplete; keeping existing manuscript figures.')
    methods=['input','jarvisevo','monetgpt','veraretouch','ours'] if has_jarvis else ['input','monetgpt','veraretouch','ours','reference']
    titles={'input':'Input','jarvisevo':'JarvisEvo','monetgpt':'MonetGPT','veraretouch':'VeraRetouch','ours':'Ours','reference':'Reference'}
    provenance=[]
    lines=[r'\begin{figure}[!ht]',r'\centering\begingroup\small',r'\setlength{\tabcolsep}{1.5pt}',
           r'\renewcommand{\tabularxcolumn}[1]{m{#1}}',r'\renewcommand{\arraystretch}{.85}',
           r'\begin{tabularx}{\linewidth}{@{}*{5}{>{\centering\arraybackslash}X}@{}}',
           ' & '.join(r'\textbf{'+titles[m]+'}' for m in methods)+r'\\']
    for r in rows:
        src=Path(r['folder']);dst=OUT/r['id'];dst.mkdir(parents=True,exist_ok=True)
        assert sha(src/'input.png')==r['input_sha256']
        files={}
        for m in methods:
            file=src/f'{m}.png'
            with Image.open(file) as image:
                image.load();assert list(image.size)==r['size'],(r['id'],m,image.size)
            shutil.copyfile(file,dst/f'{m}.png');files[m]=sha(file)
        lines += [r'\addlinespace[2pt]',r'\multicolumn{5}{@{}l@{}}{\fontsize{8}{9}\selectfont\itshape\strut '+tex(r['instruction_short'])+r'}\\[-1pt]',
                  ' & '.join(r'\includegraphics[width=\linewidth]{'+str((dst/f'{m}.png').relative_to(PAPER))+'}' for m in methods)+r'\\']
        provenance.append(dict(id=r['id'],source_id=r['source_id'],input_sha256=r['input_sha256'],
                               instruction=r['instruction'],instruction_short=r['instruction_short'],output_hashes=files))
    lines += [r'\end{tabularx}\endgroup',r'\caption{Instruction-guided retouching on four author-selected training-domain images. Ours uses PIX@3200 with fresh reasoning and predicted subject support. All methods receive the same input and full instruction; short instructions are shown for display.}',r'\label{fig:qualitative}',r'\end{figure}']
    content='\n'.join(lines)+'\n'
    (OUT/'comparison.tex').write_text(content)
    (OUT/'provenance.json').write_text(json.dumps(dict(checkpoint=ours['checkpoint'],methods=methods,rows=provenance),indent=2)+'\n')
    draft=PAPER/'drafts/selected_four_pix3200';draft.mkdir(parents=True,exist_ok=True)
    source=draft/'comparison.tex'
    source.write_text(r'\documentclass{article}\usepackage{iclr2027/iclr2027_conference,times,graphicx,booktabs,tabularx}\begin{document}'+'\n'+content+r'\end{document}'+'\n')
    compile_pdf(source,draft/'comparison.pdf')
    if a.apply:(PAPER/'figures/qualitative_layout.tex').write_text(content)
    print(json.dumps(dict(methods=methods,applied=a.apply,pdf=str(draft/'comparison.pdf'))),flush=True)


if __name__=='__main__':main()
