"""Four-image inference-time text ablation; no model retraining or mask changes."""
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from tools.build_appendix_paper_samples import PAPER,tex
from tools.build_appendix_local_detail import compile_pdf

WORK=Path('/home/bc/data/runs/paper_selected_four_pix3200_20260920')
ASSETS=PAPER/'figures/selected_four_reasoning_ablation'


def main():
    rows=json.loads((WORK/'inputs.json').read_text())['records']
    baseline=json.loads((WORK/'ours.json').read_text());trial=json.loads((WORK/'ours_no_reasoning.json').read_text())
    assert baseline['checkpoint']['sha256']==trial['checkpoint']['sha256']
    assert len(trial['rows'])==4 and all(r['ok'] and r['cot_tokens']==6 for r in trial['rows'])
    out=PAPER/'drafts/selected_four_no_reasoning';out.mkdir(parents=True,exist_ok=True)
    lines=[r'\documentclass{article}\usepackage{iclr2027/iclr2027_conference,times,graphicx,booktabs,tabularx}',
           r'\begin{document}\begin{figure}[!ht]\centering\begingroup',
           r'\fontsize{8}{9}\selectfont\setlength{\tabcolsep}{1.5pt}\renewcommand{\arraystretch}{0}',
           r'\renewcommand{\tabularxcolumn}[1]{m{#1}}',r'\begin{tabularx}{\linewidth}{@{}*{4}{>{\centering\arraybackslash}X}@{}}',
           r'Input & With reasoning & Without stage reasoning & Reference\\']
    summary=[]
    for r in rows:
        folder=Path(r['folder']);gt=np.asarray(Image.open(folder/'reference.png'),dtype=np.float32)/255
        assets=ASSETS/r['id'];assets.mkdir(parents=True,exist_ok=True)
        for name in ['input','ours','ours_no_reasoning','reference']:
            shutil.copyfile(folder/f'{name}.png',assets/f'{name}.png')
        result=dict(id=r['id'])
        for name in ['ours','ours_no_reasoning']:
            image=np.asarray(Image.open(folder/f'{name}.png'),dtype=np.float32)/255
            assert image.shape==gt.shape
            result[name]=dict(L1_x100=float(np.abs(image-gt).mean())*100,
                              PSNR=float(-10*np.log10(max(float(np.square(image-gt).mean()),1e-12))))
        summary.append(result)
        lines += [r'\addlinespace[2pt]',r'\multicolumn{4}{@{}l@{}}{\itshape\strut '+tex(r['instruction_short'])+r'}\\',
                  ' & '.join(r'\includegraphics[width=\linewidth]{'+str((assets/f'{name}.png').relative_to(PAPER))+'}'
                             for name in ['input','ours','ours_no_reasoning','reference'])+r'\\']
    lines += [r'\end{tabularx}\endgroup\setlength{\abovecaptionskip}{4pt}',
              r'\caption{PIX@3200 with and without generated stage reasoning. Input, instruction, checkpoint, and predicted subject mask are fixed; stage/readout markers remain.}',
              r'\end{figure}\end{document}']
    source=out/'comparison.tex';source.write_text('\n'.join(lines)+'\n');compile_pdf(source,out/'comparison.pdf')
    report=dict(checkpoint=trial['checkpoint'],protocol='Remove six-stage reasoning text at inference; keep instruction, stage/readout tokens, and the same input-predicted subject masks. Photometric supports follow each rollout current state.',
                scope='Four author-selected training examples; not a retrained or full-benchmark comparison.',rows=summary)
    report['mean']={name:{metric:float(np.mean([r[name][metric] for r in summary])) for metric in ['L1_x100','PSNR']}
                    for name in ['ours','ours_no_reasoning']}
    (WORK/'reasoning_ablation.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
