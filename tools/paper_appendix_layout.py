"""Select appendix assets and compose their labels/layout only in LaTeX."""
import argparse
import json
from pathlib import Path
import shutil

from PIL import Image,ImageDraw,ImageFont

from tools.paper_appendix_qualitative import OUT,WORK,dump,sha

PAPER=Path('/home/bc/VeraRetouch/EPR/ICLR2027')


def preview():
    # Internal contact sheets are selection aids, never manuscript assets.
    for kind,filename in [('chain','chain_recovery_results.json'),('infer','style_inference_results.json')]:
        rows=json.loads((WORK/filename).read_text())
        if kind=='infer':
            rows=sorted([r for r in rows if r['identity_l1']>4 and r['l1']<.8*r['identity_l1']],key=lambda r:r['l1'])[:16]
        cols=2;cw,ch=630,215
        sheet=Image.new('RGB',(cols*cw,((len(rows)+cols-1)//cols)*ch),'white')
        draw=ImageDraw.Draw(sheet);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
        for i,r in enumerate(rows):
            folder=Path(r['folder']);names=['recovery_0','reference','recovery_6'] if kind=='chain' else ['input','target','prediction']
            x=(i%cols)*cw;y=(i//cols)*ch
            label=f"{folder.name}  error={r.get('final_l1',r.get('l1')):.3f}  initial={r.get('initial_l1',r.get('identity_l1')):.3f}"
            draw.text((x+4,y+2),label,font=font,fill='black')
            for c,name in enumerate(names):
                im=Image.open(folder/(name+'.png')).convert('RGB');im.thumbnail((200,175))
                sheet.paste(im,(x+c*210+(200-im.width)//2,y+28))
        sheet.save(WORK/(kind+'_selection_preview.png'))


def tex(s):
    table={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_',
           '{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}',
           '<':r'\textless{}','>':r'\textgreater{}'}
    return ''.join(table.get(c,c) for c in s).replace('—',', ').replace('–','--')


def export(chain_names,infer_names):
    chains=json.loads((WORK/'chain_recovery_results.json').read_text())
    models=json.loads((WORK/'style_inference_results.json').read_text())
    selected=[]
    def install(rows,names,kind):
        picked=[]
        for index,name in enumerate(names,1):
            r=next(r for r in rows if Path(r['folder']).name==name)
            folder=OUT/f'{kind}_{index:02d}';folder.mkdir(exist_ok=True)
            files=['reference']+[f'recovery_{j}' for j in range(7)]+[f'mask_{j}' for j in range(1,7)] if kind=='recovery' else ['input','target','prediction']
            provenance=dict(r);provenance['output_files']={}
            for f in files:
                target=folder/(f+'.png');shutil.copyfile(Path(r['folder'])/(f+'.png'),target)
                provenance['output_files'][f]=dict(file=str(target.relative_to(PAPER)),sha256=sha(target))
            # The internal trace stores full origins. Public PDF contains case IDs only.
            dump(folder/'provenance.json',provenance);picked.append((r,folder));selected.append(provenance)
        return picked
    selected_chains=install(chains,chain_names,'recovery')
    selected_models=install(models,infer_names,'inference')
    dump(OUT/'selection_manifest.json',dict(
        chain_candidates=16,inference_candidates=48,
        policy='Representative successful training examples selected for visible edit, low error, distinct scenes. Not random test-set performance.',
        selected=selected))
    prefix='figures/appendix_qualitative_20260919'
    lines=[r'\section{MaskGrade Language and Annotation}',r'\label{app:maskgrade_annotation}',
      'We document the training-language distribution, the production annotation prompts, and representative records used to supervise editing.',
      r'\subsection{Instruction Vocabulary}',
      'We count one instruction per record over the 73,853 entries in the color-language training manifest.',
      'The deterministic instruction selection gives 51,976 long, 14,574 medium, and 7,303 short requests.',
      'English words are lowercased and counted after removing general stopwords and task boilerplate; fixed scaffolds and reasoning paragraphs are excluded.',
      'The median instruction length is 90 words, with 10th and 90th percentiles of 21 and 99.',
      r'Figure~\ref{fig:maskgrade_vocabulary} visualizes the vocabulary; word size represents frequency, not scene-category coverage.',
      r'\begin{figure}[!htbp]',r'\centering',
      r'\begin{minipage}[c]{.57\linewidth}\centering',
      r'\includegraphics[width=\linewidth]{'+prefix+'/instruction_wordcloud.png}',r'\end{minipage}\hfill',
      r'\begin{minipage}[c]{.40\linewidth}\centering',
      r'\includegraphics[width=\linewidth]{'+prefix+'/instruction_frequency.pdf}',r'\end{minipage}',
      r'\caption{Color, tone, and region vocabulary in MaskGrade training instructions. The word cloud shows 70 frequent content words; the frequency chart reports the 12 most frequent tokens under the same counting protocol.}',
      r'\label{fig:maskgrade_vocabulary}',r'\end{figure}',
      r'\subsection{Annotation Inputs and Output Contract}',
      'The production annotator is GPT-5.6 Terra.',
      'It receives three images in order: the current image, its target appearance, and a two-row, seven-column stage breakdown, together with a facts sheet extracted from the construction record.',
      'In the archived assets, the first two inputs correspond to the after and before images, respectively.',
      'The annotator returns six ordered records with observation, mask, and adjustment fields, followed by long (60--110 words), medium (20--35 words), and short (5--12 words) user instructions.',
      'The parser checks required fields, nonempty text, stage count, and the English-language format.',
      'A separate numerical reconciliation records mask, band, direction, and magnitude discrepancies for inspection; it is a diagnostic report rather than a mandatory semantic acceptance filter.',
      'The student receives the current image and a user request; the target and breakdown images belong to annotation only.',
      r'\subsection{Production Prompt}',
      'The following system and task prompts are transcribed from the frozen annotation configuration.',
      r'\noindent\textbf{System prompt.}',r'\begin{quote}\small']
    lines.append(tex((OUT/'system_prompt.txt').read_text()).strip());lines+=[r'\end{quote}',r'\noindent\textbf{Task prompt.}',r'\begin{quote}\small',tex((OUT/'user_head_prompt.txt').read_text()).strip(),r'\end{quote}',r'\FloatBarrier']
    lines += [r'\section{MaskGrade Training Examples and Stepwise Recovery}',r'\label{app:maskgrade_examples}',
       'The following selected training examples originate from Unsplash according to the frozen source manifest.',
       'They illustrate data construction and recovery, rather than held-out model performance.',
       'For each recorded transition, we fit an absolute color code using its recorded soft support, then execute these fixed codes sequentially from the degraded endpoint.',
       'The shown intermediate images are outputs of this recovery rollout, not copies of the target states.',
       'Supports already contain the recorded strength; it is not applied a second time.']
    labels={'global':'Global','hue':'Hue','lum_shadow':'Shadows','lum_mid':'Midtones','lum_high':'Highlights','geom':'Spatial'}
    for i,(r,folder) in enumerate(selected_chains,1):
        p=str(folder.relative_to(PAPER))
        lines += [r'\subsection{Recovery Example '+str(i)+'}',
           f'This six-stage training example reduces pixel MAE $\\times100$ from {r["initial_l1"]:.2f} to {r["final_l1"]:.2f} under target-conditioned code recovery.',
           r'\begin{figure}[!htbp]',r'\centering',r'\setlength{\tabcolsep}{2pt}',r'\begin{tabular}{cccc}',
           r'Input & After '+labels[r['stage_order'][0]]+' & After '+labels[r['stage_order'][1]]+' & After '+labels[r['stage_order'][2]]+r' \\',
           ' & '.join(r'\includegraphics[width=.235\linewidth]{'+p+f'/recovery_{j}.png'+'}' for j in range(4))+r' \\[4pt]',
           'After '+labels[r['stage_order'][3]]+' & After '+labels[r['stage_order'][4]]+' & After '+labels[r['stage_order'][5]]+r' & Reference \\',
           ' & '.join([r'\includegraphics[width=.235\linewidth]{'+p+f'/recovery_{j}.png'+'}' for j in range(4,7)]+[r'\includegraphics[width=.235\linewidth]{'+p+'/reference.png}'])+r' \\',
           r'\end{tabular}',r'\caption{Sequential recovery on Unsplash-derived training example '+str(i)+'. Each image is a separate execution state; the final column gives the reference. Color codes are fitted from recorded state pairs.}',r'\end{figure}',
           r'\noindent\textbf{Stage supports.}',r'\begin{center}',r'\setlength{\tabcolsep}{2pt}',r'\begin{tabular}{cccccc}',
           ' & '.join(labels[k] for k in r['stage_order'])+r' \\',
           ' & '.join(r'\includegraphics[width=.153\linewidth]{'+p+f'/mask_{j}.png'+'}' for j in range(1,7))+r' \\',r'\end{tabular}',r'\end{center}',
           'Grayscale encodes the original soft execution weight on a fixed $[0,1]$ scale.',r'\FloatBarrier']
    # Show one actual language record, separate from recovered-operator measurements.
    r=selected_chains[0][0];a=r['annotation']
    lines += [r'\subsection{A Complete Annotation Record}',
      'This archived annotation accompanies the first recovery example. Its wording is reproduced as a dataset record, not as a new prediction or a numerical description of the fitted codes.',
      r'\noindent\textbf{Short request.} '+tex(a['instruction_short']),
      r'\par\noindent\textbf{Medium request.} '+tex(a['instruction_medium']),
      r'\par\noindent\textbf{Long request.} '+tex(a['instruction_long'])]
    for c in a['cot']:
        lines += [r'\paragraph{Move '+str(c['step'])+'}',
          r'\textbf{Observation:} '+tex(c['observation'])+r'\par',
          r'\textbf{Mask:} '+tex(c['mask'])+r'\par',
          r'\textbf{Adjustment:} '+tex(c['adjustment'])+r'\par']
    lines += [r'\FloatBarrier',r'\section{Training-Domain Instruction-Guided Retouching}',r'\label{app:training_qualitative}',
      'We show selected successful training-domain examples from Unsplash-derived style records using the continuous-readout checkpoint at update 800.',
      'The examples were selected from 48 fixed candidate images for visible editing changes, reconstruction quality, and distinct scene content.',
      'Each model prediction is newly rendered from the input and the cached base-SFT-generated editing description; the target image is used only for comparison.',
      'These illustrations document in-domain behavior and are separate from the benchmark results.']
    for i,(r,folder) in enumerate(selected_models,1):
        p=str(folder.relative_to(PAPER))
        lines += [r'\subsection{Inference Example '+str(i)+'}',r'\noindent\textbf{Request.} '+tex(r['instruction']),
           r'\begin{figure}[!htbp]',r'\centering',r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{ccc}',r'Input & Target style & Model output \\',
           ' & '.join(r'\includegraphics[width=.32\linewidth]{'+p+'/'+name+'.png}' for name in ['input','target','prediction'])+r' \\',r'\end{tabular}',
           r'\caption{Selected Unsplash training example '+str(i)+f'. Model-output MAE $\\times100$ is {r["l1"]:.2f}, compared with {r["identity_l1"]:.2f} for the unchanged input.'+'}',r'\end{figure}',
           r'\noindent\textbf{Cached model-generated editing description.}',r'\begin{quote}\small',tex(r['reasoning']),r'\end{quote}',r'\FloatBarrier']
    text='\n'.join(lines)+'\n'
    # Enforce narrative order: finish each visual before its following prose.
    text=text.replace(r'\end{figure}',r'\end{figure}'+'\n'+r'\FloatBarrier')
    text=text.replace(r'\begin{tabular}{ccc}',r'\begin{tabular}{@{}ccc@{}}')
    text=text.replace('Cached model-generated editing description.','Model-generated editing description (cached).')
    for i,name in enumerate(['Backlit Portrait','Woodland Lake','Poolside Color'],1):
        text=text.replace('Recovery Example '+str(i)+'}',name+'}')
        text=text.replace(r'\subsection{'+name+'}',r'\Needspace{.70\textheight}'+'\n'+r'\subsection{'+name+'}')
    for i,name in enumerate(['Warm Fur Tones','Cool Waterfall Contrast','Warm Interior'],1):
        text=text.replace('Inference Example '+str(i)+'}',name+'}')
        text=text.replace(r'\subsection{'+name+'}',r'\Needspace{.52\textheight}'+'\n'+r'\subsection{'+name+'}')
    (WORK/'appendix_qualitative_generated.tex').write_text(text)
    dump(WORK/'selected_metrics.json',dict(recovery=[dict(key=r['key'],initial=r['initial_l1'],final=r['final_l1']) for r,f in selected_chains],inference=[dict(key=r['key'],l1=r['l1'],identity=r['identity_l1']) for r,f in selected_models]))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['preview','export'])
    p.add_argument('--chains',default='');p.add_argument('--inference',default='')
    a=p.parse_args()
    if a.mode=='preview':preview()
    else:export(a.chains.split(','),a.inference.split(','))
