"""Compose traceable sample figures in LaTeX at the actual ICLR text width.

No photo pixels are synthesized or modified. Panels use separately archived
images. The generated PDFs contain searchable text and no page furniture;
the ICLR document supplies its own caption, page number, and review ruler.
"""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

PAPER = Path('/home/bc/VeraRetouch/EPR/ICLR2027')
ASSETS = PAPER / 'figures/appendix_qualitative_20260919'
WORK = Path('/home/bc/data/runs/paper_appendix_qualitative_20260919')
OUT = PAPER / 'figures/appendix_paper_samples'
PREFIX = 'figures/appendix_qualitative_20260919'
LABELS = {'global': 'Global', 'hue': 'Hue', 'lum_shadow': 'Shadows',
          'lum_mid': 'Midtones', 'lum_high': 'Highlights', 'geom': 'Spatial'}
MODEL_CASES = [('inference_01', 'Warm fur tones'),
               ('inference_02', 'Cool waterfall contrast'),
               ('inference_03', 'Warm interior'),
               ('inference_04', 'Autumn foliage'),
               ('inference_05', 'Coastal light'),
               ('inference_06', 'Urban tones'),
               ('inference_07', 'Sunset silhouettes')]
RECOVERY_CASES = [('recovery_01', 'Backlit portrait'),
                  ('recovery_02', 'Woodland lake'),
                  ('recovery_03', 'Poolside color')]


def tex(value):
    escape = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
              '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
              '~': r'\textasciitilde{}', '^': r'\textasciicircum{}',
              '<': r'\textless{}', '>': r'\textgreater{}'}
    value = str(value).replace('—', '---').replace('–', '--')
    value = value.replace('’', "'").replace('“', '``').replace('”', "''")
    return ''.join(escape.get(c, c) for c in value)


def install_assets():
    for number, name in [(4, 'infer_23'), (5, 'infer_06'), (6, 'infer_35'), (7, 'infer_20')]:
        folder = ASSETS / f'inference_{number:02d}'
        folder.mkdir(exist_ok=True)
        record = json.loads((WORK / name / 'provenance.json').read_text())
        record['output_files'] = {}
        for filename in ['input.png', 'prediction.png', 'target.png']:
            dest = folder / filename
            shutil.copyfile(WORK / name / filename, dest)
            record['output_files'][dest.stem] = {
                'file': str(dest.relative_to(PAPER)),
                'sha256': hashlib.sha256(dest.read_bytes()).hexdigest()}
        (folder / 'provenance.json').write_text(json.dumps(record, indent=2) + '\n')
    for ident, _ in RECOVERY_CASES:
        folder = ASSETS / ident
        record = json.loads((folder / 'provenance.json').read_text())
        for step in range(1, 7):
            shutil.copyfile(Path(record['folder']) / f'recorded_{step}.png',
                            folder / f'recorded_{step}.png')
    selection = json.loads((ASSETS / 'selection_manifest.json').read_text())
    selection['selected'] = [json.loads((ASSETS / ident / 'provenance.json').read_text())
                             for ident, _ in RECOVERY_CASES + MODEL_CASES]
    selection['displayed_recovery_cases'] = len(RECOVERY_CASES)
    selection['displayed_inference_cases'] = len(MODEL_CASES)
    (ASSETS / 'selection_manifest.json').write_text(json.dumps(selection, indent=2) + '\n')


def panel(ident, name, label, width, height):
    return (rf'\begin{{minipage}}[t]{{{width}\linewidth}}\centering' + '\n'
            + rf'\includegraphics[width=\linewidth,height={height}in,keepaspectratio]{{{PREFIX}/{ident}/{name}.png}}\par' + '\n'
            + r'{\small ' + label + r'}\end{minipage}')


def row(ident, names, labels, width, height):
    return '\n\\hfill\n'.join(panel(ident, name, label, width, height)
                               for name, label in zip(names, labels)) + '\n\\par\\medskip\n'


def heading(text):
    return r'\noindent{\bfseries ' + tex(text) + r'}\par\smallskip' + '\n'


def field(label, value):
    return r'\noindent\textcolor{samplegreen}{\textbf{' + label + '}} ' + tex(value) + '\n\\par\\smallskip\n'


def document(content):
    # Figure canvas dimensions, not a modification to the manuscript template.
    return (r'''\documentclass[10pt,border=0pt]{standalone}
\usepackage{times,graphicx,xcolor}
\definecolor{samplegreen}{RGB}{28,88,76}
\begin{document}
\begin{minipage}{5.5in}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0pt}
\small
''' + content + '\n' + r'''\end{minipage}
\end{document}
''')


def recovery(ident, title, record):
    body = heading(title)
    body += row(ident, ['recovery_0', 'reference', 'recovery_6'],
                ['Degraded input', 'Reference', 'Recovered output'], '.323', 1.8)
    body += heading('Recorded degradation: reference to input')
    body += row(ident, ['reference'] + [f'recorded_{j}' for j in range(5, 0, -1)] + ['recovery_0'],
                [r'$d_0$', r'$d_1$', r'$d_2$', r'$d_3$', r'$d_4$', r'$d_5$', r'$d_6$'], '.138', .81)
    body += heading('Closed-form recovery: input to restored output')
    body += row(ident, [f'recovery_{j}' for j in range(7)],
                [r'$z_0$', r'$z_1$', r'$z_2$', r'$z_3$', r'$z_4$', r'$z_5$', r'$z_6$'], '.138', .81)
    body += heading('Supports in recovery order')
    body += row(ident, [f'mask_{j}' for j in range(1, 7)],
                [f'{j+1}. {LABELS[k]}' for j, k in enumerate(record['stage_order'])], '.161', .63)
    body += field('Request.', record['annotation']['instruction_medium'])
    body += (r'\noindent MAE $\times100$: ' + f'{record["initial_l1"]:.2f}'
             + r' (input) $\rightarrow$ ' + f'{record["final_l1"]:.2f}' + ' (recovered output).\n')
    return body


def annotation(ident, title, record, part):
    body = heading(f'{title}: moves {part*3+1}--{part*3+3}')
    for c in record['annotation']['cot'][part*3:part*3+3]:
        step = c['step']
        body += r'\noindent\rule{\linewidth}{.3pt}\par\smallskip' + '\n'
        body += heading(f'Move {step} / {LABELS[record["stage_order"][step-1]]}')
        body += r'\begin{minipage}[t]{.80\linewidth}\vspace{0pt}' + '\n'
        for label, key in [('Observation.', 'observation'), ('Mask.', 'mask'), ('Adjustment.', 'adjustment')]:
            body += field(label, c[key])
        body += r'\end{minipage}\hfill\begin{minipage}[t]{.17\linewidth}\vspace{0pt}\centering' + '\n'
        body += rf'\includegraphics[width=\linewidth,height=.65in,keepaspectratio]{{{PREFIX}/{ident}/mask_{step}.png}}\par'
        body += r'{\small Soft support}\end{minipage}\par\medskip' + '\n'
    return body


def model(ident, title, record):
    body = heading(title)
    body += row(ident, ['input', 'prediction'], ['Input', 'Model output'], '.486', 2.65)
    body += r'\noindent\rule{\linewidth}{.3pt}\par\smallskip' + '\n'
    body += field('User instruction.', record['instruction'])
    body += r'\medskip' + '\n' + heading('Editing description')
    raw = re.sub(r'<vr_stage_\d+>', '', record['reasoning']).strip()
    match = re.search(r'Observation:\s*(.*?)\s*Mask:\s*(.*?)\s*Adjustment:\s*(.*)', raw, re.S)
    if not match:
        raise ValueError(f'Invalid reasoning structure: {ident}')
    for i, label in enumerate(['Observation.', 'Mask.', 'Adjustment.'], 1):
        body += field(label, match[i])
    return body


def main():
    OUT.mkdir(exist_ok=True)
    install_assets()
    figures = []
    for ident, title in RECOVERY_CASES:
        record = json.loads((ASSETS / ident / 'provenance.json').read_text())
        caption = ('Six-step degradation and recovery on a selected Unsplash training example. '
                   'Both sequences read left to right: recorded degradation above, sequential execution '
                   'of codes fitted from adjacent state pairs below. Supports include the recorded '
                   'strength and use a common zero-to-one scale. This is target-conditioned recovery.')
        figures.append((ident, recovery(ident, title, record), caption, 'recovery'))
    for ident, title in RECOVERY_CASES:
        record = json.loads((ASSETS / ident / 'provenance.json').read_text())
        for part in range(2):
            figures.append((f'{ident}_annotation_{part+1}', annotation(ident, title, record, part),
                            f'Archived MaskGrade annotation for {title.lower()}, moves {part*3+1}--{part*3+3}. '
                            'Text and supports correspond to the selected constructed training trajectory.', 'annotation'))
    for ident, title in MODEL_CASES:
        record = json.loads((ASSETS / ident / 'provenance.json').read_text())
        figures.append((ident, model(ident, title, record),
                        f'Instruction-guided retouching: {title.lower()}. '
                        'Selected Unsplash training example using the continuous-readout checkpoint at update 800. '
                        'The description is cached base-SFT output; color readout and rendering were rerun without the target.', 'model'))
    manifest = []
    section = []
    previous = None
    for ident, content, caption, kind in figures:
        source = OUT / f'{ident}.tex'
        source.write_text(document(content))
        with tempfile.TemporaryDirectory(prefix='chiaro-figure-') as build:
            result = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error',
                                     f'-output-directory={build}', str(source)], cwd=PAPER,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if result.returncode or 'Overfull' in result.stdout:
                raise RuntimeError(result.stdout[-6000:])
            shutil.copyfile(Path(build) / f'{ident}.pdf', OUT / f'{ident}.pdf')
        info = subprocess.check_output(['pdfinfo', str(OUT / f'{ident}.pdf')], text=True)
        width, height = map(float, re.search(r'Page size:\s+([\d.]+) x ([\d.]+)', info).groups())
        if abs(width - 396) > .1 or height > 555:
            raise ValueError(f'{ident} exceeds safe figure canvas: {width}x{height}')
        manifest.append({'id': ident, 'kind': kind, 'width_pt': width, 'height_pt': height,
                         'caption': caption, 'sha256': hashlib.sha256((OUT / f'{ident}.pdf').read_bytes()).hexdigest()})
        section.append(r'\clearpage')
        if kind != previous:
            headings = {'recovery': ('MaskGrade Degradation and Recovery', 'app:maskgrade_examples'),
                        'annotation': ('Complete MaskGrade Annotation Records', 'app:maskgrade_records'),
                        'model': ('Instruction-Guided Retouching Examples', 'app:training_qualitative')}
            name, label = headings[kind]
            section.extend([r'\section{' + name + '}', r'\label{' + label + '}'])
            previous = kind
        section.extend([r'\begin{figure}[!ht]', r'\centering',
                        rf'\includegraphics[width=\linewidth]{{figures/appendix_paper_samples/{ident}.pdf}}',
                        r'\caption{' + caption + '}', r'\label{fig:sample_' + ident + '}', r'\end{figure}', r'\FloatBarrier'])
        print(f'{ident}: {width:.1f} x {height:.1f} pt', flush=True)
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    # Keep generated section separate; the author-approved production prompts are untouched.
    (PAPER / 'sections/appendix/sample_figures.tex').write_text('\n'.join(section) + '\n')


if __name__ == '__main__':
    main()
