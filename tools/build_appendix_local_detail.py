"""Traceable local-edit zooms and true adjacent-output residual maps.

Photos are never recolored for display. Matched crops use one coordinate box.
Heatmaps encode 100*mean_RGB(abs(clip(z_k)-clip(z_{k-1}))). A single, untruncated
scale is shared by all six steps of each case. Composition happens in LaTeX.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, colors, cm

from tools.build_appendix_paper_samples import PAPER, tex

WORK = Path('/home/bc/data/runs/paper_appendix_expansion_20260919')
ASSETS = PAPER/'figures/appendix_local_detail'


def crop_box(energy, fraction=.34):
    h, w = energy.shape
    cw = max(1, round(w*fraction))
    # Portrait-shaped crops waste most of a horizontal comparison cell.
    # Use a wider detail window without distorting any sampled pixels.
    ch = max(1, min(round(h*fraction), round(cw*.75)))
    integral = np.pad(energy.astype(np.float64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    totals = integral[ch:, cw:]-integral[:-ch, cw:]-integral[ch:, :-cw]+integral[:-ch, :-cw]
    y, x = np.unravel_index(totals.argmax(), totals.shape)
    return [int(x), int(y), int(x+cw), int(y+ch)]


def detail_energy(change, rgb):
    gray = np.asarray(rgb, dtype=np.float32).mean(-1)
    gy, gx = np.gradient(gray)
    edges = np.abs(gx)+np.abs(gy)
    return change*(.25+np.minimum(edges/max(float(edges.mean()), 1e-6), 4))


def compile_pdf(source, destination):
    with tempfile.TemporaryDirectory(prefix='appendix-local-') as temp:
        result = subprocess.run(['pdflatex', '-halt-on-error', '-interaction=nonstopmode',
                                 f'-output-directory={temp}', str(source)], cwd=PAPER,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if result.returncode or 'Overfull' in result.stdout:
            raise RuntimeError(result.stdout[-5000:])
        shutil.copyfile(Path(temp)/(source.stem+'.pdf'), destination)


def photo(folder, name, width, height, box=None, size=None):
    source = folder/(name+'.png')
    display = folder/(name+'_display.png')
    # Full-frame thumbnails are printed small; avoid embedding repeated
    # full-resolution rasters. Actual matched crops retain their source pixels.
    if not display.exists() and 'zoom' not in name:
        frame = Image.open(source)
        if max(frame.size) > 384:
            frame.thumbnail((384,384), Image.Resampling.LANCZOS)
            frame.save(display)
    path = str((display if display.exists() else source).relative_to(PAPER))
    include = rf'\includegraphics[width={width}\linewidth,height={height}in,keepaspectratio]{{{path}}}'
    if box is None:
        return include
    w, h = size; x0, y0, x1, y1 = box
    return (r'\begin{tikzpicture}\node[anchor=south west,inner sep=0] (pic) at (0,0) {' + include + r'};'
            + r'\begin{scope}[x={(pic.south east)},y={(pic.north west)}]'
            + rf'\draw[draw=orange!90!black,line width=.55pt] ({x0/w:.6f},{1-y1/h:.6f}) rectangle ({x1/w:.6f},{1-y0/h:.6f});'
            + r'\end{scope}\end{tikzpicture}')


def build_case(record, index):
    original = Path(record['folder']); folder = ASSETS/f'case_{index:02d}'
    folder.mkdir(parents=True, exist_ok=True)
    if (original/'float_states.npz').exists():
        with np.load(original/'float_states.npz') as saved:
            states = np.clip(saved['recovery'], 0, 1)
        precision = 'clipped float32 execution states, before PNG quantization'
    else:
        states = np.stack([np.asarray(Image.open(original/f'recovery_{j}.png'), dtype=np.float32)/255 for j in range(7)])
        precision = 'decoded archived uint8 execution outputs'
    delta = np.abs(np.diff(states, axis=0)).mean(-1)*100
    maximum = max(1., float(np.ceil(delta.max())))
    cmap = matplotlib.colormaps['inferno']
    crop_records = []
    for j in range(7):
        shutil.copyfile(original/f'recovery_{j}.png', folder/f'recovery_{j}.png')
    shutil.copyfile(original/'reference.png', folder/'reference.png')
    for j in range(1, 7):
        shutil.copyfile(original/f'mask_{j}.png', folder/f'mask_{j}.png')
        box = crop_box(detail_energy(delta[j-1], states[j-1]))
        before = Image.open(folder/f'recovery_{j-1}.png')
        after = Image.open(folder/f'recovery_{j}.png')
        before.crop(box).save(folder/f'before_zoom_{j}.png')
        after.crop(box).save(folder/f'after_zoom_{j}.png')
        rgba = (cmap(delta[j-1]/maximum)*255).astype(np.uint8)
        Image.fromarray(rgba[..., :3]).save(folder/f'residual_{j}.png')
        crop_records.append(dict(step=j, box_xyxy=box, source_size=list(after.size),
                                 mean_change=float(delta[j-1].mean()), max_change=float(delta[j-1].max())))
    fig, axis = plt.subplots(figsize=(3.7, .40))
    fig.subplots_adjust(left=.025, right=.97, top=.90, bottom=.60)
    colorbar = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(0, maximum), cmap=cmap), cax=axis, orientation='horizontal')
    colorbar.set_ticks([0, maximum/2, maximum]); colorbar.ax.tick_params(labelsize=8, length=2)
    colorbar.set_label('Actual change (mean absolute RGB difference x100)', fontsize=8, labelpad=1)
    fig.savefig(folder/'colorbar.pdf', bbox_inches='tight', pad_inches=.02); plt.close(fig)
    manifest = dict(**record, residual_definition='100*mean_RGB(abs(clip(z_k,0,1)-clip(z_(k-1),0,1)))',
                    residual_state_precision=precision, residual_scale=[0, maximum],
                    crop_policy='fixed-size window maximizing texture-weighted actual adjacent-output change; identical before/after coordinates',
                    crops=crop_records)
    (folder/'provenance.json').write_text(json.dumps(manifest, indent=2)+'\n')
    labels = ['Global', 'Hue', 'Shadows', 'Midtones', 'Highlights', 'Spatial']
    lines = [r'\documentclass[10pt,border=0pt]{standalone}', r'\usepackage{times,graphicx,tikz,xcolor,array}',
             r'\begin{document}\begin{minipage}{5.5in}\fontsize{8.5}{10}\selectfont\setlength{\parindent}{0pt}',
             r'\textbf{Local recovery '+str(index)+r'}\par\smallskip',
             r'\textbf{Request.} '+tex(record['annotation']['instruction_medium'])+r'\par\smallskip',
             r'\begin{minipage}{.49\linewidth}\centering', photo(folder, 'recovery_0', 1, .95),
             r'\par Degraded input\end{minipage}\hfill\begin{minipage}{.49\linewidth}\centering',
             photo(folder, 'recovery_6', 1, .95), r'\par Recovered output\end{minipage}\par\medskip',
             r'\setlength{\tabcolsep}{1pt}\begin{tabular}{@{}*{5}{>{\centering\arraybackslash}p{.192\linewidth}}@{}}',
             r'Current state & Support & Before (zoom) & After (zoom) & Actual residual\\']
    for j in range(1, 7):
        caption = record['annotation']['cot'][j-1]['observation']
        # Original first sentence only, not an invented interpretation.
        sentence = caption.split('. ')[0].rstrip('.')+'.'
        if len(sentence.split()) > 28:
            sentence = ' '.join(sentence.split()[:28]).rstrip('.,;:')+'...'
        lines += [r'\multicolumn{5}{@{}p{\linewidth}@{}}{\vspace{3pt}\textbf{'+str(j)+'. '+labels[j-1]+r'.} '+tex(sentence)+r'}\\[2pt]']
        crop = crop_records[j-1]
        panels = [photo(folder, f'recovery_{j}', 1, .52, crop['box_xyxy'], crop['source_size']),
                  photo(folder, f'mask_{j}', 1, .52), photo(folder, f'before_zoom_{j}', 1, .52),
                  photo(folder, f'after_zoom_{j}', 1, .52), photo(folder, f'residual_{j}', 1, .52)]
        lines += [' & '.join(panels)+r'\\']
    lines += [r'\end{tabular}\par\smallskip\centering',
              r'\includegraphics[width=.72\linewidth]{'+str((folder/'colorbar.pdf').relative_to(PAPER))+'}',
              r'\end{minipage}\end{document}']
    source = folder/'detail.tex'; source.write_text('\n'.join(lines)+'\n')
    compile_pdf(source, folder/'detail.pdf')
    return str((folder/'detail.pdf').relative_to(PAPER))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--cases'); p.add_argument('--preview', action='store_true')
    args = p.parse_args(); records = json.loads((WORK/'local_results.json').read_text())
    if args.preview:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
        for part in range(2):
            canvas = Image.new('RGB', (1200, 1260), 'white'); draw = ImageDraw.Draw(canvas)
            for i, r in enumerate(records[part*12:part*12+12]):
                folder = Path(r['folder']); x=(i%2)*600; y=(i//2)*210
                draw.text((x+3, y+2), f'{folder.name}: local {r["local_amplitude"]:.1f}; area {r["local_area"]:.2f} (internal selection)', fill='black', font=font)
                for j, name in enumerate(['recovery_5', 'recovery_6', 'mask_6']):
                    frame = Image.open(folder/(name+'.png')).convert('RGB'); frame.thumbnail((195,180))
                    canvas.paste(frame,(x+j*200+(195-frame.width)//2, y+25))
            canvas.save(WORK/f'local_selection_internal_{part}.png')
        return
    if not args.cases:
        p.error('--cases is required unless --preview is set')
    by_name = {Path(r['folder']).name:r for r in records}
    selected = [by_name[name] for name in args.cases.split(',')]
    output = [build_case(row, index) for index, row in enumerate(selected, 1)]
    (ASSETS/'selection.json').write_text(json.dumps(dict(candidates=len(records), cases=selected, files=output), indent=2)+'\n')
    print(json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
