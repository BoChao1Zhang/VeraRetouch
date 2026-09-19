"""Legacy standalone browser cards, NOT for direct appendix insertion.

Letter-width cards collide with the ICLR review ruler when used as full-page
overlays. Use build_appendix_paper_samples.py for the paper's text-width PDFs.
This exporter is retained only to reproduce the earlier independent preview.
"""
import argparse
import base64
import html
import json
from pathlib import Path
import re
import subprocess
import tempfile

ROOT=Path('/home/bc/VeraRetouch/EPR/ICLR2027')
ASSETS=ROOT/'figures/appendix_qualitative_20260919'
DEST=ROOT/'figures/appendix_sample_pdfs'
LABELS={'global':'Global','hue':'Hue','lum_shadow':'Shadows','lum_mid':'Midtones','lum_high':'Highlights','geom':'Spatial'}
STYLE='''
@page{size:Letter;margin:.55in .60in .55in}
*{box-sizing:border-box}body{margin:0;color:#20332f;font:10pt/1.48 Arial,Helvetica,sans-serif;-webkit-print-color-adjust:exact;print-color-adjust:exact}
.page{break-after:page}.page:last-child{break-after:auto}
header{display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #d9e2dc;padding-bottom:9px;margin-bottom:15px;font-size:8pt;color:#66736d}
.brand{font-weight:bold;font-size:11pt;color:#176b52}h1{margin:0 0 4px;font-size:21pt;line-height:1.15;font-weight:650}h2{font-size:12pt;margin:0 0 10px}h3{font-size:10pt;margin:0 0 5px}.subtitle{font-size:9pt;color:#63736c;margin:0 0 12px}
.tags{display:flex;gap:6px;margin-bottom:14px}.tag{font-size:7.5pt;color:#176b52;background:#edf5ef;border:1px solid #d8e6dc;border-radius:12px;padding:2px 8px}
.panel{border:1px solid #dce3dd;border-radius:10px;padding:13px;margin-bottom:13px;break-inside:avoid}
.request{border-left:3px solid #bdd4c3;padding-left:12px;font:11pt/1.5 Georgia,serif;margin:0}
.row{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}figure{margin:0;min-width:0}.row img{width:100%;height:1.55in;object-fit:contain;background:#f0f2ec;border-radius:5px}figcaption{font-size:8pt;margin-top:5px;display:flex;justify-content:space-between}.metrics{border-top:1px solid #dce3dd;display:flex;gap:26px;padding-top:10px;margin-top:12px;font-size:8pt}.metrics b{font-size:14pt;margin-right:5px}.good{color:#176b52}.muted{color:#66736d}.note{font-size:8pt;color:#66736d;margin:9px 0 0}
.states{display:grid;grid-template-columns:repeat(4,1fr);gap:9px}.states img{width:100%;height:1.02in;object-fit:contain;background:#f0f2ec;border-radius:4px}.states figcaption{font-size:7pt}
.masks{display:grid;grid-template-columns:repeat(6,1fr);gap:7px}.masks img{width:100%;height:.65in;object-fit:contain;background:#eee}.masks figcaption{font-size:7pt}.annotation{background:#fbfcf9}.annotation p{margin:5px 0;font-size:9pt;line-height:1.45}.field{font-size:8pt;color:#176b52;font-weight:bold;letter-spacing:.04em}.stepbar{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}.stepnumber{font-size:8pt;color:#176b52;background:#eaf3ec;padding:3px 8px;border-radius:7px}.step-content{display:grid;grid-template-columns:minmax(0,1fr) .85in;gap:10px}.step-content img{width:.85in;max-height:.8in;object-fit:contain}.step-content .masklabel{font-size:7pt;color:#66736d;text-align:center}.footer{font-size:7pt;color:#7a867f;margin-top:10px;padding-top:7px;border-top:1px solid #dce3dd;overflow-wrap:anywhere}
'''


def esc(x):return html.escape(str(x))
def photo(path):return 'data:image/png;base64,'+base64.b64encode(Path(path).read_bytes()).decode()
def img(path,caption):return f'<figure><img src="{photo(path)}" alt="{esc(caption)}"><figcaption>{esc(caption)}</figcaption></figure>'
def head(label):return f'<header><span class="brand">ChiaroRetouch</span><span>{esc(label)}</span></header>'
def para(label,text):return f'<p><span class="field">{label}</span><br>{esc(text)}</p>'


def render(ident,title):
    folder=ASSETS/ident;r=json.loads((folder/'provenance.json').read_text());recovery=ident.startswith('recovery')
    kind='MASKGRADE / STEPWISE RECOVERY' if recovery else 'MODEL / INSTRUCTION-GUIDED EDITING'
    request=r['annotation']['instruction_medium'] if recovery else r['instruction']
    initial=r['initial_l1'] if recovery else r['identity_l1'];final=r['final_l1'] if recovery else r['l1']
    frames=['recovery_0','reference','recovery_6'] if recovery else ['input','target','prediction']
    labels=['Input','Reference','Recovered output'] if recovery else ['Input','Target style','Model output']
    badge='Recorded-state code fitting' if recovery else 'R best@800'
    body=f'<section class="page">{head(kind)}<h1>{esc(title)}</h1><p class="subtitle">'+('Six sequential color updates with recorded spatial supports.' if recovery else 'A representative instruction-guided edit from the training domain.')+'</p>'
    body+=f'<div class="tags"><span class="tag">Unsplash</span><span class="tag">Training example</span><span class="tag">{badge}</span></div>'
    body+=f'<div class="panel"><h2>Editing request</h2><p class="request">{esc(request)}</p></div>'
    body+='<div class="panel"><h2>Input and result</h2><div class="row">'+''.join(img(folder/(f+'.png'),label) for f,label in zip(frames,labels))+'</div>'
    body+=f'<div class="metrics"><span><b>{initial:.2f}</b>Input error</span><span class="good"><b>{final:.2f}</b>Output error</span><span class="muted">MAE ×100 vs. reference</span></div>'
    body+='<p class="note">'+('Codes are fitted from recorded adjacent states and replayed sequentially. This is target-conditioned recovery.' if recovery else 'The readout predicts continuous color parameters; the target is used only for comparison.')+'</p></div>'
    if recovery:
        body+='<div class="panel"><h2>Restoration sequence</h2><div class="states">'
        for j in range(7):body+=img(folder/f'recovery_{j}.png','Input' if j==0 else f'{j:02d} · '+LABELS[r['stage_order'][j-1]])
        body+=img(folder/'reference.png','Reference')+'</div></div>'
        body+='<div class="panel"><h2>Soft execution supports</h2><div class="masks">'
        body+=''.join(img(folder/f'mask_{j}.png',LABELS[r['stage_order'][j-1]]) for j in range(1,7))
        body+='</div><p class="note">Common 0–1 scale; recorded weights already include the construction strength.</p></div>'
    else:
        raw=re.sub(r'<vr_stage_\d+>','',r['reasoning']).strip()
        m=re.search(r'Observation:\s*(.*?)\s*Mask:\s*(.*?)\s*Adjustment:\s*(.*)',raw,re.S)
        body+='<div class="panel annotation"><h2>Model-generated editing description</h2>'
        body+=''.join(para(label,m[i+1]) for i,label in enumerate(['OBSERVATION','MASK','ADJUSTMENT']))
        body+='<p class="note">Cached base-SFT description; continuous readout and rendering were rerun for this example.</p></div>'
    body+=f'<div class="footer">Sample {esc(r["source_id"])} · Selected qualitative training example</div></section>'
    if recovery:
        for part in range(2):
            body+=f'<section class="page">{head("MASKGRADE / ARCHIVED ANNOTATION")}<h1>{esc(title)}</h1><p class="subtitle">Stage descriptions · {part*3+1}–{part*3+3}</p>'
            body+='<p class="note" style="margin-bottom:14px">Original annotation associated with the constructed trajectory. Numerical descriptions are annotation records, not newly generated readout predictions.</p>'
            for c in r['annotation']['cot'][part*3:part*3+3]:
                s=c['step'];body+=f'<div class="panel annotation"><div class="stepbar"><h2 style="margin:0">{LABELS[r["stage_order"][s-1]]}</h2><span class="stepnumber">MOVE {s:02d}</span></div><div class="step-content"><div>'
                body+=''.join(para(label,c[field]) for label,field in [('OBSERVATION','observation'),('MASK','mask'),('ADJUSTMENT','adjustment')])
                body+=f'</div><div><img src="{photo(folder/f"mask_{s}.png")}" alt="Stage mask"><div class="masklabel">Soft support</div></div></div></div>'
            body+=f'<div class="footer">Sample {esc(r["source_id"])} · Full six-move annotation retained</div></section>'
    compact='''
    body{font-size:9pt;line-height:1.4}header{margin-bottom:10px;padding-bottom:6px}
    h1{font-size:18pt}h2{font-size:11pt;margin-bottom:7px}.subtitle{margin-bottom:9px}
    .tags{margin-bottom:10px}.panel{padding:10px;margin-bottom:10px}.request{font-size:10pt;line-height:1.4}
    .row img{height:1.23in}.states img{height:.76in}.masks img{height:.42in}
    .metrics{padding-top:7px;margin-top:8px}.metrics b{font-size:12pt}
    .annotation p{font-size:8.5pt;line-height:1.35;margin:4px 0}.stepbar{margin-bottom:5px}
    .annotation .field{font-size:7.5pt}.note{font-size:7.7pt;margin-top:6px}.footer{margin-top:7px}
    .step-content{display:flex;gap:10px}.step-content>div:first-child{flex:1;min-width:0}
    .step-content>div:last-child{width:.85in;flex:0 0 .85in}.annotation p{overflow-wrap:anywhere}
    '''
    html_doc=f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{esc(title)}</title><style>{STYLE}{compact}</style></head><body>{body}</body></html>'
    visible_text=re.sub(r'data:image/png;base64,[A-Za-z0-9+/=]+','',html_doc)
    if re.search(r'\b(?:nvidia|gpu|cuda|h100|a100)\b',visible_text,re.I):raise ValueError('Hardware text')
    return html_doc


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--only',default='');args=ap.parse_args()
    DEST.mkdir(parents=True,exist_ok=True)
    names=[('recovery_01','Backlit Portrait'),('recovery_02','Woodland Lake'),('recovery_03','Poolside Color'),
           ('inference_01','Warm Fur Tones'),('inference_02','Cool Waterfall Contrast'),('inference_03','Warm Interior')]
    outputs=[]
    for ident,title in names:
        if args.only and args.only!=ident:continue
        source=DEST/(ident+'.html');source.write_text(render(ident,title))
        pdf=DEST/(ident+'.pdf')
        with tempfile.TemporaryDirectory(prefix='chiaro-print-') as profile:
            command=['google-chrome','--headless','--no-sandbox','--disable-gpu',
                     '--disable-dev-shm-usage',f'--user-data-dir={profile}',
                     '--no-pdf-header-footer','--run-all-compositor-stages-before-draw',
                     '--virtual-time-budget=3000',f'--print-to-pdf={pdf}',source.as_uri()]
            subprocess.run(command,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=60)
        info=subprocess.check_output(['pdfinfo',str(pdf)],text=True)
        pages=int(re.search(r'Pages:\s+(\d+)',info)[1])
        outputs.append(dict(id=ident,pdf=str(pdf.relative_to(ROOT)),pages=pages))
        print(json.dumps(outputs[-1]),flush=True)
    (DEST/'print_manifest.json').write_text(json.dumps(outputs,indent=2)+'\n')


if __name__=='__main__':main()
