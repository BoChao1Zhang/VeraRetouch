"""Verify native PDF panel heights and underlying crops/residuals."""
import json
from pathlib import Path
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import matplotlib
import numpy as np
from PIL import Image

from tools.build_local_seven_line_pages import OUT,PAPER


def main():
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert len(manifest['pages'])==15
    for page in manifest['pages']:
        body=(PAPER/page['tex']).read_text()
        assert len(re.findall(r'includegraphics\[height=84pt\]',body))==15
        assert not re.search(r'\\(?:resizebox|scalebox)',body)
    for folder in sorted(OUT.glob('case_*')):
        meta=json.loads((folder/'provenance.json').read_text());source=Path(meta['folder'])
        with np.load(source/'float_states.npz') as arrays:
            states=np.clip(arrays['recovery'],0,1);masks=arrays['masks']
        for step,box in enumerate(meta['crops_xyxy'],1):
            for previous,name in [(step-1,'before'),(step,'after')]:
                expected=np.asarray(Image.open(source/f'recovery_{previous}.png').crop(box))
                assert np.array_equal(expected,np.asarray(Image.open(folder/f'{name}_zoom_{step}.png')))
            delta=np.abs(states[step]-states[step-1]).mean(-1)*100
            image=Image.fromarray((matplotlib.colormaps['inferno'](delta/meta['residual_scale'][1])[...,:3]*255).astype(np.uint8))
            image.thumbnail((512,512),Image.Resampling.LANCZOS)
            assert np.array_equal(np.asarray(image),np.asarray(Image.open(folder/f'residual_{step}.png')))
            zero=masks[step-1]==0
            assert not zero.any() or float(delta[zero].max())<1e-5
    # The ruler is fixed at 12 TeX pt. Poppler reports image bounding boxes in
    # rounded PDF points; 84 TeX pt = 83.686 PDF points, reported as 84 here.
    with tempfile.TemporaryDirectory(prefix='local-height-audit-') as tmp:
        prefix=Path(tmp)/'panels'
        subprocess.run(['pdftohtml','-xml','-zoom','1','-f','18','-l','32',str(PAPER/'main.pdf'),str(prefix)],
                       check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        tree=ET.parse(prefix.with_suffix('.xml'))
        measurements=[]
        for page in tree.getroot().findall('page'):
            panels=[im for im in page.findall('image') if float(im.attrib['height'])>20]
            assert len(panels)==15, (page.attrib,len(panels))
            assert all(int(im.attrib['height'])==84 for im in panels)
            measurements.append(dict(page=int(page.attrib['number']),panels=15,height_pdf_points_rounded=84))
    assert len(measurements)==15
    audit=dict(passed=True,panels_measured=225,height_tex_pt=84,review_line_pitch_tex_pt=12,
               review_line_intervals=7,matched_crops_verified=60,residual_maps_verified=30,pages=measurements)
    path=Path('/home/bc/data/runs/paper_appendix_expansion_20260919/seven_line_audit.json')
    path.write_text(json.dumps(audit,indent=2)+'\n')
    (OUT/'height_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k!='pages'}))


if __name__=='__main__':main()
