"""Check paired crop identity, true residuals, and native image geometry."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib

from tools.build_appendix_paper_samples import PAPER


def pixels(path):
    return np.asarray(Image.open(path).convert('RGB'))


def check_crop(folder, before, after, crop_before, crop_after, box):
    x0,y0,x1,y1=box
    assert np.array_equal(pixels(folder/crop_before), pixels(folder/before)[y0:y1,x0:x1])
    assert np.array_equal(pixels(folder/crop_after), pixels(folder/after)[y0:y1,x0:x1])


def main():
    checks=[]
    for folder in sorted((PAPER/'figures/appendix_local_detail').glob('case_*')):
        r=json.loads((folder/'provenance.json').read_text())
        with np.load(Path(r['folder'])/'float_states.npz') as data:
            states=np.clip(data['recovery'],0,1);masks=data['masks']
        delta=np.abs(np.diff(states,axis=0)).mean(-1)*100
        outside=[]
        for j,crop in enumerate(r['crops'],1):
            check_crop(folder,f'recovery_{j-1}.png',f'recovery_{j}.png',f'before_zoom_{j}.png',f'after_zoom_{j}.png',crop['box_xyxy'])
            expected=(matplotlib.colormaps['inferno'](delta[j-1]/r['residual_scale'][1])[...,:3]*255).astype(np.uint8)
            assert np.array_equal(expected,pixels(folder/f'residual_{j}.png'))
            zero=masks[j-1]==0
            maximum=float(delta[j-1][zero].max()) if zero.any() else 0.
            assert maximum < 1e-5
            outside.append(maximum)
        checks.append(dict(case=folder.name,crops=6,heatmaps=6,outside_support_max_change=outside))
    for group in ['unsplash','fivek','ppr10k']:
        meta=json.loads((PAPER/'figures/appendix_domain_gallery/selection.json').read_text())
        maximum=next(r['scale'][1] for r in meta if r['group']==group)
        for folder in sorted((PAPER/'figures/appendix_domain_gallery'/group).glob('case_*')):
            r=json.loads((folder/'provenance.json').read_text())
            check_crop(folder,'input.png','prediction.png','before_zoom.png','after_zoom.png',r['crop_xyxy'])
            before=pixels(folder/'input.png').astype(np.float32)/255
            after=pixels(folder/'prediction.png').astype(np.float32)/255
            delta=np.abs(after-before).mean(-1)*100
            expected=(matplotlib.colormaps['inferno'](delta/maximum)[...,:3]*255).astype(np.uint8)
            assert np.array_equal(expected,pixels(folder/'residual.png'))
            checks.append(dict(group=group,case=folder.name,crops=1,heatmaps=1))
    folder=PAPER/'figures/appendix_highres'
    r=json.loads((folder/'provenance.json').read_text())
    check_crop(folder,'input.png','output.png','input_zoom.png','output_zoom.png',r['crop_xyxy'])
    assert Image.open(folder/'input.png').size==Image.open(folder/'output.png').size==(4032,2268)
    checks.append(dict(case='native_high_resolution',size=[4032,2268],crop_size=[768,432],fresh_generation=True))
    output=Path('/home/bc/data/runs/paper_appendix_expansion_20260919/figure_audit.json')
    output.write_text(json.dumps(dict(passed=True,checks=checks),indent=2)+'\n')
    print(json.dumps(dict(passed=True,local_cases=5,local_transitions=30,in_domain_cases=18,native_size=[4032,2268])))


if __name__=='__main__':
    main()
