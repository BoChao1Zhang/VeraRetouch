"""Install author-selected benchmark outputs with source-byte hash checks."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from PIL import Image

REPO=Path('/home/bc/VeraRetouch')
PAPER=REPO/'EPR/ICLR2027'
GALLERY=REPO/'outputs/figure4_selected_review_20260921'
METHODS=['input','jarvisevo','monetgpt','veraretouch','ours']
SHORT={232:"Warm the scene, enrich colors and contrast, and sharpen the dog's fur.",
       280:'Add a moody blue-green tone, darken the corners, and sharpen the image.',
       307:'Create a cinematic sunset with deep blacks, warm colors, and strong contrast.',
       91:'Brighten and warm the city, amplify the golden glow, and soften detail.',
       295:'Darken the background so the red flower stands out.',
       360:'Darken the background and enrich warm skin tones for a dramatic spotlight effect.'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('selection',type=Path);args=ap.parse_args()
    selection=json.loads(args.selection.read_text())
    rows=selection['selected'];assert len(rows)==6 and len({r['sample_id'] for r in rows})==6
    gallery={r['id']:r for r in json.loads((GALLERY/'records.json').read_text())}
    assert [r['orientation'] for r in rows]==['landscape']*3+['portrait']*3
    provenance=[]
    for item in rows:
        r=gallery[item['sample_id']]
        assert r['complete'] and item['mode']==r['mode']=='nocot'
        assert item['execution']=='six-stage' and item['checkpoint']=='PIX3200'
        folder=PAPER/'figures/qualitative_selected_six'/r['id'];folder.mkdir(parents=True,exist_ok=True)
        images={};display_images={}
        for method in METHODS:
            source=GALLERY/r['images'][method]
            raw=source.read_bytes()
            assert hashlib.sha256(raw).hexdigest()==item['sha256'][method]
            assert list(Image.open(source).size)==r['size']
            dest=folder/(method+source.suffix)
            shutil.copyfile(source,dest)
            images[method]=str(dest.relative_to(PAPER))
            image=Image.open(source).convert('RGB')
            if image.width>640:
                image=image.resize((640,round(640*image.height/image.width)),Image.Resampling.LANCZOS)
            display=folder/(method+'_display.png')
            image.save(display)
            display_images[method]=str(display.relative_to(PAPER))
        provenance.append(dict(sample_id=r['id'],orientation=r['orientation'],mode=r['mode'],
                               checkpoint='PIX3200',execution='six-stage',instruction=r['instruction'],
                               display_instruction=SHORT[r['number']],images=images,display_images=display_images,
                               display_transform='uniform width 640 px, aspect-preserving Lanczos; no crop or color adjustment',
                               source_sha256=item['sha256']))
    (PAPER/'figures/qualitative_selected_six/provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(dict(n=6,order=[r['sample_id'] for r in provenance],all_hashes_verified=True)))


if __name__=='__main__':main()
