"""Align the four existing Figure 4 IDs with the main-table public protocol."""
import hashlib
import json
from pathlib import Path
import sqlite3

from PIL import Image

PAPER=Path('/home/bc/VeraRetouch/EPR/ICLR2027')
RUN=Path('/home/bc/nfsvfs/bc/data/runs/epr077_fivek_ppr10k_20260920/predictions')


def main():
    from veraretouch_sprf.e2e.bench import samples_for
    old=json.loads((PAPER/'figures/qualitative_instructions.json').read_text())
    records={bench:{r['sample_id']:r for r in samples_for(bench)[0]} for bench in ['fivek','ppr10k']}
    provenance=[]
    for entry in old['examples']:
        bench,sid=entry['dataset'],entry['sample_id']
        row=records[bench][sid]
        if row['instruction']!=entry['public_test_instruction']:
            raise ValueError('Archived public instruction mismatch')
        method=f'ours_local_pix3200_{bench}_nocot'
        facts=json.loads((RUN/method/f'epr077_{method}.json').read_text())
        assert facts['checkpoint']['step']==3200 and facts['text_mode']=='empty'
        dest=PAPER/'figures/qualitative_matched_pix3200'/f'{bench}_{sid}'
        dest.mkdir(parents=True,exist_ok=True)
        sources=dict(input=Path(row['input_path']),ours=RUN/method/'instr_real'/f'{sid}.png')
        for name in ['jarvisevo','monetgpt','veraretouch']:
            sources[name]=PAPER/entry['images'][name]
        images={k:Image.open(p).convert('RGB') for k,p in sources.items()}
        reference_ratio=images['input'].width/images['input'].height
        if any(abs(im.width/im.height-reference_ratio)>.01 for im in images.values()):
            raise ValueError('Aspect-ratio mismatch in existing baseline assets')
        hashes={}
        for name,im in images.items():
            im.save(dest/(name+'.png'))
            hashes[name]=hashlib.sha256(im.tobytes()).hexdigest()
        provenance.append(dict(bench=bench,sample_id=sid,instruction=row['instruction'],
                               protocol='public benchmark input and primary instruction; single-edit PIX3200 no-CoT',
                               checkpoint_sha256=facts['checkpoint']['sha256'],pixel_sha256=hashes))
    (PAPER/'figures/qualitative_matched_pix3200/provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(dict(samples=len(provenance),checkpoint='PIX3200',mode='single-edit no-CoT')))


if __name__=='__main__':main()
