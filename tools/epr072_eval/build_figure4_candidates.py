"""Author-selected ArtEdit candidates, preserving per-sample CoT choice."""
import hashlib
import json
from pathlib import Path
import re

from PIL import Image

REPO=Path('/home/bc/VeraRetouch')
OUT=REPO/'outputs/figure4_selected_review_20260921'
TOKENS='091 0195 0199 0205 0208 0214 0220 0222 0225 0228 0229 0232 -243 249_COT 0259 0267 0268 0273 0280 0285 0288 0295 0297 303 305 306 307 0314 321 322 324 327 329 344 345 349 353 356 357 360 371 375 377_COT 383 386 390 393 394_COT 396 398'.split()
BASE=Path('/home/bc/data/runs/pubbench_artedit')
METHODS=dict(jarvisevo=BASE/'jarvisevo_lr/en/instr_real',
             monetgpt=BASE/'monetgpt/en/instr_real',veraretouch=BASE/'veraretouch/en/real')


def main():
    pairs={r['id']:r for r in json.loads((REPO/'outputs/sixstage_cot_pair_review/records.json').read_text())}
    OUT.mkdir(parents=True,exist_ok=True)
    records=[]
    for rank,token in enumerate(TOKENS):
        number=int(re.search(r'\d+',token).group())
        sid=f'artedit_en_{number:04d}'
        mode='cot' if token.endswith('_COT') else 'nocot'
        prior=pairs[sid]
        pair_dir=REPO/'outputs/sixstage_cot_pair_review'/sid
        sources=dict(input=(pair_dir/'input.jpg').resolve(),
                     ours=(pair_dir/(mode+'.png')).resolve(),
                     reference=(pair_dir/'reference.jpg').resolve())
        for method,folder in METHODS.items():
            sources[method]=folder/(sid+'.png')
        folder=OUT/sid;folder.mkdir(exist_ok=True)
        w,h=Image.open(sources['input']).size
        row=dict(id=sid,number=number,original_token=token,rank=rank,mode=mode,
                 orientation='landscape' if w>=h else 'portrait',size=[w,h],
                 instruction=prior['instruction'],images={},missing=[],geometry_mismatch=[],sha256={})
        for method,source in sources.items():
            if not source.is_file():
                row['missing'].append(method);continue
            size=Image.open(source).size
            if size!=(w,h):row['geometry_mismatch'].append(dict(method=method,size=list(size)))
            dest=folder/(method+source.suffix)
            if dest.is_symlink():assert dest.resolve()==source.resolve()
            elif dest.exists():raise ValueError(f'Refusing overwrite {dest}')
            else:dest.symlink_to(source)
            row['images'][method]=str(dest.relative_to(OUT))
            row['sha256'][method]=hashlib.sha256(source.read_bytes()).hexdigest()
        row['complete']=not row['missing'] and not row['geometry_mismatch']
        records.append(row)
    assert len({r['id'] for r in records})==len(TOKENS)
    records.sort(key=lambda r:(r['orientation']!='landscape',r['rank']))
    (OUT/'records.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
    (OUT/'index.html').write_text((REPO/'tools/epr072_eval/figure4_candidates.html').read_text())
    audit=dict(n=len(records),landscape=sum(r['orientation']=='landscape' for r in records),
               portrait=sum(r['orientation']=='portrait' for r in records),
               with_cot=[r['number'] for r in records if r['mode']=='cot'],
               incomplete=[r['id'] for r in records if not r['complete']],
               ordering='landscape first, portrait second; author order retained within each group')
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit),flush=True)


if __name__=='__main__':main()
