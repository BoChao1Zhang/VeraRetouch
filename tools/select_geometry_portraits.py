"""Select human-photo candidates by recorded geometry, never by CoT wording."""
import csv
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path
import sqlite3
import argparse
from PIL import Image

from tools.paper_appendix_qualitative import PLAN, RECORDS, REPO

ROOT=Path('/home/bc/data/runs/paper_geometry_portraits_20260920')
INDEX=Path('/home/bc/data/runs/epr055_sprfv3/refit_20260911/lean_index.json')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--extend-per-family',type=int);args=parser.parse_args()
    ROOT.mkdir(exist_ok=True)
    if args.extend_per_family:
        groups=json.loads((ROOT/'all_candidates.json').read_text())
        selected=json.loads((ROOT/'local_candidates.json').read_text())
        index=json.loads(INDEX.read_text())
        sizes={}
        for kind in ['band','linear','radial']:
            seen={r['source_id'] for r in selected if r['geometry_kind']==kind}
            choices=sorted(groups[kind],key=lambda r:hashlib.sha256(('geometry-review:'+r['key']).encode()).hexdigest())
            for portrait in [True,False]:
                for r in choices:
                    if len(seen)>=args.extend_per_family:break
                    sid=r['source_id']
                    if sid in seen:continue
                    if sid not in sizes:
                        with Image.open(index[r['key']]['png']) as im:sizes[sid]=im.size
                    w,h=sizes[sid]
                    if (h>w)!=portrait:continue
                    selected.append(r);seen.add(sid)
                if len(seen)>=args.extend_per_family:break
            assert len(seen)==args.extend_per_family,(kind,len(seen))
            print(json.dumps(dict(family=kind,selected=len(seen))),flush=True)
        (ROOT/'local_candidates.json').write_text(json.dumps(selected,indent=2)+'\n')
        print(json.dumps(dict(selected=dict(Counter(r['geometry_kind'] for r in selected)),preserved_prefix=90)),flush=True)
        return
    sources={r['source_id']:r for r in csv.DictReader(open(REPO/'tools/data_splits/splits_sources.csv'))}
    train={r['key'] for r in json.loads(PLAN.read_text())['records']}
    humans=set()
    for line in (REPO/'EPR/ICLR2027/figures/appendix_scene_words/counting_trace.jsonl').open():
        r=json.loads(line)
        if set(r['scene_terms'])&{'Portraits','Women','Men','Children','Couples','Weddings'}:
            humans.add(r['source_id'])
    annotations={}
    for line in RECORDS.open():
        r=json.loads(line);sid=r['key'].split('.rep')[0]
        if r['key'] in train and sid in humans and sources.get(sid,{}).get('pool') in {'ppr10k','unsplash','fivek_gold'}:
            annotations[r['key']]=r['answer']
    index=json.loads(INDEX.read_text());shards=defaultdict(list)
    for key in annotations:shards[index[key]['dir']].append(key)
    groups=defaultdict(list);counts=Counter()
    for path,keys in sorted(shards.items()):
        with (Path(path)/'pairs.jsonl').open('rb') as f:
            for key in sorted(keys,key=lambda k:index[k]['off']):
                ent=index[key];f.seek(ent['off']);r=json.loads(f.read(ent['len']))
                kind=r['mask']['geom'];counts[kind]+=1
                sid=key.split('.rep')[0]
                groups[kind].append(dict(key=key,source_id=sid,annotation=annotations[key],
                    instruction=annotations[key]['instruction_medium'],pool=sources[sid]['pool'],split='train',
                    geometry_kind=kind,geometry_parameters=r['mask'].get('geom_params'),
                    geometry_source=dict(shard=path,offset=ent['off'],length=ent['len'])))
    selected=[]
    for kind,rows in sorted(groups.items()):
        rows.sort(key=lambda r:hashlib.sha256(('geometry-review:'+r['key']).encode()).hexdigest())
        seen=set();subset=[]
        for r in rows:
            if r['source_id'] in seen:continue
            seen.add(r['source_id']);subset.append(r)
            if len(subset)>=30:break
        if kind not in {'semantic','subject'}:selected.extend(subset)
    (ROOT/'geometry_counts.json').write_text(json.dumps(dict(counts),indent=2)+'\n')
    (ROOT/'all_candidates.json').write_text(json.dumps(dict(groups),indent=2)+'\n')
    (ROOT/'local_candidates.json').write_text(json.dumps(selected,indent=2)+'\n')
    print(json.dumps(dict(recorded_counts=dict(counts),selected=dict(Counter(r['geometry_kind'] for r in selected)))),flush=True)


if __name__=='__main__':main()
