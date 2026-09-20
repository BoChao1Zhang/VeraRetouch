"""Diverse, image-deduplicated landscape shortlist for hue/tonal-mask review."""
from collections import Counter,defaultdict,deque
import csv
import hashlib
import json
from pathlib import Path
import re
from tools.paper_appendix_qualitative import PLAN,RECORDS,REPO

WORK=Path('/home/bc/data/runs/paper_landscape_review500_20260920')
EXCLUDED={'Women','Men','People','Children','Portraits','Couples','Weddings','Dogs','Cats','Horses','Food','Fruit','Rooms','Interiors','Still life','Macro'}
SCENES=[('Night',{'Night'}),('Desert',{'Desert'}),('Snow and ice',{'Snow'}),
        ('Waterfalls and rivers',{'Waterfalls','Rivers'}),('Coast and sea',{'Coast','Sea','Beaches'}),
        ('Lakes',{'Lakes'}),('Mountains',{'Mountains','Rocks','Hills'}),
        ('Forest and woodland',{'Forests','Trees','Foliage'}),('Fields and grassland',{'Fields','Grass'}),
        ('City and architecture',{'Cities','Buildings','Architecture','Bridges','Streets','Temples','Churches'}),
        ('Dawn and dusk',{'Sunrise','Sunset'}),('Open sky',{'Sky','Clouds','Landscape'})]


def style(text):
    for label,pattern in [('Vintage / film',r'vintage|retro|film|cinematic'),('Soft / muted',r'muted|soft|subdued|gentle'),
                          ('Rich / contrast',r'contrast|dramatic|richer|deep'),('Warm',r'warm|golden'),
                          ('Cool',r'cool|blue'),('Bright / airy',r'bright|airy')]:
        if re.search(pattern,text,re.I):return label
    return 'Natural / balanced'


def main():
    WORK.mkdir(exist_ok=True)
    sources={r['source_id']:r for r in csv.DictReader(open(REPO/'tools/data_splits/splits_sources.csv'))}
    train={r['key'] for r in json.loads(PLAN.read_text())['records']}
    old=json.loads((REPO/'outputs/local_review500_20260920/combined_results.json').read_text())
    excluded={r['source_id'] for r in old}
    eligible={}
    for line in (REPO/'EPR/ICLR2027/figures/appendix_scene_words/counting_trace.jsonl').open():
        r=json.loads(line);sid=r['source_id'];terms=set(r['scene_terms'])
        if sid in excluded or terms&EXCLUDED or r['canonical_key'] not in train:continue
        if sources.get(sid,{}).get('pool') not in {'unsplash','fivek_gold'}:continue
        scene=next((label for label,words in SCENES if terms&words),None)
        if scene:eligible[r['canonical_key']]=dict(r,scene_bucket=scene)
    bins=defaultdict(list)
    for line in RECORDS.open():
        r=json.loads(line);key=r['key']
        if key not in eligible:continue
        meta=eligible[key];sid=meta['source_id'];answer=r['answer'];instruction=answer['instruction_medium']
        bucket=style(instruction)
        row=dict(key=key,source_id=sid,annotation=answer,instruction=instruction,pool=sources[sid]['pool'],split='train',
                 scene_bucket=meta['scene_bucket'],style_bucket=bucket,scene_terms=meta['scene_terms'],intent_terms=meta['intent_terms'])
        bins[(row['scene_bucket'],bucket)].append(row)
    queues={k:deque(sorted(rows,key=lambda r:hashlib.sha256(('landscape-review-v1:'+r['key']).encode()).hexdigest())) for k,rows in bins.items()}
    # Round-robin within scenes and style families, not just a single photo genre.
    by_scene=defaultdict(list)
    for key in queues:by_scene[key[0]].append(key)
    selected=[];style_cursor=Counter()
    while len(selected)<500:
        advanced=False
        for scene in sorted(by_scene):
            choices=sorted(k for k in by_scene[scene] if queues[k])
            if not choices:continue
            key=choices[style_cursor[scene]%len(choices)];style_cursor[scene]+=1
            row=queues[key].popleft();row['review_focus_step']=2+(len(selected)%4);selected.append(row);advanced=True
            if len(selected)==500:break
        if not advanced:raise RuntimeError(f'Only {len(selected)} eligible landscapes')
    assert len({r['source_id'] for r in selected})==500
    (WORK/'local_candidates.json').write_text(json.dumps(selected,indent=2)+'\n')
    report=dict(count=500,eligible=len(eligible),scene_buckets=dict(Counter(r['scene_bucket'] for r in selected)),
                style_buckets=dict(Counter(r['style_bucket'] for r in selected)),focus_steps=dict(Counter(r['review_focus_step'] for r in selected)),
                note='Sampling groups from annotation vocabulary, not verified scene-class labels. 500 different source IDs; no overlap with first-round portrait gallery.')
    (WORK/'selection_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
