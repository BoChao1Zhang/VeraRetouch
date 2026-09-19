"""Annotation-derived scene/subject and retouching vocabulary, image-deduplicated.

Counts are document frequencies over unique source-image IDs, not inferred
scene-class proportions. Lexical families and raw counts are exported for audit.
Each output asset contains a single word cloud; composition is done in LaTeX.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np

ROOT=Path('/home/bc/VeraRetouch')
PAPER=ROOT/'EPR/ICLR2027'
OUT=PAPER/'figures/appendix_scene_words'
RECORDS=Path('/home/bc/data/runs/epr051_vlmsft/snap_sft2/records.jsonl')
PLAN=Path('/home/bc/data/runs/epr058_color/cache_prepare_ar1600_v1/plan_trainfull.json')

# Surface forms only: no scene is assigned from its color or editing operation.
SCENES={
 'Sky':r'sky|skies', 'Trees':r'tree|trees', 'Flowers':r'flower|flowers|floral|bouquet|bouquets',
 'Buildings':r'building|buildings', 'Water':r'water', 'Women':r'woman|women',
 'Men':r'man|men', 'People':r'person|people', 'Children':r'child|children|boy|boys|girl|girls|baby|babies',
 'Portraits':r'portrait|portraits|portraiture', 'Couples':r'couple|couples',
 'Weddings':r'wedding|weddings|bridal|bride|brides|groom|grooms',
 'Mountains':r'mountain|mountains|mountainous', 'Forests':r'forest|forests|woodland|woodlands',
 'Lakes':r'lake|lakes', 'Sea':r'sea|ocean|oceans|seascape|seascapes',
 'Beaches':r'beach|beaches', 'Coast':r'coast|coastal|coastline|coastlines',
 'Clouds':r'cloud|clouds', 'Snow':r'snow|snowy', 'Grass':r'grass|grasses|grassy',
 'Foliage':r'foliage|leaf|leaves', 'Rocks':r'rock|rocks|rocky|boulder|boulders',
 'Gardens':r'garden|gardens', 'Fields':r'field|fields|meadow|meadows',
 'Desert':r'desert|deserts|dune|dunes', 'Rivers':r'river|rivers|stream|streams',
 'Waterfalls':r'waterfall|waterfalls', 'Hills':r'hill|hills|hillside|hillsides',
 'Streets':r'street|streets', 'Roads':r'road|roads', 'Bridges':r'bridge|bridges',
 'Cities':r'city|cities|cityscape|cityscapes', 'Architecture':r'architecture|architectural',
 'Interiors':r'interior|interiors|indoors|indoor', 'Rooms':r'room|rooms',
 'Cafes':r'cafe|cafes|café|cafés', 'Windows':r'window|windows',
 'Food':r'food|meal|meals|dish|dishes', 'Fruit':r'fruit|fruits',
 'Tables':r'table|tables', 'Furniture':r'furniture|chair|chairs|sofa|sofas',
 'Dogs':r'dog|dogs|puppy|puppies', 'Cats':r'cat|cats|kitten|kittens',
 'Birds':r'bird|birds|gull|gulls', 'Horses':r'horse|horses',
 'Animals':r'animal|animals|wildlife', 'Boats':r'boat|boats|sailboat|sailboats',
 'Cars':r'car|cars|vehicle|vehicles', 'Trains':r'train|trains|railway',
 'Bicycles':r'bicycle|bicycles|bike|bikes', 'Statues':r'statue|statues|sculpture|sculptures',
 'Temples':r'temple|temples', 'Churches':r'church|churches|cathedral|cathedrals',
 'Night':r'night|nighttime|night-time|nightscape|nightscapes',
 'Sunset':r'sunset|sunsets|dusk', 'Sunrise':r'sunrise|sunrises|dawn',
 'Still life':r'still[- ]life', 'Landscape':r'landscape|landscapes',
 'Macro':r'macro|close[- ]up', 'Mountaineering':r'climber|climbers|mountaineering',
}
INTENTS={
 'Warm tones':r'warm|warmer|warmth|warming', 'Cool tones':r'cool|cooler|cooling|coolness',
 'Contrast':r'contrast', 'Saturation':r'saturation|saturate|saturated',
 'Brightness':r'bright|brighter|brighten|brightening|brightness',
 'Shadows':r'shadow|shadows', 'Highlights':r'highlight|highlights',
 'Midtones':r'midtone|midtones|mid-tone|mid-tones',
 'Skin tones':r'skin|complexion', 'Depth':r'depth|dimensional|dimension',
 'Detail':r'detail|details', 'Texture':r'texture|textures',
 'Natural':r'natural|naturally|realistic', 'Softness':r'soft|softer|soften|softening',
 'Separation':r'separate|separation|stand out', 'Color balance':r'balance|balanced|rebalance',
 'Clarity':r'clear|clearer|clarity|crisp|crisper', 'Atmosphere':r'atmosphere|atmospheric|mood|moody',
 'Vibrance':r'vivid|vibrant|vibrance|vibrancy', 'Muted':r'muted|mute|subdued',
 'Cinematic':r'cinematic|cinema|filmic', 'Vintage':r'vintage|nostalgic|nostalgia|retro',
 'Glow':r'glow|glowing|luminous', 'Neutral':r'neutral|neutralize|neutralise',
 'Exposure':r'exposure|underexposed|overexposed', 'Clean look':r'clean|cleaner',
 'Richness':r'rich|richer|richness|enrich', 'Subject focus':r'focus|subject',
 'Gentleness':r'gentle|gently', 'Dramatic':r'dramatic|dramatically',
 'Colorfulness':r'colourful|colorful',
 'Depth of color':r'deep|deeper|deepen', 'Airy':r'airy',
}


def compile_lexicon(lexicon):
    return {name:re.compile(r'\b(?:'+pattern+r')\b',re.I) for name,pattern in lexicon.items()}


def extract():
    OUT.mkdir(exist_ok=True)
    plan=PLAN.read_bytes(); keys={r['key'] for r in json.loads(plan)['records']}
    canonical={}; found=set(); digest=hashlib.sha256()
    with RECORDS.open('rb') as stream:
        for line in stream:
            digest.update(line); r=json.loads(line); key=r['key']
            if key not in keys:
                continue
            if key in found:
                raise ValueError('Duplicate chain key')
            found.add(key)
            source=key.split('.rep')[0]
            if source in canonical and canonical[source]['key'] < key:
                continue
            answer=r['answer']
            canonical[source]=dict(key=key,
                observations=' '.join(c['observation'] for c in answer['cot']),
                instruction=answer['instruction_medium'])
    if found!=keys:
        raise ValueError(f'Incomplete corpus: {len(found)}/{len(keys)}')
    scene_rx, intent_rx=compile_lexicon(SCENES), compile_lexicon(INTENTS)
    scene, intent=Counter(),Counter(); traces=[]
    for source,row in sorted(canonical.items()):
        sw=[name for name,pattern in scene_rx.items() if pattern.search(row['observations'])]
        iw=[name for name,pattern in intent_rx.items() if pattern.search(row['instruction'])]
        scene.update(sw);intent.update(iw)
        traces.append(dict(source_id=source,canonical_key=row['key'],scene_terms=sw,intent_terms=iw))
    for name,counts in [('scene',scene),('intent',intent)]:
        with (OUT/(name+'_counts.csv')).open('w') as stream:
            writer=csv.writer(stream,lineterminator='\n');writer.writerow(['term','source_image_count'])
            writer.writerows(sorted(counts.items(),key=lambda x:(-x[1],x[0])))
    stats=dict(n_chains=len(found),n_source_images=len(canonical),
               source_selection='lexicographically smallest training-chain key for each source-image ID',
               scene_field='answer.cot[*].observation, combined over six moves',
               intent_field='answer.instruction_medium of the same canonical record',
               count='one presence per lexical family per source-image ID; overlapping terms allowed',
               scene_lexicon=SCENES,intent_lexicon=INTENTS,
               plan_sha256=hashlib.sha256(plan).hexdigest(),records_sha256=digest.hexdigest(),
               scene_matched_images=sum(bool(r['scene_terms']) for r in traces),
               scene_counts=dict(scene.most_common()),intent_counts=dict(intent.most_common()))
    (OUT/'statistics.json').write_text(json.dumps(stats,indent=2)+'\n')
    with (OUT/'counting_trace.jsonl').open('w') as stream:
        for row in traces:
            stream.write(json.dumps(row)+'\n')
    print(json.dumps(dict(n_chains=len(found),n_images=len(canonical),top_scene=scene.most_common(15))),flush=True)


def draw():
    sys.path.insert(0,'/tmp/codex-appendix-wordcloud')
    from wordcloud import WordCloud
    stats=json.loads((OUT/'statistics.json').read_text())
    width,height=960,780
    yy,xx=np.ogrid[:height,:width]
    # Softly rounded envelope rather than a hard rectangle or decorative icon.
    mask=np.where(((xx-width/2)/(width*.49))**4+((yy-height/2)/(height*.48))**4>1,255,0).astype(np.uint8)
    palette=['#276f95','#008879','#794ca7','#ad3b80','#a36520']
    def color(word,**kwargs):
        return palette[int(hashlib.sha256(word.encode()).hexdigest()[:6],16)%len(palette)]
    rendered=[]
    for kind,limit in [('scene',36),('intent',27)]:
        counts=dict(sorted(stats[kind+'_counts'].items(),key=lambda x:(-x[1],x[0]))[:limit])
        layouts=[]
        for maximum_font in [124,112,100,88]:
            candidate=WordCloud(width=width,height=height,mask=mask,background_color='white',
                 font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                 max_words=limit,min_font_size=41,max_font_size=maximum_font,
                 relative_scaling=.5,margin=6,prefer_horizontal=.92,random_state=19,
                 collocations=False,color_func=color,repeat=False).generate_from_frequencies(counts)
            layouts.append(candidate)
            if len(candidate.layout_)>=limit-2:
                break
        cloud=max(layouts,key=lambda c:len(c.layout_))
        cloud.to_file(str(OUT/(kind+'_wordcloud.png')))
        svg=cloud.to_svg(embed_font=False).replace('<svg ',f'<svg viewBox="0 0 {width} {height}" ',1)
        (OUT/(kind+'_wordcloud.svg')).write_text(svg)
        html=('<!doctype html><html><head><meta charset="utf-8"><style>'
              '@page{size:2.65in '+str(2.65*height/width)+'in;margin:0}'
              'html,body{margin:0;padding:0}svg{display:block;width:2.65in;height:auto}'
              '</style></head><body>'+svg+'</body></html>')
        source=OUT/(kind+'_wordcloud.html');source.write_text(html)
        pdf=OUT/(kind+'_wordcloud.pdf')
        with tempfile.TemporaryDirectory(prefix='scene-cloud-print-') as profile:
            subprocess.run(['google-chrome','--headless','--no-sandbox','--disable-gpu',
                            '--no-pdf-header-footer',f'--user-data-dir={profile}',
                            f'--print-to-pdf={pdf}',source.as_uri()],check=True,
                           stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=60)
        sizes=[item[1]*2.65*72/width for item in cloud.layout_]
        if min(sizes)<8:
            raise ValueError('Cloud text is too small at manuscript size')
        rendered.append(dict(kind=kind,shown=len(cloud.layout_),smallest_font_pt=min(sizes),
                             displayed=[dict(term=item[0][0],count=counts[item[0][0]]) for item in cloud.layout_]))
    (OUT/'render_audit.json').write_text(json.dumps(rendered,indent=2)+'\n')
    print(json.dumps(rendered),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['extract','draw'])
    globals()[p.parse_args().mode]()


if __name__=='__main__':
    main()
