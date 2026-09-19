"""Auditable appendix assets: training-language statistics, recovery, and inference.

Each exported paper image contains exactly one state, mask or plot. Layout is TeX.
No training data or checkpoints are changed.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import tomllib

import numpy as np
from PIL import Image

REPO=Path('/home/bc/VeraRetouch')
OUT=REPO/'EPR/ICLR2027/figures/appendix_qualitative_20260919'
WORK=Path('/home/bc/data/runs/paper_appendix_qualitative_20260919')
RECORDS=Path('/home/bc/data/runs/epr051_vlmsft/snap_sft2/records.jsonl')
PLAN=Path('/home/bc/data/runs/epr058_color/cache_prepare_ar1600_v1/plan_trainfull.json')
PROMPT=REPO/'experiments/prs/EPR-051_masked-restore-production/epr051_cot.toml'


def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def prepare():
    from veraretouch_sprf.data.cot_text import instruction_for
    sys.path.insert(0,'/tmp/codex-appendix-wordcloud')
    from wordcloud import WordCloud,STOPWORDS
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True)
    keys={r['key'] for r in json.loads(PLAN.read_text())['records']}
    sources={r['source_id']:r for r in csv.DictReader(open(REPO/'tools/data_splits/splits_sources.csv'))}
    stop=set(STOPWORDS)|set('image photo photograph current make please look give create scene across overall keep turn bring feel apply'.split())
    # Remove annotation-protocol scaffolding, not unwanted empirical outcomes.
    protocol_stop=set('mask masks grade grading finish toward towards use using used start starting globally global weighted weights weight broad presence specified controlled without keeping softly gently separately pass passes slightly followed following covering band bands core side frame roll-offs roll-off recipe recipes step steps first next finally then requested exact numbers parameter parameters stronger build building'.split())
    forms={'greens':'green','blues':'blue','reds':'red','cyans':'cyan','yellows':'yellow',
           'oranges':'orange','purples':'purple','magentas':'magenta','colours':'color',
           'colour':'color','colors':'color','colour-range':'color range','midtone':'midtones',
           'highlight':'highlights','shadow':'shadows','skin-tones':'skin tones',
           'tones':'tone','details':'detail','textures':'texture','lifting':'lift',
           'brightening':'brighten','enriching':'enrich','deepening':'deepen',
           'preserving':'preserve','cooler':'cool','warmer':'warm','richer':'rich'}
    counts=Counter();raw_counts=Counter();lengths=[];pool=Counter();tiers=Counter();found=set();annotations=[]
    candidates=[]
    for line in RECORDS.open():
        r=json.loads(line);key=r['key']
        if key not in keys:continue
        if key in found:raise ValueError('Duplicate training annotation')
        found.add(key);instruction,tier=instruction_for(r,key)
        words=re.findall(r"[a-z]+(?:-[a-z]+)?",instruction.lower())
        tokens=[w for w in words if w not in stop and len(w)>2]
        raw_counts.update(tokens)
        counts.update(forms.get(w,w) for w in tokens if w not in protocol_stop)
        lengths.append(len(words));tiers[tier]+=1
        source=key.split('.rep')[0];provenance=sources.get(source,{})
        pool[provenance.get('pool','unresolved')]+=1
        if provenance.get('pool')=='unsplash' and provenance.get('split')=='train':
            candidates.append(dict(key=key,source_id=source,annotation=r['answer'],
                                   annotation_prompt_sha=r.get('prompt_sha256'),instruction=instruction,
                                   pool='unsplash',split='train'))
    if found!=keys:raise ValueError(f'Missing {len(keys-found)} training annotations')
    # Fixed candidate sample, one chain per photograph, before inspecting recovery.
    candidates.sort(key=lambda r:hashlib.sha256(('appendix-v1:'+r['key']).encode()).hexdigest())
    seen=set();selected=[]
    for r in candidates:
        if r['source_id'] in seen:continue
        seen.add(r['source_id']);selected.append(r)
        if len(selected)==16:break
    dump(WORK/'chain_candidates.json',selected)
    stats=dict(n=len(found),source_counts=dict(pool),instruction_tiers=dict(tiers),
       instruction_word_count=dict(median=float(np.median(lengths)),p10=float(np.percentile(lengths,10)),
                                   p90=float(np.percentile(lengths,90))),
       counting='One deterministically selected instruction per trainfull chain; lowercase English tokens; no reasoning fields or repeated scaffold.',
       stopwords=sorted(stop),additional_protocol_stopwords=sorted(protocol_stop),
       word_form_merges=forms,cleaning_version='content-v2',
       top_words=counts.most_common(100),plan_sha256=sha(PLAN),
       records_sha256=sha(RECORDS),source_table_sha256=sha(REPO/'tools/data_splits/splits_sources.csv'))
    dump(OUT/'language_statistics.json',stats)
    with (OUT/'word_frequencies.csv').open('w') as f:
        writer=csv.writer(f,lineterminator='\n');writer.writerow(['word','count']);writer.writerows(counts.most_common())
    with (OUT/'word_frequencies_raw.csv').open('w') as f:
        writer=csv.writer(f,lineterminator='\n');writer.writerow(['word','count']);writer.writerows(raw_counts.most_common())
    wc=WordCloud(width=1500,height=850,background_color='white',max_words=65,
                 font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                 random_state=19,collocations=False,prefer_horizontal=1,
                 colormap='Dark2',relative_scaling=.55,margin=7,
                 max_font_size=150,min_font_size=18).generate_from_frequencies(counts)
    wc.to_file(str(OUT/'instruction_wordcloud.png'))
    (OUT/'instruction_wordcloud.svg').write_text(wc.to_svg(embed_font=False))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(3.25,2.8),layout='constrained')
    common=counts.most_common(12)[::-1]
    ax.barh([w for w,n in common],[n for w,n in common],color='#0072B2')
    ax.set_xlabel('Token occurrences')
    ax.spines[['top','right']].set_visible(False)
    fig.savefig(OUT/'instruction_frequency.pdf');plt.close(fig)
    cfg=tomllib.loads(PROMPT.read_text());prompt=cfg['prompt']
    for name in ['system','user_head']:
        (OUT/(name+'_prompt.txt')).write_text(prompt[name])
    dump(OUT/'annotation_protocol.json',dict(model='gpt-5.6-terra',
         system_sha256=hashlib.sha256(prompt['system'].encode()).hexdigest(),
         user_head_sha256=hashlib.sha256(prompt['user_head'].encode()).hexdigest(),
         prompt_source=str(PROMPT),images=['CURRENT: after','TARGET: before','2x7 breakdown sheet'],
         numeric_reconcile='diagnostic only; not an acceptance filter',
         student_instruction_mix={'long':70,'medium':20,'short':10},
         output_fields=['cot[6]: step, observation, mask, adjustment',
                        'instruction_long','instruction_medium','instruction_short']))
    print(json.dumps(dict(n=stats['n'],sources=pool,candidates=len(selected)),ensure_ascii=False),flush=True)


def image(path,x):
    import torch
    if isinstance(x,torch.Tensor):x=x.detach().cpu().numpy()
    Image.fromarray(np.rint(np.clip(x,0,1)*255).astype(np.uint8)).save(path)


def cloud():
    """Layout-only refresh from the counted corpus, avoiding another data scan."""
    sys.path.insert(0,'/tmp/codex-appendix-wordcloud')
    from wordcloud import WordCloud
    counts={r['word']:int(r['count']) for r in csv.DictReader((OUT/'word_frequencies.csv').open())}
    wc=WordCloud(width=1500,height=850,background_color='white',max_words=65,
        font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',random_state=19,
        collocations=False,prefer_horizontal=1,colormap='Dark2',relative_scaling=.55,
        margin=7,max_font_size=150,min_font_size=18).generate_from_frequencies(counts)
    wc.to_file(str(OUT/'instruction_wordcloud.png'))
    (OUT/'instruction_wordcloud.svg').write_text(wc.to_svg(embed_font=False))


def recover():
    import torch
    from veraretouch_sprf.readout import mixed_data as MX,multistage_data as MD,mixed_codes as MC
    from tools.epr059_glutbasis.g2_capacity import GlutBasis
    from veraretouch_sprf.data.stage_targets import LutVolumes
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.12)
    selected=json.loads((WORK/'chain_candidates.json').read_text())
    keys=[r['key'] for r in selected]
    seed=WORK/'subject_paths.json'
    if not seed.exists():shutil.copyfile(MX.ROOT/'stage2_chain/subject_paths.json',seed)
    payload,mapping,_=MX.MixedChainSource.collect_subjects(keys,cache_path=seed)
    source=MX.MixedChainSource(subject_payloads=payload,subject_mapping=mapping);source.setup()
    run=json.loads((MD.BK_RUN/'run_args.json').read_text())['config']
    bank=LutVolumes(run['data']['lut_bank_dir'],64);basis=GlutBasis(str(MD.GEOMETRY),'cuda:0')
    results=[]
    for i,r in enumerate(selected):
        law=source.chain(r['key']);entry=source.ds.index[r['key']];journal=source.ds.journal(r['key'])
        beta=law['beta'].to('cuda:0');states=MD.chain_states(law['x0'],beta,law['luts'],bank,'cuda:0')
        codes=[MC.solve_support_code(states[s+1],states[s],beta[s],basis.geometry) for s in range(6)]
        current=states[-1];replay=[current];errors=[]
        for slot in range(5,-1,-1):
            current=MC.apply_support(codes[slot],current,beta[slot],basis.geometry)
            replay.append(current);errors.append(float((current-states[slot]).abs().mean())*100)
        folder=WORK/f'chain_{i:02d}';folder.mkdir(exist_ok=True)
        h,w=law['hw']
        image(folder/'reference.png',states[0].reshape(h,w,3))
        for j,frame in enumerate(replay):image(folder/f'recovery_{j}.png',frame.reshape(h,w,3))
        for j,slot in enumerate(range(5,-1,-1),1):
            image(folder/f'mask_{j}.png',beta[slot].reshape(h,w))
            image(folder/f'recorded_{j}.png',states[slot].reshape(h,w,3))
        result=dict(**r,folder=str(folder),stage_order=[step['kind'] for step in journal['steps']][::-1],
             initial_l1=float((states[-1]-states[0]).abs().mean())*100,stage_l1=errors,
             final_l1=errors[-1],source_path=journal['source_path'],
             support='recorded strength-scaled beta; no second strength multiplication',
             execution='Codes fitted on recorded adjacent state pairs, then replayed sequentially on the current predicted state. Target-conditioned reconstruction, not VLM inference.',
             geometry_sha256=sha(MD.GEOMETRY),lut_ids=law['luts'])
        dump(folder/'provenance.json',result);results.append(result)
        dump(WORK/'chain_recovery_results.json',results)
        print(json.dumps(dict(i=i,key=r['key'],initial=result['initial_l1'],final=result['final_l1'])),flush=True)


def infer():
    import torch
    from veraretouch_sprf.readout import data as SD,epr071_data as E71,mixed_data as MX
    from tools.epr071_val50_diag import load_checkpoint,PROTOSET
    from veraretouch_sprf.readout import select_train as ST,artedit_eval as AE
    from q3vl.whatb.lutdata import LutBank
    from tools.epr059_glutbasis.common import BANK_DIR
    from veraretouch_sprf.data.cot_text import target_text
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.20)
    rows,codes,_,_=SD.load_sources([dict(name='style30k',root=str(SD.STYLE_ROOT),
                                         split='style_train',codes=str(SD.STYLE_CODES))])
    rows,override,_=E71.style_override(rows,journals=str(E71.STORE/'journals'))
    manifest=json.loads((SD.STYLE_ROOT/'style_train_plan.json').read_text())
    meta={r['key']:r for r in manifest['records']}
    sources={r['source_id']:r for r in csv.DictReader(open(REPO/'tools/data_splits/splits_sources.csv'))}
    rows=[r for r in rows if sources.get(meta[r['key']]['source_image_id'],{}).get('pool')=='unsplash']
    rows.sort(key=lambda r:hashlib.sha256(('appendix-style:'+r['key']).encode()).hexdigest())
    chosen=[];seen=set()
    for r in rows:
        source_id=meta[r['key']]['source_image_id']
        if source_id in seen:continue
        chosen.append(r);seen.add(source_id)
        if len(chosen)==48:break
    bank=ST.Bank('cuda:0',path=PROTOSET);model,facts=load_checkpoint(E71.STORE/'train/R/run/best.pt',bank,'cuda:0')
    model.model.eval();model.head.eval();mean,std,_=E71.load_scaler()
    mean=torch.as_tensor(mean,device='cuda:0');std=torch.as_tensor(std,device='cuda:0')
    ds=MX.SingleItems(chosen,model.processor,codes,model.readout_ids,override=override)
    renderer=AE.GlutRenderer(AE.GEOMETRY,'cuda:0');lut=LutBank(BANK_DIR,cache_size=64)
    results=[]
    with torch.no_grad():
        for start in range(0,len(ds),2):
            sb=MX.collate_mixed([ds[i] for i in range(start,min(start+2,len(ds)))])['single']
            _,q,_=model(sb['ids'],sb['images']);raw=q*std+mean
            for j,key in enumerate(sb['keys']):
                i=start+j;r=chosen[i];frame=sb['image_rgb'][j].to('cuda:0').float()/255
                z=frame.reshape(-1,3);gt=lut.apply(z,r['lut_id']).clamp(0,1)
                pred=renderer.apply(raw[j].reshape(3,-1),z).clamp(0,1)
                folder=WORK/f'infer_{i:02d}';folder.mkdir(exist_ok=True)
                for name,value in [('input',frame),('target',gt.reshape_as(frame)),('prediction',pred.reshape_as(frame))]:
                    image(folder/(name+'.png'),value)
                ov=override[key]
                result=dict(key=key,source_id=meta[key]['source_image_id'],pool='unsplash',split='train',
                    folder=str(folder),instruction=ov['request'],lut_id=r['lut_id'],
                    checkpoint=facts,reasoning=model.tokenizer.decode(ov['cot_ids'],skip_special_tokens=False),
                    reasoning_source='cached base-SFT generated description; model readout inferred anew; target not provided',
                    l1=float((pred-gt).abs().mean())*100,identity_l1=float((z-gt).abs().mean())*100,
                    source_index=meta[key]['source_index_record'])
                dump(folder/'provenance.json',result);results.append(result)
                dump(WORK/'style_inference_results.json',results)
                print(json.dumps(dict(i=i,key=key,l1=result['l1'],identity=result['identity_l1'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','recover','infer','cloud'])
    globals()[p.parse_args().mode]()
