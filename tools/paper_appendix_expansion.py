"""Prepare additional in-domain outputs and measurable local-change assets."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3

import numpy as np
from PIL import Image

from tools.paper_appendix_qualitative import REPO, WORK as OLD, dump, image

WORK = Path('/home/bc/data/runs/paper_appendix_expansion_20260919')


def preview():
    from PIL import ImageDraw, ImageFont
    records = json.loads((WORK/'inference_results.json').read_text())
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    for pool in ['fivek_gold', 'ppr10k']:
        rows = [r for r in records if r['pool'] == pool and r['l1'] < .85*r['identity_l1']]
        rows.sort(key=lambda r:r['l1'])
        canvas = Image.new('RGB', (1200, ((len(rows)+1)//2)*200), 'white'); draw = ImageDraw.Draw(canvas)
        for i, r in enumerate(rows):
            folder = Path(r['folder']); x=(i%2)*600; y=(i//2)*200
            draw.text((x+2,y+2), f'{folder.name}: {r["identity_l1"]:.2f} -> {r["l1"]:.2f} (internal)', font=font, fill='black')
            for j, name in enumerate(['input','prediction','target']):
                frame=Image.open(folder/(name+'.png')); frame.thumbnail((195,170))
                canvas.paste(frame,(x+j*200+(195-frame.width)//2,y+23))
        canvas.save(WORK/(pool+'_internal_preview.png'))


def portrait_preview():
    """Internal selection sheet only; never a manuscript asset."""
    from PIL import ImageDraw, ImageFont
    records=json.loads((WORK/'local_results.json').read_text())
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    canvas=Image.new('RGB',(1600,((len(records)+3)//4)*260),'white')
    draw=ImageDraw.Draw(canvas)
    for i,r in enumerate(records):
        x=(i%4)*400;y=(i//4)*260;folder=Path(r['folder'])
        size=Image.open(folder/'reference.png').size
        draw.text((x+3,y+3),f'{i:02d} {size} local {r["local_amplitude"]:.2f}',fill='black',font=font)
        for j,s in enumerate([5,6]):
            frame=Image.open(folder/f'recovery_{s}.png');frame.thumbnail((195,230))
            canvas.paste(frame,(x+j*200+(195-frame.width)//2,y+25))
    canvas.save(WORK/'portrait_selection_internal.png')


def choose_local():
    from tools.paper_appendix_qualitative import PLAN, RECORDS
    sources = {r['source_id']: r for r in csv.DictReader(open(REPO/'tools/data_splits/splits_sources.csv'))}
    keys = {r['key'] for r in json.loads(PLAN.read_text())['records']}
    old_ids = {r['source_id'] for r in json.loads((OLD/'chain_candidates.json').read_text())}
    candidates = []
    for line in RECORDS.open():
        row = json.loads(line); key = row['key']; source = key.split('.rep')[0]
        if key not in keys or source in old_ids or sources.get(source, {}).get('pool') != 'unsplash':
            continue
        annotation = row['answer']; last = annotation['cot'][-1]
        if not re.search(r'segment|main.subject', last['mask'], re.I):
            continue
        values = [float(s) for s in re.findall(r'(?<![A-Za-z])0\.\d+', last['mask'])]
        if not values or not .08 < max(values) < .45:
            continue
        candidates.append(dict(key=key, source_id=source, annotation=annotation,
                               instruction=annotation['instruction_medium'], pool='unsplash', split='train'))
    candidates.sort(key=lambda r: hashlib.sha256(('local-visibility-v2:'+r['key']).encode()).hexdigest())
    seen = set(); selected = []
    for row in candidates:
        if row['source_id'] in seen:
            continue
        selected.append(row); seen.add(row['source_id'])
        if len(selected) == 24:
            break
    dump(WORK/'local_candidates.json', selected)
    print(json.dumps(dict(eligible=len(candidates), chosen=len(selected))), flush=True)


def choose_landscape():
    from tools.paper_appendix_qualitative import RECORDS
    scene_file=REPO/'EPR/ICLR2027/figures/appendix_scene_words/counting_trace.jsonl'
    sources={r['source_id']:r for r in csv.DictReader(open(REPO/'tools/data_splits/splits_sources.csv'))}
    eligible=[]
    for line in scene_file.open():
        row=json.loads(line); terms=set(row['scene_terms'])
        if sources.get(row['source_id'],{}).get('pool')!='unsplash':
            continue
        if not {'Sky','Clouds'}.issubset(terms):
            continue
        if not terms & {'Mountains','Lakes','Sea','Coast','Landscape','Hills','Rocks','Fields'}:
            continue
        eligible.append(row)
    eligible.sort(key=lambda r:hashlib.sha256(('hue-landscape-v1:'+r['canonical_key']).encode()).hexdigest())
    selected_keys={r['canonical_key'] for r in eligible[:40]}
    selected=[]
    for line in RECORDS.open():
        row=json.loads(line)
        if row['key'] not in selected_keys:
            continue
        answer=row['answer']
        selected.append(dict(key=row['key'],source_id=row['key'].split('.rep')[0],annotation=answer,
                             instruction=answer['instruction_medium'],pool='unsplash',split='train'))
    selected.sort(key=lambda r:hashlib.sha256(('hue-landscape-v1:'+r['key']).encode()).hexdigest())
    if len(selected)!=len(selected_keys):
        raise ValueError('Missing landscape annotations')
    dump(WORK/'local_candidates.json',selected)
    print(json.dumps(dict(eligible=len(eligible),chosen=len(selected))),flush=True)


def choose_portraits(count=24):
    """Deterministic training-source shortlist; visual selection follows recovery."""
    from tools.paper_appendix_qualitative import RECORDS, PLAN
    sources = {r['source_id']: r for r in csv.DictReader(open(REPO/'tools/data_splits/splits_sources.csv'))}
    train_keys = {r['key'] for r in json.loads(PLAN.read_text())['records']}
    trace = REPO/'EPR/ICLR2027/figures/appendix_scene_words/counting_trace.jsonl'
    eligible = set()
    for line in trace.open():
        row = json.loads(line); terms = set(row['scene_terms'])
        if sources.get(row['source_id'], {}).get('pool') not in {'unsplash', 'ppr10k', 'fivek_gold'}:
            continue
        if terms & {'Portraits', 'Women', 'Men', 'Children'} and not terms & {'Dogs', 'Cats', 'Animals'}:
            eligible.add(row['source_id'])
    candidates = []
    for line in RECORDS.open():
        row = json.loads(line); key = row['key']; source_id = key.split('.rep')[0]
        if source_id not in eligible or key not in train_keys:
            continue
        answer = row['answer']; last = answer['cot'][-1]
        if not re.search(r'segment|main.subject', last['mask'], re.I):
            continue
        if not re.search(r'face|skin|woman|man\b|girl|boy|child|person', last['observation'], re.I):
            continue
        values = [float(s) for s in re.findall(r'(?<![A-Za-z])0\.\d+', last['mask'])]
        if not values or not .10 < max(values) < .65:
            continue
        candidates.append(dict(key=key, source_id=source_id, annotation=answer,
                               instruction=answer['instruction_medium'],
                               pool=sources[source_id]['pool'], split='train'))
    candidates.sort(key=lambda r: hashlib.sha256(('portrait-local-v1:'+r['key']).encode()).hexdigest())
    selected = []; seen = set()
    for row in candidates:
        if row['source_id'] in seen:
            continue
        selected.append(row); seen.add(row['source_id'])
        if len(selected) == count:
            break
    dump(WORK/'local_candidates.json', selected)
    print(json.dumps(dict(eligible=len(candidates), chosen=len(selected))), flush=True)


def recover():
    import torch
    from veraretouch_sprf.readout import mixed_data as MX, multistage_data as MD, mixed_codes as MC
    from tools.epr059_glutbasis.g2_capacity import GlutBasis
    from veraretouch_sprf.data.stage_targets import LutVolumes
    device = 'cuda:0'; torch.cuda.set_device(device); torch.set_num_threads(3)
    torch.cuda.set_per_process_memory_fraction(.08, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    selected = json.loads((WORK/'local_candidates.json').read_text())
    seed = WORK/'subject_paths.json'
    if not seed.exists():
        shutil.copyfile(OLD/'subject_paths.json', seed)
    payload, mapping, _ = MX.MixedChainSource.collect_subjects([r['key'] for r in selected], cache_path=seed)
    source = MX.MixedChainSource(subject_payloads=payload, subject_mapping=mapping); source.setup()
    run = json.loads((MD.BK_RUN/'run_args.json').read_text())['config']
    bank = LutVolumes(run['data']['lut_bank_dir'], 64)
    basis = GlutBasis(str(MD.GEOMETRY), device)
    results_path=WORK/'local_results.json'
    output=json.loads(results_path.read_text()) if results_path.exists() else []
    for i,old in enumerate(output):
        if i>=len(selected) or old['key']!=selected[i]['key']:
            raise ValueError('Refusing to renumber previously rendered recovery cases')
    for i, row in enumerate(selected):
        if i<len(output):continue
        law = source.chain(row['key']); journal = source.ds.journal(row['key'])
        beta = law['beta'].to(device)
        states = MD.chain_states(law['x0'], beta, law['luts'], bank, device)
        codes = [MC.solve_support_code(states[s+1], states[s], beta[s], basis.geometry) for s in range(6)]
        current = states[-1]; replay = [current]; errors = []
        for slot in range(5, -1, -1):
            current = MC.apply_support(codes[slot], current, beta[slot], basis.geometry)
            replay.append(current); errors.append(float((current-states[slot]).abs().mean())*100)
        folder = WORK/f'local_{i:02d}'; folder.mkdir(exist_ok=True)
        h, w = law['hw']
        image(folder/'reference.png', states[0].reshape(h, w, 3))
        for j, frame in enumerate(replay):
            image(folder/f'recovery_{j}.png', frame.reshape(h, w, 3))
        for j, slot in enumerate(range(5, -1, -1), 1):
            image(folder/f'mask_{j}.png', beta[slot].reshape(h, w))
            image(folder/f'recorded_{j}.png', states[slot].reshape(h, w, 3))
        np.savez_compressed(folder/'float_states.npz',
                            recovery=torch.stack(replay).cpu().numpy().reshape(7, h, w, 3),
                            masks=beta.flip(0).cpu().numpy().reshape(6, h, w))
        delta = (replay[6]-replay[5]).abs().mean(-1)
        row = dict(**row, folder=str(folder), stage_order=[s['kind'] for s in journal['steps']][::-1],
                   initial_l1=float((states[-1]-states[0]).abs().mean())*100,
                   stage_l1=errors, final_l1=errors[-1],
                   local_amplitude=float((delta*beta[0]).sum()/beta[0].sum())*100,
                   local_area=float((beta[0]>.01).float().mean()),
                   execution='Target-conditioned adjacent-state code fitting and sequential replay; not model inference.')
        dump(folder/'provenance.json', row); output.append(row); dump(WORK/'local_results.json', output)
        print(json.dumps(dict(case=folder.name, final=row['final_l1'], local=row['local_amplitude'], area=row['local_area'])), flush=True)


def infer():
    import torch
    from veraretouch_sprf.readout import data as SD, epr071_data as E71, mixed_data as MX
    from tools.epr071_val50_diag import load_checkpoint, PROTOSET
    from veraretouch_sprf.readout import select_train as ST, artedit_eval as AE
    from q3vl.whatb.lutdata import LutBank
    from tools.epr059_glutbasis.common import BANK_DIR
    device = 'cuda:0'
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.20, device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows, codes, _, _ = SD.load_sources([dict(name='style30k', root=str(SD.STYLE_ROOT),
                                             split='style_train', codes=str(SD.STYLE_CODES))])
    rows, override, _ = E71.style_override(rows, journals=str(E71.STORE/'journals'))
    meta = {r['key']: r for r in json.loads((SD.STYLE_ROOT/'style_train_plan.json').read_text())['records']}
    sources = {r['source_id']: r for r in csv.DictReader(open(REPO/'tools/data_splits/splits_sources.csv'))}
    selected = []
    for pool in ['fivek_gold', 'ppr10k']:
        candidates = [r for r in rows if sources.get(meta[r['key']]['source_image_id'], {}).get('pool') == pool]
        candidates.sort(key=lambda r: hashlib.sha256(('appendix-expand:'+r['key']).encode()).hexdigest())
        seen = set()
        for r in candidates:
            source = meta[r['key']]['source_image_id']
            if source in seen:
                continue
            seen.add(source)
            selected.append((pool, r))
            if len(seen) == 20:
                break
        print(pool, len(seen), flush=True)
    WORK.mkdir(exist_ok=True)
    bank = ST.Bank(device, path=PROTOSET)
    model, facts = load_checkpoint(E71.STORE/'train/R/run/best.pt', bank, device)
    model.model.eval(); model.head.eval()
    mean, std, _ = E71.load_scaler()
    mean = torch.as_tensor(mean, device=device); std = torch.as_tensor(std, device=device)
    ds = MX.SingleItems([r for _, r in selected], model.processor, codes, model.readout_ids, override=override)
    renderer = AE.GlutRenderer(AE.GEOMETRY, device)
    lut = LutBank(BANK_DIR, cache_size=64)
    output = []
    with torch.no_grad():
        for i, (pool, r) in enumerate(selected):
            batch = MX.collate_mixed([ds[i]])['single']
            _, q, _ = model(batch['ids'], batch['images'])
            raw = q[0]*std+mean
            rgb = batch['image_rgb'][0].to(device).float()/255
            z = rgb.reshape(-1, 3)
            gt = lut.apply(z, r['lut_id']).clamp(0, 1)
            pred = renderer.apply(raw.reshape(3, -1), z).clamp(0, 1)
            folder = WORK/f'{pool}_{i:02d}'; folder.mkdir(exist_ok=True)
            for name, value in [('input', rgb), ('target', gt.reshape_as(rgb)), ('prediction', pred.reshape_as(rgb))]:
                image(folder/(name+'.png'), value)
            np.save(folder/'raw_code.npy', raw.cpu().numpy())
            record = dict(key=r['key'], source_id=meta[r['key']]['source_image_id'], pool=pool,
                          split='train', task='synthetic style on source-dataset photographs',
                          target_type='LUT-rendered style target, not expert-retouched benchmark target',
                          folder=str(folder), instruction=override[r['key']]['request'],
                          reasoning=model.tokenizer.decode(override[r['key']]['cot_ids'], skip_special_tokens=False),
                          reasoning_source='cached base-SFT description; newly executed continuous readout',
                          checkpoint=facts, l1=float((pred-gt).abs().mean())*100,
                          identity_l1=float((z-gt).abs().mean())*100,
                          source_index=meta[r['key']]['source_index_record'])
            dump(folder/'provenance.json', record); output.append(record)
            dump(WORK/'inference_results.json', output)
            print(json.dumps(dict(case=folder.name, l1=record['l1'], initial=record['identity_l1'])), flush=True)


def main():
    global WORK
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['infer', 'choose_local', 'choose_landscape', 'choose_portraits', 'recover', 'preview', 'portrait_preview'])
    p.add_argument('--work',type=Path,default=WORK)
    p.add_argument('--count',type=int,default=24)
    args=p.parse_args();WORK=args.work;WORK.mkdir(exist_ok=True)
    if args.mode=='choose_portraits':choose_portraits(args.count)
    else:globals()[args.mode]()


if __name__ == '__main__':
    main()
