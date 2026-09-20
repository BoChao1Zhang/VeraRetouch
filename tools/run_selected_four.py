"""Fresh target-free inference for the four author-selected qualitative cases.

Only prepare() reads the gallery manifest. Inference phases consume input images
and the same natural-language instruction; no reference, cached CoT or fit code.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import time
import numpy as np
from PIL import Image

REPO=Path('/home/bc/VeraRetouch')
WORK=Path('/home/bc/data/runs/paper_selected_four_pix3200_20260920')
CKPT=Path('/home/bc/data/runs/epr072_local_continuation_20260919/final_step3200/PIX_step3200.pt')
IDS=[134,167,294,364]
USER_SELECTION=Path('/home/bc/.codex/attachments/3697b8f0-a2e4-4798-b689-09eacd9e3e41/local-selected (2).json')


def dump(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.partial');temp.write_text(json.dumps(data,indent=2,default=str)+'\n');temp.replace(path)


def prepare():
    gallery=json.loads((REPO/'outputs/local_review500_20260920/combined_results.json').read_text())
    by_id={r['number']:r for r in gallery}
    selected=json.loads(USER_SELECTION.read_text())
    for item in selected['selected']:
        actual=by_id[int(item['number'])]
        assert item['key']==actual['key'] and item['source_id']==actual['source_id']
    dump(WORK/'user_selection.json',selected)
    records=[]
    for n in IDS:
        r=by_id[n];folder=WORK/f'case_{n:03d}';folder.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(Path(r['folder'])/'recovery_0.png',folder/'input.png')
        shutil.copyfile(Path(r['folder'])/'reference.png',folder/'reference.png')
        records.append(dict(id=f'selected_{n:03d}',sample_id=f'selected_{n:03d}',number=n,source_id=r['source_id'],
                            key=r['key'],input_png=str(folder/'input.png'),image=str(folder/'input.png'),
                            folder=str(folder),size=r['image_size'],instruction=r['instruction'],
                            instruction_short=r['annotation']['instruction_short'],lane='local',
                            input_sha256=hashlib.sha256((folder/'input.png').read_bytes()).hexdigest()))
    dump(WORK/'inputs.json',dict(records=records,rows=records,checkpoint=str(CKPT),
         input_protocol='Original degraded z0 and full instruction shared by every method; target is not an inference input.'))
    jobs=[]
    for r in records:
        h,w=r['size'][1],r['size'][0];sid=r['id']
        common=dict(sample_id=sid,bench='selected_train',arm='instr_real',raw_path=r['image'],input_path=r['image'],
                    gt_path=None,instruction=r['instruction'],instruction_field='instruction',expect_hw=[h,w],raw_h=h,raw_w=w)
        jobs.append(dict(common,model='jarvisevo',out_json=str(WORK/'jarvisevo_params'/f'{sid}.json')))
    (WORK/'jarvisevo_jobs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in jobs))
    for r in jobs:
        sid=r['sample_id'];r.update(model='monetgpt',style='balanced',out_tif=str(WORK/'monetgpt'/f'{sid}.tif'),
                                  out_txt=str(WORK/'monetgpt'/f'{sid}.txt'),out_json=str(WORK/'monetgpt'/f'{sid}.json'))
    (WORK/'monetgpt_jobs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in jobs))
    print(json.dumps(dict(selected=len(selected['selected']),run_ids=IDS,work=str(WORK))),flush=True)


def setup():
    import torch
    torch.set_num_threads(4);torch.manual_seed(20260920);torch.cuda.set_per_process_memory_fraction(.28)
    torch.backends.cuda.matmul.allow_tf32=False
    return json.loads((WORK/'inputs.json').read_text())['records']


def where():
    rows=setup()
    from veraretouch_sprf.e2e.whereq import SubjQPredictor
    predictor=SubjQPredictor(device='cuda:0')
    report=[]
    for r in rows:
        start=time.monotonic();pred=predictor.predict(Image.open(r['image']).convert('RGB'),r['instruction'])
        soft=pred['soft'].detach().cpu().numpy().astype(np.float32)
        np.save(Path(r['folder'])/'predicted_subject.npy',soft)
        Image.fromarray(np.rint(soft*255).astype(np.uint8)).save(Path(r['folder'])/'predicted_subject.png')
        row=dict(sample_id=r['id'],seconds=time.monotonic()-start,
                 facts={k:v for k,v in pred.items() if k not in {'soft','hard'}},mask_mean=float(soft.mean()))
        report.append(row);dump(WORK/'where.json',dict(model=predictor.facts(),target_access=False,rows=report));print(r['id'],flush=True)


def cot():
    rows=setup()
    from veraretouch_sprf.readout.model import ReadoutVLM,AR_ADAPTER
    from tools.epr075_fig4.cot import generate,IM_END
    holder=ReadoutVLM(device='cuda:0',grad_checkpointing=False)
    holder.model.base_model.set_adapter([AR_ADAPTER]);holder.model.active_adapter=AR_ADAPTER;holder.model.eval()
    generated=generate(holder.model,holder.processor,holder.pad_id,rows,lambda r:r['instruction'],'',2048,(IM_END,),'cuda:0',batch=1)
    for r in generated:r['text']=holder.tokenizer.decode(r['token_ids'],skip_special_tokens=False)
    dump(WORK/'fresh_cot.json',dict(engine='AR1600 generation, input image and instruction only',target_access=False,rows=generated))


def ours(no_reasoning=False):
    rows=setup()
    import torch
    from tools.epr071_val50_diag import load_checkpoint,PROTOSET
    from veraretouch_sprf.readout import select_train as ST,epr071_data as E71,artedit_eval as AE,multistage_data as MD
    from veraretouch_sprf.readout.epr072_local_train import forward_chain
    from veraretouch_sprf.readout.multistage_loss import ChainPixelL1
    from veraretouch_sprf.data import q3vl_text as T
    from veraretouch_sprf.models.vlm import q3vl_common as Q
    from veraretouch_sprf.e2e.masks import state_alphas_device
    from tools.epr075_fig4.render_ours import sequence_six
    bank=ST.Bank('cuda:0',path=PROTOSET);model,facts=load_checkpoint(CKPT,bank,'cuda:0')
    model.model.eval();model.head.eval()
    mean,std,_=E71.load_scaler();mean=torch.as_tensor(mean,device='cuda:0');std=torch.as_tensor(std,device='cuda:0')
    pix=ChainPixelL1(MD.GEOMETRY,'cuda:0',chunk=196608,stride=1)
    journals={} if no_reasoning else {r['sample_id']:r for r in json.loads((WORK/'fresh_cot.json').read_text())['rows']}
    prefix='ours_no_reasoning' if no_reasoning else 'ours'
    report=[]
    for r in rows:
        folder=Path(r['folder']);pil=Image.open(r['image']).convert('RGB');w,h=pil.size
        small,geometry=Q.prepare_image_spec5(pil);enc=T.encode_prompt(model.processor,small,r['instruction'])
        prompt=enc['input_ids'][0].tolist()
        if no_reasoning:
            segments=[[int(token)] for token in model.stage_ids]
            journal=dict(token_ids=list(model.stage_ids))
        else:
            journal=journals[r['id']];assert prompt==journal['prompt_token_ids']
            segments,reason=AE.segments_of(journal,model.stage_ids)
            if segments is None:raise ValueError(f'{r["id"]}: invalid fresh reasoning: {reason}')
        ids,groups=sequence_six(prompt,segments,model.readout_ids)
        images=[dict(pixel_values=enc['pixel_values'],image_grid_thw=enc['image_grid_thw'])]
        with torch.no_grad():codes=(forward_chain(model,[ids],[groups],images)*std+mean)[0]
        rgb=np.asarray(pil,dtype=np.float32)/255;current=torch.as_tensor(rgb.reshape(-1,3),device='cuda:0')
        subject=torch.as_tensor(np.load(folder/'predicted_subject.npy'),device='cuda:0')
        with torch.no_grad():
            for k,slot in enumerate(range(5,-1,-1),1):
                fields,_=state_alphas_device(current.reshape(h,w,3),subject,info=False,validate=False)
                current=pix.apply_code(codes[slot].reshape(3,-1),current,fields[slot].reshape(-1))
                Image.fromarray(np.rint(current.reshape(h,w,3).clamp(0,1).cpu().numpy()*255).astype(np.uint8)).save(folder/f'{prefix}_stage{k}.png')
        shutil.copyfile(folder/f'{prefix}_stage6.png',folder/f'{prefix}.png')
        np.save(folder/f'{prefix}_predicted_codes.npy',codes.detach().cpu().numpy())
        report.append(dict(id=r['id'],ok=True,input_sha256=r['input_sha256'],cot_tokens=len(journal['token_ids'])))
        report[-1]['stage_reasoning_text_tokens']=0 if no_reasoning else len(journal['token_ids'])-7
        report[-1]['stage_markers']=6
        dump(WORK/f'{prefix}.json',dict(checkpoint=facts,target_access=False,reasoning=('stage markers only, no generated stage reasoning' if no_reasoning else 'fresh AR1600'),
             support='same input-predicted SUBJQ mask; photometric masks recomputed on current states',rows=report));print(r['id'],flush=True)


def ours_no_reasoning():
    ours(no_reasoning=True)


def vera():
    rows=setup()
    from tools.appendix_latency_benchmark import load_vera
    call,facts=load_vera();report=[]
    for r in rows:
        pred,extra=call(r)
        if [pred.shape[1],pred.shape[0]]!=r['size']:raise ValueError('VeraRetouch output size mismatch')
        Image.fromarray(pred).save(Path(r['folder'])/'veraretouch.png')
        report.append(dict(id=r['id'],ok=True,**extra));dump(WORK/'veraretouch.json',dict(config=facts,rows=report));print(r['id'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','where','cot','ours','ours_no_reasoning','vera']);a=p.parse_args();globals()[a.phase]()
