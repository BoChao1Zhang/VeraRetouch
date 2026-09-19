"""EPR072LOCAL: matched pixel-only / pixel+InfoNCE six-stage continuation.

Keeps EPR-071 R's base, all-layer LoRA, eight shared readout tokens and scaler.
Replay batches use style30k + all MMArt. Local batches use every valid chain.
No code-MSE in either arm. No benchmark oracle examples enter training.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from veraretouch_sprf.readout import data as SD, mixed_data as MX
from veraretouch_sprf.readout import multistage_data as MD, epr071_data as E71
from veraretouch_sprf.readout import select_train as ST, mixed_codes as MC
from veraretouch_sprf.readout.multistage_loss import ChainPixelL1, infonce
from veraretouch_sprf.readout.select_model import SEL_ADAPTER
from tools.epr071_val50_diag import load_checkpoint, JOURNAL, HALVES
from veraretouch_sprf.e2e.masks import state_alphas
from veraretouch_sprf.readout.epr072_local_io import (
    select_style_shards,ResidentFiles,CachedStyle,install_chain_residency)

ROOT=Path('/home/bc/data/runs/epr072_local_continuation_20260919')
PPR=Path('/home/bc/nfsvfs/bc/data/runs/epr072_mmart_ppr10k_20260919')
STYLE_SPEC=MX.STAGE1_SOURCES


def emit(event,**kw):
    print(json.dumps(dict(event=event,time=time.time(),**kw),default=str),flush=True)


def write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value,indent=2,default=str)+'\n'); tmp.replace(path)


def digest_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()


def forward_chain(model,sequences,groups,images):
    """Shared readout IDs can recur; positions, not ID, identify each stage."""
    hidden,_=model.hidden(sequences,images,autocast=True)
    index=torch.tensor(groups,device=hidden.device)
    if index.shape[1:]!=(6,model.k): raise ValueError(f'Invalid stage groups {index.shape}')
    for seq,positions in zip(sequences,groups):
        for pos in positions:
            if [seq[p] for p in pos]!=model.readout_ids: raise ValueError('readout position drift')
    gather=index.reshape(len(sequences),6*model.k,1).expand(-1,-1,hidden.shape[-1])
    u=hidden.gather(1,gather).reshape(len(sequences)*6,-1)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        q=model.head.residual(model.head.norm(u))
    return q.float().reshape(len(sequences),6,-1).flip(1)


def local_nce(pred,target,normalized_bank,temperature=.07):
    """Recovered inverse-action code is positive; fixed bank provides negatives."""
    q=F.normalize(pred.reshape(1,-1),dim=-1)
    t=F.normalize(target.detach().reshape(1,-1),dim=-1)
    positive=(q*t).sum(-1,keepdim=True)/temperature
    negatives=q@normalized_bank.T/temperature
    duplicates=(t@normalized_bank.T)>1-1e-6
    negatives=negatives.masked_fill(duplicates,float('-inf'))
    return F.cross_entropy(torch.cat((positive,negatives),1),
                           torch.zeros(1,dtype=torch.long,device=q.device))


def current_supports(states,recorded_beta,strength,hw):
    """Current photometric masks; recorded geometric/subject support as teacher.

    Geometry does not move under color edits. The subject/where predictor is
    not trained in this controlled color-readout experiment.
    """
    subject=(recorded_beta[0]/strength).clamp(0,1).reshape(*hw) if strength>0 else \
        torch.zeros(hw,device=states.device)
    fields=[]
    for slot in range(6):
        masks,_=state_alphas(states[slot+1].reshape(*hw,3),subject)
        fields.append(masks[slot].to(states.device).reshape(-1))
    return torch.stack(fields)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--arm',choices=['PIX','NCE'],required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--start',default=str(E71.STORE/'train/R/run/best.pt'))
    ap.add_argument('--style-n',type=int,default=15000)
    ap.add_argument('--total-steps',type=int,default=6250)
    ap.add_argument('--micro',type=int,default=4)
    ap.add_argument('--batch',type=int,default=16)
    ap.add_argument('--seed',type=int,default=20260918)
    ap.add_argument('--eval-every',type=int,default=400)
    ap.add_argument('--smoke',type=int,default=0)
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    run=out/'run'; run.mkdir(exist_ok=True)
    if (run/'steps.jsonl').exists(): raise SystemExit('Run already exists: use a fresh --out')
    device='cuda:0'; torch.set_num_threads(4)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.80)
    if args.batch%args.micro: raise ValueError('batch must be divisible by micro')
    accum=args.batch//args.micro; weight_nce=.1 if args.arm=='NCE' else 0.
    steps=args.smoke or args.total_steps
    begun=time.monotonic()
    snapshot=out/'frozen_sources'; snapshot.mkdir(exist_ok=True)
    files=[Path(__file__),Path(ST.__file__),Path(MC.__file__),Path(MX.__file__),
           Path(MD.__file__),Path(E71.__file__)]
    source_hashes={str(p):digest_file(p) for p in files}
    for i,p in enumerate(files): shutil.copyfile(p,snapshot/f'{i}_{p.name}')
    write(out/'source_manifest.json',dict(argv=os.sys.argv,hashes=source_hashes))
    emit('start',arm=args.arm,device=device,visible=os.environ.get('CUDA_VISIBLE_DEVICES'))

    # Fix both streams independently of model RNG, and include every valid chain.
    spec=json.loads(Path(STYLE_SPEC).read_text())['sources']
    spec=[s for s in spec if s['name']=='style30k']
    if len(spec)!=1: raise ValueError('Expected exactly one style30k source')
    style_rows,style_codes,style_facts,_=SD.load_sources(spec)
    eligible,override,rewrite_facts=E71.style_override(style_rows,journals=str(E71.STORE/'journals'))
    if len(eligible)<args.style_n:
        raise ValueError(f'Only {len(eligible)} rewritten style30k records for {args.style_n}')
    style_rows,style_shards,selected_shards=select_style_shards(
        eligible,E71.STORE/'journals',args.style_n,args.seed)
    override={r['key']:override[r['key']] for r in style_rows}
    del eligible
    emit('style_ready',n=len(style_rows),selected_shards=selected_shards)
    plan=MD.read_stage2_manifest_multi(str(MX.CHAIN_MANIFEST),splits=['trainfull','heldout'])
    plan['records']=[r for r in plan['records'] if r['format_valid']]
    plan['n']=len(plan['records'])
    for i,r in enumerate(plan['records']): r['seq']=i
    valplan=MD.build_chain_plan('val',limit=16)
    for r in valplan['records']: r['split']='val'
    keys=[r['key'] for r in plan['records']]+[r['key'] for r in valplan['records']]
    cache=out/'subject_paths.json'
    seed_cache=MX.ROOT/'stage2_chain/subject_paths.json'
    if seed_cache.exists(): shutil.copyfile(seed_cache,cache)
    payloads,mapping,subject_facts=MX.MixedChainSource.collect_subjects(keys,cache_path=cache)
    source=MX.MixedChainSource(subject_payloads=payloads,subject_mapping=mapping); source.setup()
    resident=ResidentFiles(budget_gib=12)
    install_chain_residency(source,resident)
    instructions={}
    for split in ['trainfull','heldout','val']:
        splitkeys=[r['key'] for r in plan['records']+valplan['records'] if r.get('split','trainfull')==split]
        if splitkeys: instructions.update(MD.load_instructions(splitkeys,split))
    emit('chains_ready',n=plan['n'],subjects=subject_facts)
    bank=ST.Bank(device,path=E71.PROTOSET)
    mean,std,scaler_step=E71.load_scaler()
    mean_t=torch.from_numpy(mean).to(device); std_t=torch.from_numpy(std).to(device)
    bank_std=(bank.codes.reshape(len(bank.ids),-1)-mean_t)/std_t
    norm_bank=F.normalize(bank_std,dim=-1)
    model,initial=load_checkpoint(Path(args.start),bank,device)
    if initial['arm']!='R' or initial['k_readout']!=8: raise ValueError('Expected stage1 R k8')
    model.model.base_model.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={'use_reentrant':False})
    model.model.base_model.model.enable_input_require_grads()
    # The unused prototype classification head remains frozen in both R arms.
    for p in model.head.select.parameters(): p.requires_grad_(False)
    groups=[model.readout_ids[:] for _ in range(6)]
    style=CachedStyle(style_rows,model.processor,style_codes,model.readout_ids,
                       name='style30k',override=override,shard_of=style_shards,resident=resident)
    mmart=E71.MMArtItems([str(E71.STORE),str(PPR)],model.processor,model.readout_ids,
                         label_index=bank.index)
    local=MX.ChainItems(plan,model.processor,groups,source,instructions,name='local')
    val=MX.ChainItems(valplan,model.processor,groups,source,instructions,split='val',name='val')
    queues={
        'global':MX.InterleavedQueue([('style',style),('mmart',mmart)],seed=args.seed,
                                     name='global',block=512),
        'local':MX.InterleavedQueue([('local',local)],seed=args.seed+1,name='local',block=512)}
    loaders={name:MX.LoaderCycle(DataLoader(q,batch_size=args.micro,num_workers=0,
                 collate_fn=MX.collate_mixed,drop_last=True),queue=q) for name,q in queues.items()}
    from veraretouch_sprf.data.stage_targets import LutVolumes
    from q3vl.whatb.lutdata import LutBank
    from tools.epr059_glutbasis.common import BANK_DIR
    run_args=json.loads((MD.BK_RUN/'run_args.json').read_text())['config']
    chain_bank=LutVolumes(run_args['data']['lut_bank_dir'],4200)
    style_bank=LutBank(BANK_DIR,cache_size=4200)
    pix=ChainPixelL1(MD.GEOMETRY,device,stride=1)
    opt_groups=model.trainable_groups(1e-4,1e-3,1e-3,.01)
    base_lr=[g['lr'] for g in opt_groups]
    optimizer=torch.optim.AdamW([{k:v for k,v in g.items() if k!='name'} for g in opt_groups],
                                betas=(.9,.999),eps=1e-8)
    config=dict(epr='EPR072LOCAL',arm=args.arm,init=initial,scaler=str(E71.SCALER),
        scaler_step=scaler_step,geometry=str(MD.GEOMETRY),style_n=len(style),mmart=mmart.facts(),
        local_n=len(local),batch=args.batch,micro=args.micro,total_steps=steps,
        loss=dict(pixel=1,code_mse=0,infonce=weight_nce,temperature=.07),
        beta='current photometric; recorded subject support teacher; support-only, strength absorbed',
        slots=[0,1,2,3,4,5],subject='recorded geometric support; where head not trained',
        local_positive='fresh restoration code from current support; bank negatives',
        global_positive='style own LUT / MMArt existing best rendered prototype',
        model=model.facts,seed=args.seed,queue_cycle=['global','local'],
        local_journal='existing six-stage AR1600 cache',
        io=dict(whole_file_budget_gib=12,block=512,style_selected_tar_groups=selected_shards),
        style_journal='same EPR071 base-SFT journal and rewritten request',
        caveat='6250 updates with 1:1 replay sees about 50000 local chains; all valid chains are in the sampling pool')
    write(out/'config.json',config)
    write(out/'sample_manifest.json',dict(style=[r['key'] for r in style_rows],
          mmart=[r['key'] for r in mmart.samples],local=[r['key'] for r in plan['records']]))
    emit('ready',config=config)
    evaluator=ST.SelectEval(model,bank,device,micro=2,num_workers=0,render_chunk=6,
        halves_path=HALVES,arm='R',code_mean=mean,code_std=std,journal=str(JOURNAL))
    best=float('inf'); anchors={}; log=(run/'steps.jsonl').open('w')
    _,guard_samples=SD.load_split('style_val')
    guard_samples=guard_samples[:32]
    guard_ds=SD.StyleReadoutDataset(guard_samples,model.processor,'cot',
                                   SD.CodePack(SD.STYLE_CODES),with_image=True)

    @torch.no_grad()
    def style_guard():
        model.model.eval();model.head.eval();errors=[]
        for j in range(0,len(guard_ds),2):
            b=SD.collate([guard_ds[i] for i in range(j,min(j+2,len(guard_ds)))])
            _,q,_=model([model.sequences(body) for body in b['bodies']],b['images'])
            raw=q*std_t+mean_t
            for i,image in enumerate(b['image_rgb']):
                z=image.to(device).reshape(-1,3).float()/255
                target=style_bank.apply(z,b['lut_ids'][i]).clamp(0,1)
                prediction=pix.apply_code(raw[i].reshape(3,-1),z,torch.ones(len(z),device=device))
                errors.append(float((prediction.clamp(0,1)-target).abs().mean()))
        model.model.train();model.head.train()
        return float(np.mean(errors))

    def save(step,tag,extra=None):
        blob=dict(arm='R',experiment_arm=args.arm,step=step,tag=tag,k_readout=model.k,
          n_class=len(bank.ids),bank_ids=bank.ids,residual='on',
          head={k:v.detach().cpu() for k,v in model.head.state_dict().items()},
          readout_emb=model.readout_emb.detach().cpu(),
          lora={n:p.detach().cpu() for n,p in model.model.named_parameters()
                if 'lora_' in n and f'.{SEL_ADAPTER}.' in n},
          code_mean=mean,code_std=std,config=config,extra=extra)
        path=run/f'{tag}.pt'; tmp=path.with_suffix('.partial'); torch.save(blob,tmp);tmp.replace(path)

    def local_terms(cb,scored=True):
        predicted=forward_chain(model,cb['ids'],cb['groups'],cb['images'])
        raw=predicted*std_t+mean_t
        losses=[]; nces=[]; perstage=[]; active=0
        for i,key in enumerate(cb['keys']):
            with torch.no_grad():
                recorded=cb['beta'][i].to(device).float()
                states=MD.chain_states(cb['x0'][i],recorded,cb['luts'][i],chain_bank,device)
                beta=current_supports(states,recorded,float(cb['s'][i]),cb['hw'][i])
            one=[]
            for slot in range(5,-1,-1):
                z,target,a=states[slot+1],states[slot],beta[slot]
                one.append(pix.slot_loss(raw[i,slot].reshape(3,-1),z,target,a))
                if weight_nce and scored and bool((a>0).any()):
                    cachekey=(key,slot)
                    if cachekey not in anchors:
                        anchors[cachekey]=MC.solve_support_code(z,target,a,pix.basis.geometry).cpu()
                    qt=(anchors[cachekey].to(device).reshape(-1)-mean_t)/std_t
                    nces.append(local_nce(predicted[i,slot],qt,norm_bank))
                    active+=1
            losses.append(torch.stack(one).mean())
            perstage.append([float(v.detach()) for v in one])
        loss=torch.stack(losses).mean()
        return loss,torch.stack(nces).mean() if nces else loss.new_zeros(()),perstage

    @torch.no_grad()
    def evaluate(step):
        nonlocal best
        model.model.eval();model.head.eval()
        one=[];roll=[]
        for start in range(0,len(val),2):
            cb=MX.collate_mixed([val[j] for j in range(start,min(start+2,len(val)))])['chain']
            q=forward_chain(model,cb['ids'],cb['groups'],cb['images'])
            raw=q*std_t+mean_t
            for i,key in enumerate(cb['keys']):
                recorded=cb['beta'][i].to(device).float()
                states=MD.chain_states(cb['x0'][i],recorded,cb['luts'][i],chain_bank,device)
                fields=current_supports(states,recorded,float(cb['s'][i]),cb['hw'][i])
                current=states[-1]; step_losses=[]
                for slot in range(5,-1,-1):
                    step_losses.append(float(pix.slot_loss(raw[i,slot].reshape(3,-1),
                                             states[slot+1],states[slot],fields[slot])))
                    subject=fields[0].reshape(*cb['hw'][i])
                    live,_=state_alphas(current.reshape(*cb['hw'][i],3),subject)
                    current=pix.apply_code(raw[i,slot].reshape(3,-1),current,
                                           live[slot].to(device).reshape(-1))
                one.append(float(np.mean(step_losses)))
                roll.append(float((current.clamp(0,1)-states[0]).abs().mean()))
        global_scores=evaluator.artedit('val50')
        style_error=style_guard()
        result=dict(step=step,style_guard_l1=style_error,
                    style_ratio_to_initial=style_error/style_baseline,
                    local_single_l1=float(np.mean(one)),
                    local_rollout_l1=float(np.mean(roll)),global_val50=global_scores)
        write(run/f'evaluations/step{step:06d}.json',result)
        save(step,'latest',result)
        if result['local_rollout_l1']<best and style_error<=1.05*style_baseline:
            best=result['local_rollout_l1'];save(step,'best',result)
        emit('eval',**result)
        model.model.train();model.head.train()

    style_baseline=style_guard()
    write(out/'initial_style_guard.json',dict(n=len(guard_ds),l1=style_baseline))
    emit('initial_style_guard',n=len(guard_ds),l1=style_baseline)
    model.model.train();model.head.train()
    for step in range(steps):
        lane='global' if step%2==0 else 'local'
        mult=min((step+1)/100,1.)*.5*(1+math.cos(math.pi*max(step-100,0)/max(steps-100,1)))
        for g,lr in zip(optimizer.param_groups,base_lr):g['lr']=lr*mult
        optimizer.zero_grad(set_to_none=True); values=[]; keys=[]; stages=[]
        for _ in range(accum):
            batch=loaders[lane].next();keys+=batch['keys']
            if lane=='local':
                pixel,nce,parts=local_terms(batch['chain']);stages+=parts
            else:
                sb=batch['single'];_,q,_=model(sb['ids'],sb['images']);raw=q*std_t+mean_t
                terms=[]
                for i in range(len(sb['keys'])):
                    z=sb['image_rgb'][i].to(device).reshape(-1,3).float()/255
                    with torch.no_grad():
                        target=(sb['target_rgb'][i].to(device).reshape(-1,3).float()/255
                           if sb['target_modes'][i]=='image' else style_bank.apply(z,sb['lut_ids'][i]).clamp(0,1))
                    terms.append(pix.slot_loss(raw[i].reshape(3,-1),z,target,torch.ones(len(z),device=device)))
                pixel=torch.stack(terms).mean()
                nce=infonce(q,bank_std,bank.labels(sb['lut_ids'],device),.07) if weight_nce else pixel.new_zeros(())
            loss=pixel+weight_nce*nce
            if not torch.isfinite(loss): raise RuntimeError(f'nonfinite loss step{step}')
            (loss/accum).backward(); values.append((float(pixel.detach()),float(nce.detach()),float(loss.detach())))
        grad=float(torch.nn.utils.clip_grad_norm_([p for _,p in model.trainable_tensors()],1.))
        if not math.isfinite(grad): raise RuntimeError('nonfinite gradients')
        optimizer.step()
        row=dict(step=step+1,queue=lane,pixel=float(np.mean([v[0] for v in values])),
             nce=float(np.mean([v[1] for v in values])),loss=float(np.mean([v[2] for v in values])),
             grad_norm=grad,keys_sha=hashlib.sha256('\n'.join(keys).encode()).hexdigest(),
             stage_losses=np.mean(stages,axis=0).tolist() if stages else None,
             gpu_gib=torch.cuda.max_memory_allocated()/2**30,rss_gib=MX.rss_gib(),io=resident.facts(),
             elapsed=time.monotonic()-begun)
        log.write(json.dumps(row)+'\n');log.flush();emit('train',**row)
        if args.smoke and step+1==steps:save(step+1,'smoke')
        elif (step+1)%args.eval_every==0:evaluate(step+1)
    if not args.smoke:
        evaluate(steps);save(steps,'final')
        write(run/'global_full_final.json',evaluator.artedit('full'))
    emit('done',steps=steps,arm=args.arm,seconds=time.monotonic()-begun)


if __name__=='__main__':main()
