"""EPR-081: before + final instruction -> one global code, after-only MAE.

This trainer cannot read the original trajectory cache. All examples, including
replay, cross the exact-schema pair-only boundary in epr081_pairs.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from veraretouch_sprf.readout.epr081_pairs import PairStore

INIT = Path('/home/bc/nfsvfs/bc/data/runs/epr071_mmart_20260918/train/R/run/best.pt')
INIT_SHA = '4309e517e524fdf74a08881ad017b009036a4d217aad6adc5e94f825edfb9fbf'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda:f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def multiplier(step, horizon=6250):
    return min((step+1)/100, 1.) * .5 * (1+math.cos(math.pi*max(step-100,0)/max(horizon-100,1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--start', default=str(INIT))
    ap.add_argument('--smoke', type=int, default=0)
    ap.add_argument('--micro', type=int, default=4)
    ap.add_argument('--mem-fraction', type=float, default=.78)
    ap.add_argument('--save-every', type=int, default=400)
    args = ap.parse_args()
    if 16 % args.micro: raise ValueError('micro must divide effective batch 16')
    root, out = Path(args.pairs), Path(args.out)
    audit = json.loads((root/'export_audit.json').read_text())
    if not audit['complete'] and not args.smoke:
        raise ValueError('Cannot train an incomplete pair export')
    if sha(args.start) != INIT_SHA: raise ValueError('Shared initialization SHA differs')
    store = PairStore(root)
    plan = json.loads((root/'batch_plan.json').read_text())['batches']
    if len(plan) != 3200: raise ValueError('Expected exactly 3,200 matched batches')
    for b in plan[:args.smoke or 3200]:
        if len(b['keys']) != 16 or any(k not in store.rows for k in b['keys']):
            raise ValueError('Missing pair or wrong effective batch')
        if hashlib.sha256('\n'.join(b['keys']).encode()).hexdigest() != b['keys_sha']:
            raise ValueError('Batch digest mismatch')
    out.mkdir(parents=True, exist_ok=True)
    if (out/'steps.jsonl').exists(): raise ValueError('Refusing to overwrite a run')
    from veraretouch_sprf.readout import select_train as ST, epr071_data as E71
    from veraretouch_sprf.readout.multistage_loss import ChainPixelL1
    from veraretouch_sprf.readout.multistage_data import GEOMETRY
    from veraretouch_sprf.readout.select_model import SEL_ADAPTER
    from tools.epr071_val50_diag import load_checkpoint
    torch.set_num_threads(4); torch.manual_seed(20260918); np.random.seed(20260918)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
    device = 'cuda:0'
    bank = ST.Bank(device, path=E71.PROTOSET)
    model, initial = load_checkpoint(Path(args.start), bank, device)
    if initial['arm'] != 'R' or model.k != 8: raise ValueError('Expected R/k8 initialization')
    model.model.base_model.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={'use_reentrant':False})
    model.model.base_model.model.enable_input_require_grads()
    for p in model.head.select.parameters(): p.requires_grad_(False)
    mean, std, _ = E71.load_scaler()
    mean_t, std_t = torch.from_numpy(mean).to(device), torch.from_numpy(std).to(device)
    pix = ChainPixelL1(GEOMETRY, device, chunk=196608, stride=2)
    groups = model.trainable_groups(1e-4, 1e-3, 1e-3, .01)
    base_lr = [g['lr'] for g in groups]
    opt = torch.optim.AdamW([{k:v for k,v in g.items() if k!='name'} for g in groups],
                           betas=(.9,.999), eps=1e-8)
    config = dict(epr='EPR081', arm='GLOBAL_ENDPOINT', init=initial, init_sha256=INIT_SHA,
        batch=16, micro=args.micro, total_steps=3200, schedule_steps=6250, seed=20260918,
        loss='mean full-image un-clipped MAE at pixel stride 2; beta=1',
        input='before image + final user instruction + one eight-token readout group',
        forbidden='CoT, stage markers, supports, intermediate states, code/prototype targets',
        replay='same pair-only boundary as former local stream',
        initialization_caveat='shared pretrained initialization may already encode process learning',
        optimizer=dict(lora_lr=1e-4, head_lr=1e-3, embed_lr=1e-3, weight_decay=.01,
                       betas=[.9,.999], eps=1e-8, clip_grad=1.),
        pairs_sha256=sha(root/'pairs.json'), batch_plan_sha256=sha(root/'batch_plan.json'),
        trainer_sha256=sha(__file__), data_sha256=sha(Path(__file__).with_name('epr081_pairs.py')))
    (out/'config.json').write_text(json.dumps(config, indent=2)+'\n')

    def save(step, tag):
        blob = dict(arm='R', experiment_arm='GLOBAL_ENDPOINT', step=step, tag=tag,
            k_readout=model.k, n_class=len(bank.ids), bank_ids=bank.ids, residual='on',
            head={k:v.detach().cpu() for k,v in model.head.state_dict().items()},
            readout_emb=model.readout_emb.detach().cpu(),
            lora={n:p.detach().cpu() for n,p in model.model.named_parameters()
                  if 'lora_' in n and f'.{SEL_ADAPTER}.' in n},
            code_mean=mean, code_std=std, config=config)
        tmp = out/(tag+'.partial'); torch.save(blob,tmp); tmp.replace(out/(tag+'.pt'))

    model.model.train(); model.head.train(); begun=time.time()
    steps=args.smoke or 3200
    with (out/'steps.jsonl').open('w', buffering=1) as log:
        for step,b in enumerate(plan[:steps]):
            for g,lr in zip(opt.param_groups,base_lr): g['lr']=lr*multiplier(step)
            opt.zero_grad(set_to_none=True); losses=[]
            for start in range(0,16,args.micro):
                items=[store.item(k,model.processor,model.readout_ids,model.stage_ids)
                       for k in b['keys'][start:start+args.micro]]
                _,q,_=model([r['ids'] for r in items],[r['image'] for r in items])
                raw=q*std_t+mean_t
                terms=[]
                for i,row in enumerate(items):
                    before,after=row['before'].to(device),row['after'].to(device)
                    terms.append(pix.slot_loss(raw[i].reshape(3,-1),before,after,
                                               torch.ones(len(before),device=device)))
                loss=torch.stack(terms).mean()
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite pair loss')
                (loss*args.micro/16).backward(); losses.append(float(loss.detach()))
            grad=float(torch.nn.utils.clip_grad_norm_([p for _,p in model.trainable_tensors()],1.))
            if not math.isfinite(grad): raise RuntimeError('Nonfinite gradients')
            if args.smoke:
                surfaces = dict(readout=[model.readout_emb], head=list(model.head.residual.parameters()),
                                lora=[p for n,p in model.model.named_parameters()
                                      if 'lora_' in n and f'.{SEL_ADAPTER}.' in n])
                for name,params in surfaces.items():
                    magnitude=sum(float(p.grad.detach().abs().sum()) for p in params if p.grad is not None)
                    if not math.isfinite(magnitude) or magnitude <= 0:
                        raise RuntimeError(f'Smoke gradient surface disconnected: {name}')
            opt.step()
            row=dict(step=step+1,queue=b['queue'],keys_sha=b['keys_sha'],loss=float(np.mean(losses)),
                     grad_norm=grad,elapsed=time.time()-begun)
            log.write(json.dumps(row)+'\n'); print(json.dumps(row),flush=True)
            if (step+1)%args.save_every==0: save(step+1,f'step{step+1:06d}')
        save(steps,'smoke' if args.smoke else 'final')
    if args.smoke:
        # State round-trip integrity without constructing a second full base model.
        blob=torch.load(out/'smoke.pt',map_location='cpu',weights_only=False)
        for k,v in model.head.state_dict().items():
            if not torch.equal(v.detach().cpu(),blob['head'][k]): raise RuntimeError('Save round-trip mismatch')
        model.model.eval(); model.head.eval()
        with torch.no_grad():
            item=store.item(plan[1]['keys'][0],model.processor,model.readout_ids,model.stage_ids)
            # Target is not an argument of either predictor or global executor.
            _,q,_=model([item['ids']],[item['image']])
            z=item['before'].to(device)
            rendered=pix.apply_code((q[0]*std_t+mean_t).reshape(3,-1),z,torch.ones(len(z),device=device))
            if not torch.isfinite(rendered).all(): raise RuntimeError('Nonfinite target-free rendering')
            score=float((rendered.clamp(0,1)-item['after'].to(device)).abs().mean())
        report=dict(event='smoke_pass',steps=steps,save_reload=True,
                    gradients='head/readout/LoRA finite and nonzero',target_free_render_l1=score)
        (out/'smoke_pass.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)


if __name__=='__main__': main()
