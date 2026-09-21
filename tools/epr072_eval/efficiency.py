"""Exclusive-device latency test for the actual six-stage no-CoT pipeline.

50 deterministic ArtEdit inputs; matched short-side-512 comparison plus ours
512/1024/2048/4096 scaling. Larger inputs are resized stress tests, not native
high-resolution photographs. Never compared with a baseline's different size.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import time
import traceback

import numpy as np
from PIL import Image

from tools.epr072_eval.public_six_stage import dump, digest


class DeviceBusy(RuntimeError):
    """Transient contention; the scheduler may retry in a fresh process."""


def require_exclusive():
    pids = other_processes()
    if pids:
        raise DeviceBusy(f'Device is shared with PIDs {pids}; timing is invalid')


def test_size(original, short_side):
    scale = short_side / min(original)
    return tuple(max(64, round(v * scale / 64) * 64) for v in original)


def other_processes():
    device = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    uuid = subprocess.check_output(['nvidia-smi','-i',device,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    lines = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    return [int(s.split(',')[1]) for s in lines.splitlines()
            if s.split(',')[0].strip()==uuid and int(s.split(',')[1])!=os.getpid()]


def load_ours():
    import torch
    from tools.epr072_eval.public_six_stage import Predictor
    from veraretouch_sprf.e2e import masks
    predictor = Predictor('cuda:0',65536)
    support_fn, apply_fn = masks.state_alphas_device, predictor.pix.apply_code
    times=[]
    def measured(fn):
        def call(*a,**kw):
            torch.cuda.synchronize(); start=time.perf_counter()
            result=fn(*a,**kw)
            torch.cuda.synchronize(); times.append(time.perf_counter()-start)
            return result
        return call
    masks.state_alphas_device=measured(support_fn)
    predictor.pix.apply_code=measured(apply_fn)
    def call(row):
        times.clear()
        rgb=np.asarray(Image.open(row['image']).convert('RGB'))
        output,_,stages,_=predictor.predict(rgb,row['instruction'])
        if len(times)!=12 or len(stages)!=6:
            raise ValueError('Expected six support calculations and six color updates')
        return output,dict(render_seconds=sum(times))
    return call,dict(checkpoint=predictor.facts,stage_text='none',subject='generated where span + SUBJQ',
                     render_scope='six current-state support calculations and six color updates; excludes output quantization/host transfer')


def loader(method):
    if method=='ours':return load_ours()
    if method=='veraretouch':
        from tools.appendix_latency_benchmark import load_vera
        return load_vera()
    import torch
    from q3vl.whatb.pubbench.epr038c_render import build_pipeline,METHODS
    pipe=build_pipeline(method,'cuda:0')
    params=dict(METHODS[method]['params']);params.pop('resolution',None)
    if method=='flux_kontext_dev':
        params['num_inference_steps']=50  # Installed pipeline's default, now explicit.
    def call(row):
        image=Image.open(row['image']).convert('RGB')
        w,h=image.size
        kwargs=dict(prompt=row['instruction'],generator=torch.Generator(device='cpu').manual_seed(19),
                    height=h,width=w,**params)
        if method == 'flux_kontext_dev':
            # Explicit H/W alone still normalizes area to 1024^2 in this
            # installed pipeline. Both public options must be overridden.
            kwargs.update(max_area=w*h, _auto_resize=False)
        elif method == 'instructpix2pix':
            # This pipeline takes its output grid from image preprocessing;
            # height/width are not supported call arguments in this version.
            kwargs.pop('height');kwargs.pop('width')
        kwargs['image']=[image] if method=='qwen_image_edit_2511' else image
        output=pipe(**kwargs).images[0]
        return np.asarray(output),{}
    return call,dict(checkpoint=str(METHODS[method]['path']),params=params,
                     output_geometry='explicit requested width/height; no post-resize',
                     condition_geometry=('matched input grid; max_area=W*H, _auto_resize=False'
                                         if method=='flux_kontext_dev' else
                                         'official internal conditioning preprocessing retained'),
                     dtype='bfloat16',seed=19)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--method',choices=['ours','veraretouch','instructpix2pix','flux_kontext_dev','qwen_image_edit_2511'],required=True)
    ap.add_argument('--short-side',type=int,default=512)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--repeats',type=int,default=3)
    ap.add_argument('--n',type=int,default=50)
    ap.add_argument('--preflight',action='store_true',help='Validate inputs and size limits without loading models or using the device')
    args=ap.parse_args()
    if args.n < 1 or args.repeats < 1 or args.short_side < 64:
        ap.error('n/repeats must be positive and short-side at least 64')
    import torch
    from veraretouch_sprf.readout.artedit_eval import load_split,load_rgb_u8
    torch.set_num_threads(4);torch.manual_seed(19)
    torch.backends.cuda.matmul.allow_tf32=False
    records=load_split('full')[0]
    records=sorted(records,key=lambda r:hashlib.sha256(('efficiency-20260921:'+r['sample_id']).encode()).hexdigest())[:args.n]
    args.out.mkdir(parents=True,exist_ok=True)
    if len(records) != args.n:
        raise ValueError('Requested sample count exceeds available records')
    if args.method == 'ours':
        oversized=[]
        for r in records:
            with Image.open(r['input_path']) as frame:
                size=test_size(frame.size,args.short_side)
            if size[0]*size[1] > 2**24:
                oversized.append(dict(id=r['sample_id'],size=size))
        if oversized:
            dump(args.out/'unsupported.json',dict(complete=False,reason='Current exact support implementation uses torch.quantile with a 2^24-pixel limit',samples=oversized))
            raise NotImplementedError('Requested grid exceeds exact support quantile limit; see unsupported.json')
    rows=[]
    with tempfile.TemporaryDirectory(prefix='efficiency-inputs-',dir='/dev/shm') as temporary:
        prepared=[]
        for r in records:
            image=Image.fromarray(load_rgb_u8(r['input_path']))
            original=list(image.size)
            size=test_size(original,args.short_side)
            image=image.resize(size,Image.Resampling.LANCZOS)
            path=Path(temporary)/(r['sample_id']+'.png');image.save(path,compress_level=1)
            prepared.append(dict(id=r['sample_id'],image=str(path),size=list(size),original_size=original,
                                 input_sha256=digest(r['input_path']),prepared_sha256=digest(path),
                                 instruction=r['instruction']))
        input_audit=[{k:v for k,v in r.items() if k!='image'} for r in prepared]
        dump(args.out/'inputs.json',dict(records=input_audit))
        oversized=[r['id'] for r in prepared if r['size'][0]*r['size'][1] > 2**24]
        if args.method=='ours' and oversized:
            # The exact current-state luminance support uses torch.quantile,
            # whose input limit is 2^24. Do not silently subsample or swap the
            # support algorithm just to make a resolution claim.
            dump(args.out/'unsupported.json',dict(complete=False,reason='Current exact support implementation uses torch.quantile with a 2^24-pixel limit',sample_ids=oversized))
            raise NotImplementedError('Requested grid exceeds exact support quantile limit; see unsupported.json')
        if args.preflight:
            dump(args.out/'preflight.json',dict(valid_inputs=True,n=len(prepared),device_used=False))
            return
        require_exclusive()
        call,config=loader(args.method)
        require_exclusive()
        dump(args.out/'protocol.json',dict(method=args.method,short_side=args.short_side,n=args.n,
             repeats=args.repeats,warmup=2,config=config,sample_ids=[r['id'] for r in prepared],
             input_sizes=[dict(id=r['id'],size=r['size'],original=r['original_size']) for r in prepared],
             scope='PNG decode, VLM preprocessing, generated locator text where applicable, readout, masks, image execution and host output; excludes model loading, input test-grid construction and disk saving',
             scaling='All methods receive identical resized input pixels; enlarged grids are scaling stress tests',
             source_sha256=digest(Path(__file__)),predictor_sha256=digest(Path(__file__).with_name('public_six_stage.py')),
             private_runtime=dict(device=torch.cuda.get_device_name(),torch=torch.__version__,batch=1)))
        for _ in range(2):
            require_exclusive()
            call(prepared[0]);torch.cuda.synchronize()
            require_exclusive()
        for r in prepared:
            samples=[]
            for repeat in range(args.repeats):
                require_exclusive()
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                pred,extra=call(r);torch.cuda.synchronize();elapsed=time.perf_counter()-start
                require_exclusive()
                if [pred.shape[1],pred.shape[0]]!=r['size']:
                    raise ValueError('Output geometry mismatch; no resize is allowed')
                samples.append(dict(seconds=elapsed,render_seconds=extra.get('render_seconds'),
                                    peak_allocated_bytes=torch.cuda.max_memory_allocated()))
            rows.append(dict(id=r['id'],size=r['size'],samples=samples))
            dump(args.out/'partial_results.json',dict(complete=False,n=len(rows),rows=rows))
            dump(args.out/'progress.json',dict(n=len(rows),expected=len(prepared)))
            print(json.dumps(dict(method=args.method,short_side=args.short_side,n=len(rows),seconds=float(np.median([s['seconds'] for s in samples])))),flush=True)
        require_exclusive()
    dump(args.out/'results.json',dict(complete=True,n=len(rows),rows=rows,
         median_seconds=float(np.median([np.median([s['seconds'] for s in r['samples']]) for r in rows])),
         median_render_seconds=float(np.median([s['render_seconds'] for r in rows for s in r['samples']])) if args.method=='ours' else None))


if __name__=='__main__':
    import sys
    try:
        main()
    except DeviceBusy:
        traceback.print_exc()
        sys.exit(75)
