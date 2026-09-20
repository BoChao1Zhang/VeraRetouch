"""PIX-3200 target-free six-stage public evaluation, stage-text disabled.

Six codes are read jointly, then executed sequentially on current states.
The separate SUBJQ locator still generates its where span; no claim of a
completely generation-free pipeline is made. Targets enter scoring only.
Outputs are independent resumable tar shards, never the single-edit board.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import sqlite3  # Must precede torch in this environment.
import tarfile
import tempfile
import time

import numpy as np
from PIL import Image
import torch

CKPT = Path('/home/bc/data/runs/epr072_local_continuation_20260919/final_step3200/PIX_step3200.pt')
CKPT_SHA = '284b8fb5e47285c171caba9c32f9452c73b701c5a7ca713a5d87c7cc9d1d9a40'
ORDER = ('global', 'hue', 'shadows', 'midtones', 'highlights', 'subject')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value, indent=2, default=str)+'\n')
    tmp.replace(path)


class Predictor:
    def __init__(self, device, chunk):
        from veraretouch_sprf.e2e.whereq import SubjQPredictor
        from tools.epr071_val50_diag import load_checkpoint, PROTOSET
        from veraretouch_sprf.readout import select_train as ST, epr071_data as E71, multistage_data as MD
        from veraretouch_sprf.readout.multistage_loss import ChainPixelL1
        if digest(CKPT) != CKPT_SHA:
            raise ValueError('Checkpoint differs from frozen PIX-3200')
        self.device = device
        self.where = SubjQPredictor(device=device)
        bank = ST.Bank(device, path=PROTOSET)
        self.model, self.facts = load_checkpoint(CKPT, bank, device)
        self.model.model.eval()
        self.model.head.eval()
        mean, std, _ = E71.load_scaler()
        self.mean = torch.as_tensor(mean, device=device)
        self.std = torch.as_tensor(std, device=device)
        self.pix = ChainPixelL1(MD.GEOMETRY, device, chunk=chunk, stride=1)
        self.where_facts = self.where.facts()

    @torch.inference_mode()
    def predict(self, rgb, instruction):
        # No target path, target tensor, recorded mask, or fitted code is accepted.
        from veraretouch_sprf.data import q3vl_text as T
        from veraretouch_sprf.models.vlm import q3vl_common as Q
        from veraretouch_sprf.readout.epr072_local_train import forward_chain
        from veraretouch_sprf.e2e.masks import state_alphas_device
        from tools.epr075_fig4.render_ours import sequence_six
        pil = Image.fromarray(rgb)
        h, w = rgb.shape[:2]
        located = self.where.predict(pil, instruction)
        subject = located['soft'].to(self.device)
        if tuple(subject.shape) != (h, w):
            raise ValueError('Predicted subject support size mismatch')
        small, _ = Q.prepare_image_spec5(pil)
        enc = T.encode_prompt(self.model.processor, small, instruction)
        segments = [[int(token)] for token in self.model.stage_ids[:6]]
        ids, groups = sequence_six(enc['input_ids'][0].tolist(), segments,
                                   self.model.readout_ids)
        images = [dict(pixel_values=enc['pixel_values'], image_grid_thw=enc['image_grid_thw'])]
        codes = (forward_chain(self.model, [ids], [groups], images)*self.std+self.mean)[0]
        if tuple(codes.shape) != (6, 780) or not torch.isfinite(codes).all():
            raise ValueError(f'Invalid six-stage codes: {tuple(codes.shape)}')
        current = torch.as_tensor(rgb.astype(np.float32)/255, device=self.device).reshape(-1, 3)
        stage_rows = []
        for stage, slot in enumerate(range(5, -1, -1)):
            fields, _ = state_alphas_device(current.reshape(h, w, 3), subject,
                                            info=False, validate=False)
            support = fields[slot].reshape(-1)
            updated = self.pix.apply_code(codes[slot].reshape(3, -1), current, support)
            if not torch.isfinite(updated).all():
                raise ValueError(f'Nonfinite image at stage {stage+1}')
            zero = support == 0
            unchanged = bool(torch.equal(updated[zero], current[zero]))
            if not unchanged:
                raise ValueError('Zero-support identity check failed')
            stage_rows.append(dict(stage=stage+1, slot=slot, name=ORDER[stage],
                                   support_mean=float(support.mean()),
                                   actual_change_l1=float((updated-current).abs().mean()),
                                   zero_support_pixels=int(zero.sum()),
                                   zero_support_unchanged=unchanged))
            current = updated
            del fields, support, zero
        pred = np.rint(current.reshape(h, w, 3).clamp(0, 1).cpu().numpy()*255).astype(np.uint8)
        facts = {k: v for k, v in located.items() if k not in ('soft', 'hard')}
        return pred, codes.cpu().numpy(), stage_rows, facts


def metrics(pred, target, device):
    from q3vl.whatb.pubbench.epr035d_metrics import psnr, ssim
    from q3vl.whatb.colorimetry import delta_e00_srgb
    if pred.shape != target.shape:
        raise ValueError(f'Reference size mismatch: {pred.shape} vs {target.shape}')
    diff = (pred.astype(np.float64)-target.astype(np.float64))/255
    with torch.inference_mode():
        de = float(delta_e00_srgb(torch.from_numpy(pred.astype(np.float32)/255).to(device),
                                  torch.from_numpy(target.astype(np.float32)/255).to(device)).mean())
    return dict(l1=float(np.abs(diff).mean()), l2=float((diff**2).mean()),
                psnr=float(psnr(pred, target)), ssim=float(ssim(pred, target, device=device)), de00=de)


def records_for(bench):
    if bench == 'artedit':
        from veraretouch_sprf.readout.artedit_eval import load_split
        return load_split('full')[0]
    from veraretouch_sprf.e2e.bench import samples_for
    return [dict(r) for r in samples_for(bench)[0]]


def tar_add(pack, name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0
    pack.addfile(info, io.BytesIO(data))


def run_bench(predictor, bench, args):
    from veraretouch_sprf.readout.artedit_eval import load_rgb_u8
    records = records_for(bench)
    if args.limit:
        records = records[:args.limit]
    out = args.out/bench
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(protocol='six-stage joint readout, sequential current-state execution',
                    checkpoint=str(CKPT), checkpoint_sha256=CKPT_SHA,
                    stage_text='disabled; six stage markers and six readout groups retained',
                    subject='SUBJQ predicted from input + instruction; generated where span',
                    target_access_during_inference=False, slots=[5, 4, 3, 2, 1, 0],
                    intermediate_clipping=False, final_clipping=True, native_resolution=True,
                    source_sha256=digest(Path(__file__)),
                    sample_ids=[r['sample_id'] for r in records], limit=args.limit,
                    shard_size=args.shard_size)
    path = out/'manifest.json'
    if path.exists() and read_json(path) != manifest:
        raise ValueError('Existing output has a different frozen protocol')
    dump(path, manifest)
    dump(out/'model_facts.json', dict(readout=predictor.facts, subject=predictor.where_facts))
    all_rows = []
    for start in range(0, len(records), args.shard_size):
        batch = records[start:start+args.shard_size]
        stem = f'part_{start:04d}'
        done = out/(stem+'.json')
        tar_out = out/(stem+'.tar')
        if done.exists():
            result = read_json(done)
            if result['sample_ids'] != [r['sample_id'] for r in batch] or digest(tar_out) != result['tar_sha256']:
                raise ValueError('Invalid completed shard')
            all_rows.extend(result['rows'])
            continue
        rows = []
        with tempfile.TemporaryDirectory(prefix='sixstage-', dir='/dev/shm') as temp:
            tar_path = Path(temp)/(stem+'.tar')
            with tarfile.open(tar_path, 'w') as pack:
                for record in batch:
                    t0 = time.monotonic()
                    rgb = load_rgb_u8(record['input_path'])
                    pred, codes, stages, where = predictor.predict(rgb, record['instruction'])
                    # Only now is the target loaded for reference scoring.
                    target = load_rgb_u8(record['gt_path'])
                    scores = metrics(pred, target, args.device)
                    sid = record['sample_id']
                    png = io.BytesIO()
                    Image.fromarray(pred).save(png, format='PNG', compress_level=1)
                    tar_add(pack, sid+'.png', png.getvalue())
                    buf = io.BytesIO()
                    np.save(buf, codes)
                    tar_add(pack, sid+'.codes.npy', buf.getvalue())
                    row = dict(sample_id=sid, h=int(pred.shape[0]), w=int(pred.shape[1]),
                               input_sha256=digest(record['input_path']),
                               prediction_sha256=hashlib.sha256(png.getvalue()).hexdigest(),
                               instruction=record['instruction'], stages=stages, where=where,
                               **scores, shared_workload_seconds=time.monotonic()-t0)
                    rows.append(row)
                    print(json.dumps(dict(bench=bench, sample_id=sid, completed=start+len(rows),
                                           total=len(records), stages=len(stages), **scores)), flush=True)
            staging = tar_out.with_suffix('.tar.partial')
            shutil.copyfile(tar_path, staging)
            staging.replace(tar_out)
        result = dict(sample_ids=[r['sample_id'] for r in batch], rows=rows, tar_sha256=digest(tar_out))
        dump(done, result)
        all_rows.extend(rows)
        dump(out/'progress.json', dict(n_completed=len(all_rows), n_expected=len(records)))
    summary = dict(bench=bench, n=len(all_rows), complete=len(all_rows)==len(records),
                   checkpoint_sha256=CKPT_SHA, protocol=manifest,
                   means={k: float(np.mean([r[k] for r in all_rows]))
                          for k in ('l1', 'l2', 'psnr', 'ssim', 'de00')})
    dump(out/'summary.json', summary)
    print('BENCH_COMPLETE '+json.dumps(summary), flush=True)


def read_json(path):
    return json.loads(Path(path).read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bench', nargs='+', choices=['artedit', 'fivek', 'ppr10k'], default=['artedit'])
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--shard-size', type=int, default=16)
    ap.add_argument('--chunk', type=int, default=65536)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--mem-fraction', type=float, default=.28)
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20260920)
    np.random.seed(20260920)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
    predictor = Predictor(args.device, args.chunk)
    print('MODEL_READY', flush=True)
    for bench in args.bench:
        run_bench(predictor, bench, args)


if __name__ == '__main__':
    main()
