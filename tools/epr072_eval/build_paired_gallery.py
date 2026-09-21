"""Pair actual public-benchmark outputs; optional first-eight code replay.

No color grading or metric-dependent selection. Final outputs remain the
archived scored PNGs. Stages 1--5 are replayed with frozen saved codes;
stage 6 uses the original output (no subject mask is invented).
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import tarfile

import numpy as np
from PIL import Image

REPO = Path('/home/bc/VeraRetouch')
OUT = REPO/'outputs/sixstage_cot_pair_review'
RUN = Path('/home/bc/nfsvfs/bc/data/runs/epr072_sixstage_compare_20260921')
ROOTS = dict(cot=RUN/'with_cot/artedit', nocot=Path('/home/bc/nfsvfs/bc/data/runs/epr072_sixstage_public_20260920/full/artedit'))
PRED = Path('/home/bc/data/runs/epr072_sixstage_compare_20260921/predictions')


def load(path):
    return json.loads(path.read_text())


def link(source, target):
    if target.is_symlink():
        assert target.resolve() == source.resolve()
    elif target.exists():
        raise ValueError(f'Refusing to replace {target}')
    else:
        target.symlink_to(source)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--replay', type=int, default=8)
    args = ap.parse_args()
    from veraretouch_sprf.readout import artedit_eval as AE
    records, _ = AE.load_split('full')
    maps, scores = {}, {}
    for mode, root in ROOTS.items():
        rows = {}
        for file in sorted(root.glob('part_*.json')):
            for row in load(file)['rows']:
                rows[row['sample_id']] = dict(row, archive=str(file.with_suffix('.tar')))
        maps[mode] = rows
        scores[mode] = {r['sample_id']:r for r in
                        (json.loads(line) for line in (RUN/'scores'/mode/'rows_viescore_en.jsonl').read_text().splitlines())
                        if r.get('sc') is not None}
    OUT.mkdir(parents=True, exist_ok=True)
    if args.replay:
        import torch
        from veraretouch_sprf.readout.multistage_loss import ChainPixelL1
        from veraretouch_sprf.e2e.masks import state_alphas_device
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.set_per_process_memory_fraction(.08)
        pix = ChainPixelL1(AE.GEOMETRY, 'cuda:0', chunk=65536, stride=1)
    journal = AE.JournalPack(AE.default_journal('six'))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('/home/bc/data/runs/epr051_vlmsft/sft_s1f_full/ckpt_epoch1/tokenizer')
    public = []
    for index, r in enumerate(records):
        sid = r['sample_id']
        folder = OUT/sid
        folder.mkdir(exist_ok=True)
        assert maps['cot'][sid]['input_sha256'] == maps['nocot'][sid]['input_sha256']
        assert maps['cot'][sid]['instruction'] == maps['nocot'][sid]['instruction'] == r['instruction']
        link(Path(r['input_path']), folder/'input.jpg')
        link(Path(r['gt_path']), folder/'reference.jpg')
        result = dict(id=sid, instruction=r['instruction'], replay=index<args.replay,
                      cot_text=tokenizer.decode(journal.get(sid)['token_ids'], skip_special_tokens=False), modes={})
        for mode in ROOTS:
            row = maps[mode][sid]
            source = PRED/mode/(sid+'.png')
            assert hashlib.sha256(source.read_bytes()).hexdigest() == row['prediction_sha256']
            link(source, folder/(mode+'.png'))
            result['modes'][mode] = dict(psnr=row['psnr'], l1=row['l1']*100,
                                         de00=row['de00'], stages=row['stages'],
                                         sc=scores[mode][sid]['sc'], pq=scores[mode][sid]['pq'])
            if index < args.replay:
                with tarfile.open(row['archive']) as pack:
                    codes = np.load(io.BytesIO(pack.extractfile(sid+'.codes.npy').read()))
                rgb = AE.load_rgb_u8(r['input_path'])
                h,w = rgb.shape[:2]
                current = torch.as_tensor(rgb.astype(np.float32)/255, device='cuda:0').reshape(-1,3)
                subject = torch.zeros((h,w),device='cuda:0')
                checks = []
                with torch.inference_mode():
                    for stage, slot in enumerate(range(5,0,-1),1):
                        fields,_ = state_alphas_device(current.reshape(h,w,3),subject,info=False,validate=False)
                        code = torch.as_tensor(codes[slot].reshape(3,-1),device='cuda:0')
                        updated = pix.apply_code(code,current,fields[slot].reshape(-1))
                        error = abs(float((updated-current).abs().mean())-row['stages'][stage-1]['actual_change_l1'])
                        if error > 1e-6:
                            raise ValueError(f'Replay mismatch {sid} {mode} stage{stage}: {error}')
                        checks.append(error)
                        current = updated
                        pixels = np.rint(current.reshape(h,w,3).clamp(0,1).cpu().numpy()*255).astype(np.uint8)
                        Image.fromarray(pixels).save(folder/f'{mode}_z{stage}.png',compress_level=1)
                result['modes'][mode]['replay_change_errors'] = checks
        public.append(result)
        if index < args.replay or (index+1)%100 == 0:
            print(f'Prepared {index+1}/{len(records)}',flush=True)
    (OUT/'records.json').write_text(json.dumps(public,ensure_ascii=False))
    (OUT/'index.html').write_text((REPO/'tools/epr072_eval/paired_gallery.html').read_text())
    print(json.dumps(dict(n=len(public), replay=args.replay, output=str(OUT))),flush=True)


if __name__ == '__main__':
    main()
