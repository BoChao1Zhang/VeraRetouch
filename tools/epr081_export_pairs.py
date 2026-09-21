"""Offline-only endpoint export; trajectory annotations never enter the trainer.

Freezes the exact 3,200 matched-control batch order before building float pairs.
Construction runs on CPU by default and does not contend with timing jobs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from veraretouch_sprf.readout import data as SD, mixed_data as MX
from veraretouch_sprf.readout import multistage_data as MD, epr071_data as E71
from veraretouch_sprf.readout.epr072_local_io import (
    select_style_shards, ResidentFiles, install_chain_residency, ShardGroupQueue,
    LooseImageAdapter)
from veraretouch_sprf.readout.epr081_pairs import array_sha, validate_row

CONTROL = Path('/home/bc/data/runs/epr072_local_continuation_20260919/ENDPT2/full')
PPR = Path('/home/bc/nfsvfs/bc/data/runs/epr072_mmart_ppr10k_20260919')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def write(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)


class Refs:
    def __init__(self, records, shard):
        self.records, self.shard = records, shard
    def __len__(self): return len(self.records)
    def __getitem__(self, i): return self.records[i]['key']
    def shard_key(self, i): return self.shard(i)


def setup(out):
    spec = [s for s in json.loads(Path(MX.STAGE1_SOURCES).read_text())['sources']
            if s['name'] == 'style30k']
    style, _, _, _ = SD.load_sources(spec)
    eligible, _, _ = E71.style_override(style, str(E71.STORE / 'journals'))
    style, shards, _ = select_style_shards(eligible, E71.STORE / 'journals', 15000, 20260918)
    rewrite = E71.style_rewrite_map()[0]
    mmart = E71.MMArtItems([str(E71.STORE), str(PPR)], None, [], label_index=None)
    plan = MD.read_stage2_manifest_multi(str(MX.CHAIN_MANIFEST), splits=['trainfull', 'heldout'])
    records = [r for r in plan['records'] if r['format_valid']]
    payloads, mapping, _ = MX.MixedChainSource.collect_subjects(
        [r['key'] for r in records], cache_path=CONTROL / 'subject_paths.json')
    source = MX.MixedChainSource(subject_payloads=payloads, subject_mapping=mapping)
    source.setup()
    resident = ResidentFiles(budget_gib=10, prefetch=False)
    loose = ResidentFiles(budget_gib=1.5)
    install_chain_residency(source, resident, loose)
    sr = Refs(style, lambda i: ('style', shards[style[i]['key']]))
    mr = Refs(mmart.samples, mmart.shard_key)
    lr = Refs(records, lambda i: ('local', source.ds.index[records[i]['key']]['dir']))
    queues = dict(global_=MX.InterleavedQueue([('style', sr), ('mmart', mr)],
                      seed=20260918, name='global', block=512),
                  local=ShardGroupQueue([('local', lr)], seed=20260919, name='local',
                      resident=resident, budget_bytes=10 * 2**30, block=64, max_dirs=8))
    queues['global'] = queues.pop('global_')
    pool = dict(style=[r['key'] for r in style], mmart=[r['key'] for r in mmart.samples],
                local=[r['key'] for r in records])
    if pool != json.loads((CONTROL / 'sample_manifest.json').read_text()):
        raise ValueError('Sample pools differ from matched control')
    positions = dict(global_=0, local=0); positions['global'] = positions.pop('global_')
    epochs = dict(global_=0, local=0); epochs['global'] = epochs.pop('global_')
    expected = [json.loads(s) for s in (CONTROL / 'run/steps.jsonl').read_text().splitlines()]
    batches = []
    for step in range(3200):
        lane = 'global' if step % 2 == 0 else 'local'
        q = queues[lane]; keys = []
        for _ in range(4):  # matched four-item fetch boundary/drop_last
            if positions[lane] + 4 > len(q):
                epochs[lane] += 1; q.reshuffle(epochs[lane]); positions[lane] = 0
            for i in range(positions[lane], positions[lane] + 4):
                p, j = q.order[i]; keys.append(q.parts[p][1][j])
            positions[lane] += 4
        digest = hashlib.sha256('\n'.join(keys).encode()).hexdigest()
        if digest != expected[step]['keys_sha'] or lane != expected[step]['queue']:
            raise ValueError(f'Batch order mismatch at update {step+1}')
        batches.append(dict(step=step+1, queue=lane, keys=keys, keys_sha=digest))
    write(out / 'batch_plan.json', dict(schema='epr081-batches-v1', batches=batches))
    write(out / 'pool_audit.json', dict(pool=pool, control=str(CONTROL),
          control_log_sha256=sha(CONTROL / 'run/steps.jsonl'), matched_updates=3200))
    instructions = {}
    for split in ('trainfull', 'heldout'):
        keys = [r['key'] for r in records if r.get('split', 'trainfull') == split]
        if keys: instructions.update(MD.load_instructions(keys, split))
    return source, style, rewrite, mmart, instructions, batches, loose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0, help='preflight only; never train incomplete export')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--compare-to', help='verify reconstructed endpoint arrays against a prior export')
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if (out / 'pairs.json').exists(): raise ValueError('Completed export exists')
    source, style, rewrite, mmart, instructions, batches, loose = setup(out)
    from veraretouch_sprf.data.stage_targets import LutVolumes
    from q3vl.whatb.lutdata import LutBank
    from tools.epr059_glutbasis.common import BANK_DIR
    run_args = json.loads((MD.BK_RUN / 'run_args.json').read_text())['config']
    chain_bank = LutVolumes(run_args['data']['lut_bank_dir'], 4200)
    style_bank = LutBank(BANK_DIR, cache_size=4200)
    styles = {r['key']: r for r in style}
    pairs = {r['key']: r for r in mmart.samples}
    image_reader = LooseImageAdapter(loose, SD.STYLE_ROOT)
    needed = list(dict.fromkeys(k for b in batches for k in b['keys']))
    if args.limit:
        # Include at least one local endpoint in a tiny preflight.
        needed = list(dict.fromkeys(batches[0]['keys'][:args.limit//2] +
                                   batches[1]['keys'][:args.limit-args.limit//2]))
    records = []; started = time.time()
    reference = None
    max_difference = dict(before=0., after=0.)
    if args.compare_to:
        from veraretouch_sprf.readout.epr081_pairs import PairStore
        reference = PairStore(args.compare_to)
    for start in range(0, len(needed), 32):
        arrays = {}; pending = []; shard = f'pairs-{start//32:05d}.npz'
        for j, key in enumerate(needed[start:start+32]):
            if key in styles:
                row = styles[key]; visual = np.asarray(image_reader.image(row), dtype=np.uint8).copy()
                before = torch.from_numpy(visual).to(args.device).float()/255
                after = style_bank.apply(before.reshape(-1, 3), row['lut_id']).clamp(0,1).reshape(before.shape)
                instruction = rewrite[key][1]
            elif key in pairs:
                row = pairs[key]
                im, target, _ = E71.load_pair(row['expert'], row['base'],
                    before_path=row.get('before_path'), after_path=row.get('after_path'))
                visual = np.asarray(im, dtype=np.uint8).copy()
                before = torch.from_numpy(visual).float()/255
                after = torch.from_numpy(np.asarray(target, dtype=np.uint8).copy()).float()/255
                instruction = row['instruction']
            else:
                law = source.chain(key)
                states = MD.chain_states(law['x0'], law['beta'], law['luts'], chain_bank, args.device)
                before, after = states[-1].reshape(*law['hw'],3), states[0].reshape(*law['hw'],3)
                visual = np.asarray(source.vlm_image(key)[0], dtype=np.uint8).copy()
                instruction = instructions[key]
            aa = dict(visual=visual, before=before.cpu().numpy().astype(np.float32),
                      after=after.cpu().numpy().astype(np.float32))
            if reference is not None:
                rr = reference.arrays(key)
                if not np.array_equal(aa['visual'], rr['visual']):
                    raise ValueError(f'Visual before mismatch: {key}')
                for name in ('before', 'after'):
                    delta = float(np.abs(aa[name] - rr[name]).max())
                    max_difference[name] = max(max_difference[name], delta)
                    if delta > 2e-6: raise ValueError(f'Float endpoint parity failure: {key}/{name}: {delta}')
            prefix = str(j)
            record = dict(key=key, instruction=instruction, shard=shard, prefix=prefix,
                          sha256={n:array_sha(a) for n,a in aa.items()})
            validate_row(record)
            for n,a in aa.items(): arrays[prefix+'_'+n] = a
            pending.append(record)
        tmp = out / (shard + '.partial')
        with tmp.open('wb') as f: np.savez_compressed(f, **arrays)
        tmp.replace(out / shard); records.extend(pending)
        write(out / 'progress.json', dict(exported=len(records), total=len(needed), seconds=time.time()-started))
        print(json.dumps(dict(event='export', n=len(records), total=len(needed))), flush=True)
    write(out / 'pairs.json', dict(schema='epr081-pairs-v1', rows=records))
    write(out / 'export_audit.json', dict(complete=not args.limit, n=len(records),
        float_endpoints=True, visual='matched control observed before image',
        exporter_sha256=sha(__file__), device=args.device,
        source_sha256={str(Path(m.__file__)):sha(m.__file__) for m in (SD, MX, MD, E71)},
        parity_reference=args.compare_to, max_float_difference=max_difference if reference else None,
        note='Only offline construction reads trajectories; training boundary is pair-only.'))


if __name__ == '__main__': main()
