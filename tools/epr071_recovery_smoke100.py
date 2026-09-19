"""R@800 smoke + fresh pair-conditioned recovery on 100 fixed ArtEdit pairs.

Oracle fits use target pixels and are capacity diagnostics, not model predictions.
The inverse baseline inverts a FITTED target->input LUT on 4096 fixed pixels;
the unknown physical forward editing operator is not available on ArtEdit.
"""
import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3  # Load sqlite runtime before torch.
import time

import numpy as np
import torch

from tools.epr071_val50_diag import (STORE, PROTOSET, JOURNAL, HALVES,
                                    load_checkpoint, internal_l1, geometry_of,
                                    reach_rows, sha256_file, stats)
from veraretouch_sprf.readout import artedit_eval as AE, epr071_data as E71
from veraretouch_sprf.readout import select_train as ST, mixed_codes as MC
from tools.epr059_glutbasis.common import apply_codes, glut_features
from tools.epr065_dict_reach import render_u8
from q3vl.whatb.lutdata import apply_lut_volume


def emit(event, **kwargs):
    print(json.dumps(dict(event=event, **kwargs)), flush=True)


def write_json(path, obj):
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


@torch.no_grad()
def invert_fitted_lut(volume, observed, iterations=30):
    """Damped GN; finite differences use their actual clipped spacing."""
    value = observed.clone()
    eye = torch.eye(3, device=value.device)
    best = value.clone()
    best_error = (apply_lut_volume(volume, value) - observed).square().sum(-1)
    for _ in range(iterations):
        residual = apply_lut_volume(volume, value) - observed
        columns = []
        for axis in range(3):
            offset = torch.zeros(3, device=value.device)
            offset[axis] = .5 / 32
            plus, minus = (value + offset).clamp(0, 1), (value - offset).clamp(0, 1)
            spacing = (plus[:, axis] - minus[:, axis]).clamp_min(1e-8)
            columns.append((apply_lut_volume(volume, plus) -
                            apply_lut_volume(volume, minus)) / spacing[:, None])
        jac = torch.stack(columns, -1)
        system = jac.transpose(-1, -2) @ jac + 1e-3 * eye
        rhs = (jac.transpose(-1, -2) @ residual[..., None])
        delta = torch.linalg.solve(system, rhs).squeeze(-1)
        value = (value - delta).clamp(0, 1)
        error = (apply_lut_volume(volume, value) - observed).square().sum(-1)
        improve = error < best_error
        best = torch.where(improve[:, None], value, best)
        best_error = torch.minimum(best_error, error)
    return best, best_error


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--micro', type=int, default=2,
                    help='Match the original EPR-071 evaluation batch size.')
    ap.add_argument('--checkpoint', default=str(STORE / 'train/R/run/best.pt'))
    ap.add_argument('--seed', type=int, default=20260919)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'summary.json').exists():
        raise SystemExit('Finished output already exists; use a new --out.')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.45)
    val, _ = AE.load_split('val50')
    rest, _ = AE.load_split('rest')
    chosen = np.random.default_rng(args.seed).choice(len(rest), 50, replace=False)
    records = val + [rest[int(i)] for i in sorted(chosen)]
    assert len(records) == 100 and len({r['sample_id'] for r in records}) == 100
    ids = [r['sample_id'] for r in records]
    manifest = dict(seed=args.seed, n=100, model_micro=args.micro,
                    selection='all val50 + 50 seeded rest',
                    sample_ids=ids, records=records, geometry=str(AE.GEOMETRY),
                    fit='input->target fresh ridge LS; target available to oracle only',
                    inverse='fit target->input; damped GN inverse on 4096 common pixels',
                    bank='existing 5075-prototype GT top20-screen/full-res oracle',
                    source_sha256=sha256_file(Path(__file__)))
    if (out / 'manifest.json').exists():
        old = json.loads((out / 'manifest.json').read_text())
        if old['sample_ids'] != ids:
            raise SystemExit('Resume sample selection changed')
    write_json(out / 'manifest.json', manifest)
    shutil.copyfile(__file__, out / 'source_main.py')
    checkpoint = out / 'R_best800_snapshot.pt'
    if not checkpoint.exists():
        shutil.copyfile(args.checkpoint, checkpoint)
    started = time.monotonic()
    bank = ST.Bank(args.device, path=PROTOSET)
    mean, std, scaler_step = E71.load_scaler()
    cached_predictions = out / 'model_predictions.npz'
    if cached_predictions.exists():
        raw_model = np.load(cached_predictions)['raw_codes']
        facts = json.loads((out / 'checkpoint.json').read_text())
        emit('reuse_model_predictions', step=facts['step'])
    else:
        model, facts = load_checkpoint(checkpoint, bank, args.device)
        if facts['arm'] != 'R' or facts['step'] != 800:
            raise SystemExit(f'Expected R@800, got {facts}')
        model.model.eval()
        model.head.eval()
        evaluator = ST.SelectEval(model, bank, args.device, micro=args.micro, num_workers=0,
                                  render_chunk=3, halves_path=HALVES, arm='R',
                                  code_mean=mean, code_std=std, journal=str(JOURNAL))
        _, standardized = evaluator.predict(records)
        raw_model = standardized * std + mean
        np.savez(cached_predictions, raw_codes=raw_model)
        write_json(out / 'checkpoint.json', facts)
        del evaluator, model, standardized
        gc.collect()
        torch.cuda.empty_cache()
        emit('model_complete', seconds=time.monotonic()-started, **facts)
    renderer = AE.GlutRenderer(AE.GEOMETRY, args.device)
    geometry = renderer.basis.geometry
    reaches = reach_rows()
    mean_t = torch.as_tensor(mean, device=args.device)
    std_t = torch.as_tensor(std, device=args.device)
    bank_std = (bank.codes.reshape(len(bank.ids), -1) - mean_t) / std_t
    # Existing val50 cache is compared, never silently reused as the fresh solve.
    from tools.epr062_val50_diag import oracle_codes
    old_codes, old_facts = oracle_codes()
    manifest['old_cache'] = old_facts
    manifest['scaler_step'] = scaler_step
    write_json(out / 'manifest.json', manifest)
    all_codes, forward_codes, rows = [], [], []
    stream = (out / 'rows.jsonl').open('w')
    for i, rec in enumerate(records):
        tick = time.monotonic()
        src, gt = AE.load_rgb_u8(rec['input_path']), AE.load_rgb_u8(rec['gt_path'])
        if src.shape != gt.shape:
            raise ValueError(f"Unaligned dimensions: {rec['sample_id']}")
        z = torch.as_tensor(src.astype(np.float32)/255, device=args.device).reshape(-1,3)
        target = torch.as_tensor(gt.astype(np.float32)/255, device=args.device).reshape(-1,3)
        # This is recovery in the actual desired direction, not forward grading
        # and not selection from the prototype bank.
        code = MC.solve_support_code(z, target, None, geometry)
        forward = MC.solve_support_code(target, z, None, geometry)
        all_codes.append(code.cpu().numpy())
        forward_codes.append(forward.cpu().numpy())
        fitted = MC.apply_support(code, z, None, geometry)
        proto_index = int(reaches[rec['sample_id']]['proto5075_best_index'])
        codes = torch.stack((torch.as_tensor(raw_model[i], device=args.device).reshape_as(code),
                             code, bank.codes[proto_index]))
        volumes = apply_codes(renderer.phi, codes).clamp(0,1).reshape(-1,33,33,33,3)
        volumes = volumes.permute(0,4,1,2,3).contiguous()
        rendered_u8 = render_u8(volumes, z)
        row = dict(sample_id=rec['sample_id'], group='val50' if i < 50 else 'rest50',
                   pixels=len(z), identity_l1=float((z.double()-target.double()).abs().mean())*100,
                   continuous_fit_l1=float((fitted-target).abs().mean())*100,
                   continuous_clipped_l1=float((fitted.clamp(0,1)-target).abs().mean())*100)
        for j, name in enumerate(('model','closed','bank')):
            prediction = rendered_u8[j].double()/255
            metrics = geometry_of(z.double(),target.double(),prediction)
            row.update({f'{name}_{k}':v for k,v in metrics.items()})
            row[f'{name}_l1'] *= 100
            gt_magnitude = float((target-z).abs().mean())
            row[f'{name}_amplitude_ratio'] = metrics['m_pred']/max(gt_magnitude,1e-12)
            if name == 'closed':
                error = (prediction-target.double()).square().mean()
                row['closed_psnr'] = float(-10*torch.log10(error.clamp_min(1e-20)))
        distances = ((bank_std - (code.flatten()-mean_t)/std_t)**2).mean(-1)
        row['code_bank_min_standardized_rmse'] = float(distances.min().sqrt())
        row['code_bank_nearest_index'] = int(distances.argmin())
        row['bank_reference_l1_gap'] = abs(row['bank_l1']/100 -
                                          float(reaches[rec['sample_id']]['proto5075_best_l1']))
        if rec['sample_id'] in old_codes:
            old = torch.as_tensor(old_codes[rec['sample_id']], device=args.device).reshape_as(code)
            row['old_code_max_abs_diff'] = float((code-old).abs().max())
            old_render = renderer.apply(old,z).clamp(0,1)
            row['old_cached_l1'] = float(((old_render*255+.5).to(torch.uint8).double()/255-target.double()).abs().mean())*100
        # Paired inverse comparison on exactly the same reproducible subset.
        gen = torch.Generator().manual_seed(args.seed+i)
        sel = torch.randperm(len(z), generator=gen)[:4096].to(args.device)
        inverse, inv_equation_error = invert_fitted_lut(renderer.volume(forward),z[sel])
        direct = apply_lut_volume(volumes[1:2],z[sel]).clamp(0,1)
        row['inverse_probe_l1'] = float((inverse-target[sel]).abs().mean())*100
        row['direct_probe_l1'] = float((direct-target[sel]).abs().mean())*100
        row['inverse_equation_rmse'] = float(inv_equation_error.mean().sqrt())
        row['forward_fit_probe_l1'] = float((renderer.apply(forward,target[sel])-z[sel]).abs().mean())*100
        row['seconds'] = time.monotonic()-tick
        rows.append(row)
        stream.write(json.dumps(row)+'\n'); stream.flush()
        emit('sample', index=i+1, sample_id=rec['sample_id'], model=row['model_l1'],
             closed=row['closed_l1'], bank=row['bank_l1'], seconds=row['seconds'])
        del z,target,fitted,rendered_u8,volumes,codes,code,forward
    stream.close()
    np.savez(out / 'rebuilt_supervision_codes.npz', sample_ids=np.array(ids),
             restoration_codes=np.stack(all_codes), forward_codes=np.stack(forward_codes))
    with (out/'rows.csv').open('w') as f:
        fields=sorted({k for row in rows for k in row})
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    groups={}
    for group in ('all','val50','rest50'):
        selected=[r for r in rows if group=='all' or r['group']==group]
        metrics={k:stats([r[k] for r in selected]) for k in rows[-1]
                 if isinstance(rows[-1][k],(float,int)) and all(k in r for r in selected)}
        groups[group]=dict(n=len(selected),metrics=metrics,
                          closed_beats_bank=sum(r['closed_l1']<r['bank_l1'] for r in selected),
                          closed_beats_model=sum(r['closed_l1']<r['model_l1'] for r in selected),
                          direct_beats_inverse=sum(r['direct_probe_l1']<r['inverse_probe_l1'] for r in selected))
    logged, log_path=internal_l1(STORE/'train/R/run',800)
    smoke=groups['val50']['metrics']['model_l1']['mean']
    summary=dict(checkpoint=facts,groups=groups,seconds=time.monotonic()-started,
                 smoke_reference=logged,smoke_here=smoke,smoke_abs_diff=abs(smoke-logged),
                 smoke_log=log_path,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
                 limitations=['Target-conditioned fitting, not target-free model inference.',
                              'ArtEdit forward operator unknown; inverse baseline uses a fitted forward LUT.',
                              'Inverse comparison uses 4096 pixels per image; main errors use all pixels.',
                              'Bank comparator uses an existing downsample-screen/top20 full-resolution search.',
                              '50 hard validation + 50 rest samples is not the full-400 benchmark distribution.'])
    write_json(out/'summary.json',summary)
    if summary['smoke_abs_diff'] > 1e-4:
        raise RuntimeError(f'Smoke did not reproduce training log: {summary["smoke_abs_diff"]}')
    emit('complete',out=str(out),seconds=summary['seconds'],smoke_abs_diff=summary['smoke_abs_diff'])


if __name__=='__main__':
    main()
