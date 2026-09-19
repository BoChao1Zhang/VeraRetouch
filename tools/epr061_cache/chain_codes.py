"""EPR-061-D2 step 1: per-chain, per-stage closed-form GLUT codes for the degradation chains.

For every chain in ``plan_trainfull[:70000] + plan_val + plan_heldout`` and every colour
stage ``k in 1..5`` (STEP_KIND = geom, lum_high, lum_mid, lum_shadow, hue, global; slot 0 =
subject/where is skipped) the chain is rebuilt with the journal's own rebuilders and two
codes are solved on the frozen ``geometry_N64`` basis (D = 260, ridge 1e-2, trace-scaled,
no Laplacian):

  allpix     absolute-color E4 ridge least squares over EVERY canonical pixel,
             H=beta*phi(z_k), T=(prev_k-z_k)+beta*z_k.
  uniform    unweighted ridge least squares over 4,096 uniform [0,1]^3 queries,
             target F(x) = 2*x - L_k(x). (identity plus the image-free residual
             contrast; not an exact inverse LUT)

Both are stored in one member ``<split>/<seq>_<slot>.f32`` of shape (2, 3, 260); row 0 is
allpix, row 1 is uniform.  The sqlite index carries lut_id, strength, beta coverage, code
norms, the split-half stability sample and the rendered-M capacity columns.

CPU workers do the journal read, the asset read and ``alpha_fields``; the parent does the
chain rendering, the features, the solves and the M pass on the GPU.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import resource
import sqlite3  # Load sqlite's C++ runtime before torch (shared training environment).
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, '/home/bc/VeraRetouch')
from tools.epr061_cache import chain_common as C

WORKER = {}
INDEX = None


# --------------------------------------------------------------------------- #
# CPU side
# --------------------------------------------------------------------------- #
def init_worker():
    import torch as _torch
    from veraretouch_sprf.data import train_stage0 as T0
    _torch.set_num_threads(1)
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard), hard))
    WORKER['reader'] = C.AssetReader(T0)
    WORKER['T0'] = T0
    WORKER['index'] = INDEX          # inherited copy-on-write across the fork
    return os.getpid()


def prepare(task):
    """-> (key, seq, split, src uint8 HW3, after uint8 HW3, beta float32 6xP, row facts)."""
    key, seq, split = task
    began = time.monotonic()
    T0, reader, index = WORKER['T0'], WORKER['reader'], WORKER['index']
    for attempt in range(4):
        try:
            row = C.journal_row(index, key)
            entry = index[key]
            src = reader.uint8(entry, f"{row['id']}.src.png")
            after = reader.uint8(entry, entry['asset'])
            break
        except OSError:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    if src.shape != after.shape:
        raise ValueError(f'Target/input geometry mismatch: {key}')
    target = torch.from_numpy(src.astype(np.float32) / 255.0)
    beta = T0.alpha_fields(row, target).reshape(6, -1).contiguous()
    return dict(key=key, seq=seq, split=split, src=src, after=after,
                beta=beta.numpy(), luts=list(row['luts']), grids=list(row['grids']),
                strength=float(row['calib']['s']), size=list(row['size']),
                geom=row['mask']['geom'], cpu_seconds=time.monotonic() - began)


def prepare_guarded(task):
    try:
        return prepare(task)
    except BaseException as error:              # a single bad chain must not stop the run
        return dict(key=task[0], seq=task[1], split=task[2], error=repr(error)[:400])


# --------------------------------------------------------------------------- #
# GPU side
# --------------------------------------------------------------------------- #
class Solver:
    """One feature buffer per chain, reused by the gram, the solve and the M pass."""

    def __init__(self, device, geometry=C.GEOMETRY, ridge=C.RIDGE, chunk=131072):
        from veraretouch_sprf.data import stage_targets
        self.device = device
        self.glut = C.Glut(geometry, device, ridge)
        self.stage_targets = stage_targets
        self.chunk = chunk

    def features(self, colors, out):
        """colors (S,P,3) -> out (S,P,D), filled in pixel chunks."""
        for start in range(0, colors.shape[1], self.chunk):
            stop = min(start + self.chunk, colors.shape[1])
            out[:, start:stop] = self.glut.features(colors[:, start:stop])
        return out

    def solve_absolute(self, gram, rhs):
        codes, empty = self.glut.solve(gram, rhs)
        # Empty masks carry no information. Store the absolute identity code,
        # while the legacy residual solver retains its zero-code convention.
        identity = torch.zeros_like(codes)
        identity[..., :, -4:-1] = torch.eye(3, device=codes.device, dtype=codes.dtype)
        codes = torch.where(empty[..., None, None], identity, codes)
        return codes, empty

    def statistics(self, phi, colors, prev, beta, rows=None):
        """Absolute-color E4 normal equations, optionally on selected pixels.

        ``rows`` selects a pixel subset (the split-half control).  Per-stage ``mm`` rather
        than one ``bmm`` -- measured 2x on this card.
        """
        grams, rhs = [], []
        for stage in range(beta.shape[0]):
            select = slice(None) if rows is None else rows
            gram, target = C.absolute_statistics(phi[stage][select], colors[stage][select],
                                                 prev[stage][select], beta[stage][select])
            grams.append(gram)
            rhs.append(target)
        return torch.stack(grams), torch.stack(rhs)

    def render_error(self, phi, colors, prev, beta, variants):
        """255 * max-channel |z + beta*(F(z)-z) - prev| for each code variant."""
        stages, pixels = beta.shape
        support = (beta > 0).sum(-1).clamp_min(1)
        out = {}
        for name, codes in variants.items():
            total = torch.zeros(stages, device=self.device)
            active = torch.zeros(stages, device=self.device)
            for stage in range(stages):
                predicted = colors[stage] if codes is None else (
                    colors[stage] + beta[stage].unsqueeze(-1)
                    * (phi[stage] @ codes[stage].transpose(0, 1) - colors[stage]))
                error = (predicted - prev[stage]).abs().amax(-1) * 255.0
                total[stage] = error.sum()
                active[stage] = (error * (beta[stage] > 0)).sum()
            out[name] = (total / pixels, active / support)
        return out


def pixel_chunks(pixels, chunk):
    return [(s, min(s + chunk, pixels)) for s in range(0, pixels, chunk)]


@torch.no_grad()
def solve_chain(solver, bank, sample, split_half):
    device = solver.device
    src = torch.from_numpy(sample['src']).to(device).float().div_(255.0)
    after = torch.from_numpy(sample['after']).to(device).float().div_(255.0)
    beta = torch.from_numpy(sample['beta']).to(device)
    states, actions = solver.stage_targets.build_intermediates(
        src.reshape(-1, 3), beta, sample['luts'], bank, device=device)
    drift = float((states[-1] - after.reshape(-1, 3)).abs().max()) * 255.0
    slots = torch.tensor(C.COLOR_SLOTS, device=device)
    z, prev, b = states[slots + 1], states[slots], beta[slots]
    del states, actions, src, after, beta
    stages, pixels = b.shape
    phi = torch.empty(stages, pixels, solver.glut.size, device=device)
    solver.features(z, phi)
    gram, rhs = solver.statistics(phi, z, prev, b)
    codes, empty = solver.solve_absolute(gram, rhs)
    stability = None
    if split_half:
        rows = torch.arange(pixels, device=device)
        left, _ = solver.solve_absolute(*solver.statistics(phi, z, prev, b, rows[rows % 2 == 0]))
        right, _ = solver.solve_absolute(*solver.statistics(phi, z, prev, b, rows[rows % 2 == 1]))
        stability = (2.0 * (left - right).norm(dim=(-1, -2))
                     / (left + right).norm(dim=(-1, -2)).clamp_min(1e-12))
    queries = torch.stack([C.queries_for(f"{sample['key']}|{slot}")
                           for slot in C.COLOR_SLOTS]).to(device)
    lut_targets = []
    for position, slot in enumerate(C.COLOR_SLOTS):
        volume = bank.get(sample['luts'][slot], device, torch.float32)
        lut_targets.append(2 * queries[position]
                           - solver.stage_targets.apply_lut_px(volume, queries[position]))
    lut_targets = torch.stack(lut_targets)
    phi_q = solver.glut.features(queries)
    codes_uniform, _ = solver.glut.solve(phi_q.transpose(-1, -2) @ phi_q,
                                         phi_q.transpose(-1, -2) @ lut_targets)
    measure = solver.render_error(phi, z, prev, b,
                                  dict(all=codes, uniform=codes_uniform, identity=None))
    del phi
    return dict(drift=drift, codes=codes, codes_uniform=codes_uniform, empty=empty,
                stability=stability, measure=measure,
                coverage=(b > 0).float().mean(-1), beta_mean=b.mean(-1), pixels=pixels)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def tasks_for(splits, records_by_split, cursor):
    for split in splits:
        start = cursor.get(split, 0)
        for record in records_by_split[split][start:]:
            yield (record['key'], record['seq'], split)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits', default=','.join(C.SPLITS))
    parser.add_argument('--limit', type=int, default=0, help='chains per split (pilot)')
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--chunk', type=int, default=131072)
    parser.add_argument('--split-half-every', type=int, default=50)
    parser.add_argument('--name', default=C.ABSOLUTE_CACHE_NAME)
    parser.add_argument('--out', default=str(C.OUT))
    parser.add_argument('--commit-every', type=int, default=512)
    parser.add_argument('--gpu-fraction', type=float, default=0.20,
                        help='hard cap as a fraction of the card (0.20 = 19.6 GB on H100-80/97GB)')
    parser.add_argument('--no-resume', action='store_true')
    args = parser.parse_args()
    splits = args.splits.split(',')
    began = time.monotonic()

    global INDEX
    index = INDEX = C.load_index()
    records_by_split, plan_hashes = {}, {}
    for split in splits:
        plan, records = C.plan_records(split)
        if args.limit:
            records = records[:args.limit]
        records_by_split[split] = records
        plan_hashes[split] = plan['plan_sha256']
    ra, law = C.bind_build_law(index, records_by_split[splits[0]][0]['key'])

    from veraretouch_sprf.fit import subject_source
    wanted = [r['key'] for split in splits for r in records_by_split[split]]
    scan_began = time.monotonic()
    paths = C.subject_paths(index, wanted)
    payloads, mapping, subject_audit = subject_source.collect(paths)
    subject_audit['scan_seconds'] = time.monotonic() - scan_began
    subject_audit['chains_scanned'] = len(wanted)
    subject_audit['chains_needing_subject'] = len(paths)
    subject_source.install(payloads, mapping)
    print(json.dumps(dict(phase='subject', **subject_audit)), flush=True)

    meta = dict(task='EPR-061-D2', geometry=str(C.GEOMETRY), ridge=C.RIDGE,
                code_semantics=C.ABSOLUTE_CODE_SEMANTICS,
                executor='z + beta * (phi(z) @ W.T - z)',
                support_semantics='recorded strength-scaled beta; not support-only alpha',
                dims=260, stages=json.dumps(list(C.STAGES)), color_slots=json.dumps(list(C.COLOR_SLOTS)),
                build_law=json.dumps(law), plan_sha256=json.dumps(plan_hashes),
                lut_bank=ra['data']['lut_bank_dir'], refit_index=str(C.REFIT_ROOT / 'lean_index.json'),
                member_layout='(2,3,260) float32: row 0 = allpix beta^2-weighted, row 1 = uniform 4096',
                fit_allpix='E4 absolute fit: H=beta*phi(z), T=(prev-z)+beta*z',
                fit_uniform=f'unweighted absolute contrast target 2*q-L(q), {C.N_QUERIES} queries, seed sha256({C.QUERY_TAG}:<key>|<slot>)',
                trainfull_limit=C.TRAINFULL_LIMIT, limit=args.limit,
                split_half_every=args.split_half_every,
                counts=json.dumps({s: len(r) for s, r in records_by_split.items()}))

    writer = C.ResumableTar(args.name, out=Path(args.out), meta=meta, resume=not args.no_resume)
    pending_splits = [s for s in splits if s not in writer.done]
    if not pending_splits:
        print(json.dumps(dict(phase='already_complete', members=writer.members)), flush=True)
        return

    context = mp.get_context('fork')
    executor = ProcessPoolExecutor(max_workers=args.workers, mp_context=context,
                                   initializer=init_worker)
    # Force the fork before the parent touches CUDA so the index and the subject payloads
    # stay copy-on-write shared instead of being pickled per worker.
    warm = [executor.submit(_noop) for _ in range(args.workers * 2)]
    [f.result() for f in warm]

    device = torch.device('cuda:0')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(args.gpu_fraction)
    solver = Solver(device, chunk=args.chunk)
    from veraretouch_sprf.data import stage_targets
    bank = stage_targets.LutVolumes(ra['data']['lut_bank_dir'], 4200)

    lookahead = args.workers * 3
    stream = tasks_for(pending_splits, records_by_split, writer.cursor)
    futures, order = {}, []
    done, failures = 0, 0
    gpu_seconds, cpu_seconds = 0.0, 0.0
    last = time.monotonic()
    totals = {s: len(records_by_split[s]) for s in splits}

    def pump():
        while len(futures) < lookahead:
            task = next(stream, None)
            if task is None:
                return
            futures[task] = executor.submit(prepare_guarded, task)
            order.append(task)

    pump()
    try:
        while order:
            task = order.pop(0)
            sample = futures.pop(task).result()
            pump()
            key, seq, split = task
            if 'error' in sample:
                writer.fail(key, split, seq, sample['error'])
                failures += 1
            else:
                cpu_seconds += sample['cpu_seconds']
                started = time.monotonic()
                try:
                    result = solve_chain(solver, bank, sample,
                                         args.split_half_every and seq % args.split_half_every == 0)
                    if result['drift'] > 1.1:
                        raise ValueError(f"Reconstruction drift {result['drift']:.4f}")
                    codes = result['codes'].cpu().numpy()
                    codes_uniform = result['codes_uniform'].cpu().numpy()
                    stability = None if result['stability'] is None else result['stability'].cpu().numpy()
                    coverage = result['coverage'].cpu().numpy()
                    beta_mean = result['beta_mean'].cpu().numpy()
                    measure = {n: (a.cpu().numpy(), b.cpu().numpy()) for n, (a, b) in result['measure'].items()}
                    empty = result['empty'].cpu().numpy()
                    for position, slot in enumerate(C.COLOR_SLOTS):
                        payload = np.stack((codes[position], codes_uniform[position]))
                        writer.add(f'{split}/{seq:06d}_{slot}.f32', payload.reshape(6, 260),
                                   key=f'{key}|{slot}', split=split, seq=seq, slot=slot,
                                   chain_key=key, lut_id=sample['luts'][slot],
                                   strength=sample['strength'],
                                   beta_coverage=float(coverage[position]),
                                   beta_mean=float(beta_mean[position]),
                                   code_fro=float(np.linalg.norm(codes[position])),
                                   code_fro_uniform=float(np.linalg.norm(codes_uniform[position])),
                                   m_all_mean=float(measure['all'][0][position]),
                                   m_active_mean=float(measure['all'][1][position]),
                                   m_uniform_all_mean=float(measure['uniform'][0][position]),
                                   m_uniform_active_mean=float(measure['uniform'][1][position]),
                                   m_identity_all_mean=float(measure['identity'][0][position]),
                                   m_identity_active_mean=float(measure['identity'][1][position]),
                                   splithalf_rel=None if stability is None else float(stability[position]),
                                   empty_stage=int(empty[position]))
                    done += 1
                except BaseException as error:
                    writer.fail(key, split, seq, repr(error)[:400])
                    failures += 1
                gpu_seconds += time.monotonic() - started
            writer.cursor[split] = seq + 1
            processed = done + failures
            if processed % args.commit_every == 0 or not order:
                writer.commit()
                elapsed = time.monotonic() - began
                remaining = sum(totals[s] for s in pending_splits) - processed
                print(json.dumps(dict(phase='codes', split=split, seq=seq, done=done,
                                      failures=failures, members=writer.members,
                                      chains_per_second=processed / max(elapsed, 1e-3),
                                      gpu_seconds_per_chain=gpu_seconds / max(done, 1),
                                      cpu_seconds_per_chain=cpu_seconds / max(done, 1),
                                      eta_hours=remaining / max(processed / max(elapsed, 1e-3), 1e-9) / 3600,
                                      rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
                                      elapsed=elapsed)), flush=True)
                last = time.monotonic()
        for split in pending_splits:
            writer.done.add(split)
        writer.commit()
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    stats = writer.close(meta=dict(done=done, failures=failures,
                                   seconds=time.monotonic() - began,
                                   subject_audit=json.dumps(subject_audit)))
    C.atomic_json(Path(args.out) / f'{args.name}.summary.json',
                  dict(meta={k: v for k, v in meta.items()}, done=done, failures=failures,
                       seconds=time.monotonic() - began, **stats))
    print(json.dumps(dict(phase='complete', done=done, failures=failures, **stats)), flush=True)


def _noop():
    return os.getpid()


if __name__ == '__main__':
    main()
