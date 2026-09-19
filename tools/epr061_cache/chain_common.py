"""EPR-061-D2 shared helpers: chain sources, GLUT basis, closed-form codes, tar+sqlite writer.

Nothing here writes into any existing run root.  The only writable destination is
``OUT`` (the rclone VFS mount).
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import sqlite3
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

OUT = Path('/home/bc/nfsvfs/bc/data/runs/epr061_fulldata_20260917/chain_codes')
SCRATCH = Path('/dev/shm/epr061_chain_codes')
GEOMETRY = Path('/home/bc/nfsvfs/bc/data/runs/epr059_glutbasis_20260916/geometry_N64.pt')
EXPORT = Path('/home/bc/data/runs/epr058_color/cache_prepare_ar1600_v1')
JOURNAL_ROOT = Path('/home/bc/data/runs/epr058_color/cache_journal_ar1600')
REFIT_ROOT = Path('/home/bc/data/runs/epr055_sprfv3/refit_20260911')
BK_RUN = Path('/home/bc/data/runs/epr051_sprf/bkfull_adagn_ff_affhead')
FAMILY_SPLIT = Path('/home/bc/data/runs/epr058_color/family_split.json')
STYLE_ROOT = Path('/home/bc/data/runs/epr058_color/stylelocal512_20260916_v1')
STYLE_CODES = Path('/home/bc/nfsvfs/bc/data/runs/epr059_glutbasis_20260916')
PROCESS_ROOT = Path('/home/bc/data/runs/epr058_color/process20k_glut033_p16384_gpu1_v1')
PROCESS_PLAN = PROCESS_ROOT / 'pixel_plan.json'   # the plan the p16384 packs were committed against
PROCESS_PARENT_PLAN = Path('/home/bc/data/runs/epr058_color/process20k_gpu1_v1/data_plan.json')

RIDGE = 1e-2
ABSOLUTE_CODE_SEMANTICS = 'absolute_color_v2'
LEGACY_CODE_SEMANTICS = 'unmasked_residual_v1'
ABSOLUTE_CACHE_NAME = 'codes_chain_absolute_v2'
QUERY_TAG = 'EPR061-D2'
N_QUERIES = 4096
# STEP_KIND order, asserted against the build law at run time.
STAGES = ('geom', 'lum_high', 'lum_mid', 'lum_shadow', 'hue', 'global')
COLOR_SLOTS = (1, 2, 3, 4, 5)
SPLITS = ('val', 'heldout', 'trainfull')
# 2026-09-17 用户裁定：heldout1464 不再留作评测，全链进阶段二训练；val128 留作次要监控。
# 0 = 取该 split 的全部计划记录。
TRAINFULL_LIMIT = 0
TRAIN_SEQUENCE = ('trainfull', 'heldout')   # 训练流拼接顺序
EVAL_SPLITS = ('val',)


def sha256_bytes(blob):
    return hashlib.sha256(blob).hexdigest()


def digest_json(payload):
    return sha256_bytes(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode())


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + '\n')
    tmp.replace(path)
    return path


# --------------------------------------------------------------------------- #
# chain sources
# --------------------------------------------------------------------------- #
def load_index():
    return json.loads((REFIT_ROOT / 'lean_index.json').read_text())


def bind_build_law(index, key):
    """Load the build TOML globals into ``epr050_build_degradation`` (STEP_KIND, PIX_CHUNK)."""
    from veraretouch_sprf.data import train_stage0 as T0
    ra = json.loads((BK_RUN / 'run_args.json').read_text())['config']
    T0.ASSET_MODE = ra['data']['asset_source']
    law = T0.bind_build_config([dict(shard=index[key]['dir'])],
                               ra['guard']['build_config_sha256_allowed'])
    if law['step_kind'] != list(STAGES) or law['n_steps'] != 6:
        raise ValueError(f'Unexpected build law: {law}')
    return ra, law


def plan_records(split):
    plan = json.loads((EXPORT / f'plan_{split}.json').read_text())
    records = plan['records']
    if split == 'trainfull' and TRAINFULL_LIMIT:
        records = records[:TRAINFULL_LIMIT]
    return plan, records


def journal_row(index, key):
    entry = index[key]
    with open(Path(entry['dir']) / 'pairs.jsonl', 'rb') as stream:
        stream.seek(entry['off'])
        row = json.loads(stream.read(entry['len']))
    if row['id'] != key.split('|')[0]:
        raise ValueError(f'Journal offset mismatch: {key}')
    if abs(float(row['calib']['s']) - entry['calib_s']) > 1e-9:
        raise ValueError(f'Strength mismatch: {key}')
    return row


def subject_paths(index, keys, workers=8):
    """``subject_png`` of every key whose mask geometry needs a hard subject mask.

    Same offset-direct read as ``epr055_refit_v2.journal_rows``, fanned out over shards
    because the serial scan of 71k rows costs ~19 min on this NFS mount.
    """
    from concurrent.futures import ThreadPoolExecutor
    groups = {}
    for key in keys:
        groups.setdefault(index[key]['dir'], []).append(key)

    def scan(item):
        shard, group = item
        found = []
        for attempt in range(6):
            try:
                with (Path(shard) / 'pairs.jsonl').open('rb') as stream:
                    for key in group:
                        entry = index[key]
                        stream.seek(entry['off'])
                        row = json.loads(stream.read(entry['len']))
                        if row['id'] != key.split('|')[0]:
                            raise ValueError(f'Journal offset mismatch: {key}')
                        if row['mask']['geom'] in ('subject', 'semantic'):
                            found.append(row['subject_png'])
                return found
            except OSError:
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [path for chunk in pool.map(scan, groups.items()) for path in chunk]


class AssetReader:
    """``assets/`` first, ``archive/`` indexed tar otherwise -- the ObservableDataset rule."""

    def __init__(self, t0, limit=8):
        from collections import OrderedDict
        self.t0 = t0
        self.limit = limit
        self.archives = OrderedDict()

    def raw(self, entry, name):
        direct = Path(entry['dir']) / 'assets' / name
        if direct.is_file():
            return direct.read_bytes()
        from veraretouch_sprf.data.archive_assets import ArchiveAssets
        if entry['dir'] not in self.archives:
            self.archives[entry['dir']] = ArchiveAssets(Path(entry['dir']), self.t0)
            while len(self.archives) > self.limit:
                _, evicted = self.archives.popitem(last=False)
                for descriptor in evicted._fds.values():
                    os.close(descriptor)
        self.archives.move_to_end(entry['dir'])
        return self.archives[entry['dir']].read(name)

    def uint8(self, entry, name):
        with Image.open(io.BytesIO(self.raw(entry, name))) as image:
            return np.asarray(image.convert('RGB'), dtype=np.uint8)


# --------------------------------------------------------------------------- #
# GLUT basis / closed form (EPR-059 G1/G3 口径, ridge trace-scaled, no Laplacian)
# --------------------------------------------------------------------------- #
class Glut:
    def __init__(self, path, device, ridge=RIDGE):
        from q3vl.whatb.generator import SharedGeometry
        from tools.epr059_glutbasis.common import glut_features
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        self.geometry = SharedGeometry(int(checkpoint['n_gauss'])).to(device)
        self.geometry.load_state_dict(checkpoint['state_dict'])
        for parameter in self.geometry.parameters():
            parameter.requires_grad_(False)
        self._features = glut_features
        self.size = 4 * int(checkpoint['n_gauss']) + 4
        self.ridge = float(ridge)
        self.device = device
        self.eye = torch.eye(self.size, device=device)
        self.info = dict(path=str(path), n_gauss=int(checkpoint['n_gauss']),
                         step=int(checkpoint['step']), dims=self.size, ridge=self.ridge,
                         script_sha256=checkpoint.get('script_sha256'))

    def features(self, colors):
        return self._features(colors, self.geometry)

    def solve(self, gram, rhs):
        """``gram`` (..., D, D), ``rhs`` (..., D, 3) -> ``W`` (..., 3, D).

        An empty stage (beta identically zero, so gram == 0) gets the identity added the
        way ``sprf_v3.solve_coefficients`` does it, which returns the zero code instead of
        raising on a singular system.
        """
        scale = gram.diagonal(dim1=-2, dim2=-1).sum(-1) / self.size
        system = gram + (self.ridge * scale)[..., None, None] * self.eye
        system = system + (scale == 0)[..., None, None] * self.eye
        try:                       # SPD by construction; Cholesky is ~4x cusolver's LU here
            weights = torch.cholesky_solve(rhs, torch.linalg.cholesky(system)).transpose(-1, -2)
        except RuntimeError:
            weights = torch.linalg.solve(system, rhs).transpose(-1, -2)
        if not torch.isfinite(weights).all():
            raise RuntimeError('Non-finite coefficient solve')
        return weights, scale == 0


def weighted_statistics(features, target, beta):
    """``sufficient_statistics`` 口径: both sides scaled by beta, so the fit is beta^2 weighted."""
    weighted = features * beta.unsqueeze(-1)
    return (weighted.transpose(-1, -2) @ weighted,
            weighted.transpose(-1, -2) @ (target * beta.unsqueeze(-1)))


def absolute_statistics(features, colors, previous, beta):
    """E4 normal equations: H=beta*phi, T=(previous-colors)+beta*colors.

    Inputs support leading batch/stage dimensions. No mask division is used;
    the returned code represents absolute F(c), for z+beta*(F(z)-z).
    """
    if features.shape[:-1] != colors.shape[:-1] or colors.shape != previous.shape:
        raise ValueError('Feature/state pixel dimensions do not match')
    if colors.shape[-1] != 3 or beta.shape != colors.shape[:-1]:
        raise ValueError('Expected RGB states and one weight per pixel')
    h = features * beta.unsqueeze(-1)
    target = previous - colors + beta.unsqueeze(-1) * colors
    return h.transpose(-1, -2) @ h, h.transpose(-1, -2) @ target


def cache_semantics(index_path):
    """Old, unversioned D2 packs contain residual codes, not E4 absolute codes."""
    with sqlite3.connect(f'file:{Path(index_path)}?mode=ro', uri=True) as db:
        meta = dict(db.execute('SELECT k, v FROM meta'))
    return meta.get('code_semantics', LEGACY_CODE_SEMANTICS)


def check_cache_semantics(index_path, expected):
    actual = cache_semantics(index_path)
    if actual != expected:
        raise ValueError(f'{index_path}: code semantics {actual!r}, expected {expected!r}; '
                         'use a new output name and rebuild from recorded state pairs')


def queries_for(key, n=N_QUERIES, tag=QUERY_TAG):
    seed = int.from_bytes(hashlib.sha256(f'{tag}:{key}'.encode()).digest()[:8], 'little')
    return torch.rand((n, 3), generator=torch.Generator().manual_seed(seed % (2 ** 63 - 1)))


# --------------------------------------------------------------------------- #
# tar + sqlite, written straight into OUT so a long run is resumable
# --------------------------------------------------------------------------- #
COLUMNS = ('name', 'key', 'split', 'seq', 'slot', 'chain_key', 'lut_id', 'strength',
           'beta_coverage', 'beta_mean', 'code_fro', 'code_fro_uniform',
           'm_all_mean', 'm_active_mean', 'm_uniform_all_mean', 'm_uniform_active_mean',
           'm_identity_all_mean', 'm_identity_active_mean',
           'empty_stage', 'splithalf_rel', 'offset', 'size', 'sha256', 'rows', 'cols', 'dtype')


class ResumableTar:
    """USTAR members appended straight to ``OUT``; index kept in /dev/shm and copied at commits."""

    def __init__(self, name, out=OUT, meta=None, resume=True):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.tar_path = self.out / f'{name}.tar'
        self.db_path = SCRATCH / f'{name}.index.sqlite'
        self.out_db = self.out / f'{name}.index.sqlite'
        expected = (meta or {}).get('code_semantics')
        # Check BEFORE opening/truncating a pack or overwriting its metadata.
        if expected and self.out_db.exists():
            check_cache_semantics(self.out_db, expected)
        self.state_path = self.out / f'{name}.progress.json'
        state = json.loads(self.state_path.read_text()) if (resume and self.state_path.is_file()) else None
        if state and self.out_db.is_file():
            shutil.copyfile(self.out_db, self.db_path)
            self.stream = self.tar_path.open('r+b')
            self.stream.truncate(state['offset'])
            self.stream.seek(state['offset'])
            self.done = set(state['done_splits'])
            self.cursor = dict(state['cursor'])
        else:
            self.db_path.unlink(missing_ok=True)
            self.stream = self.tar_path.open('wb')
            self.done, self.cursor = set(), {}
        self.db = sqlite3.connect(self.db_path)
        self.db.executescript(
            'CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);'
            f'CREATE TABLE IF NOT EXISTS members({", ".join(COLUMNS)});'
            'CREATE UNIQUE INDEX IF NOT EXISTS members_name ON members(name);'
            'CREATE INDEX IF NOT EXISTS members_key ON members(key);'
            'CREATE INDEX IF NOT EXISTS members_chain ON members(chain_key);'
            'CREATE INDEX IF NOT EXISTS members_lut ON members(lut_id);'
            'CREATE INDEX IF NOT EXISTS members_split ON members(split, seq);'
            'CREATE TABLE IF NOT EXISTS failures(chain_key TEXT, split TEXT, seq INTEGER,'
            ' reason TEXT);')
        for key, value in (meta or {}).items():
            self.db.execute('INSERT OR REPLACE INTO meta VALUES(?,?)', (key, str(value)))
        self.db.commit()
        self.members = self.db.execute('SELECT count(*) FROM members').fetchone()[0]

    def add(self, member, payload, **fields):
        if len(member) >= 100:
            raise ValueError(f'member name too long for USTAR: {member}')
        blob = np.ascontiguousarray(payload, dtype=np.float32).tobytes()
        info = tarfile.TarInfo(member)
        info.size = len(blob)
        self.stream.write(info.tobuf(format=tarfile.USTAR_FORMAT))
        offset = self.stream.tell()
        self.stream.write(blob)
        self.stream.write(b'\0' * ((-len(blob)) % 512))
        row = dict(fields, name=member, offset=offset, size=len(blob),
                   sha256=sha256_bytes(blob), rows=payload.shape[0], cols=payload.shape[1],
                   dtype='float32')
        self.db.execute(f'INSERT OR REPLACE INTO members({", ".join(COLUMNS)}) '
                        f'VALUES({", ".join(":" + c for c in COLUMNS)})',
                        {c: row.get(c) for c in COLUMNS})
        self.members += 1

    def fail(self, chain_key, split, seq, reason):
        self.db.execute('INSERT INTO failures VALUES(?,?,?,?)', (chain_key, split, seq, reason))

    def commit(self, cursor=None, done=None):
        if cursor is not None:
            self.cursor.update(cursor)
        if done:
            self.done.add(done)
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.db.commit()
        shutil.copyfile(self.db_path, self.out_db.with_suffix('.sqlite.partial'))
        self.out_db.with_suffix('.sqlite.partial').replace(self.out_db)
        atomic_json(self.state_path, dict(offset=self.stream.tell(), members=self.members,
                                          cursor=self.cursor, done_splits=sorted(self.done),
                                          time=time.time()))

    def close(self, meta=None):
        self.stream.write(b'\0' * 1024)
        self.stream.flush()
        os.fsync(self.stream.fileno())
        size = self.stream.tell()
        self.stream.close()
        for key, value in (meta or {}).items():
            self.db.execute('INSERT OR REPLACE INTO meta VALUES(?,?)', (key, str(value)))
        self.db.execute('INSERT OR REPLACE INTO meta VALUES(?,?)', ('members', str(self.members)))
        self.db.execute('INSERT OR REPLACE INTO meta VALUES(?,?)', ('tar_bytes', str(size)))
        self.db.commit()
        self.db.close()
        shutil.copyfile(self.db_path, self.out_db.with_suffix('.sqlite.partial'))
        self.out_db.with_suffix('.sqlite.partial').replace(self.out_db)
        return dict(members=self.members, tar_bytes=size)
