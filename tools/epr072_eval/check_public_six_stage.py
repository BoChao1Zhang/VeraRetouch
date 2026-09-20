"""CPU-only acceptance checks for archived six-stage public predictions."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('folder', type=Path)
    args = ap.parse_args()
    manifest = json.loads((args.folder/'manifest.json').read_text())
    summary = json.loads((args.folder/'summary.json').read_text())
    assert manifest['slots'] == [5, 4, 3, 2, 1, 0]
    assert manifest['target_access_during_inference'] is False
    ids, values = [], {k: [] for k in ('l1', 'l2', 'psnr', 'ssim', 'de00')}
    for part in sorted(args.folder.glob('part_*.json')):
        info = json.loads(part.read_text())
        archive = part.with_suffix('.tar')
        digest = hashlib.sha256()
        with archive.open('rb') as f:
            for block in iter(lambda: f.read(1 << 20), b''):
                digest.update(block)
        assert digest.hexdigest() == info['tar_sha256']
        with tarfile.open(archive) as pack:
            for r in info['rows']:
                sid = r['sample_id']
                ids.append(sid)
                assert [s['slot'] for s in r['stages']] == manifest['slots']
                assert all(s['zero_support_unchanged'] for s in r['stages'])
                assert all(np.isfinite(s['actual_change_l1']) for s in r['stages'])
                png = pack.extractfile(sid+'.png').read()
                assert hashlib.sha256(png).hexdigest() == r['prediction_sha256']
                image = Image.open(io.BytesIO(png))
                image.load()
                assert image.size == (r['w'], r['h'])
                codes = np.load(io.BytesIO(pack.extractfile(sid+'.codes.npy').read()))
                assert codes.shape == (6, 780) and np.isfinite(codes).all()
                for k in values:
                    assert np.isfinite(r[k])
                    values[k].append(r[k])
    assert ids == manifest['sample_ids'] and len(ids) == len(set(ids))
    assert summary['complete'] and summary['n'] == len(ids)
    for k, v in values.items():
        assert abs(float(np.mean(v))-summary['means'][k]) < 1e-12
    print(json.dumps(dict(status='PASS', n=len(ids), stages_per_image=6,
                          native_size=True, zero_support_identity=True,
                          archive_and_code_integrity=True)))


if __name__ == '__main__':
    main()
