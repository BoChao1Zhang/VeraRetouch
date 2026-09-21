"""Score the archived ENDPT2 no-CoT single/global predictions, never six-stage PNGs."""
import hashlib
import json
from pathlib import Path
import sqlite3
import tarfile

from tools.epr080_sixstage_control.score_six import (
    OUT, CKPT_SHA, MANIFESTS, manifest_ids, sha256_file, scored_ids)


def main():
    method = 'epr080_endpt2_single_nocot'
    root = OUT / 'single_stage' / method
    stem = root / f'epr076_{method}'
    meta = json.loads(stem.with_suffix('.json').read_text())
    assert meta['n'] == 400 and meta['text_mode'] == 'empty'
    assert meta['checkpoint']['sha256'] == CKPT_SHA['endpt2']
    assert meta['executor']['mode'] == 'single'
    archive = stem.with_suffix('.tar')
    expected = meta['pack'] if 'pack' in meta else None
    if expected is None:
        expected = next(v for v in meta.values() if isinstance(v, dict) and 'n_members' in v)
    assert sha256_file(archive) == expected['tar_sha256']
    index = Path(str(stem) + '.index.sqlite')
    assert sha256_file(index) == expected['index_sha256']
    with sqlite3.connect(f'file:{index}?mode=ro', uri=True) as db:
        rows = db.execute('SELECT name,sample_id,sha256 FROM members').fetchall()
    ids = {r[1] for r in rows}
    assert ids == manifest_ids('artedit') and len(rows) == 400
    pred = Path('/dev/shm/epr080_single_viescore_nocot')
    pred.mkdir(exist_ok=True)
    with tarfile.open(archive) as pack:
        for name, sid, digest in rows:
            assert Path(sid).name == sid and name == sid + '.png'
            blob = pack.extractfile(name).read()
            assert hashlib.sha256(blob).hexdigest() == digest
            (pred / name).write_bytes(blob)
    out = OUT / 'scores' / 'endpt2_single' / 'nocot'
    out.mkdir(parents=True, exist_ok=True)
    from q3vl.whatb.pubbench import epr038b_metrics as M
    M.METHODS[method] = dict(dir=pred, instruction_field='instruction')
    from q3vl.whatb.pubbench import epr038b_viescore as V
    V.JUDGE_METHODS = [method]
    argv = ['--method', method, '--out-root', str(out),
            '--manifest', str(MANIFESTS['artedit']), '--judge-model', 'gpt-5.6-terra',
            '--attempts', '6', '--judge-long-edge', '1024']
    path = out / 'rows_viescore_en.jsonl'
    for attempt in range(4):
        if scored_ids(path, method, 'sc') == ids:
            break
        V.main(argv)
    assert scored_ids(path, method, 'sc') == ids, 'Incomplete scores; resume safely'
    print('SCORE_COMPLETE single/global ENDPT2 no-CoT n=400', flush=True)


if __name__ == '__main__':
    main()
