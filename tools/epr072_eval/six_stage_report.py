"""Publish complete-only paired metric cells and exact coverage counts."""
import argparse
import csv
from datetime import datetime
import io
import json
import math
from pathlib import Path
import statistics
import subprocess

ROOT = Path('/home/bc/nfsvfs/bc/data/runs/epr072_sixstage_compare_20260921')
NO_COT = Path('/home/bc/nfsvfs/bc/data/runs/epr072_sixstage_public_20260920/full')
FIELDS = ['L1_x100', 'L2_x1000', 'PSNR', 'SSIM', 'DE00', 'SC', 'PQ', 'O', 'ArtiMuse', 'QAlign', 'DeQA']


def load(path):
    return json.loads(path.read_text()) if path.exists() else {}


def rows(path):
    if not path.exists():
        return []
    result = []
    lines = path.read_bytes().splitlines(keepends=True)
    for i, line in enumerate(lines):
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines)-1 or line.endswith(b'\n'):
                raise
    return result


def score(path, mode, column):
    valid = {}
    for row in rows(path):
        value = row.get(column)
        if (row['method'] == 'ours_local_pix3200_six_'+mode
                and isinstance(value, (int, float)) and math.isfinite(value)):
            valid[row['sample_id']] = value
    # Validate identities, not merely row count.
    expected = set(load(NO_COT/'artedit/manifest.json').get('sample_ids', []))
    return (statistics.mean(valid.values()) if len(expected) == 400 and set(valid) == expected else None,
            len(valid))


def atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.partial')
    tmp.write_text(text)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stop-timer')
    args = ap.parse_args()
    table, objective = [], []
    for mode, folder in [('nocot', NO_COT), ('cot', ROOT/'with_cot')]:
        result = dict(mode=mode, **{k: None for k in FIELDS})
        summary = load(folder/'artedit/summary.json')
        counts = {}
        for field, key, scale in [('L1_x100','l1',100), ('L2_x1000','l2',1000),
                                  ('PSNR','psnr',1), ('SSIM','ssim',1), ('DE00','de00',1)]:
            complete = summary.get('complete') and summary.get('n') == 400
            result[field] = summary['means'][key]*scale if complete else None
            counts[field] = summary.get('n',0) if complete else load(folder/'artedit/progress.json').get('n_completed',0)
        scores = ROOT/'scores'/mode
        for field, key in [('SC','sc'), ('PQ','pq'), ('O','o')]:
            result[field], counts[field] = score(scores/'rows_viescore_en.jsonl', mode, key)
        for field, name in [('ArtiMuse','artimuse'), ('QAlign','qalign'), ('DeQA','deqa')]:
            result[field], counts[field] = score(scores/f'rows_iaa_{name}_en.jsonl', mode, 'score')
        result['counts'] = counts
        result['complete'] = all(result[k] is not None for k in FIELDS)
        table.append(result)
        for bench, expected in [('fivek',498), ('ppr10k',492)]:
            summary = load(folder/bench/'summary.json')
            complete = bool(summary.get('complete') and summary.get('n') == expected)
            objective.append(dict(mode=mode, bench=bench, n=summary.get('n',0), complete=complete,
                                  means=summary.get('means') if complete else None))
    units = ['epr072-sixstage-cot-pipeline', 'epr072-sixstage-paired-iaa',
             'epr072-sixstage-nocot-judge', 'epr072-sixstage-cot-judge']
    services = {u: subprocess.run(['systemctl','--user','show',u+'.service','-p','ActiveState','-p','ExecMainStatus'],
                                  text=True, capture_output=True).stdout.strip() for u in units}
    complete = all(r['complete'] for r in table+objective)
    report = dict(time=datetime.now().astimezone().isoformat(), artedit=table,
                  objective=objective, services=services, complete=complete)
    atomic(ROOT/'comparison.json', json.dumps(report, indent=2)+'\n')
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=['mode']+FIELDS+['complete'])
    writer.writeheader()
    for r in table:
        writer.writerow({k:r[k] for k in writer.fieldnames})
    atomic(ROOT/'comparison.csv', buffer.getvalue())
    text = '# Six-stage PIX-3200: paired CoT comparison\n\n'
    text += 'Cells appear only after all 400 ArtEdit samples have valid scores; — means incomplete.\n\n'
    text += '| Mode | '+' | '.join(FIELDS)+' |\n|---|'+'---:|'*len(FIELDS)+'\n'
    for r in table:
        text += '| '+r['mode']+' | '+' | '.join('—' if r[k] is None else f'{r[k]:.4f}' for k in FIELDS)+' |\n'
    text += '\nCoverage per column: '+json.dumps({r['mode']:r['counts'] for r in table})+'\n'
    atomic(ROOT/'comparison.md', text)
    print(json.dumps(report), flush=True)
    if complete and args.stop_timer:
        subprocess.run(['systemctl','--user','stop',args.stop_timer], check=True)


if __name__ == '__main__':
    main()
