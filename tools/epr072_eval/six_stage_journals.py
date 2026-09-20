"""Generate input-only AR1600 six-stage CoT for the expert benchmarks."""
import argparse
import json
from pathlib import Path
import sqlite3

import torch

from tools.epr072_eval.public_six_stage import records_for, dump, digest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--batch', type=int, default=8)
    args = ap.parse_args()
    from veraretouch_sprf.readout.model import ReadoutVLM, AR_ADAPTER
    from veraretouch_sprf.readout.artedit_eval import segments_of
    from tools.epr075_fig4.cot import generate, IM_END
    torch.set_num_threads(4)
    torch.manual_seed(20260920)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.25)
    holder = ReadoutVLM(device='cuda:0', grad_checkpointing=False)
    holder.model.base_model.set_adapter([AR_ADAPTER])
    holder.model.active_adapter = AR_ADAPTER
    holder.model.eval()
    for bench in ('fivek', 'ppr10k'):
        path = args.out/(bench+'.json')
        prior = json.loads(path.read_text()) if path.exists() else {}
        rows = prior.get('rows', [])
        seen = {r['sample_id'] for r in rows}
        records = [dict(sample_id=r['sample_id'], input_png=r['input_path'], instruction=r['instruction'])
                   for r in records_for(bench)]
        todo = [r for r in records if r['sample_id'] not in seen]
        for start in range(0, len(todo), args.batch):
            block = todo[start:start+args.batch]
            new = generate(holder.model, holder.processor, holder.pad_id, block,
                           lambda r:r['instruction'], '', 2048, (IM_END,), 'cuda:0', batch=args.batch)
            for r, source in zip(new, block):
                seg, reason = segments_of(r, holder.stage_ids)
                r.update(ok=seg is not None, reason=reason,
                         input_sha256=digest(source['input_png']), instruction=source['instruction'])
            rows.extend(new)
            dump(path, dict(engine='AR1600, AR-B1 adapter only, HF greedy',
                            batch=args.batch, max_new_tokens=2048, target_access=False,
                            expected=len(records), n=len(rows), n_valid=sum(r['ok'] for r in rows),
                            frozen=holder.frozen_meta, rows=rows))
            print(f'JOURNAL {bench} {len(rows)}/{len(records)} valid={sum(r["ok"] for r in rows)}', flush=True)
        if {r['sample_id'] for r in rows} != {r['sample_id'] for r in records}:
            raise ValueError('Journal coverage mismatch')
        if not all(r['ok'] for r in rows):
            raise ValueError('Malformed generated CoT retained for audit; no silent repair/fallback')


if __name__ == '__main__':
    main()
