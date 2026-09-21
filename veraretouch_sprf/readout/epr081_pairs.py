"""Strict pair-only boundary for instruction-conditioned global continuation.

No trajectory reader, prototype target, CoT cache, or support is imported here.
The offline exporter alone knows how an endpoint pair was constructed.
"""
from collections import OrderedDict
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

FIELDS = {'key', 'instruction', 'shard', 'prefix', 'sha256'}
ARRAYS = ('visual', 'before', 'after')


def array_sha(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def validate_row(row):
    if set(row) != FIELDS:
        raise ValueError(f'Pair-only schema violation: {set(row) ^ FIELDS}')
    if not row['instruction'].strip():
        raise ValueError('Final user instruction is required')
    forbidden = ('<vr_stage_', 'Observation:', '\nMask:', '\nAdjustment:',
                 'Describe the requested forward', 'Finish the move with')
    if any(x in row['instruction'] for x in forbidden):
        raise ValueError(f"Process scaffold in final instruction: {row['key']}")
    if set(row['sha256']) != set(ARRAYS):
        raise ValueError('Every endpoint array must have a source digest')


def single_sequence(prompt_ids, readout_ids, stage_ids):
    if len(readout_ids) != 8 or len(set(readout_ids)) != 8:
        raise ValueError('Expected one shared eight-token readout group')
    if set(prompt_ids) & (set(readout_ids) | set(stage_ids)):
        raise ValueError('Process/readout tokens leaked into user prompt')
    return list(prompt_ids) + list(readout_ids)


class PairStore:
    """Whole-shard compressed NPZ residency with a bounded LRU (no tiny reads)."""
    def __init__(self, root, keep=2):
        self.root = Path(root)
        doc = json.loads((self.root / 'pairs.json').read_text())
        if set(doc) != {'schema', 'rows'} or doc['schema'] != 'epr081-pairs-v1':
            raise ValueError('Invalid pair manifest')
        self.rows = {}
        for row in doc['rows']:
            validate_row(row)
            if row['key'] in self.rows:
                raise ValueError('Duplicate pair key')
            self.rows[row['key']] = row
        self.keep = int(keep)
        self.cache = OrderedDict()

    def arrays(self, key):
        row = self.rows[key]
        name = row['shard']
        if name not in self.cache:
            while len(self.cache) >= self.keep:
                _, old = self.cache.popitem(last=False)
                old.close()
            self.cache[name] = np.load(io.BytesIO((self.root / name).read_bytes()), allow_pickle=False)
        self.cache.move_to_end(name)
        arrays = {n: self.cache[name][row['prefix'] + '_' + n] for n in ARRAYS}
        for n, value in arrays.items():
            if array_sha(value) != row['sha256'][n]:
                raise ValueError(f'Corrupt endpoint array {key}/{n}')
        if arrays['before'].dtype != np.float32 or arrays['after'].dtype != np.float32:
            raise ValueError('Loss endpoints must remain float32')
        if arrays['before'].shape != arrays['after'].shape:
            raise ValueError('Unaligned endpoints')
        return arrays

    def item(self, key, processor, readout_ids, stage_ids):
        from veraretouch_sprf.data.q3vl_text import encode_prompt
        a = self.arrays(key)
        row = self.rows[key]
        enc = encode_prompt(processor, Image.fromarray(a['visual']), row['instruction'])
        ids = single_sequence(enc['input_ids'][0].tolist(), readout_ids, stage_ids)
        return dict(key=key, ids=ids,
                    image={n: enc[n] for n in ('pixel_values', 'image_grid_thw')},
                    before=torch.from_numpy(a['before']).reshape(-1, 3),
                    after=torch.from_numpy(a['after']).reshape(-1, 3))
