"""CPU boundary and matched-budget tests, independent of private data."""
import unittest
from veraretouch_sprf.readout.epr081_pairs import validate_row, single_sequence
from veraretouch_sprf.readout.epr081_train import multiplier


class BoundaryTests(unittest.TestCase):
    def row(self):
        return dict(key='x',instruction='Brighten the face while preserving the sky.',
                    shard='x.npz',prefix='0',sha256=dict(visual='a',before='b',after='c'))
    def test_only_pair_fields(self):
        validate_row(self.row())
        for name in ('mask','cot','states','lut_id','code','stage_positions'):
            r=self.row();r[name]='leak'
            with self.assertRaises(ValueError):validate_row(r)
    def test_no_scaffold(self):
        r=self.row();r['instruction']+='\nObservation: injected process'
        with self.assertRaises(ValueError):validate_row(r)
    def test_single_readout(self):
        ids=list(range(100,108));self.assertEqual(single_sequence([1,2],ids,[50]),[1,2]+ids)
        for prompt in ([1,50],[1,100]):
            with self.assertRaises(ValueError):single_sequence(prompt,ids,[50])
    def test_schedule(self):
        import math
        for step in range(3200):
            expected=min((step+1)/100,1.)*.5*(1+math.cos(math.pi*max(step-100,0)/6150))
            self.assertEqual(multiplier(step),expected)
    def test_float_pair_roundtrip_and_corruption(self):
        import json
        from pathlib import Path
        import tempfile
        import numpy as np
        from veraretouch_sprf.readout.epr081_pairs import PairStore, array_sha
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            before=np.full((2,3,3),0.1234567,dtype=np.float32)
            arrays=dict(visual=np.zeros((2,3,3),dtype=np.uint8),before=before,after=before*.7)
            row=self.row();row['sha256']={n:array_sha(v) for n,v in arrays.items()}
            np.savez_compressed(root/'x.npz',**{'0_'+n:v for n,v in arrays.items()})
            (root/'pairs.json').write_text(json.dumps(dict(schema='epr081-pairs-v1',rows=[row])))
            store=PairStore(root)
            self.assertTrue(np.array_equal(store.arrays('x')['before'],before))
            store.rows['x']['sha256']['before']='wrong'
            with self.assertRaises(ValueError):store.arrays('x')


if __name__=='__main__':unittest.main()
