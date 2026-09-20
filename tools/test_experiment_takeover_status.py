import json
from pathlib import Path
import tempfile
import unittest

from tools.experiment_takeover_status import endpoint_status, jsonl, style_status


class TakeoverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def put(self, name, value):
        path = self.root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value)+'\n')
        return path

    def test_partial_tail_only(self):
        path = self.put('rows.jsonl', {'key': 'a'})
        with path.open('a') as f:
            f.write('{"key":')
        self.assertEqual(jsonl(path), [{'key': 'a'}])
        with path.open('a') as f:
            f.write('\n')
        with self.assertRaises(json.JSONDecodeError):
            jsonl(path)

    def endpoint_fixture(self):
        self.put('config.json', dict(arm='ENDPT', total_steps=3200,
                                     schedule_steps=6250))
        self.put('run/steps.jsonl', dict(step=3200))
        self.put('run/evaluations/step003200.json', dict(step=3200,
                 local_single_l1=.02, local_rollout_l1=.05))
        self.put('run/global_full_final.json', dict(n=400, groups={'all': {
            'top1': {k: {'n': 400, 'mean': 1.0}
                     for k in ('l1', 'l2', 'psnr', 'de00')}}}))

    def test_requires_final_not_best(self):
        self.endpoint_fixture()
        (self.root/'run/best.pt').touch()
        self.assertFalse(endpoint_status(self.root)['ready'])
        (self.root/'run/final.pt').touch()
        self.assertTrue(endpoint_status(self.root)['ready'])
        self.put('config.json', dict(arm='ENDPT', total_steps=3200,
                                     schedule_steps=3200))
        self.assertFalse(endpoint_status(self.root)['ready'])

    def test_rejects_val50_and_nan(self):
        self.endpoint_fixture()
        (self.root/'run/final.pt').touch()
        full = json.loads((self.root/'run/global_full_final.json').read_text())
        full['groups']['all']['top1']['l1']['n'] = 50
        self.put('run/global_full_final.json', full)
        self.assertFalse(endpoint_status(self.root)['ready'])
        full['groups']['all']['top1']['l1'] = {'n': 400, 'mean': float('nan')}
        self.put('run/global_full_final.json', full)
        self.assertFalse(endpoint_status(self.root)['ready'])

    def test_style_requires_unique_complete_pairs(self):
        row = dict(method='m', lut_id='L', image_id='001',
                   **{k: 1.0 for k in ('psnr', 'ssim', 'de76', 'de00', 'lpips')})
        path = self.put('rows_0.jsonl', row)
        self.assertTrue(style_status(self.root, ['m'], ['L'], ['001'])['m']['ready'])
        self.assertFalse(style_status(self.root, ['m'], ['L'], ['001', '002'])['m']['ready'])
        with path.open('a') as f:
            f.write(json.dumps(row)+'\n')
        self.assertFalse(style_status(self.root, ['m'], ['L'], ['001'])['m']['ready'])


if __name__ == '__main__':
    unittest.main()
