import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.epr072_eval import six_stage_report as R


class ReportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root/'artedit').mkdir()
        (self.root/'artedit/manifest.json').write_text(json.dumps({'sample_ids': [str(i) for i in range(400)]}))
        self.file = self.root/'rows.jsonl'

    def tearDown(self):
        self.temp.cleanup()

    def write_rows(self, n, method='ours_local_pix3200_six_cot'):
        self.file.write_text(''.join(json.dumps(dict(sample_id=str(i), method=method, sc=8))+'\n' for i in range(n)))

    def test_incomplete_not_reported(self):
        self.write_rows(399)
        with patch.object(R, 'NO_COT', self.root):
            self.assertEqual(R.score(self.file, 'cot', 'sc'), (None,399))

    def test_complete_and_deduplicated(self):
        self.write_rows(400)
        with self.file.open('a') as f:
            f.write(json.dumps(dict(sample_id='0', method='ours_local_pix3200_six_cot', sc=8))+'\n')
        with patch.object(R, 'NO_COT', self.root):
            self.assertEqual(R.score(self.file, 'cot', 'sc'), (8,400))
            self.assertEqual(R.score(self.file, 'nocot', 'sc'), (None,0))

    def test_wrong_id_rejected(self):
        self.write_rows(401)
        with patch.object(R, 'NO_COT', self.root):
            self.assertEqual(R.score(self.file, 'cot', 'sc'), (None,401))

    def test_partial_append_tolerated(self):
        self.write_rows(1)
        with self.file.open('a') as f:
            f.write('{')
        self.assertEqual(len(R.rows(self.file)),1)


if __name__ == '__main__':
    unittest.main()
