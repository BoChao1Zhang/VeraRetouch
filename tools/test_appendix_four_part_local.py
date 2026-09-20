"""Integrity checks for the author-approved appendix panels."""
import json
import subprocess
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tools.build_appendix_four_part_local import ASSETS, DRAFTS
from tools.preview_three_part_local import window_sum


class LocalPanelsTest(unittest.TestCase):
    def test_integral_windows(self):
        a=np.arange(30).reshape(5,6)
        expected=np.array([[a[y:y+2,x:x+3].sum() for x in range(4)] for y in range(4)])
        np.testing.assert_array_equal(window_sum(a,3,2),expected)

    def test_true_crops_and_zero_outside(self):
        records=sorted(ASSETS.glob('case_*/audit.json'))
        self.assertEqual(len(records),6)
        for audit in records:
            r=json.loads(audit.read_text());folder=audit.parent;src=Path(r['source_folder'])
            with np.load(src/'float_states.npz') as data:
                masks=data['masks'];states=data['recovery']
            self.assertEqual(len(r['picked']),2)
            self.assertEqual([s['step'] for s in r['annotation']['cot']],list(range(1,7)))
            for selected in r['picked']:
                step=selected['step']
                for area in ['inside','outside']:
                    box=selected[area]['box'];x0,y0,x1,y1=box
                    support=masks[step-1,y0:y1,x0:x1]
                    self.assertTrue((support>0).all() if area=='inside' else (support==0).all())
                    for when,slot in [('before',step-1),('after',step)]:
                        expected=np.asarray(Image.open(src/f'recovery_{slot}.png').convert('RGB').crop(box))
                        actual=np.asarray(Image.open(folder/f'{step}_{area}_{when}.png'))
                        np.testing.assert_array_equal(actual,expected)
                    if area=='outside':
                        np.testing.assert_array_equal(states[step-1,y0:y1,x0:x1],states[step,y0:y1,x0:x1])
            if r['image_size'][1]>r['image_size'][0]:
                self.assertEqual(r['pool'],'ppr10k')
                self.assertIn(6,[s['step'] for s in r['picked']])

    def test_compiled_pages(self):
        expected=[2,2,2,3,3,3]
        for i,count in enumerate(expected,1):
            pdf=DRAFTS/f'case_{i:02d}.pdf'
            info=subprocess.check_output(['pdfinfo',str(pdf)],text=True)
            pages=int(next(line.split(':')[1] for line in info.splitlines() if line.startswith('Pages:')))
            self.assertEqual(pages,count)
            text=subprocess.check_output(['pdftotext',str(pdf),'-'],text=True)
            for heading in ['Before / After / GT','Iterative recovery','Local changes','Stagewise reasoning']:
                self.assertIn(heading,text)
            self.assertIn('target-conditioned',text)
            self.assertNotRegex(text,r'(?i)NVIDIA|\bGPU\b|\bCUDA\b')


if __name__=='__main__':unittest.main()
