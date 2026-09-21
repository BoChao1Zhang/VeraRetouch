"""Check expert-table values and ranking macros against full-precision data."""
import json
from pathlib import Path
import re

PAPER=Path('/home/bc/VeraRetouch/EPR/ICLR2027')


def main():
    data=json.loads((PAPER/'tables/expert_results_20260921.json').read_text())['tables']
    checks=0
    for bench,rows in data.items():
        lines=(PAPER/'tables/objective_pair.tex').read_text().splitlines()
        metrics=[('L1','L1$',2,False),('L2','L2$',2,False),('PSNR','PSNR$',2,True),
                 ('SSIM','SSIM$',3,True),('DE00',r'$\Delta E',2,False)]
        for row in rows:
            prefix=r'\textbf{Ours}' if row['method'].startswith('Ours') else row['method']
            line=next(s.strip() for s in lines if s.strip().startswith(prefix))
            cells=line.split('&')[1:]
            assert len(cells)==10
            cells=cells[:5] if bench=='fivek' else cells[5:]
            assert len(cells)==5
            for (key,_,digits,high),cell in zip(metrics,cells):
                ranks=sorted(set(r[key] for r in rows),reverse=high)
                value=re.search(r'\d+\.\d+',cell).group()
                assert value==f'{row[key]:.{digits}f}',(bench,key,row['method'],cell)
                assert ('\\bestmetric{' in cell)==(row[key]==ranks[0])
                assert ('\\secondmetric{' in cell)==(row[key]==ranks[1])
                checks+=1
        assert r'\textbf{Ours}' in '\n'.join(lines)
        ours=rows[-1]
        assert ours['execution']=='six-stage' and ours['stage_text']=='none'
        assert ours['n']==(498 if bench=='fivek' else 492)
    print(json.dumps(dict(status='PASS',metric_cells=checks,tables=1,benchmarks=2,
                          ranking='verified before rounding',ours='six-stage without CoT')))


if __name__=='__main__':main()
