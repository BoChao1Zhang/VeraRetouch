"""Validate an author-chosen four-family set; output a manifest, never publish it."""
import argparse
from collections import Counter
import json
from pathlib import Path

FAMILIES={'semantic','radial','band','linear'}
WEB=Path('/home/bc/VeraRetouch/outputs/local_review500_20260920')
SEMANTIC=Path('/home/bc/data/runs/paper_appendix_portraits_20260920/local_02/provenance.json')
GROUPS=Path('/home/bc/data/runs/paper_geometry_portraits_20260920/all_candidates.json')


def make_selection(ids,records):
    if len(ids) not in {3,4} or len(set(ids))!=len(ids):raise ValueError('Choose three new geometric cases, or four distinct cases.')
    chosen=[next(r for r in records if r['number']==n) for n in ids]
    if len(chosen)==3:
        semantic=json.loads(SEMANTIC.read_text())
        groups=json.loads(GROUPS.read_text())
        assert semantic['key'] in {r['key'] for r in groups['semantic']}
        chosen.insert(0,dict(semantic,geometry_kind='semantic'))
    counts=Counter(r['geometry_kind'] for r in chosen)
    if set(counts)!=FAMILIES or any(n!=1 for n in counts.values()):
        raise ValueError(f'One case per family is required, got {dict(counts)}')
    return [dict(title={'semantic':'Semantic support','radial':'Radial support','band':'Band support','linear':'Linear-gradient support'}[r['geometry_kind']],
                 provenance=str(Path(r['folder'])/'provenance.json'),geometry_kind=r['geometry_kind'],
                 require_spatial=True,subject_priority=r['geometry_kind']=='semantic') for r in chosen]


def main():
    p=argparse.ArgumentParser();p.add_argument('--ids',required=True);p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    ids=[int(n.strip()) for n in args.ids.split(',')]
    result=make_selection(ids,json.loads((WEB/'combined_results.json').read_text()))
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
