"""Build an offline, self-contained index around existing per-sample assets."""
import json
from pathlib import Path
from tools.build_local_review500 import ROOT,WEB


def main():
    records=json.loads((ROOT/'results.json').read_text())
    public=[]
    for r in records:
        public.append({k:r[k] for k in ['number','source_id','key','instruction','annotation','pool','image_size',
                                       'portrait','local_amplitude','background','crop','residual_max']})
        for image in ['input','before','after','gt','support','residual','crop_before','crop_after','z1','z2','z3','z4']:
            assert (WEB/f'case_{r["number"]:03d}'/f'{image}.png').is_file()
    html=(Path(__file__).with_name('local_review_gallery.html')).read_text()
    encoded=json.dumps(public,ensure_ascii=False).replace('<','\\u003c')
    (WEB/'index.html').write_text(html.replace('__RECORDS__',encoded))
    (WEB/'index.json').write_text(json.dumps(public,ensure_ascii=False,indent=2)+'\n')
    (WEB/'README.txt').write_text('打开 index.html，勾选照片后点击“导出选中编号”。这些是目标条件闭式恢复候选，不是模型预测。\n')
    print(json.dumps(dict(samples=len(records),portrait=sum(r['portrait'] for r in records),index=str(WEB/'index.html'))))


if __name__=='__main__':main()
