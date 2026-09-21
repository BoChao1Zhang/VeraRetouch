"""Run EPR-081 only after the already-authorized efficiency queue finishes.

Own queue lock; no process-name killing, no interference with existing jobs.
"""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('/home/bc/nfsvfs/bc/data/runs/epr081_endpoint_global_20260921')
EFF=Path('/home/bc/nfsvfs/bc/data/runs/epr072_efficiency_20260921/queue.json')
PY='/home/bc/envs/q3vl_sft/bin/python'
REPO=Path('/home/bc/VeraRetouch')


def write(state,**extra):
    p=ROOT/'queue.json';tmp=p.with_suffix('.partial')
    tmp.write_text(json.dumps(dict(state=state,time=time.time(),**extra),indent=2)+'\n');tmp.replace(p)


def idle():
    uuid=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid',
                                  '--format=csv,noheader'],text=True).strip()
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                                  '--format=csv,noheader'],text=True)
    return not any(s.split(',')[0].strip()==uuid for s in rows.splitlines())


def wait_resources():
    stable=0
    while stable<4:
        try: finished=json.loads(EFF.read_text()).get('state')=='finished'
        except (OSError,ValueError): finished=False
        stable=stable+1 if finished and idle() else 0
        write('waiting_after_efficiency',efficiency_finished=finished,idle_checks=stable)
        time.sleep(15)


def launch(label,args):
    wait_resources()
    env=dict(os.environ,PYTHONPATH=str(REPO),CUDA_VISIBLE_DEVICES='0',
             LD_LIBRARY_PATH='/home/bc/miniconda3/envs/llm_factory/lib',TOKENIZERS_PARALLELISM='false')
    with (ROOT/(label+'.log')).open('a') as log:
        child=subprocess.Popen([PY,'-u']+args,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
        write('running',job=label,pid=child.pid)
        code=child.wait()
    if code:
        write('failed',job=label,returncode=code)
        raise SystemExit(code)


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    preflight=ROOT/'preflight_pairs'
    if not (preflight/'pairs.json').exists():
        write('blocked',reason='CPU pair-export preflight must pass before queueing')
        return
    if not (ROOT/'smoke/smoke_pass.json').exists():
        if not (ROOT/'preflight_gpu/export_audit.json').exists():
            launch('float_parity',['tools/epr081_export_pairs.py','--out',str(ROOT/'preflight_gpu'),
                '--device','cuda:0','--limit','32','--compare-to',str(preflight)])
        launch('smoke',['-m','veraretouch_sprf.readout.epr081_train',
            '--pairs',str(preflight),'--out',str(ROOT/'smoke'),'--smoke','2'])
    if not (ROOT/'pairs/export_audit.json').exists():
        launch('export',['tools/epr081_export_pairs.py','--out',str(ROOT/'pairs'),'--device','cuda:0'])
    if not json.loads((ROOT/'pairs/export_audit.json').read_text())['complete']:
        raise RuntimeError('Full endpoint export incomplete')
    launch('train',['-m','veraretouch_sprf.readout.epr081_train',
        '--pairs',str(ROOT/'pairs'),'--out',str(ROOT/'train')])
    write('finished',checkpoint=str(ROOT/'train/final.pt'))


if __name__=='__main__':main()
