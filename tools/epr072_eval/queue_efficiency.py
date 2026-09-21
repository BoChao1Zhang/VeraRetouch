"""Wait for exclusive card 0 and run only serial latency jobs.

Never stops other tasks. If sharing starts during timing, terminate only this
queue's exact child and invalidate the incomplete measurement.
"""
import json
import fcntl
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('/home/bc/nfsvfs/bc/data/runs/epr072_efficiency_20260921')
REPO=Path('/home/bc/VeraRetouch')
PY='/home/bc/envs/q3vl_sft/bin/python'


def write(name,value):
    ROOT.mkdir(parents=True,exist_ok=True)
    p=ROOT/name;t=p.with_suffix('.partial');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(p)


def processes():
    uuid=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    return {int(s.split(',')[1]) for s in text.splitlines() if s.split(',')[0].strip()==uuid}


def wait_idle(label):
    stable=0
    while stable<4:
        pids=processes();stable=stable+1 if not pids else 0
        write('queue.json',dict(state='waiting_for_exclusive_card0',job=label,pids=sorted(pids),idle_checks=stable,time=time.time()))
        time.sleep(15)


def launch(cmd,label,env):
    wait_idle(label)
    with (ROOT/(label+'.log')).open('a') as log:
        child=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
        write('queue.json',dict(state='running',job=label,pid=child.pid,time=time.time()))
        while child.poll() is None:
            others=processes()-{child.pid}
            if others:
                child.terminate()
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                write('queue.json',dict(state='timing_interrupted',job=label,other_pids=sorted(others),time=time.time()))
                return 'retry'
            time.sleep(10)
    if child.returncode == 75:
        return 'retry'
    if child.returncode:
        write('queue.json',dict(state='failed',job=label,returncode=child.returncode,time=time.time()))
        write(label+'_failure.json',dict(complete=False,returncode=child.returncode,log=str(ROOT/(label+'.log')),time=time.time()))
        return 'failed'
    return 'complete'


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'queue.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=dict(os.environ,PYTHONPATH=str(REPO),CUDA_VISIBLE_DEVICES='0',TOKENIZERS_PARALLELISM='false',
             LD_LIBRARY_PATH='/home/bc/miniconda3/envs/llm_factory/lib')
    jobs=[('ours',512,PY),('veraretouch',512,'/home/bc/data/external/veraretouch/venv/bin/python'),
          ('instructpix2pix',512,'/home/bc/data/external/diffusers_edit/venv/bin/python'),
          ('flux_kontext_dev',512,'/home/bc/data/external/diffusers_edit/venv/bin/python'),
          ('qwen_image_edit_2511',512,'/home/bc/data/external/diffusers_edit/venv/bin/python')]
    jobs += [('ours',size,PY) for size in (1024,2048,4096)]
    statuses={}
    for method,size,python in jobs:
        label=f'{method}_{size}';out=ROOT/label
        if (out/'results.json').exists() and json.loads((out/'results.json').read_text()).get('complete'):
            statuses[label]='complete';continue
        command=[python,'-m','tools.epr072_eval.efficiency','--method',method,'--short-side',str(size),'--out',str(out)]
        method_env=dict(env)
        if method!='ours':method_env.pop('LD_LIBRARY_PATH',None)
        while True:
            status=launch(command,label,method_env)
            if status!='retry':break
        statuses[label]=status
        write('jobs.json',statuses)
    write('queue.json',dict(state='finished',all_complete=all(s=='complete' for s in statuses.values()),jobs=statuses,time=time.time()))


if __name__=='__main__':main()
