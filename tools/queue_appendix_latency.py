"""Wait for existing training and an idle device, then run serial benchmarks.

Never signals external jobs. If another compute job arrives, terminate only
the benchmark child and wait again. State/result files are the durable record.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from PIL import Image

REPO = Path('/home/bc/VeraRetouch')
WORK = Path('/home/bc/data/runs/paper_appendix_expansion_20260919')
OUT = WORK/'latency'
TRAIN_PIDS = (3127669, 3127726)
DEVICE = '1'
PY = '/home/bc/envs/q3vl_sft/bin/python'


def status(state, **extra):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'queue_status.json').write_text(json.dumps(dict(state=state, updated=time.time(), **extra), indent=2)+'\n')


def active_training():
    return [pid for pid in TRAIN_PIDS if Path(f'/proc/{pid}/cmdline').exists()
            and b'epr072_local_train' in Path(f'/proc/{pid}/cmdline').read_bytes()]


def device_processes():
    uuid = subprocess.check_output(['nvidia-smi', '-i', DEVICE, '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    listing = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    return {int(line.split(',')[1]) for line in listing.splitlines() if line.split(',')[0].strip() == uuid}


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    old = Path('/home/bc/data/runs/paper_appendix_qualitative_20260919')
    records = json.loads((old/'style_inference_results.json').read_text())
    records += json.loads((WORK/'inference_results.json').read_text())
    selected = []
    inputs = OUT/'inputs'; inputs.mkdir(exist_ok=True)
    for pool in ['unsplash', 'fivek_gold', 'ppr10k']:
        candidates = [r for r in records if r['pool'] == pool]
        candidates.sort(key=lambda r: hashlib.sha256(('latency-frozen-v1:'+r['key']).encode()).hexdigest())
        if len(candidates) < 5:
            raise RuntimeError(f'Missing inference samples for {pool}')
        for i, row in enumerate(candidates[:5]):
            ident = f'{pool}_{i:02d}'
            frame = Image.open(Path(row['folder'])/'input.png').convert('RGB')
            scale = 512/max(frame.size)
            size = tuple(max(64, round(v*scale/64)*64) for v in frame.size)
            frame = frame.resize(size, Image.Resampling.LANCZOS)
            path = inputs/(ident+'.png'); frame.save(path)
            selected.append(dict(id=ident, pool=pool, key=row['key'], image=str(path),
                                 size=list(frame.size), instruction=row['instruction'],
                                 input_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (OUT/'manifest.json').write_text(json.dumps(dict(records=selected), indent=2)+'\n')
    high = WORK/'highres'; high.mkdir(exist_ok=True)
    destination = high/'unsplash_blanca_native.jpg'
    if not destination.exists():
        shutil.copyfile('/tmp/chiaro-unsplash-blanca.jpg', destination)
    size = Image.open(destination).size
    if size != (4032, 2268):
        raise ValueError('Unexpected native-resolution input')
    source = dict(url='https://unsplash.com/photos/4qKVQYOluDk', photographer='Yoshi Takekawa',
                  license='Unsplash License', native_size=list(size),
                  download='https://images.unsplash.com/photo-1476041178066-aa562074def7?fm=jpg&q=95',
                  sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
    (high/'source.json').write_text(json.dumps(source, indent=2)+'\n')
    (high/'manifest.json').write_text(json.dumps(dict(records=[dict(id='blanca_native', image=str(destination),
       size=list(size), instruction='Give the lake a warmer, brighter late-afternoon mood, lift the dark trees gently, and preserve the cloud and mountain detail.')]), indent=2)+'\n')


def launch(command, label, env):
    log = OUT/(label+'.log')
    with log.open('a') as handle:
        child = subprocess.Popen(command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT)
        status('running', job=label, child_pid=child.pid)
        while child.poll() is None:
            others = device_processes()-{child.pid}
            if others:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait()
                status('resource_conflict', job=label, other_pids=sorted(others))
                return False
            time.sleep(10)
    if child.returncode:
        status('failed', job=label, exit_code=child.returncode, log=str(log))
        raise RuntimeError(f'{label} failed; inspect {log}')
    return True


def wait_idle():
    stable = 0
    while stable < 8:
        training, processes = active_training(), device_processes()
        stable = stable+1 if not training and not processes else 0
        status('waiting_for_training_and_exclusive_device', training_pids=training,
               other_pids=sorted(processes), idle_checks=stable)
        time.sleep(15)


def summarize():
    import numpy as np
    lines=['# 推理耗时实测（中文待确认）', '',
           '同一批 15 张训练域照片与指令，公共输入长边 512；模型预载入、预热 2 次、每图 3 次。', '',
           '| 模型 | 完整流程中位数 / 秒 | P10 / 秒 | P90 / 秒 | 渲染子阶段中位数 / 秒 |',
           '|---|---:|---:|---:|---:|']
    summaries=[]
    for method in ['ours','veraretouch','instructpix2pix']:
        data=json.loads((OUT/method/'results.json').read_text())
        if len(data['records']) != 15:
            raise ValueError(f'Incomplete measurement: {method}')
        values=[r['median_seconds'] for r in data['records']]
        renders=[x['render_seconds'] for r in data['records'] for x in r['repetitions'] if x.get('render_seconds') is not None]
        summary=dict(method=method,n=15,median=float(np.median(values)),p10=float(np.percentile(values,10)),p90=float(np.percentile(values,90)),render_median=float(np.median(renders)) if renders else None)
        summaries.append(summary)
        render=f'{summary["render_median"]:.3f}' if renders else '未单独拆分'
        lines.append(f'| {method} | {summary["median"]:.3f} | {summary["p10"]:.3f} | {summary["p90"]:.3f} | {render} |')
    lines += ['', '完整流程包含输入解码、预处理、重新生成描述（适用时）、参数预测、图像执行、同步及结果回传；不含权重加载、文件保存。',
              '渲染子阶段不与其他模型的完整推理时间直接比较。原始设置、输出尺寸、每次耗时与运行审计见对应 results.json。']
    (OUT/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    (OUT/'results.zh.md').write_text('\n'.join(lines)+'\n')


def main():
    env = dict(os.environ, PYTHONPATH=str(REPO), CUDA_VISIBLE_DEVICES=DEVICE,
               LD_LIBRARY_PATH='/home/bc/miniconda3/envs/llm_factory/lib')
    wait_idle()
    results = WORK/'inference_results.json'
    if not results.exists() or len(json.loads(results.read_text())) != 40:
        command = [PY, str(REPO/'tools/paper_appendix_expansion.py'), 'infer']
        while not launch(command, 'complete_in_domain_outputs', env):
            wait_idle()
    prepare()
    jobs = [('ours', PY),
            ('veraretouch', '/home/bc/data/external/veraretouch/venv/bin/python'),
            ('instructpix2pix', '/home/bc/data/external/diffusers_edit/venv/bin/python')]
    for method, python in jobs:
        wait_idle()
        method_env = dict(env)
        if method != 'ours':
            method_env.pop('LD_LIBRARY_PATH', None)
        command = [python, str(REPO/'tools/appendix_latency_benchmark.py'), '--method', method,
                   '--manifest', str(OUT/'manifest.json'), '--out', str(OUT/method)]
        while not launch(command, method, method_env):
            wait_idle()
    wait_idle()
    command = [PY, str(REPO/'tools/appendix_latency_benchmark.py'), '--method', 'ours',
               '--manifest', str(WORK/'highres/manifest.json'), '--out', str(WORK/'highres/output'),
               '--warmup', '1', '--repeats', '3']
    while not launch(command, 'native_high_resolution', env):
        wait_idle()
    summarize()
    status('complete', results=[str(OUT/m/'results.json') for m, _ in jobs],
           high_resolution=str(WORK/'highres/output/results.json'))


if __name__ == '__main__':
    main()
