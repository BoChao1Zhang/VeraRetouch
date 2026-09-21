"""Count floating checkpoint tensors from headers, without loading model weights."""
import hashlib
import json
import math
from pathlib import Path
import struct


def count(root):
    root = Path(root)
    files = sorted(root.glob('*.safetensors'))
    if not files:
        raise ValueError(f'No safetensors at {root}')
    total = 0
    groups = {}
    hashes = {}
    for path in files:
        with path.open('rb') as stream:
            size = struct.unpack('<Q', stream.read(8))[0]
            raw = stream.read(size)
        hashes[path.name] = hashlib.sha256(raw).hexdigest()
        for name, tensor in json.loads(raw).items():
            if name == '__metadata__' or tensor['dtype'] not in ('F32','F16','BF16','F64'):
                continue
            n = math.prod(tensor['shape'])
            total += n
            prefix = '.'.join(name.split('.')[:2])
            groups[prefix] = groups.get(prefix, 0) + n
    return dict(path=str(root), floating_elements=total, groups=groups, headers=hashes)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--public-output', type=Path)
    args = parser.parse_args()
    base = Path('/home/bc/data/models')
    roots = {name: base/name for name in ('monetGPT','JarvisArt','JarvisEvo','VeraRetouch','EyeControl')}
    for model, components in {
        'instruct-pix2pix': ['unet','vae','text_encoder'],
        'FLUX.1-Kontext-dev': ['transformer','vae','text_encoder','text_encoder_2'],
        'Qwen-Image-Edit-2511': ['transformer','vae','text_encoder'],
    }.items():
        for component in components:
            roots[f'{model}/{component}'] = base/model/component
    roots['ours_readout_base'] = Path('/home/bc/data/runs/epr051_vlmsft/sft_s1f_full/ckpt_epoch1/model')
    roots['ours_subject_base'] = Path('/home/bc/data/runs/q3vl_base_sft_v2seg_20260814/checkpoint-4976')
    results = {name: count(root) for name, root in roots.items()}
    dest = Path('/tmp/paper_parameter_audit.json')
    dest.write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps({k: v['floating_elements'] for k,v in results.items()},indent=2))
    n = {k: v['floating_elements'] for k,v in results.items()}
    pipeline = lambda prefix: sum(v for k,v in n.items() if k.startswith(prefix+'/'))
    config = json.loads(Path('/home/bc/data/runs/epr072_local_continuation_20260919/ENDPT2/full/config.json').read_text())['model']
    one = n['ours_readout_base'] + sum(config[k] for k in ('n_lora_params','n_readout_emb','n_head_params'))
    six = one + n['ours_subject_base'] + 4099989
    totals = {'EyeControl': pipeline('FLUX.1-Kontext-dev')+n['EyeControl'],
              'FLUX.1 Kontext': pipeline('FLUX.1-Kontext-dev'),
              'Qwen-Image-Edit': pipeline('Qwen-Image-Edit-2511'),
              'InstructPix2Pix': pipeline('instruct-pix2pix'),
              'MonetGPT': n['monetGPT'], 'JarvisArt': n['JarvisArt'],
              'JarvisEvo': n['JarvisEvo'], 'VeraRetouch': n['VeraRetouch'],
              'Ours (One-stage)': one, 'Ours (Six-stage)': six}
    if args.public_output:
        args.public_output.write_text(json.dumps(dict(
            table_nominal_backbone_billions={'EyeControl':12,'FLUX.1 Kontext':12,
                'Qwen-Image-Edit':20,'InstructPix2Pix':0.86,'MonetGPT':7,
                'JarvisArt':7,'JarvisEvo':8,'VeraRetouch':0.5,
                'Ours (One-stage)':4,'Ours (Six-stage)':4},
            table_scope='Nominal principal language/denoising backbone scale, not total deployment weights. All expert-table Ours rows use the six-stage configuration.',
            definition='Approximate checkpoint parameter inventory, in billions; frozen components and retained heads/adapters included, tied weights stored once; small floating buffers included in header counts.',
            totals=totals, components=n,
            readout_additions={k:config[k] for k in ('n_lora_params','n_readout_emb','n_head_params')},
            subject_head=4099989,
            expert_tables='Ours uses six-stage execution',
            exclusions='No judge models or proprietary Lightroom application internals; instruction-mode VeraRetouch does not use the separate reference encoder.',
            header_sha256={k:v['headers'] for k,v in results.items()}),indent=2)+'\n')
