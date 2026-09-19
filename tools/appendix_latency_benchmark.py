"""Matched-input, warm-model latency experiment, plus native-resolution inference.

Called serially in each method's existing environment. Timed intervals include
image decoding, preprocessing, fresh text generation where applicable, readout,
rendering, synchronization and return to host; checkpoint loading and saving
are excluded. No cached rationale is used for the latency measurement.
"""
import argparse
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np
from PIL import Image

REPO = Path('/home/bc/VeraRetouch')
sys.path.insert(0, str(REPO))


def load_ours():
    import torch
    from tools.epr071_val50_diag import load_checkpoint, PROTOSET
    from veraretouch_sprf.readout import select_train as ST, epr071_data as E71, artedit_eval as AE
    from veraretouch_sprf.data import q3vl_text as T
    from q3vl.train.imageproc import prepare_image
    from tools.epr061_cache.common import STYLE_SUFFIX
    from q3vl.whatb.lutdata import apply_lut_volume
    bank = ST.Bank('cuda:0', path=PROTOSET)
    model, facts = load_checkpoint(E71.STORE/'train/R/run/best.pt', bank, 'cuda:0')
    model.model.eval(); model.head.eval()
    mean, std, _ = E71.load_scaler()
    mean = torch.as_tensor(mean, device='cuda:0'); std = torch.as_tensor(std, device='cuda:0')
    renderer = AE.GlutRenderer(AE.GEOMETRY, 'cuda:0')

    @torch.inference_mode()
    def call(row):
        frame = Image.open(row['image']).convert('RGB')
        small, _ = prepare_image(frame)
        encoded = T.encode_prompt(model.processor, small, row['instruction']+STYLE_SUFFIX)
        tensors = {k: v.to('cuda:0') for k, v in encoded.items()}
        # Training's rationale generator is the frozen base SFT, not the
        # readout adapter. Fresh generation stops at the same stage marker.
        with model.model.disable_adapter():
            generated = model.model.generate(**tensors, do_sample=False,
                        max_new_tokens=768, eos_token_id=list(model.stage_ids),
                        pad_token_id=model.pad_id, use_cache=True)
        prompt = encoded['input_ids'][0].tolist()
        cot = generated[0, len(prompt):].tolist()
        if not cot or cot[-1] != model.stage_ids[0]:
            raise ValueError('Fresh rationale did not complete the required first stage')
        seq = model.sequences(prompt, cot)
        images = [dict(pixel_values=encoded['pixel_values'], image_grid_thw=encoded['image_grid_thw'])]
        _, q, _ = model([seq], images)
        raw = (q[0]*std+mean).reshape(3, -1)
        torch.cuda.synchronize()
        began = time.perf_counter()
        source = np.asarray(frame, dtype=np.float32)/255
        volume = renderer.volume(raw)
        flat = source.reshape(-1, 3); pieces = []
        # Bounded native-grid application; no output resizing or resynthesis.
        for start in range(0, len(flat), 1048576):
            rgb = torch.as_tensor(flat[start:start+1048576], device='cuda:0')
            pieces.append(apply_lut_volume(volume, rgb).clamp(0, 1).cpu().numpy())
        pred = np.concatenate(pieces).reshape(source.shape)
        output = np.rint(pred*255).astype(np.uint8)
        torch.cuda.synchronize()
        return output, dict(render_seconds=time.perf_counter()-began,
                            generated_tokens=len(cot), rationale=model.tokenizer.decode(cot, skip_special_tokens=False),
                            raw_code=raw.cpu().numpy().tolist())
    return call, dict(checkpoint=facts, generation='greedy base SFT, max 768, stage-token stop',
                      rendering='33-cubed baked LUT, native pixel grid, chunks of 1048576 pixels')


def load_vera():
    import torch
    root = Path('/home/bc/data/external/veraretouch/repo')
    sys.path.insert(0, str(root)); os.chdir(root)
    import inference as V
    path = '/home/bc/data/external/veraretouch/checkpoints/VeraRetouch'
    with open(root/'configs/infer_config.yaml') as stream:
        cfg = V.Box(V.yaml.safe_load(stream))
    cfg.project_name = 'test_no_instruct'; cfg.freeze_retouch_decoder = True
    tokenizer = V.AutoTokenizer.from_pretrained(path, model_max_length=4096, padding_side='right', use_fast=False)
    model = V.VeraRetouchForCausalLLM_Unified.from_pretrained(path, config_add=cfg, torch_dtype=torch.bfloat16).cuda().eval()
    tokens = [V.DEFAULT_RETOUCH_LIGHT_TOKEN, V.DEFAULT_RETOUCH_COLORTEMP_TOKEN, V.DEFAULT_RETOUCH_COLORMIXER_TOKEN]
    model.register_special_token_idx(*[tokenizer(t, add_special_tokens=False).input_ids[0] for t in tokens])
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    processor = model.get_vision_tower().image_processor
    collator = V.DataCollatorForUnifiedTestDataset(tokenizer=tokenizer)

    @torch.inference_mode()
    def call(row):
        ds = V.Infer_Style_Dataset(img_paths=[row['image']], prompts=[row['instruction']], tokenizer=tokenizer, image_processor=processor)
        batch = collator([ds[0]])
        outputs, text = model._generate(tokenizer=tokenizer, inputs=batch['input_ids'].cuda(),
                  attention_mask=batch['attention_mask'].cuda(),
                  images=[x.to(torch.bfloat16).cuda() for x in batch['images']],
                  image_sizes=batch['image_sizes'],
                  retouch_masks=[x.to(torch.bfloat16).cuda() for x in batch['retouch_masks']],
                  input_imgs=[x.to(torch.bfloat16).cuda() for x in batch['input_imgs']],
                  do_sample=False, temperature=0, top_p=1, num_beams=1,
                  max_new_tokens=4096, output_hidden_states=True, return_dict_in_generate=True, chunk=-1)
        # Official saver uses cv2.imwrite, so outputs are BGR.
        return np.asarray(outputs[0])[..., ::-1].copy(), dict(rationale=str(text[0]))
    return call, dict(checkpoint=path, generation='greedy, max 4096', mode='style')


def load_ip2p():
    import torch
    from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
    path = '/home/bc/data/models/instruct-pix2pix'
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(path, torch_dtype=torch.bfloat16,
               safety_checker=None, requires_safety_checker=False).to('cuda:0')
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    @torch.inference_mode()
    def call(row):
        frame = Image.open(row['image']).convert('RGB')
        output = pipe(prompt=row['instruction'], image=frame, num_inference_steps=100,
                      guidance_scale=7.5, image_guidance_scale=1.5,
                      generator=torch.Generator('cuda:0').manual_seed(19)).images[0]
        return np.asarray(output), dict(steps=100)
    return call, dict(checkpoint=path, steps=100, guidance_scale=7.5, image_guidance_scale=1.5, seed=19)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['ours', 'veraretouch', 'instructpix2pix'], required=True)
    p.add_argument('--manifest', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=3); p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--purpose', choices=['latency', 'qualitative'], default='latency')
    args = p.parse_args()
    import torch
    torch.set_num_threads(4); torch.manual_seed(19)
    if args.purpose == 'qualitative':
        torch.cuda.set_per_process_memory_fraction(.20)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = json.loads(args.manifest.read_text())['records']
    args.out.mkdir(parents=True, exist_ok=True)
    call, config = {'ours': load_ours, 'veraretouch': load_vera, 'instructpix2pix': load_ip2p}[args.method]()
    for _ in range(args.warmup):
        call(rows[0]); torch.cuda.synchronize()
    results = []
    for row in rows:
        samples = []
        for repeat in range(args.repeats):
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter(); output, extra = call(row); torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
            if (output.shape[1], output.shape[0]) != tuple(row['size']):
                raise ValueError(f'Output geometry mismatch: {row["id"]}')
            samples.append(dict(seconds=elapsed, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                render_seconds=extra.get('render_seconds')))
        Image.fromarray(output).save(args.out/(row['id']+'.png'))
        result = dict(id=row['id'], image=row['image'], size=row['size'], instruction=row['instruction'],
                      repetitions=samples, median_seconds=float(np.median([r['seconds'] for r in samples])), **extra)
        results.append(result)
        payload = dict(method=args.method, purpose=args.purpose, config=config, warmup=args.warmup,
                       repeats=args.repeats, records=results,
                       private_runtime_audit=dict(device=torch.cuda.get_device_name(), torch=torch.__version__,
                                                  inference_dtype='bfloat16', batch_size=1),
                       timing_scope='decode, preprocess, fresh generation, code prediction, rendering, synchronization, host output; excludes model loading and saving')
        (args.out/'results.json').write_text(json.dumps(payload, indent=2)+'\n')
        print(json.dumps(dict(method=args.method, id=row['id'], seconds=result['median_seconds'])), flush=True)


if __name__ == '__main__':
    main()
