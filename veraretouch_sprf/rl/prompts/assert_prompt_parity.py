#!/usr/bin/env python3
"""EPR-052 ENG-2 守卫：ms-swift template.encode 的学生 prompt 与项目 q3vl_text.prompt_text 的 processor 编码逐 id 相等；
教师 prompt 的 input_ids 前缀与学生相等直到指令末尾（SURVEY v2 §F.1 两条预注册守卫）。容器内运行：
  PYTHONPATH=/workspace/VeraRetouch python -m veraretouch_sprf.rl.prompts.assert_prompt_parity --model /data/runs/epr052_rl/s1f_epoch1_merged --jsonl ... --out ...
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from PIL import Image
import torch

from veraretouch_sprf.data import q3vl_text as T
from veraretouch_sprf.data import cot_text as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--jsonl', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--path-map', default='', help='e.g. /data=/home/bc/data（宿主上跑时）')
    a = ap.parse_args()
    from swift import get_processor, get_template   # main@236a1f19：get_model_processor/get_processor（无 get_model_tokenizer）
    processor = get_processor(a.model, model_type='qwen3_vl')
    template = get_template(processor, max_length=None)
    template.set_mode('transformers')   # prompt-only：与 rollout 前 resample_encode（remove_response 后 encode）同口径
    tok = processor.tokenizer
    im_end = tok.convert_tokens_to_ids('<|im_end|>')
    stage_ids = [tok.convert_tokens_to_ids(t) for t in C.STAGE_TOKENS]
    rows = [json.loads(l) for l in open(a.jsonl)]
    res = []
    for r in rows:
        img_path = r['images'][0]
        if a.path_map:
            s, d = a.path_map.split('=', 1); img_path = img_path.replace(s, d, 1)
        img = Image.open(img_path).convert('RGB')
        instr = r['messages'][0]['content']
        assert instr.startswith('<image>'); instr = instr[len('<image>'):]
        # 项目口径：prompt_text + processor(do_resize=False)（图已是 spec-5 几何）
        ref = processor(text=[T.prompt_text(instr)], images=[img], do_resize=False, return_tensors='pt')
        ref_ids = ref['input_ids'][0].tolist(); ref_grid = ref['image_grid_thw'][0].tolist()
        # ms-swift 口径
        enc = template.encode({'messages': r['messages'], 'images': [img_path]})
        sw_ids = list(enc['input_ids']); sw_grid = torch.as_tensor(enc['image_grid_thw']).reshape(-1, 3)[0].tolist()
        enc_t = template.encode({'messages': [{'role': 'user', 'content': r['teacher_prompt']}], 'images': [img_path]})
        t_ids = list(enc_t['input_ids'])
        first_diff = next((i for i, (x, y) in enumerate(zip(ref_ids, sw_ids)) if x != y), None)
        if first_diff is None and len(ref_ids) != len(sw_ids):
            first_diff = min(len(ref_ids), len(sw_ids))
        k = max(i for i, t in enumerate(sw_ids) if t == im_end)      # 学生指令末尾 <|im_end|> 位置
        t_pref = next((i for i in range(k) if t_ids[i] != sw_ids[i]), None)
        n_stage_in_teacher = sum(1 for t in t_ids if t in stage_ids)
        res.append(dict(key=r['key'], student_len_ref=len(ref_ids), student_len_swift=len(sw_ids),
                        student_ids_equal=(ref_ids == sw_ids), first_diff_idx=first_diff,
                        grid_ref=ref_grid, grid_swift=sw_grid, grid_equal=(ref_grid == sw_grid),
                        n_image_tokens_ref=sum(1 for t in ref_ids if t == tok.convert_tokens_to_ids('<|image_pad|>')),
                        n_image_tokens_swift=sum(1 for t in sw_ids if t == tok.convert_tokens_to_ids('<|image_pad|>')),
                        teacher_len=len(t_ids), teacher_prefix_equal_upto_instr_end=(t_pref is None), teacher_prefix_first_diff=t_pref,
                        prefix_len_checked=k, teacher_stage_tokens_single=(n_stage_in_teacher == 6),
                        student_text_swift=tok.decode(sw_ids[-8:]), student_text_ref=tok.decode(ref_ids[-8:])))
        if not res[-1]['student_ids_equal']:
            i = first_diff
            res[-1]['diff_context'] = dict(ref=tok.decode(ref_ids[max(0, i - 5): i + 5]), swift=tok.decode(sw_ids[max(0, i - 5): i + 5]))
    summ = dict(n=len(res), student_ids_equal=sum(r['student_ids_equal'] for r in res), grid_equal=sum(r['grid_equal'] for r in res),
                teacher_prefix_equal=sum(r['teacher_prefix_equal_upto_instr_end'] for r in res),
                teacher_stage_tokens_single=sum(r['teacher_stage_tokens_single'] for r in res),
                student_len=[r['student_len_swift'] for r in res], teacher_len=[r['teacher_len'] for r in res], rows=res)
    Path(a.out).write_text(json.dumps(summ, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summ.items() if k != 'rows'}, ensure_ascii=False))
    for r in res:
        print(r['key'], 'student_equal', r['student_ids_equal'], 'first_diff', r['first_diff_idx'], 'grid', r['grid_ref'], r['grid_swift'],
              'teacher_prefix_equal', r['teacher_prefix_equal_upto_instr_end'], 'lens', r['student_len_swift'], r['teacher_len'])


if __name__ == '__main__':
    main()
