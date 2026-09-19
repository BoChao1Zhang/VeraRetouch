"""Build a local, dependency-free HTML sample browser from verified assets."""
import json
from pathlib import Path
import re

ROOT=Path('/home/bc/VeraRetouch/EPR/ICLR2027')
ASSETS=ROOT/'figures/appendix_qualitative_20260919'


def main():
    samples=[]
    names=[('recovery_01','逆光中的人物','Backlit portrait'),
           ('recovery_02','雾中的林湖','Woodland lake'),
           ('recovery_03','泳池的鲜明色彩','Poolside color'),
           ('inference_01','狐狸的温暖毛色','Warm fur tones'),
           ('inference_02','瀑布的冷色对比','Cool waterfall'),
           ('inference_03','咖啡馆的暖色氛围','Warm interior'),
           ('inference_04','秋叶的暖色层次','Autumn foliage'),
           ('inference_05','洞穴与海岸光线','Coastal light'),
           ('inference_06','城市俯瞰的色调','Urban tones'),
           ('inference_07','日落中的人物剪影','Sunset silhouettes')]
    for ident,title,english in names:
        r=json.loads((ASSETS/ident/'provenance.json').read_text())
        recovery=ident.startswith('recovery')
        prefix='../figures/appendix_qualitative_20260919/'+ident+'/'
        s=dict(id=ident,kind='recovery' if recovery else 'inference',title=title,english=english,
          description=('六阶段恢复回放：查看颜色变化、空间支持与原始标注。' if recovery else
                       '指令引导的训练域样例：查看目标风格、实际模型输出与编辑描述。'),
          input=prefix+('recovery_0.png' if recovery else 'input.png'),
          target=prefix+('reference.png' if recovery else 'target.png'),
          initial=r['initial_l1'] if recovery else r['identity_l1'],
          final=r['final_l1'] if recovery else r['l1'],
          facts={'来源':'Unsplash（来源表核对）','划分':'Training example','样本键':r['key']})
        if recovery:
            a=r['annotation'];s.update(instructions={k:a['instruction_'+k] for k in ['short','medium','long']},
               states=[prefix+f'recovery_{i}.png' for i in range(7)],
               masks=[prefix+f'mask_{i}.png' for i in range(1,7)],order=r['stage_order'],reasoning=a['cot'])
            s['facts'].update({'执行方式':'记录相邻状态求码；固定代码顺序回放','支持':'原始 soft beta，已包含强度',
                              '文本来源':'原始标注记录；非本次新生成'})
        else:
            raw=re.sub(r'<vr_stage_\d+>','',r['reasoning']).strip()
            m=re.search(r'Observation:\s*(.*?)\s*Mask:\s*(.*?)\s*Adjustment:\s*(.*)',raw,re.S)
            if not m:raise ValueError('Missing structured reasoning: '+ident)
            s.update(instructions={'medium':r['instruction']},result=prefix+'prediction.png',
                     reasoning=[dict(observation=m[1],mask=m[2],adjustment=m[3])])
            s['facts'].update({'模型':'R best@800 · continuous readout','输出':'实际预测颜色码与渲染结果',
                              '文本来源':'缓存的 base-SFT 模型生成描述'})
        samples.append(s)
    data=dict(samples=samples,system=(ASSETS/'system_prompt.txt').read_text(),
              task=(ASSETS/'user_head_prompt.txt').read_text())
    template=(ROOT/'previews/qualitative_samples.template.html').read_text()
    result=template.replace('__SAMPLE_DATA__',json.dumps(data,ensure_ascii=False).replace('<','\\u003c'))
    if re.search(r'\b(?:nvidia|gpu|cuda|h100|a100)\b',result,re.I):
        raise ValueError('Hardware text in public sample page')
    output=ROOT/'previews/qualitative_samples.html';output.write_text(result)
    print(json.dumps(dict(path=str(output),samples=len(samples),network_dependencies=0)))


if __name__=='__main__':main()
