# 源自 experiments/prs/EPR-051_masked-restore-production/stage0/sprf/vlmsft/parity_report.py @ 2026-09-06（原件未入 git；原 sha256 见 veraretouch_sprf/PROVENANCE.md），逐字复制/仅改导入与路径解析（EPR-052）。
"""810 divergence statistics: batch-1 vs batch-8 greedy, per the 2026-09-06 ruling.

Reporting only -- NOT a gate.  The replay contract is fixed batch composition +
batch_manifest, not bitwise equality with batch-1 (bf16 reduction order differs).
"""
import json, sys
from pathlib import Path
import torch

RH = Path("/home/bc/data/runs/epr051_vlmsft/rehearse8")
a_t = json.loads((RH/"par1"/"samples_par1.json").read_text())
b_t = json.loads((RH/"par8"/"samples_par8.json").read_text())
A = torch.load(RH/"par1"/"latents_par1.pt", map_location="cpu")
B = torch.load(RH/"par8"/"latents_par8.pt", map_location="cpu")
ai = {k: i for i, k in enumerate(A["keys"])}
bi = {k: i for i, k in enumerate(B["keys"])}

keys = [k for k in A["keys"] if k in bi]
rows, n_ident = [], 0
for k in keys:
    ta, tb = a_t.get(k), b_t.get(k)
    ident = ta == tb
    n_ident += ident
    dv = None
    if not ident and ta is not None and tb is not None:
        dv = next((j for j in range(min(len(ta), len(tb))) if ta[j] != tb[j]),
                  min(len(ta), len(tb)))
    la, lb = A["latents"][ai[k]], B["latents"][bi[k]]
    cos = torch.nn.functional.cosine_similarity(la, lb, dim=-1)
    rows.append(dict(key=k, identical=bool(ident), first_div_char=dv,
                     len_b1=len(ta) if ta else None, len_b8=len(tb) if tb else None,
                     latent_cos_min=float(cos.min()), latent_cos_mean=float(cos.mean()),
                     latent_l2=float((la - lb).norm(dim=-1).mean()),
                     ok_b1=bool(A["ok"][ai[k]]), ok_b8=bool(B["ok"][bi[k]])))

print(f"{'key':34s} {'ident':6s} {'div@char':9s} {'len b1/b8':12s} "
      f"{'lat_cos_mean':13s} {'lat_cos_min':12s} {'lat_l2':8s}")
for r in rows:
    print(f"{r['key']:34s} {str(r['identical']):6s} "
          f"{str(r['first_div_char']):9s} "
          f"{str(r['len_b1'])+'/'+str(r['len_b8']):12s} "
          f"{r['latent_cos_mean']:<13.6f} {r['latent_cos_min']:<12.6f} "
          f"{r['latent_l2']:<8.4f}")
divs = [r["first_div_char"] for r in rows if r["first_div_char"] is not None]
summ = dict(n=len(rows), n_text_identical=n_ident, n_differing=len(rows)-n_ident,
            first_div_char_min=min(divs) if divs else None,
            first_div_char_max=max(divs) if divs else None,
            latent_cos_mean_over_keys=sum(r["latent_cos_mean"] for r in rows)/len(rows),
            latent_cos_min_over_keys=min(r["latent_cos_min"] for r in rows),
            ok_agree=sum(1 for r in rows if r["ok_b1"] == r["ok_b8"]))
print("PARITY_STATS " + json.dumps(summ))
(RH/"parity_stats.json").write_text(json.dumps(dict(summary=summ, per_key=rows), indent=1))
