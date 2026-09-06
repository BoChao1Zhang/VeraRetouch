# 源自 experiments/prs/EPR-051_masked-restore-production/stage0/sprf/vlmsft/q3vl_data.py @ 2026-09-06（原件未入 git；原 sha256 见 veraretouch_sprf/PROVENANCE.md），逐字复制/仅改导入与路径解析（EPR-052）。
"""EPR-051 / SFT+ADAPT -- dataset shared by stage 1 (CE) and stage 2 (latent).

One sample = (y image, frozen instruction) -> 6-segment CoT.
The student never sees x0, the breakdown sheet, or any journal field.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
from veraretouch_sprf import _paths as _P  # EPR-052：替代原 sys.path 注入块
_P.ensure_sys_path()

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402

from veraretouch_sprf.data import cot_text as C  # noqa: E402
from veraretouch_sprf.models.vlm import q3vl_common as Q  # noqa: E402
from veraretouch_sprf.data import q3vl_text as T  # noqa: E402


class CoTDataset(Dataset):
    def __init__(self, snapshot_dir: str, keys, processor, stage_ids,
                 include_stage_token: bool = False, max_len: int = 4096,
                 targets=None, instruction_mode: str = "fixed",
                 instruction_salt: str = C.INSTRUCTION_SALT):
        snap = Path(snapshot_dir)
        self.by_key = {}
        for line in open(snap / "records.jsonl"):
            r = json.loads(line)
            self.by_key[r["key"]] = r
        self.assets = json.loads((snap / "assets_index.json").read_text())["index"]
        self.keys = list(keys)
        self.proc = processor
        self.stage_ids = stage_ids
        self.include_stage_token = include_stage_token
        self.max_len = max_len
        self.targets = targets            # (N, 6, 128) aligned to self.keys, or None
        if instruction_mode not in ("fixed", "per_sample"):
            Q.die(f"unknown instruction_mode {instruction_mode!r}")
        self.instruction_mode = instruction_mode
        self.instruction_salt = instruction_salt

    def instruction_of(self, key, rec):
        """fixed = the frozen one-sentence prompt; per_sample = this sample's own
        editing instruction, tier chosen by sha1(salt+':'+key) at 70/20/10."""
        if self.instruction_mode == "fixed":
            return C.INSTRUCTION, "fixed"
        return C.instruction_for(rec, key, self.instruction_salt)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        key = self.keys[i]
        rec = self.by_key[key]
        img, geom = Q.prepare_image_spec5(Image.open(self.assets[key]["png"]))
        instr, tier = self.instruction_of(key, rec)
        ex = T.build_example(self.proc, img, instr, rec,
                             include_stage_token=self.include_stage_token)
        Q.assert_grid(geom, ex["image_grid_thw"][0])
        if len(ex["input_ids"]) > self.max_len:
            Q.die(f"{key}: length {len(ex['input_ids'])} > max_len {self.max_len}; "
                  "truncation would cut a span, raise max_len instead")
        ids = ex["input_ids"].tolist()
        for m, (s, e) in enumerate(ex["spans"], 1):
            want = self.stage_ids[m - 1]
            got = ids[e] if not self.include_stage_token else ids[e - 1]
            if got != want:
                Q.die(f"{key}: span {m} boundary token {got} != stage token {want}")
        out = dict(key=key, **ex, n_visual_tokens=geom.n_visual_tokens,
                   instruction_tier=tier)
        if self.targets is not None:
            out["target"] = self.targets[i]
        return out


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    ii, ll, am = [], [], []
    for b in batch:
        k = n - len(b["input_ids"])
        ii.append(torch.cat([b["input_ids"],
                             torch.full((k,), pad_id, dtype=torch.long)]))
        ll.append(torch.cat([b["labels"],
                             torch.full((k,), Q.IGNORE_INDEX, dtype=torch.long)]))
        am.append(torch.cat([torch.ones(len(b["input_ids"]), dtype=torch.long),
                             torch.zeros(k, dtype=torch.long)]))
    out = dict(keys=[b["key"] for b in batch],
               input_ids=torch.stack(ii), labels=torch.stack(ll),
               attention_mask=torch.stack(am),
               spans=[b["spans"] for b in batch],
               pixel_values=torch.cat([b["pixel_values"] for b in batch], dim=0),
               image_grid_thw=torch.cat([b["image_grid_thw"] for b in batch], dim=0),
               n_visual_tokens=[b["n_visual_tokens"] for b in batch])
    if "target" in batch[0]:
        out["target"] = torch.stack([b["target"] for b in batch])
    return out
