"""Deterministic tar-block sampling and bounded whole-file residency for EPR072."""
from collections import OrderedDict, defaultdict
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import tarfile
from types import MethodType

import numpy as np
from PIL import Image
import torch

from veraretouch_sprf.readout import mixed_data as MX
from veraretouch_sprf.data.archive_assets import ArchiveAssets


def select_style_shards(rows,journal_root,count,seed):
    """Select whole journal tar groups in seeded order; trim only the final group.

    The original style30k images are loose PNGs. Journal tar grouping is retained
    for deterministic locality; we do not substitute the different D1 style set.
    """
    by_key={r['key']:r for r in rows}; groups=defaultdict(list); shard_of={}
    for path in sorted(Path(journal_root).glob('epr071_cot_style_*.index.sqlite')):
        with sqlite3.connect(f'file:{path}?mode=ro',uri=True) as db:
            keys=[r[0] for r in db.execute('SELECT sample FROM members ORDER BY offset')]
        for key in keys:
            if key in by_key:
                groups[path.name].append(key);shard_of[key]=path.name
    if sum(map(len,groups.values()))<count:raise ValueError('Not enough style records in tar groups')
    rng=np.random.default_rng(seed);names=sorted(groups)
    chosen=[]; selected=[]
    for i in rng.permutation(len(names)):
        name=names[int(i)];keys=groups[name]
        take=keys[:max(0,count-len(chosen))]
        chosen.extend(by_key[k] for k in take); selected.append(dict(tar=name,n=len(take)))
        if len(chosen)==count:break
    return chosen,shard_of,selected


class ResidentFiles:
    def __init__(self,budget_gib=12,max_files=4096):
        self.budget=int(budget_gib*2**30);self.max_files=max_files
        self.cache=OrderedDict();self.bytes=0;self.loads=0;self.hits=0

    def get(self,path):
        path=str(path)
        if path in self.cache:
            self.hits+=1;self.cache.move_to_end(path);return self.cache[path]
        size=Path(path).stat().st_size
        if size>self.budget:raise RuntimeError(f'File exceeds residency budget: {path} {size}')
        while self.cache and (self.bytes+size>self.budget or len(self.cache)>=self.max_files):
            _,old=self.cache.popitem(last=False);self.bytes-=len(old)
        data=Path(path).read_bytes()
        if len(data)!=size:raise OSError(f'Short whole-file read: {path}')
        self.cache[path]=data;self.bytes+=len(data);self.loads+=1;return data

    def facts(self):return dict(bytes=self.bytes,files=len(self.cache),loads=self.loads,hits=self.hits)


class CachedStyle(MX.SingleItems):
    def __init__(self,*args,shard_of,resident,**kwargs):
        super().__init__(*args,**kwargs);self.shard_of=shard_of
        adapter=LooseImageAdapter(resident,self.root)
        for sample in self.samples:sample['cache']=adapter

    def shard_key(self,i):return ('style',self.shard_of[self.samples[i]['key']])


class LooseImageAdapter:
    def __init__(self,resident,root):self.resident=resident;self.root=root;self.verified=set()
    def image(self,sample):
        path=Path(sample.get('root') or self.root)/sample['image_file']
        blob=self.resident.get(path)
        if str(path) not in self.verified:
            if hashlib.sha256(blob).hexdigest()!=sample['image_sha256']:raise ValueError('Image hash mismatch')
            self.verified.add(str(path))
        return Image.open(io.BytesIO(blob)).convert('RGB')


def install_chain_residency(source,resident):
    ds=source.setup();readers={}
    def asset(self,entry,name):
        loose=Path(entry['dir'])/'assets'/name
        if loose.exists():blob=resident.get(loose)
        else:
            if entry['dir'] not in readers:
                from veraretouch_sprf.data import train_stage0
                readers[entry['dir']]=ArchiveAssets(Path(entry['dir']),train_stage0)
            reader=readers[entry['dir']]
            shard,offset,size,member=reader.index[name]
            raw=resident.get(reader.tars[shard])
            header=tarfile.TarInfo.frombuf(raw[offset-512:offset],'utf-8','strict')
            if header.name!=member or header.size!=size:raise ValueError('Tar index/header mismatch')
            blob=raw[offset:offset+size]
        with Image.open(io.BytesIO(blob)) as image:
            return torch.from_numpy(np.asarray(image.convert('RGB'),dtype=np.float32)/255.)
    ds.asset=MethodType(asset,ds)
