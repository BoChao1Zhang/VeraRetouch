# tools/scache — s 缓存服务（INF-5）

读出臂与渲染臂解耦的 s 场离线缓存层（EXPERIMENTS_v3 §INF-5、任务卡 T5）。
读出臂把 s 场写进自己的 arm 目录；渲染臂训练只读缓存、零 VLM 成本；
交叉组合 = 换一个 arm 目录名。

## 磁盘约定

```
<root>/<arm>/<img_id>__<instr_hash>.npy        # float16，默认 32x32（约 2KB/图），可配原生分辨率
<root>/<arm>/<img_id>__<instr_hash>.meta.json  # 必含: img_id/instr_hash/shape/dtype/layer/norm/created_at/arm/arm_version
```

- `instr_hash` = md5(instruction)[:12]；无指令条目 = `noinstr`（`api.instr_hash`）。
- oracle 目录的 meta 另含 `origin`：原掩膜与输入图的 `(shard, offset_data, length, sha256)`
  引用（无压缩 ustar，可 seek 直读）+ sample/group/source/mask id + 原掩膜尺寸。
- 归一化：oracle 为 `uint8/255` 线性到 [0,1]；**无任何逐图归一化**（红线）。

## API（`api.py`，仅依赖 numpy）

```python
from api import SCache, instr_hash

cache = SCache(root, arm="ro1-l17", resolution=32, arm_version="v1")  # resolution=None -> 原生分辨率
cache.write(img_id, instr_hash(instr), s, layer=17, norm={"kind": "zscore-global", ...})
entry = cache.read(img_id, ihash)          # SEntry(img_id, instr_hash, s: float16 (H,W), meta)
cache.exists(img_id, ihash); len(cache)
for e in cache.iter_entries(): ...          # 确定序（文件名排序）
for batch in cache.iter_batches(256): ...   # 批量
arr = cache.read_stack(keys)                # (N,H,W) float16
```

写入原子（tmp + rename）；写入校验形状/NaN；`extra_meta` 不得覆盖必含字段。

## oracle 构建（`oracle.py`）

从 l 系 build 的 shards 逐候选读 C_GT 掩膜（`.cgt.png`，逐候选区域掩膜、软边单通道），
PIL BOX **面积加权**下采样到 32×32，img_id=candidate_id，指令取 journal 归档 `sft.jsonl`：

```bash
python3 tools/scache/oracle.py \
  --build /mnt/nfs/bc/data/datasets/sft/prod-l1-local17k-20260731 \
  --journal /var/cache/veradata/annot_review/journal-archive/prod-l1-local17k-20260731 \
  --root <s_cache根> [--size 32] [--max-groups 100] [--sft-only]
```

## 引导上采样（`upsample.py`，torch + kornia 0.8.2）

```python
from upsample import upsample_s, iou_at
s_full = upsample_s(s32, guide_rgb, kernel_size=None, eps=1e-4, subsample=1, device="cuda")
```

bilinear 到 guide 分辨率 → `kornia.filters.guided_blur(guidance, input, kernel_size, eps,
border_type, subsample)`（签名已对 0.8.2 源码核实）。`kernel_size=None` 自动取 ~scale/4 的奇数
（prod-l1 扫描实证小核更优，大核摊薄小掩膜）；eps 归一化域 1e-4~1e-2；`subsample>1` 走
Fast Guided Filter（内部要求整除，模块已自动 pad/crop）。guide 支持 RGB / 灰度（`gray_guide=True`）。

## 自检（`selfcheck.py`）

prod-l1 前 100 组 写→读→上采样往返 + IoU@0.5 报告（含纯 bilinear 对照）+ 三列拼图
（原掩膜 / 32×32 缓存 / 上采样恢复，best/median/worst）：

```bash
CUDA_VISIBLE_DEVICES=0 python3 tools/scache/selfcheck.py \
  --out experiments/tooling-wave1/scache [--groups 100] [--size 32] [--device cuda]
```

判据（任务卡 T5）：IoU(mean) > 0.95。2026-08-02 实测：**mean 0.9695 ✅**
（median 0.997，min 0.580——小面积软边掩膜的表示极限，详见
`experiments/tooling-wave1/scache/REPORT.md`）。
