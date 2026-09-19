# Local 监督数学核查与重训入口说明

## 结论与适用范围

1. EPR-071 的 `name: local` 行属于裁定后的全图style流：`codes_local_full_*`、自身LUT标签、全图LUT像素目标、beta=1。它不是已经训练完成的六段local。当前求码/像素损失未出现“残差当绝对颜色”的论文公式错误，不需要因此废弃或重建这批监督。
2. `mixed_codes.solve_support_code` 已正确求解 H=a*phi、T=(out−z)+a*z；`mixed_train*` 的local路径按实际支持重新求码。`multistage_loss.ChainPixelL1` 使用 z+beta*(F(z)−z)，两者一致。
3. 旧D2 `codes_chain` 是残差码，旧生成与审计内部使用 z+beta*F_residual(z)，本身并非错误的残差模型。但该码直接送入绝对颜色执行器会产生错误，二者输出相差 beta*z。已修正新链缓存的输出契约，防止后续续训跨接口混用。
4. EPR-061 的幅度问题不能由论文公式错误直接归因。当前核查未证明其历史像素训练读取了旧残差码。EPR-071 的局部文字与全图目标差异属于已确认的数据语义口径，未在本轮擅自更改。

## 已实施的修复

- `chain_codes.py` 新默认名为 `codes_chain_absolute_v2`。全像素拟合用 H=beta*phi(z)、T=(prev−z)+beta*z，输出是绝对F的系数。
- 求码后的重建误差统一按 z+beta*(F(z)−z)计算；空支持采用恒等代码约定。
- uniform对照也转换到绝对颜色语义，目标为2q−L(q)，保留为图像无关对照，不称为LUT精确逆函数。
- 新缓存记录 `code_semantics=absolute_color_v2`、执行公式、支持口径。其beta仍包含记录的强度s；不能直接把这些码与support-only alpha混用。后者应沿用`mixed_codes.solve_support_code(z, prev, alpha, ...)`重解。
- 续写前检查语义，不匹配时在打开/截断tar或改写meta前报错；即使`--no-resume`也不覆盖旧语义文件。
- 审计与stage2 manifest使用新默认名称，明确标注码与支持语义。旧审计若要回溯，可在Git历史中取得原工具；本轮不原地改写旧缓存。

## 后续六段重训的数据准备

仅当新的训练入口需要预计算的绝对颜色码标签时，才需要生成这份新缓存。使用在线正确求码路径时，不需要先重建旧D2包。

下面命令为准备好的入口，**本轮没有启动全量GPU重建或训练**。显式使用新的输出目录，避免覆盖既有manifest。

```bash
export PYTHONPATH=/home/bc/VeraRetouch
export LD_LIBRARY_PATH=/home/bc/miniconda3/envs/llm_factory/lib

# 先做小样本验证（独立目录）；此步骤需要CUDA及原始链资源。
/home/bc/envs/q3vl_sft/bin/python -m tools.epr061_cache.chain_codes \
  --splits val --limit 4 --workers 2 --split-half-every 1 \
  --out /home/bc/nfsvfs/bc/data/runs/local_absolute_v2_pilot_20260919

# 完整重建：另一个独立目录，避免把pilot的“已完成”状态误用为全量完成。
/home/bc/envs/q3vl_sft/bin/python -m tools.epr061_cache.chain_codes \
  --out /home/bc/nfsvfs/bc/data/runs/local_absolute_v2_full_20260919

/home/bc/envs/q3vl_sft/bin/python -m tools.epr061_cache.chain_manifest \
  --which stage2 \
  --out /home/bc/nfsvfs/bc/data/runs/local_absolute_v2_full_20260919
```

注意：现有D2构造仍只生成slot 1..5的颜色码，slot 0为原有Subject/where约定；本轮没有把它伪装成六个全部可训练颜色动作。新增六段local续训入口需明确Subject阶段的监督与有效槽位。

## 已完成的验证

```bash
PYTHONPATH=/home/bc/VeraRetouch /home/bc/envs/q3vl_sft/bin/python \
  -m unittest tools.epr061_cache.test_absolute_chain_codes -v
```

8项CPU测试通过：增广最小二乘独立对照、像素子集与空支持、旧残差码误用反例、缓存评估与训练执行器一致、当前local求码与像素损失一致、强度吸收、拒绝混用且保持原文件字节不变、新版语义识别。未修改运行中的EPR-071训练代码或训练产物。
