## 要验证的结论

如果这份数据交付成立，我们就能说「Base SFT 与后续 8 个 Where 臂 / 8 个 What 臂读的是同一份、可从发布产物重新推导出每一个数字的数据，且 `T_lut_unseen` 里的 LUT 身份确实一次也没进过任何训练 manifest」；失败就不能说 unseen LUT generalization，报告只能退回写 held-out sample。

## 为什么需要验证它

METACANVAS §0 把「结构上支持 4000 个及更多 LUT，但是否泛化必须由 LUT-ID-disjoint 测试证明」写成了论文主张；如果评测集与训练集共享 LUT 身份或共享源图，审稿人可以一句话废掉整章结论，而这种泄漏在训练跑完之后是补不回来的。

## 怎么验的

对十个生产 build 的全部 172,580 条 SFT 行做确定性七段→两段转换、逐样本图像几何与序列长度校验、group-level LUT reserve 与四集合划分，产出 indexed tar shards + terminal manifest；然后用一套独立脚本**只读发布产物**把下面每个数字重新算一遍。

---

# S0-DATA 运行前强制校验报告（SFT spec §9 项 5-9）

数据集根：`/mnt/nfs/bc/data/datasets/sft2seg-20260804`　terminal manifest digest：`9278d721bb1e234c0a6e7ad94f8f8ae1`
生成方式：`python -m q3vl.data.cli verify` + `python -m q3vl.data.report`（可复跑）

## 0. 结论一览

| spec §9 项 | 检查 | 结果 |
|---|---|---|
| 5 | 七段→两段转换：全量计数 / 拒绝原因 / 抽样原文对照 | **PASS** |
| 6 | train / eval / dedup 三集合无交集 | **PASS** |
| 7 | indexed-shard 随机读取 / checksum / resume | **PASS** |
| 8 | 图像短边 / 长边 / 宽高比 / 32 对齐 / 视觉 token 分布 | **PASS** |
| 9 | 完整序列长度分布与 >2048 过滤数 | **PASS** |
| 2.2 | 四个隔离集合 group-level 无交集 + T_lut_unseen 的 LUT 不在训练集 | **PASS** |

**N_effective（训练）= 159,215**（定档时的 169,260 是过滤前的原始 split 数，不是最终数）

## 1. 全量计数与四个隔离集合

| split | 样本数 |
|---|---|
| T_final | 918 |
| T_lut_unseen | 433 |
| V_what | 897 |
| V_where | 896 |
| train | 159,215 |

按 build / 任务类型 / winner_confidence 分层：

```json
{
  "per_split_build": {
    "T_final": {
      "g1": 117,
      "g2": 119,
      "g3": 135,
      "g4": 123,
      "l1": 78,
      "l2": 66,
      "l3": 74,
      "l4": 65,
      "l5": 60,
      "l6": 81
    },
    "T_lut_unseen": {
      "g1": 62,
      "g2": 51,
      "g3": 71,
      "g4": 51,
      "l1": 35,
      "l2": 36,
      "l3": 23,
      "l4": 29,
      "l5": 38,
      "l6": 37
    },
    "V_what": {
      "g1": 134,
      "g2": 118,
      "g3": 134,
      "g4": 103,
      "l1": 61,
      "l2": 68,
      "l3": 61,
      "l4": 74,
      "l5": 69,
      "l6": 75
    },
    "V_where": {
      "g1": 126,
      "g2": 107,
      "g3": 145,
      "g4": 118,
      "l1": 75,
      "l2": 57,
      "l3": 53,
      "l4": 71,
      "l5": 75,
      "l6": 69
    },
    "train": {
      "g1": 21188,
      "g2": 20805,
      "g3": 21040,
      "g4": 20638,
      "l1": 12975,
      "l2": 11866,
      "l3": 12676,
      "l4": 12625,
      "l5": 12792,
      "l6": 12610
    }
  },
  "per_split_task_type": {
    "T_final": {
      "local": 424,
      "style": 494
    },
    "T_lut_unseen": {
      "local": 198,
      "style": 235
    },
    "V_what": {
      "local": 408,
      "style": 489
    },
    "V_where": {
      "local": 400,
      "style": 496
    },
    "train": {
      "local": 75544,
      "style": 83671
    }
  },
  "per_split_winner_confidence": {
    "T_final": {
      "low": 385,
      "normal": 533
    },
    "T_lut_unseen": {
      "low": 181,
      "normal": 252
    },
    "V_what": {
      "low": 330,
      "normal": 567
    },
    "V_where": {
      "low": 381,
      "normal": 515
    },
    "train": {
      "low": 65281,
      "normal": 93934
    }
  }
}
```

## 2. 拒绝报告（按阶段与原因分组）

全量账要对得上：

```text
172,580  十个 build 的 sft.jsonl 行（sft_id 无重复，与 train ∪ eval 完全相等）
     -     9  图像容器不支持（.in.dng，Pillow 只能按 TIFF 读出 raw CFA 帧）
     -  1699  dedup_drop 删除名单
     -  8371  训练侧命中被保留的 LUT identity
     -   142  未见 LUT 但落在 select 角色的源上，按纪律弃用
      162359  发布样本总数（train 159,215 + 四个评测集）
```

其余为零的项也点名一次，免得看起来是漏了：**七段结构性拒绝 0、序列超长拒绝 0、宽高比 >4:1 拒绝 0、图像损坏 0、index/checksum 失败 0、split 不一致 0**。

分组明细（合计 10,221 条）：

```json
{
  "image": {
    "image_format_unsupported": 9
  },
  "split": {
    "dedup_drop": 1699,
    "lut_reserved_for_T_lut_unseen": 8371,
    "reserved_lut_in_select_source": 142
  }
}
```

## 3. spec §9 项 5：七段 → 两段转换

- 输入七段行：172,580（十个 build 的 `sft.jsonl` 全量，sft_id 无重复）
- 七段解析成功：172,580 / 172,580；**结构性拒绝 0 条**
- 原「收束文本」存在数：0（因此一条也没有补写）
- 抽样逐字对照：24 条，全部通过 = True（明细见 `conversion_samples.md`）

每条抽样核对的六项：`where` 与 `region_scope` 逐字相等；`color` 等于其余六段按原序以 `\n` 连接；`region_scope` 未被复制进 `color`；无任何旧标签残留；「收束文本存在与否」与原文一致；`instruction` 逐字未改。

## 4. spec §9 项 6：train / eval / dedup 三集合

```json
{
  "authority_raw": {
    "train": 169260,
    "eval": 3320,
    "dedup_drop": 1699,
    "train_x_eval": 0,
    "train_x_dedup": 1667,
    "eval_x_dedup": 32
  },
  "authority_after_dedup_applied": {
    "train": 167593,
    "eval": 3288,
    "train_x_eval": 0,
    "train_x_dedup": 0,
    "eval_x_dedup": 0
  },
  "published": {
    "T_final": 918,
    "T_lut_unseen": 433,
    "V_what": 897,
    "V_where": 896,
    "train": 159215
  },
  "published_x_dedup": 0,
  "published_train_x_authority_eval": 0,
  "published_eval_x_authority_train": 0
}
```

`dedup_drop` 与原始 train/eval **相交**（1,667 / 32），且三者并集恰好等于全量 172,580，因此它是**删除名单**而非第三个平行集合；应用删除后三集合两两无交集（上表 `authority_after_dedup_applied`），发布出的任何 split 都不含 dedup id。

## 5. spec §9 项 7：indexed-shard 随机读取 / checksum / resume

```json
{
  "random_access": {
    "records": {
      "status": "ok",
      "checked": 400,
      "pool": 162359,
      "index_dir": "/mnt/nfs/bc/data/datasets/sft2seg-20260804/splits"
    },
    "images": {
      "status": "ok",
      "checked": 400,
      "pool": 162359,
      "index_dir": "/mnt/nfs/bc/data/datasets/sft2seg-20260804/splits"
    }
  },
  "resume": {
    "index_samples": 162359,
    "split_samples": 162359,
    "equal": true,
    "per_split": {
      "T_final": 918,
      "T_lut_unseen": 433,
      "V_what": 897,
      "V_where": 896,
      "train": 159215
    },
    "manifest_digest": "9278d721bb1e234c0a6e7ad94f8f8ae1eab10844602a26a1cfb0e20cab7ac840"
  },
  "indexed_datasets": {
    "records": {
      "status": "ok",
      "dataset_id": "indexed_tar_a8a17e040b320dc34fa095f582bac0f9",
      "members": 162359,
      "payload_bytes": 550209124
    },
    "images": {
      "status": "ok",
      "dataset_id": "indexed_tar_dd61c6d2221573662dfb679dce06c7f4",
      "members": 162359,
      "payload_bytes": 23679233711
    }
  }
}
```

用**训练侧自己的** reader（`q3vl.train.shards.ShardIndex/ShardStore` + `q3vl.train.dataset.Sft2SegDataset`）真读样本的结果——这是唯一一项不是「生产者自证」的检查：

```json
{
  "train": {
    "status": "ok",
    "layout": "nested",
    "n_index": 159215,
    "n_checked": 12,
    "all_pass": true,
    "checksum_verified_reads": 36,
    "examples": [
      {
        "sample_id": "sft_63fef3ffa0b87317d1886be035f39f49",
        "size_matches_plan": true,
        "geometry_reproduces": true,
        "where_matches_record": true,
        "color_matches_record": true,
        "no_legacy_tag": true
      },
      {
        "sample_id": "sft_f9bfd0633d5004baa0852df63da6100b",
        "size_matches_plan": true,
        "geometry_reproduces": true,
        "where_matches_record": true,
        "color_matches_record": true,
        "no_legacy_tag": true
      },
      {
        "sample_id": "sft_e555f86444c9adce06998ca1af6013cc",
        "size_matches_plan": true,
        "geometry_reproduces": true,
        "where_matches_record": true,
        "color_matches_record": true,
        "no_legacy_tag": true
      }
    ]
  },
  "V_where": {
    "status": "ok",
    "layout": "nested",
    "n_index": 896,
    "n_checked": 12,
    "all_pass": true,
    "checksum_verified_reads": 36,
    "examples": [
      {
        "sample_id": "sft_4decc55fa70639fa7ab9bc32d4c4d582",
        "size_matches_plan": true,
        "geometry_reproduces": true,
        "where_matches_record": true,
        "color_matches_record": true,
        "no_legacy_tag": true
      },
      {
        "sample_id": "sft_a56a109d930f3ec3f7b8581f2d2e8c8a",
        "size_matches_plan": true,
        "geometry_reproduces": true,
        "where_matches_record": true,
        "color_matches_record": true,
        "no_legacy_tag": true
      },
      {
        "sample_id": "sft_456e0429a35a8d75a4f3b3deba86609b",
        "size_matches_plan": true,
        "geometry_reproduces": true,
        "where_matches_record": true,
        "color_matches_record": true,
        "no_legacy_tag": true
      }
    ]
  },
  "V_what": {
    "status": "ok",
    "layout": "nested",
    "n_index": 897,
    "n_checked": 12,
    "all_pass": true,
    "checksum_verified_reads": 36,
    "examples": [
      {
        "sample_id": "sft_fc4faca05e15f3a99c86cf68f3a0f663",
        "size_matches_plan": true,
        "geometry_reproduces": true,
        "where_matches_record": true,
        "color_matches_record": true,
        "no_legacy_tag": true
      },
      {
        "sample_id": "sft_2b67b1aa7f8dc024f2d6678a54dedf04",
        "size_matches_plan": true,
        "geometry_reproduces": true,
        "where_matches_record": true,
        "color_matches_record": true,
        "no_legacy_tag": true
      },
      {
        "sample_id": "sft_4480e3f4b2822c304b7ec617be3e879a",
        "size_matches_plan": true,
        "geometry_reproduces": true,
        "where_matches_record": true,
        "color_matches_record": true,
        "no_legacy_tag": true
      }
    ]
  },
  "T_final": {
    "status": "ok",
    "layout": "nested",
    "n_index": 918,
    "n_checked": 12,
    "all_pass": true,
    "checksum_verified_reads": 36,
    "examples": [
      {
        "sample_id": "sft_111b9cf76a7ec1da19e40c684cdd3f21",
        "size_matches_plan": true,
        "geometry_reproduces": true,
        "where_matches_record": true,
        "color_matches_record": true,
        "no_legacy_tag": true
      },
      {
        "sample_id": "sft_28bebcffaf8a9ec74ebc7160e5d2eb34",
        "size_matches_plan": true,
        "geometry_reproduces": true,
        "where_matches_record": true,
        "color_matches_record": true,
        "no_legacy_tag": true
      },
      {
        "sample_id": "sft_d2c82d46568cf76c16aa28b40f4c50dd",
        "size_matches_plan": true,
        "geometry_reproduces": true,
        "where_matches_record": true,
        "color_matches_record": true,
        "no_legacy_tag": true
      }
    ]
  },
  "T_lut_unseen": {
    "status": "ok",
    "layout": "nested",
    "n_index": 433,
    "n_checked": 12,
    "all_pass": true,
    "checksum_verified_reads": 36,
    "examples": [
      {
        "sample_id": "sft_aab741d88b1a9e7c8413b74cc12c6527",
        "size_matches_plan": true,
        "geometry_reproduces": true,
        "where_matches_record": true,
        "color_matches_record": true,
        "no_legacy_tag": true
      },
      {
        "sample_id": "sft_daccbecfaa0d31059877dc5f4ee516bc",
        "size_matches_plan": true,
        "geometry_reproduces": true,
        "where_matches_record": true,
        "color_matches_record": true,
        "no_legacy_tag": true
      },
      {
        "sample_id": "sft_73b1013371bad717a825b6b00d07fb4e",
        "size_matches_plan": true,
        "geometry_reproduces": true,
        "where_matches_record": true,
        "color_matches_record": true,
        "no_legacy_tag": true
      }
    ]
  }
}
```

## 6. spec §9 项 8：图像契约分布

```json
{
  "violations": {},
  "orig_short_side": {
    "n": 162359,
    "min": 256,
    "p05": 360,
    "p50": 1362,
    "p95": 2854,
    "max": 6336,
    "mean": 1333.181
  },
  "out_long_side": {
    "n": 162359,
    "min": 512,
    "p05": 512,
    "p50": 768,
    "p95": 768,
    "max": 1632,
    "mean": 737.3891
  },
  "vision_tokens": {
    "n": 162359,
    "min": 256.0,
    "p05": 256.0,
    "p50": 384.0,
    "p95": 384.0,
    "max": 816.0,
    "mean": 368.6946
  },
  "vision_token_hist": {
    "256": 8322,
    "272": 286,
    "288": 437,
    "304": 1021,
    "320": 14451,
    "336": 14407,
    "352": 2410,
    "368": 1996,
    "384": 111519,
    "400": 1378,
    "416": 1813,
    "432": 837,
    "448": 2139,
    "464": 211,
    "480": 101,
    "496": 101,
    "512": 381,
    "528": 84,
    "544": 32,
    "560": 44,
    "576": 22,
    "592": 181,
    "608": 92,
    "624": 9,
    "640": 32,
    "672": 10,
    "688": 4,
    "768": 23,
    "784": 2,
    "816": 14
  },
  "aspect_in": {
    "n": 162359,
    "min": 1.0,
    "p05": 1.0166666666666666,
    "p50": 1.5,
    "p95": 1.5125553914327918,
    "max": 3.206896551724138,
    "mean": 1.4426
  },
  "aspect_rounding_error": {
    "n": 162359,
    "min": 0.0,
    "p05": 0.0,
    "p50": 0.00048828124999997224,
    "p95": 0.015745148297326986,
    "max": 0.029383886255924172,
    "mean": 0.0031
  },
  "upscaled": 23045,
  "exif_orientation_hist": {
    "1": 161164,
    "6": 13,
    "8": 1182
  },
  "format_hist": {
    "JPEG": 96710,
    "PNG": 65638,
    "TIFF": 11
  }
}
```

三个需要主 agent 知道的读数：

1. **23,045 条（14.2%）是被上采样的**——原图短边不足 512（最小 256，p05 = 360）。spec §5 写的是「等比例缩放使短边为 512」，没有下限例外，故照做；若要加下限，改 `q3vl/data` 与 `q3vl/train/imageproc` 的同一处即可（NOTES.md 决策 D-8）。
2. **宽高比 >4:1 的样本一条都没有**（实测最大 3.207），所以「长边 ≤ 2048」这一条在本语料上从来没有触发过——它是被 4:1 过滤器结构性保证的，不是被裁出来的。
3. 32 对齐带来的宽高比误差最大 0.0294，低于训练侧 `IMAGE_ASPECT_TOLERANCE = 0.032` 的门限。

入库的是按契约尺寸重编码的副本（NOTES.md 决策 D-1）。下面是把原始成员重新读出来、过一遍训练侧 `prepare_image`，再和入库图逐像素比对的结果——几何必须精确相等，唯一允许的差异是 JPEG q95 4:4:4 的量化：

```json
{
  "status": "ok",
  "n": 24,
  "size_mismatch": 0,
  "rmse_mean": 1.5041,
  "rmse_max": 2.714,
  "max_abs_diff": 21,
  "note": "difference is JPEG q95 4:4:4 quantisation only; geometry is exact",
  "all_pass": true
}
```

## 7. spec §9 项 9：完整序列长度

```json
{
  "n": 162359,
  "max": 1142,
  "p50": 653,
  "p95": 724,
  "p99": 773,
  "over_limit_remaining": 0,
  "filtered_too_long": 0,
  "histogram_256": {
    "256": 1773,
    "512": 158663,
    "768": 1885,
    "1024": 38
  }
}
```

**没有一条样本被长度过滤**：最长的完整序列只有 1142 token，p99 = 773，离 `model_max_length = 2048` 还有一倍余量。因此 spec §6 的「超长过滤」在本语料上计数为 0，而不是没做。

长度口径不是本包自己定义的：prompt 由 `q3vl.train.collator.Sft2SegCollator.build_prompt_text` 生成，target 由同一个类的 `<where>/<color>` 拼法生成。下表是「本包记录的 `total_tokens`」与「collator 真跑一遍 `encode_one` 的长度」逐条比对：

```json
{
  "n": 16,
  "placeholder_shortcut_exact": 16,
  "piecewise_equals_whole_string": 16,
  "equals_training_collator": 16,
  "all_pass": true,
  "failures": [],
  "special_token_ids": {
    "<where>": 151669,
    "</where>": 151670,
    "<color>": 151671,
    "</color>": 151672
  },
  "template_overhead_tokens": 12
}
```

## 8. T_lut_unseen 的 group-level reserve

- 保留 LUT identity：296 个（其中 259 个在 T_lut_unseen 中真实出现）
- 因此从训练集剔除：8,371 条 = **5.00%**（预算上限 5%，任务卡规定 ≤5% 直接采用）
- eval 侧被保留的样本：575，其中 433 条落在 test 角色的源上并进入 T_lut_unseen，其余落在 select 源上，按纪律弃用而非塞进选择集

| taxonomy major | eval 池 | 保留 LUT 数 | 保留 eval 样本 | 剔除训练样本 |
|---|---|---|---|---|
| 青橙暗调 | 393 | 6 | 34 | 985 |
| 黑白去色 | 365 | 18 | 52 | 918 |
| 暖调高亮 | 342 | 11 | 38 | 840 |
| 低饱褪彩 | 332 | 79 | 101 | 792 |
| 暖调复古 | 326 | 49 | 83 | 781 |
| 青绿胶片 | 313 | 29 | 60 | 798 |
| 黄绿暖调 | 313 | 10 | 40 | 774 |
| 青蓝清冷 | 312 | 6 | 25 | 770 |
| 品红冷调 | 303 | 23 | 45 | 761 |
| 复古胶片 | 287 | 65 | 97 | 952 |

## 9. 隔离审计（METACANVAS §2.2）

```json
{
  "lut_overlap_between_eval_sets": {
    "T_final|T_lut_unseen": 0,
    "V_what|T_final": 240,
    "V_what|T_lut_unseen": 0,
    "V_where|T_final": 229,
    "V_where|T_lut_unseen": 0,
    "V_where|V_what": 232
  },
  "n_reserved_luts": 259,
  "n_train_luts": 3149,
  "protocol_group_overlap_source_lut_build": {
    "T_final|T_lut_unseen": 0,
    "V_what|T_final": 0,
    "V_what|T_lut_unseen": 0,
    "V_where|T_final": 0,
    "V_where|T_lut_unseen": 0,
    "V_where|V_what": 0
  },
  "sample_id_overlap": {
    "T_final|T_lut_unseen": 0,
    "V_what|T_final": 0,
    "V_what|T_lut_unseen": 0,
    "V_where|T_final": 0,
    "V_where|T_lut_unseen": 0,
    "V_where|V_what": 0
  },
  "select_vs_test_source_overlap": 0,
  "source_group_overlap": {
    "T_final|T_lut_unseen": 201,
    "V_what|T_final": 0,
    "V_what|T_lut_unseen": 0,
    "V_where|T_final": 0,
    "V_where|T_lut_unseen": 0,
    "V_where|V_what": 0
  },
  "source_overlap": {
    "T_final|T_lut_unseen": 201,
    "V_what|T_final": 0,
    "V_what|T_lut_unseen": 0,
    "V_where|T_final": 0,
    "V_where|T_lut_unseen": 0,
    "V_where|V_what": 0
  },
  "train_x_T_lut_unseen_lut_overlap": 0,
  "train_x_eval_protocol_group_overlap": {
    "T_final": 0,
    "T_lut_unseen": 0,
    "V_what": 0,
    "V_where": 0
  },
  "train_x_eval_sample_overlap": {
    "T_final": 0,
    "T_lut_unseen": 0,
    "V_what": 0,
    "V_where": 0
  },
  "train_x_eval_source_group_overlap": {
    "T_final": 0,
    "T_lut_unseen": 0,
    "V_what": 0,
    "V_where": 0
  },
  "train_x_eval_source_overlap": {
    "T_final": 0,
    "T_lut_unseen": 0,
    "V_what": 0,
    "V_where": 0
  }
}
```

说明：`source_overlap` 中 `T_final|T_lut_unseen` 非零是**设计如此**——两者同属 test 角色，靠 LUT 身份而非源图区分；协议自己的 group 键 `(source_image_id, lut_id, build)` 在四个集合之间两两交集为 0。select 角色（V_where ∪ V_what）与 test 角色（T_final ∪ T_lut_unseen）的源集合交集为 0。

## 10. 序列与视觉 token 的分布（按 split）

```json
{
  "T_final": {
    "out_h": {
      "max": 896,
      "mean": 615.95,
      "min": 512,
      "p50": 512,
      "p95": 768
    },
    "out_w": {
      "max": 1056,
      "mean": 639.02,
      "min": 512,
      "p50": 512,
      "p95": 832
    },
    "total_tokens": {
      "max": 888,
      "mean": 650.71,
      "min": 492,
      "p50": 655,
      "p95": 722
    },
    "vision_tokens": {
      "max": 528,
      "mean": 371.49,
      "min": 256,
      "p50": 384,
      "p95": 416
    }
  },
  "T_lut_unseen": {
    "out_h": {
      "max": 896,
      "mean": 619.23,
      "min": 512,
      "p50": 512,
      "p95": 768
    },
    "out_w": {
      "max": 1056,
      "mean": 633.94,
      "min": 512,
      "p50": 512,
      "p95": 832
    },
    "total_tokens": {
      "max": 808,
      "mean": 650.86,
      "min": 501,
      "p50": 652,
      "p95": 737
    },
    "vision_tokens": {
      "max": 528,
      "mean": 370.59,
      "min": 256,
      "p50": 384,
      "p95": 416
    }
  },
  "V_what": {
    "out_h": {
      "max": 768,
      "mean": 621.56,
      "min": 512,
      "p50": 512,
      "p95": 768
    },
    "out_w": {
      "max": 1536,
      "mean": 632.79,
      "min": 512,
      "p50": 512,
      "p95": 768
    },
    "total_tokens": {
      "max": 1069,
      "mean": 651.91,
      "min": 468,
      "p50": 654,
      "p95": 722
    },
    "vision_tokens": {
      "max": 768,
      "mean": 371.18,
      "min": 256,
      "p50": 384,
      "p95": 384
    }
  },
  "V_where": {
    "out_h": {
      "max": 896,
      "mean": 594.57,
      "min": 512,
      "p50": 512,
      "p95": 768
    },
    "out_w": {
      "max": 928,
      "mean": 658.0,
      "min": 512,
      "p50": 768,
      "p95": 768
    },
    "total_tokens": {
      "max": 806,
      "mean": 653.44,
      "min": 483,
      "p50": 657,
      "p95": 730
    },
    "vision_tokens": {
      "max": 464,
      "mean": 370.29,
      "min": 256,
      "p50": 384,
      "p95": 384
    }
  },
  "train": {
    "out_h": {
      "max": 1536,
      "mean": 605.75,
      "min": 512,
      "p50": 512,
      "p95": 768
    },
    "out_w": {
      "max": 1632,
      "mean": 643.55,
      "min": 512,
      "p50": 672,
      "p95": 768
    },
    "total_tokens": {
      "max": 1142,
      "mean": 648.97,
      "min": 435,
      "p50": 653,
      "p95": 724
    },
    "vision_tokens": {
      "max": 816,
      "mean": 368.65,
      "min": 256,
      "p50": 384,
      "p95": 384
    }
  }
}
```

## 11. 训练侧怎么读

```python
from q3vl.train.shards import ShardIndex, ShardStore
from q3vl.train.dataset import Sft2SegDataset

index = ShardIndex.load('/mnt/nfs/bc/data/datasets/sft2seg-20260804/splits/train.index.jsonl')
store = ShardStore(shard_root='/', verify='checksum')  # shard 字段是绝对路径
dataset = Sft2SegDataset(index, store)                 # 每条给 where/color/instruction+图
```

- `/mnt/nfs/bc/data/datasets/sft2seg-20260804/manifest/terminal_manifest.json` 是 resume 与 step 计算的唯一权威：`steps_per_epoch = ceil(N_effective / 32)`，**不得**沿用定档时的 2645 / 5290。
- 每条记录同时带 `image.origin`（原 build 的 shard/offset/length/sha256）与 `image.baked`（契约尺寸副本），两条路径都可用；`image.out_h/out_w/vision_tokens` 是预算好的几何，训练时 `plan_geometry` 会算出同一组数。
- 每条记录带 `winner_confidence`，若主 agent 决定执行 CLAUDE.md 的「low 不进主训」纪律，训练侧可直接过滤，或重跑 `python -m q3vl.data.cli plan --drop-low-confidence`。

