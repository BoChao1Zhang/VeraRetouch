# NOTES · G2(30 源 global run + LUT 指纹标称 vs 实测 + VLM 指纹追问)

## 1. 假设 / 偏离(实施前遇到,保守默认继续,未静默拍板)

### 1.1 抽样 30 源里有 13 源的 v3.4 诊断是 low effort,`run` 硬门要求 high

- 事实:`dataset_build/agent_loop/source_annotations.py:70-72`
  ```python
  if not isinstance(provenance, Mapping) \
          or provenance.get("reasoning_effort") != "high":
      raise ValueError("offline source diagnosis was not generated at high effort")
  ```
  该常量硬编码为 `"high"`,不读 `[source_annotation] reasoning_effort`。
- 事实:`configs/agent_loop.local-v1.annotate-v34f.toml:53` 用
  `[source_annotation] reasoning_effort = "low"` 补了 v3.4 批的后半段
  (全量 5000:high 2920 / low 2080;抽样 30:high 17 / low 13)。
- 表现:直接对 30 行 manifest 跑 `run`,`_hydrate_annotations` 在第一条 low 行即抛
  `ValueError: offline source diagnosis was not generated at high effort:
  /home/bc/data/agent_loop/local-v1/annotations_v34/source_annotation_94185017b8dee1a8.json`,
  整批 0 源起跑。
- 三个候选:
  (a) 只跑 17 源 high —— 无法满足任务卡判据 accepted ≥ 24/30;
  (b) 改 `source_annotations.py` 的 high 门 —— 动的是全仓运行时判据,不在本卡授权内;
  (c) 对这 13 源用 `annotate-sources` 以 high effort 重出 v3.4 诊断。
- **采用 (c)**。理由只写事实:本卡第 2/3 步的对象是 LUT 指纹的标称/实测数字与 VLM 对指纹行的
  读数,诊断 effort 不是这两步的被测量;(a) 直接违背预注册判据,(b) 越权。
- 落地方式:新 out-dir `/home/bc/data/agent_loop/local-v1/annotations_v34_g30high/`,
  **不覆写** `annotations_v34/` 里那 13 个 low 文件(该目录按 source_id 哈希命名,同名会被
  `temporary.replace(target)` 原地覆盖,是 `docs/assets/diagnose_v34_sample30_20260824/`
  审阅板的底账)。config 为 `configs/agent_loop.g30.annotate.toml`。
- 结果口径:g30 这一批里,17 源用的是 `annotations_v34/` 原文件(与审阅板逐字相同),
  13 源用的是 `annotations_v34_g30high/` 新出的 high-effort 诊断(与审阅板不同)。
  13 源清单:
  ```
  src_5fe75aea6181ab7e src_2cd0ace0f1f60a8f src_20cda332890e3aab src_3a45dc8cbfe4bced
  src_18c8e71605067866 src_c1f26188c5f0c97e ppr10k_1679_a       src_eae1ddea2326d3a1
  src_0105c53bce9805df src_b4b1a6585541d109 src_e7389f12a5edd167 src_0f75a4b582e1f095
  src_4dfafda0e59f252d
  ```

### 1.2 「global run」按整条 run 口径跑

`cli run` 没有只跑 global 阶段的开关(`graph.py:1095-1114` 的主图固定跑到 commit_select)。
本卡按整条 run 跑,第 2 步只取 `render_record.stage='global'` 且 `status='accepted'` 的行。

### 1.3 近零阈值(第 2/3 步共用,预注册)

见 `dataset_build/tools/g30_fingerprint_ledger.py` 顶部 `NEAR_ZERO` 常量表。

### 1.4 g30 跑批期间工作区被并行改动改写(prompt revision 漂移)

- 事实:g30 的 preflight 与 run 起于 prompt registry sha `6eddedc430ae4754…`
  (preflight_key lane-1 `cb2fa79f…` / lane-2 `2e0b9bd1…`,`provider_preflight` 表实录)。
- 事实:g30 跑完后,g30b 起不来,`RuntimeError: provider_preflight_required:provider-c-lane-1`;
  当场重算 `preflight_key` 得 `d9b78dac…` / `4bc040a7…`,`prompt_revision_fingerprint()` 已变为
  `74f053a97d3cde38…`。
- 事实:`git status` 显示工作区在此期间被换过——会话开始时的 11 个 M 条目(candidates/graph/
  models/prompts/runtime/segment_fingerprints/tests 等)已消失,当前只剩
  `prompts.py` / `graph.py` / `test_agent_loop.py` 三个 M,内容是并行 G3 工作(global 轮按
  诊断增强方向 fan-out,`GLOBAL_DIRECTION_FANOUT` 进 `prompt_registry`),HEAD 也已推进到
  `c23cef0`。这是并行 agent/用户的改动,不是本卡所为。
- 事实:在 `c23cef0` 的 git worktree 里重算 `preflight_key`,得到的正是
  `cb2fa79f…` / `2e0b9bd1…`,即 g30 跑的就是 `c23cef0` 的 agent_loop 内容。
- **处置**:g30b 在 `git worktree add … c23cef0 --detach` 出来的目录里跑
  (`cd <worktree> && python -m dataset_build.agent_loop.cli --config
  /home/bc/VeraRetouch/configs/agent_loop.g30b.toml run …`),config / manifest / 凭据 /
  artifacts root 全部走绝对路径指向主仓,只有 `dataset_build/` 代码取自 `c23cef0`。
  这样 g30b 与 g30 的采样身份(model / temperature / effort / prompt revision /
  thread_revision)逐字相同,补跑不构成第二次采样。
- 未做的事:没有改工作区里那三个 M 文件,也没有 stash;并行工作原样保留。

### 1.5 未入对数表的提案

`render_record` 里 stage=`global`、status=`accepted` 的行,其 after 图的 `retention` 是
`quarantine`(`render.py:685`)。源终态提交时 `_discard_uncommitted_renders`
(`graph.py:1004`)把未进 commit 的渲染 blob 删掉,所以这些提案量得到标称行、量不到实测。
本批 3 条如此(`ledger.json` 的 `skipped`,`why="blob_missing"`)。

## 2. 结果落点

- run:g30 accepted 22 / error 5 / source_rejected 3;g30b(5 个 error 源补跑)
  accepted 1 / error 4;合并 accepted 23 / error 4 / source_rejected 3(共 30)。
  任务卡判据 accepted ≥ 24/30 未达到,补跑已按授权只做一次。
  g30b 4 个 error:`global_count_below_config_minimum` 2、`responses_output_empty` 2。
- 对数表:53 条提案(g30 51 + g30b 2),3 条未入表。
- 追问:10 条,逐槽正确率见 `docs/assets/g30_fingerprint_check_20260824/report.md`。
