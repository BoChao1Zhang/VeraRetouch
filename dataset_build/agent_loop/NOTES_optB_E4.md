# NOTES — E4:诊断 v3.4(直方图板 + 固定句式)进生产 annotate 链路

任务卡:E4。本文件只记录实施中的假设与待决策(保守默认已执行),不含结论。

## 改动清单

- 新增 `dataset_build/agent_loop/histogram_board.py`(叶子模块):`panel_stats` /
  `render_board` / `board_png`,常量 `BOARD_REVISION = "histogram-board-v3-640x604"`、
  `BOARD_SIZE = (640, 604)`、`BOARD_BIN_GEOMETRY`。
- `prompts.py`:`_DIAGNOSE_RULES` 逐字替换为 harness `V2_RULES`;新增
  `BOARD_IMAGE_ENCODING = {"format": "png", "passthrough": True}`;`_image()` 加
  `encoding` 参数;`diagnosis_request()` 改双图;
  `DIAGNOSE_PROMPT_REVISION = "diagnose-v2-histogram-board-v3.4"`;registry 新增
  `histogram_board` 与 `board_image_encoding` 两键(44 → 46)。
- `responses.py`:`_content` 新增 passthrough 分支(原字节 base64,不缩放不转 JPEG)。
- `source_annotations.py`:annotate 时用原图渲染板 → `put_bytes(retention="audit")` →
  双图请求;provenance 增 `board_revision` / `board_sha256` /
  `diagnose_prompt_revision`。
- 测试:`dataset_build/tests/test_histogram_board.py`(新);`test_agent_loop.py` 内既有
  `diagnosis_request` 调用点补第三参数,并新增双图装配 / passthrough / registry 断言。

## 假设与待决策(保守默认已执行,未静默拍板)

1. **已冻结的旧标注不重标。** `sources5k.annotated.jsonl` 及其
   `source_annotation_path` JSON 是在 `diagnose-v1` 下产出的,本次只换新 revision,
   *不* 触发任何重标;`validate_source_annotation` 也未新增对 `board_sha256` /
   `board_revision` 的必填校验,所以旧文件仍能原样加载通过。新 revision 只影响此后
   跑的 annotate run。若要让 5k 批口径统一,需要单独派一次全量重标(未做)。
2. **`diagnosis_request` 的第三参数设为必填。** 备选是让 board 可选(None 时退回单图)。
   选必填:v3.4 正文写死「You receive two images」,单图模式会让 prompt 与输入不一致且
   无法在类型层拦住。代价是既有测试调用点需同步改(已改,6 处)。
3. **registry 新增两键而不是一键。** 任务卡只点名 encoding 常量入 registry;这里另加
   `histogram_board`(板 revision + 尺寸 + 显示分箱),因为板的像素同样是模型看到的输入,
   板改了而 fingerprint 不动会违反预注册纪律。计数断言随之 44 → 46。
4. **provenance 多写了 `diagnose_prompt_revision`。** 任务卡只要求 `board_sha256` +
   `BOARD_REVISION`;多这一列是为了让后续 run 能按 revision 分批,不影响校验。
5. **板的读图口径。** `panel_stats` 用 `artifacts._open_image` 同款读取(先普通文件、再
   archive_reader),以便 archive 里的 source 也能渲染;像素与 harness 的
   `Image.open` 路径一致(同样 `exif_transpose` + `convert("RGB")`,原分辨率、不下采样)。
6. **板落盘位置与保留级别。** 用 `artifacts.put_bytes(media_type="image/png",
   retention="audit")`,与 source preview 同级(audit)。未新建独立目录。
7. **PNG 参数。** `save(..., "PNG", optimize=True)`,与 harness 落盘参数一致;确定性由
   `test_board_png_is_byte_identical_across_two_real_renders` 真实渲染两次比对字节守住。
8. **字体依赖。** 板依赖 `/usr/share/fonts/truetype/dejavu/DejaVuSans{,-Bold}.ttf`
   (与 harness 相同)。字体缺失时 `render_board` 直接抛错,未加降级字体回退——降级字体
   会改板的像素,破坏确定性。若生产机镜像不带 DejaVu,需先补字体,这一点未在本卡内验证。
9. **未做(任务卡明确排除)**:graph 在线阶段、`direction_match`、`candidates` 未动;
   `DIAGNOSIS_SCHEMA` 原样;没有发任何真实 API 请求;没有跑 annotate 全量。
10. **未纳入 registry 的板绘制常量。** 颜色、字号、面板坐标、`_SUPERSAMPLE` 只进
    `BOARD_REVISION`(改动需手工 bump),没有逐个进 registry;若担心「改了颜色忘 bump」,
    可后续把 `histogram_board.py` 的源码 sha256 也写进 registry(未做,待决策)。

## 判据运行

```
.venv/bin/python -m pytest dataset_build/tests/test_agent_loop.py \
    dataset_build/tests/test_source_histogram.py \
    dataset_build/tests/test_histogram_board.py -q
200 passed
```

`dataset_build/tests/test_canonical_orchestration.py` 与 `test_annot_contract_v4.py` 在
本环境仍是 collection error(`ModuleNotFoundError: No module named 'construct'`),
改动前后一致,与本卡无关。
