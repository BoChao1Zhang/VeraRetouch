# EPR-011 · NOTES(实施记录与协调留痕)

## 2026-08-12 提交记录(D-20 全项)

- UNIQA_FOURIER:queue id 50,gpu0,payload pid 76511,15:39:05 start,15:40:11 ready
  (`total_optimizer_steps` 1200);`ps -p` 判活通过;log
  `/home/bc/data/runs/where_b/amort_UNIQA_fourier_20260812/train.log`。
- UNIQB_NOCOORD:queue id 51,gpu1,payload pid 76643,15:39:06 start,15:40:12 ready;
  `ps -p` 判活通过(L7_LOCAL400K 恰于提交时结束,gpu1 即刻放行);log
  `/home/bc/data/runs/where_b/amort_UNIQB_nocoord_20260812/train.log`。
- step 40 双臂:BCE 0.69→0.65 下行;L_uniq_cls/L_uniq_sel 自 ~ln(4) 起降;
  area_ratio_median 0.91/0.94 在警戒带内。n=42,752 train / 400 eval,seed 20260810。

## 协调留痕(原编排会话不可达)

原「PCH 注入模块设计和相关技术分析」会话(EPR-009/010 编排方,socket 3935573)在本 EPR
挂卡前已关闭/重启,通报无法送达(一次误投到无关的声呐调研会话,已被对方指出并忽略)。
**留言如下,由用户或接续编排会话认领:**

1. EPR-011 占用 gpu0/gpu1 各约 1.5h(1200 步),完毕即释放;EPR-009/010 排卡请在其后。
2. 代码变更涉共享模块但 PCH 路径行为不变:amort/ 新增 `uniq.py` + `tests/test_uniq.py`;
   `model/trainer/losses/evaluate/viz` + `run_amort_arm.py` 增量接线(`forward_geo` 新增
   可选 `h_where/h_mask` 参数);`waves/amort_arm.sh` case 加 UNIQ。全量 amort 测试 90/90,
   PCH 相关测试零回归。
3. EPR-007/008 的 14:36 kill 非本会话所为(当时本会话仅只读检索,无任何 q/进程操作);
   声呐调研会话亦自证排除。`q events` 记录为 `q cancel`(killed while running),
   建议排查队列外的人工/其他会话操作。**两臂被杀时约 400/1336 步,未重挂——是否重挂由
   编排会话裁定(AMD-7 等待窗口本就由它们填充,现窗口被 EPR-011 占用)。**

## 假设与保守默认(按派工协议记录)

- 锚点选 P3'@1200 **evalfix** 数(0.74172)而非 H11 原数(0.7417 旧 eval):同步数且 eval
  修复一致,配对样本表取 EPR-005 交付内 per_sample jsonl。
- 语义头与词规则路由原样保留:本轮只裁几何路径;class 头替代路由只出对照列,不接线。
- `uniq_cls=uniq_sel=0.05` 为自定权重(无上游锚,已在 PROPOSAL 标注);随
  `loss_preregistration.json` 落盘,出数后不改。
- 臂 A Fourier `bands=16, scale=1.0`(SAM PositionEmbeddingRandom 默认形态);带宽敏感性
  未扫描,若臂 A 无增益,先怀疑带宽再判死坐标基(升级路径记录在案,判据不变)。
