# Robust Lightroom render server/client — 根因与需求

> 2026-06-15 ｜ 触发：VeraRetouch source_qa 用本 LR 任务 server(:8081) 批量真渲染预设做 QA，
> param 预设在并发 6 下 **62% 失败**（`lrc:error 6822 / done 4181`），错误全是笼统 `lr render failed`。
> 把并发降到 **3（匹配 3 个 client，1:1 不排队）后失败率 → ~6%**。**单预设独立渲染必成功** →
> 证明不是「Lightroom 不支持这些预设」，而是**负载下客户端退化**。本文给根因(带 file:line)+ 鲁棒性需求清单，供迭代。

## 一、根因（代码 + 现场证据）

| # | 根因 | 证据 (file:line) | 后果 |
|---|---|---|---|
| **R1** | **catalog 只进不出（主因）** | `XMPlayer.lrplugin/InitPlugin.lua:243-247`：`findPhotoByPath(photoPath)` 未命中则 `catalog:addPhoto(photoPath)`；全文件无 `removePhoto(s)`/trash。每个任务用唯一 `/tmp/lightroom_task_<uuid>/before.jpg`，`findPhotoByPath` 必 miss → 每次都新增。批处理分支同样(`:373-377`) | catalog 累积上万张 → develop/export 越来越慢 |
| **R2** | **bridge 固定 120s 导出超时** | `lrc_api_server.py:14` `BRIDGE_WAIT_TIMEOUT=120.0`，`:123` 用它等 `processed/<stem>.jpg` 落盘；catalog 慢了等不到 → 返回 None → client 收 500 → 报 failed | catalog 退化后大面积超时失败 |
| **R3** | **客户端任务目录不清理** | `lr_task_client.py:649-666` 建 `lightroom_task_<uuid>/`(before.jpg+config.lua+processed/)；upload+report 后全程无 `rmtree/unlink`(grep 为空) | 上万任务目录 + 导出图堆积，FS 变慢 |
| **R4** | **服务端结果持久化（现状=只内存逐出、盘上保留）** → **重定为设计** | `lrc_task_server.py:370` 每任务建 `results_dir/<id>/`；`_evict_completed_task:148-153` 只从**内存** dict pop、盘上 `results/<id>/processed.jpg` **保留**；现场 `lr_caches` 10,428 目录 / 2.8G | **按用户决策：server 持久化结果（durable store），盘上保留是预期行为**；client 端改为 LRU=100。仅需可选的容量/TTL 归档防单盘满，不再当 bug |
| **R5** | **无背压/过载信号** | 提交端并发 6 打 3 个 client；client 无「我过载」反馈(`ClientInfo` 仅 status/last_seen)，server 不按 client 容量限流(`get_task` 先到先得) | 退化期继续灌任务，雪上加霜 |
| **R6** | **重试与中毒不分** | 失败无区分「瞬时(超时/过载)可重试」vs「永久(预设确实不被支持)」；`report_result` 的 error 在我方只落成一句 `lr render failed` | 无法定位、无法只重试该重试的 |

> 一句话：**addPhoto 只进不出 + 两端都不清理 + 固定超时**，量一上来 Lightroom 必然退化到大面积超时。可解的工程问题，非 Lightroom 能力问题。

## 二、鲁棒性需求清单

### A. 有界缓存 / 不堆积（核心，对应 R1/R3/R4）

> **设计决策（2026-06-16，用户定）**：客户端**保留最近 100 张已编辑图片**作为热缓存（提升鲁棒性 + 兼顾性能），**最近队列以外的全部从 catalog 和本地删除**；**服务端持久化结果**（server 是结果的长期持有方，client 不长期保留）。即 A1/A2 从"catalog→0"改为"**catalog/本地 LRU=100**"，A3 从"有界淘汰"改为"**server 持久化**"。

- **A1 catalog LRU=100（根治 R1，保留热缓存）**：维护已导入照片的 LRU 队列；每次 `addPhoto` 后若 catalog 张数 > 100，对**最旧**的若干张 `catalog:removePhotos`（移出 catalog，不动盘）。catalog 稳态 ≈ 100，既根治膨胀又留热缓存。
  - 注：本工作负载探针是**固定小集合（3–6 张）被所有预设反复用**；若客户端用**稳定探针路径**（而非每任务唯一 `/tmp/<uuid>/before.jpg`），`findPhotoByPath` 会命中 → 同一探针只导入一次、catalog ≈ 探针数。可与 LRU=100 叠加（稳定路径优先、LRU 兜底）。
- **A2 客户端本地 LRU=100**：每任务的 `task_dir`（before/config + 导出 processed）也按最近 100 保留、其余 `rmtree`。upload 成功后该任务结果已在 server，客户端无需长期保留。
- **A3 服务端持久化结果（durable store）**：`results/<id>/processed.jpg`（+ 未来的 mask）由 server **长期持有**（持久化到盘，不随内存逐出而删）；client 端 LRU=100 之外不保留。明确「**server = 结果长期持有方，client = 易失热缓存**」。（可选：server 侧加按容量/TTL 的归档而非删除，避免单盘无界——但默认持久化保留。）
- **A4 失败路径也清理**：异常/超时分支同样把本地超出 LRU 的删掉、把最旧的移出 catalog（幂等清理），不能只在成功路径清。

### B. 超时与自适应（对应 R2）
- **B1 取消固定 120s**：导出等待自适应（按近期渲染 p95 / catalog 规模动态给预算）或显式分级超时。
- **B2 超时层级一致**：`bridge等导出 < client HTTP读 < server processing 重置(现 30min) < 提交端 job 超时`，避免某层先放弃造成「孤儿处理中」。

### C. 健康 / 背压 / 限流（对应 R5）
- **C1 client 健康上报**：注册/心跳带 catalog 当前张数、近 N 次渲染延迟、磁盘余量、是否在 reset。
- **C2 server 按容量派发**：依 client 健康限流/暂停派发；client 可主动 `busy`/`draining`。
- **C3 提交端尊重容量**：并发度按「在线且健康的 client 数」自动调（而非写死）。source_qa 侧 `preset_qa.run_stage2 --workers` 现为手动，可对接 C1/C2 自动化。

### D. 重试 / 中毒隔离（对应 R6）
- **D1 错误分类**：端到端透传**结构化失败原因**（plugin 报错文本 / 超时 / 导出缺失 / 预设解析失败 / catalog 慢），不再一句 `lr render failed`。`report_result` 的 `error`/`result_data` 已能带，需 plugin→client→server→提交端逐层透传。
- **D2 分类处置**：瞬时类(超时/过载)→带退避重试 N 次；永久类(预设真不被支持)→标 `unsupported` 不再重试，单独成清单供人看。
- **D3 attempts/backoff**：记录尝试次数与原因，幂等可续。

### E. 可观测性
- **E1 指标**：成功率、p50/p95 渲染延迟、catalog 张数随时间、两端磁盘占用、各失败类计数。
- **E2 关联日志**：`task_id` 贯穿 client/plugin/server/提交端，便于追一个失败任务的全链路。

### F. 自动处理 Lightroom 应用级弹窗 / 防卡死（对应新根因 R7）

**R7（新根因）：Lightroom 应用级模态弹窗阻塞 SDK 任务。** 插件用 `photo:applyDevelopSettings`（`InitPlugin.lua:258,381`）+ `catalog:withWriteAccessDo`，这些会等 Lightroom 主线程；一旦 LrC 弹出**需要人点的模态框**，SDK 调用永久阻塞 → bridge 等不到导出 → 超时/卡死。**插件/SDK 无法关闭 Adobe 原生模态框**。常见触发：
- **AI/Sensei 同意**：首次用 AI 功能（AI Denoise、Select Sky/Subject 等 AI mask）弹"启用 AI/云功能"同意框——**正好命中带 AI mask 的 local-mask 预设**。
- **账号/授权**：Adobe ID 登录、license 重新校验。
- **启动 nag**：What's New、欢迎页、"是否与云同步"、有更新可用。
- **目录类**：升级 catalog、catalog 被占用、退出时备份提示。
- **Develop 类**：缺相机配置/profile、"该预设由更新版本创建"、照片缺失。

**解决方法（分层，越靠前越根治）：**
- **F1 源头消除——把 Lightroom 预配成"无弹窗"（最优）**：一次性登录并设为不再提示；关闭自动更新、What's New、欢迎页、云同步、"显示导入对话框"、退出时备份设为"从不"、自动人脸/地点查找；用**已升级好的专用 catalog**（免升级提示）；**锁定 LrC 版本**别让它运行中自更新。多数项可在 `首选项` / `Lightroom Classic … Preferences.agprefs`(mac) / 注册表(win) 里预置 "don't show again"。
- **F2 AI/Sensei 预先同意一次**：手动跑一次 AI mask / AI Denoise 并勾"同意/不再提示"，之后批量不再弹。或：检测 AI-mask 子类型的预设单独路由/标记（`tableContainsMaskSettings`(`InitPlugin.lua:127`) 已能识别 mask，可再细分 AI mask）。
- **F3 OS 级自动点掉（兜底，覆盖未知弹窗）**：Win 用 **AutoHotkey**、mac 用 **Hammerspoon/AppleScript UI scripting** 常驻监听已知弹窗标题/按钮 → 自动点 "OK/同意/不再显示"。这是对付未知弹窗的最稳兜底。
- **F4 看门狗 + 自动恢复（防一个弹窗冻死整机）**：client 端若某任务超过 T 无导出进度 → 先发 Esc/Enter 尝试关框；仍卡 → **杀掉并重启 Lightroom**；把该任务报 failed 让 server 重排。避免单个模态框冻住整个 client 队列。
- **F5 健康态上报"阻塞中"**：client 检测到卡在弹窗 → 上报 `blocked`，server 暂停向它派发（接 C1/C2）。

### G. 导出 AI/Sensei mask 栅格（用户需求：用了 AI mask 的预设把对应 mask 也导出）

**现状**：带 mask 的预设走 `photo:updateAISettings()`（`InitPlugin.lua:261,382`，让 LrC 计算 AI mask），随后只导出**最终合成 after JPEG**——**mask 本身没有单独回传**。

**公共 SDK 能力（关键约束）**：Lightroom Classic 公共 SDK **没有"导出某个 mask 的栅格/选区位图"的 API**。AI/Sensei mask（Select Subject/Sky/People/Object、AI Denoise 等）是 Sensei 在渲染期内部计算的，develop 设置里**只存"选择主体/天空"这类指令，不含像素级选区**；几何 mask（Radial/Gradient/Brush/Range）则把**几何/参数**存在 develop 设置（XMP/config.lua）里。

**可行方案（按保真度/可行性）：**
- **G1 几何 mask → 从参数栅格化（精确，可离线）**：Radial/Gradient/Brush/Range 的几何在 XMP/`MaskGroupBasedCorrections` 里，可在我方(source_qa)按参数**精确重建** mask 栅格，无需 LrC 配合。覆盖几何类，但**不覆盖 AI 类**（AI 类无几何）。
- **G2 差分法提取（推荐，覆盖 AI + 几何，近似但实用）**：对同一探针渲染**两次**——(a) 预设原样；(b) 预设里**把该 mask 的局部校正推到极端且可识别**（如该 mask 的局部 Exposure 设 +5 或染纯品红、其余清零）——两张 after 的**逐像素差**即该 mask 的**软 alpha**（含羽化，正好适合做 region-local 的 `C_GT`）。因为 LrC 真正渲染了该选区（含 AI 选区），所以对 AI mask 也有效。多/重叠 mask 需**逐 mask 各一遍**（按 `MaskGroupBasedCorrections` 拆开、每次只极端化一个）。代价：每个 mask 多 1 次渲染。
- **G3 仅检测+标记（最低成本兜底）**：扩展 `tableContainsMaskSettings`(`InitPlugin.lua:127`) 细分 **AI mask 子类型**，把"含 AI mask"标进 `render_jobs`/`recipe_index.qa.jsonl`（`qa_local_edit` 之上再加 `qa_ai_mask`），先标记、mask 栅格留待 G2 批量补。
- **G4 mask overlay 截图（不推荐）**：驱动 LrC UI 显示 mask 叠加再截屏——非 headless、脆弱，否决。

**协议/落地（待实现，本文仅设计）**：
- LR 任务增加 `export_masks` 选项；client 对每个 mask 走 G2（或 G1 几何直接重建），把 `mask_<i>.png`(软 alpha) 连同 `processed.jpg` 一并 upload；server 持久化到 `results/<id>/masks/`。
- source_qa 侧 `lr_render` 取回 mask、`preset_qa` 把 mask 路径写进 `preset_previews`/新列，供未来 region-aware 训练当 `C_GT`。
- 优先级：先 **G3 标记**（便宜、立即可用）；需要真 mask 栅格时上 **G2 差分法**（param/AI 通吃）+ **G1**（几何精确）。**纯公共 SDK 无法直接导出 AI mask 位图，G2 是不改 Adobe、可 headless 的最现实路径。**

## 附：为什么转 `config.lua` 而不是直接加载 `.xmp`？

不是 Lightroom 不支持 xmp（xmp 就是它磁盘上的原生预设格式），而是 **LrC SDK 没有"把任意 .xmp 文件应用到某张照片"的 API**。插件实际走 `photo:applyDevelopSettings(settingsTable)`（`InitPlugin.lua:258,381`），它**只吃 Lua 设置表**；`parseLuaSettingsFile`(`:65`) 用 `loadfile`+`pcall` 期望文件 `return {…}`。SDK 里跟 xmp 沾边的两条路都不适合逐任务任意预设：① xmp 作为 **RAW 的 sidecar** 在导入/读元数据时生效——但探针是 JPEG，且是导入期机制；② 把 xmp **装进 Develop 预设文件夹**当 `LrDevelopPreset` 再 `applyDevelopPreset`——要管理预设目录 + 刷新/重启，几千个任意预设极笨重。所以 **xmp → Lua 设置表 → applyDevelopSettings** 是 SDK 原生且能带 mask 的正确做法（转换本身不是失败源，已验证 mask 块完整保留）。

## 三、source_qa（消费方）侧已就绪、可配合
- 真渲染走 `dataset_build/source_qa/lr_render.py`：`submit_task_with_files → task_status(long-poll) → download_task_result`；预设转 `config.lua`（`.xmp`→`xmp2lua` 保留 mask、`.lrtemplate`→`LuaConverter` 取 `value.settings`）。
- `render_jobs` 表已按 `(preset_content_hash, probe_id, render_engine)` 缓存、`UNIQUE` 幂等；**重跑 stage2 只重试 error/pending**，client 修好后一次性重跑即可捞回失败。
- 预设级续跑守卫：跳过已出 `preset_render=ok` 的预设，只处理失败/未渲染的。
- 若 **D1** 落地，可把结构化失败原因写进 `render_jobs.error`（现仅 `lr render failed`），并据 **D2** 决定重试/弃（标 `unsupported`）。
- 并发度建议：当前经验值 **workers=3 ≈ 1:1 匹配 3 个 client**（62%→6% 失败）；A1/B1 落地后可上调。

## 四、建议落地顺序（按性价比）
1. **A1（catalog LRU=100）+ A2（本地 LRU=100）** — 根治退化、留热缓存，单点改 `InitPlugin.lua` + `lr_task_client.py`，收益最大。
2. **F1+F2（Lightroom 预配无弹窗 + AI/Sensei 预先同意）+ F4（看门狗重启）** — 防卡死；**对带 AI mask 的 local-mask 预设尤其关键**（否则首次必弹同意框冻死）。
3. **B1（自适应超时）+ A4（失败路径清理）** — 消除误超时与堆积。
4. **A3（服务端持久化结果）** — 确认 durable，可选加容量/TTL 归档防单盘满。
5. **D1+D2（错误分类+中毒隔离）** — 让重试精准、暴露真正不兼容的预设。
6. **G（AI mask 导出）** — 先 **G3 标记**（便宜、立即可用）；需要真 mask 栅格时上 **G2 差分法**（AI+param 通吃）+ **G1**（几何精确）。
7. **C1–C3 + F3/F5（健康/背压/自动并发 + OS 自动点弹窗/阻塞上报）** — 长期稳态吞吐。

> 改完 `.lrplugin` 需重新打包推到远端 Win/Mac client 才生效（`dist/` 下有打包产物）。
