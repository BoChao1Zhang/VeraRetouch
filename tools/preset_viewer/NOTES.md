# NOTES — TOOL-PresetView-1 / -2 实施前核实记录 / 假设 / 待决策

日期 2026-08-11。工具类任务，无引用实验文档章节；任务卡「已核实事实」5 条逐条实查，
结果见下（第 1 节）。外部事实：本任务不依赖任何仓库外 URL / 论文数字，无需在线核实。

**裁定已入档（2026-08-11，主 agent）**：D1① 第三个根**保留**；D1② **不许自己碰 NFS
归档**，after 图改走 8077 的 prepare 通道（→ TOOL-PresetView-2，第 4 节）；D2 / D3 / D4
**全部照准，不改**。下文 D1–D4 保留原文作为记录。

## 1. 任务卡事实核对（逐条实查）

| # | 卡上说法 | 实查结果 |
|---|---|---|
| 1 | 密码在 `databuild.prod-g4-global25k-20260801.toml` 的 `[viewer].postgres_dsn` | **属实**。该文件 0600。代码通过 `--config` 读取，用 `psycopg.conninfo` 换 dbname，未硬编码 |
| 1 | PG 在容器 `research_pg`，`127.0.0.1:5432` | **属实**。宿主机**没有装 `psql`**，命令行查库要么走 `docker exec -i research_pg psql`，要么用 python `psycopg`（README 已按前者给命令） |
| 2 | `assets` 里 `asset_type='preset'` 共 12192 行 | **属实**（12192） |
| 2 | 关键列 / 4 个 TEXT-JSON 列 | **属实**。`preset_per_probe` 非空 8027 行（= `status='preset_meta_pass'` 的行数，两者同集） |
| 2 | `llm_qa` 按 asset_id 关联，`questionnaire` 区分轮次 | **属实**。preset 走 `questionnaire='preset'`，逐 probe（`probe_id`）× 逐 item（H1/T1/I2/P2…）一行；单个 preset 最多 780 行，8027 个 preset 有 QA。已有索引 `ix_llmqa_asset` |
| 2 | `preset_previews` 89994 行、图已全没 | **属实**（抽查 5 条 before/after 全部 `os.path.exists()=False`）。本工具不消费该表 |
| 3 | `canonical_candidates` 162 万行 / 4.2 GB，distinct preset 3522 | **属实**（reltuples 1,625,160；`pg_total_relation_size` 4248 MB；distinct preset_id 3522，与 assets 的 12192 交集 **3522，canonical-only 0**，两边同域） |
| 3 | preset_id / after_path / winner / rank 在 `payload` jsonb 里 | **部分不符**：这些同时是 `canonical_candidates` 的**普通列**且已填充（见 D2）。`canonical_groups` 同理有普通列 `source_path`，不必挖 payload |
| 3 | `after_path` 指向 `/mnt/ramstage/<build>/…`，"ramstage 上现存可读" | **不符**：`/mnt/ramstage/` 是**空目录且未挂载任何 fs**（`mount \| grep ramstage` 无输出，`ls` 只有 `.`/`..`）。**没有任何 after 图在本机**。见 D1 |
| 4 | `/img` 只放行 `/mnt/ramstage/` 与 `/home/bc/data/datasets/` | 这两个根下**一张 canonical 例图都没有**（见 D1）。已加第三个根 |
| 5 | Python 3.13 + fastapi/uvicorn/psycopg/Pillow 已装 | **属实**（psycopg 3.3.4 / fastapi 0.129.0 / uvicorn 0.41.0 / Pillow 12.2.0 / httpx 0.28.1，TestClient 可用） |

补充实查：preset 源文件 `assets.path` 本地存在 **12066 / 12192**（缺 126 个 xmp）。
`.cube` 最大 7.7 MB，故源文件展示按「只读文件头 512 KB → 取前 200 行」实现。

## 2. 待主 agent 决策

### D1（重要）ramstage 已清空 → `/img` 加了第三个白名单根，且 selfcheck 的 after 图断言被降级

**事实**：`/mnt/ramstage/` 空且未挂载，1,625,160 条 candidate 的 `after_path` **无一存在**。
`canonical_groups.source_path` 也大多失效（`/home/bc/data/datasets/presets_sources/**`、
`fivek_gold/**` 都不在本机），**唯一还在本地的一批源图是 `/home/bc/datasets/MMArt-PPR10k/`**
（注意根是 `/home/bc/datasets/`，不是卡上写的 `/home/bc/data/datasets/`）：抽 g1 build 的
3000 条 winner，392 条（约 13%）的 group 源图可读。

**我采用的保守默认**：
1. `ALLOWED_ROOTS` 在卡定的两个根之外**增加 `/home/bc/datasets/MMArt-PPR10k/`**（`app.py`
   顶部常量，带注释）。理由：否则例图功能完全是死的，一张图都出不来。realpath 归一化 +
   前缀校验照旧，白名单外与 `../` 穿越均 403（selfcheck 有断言）。
2. selfcheck 的「`/img` 对一个存在的 after_path 返回 200」**无法满足**，改成：优先找存在的
   after_path，找不到就退到存在的 group source_path，并**打印一条 `[WARN]`** 说明降级原因。
3. UI 在「例图区」检测到「有 candidate 但无一张 after 图存在」时挂一条黄条提示，避免用户
   把满屏裂图误判成工具坏了。

**请主 agent 裁定**：(a) 第三个白名单根是否保留；(b) 是否需要另派任务把 after 图从 NFS
indexed-tar 归档（`dataset_build/tools/archive_reader.py` / `land.py` 的产物）里取回来，
或改成「按需从归档流式解出」——本任务卡明确「不需要碰 NFS」，我没有做这件事。

### D2 索引建在真实列 `preset_id` 上，不是卡上写的表达式 `(payload->>'preset_id')`

`canonical_candidates.preset_id` 是**已填充的普通 text 列**，查询直接用它更快也更省。
故实际执行的是：

```sql
CREATE INDEX IF NOT EXISTS canonical_candidates_preset_idx ON canonical_candidates (preset_id);
```

已于 2026-08-11 在 `research` 库执行完毕（**这是本工具唯一的写操作，主 agent 已授权**）。
索引 11 MB；建后 `EXPLAIN ANALYZE` 走 Index Scan，单 preset 取 12 条 0.003–0.29 s
（最坏情况是 `rcp_d303ee0391f28cad` 那种 16,250 条候选的 preset）。执行时库里有两条跑了
10+ 分钟的历史慢查询，`CREATE INDEX` 的 SHARE 锁与它们的 ACCESS SHARE 不冲突，未受阻。
应用启动时还会跑一次 `IF NOT EXISTS` 兜底（`--no-index` 可关），失败只打印不拦启动。

### D3 `facets` 塞进了列表响应，没有单开 `/api/facets`

卡说「别加不在清单里的功能」，但左侧三个筛选下拉需要候选值。为不新增端点，把
verdict / fmt / pack 的 distinct + count 放在 `/api/presets` 响应的 `facets` 字段里
（assets 只有 12k 行，三条聚合查询开销可忽略）。若认为应拆成独立端点或干脆改文本框输入，
请裁定。

### D4 `verdict` 的 NULL 桶用哨兵值 `__null__`

`preset_clean_verdict IS NULL` 有 4165 行（占 34%），是最大的第二桶，不给入口不合理。
HTTP query 无法表达 NULL，故约定 `verdict=__null__` → `IS NULL`。若不喜欢哨兵值可改。

## 3. TOOL-PresetView-2 事实核对（8077 prepare 通道）

实查 `databuild_viewer/README.md` 与 `databuild_viewer/backend/app.py`（**只读，未改一
个字节**），并对运行中的 8077 打了真实请求：

| 项 | 实查结果 |
|---|---|
| 8077 在跑 | 是。`pid 2869888`，`--config databuild.prod-g4-global25k-20260801.toml --nfs-ledger /mnt/nfs-ro/bc/data/builds --host 0.0.0.0` |
| `POST /api/groups/{gid}/prepare?build_id=…` | 返回 `{schema_version, build_id, group_id, state, files:{done,total}, bytes:{done,total}, current_item, message, updated_at}`。**任务是同步登记的**：POST 刚返回 `queued`，紧接着 GET 就能查到（这条性质被用在 `_summary` 里，见下） |
| `GET` 同路径 | 未起过任务 → **404 `asset preparation not started`**（正常初始态，不是错误） |
| ready 后 `GET /img?path=<canonical>` | 200。默认是 `w=512` 缩略，要原图得带 `full=true`（本工具的 `/api/dbvimg` 无 `w` 时即转 `full=true`） |
| 未物化的 group 的 `/img` | **409 `archived image is not prepared`**，cache-only，符合 README |
| 实测耗时 | 单 group 从 POST 到 ready **约 3 秒**（`bytes.total` 2.8–7.8 MB）；一次 12 个 group 串行 POST 后 3 秒内 12/12 ready |
| `/api/health` | 10s 内没返回（疑似要摸 NFS），本工具**不依赖**它，只用 prepare + /img |

本次全部测试对 8077 的 prepare 用量：**13 个 distinct group**（手工探路 1 + selfcheck 1 +
一次 12 张例图的完整链路，其中 1 个与探路重合），在 ≤15 的预算内。

## 4. TOOL-PresetView-2 新增决策项

### D5（已按卡裁定执行）`/api/dbvimg` 只做廉价前缀过滤，权威校验交给 8077

按任务卡「选后者」：本进程只断言 `os.path.realpath(path).startswith("/mnt/ramstage/")`
（挡住 `../` 穿越和明显不属于 canonical 域的路径），真正的授权是 8077 的
`_authorized_image_path`。没有维护「只接受该 preset 详情返回过的 after_path」白名单——
那要么得存会话状态、要么每次反查一遍库，收益是零（8077 那边照样会拦）。
8077 的 403/404/409/422 原样透传，其余错误折成 502。

### D6 `POST /api/preset/{id}/prepare` 加了 `limit` 参数（默认 12 = 全量）

卡说「别加不在清单里的功能」，但 selfcheck 需要能只打扰 8077 队列**一个** group。
前端始终用默认值（全部 12 个）。同理，selfcheck 选 preset 的那条 SQL 加了
`ORDER BY preset_id`，让重复自检永远命中同一个 group，不会每跑一次就多占一个坑。
若认为 `limit` 该去掉、selfcheck 改成直接调内部函数，请裁定。

### D7 轮询侧把 `not_started` 也算终态

8077 的 POST 是同步登记任务的（实测 POST 返回后立刻可 GET），所以 POST 之后 GET 仍然
404 只可能是那次 POST 根本没登记上（例如 8077 加载的 build 集合与我们查的库不一致，
POST 直接 404 `group not found`）。这种情况继续轮询没有意义，故 `not_started` 计入
`all_done`，前端进度条会显示 `not_started:N` 让人看见。POST 侧的 404 单独标成
`not_found` 状态（区别于 GET 侧的 `not_started`）。

### D8 前端轮询上限 3 分钟（60 次 × 3s）

卡只写「3 秒轮询直到全部 ready/failed」，没给上限。加了 60 次的硬上限，到点停止并提示
「可再点按钮」，避免用户切走后页面在后台无限轮询（切换 preset 时也会主动停）。
另外重复点按钮是安全的：8077 对已 ready 的 group 自会去重。

## 5. 已知假设（自行核实过，不需决策）

- **只读实现方式**：两个业务连接都在 conninfo 里带 `options='-c default_transaction_read_only=on'`
  （会话级）。没有用「连上后 `SET`」——psycopg3 非 autocommit 下 `SET` 是事务局部的，
  commit 后会失效，等于没设。已实测：该连接执行 `CREATE TABLE` 抛 `ReadOnlySqlTransaction`；
  selfcheck 里有 `SHOW default_transaction_read_only = on` 的断言。
- **每请求新开连接**、无连接池、无鉴权、无写库（human_review 之类一概没有）——单人本地工具。
- **`statement_timeout=120s`** 加在两个业务连接上，防止误触发全表扫时卡死 worker。
- 未修改 `databuild_viewer/` 任何文件（只读了它的 README 与 `backend/app.py`）；未读
  `trash/`；未写 NFS（`/mnt/nfs*` 全程未写入，只在排查 after 图去向时 `ls` 过
  `/mnt/nfs-ro`；after 图现在完全由 8077 去归档取，本进程不碰）。
- 前端是内嵌字符串，无 npm / 无构建；`/` 只返回常量 HTML，不做 DB 查询。
- 对 8077 的 HTTP 客户端：`httpx.Timeout(30.0, connect=2.0)`（连接 2s / 读 30s，按卡）。
  连接失败一律折成结构化 `{"error": ...}` 往上抛数据，不让异常冒到路由层——所以 8077
  挂掉时详情页照常出，只有例图区降级。
