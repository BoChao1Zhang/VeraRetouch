# preset_viewer — 只读 preset 浏览界面（TOOL-PresetView-1）

浏览 `vera_source_qa.assets` 里 12,192 个 preset 的元数据 / per_probe 逐通道 / llm_qa /
源文件 / canonical build 里的真实渲染例图。只读，不改任何库表数据。

## 三行启动

```sh
# 1) 一次性建索引（可选，应用启动时也会 IF NOT EXISTS 兜底；4.2GB 表首建约 1–2 分钟）
docker exec -i research_pg psql -U research -d research -c "CREATE INDEX IF NOT EXISTS canonical_candidates_preset_idx ON canonical_candidates (preset_id);"
# 2) 自检（连两库 + 断言 + 打印 OK）
python -m tools.preset_viewer.app --config databuild.prod-g4-global25k-20260801.toml --selfcheck
# 3) 起服务（默认 127.0.0.1:8078）
python -m tools.preset_viewer.app --config databuild.prod-g4-global25k-20260801.toml --host 0.0.0.0 --port 8078
```

在仓库根目录 `/home/bc/VeraRetouch` 下跑；`python` = `/home/bc/miniconda3/bin/python`。
`--config` 指向任一 databuild TOML（0600），只读它的 `[viewer].postgres_dsn`，把 dbname
换成 `vera_source_qa` / `research` 得到两个连接串。**代码里没有硬编码密码。**
`--dbv`（默认 `http://127.0.0.1:8077`）指向主 databuild viewer，例图靠它物化，见下节。

## 接口

| 路径 | 说明 |
|---|---|
| `GET /` | 单页 UI（内嵌 HTML+原生 JS，无构建步骤） |
| `GET /api/presets` | 分页列表。`verdict`（`__null__` 表示 verdict IS NULL）/ `fmt` / `pack` / `q`（对 asset_id、preset_look_name、preset_caption ILIKE）/ `sort`=`pro_rate`(默认，NULLS LAST)\|`asset_id` / `page` / `page_size`(≤200)。返回 `total` + `items` + `facets` |
| `GET /api/preset/{asset_id}` | 详情：assets 整行（4 个 TEXT-JSON 列已解析）+ `qa`(llm_qa) + `source_file`(前 200 行，超出截断，二进制只给大小) + `candidates`(≤12 条，winner 优先 rank 升序，含 after_path / build_id / rank / winner / group 源图路径) |
| `POST /api/preset/{asset_id}/prepare` | 让 8077 物化这 ≤12 张 after 图所在的 group（逐个串行 POST，8077 自己去重）。立即返回受理列表，不等 ready。`limit=N` 只给 selfcheck 用 |
| `GET /api/preset/{asset_id}/prepare` | 逐 group 问 8077 状态并汇总：`{reachable, n_ready, n_total, all_done, groups[]}` |
| `GET /api/dbvimg?path=…&w=…` | 流式反代 8077 的 `/img`（after 图专用）。前置只放行 `/mnt/ramstage/`，权威授权由 8077 负责；8077 的 409（未物化）原样透传 |
| `GET /img?path=…&w=…` | 本地图片（before 图 / preset 源）。只放行 `/mnt/ramstage/`、`/home/bc/data/datasets/`、`/home/bc/datasets/MMArt-PPR10k/`，realpath 归一化后前缀校验，其余 403；文件不存在 404。`w` 走 Pillow 缩略 |
| `GET /healthz` | `{"ok":true,"presets":N}` |

## 例图依赖 8077（TOOL-PresetView-2）

`/mnt/ramstage/` 本机已清空，candidate 的 after 图只存在于 NFS 归档。本工具**不碰 NFS、
不自建缓存**，全程只当主 databuild viewer 的 HTTP 客户端：

1. 详情页默认只显示元数据 + 本地还在的 before 图，after 位置是「未物化」占位。
2. 点「加载例图」→ 本进程对这 ≤12 个 group **逐个串行** `POST 8077 /api/groups/{gid}/prepare?build_id=…`；
3. 前端每 3 秒轮询 `GET /api/preset/{id}/prepare`（上限 60 次 = 3 分钟），显示 `n_ready/n_total`；
4. 某个 group 转 ready，它那几张 after 图立刻换成 `/api/dbvimg?path=…&w=420`（反代 8077 `/img`）。

**前提：8077 必须在跑**（`python -m databuild_viewer.backend.app --config <同一 TOML>`）。
不可达时连接 2s 即失败，例图区挂黄条 + 占位写「8077 不可达」，页面其余部分照常。

**别狂轰**：8077 是单 worker FIFO，主人的浏览器也在用同一条队列。所以只在点按钮时才发、
逐个串行发、且从不自动预热。

## 已知现状（不是 bug）

- **before（源）图只有 MMArt-PPR10k 那批还在本地**（约 g1 build 的 10%），
  `/home/bc/data/datasets/presets_sources/**` 与 `fivek_gold/**` 均已不在本机，这些位置会裂图。
  before/after 都齐的例子：`http://127.0.0.1:8078/#/preset/rcp_5243c8d2e2b19ee6`（点「加载例图」）
- `preset_previews` 表的 before/after 图全部不存在，本工具**不消费**该表。
