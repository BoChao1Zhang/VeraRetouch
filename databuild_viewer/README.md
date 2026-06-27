# databuild viewer

datagen-v2 数据集的浏览 / 构建过程可视化 / 人工 review 网页。React + TailwindCSS 前端，
FastAPI 后端，只读浏览 Postgres `vera_source_qa` + 磁盘渲染图，唯一写操作是人工 review 落库。

```
databuild_viewer/
  backend/app.py      FastAPI：/api/* 数据接口 + /img 缩略图服务，复用 dataset_build.source_qa.db
  frontend/           Vite + React + Tailwind v4 单页应用（浏览 / 构建过程 / review 三个 Tab）
```

## 三个 Tab

- **浏览** — 左侧切 source 图 / preset，按 corpus 过滤、按 aes 分排序、分页（≤200/页）。详情：
  source 图看 2 轮 QA（验真 A/B、审美 AES merit）+ caption + aes 分；preset 看 6 探针 before/after
  + axes + VLM caption + embedding 状态。
- **构建过程** — construct 流水线的中间结果已**全部入库**。每个 `group` = 一张源图的 databuild 单元，
  按链路展开：源图 → 候选胶片条（preset 渲染 + QA + 排名 + role，after 图持久化在 render_stage）
  → SFT 产出 → DPO 偏好对。读 `construct_groups/candidates/sft/dpo` 库表。
- **人工 review** — 读预计算候选 manifest（`preset_bank_v2/review/{pairwise,scalar}_tasks.jsonl`）。
  top-2 选更好看的一张；before/after 给 after 打标量分。结果 upsert 到 `human_review` 表。

## 运行

后端（先起，默认 127.0.0.1:8077，端口可 `DBV_PORT` 覆盖）：

```bash
cd /home/bc/VeraRetouch
python -m databuild_viewer.backend.app
python -m databuild_viewer.backend.app --selfcheck   # 不起服务，跑一遍数据自检
```

前端开发（Vite 在 5173，自动把 /api、/img 代理到后端）：

```bash
cd databuild_viewer/frontend
npm install
npm run dev          # 打开 http://localhost:5173
```

前端生产构建（构建后后端直接从 `frontend/dist` 提供 SPA，单端口 8077 访问）：

```bash
cd databuild_viewer/frontend && npm run build
```

> 单人本地工具：每请求新开 DB 连接，无连接池 / 无鉴权。`/img` 只放行
> `/home/bc/data/datasets/` 下的路径。
