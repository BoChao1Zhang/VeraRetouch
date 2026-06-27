// 后端薄封装。/img 走 query 传绝对路径（后端有白名单校验）。
const j = async (u) => {
  const r = await fetch(u)
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`)
  return r.json()
}

export const api = {
  corpora: (type) => j(`/api/corpora?type=${type}`),
  list: ({ type, corpus, sort, order, page }) =>
    j(`/api/list?type=${type}&sort=${sort}&order=${order}&page=${page}&page_size=200` +
      (corpus ? `&corpus=${encodeURIComponent(corpus)}` : '')),
  asset: (id) => j(`/api/asset/${encodeURIComponent(id)}`),
  runs: () => j('/api/runs'),
  groupList: ({ run, page }) =>
    j(`/api/group/list?page=${page}&page_size=200` + (run ? `&run=${encodeURIComponent(run)}` : '')),
  group: (id) => j(`/api/group/${encodeURIComponent(id)}`),
  reviewTasks: (kind) => j(`/api/review/tasks?kind=${kind}`),
  submitReview: async (payload) => {
    const r = await fetch('/api/review', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
    if (!r.ok) throw new Error(await r.text())
    return r.json()
  },
}

// 缩略图 / 原图 URL
export const img = (path, w = 512) => path ? `/img?w=${w}&path=${encodeURIComponent(path)}` : ''
export const full = (path) => `/img?full=1&path=${encodeURIComponent(path)}`
