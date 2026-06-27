import { useState } from 'react'
import { api, img } from '../api.js'
import {
  useAsync, cx, Badge, Spinner, ErrBox, Empty, Pager,
  Workbench, Card, Thumb, KV, Json, SectionTitle,
} from '../lib/ui.jsx'

// role → 帧的色轨 + 徽章
const ROLE = {
  sft: { rail: 'bg-keep', badge: <Badge tone="keep">SFT</Badge> },
  dpo_chosen: { rail: 'bg-safelight', badge: <Badge tone="amber">DPO✓</Badge> },
  dpo_rejected: { rail: 'bg-drop', badge: <Badge tone="drop">DPO✗</Badge> },
}

export default function Build() {
  const [run, setRun] = useState(null)
  const [page, setPage] = useState(1)
  const [sel, setSel] = useState(null)

  const runs = useAsync(() => api.runs(), [])
  const list = useAsync(() => api.groupList({ run, page }), [run, page])

  const rail = (
    <div className="space-y-4">
      <div className="space-y-1.5">
        <div className="font-mono text-[11px] uppercase tracking-wider text-muted">run</div>
        <select value={run || ''} onChange={(e) => { setRun(e.target.value || null); setPage(1); setSel(null) }}
          className="w-full rounded-md border border-line bg-panel2 px-2 py-1 font-mono text-xs text-fg">
          <option value="">全部 run</option>
          {runs.data?.map((r) => (
            <option key={r.run_id} value={r.run_id}>{r.run_id.slice(0, 26)} · {r.route} ({r.n})</option>
          ))}
        </select>
      </div>
      {list.loading ? <Spinner /> : list.err ? <ErrBox err={list.err} /> : (
        <>
          <Pager total={list.data.total} page={page} onPage={setPage} unit="groups" />
          <div className="space-y-1">
            {list.data.items.map((g) => (
              <button key={g.group_id} onClick={() => setSel(g.group_id)}
                className={cx('flex w-full items-center gap-2 rounded-lg border p-1.5 text-left transition',
                  sel === g.group_id ? 'border-safelight/50 bg-safelight/10' : 'border-line hover:bg-panel2')}>
                <img src={img(g._src_path, 90)} loading="lazy" alt=""
                  className="h-11 w-11 shrink-0 rounded bg-panel2 object-cover" />
                <div className="min-w-0">
                  <div className="font-mono text-[11px] text-fg/90">
                    {g.route} · {g.n_candidates} cand{g.is_portrait ? ' · portrait' : ''}
                  </div>
                  <div className="truncate font-mono text-[10px] text-muted">{g.group_id.slice(0, 16)}</div>
                </div>
              </button>
            ))}
          </div>
          {!list.data.items.length && <Empty>construct 流水线尚未入库</Empty>}
        </>
      )}
    </div>
  )

  return (
    <Workbench rail={rail}>
      {sel ? <GroupDetail id={sel} /> : (
        <Empty>选左侧一张源图 group，查看它的 databuild 链路</Empty>
      )}
    </Workbench>
  )
}

function GroupDetail({ id }) {
  const { data, loading, err } = useAsync(() => api.group(id), [id])
  if (loading) return <Spinner />
  if (err) return <ErrBox err={err} />
  const { group: g, candidates, sft, dpo } = data
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-baseline gap-3">
        <h2 className="font-display text-base font-semibold">group {id.slice(0, 16)}</h2>
        <div className="flex gap-1.5 font-mono text-xs text-muted">
          <Badge>{g.route}</Badge>
          <span>{candidates.length} 候选</span>·<span>{sft.length} SFT</span>·<span>{dpo.length} DPO</span>
        </div>
      </div>

      {/* ① 源图 */}
      <Card>
        <SectionTitle n="①">源图 source</SectionTitle>
        <div className="grid gap-4 md:grid-cols-[320px_1fr]">
          <Thumb path={g._src_path} w={640} className="max-h-72 object-contain" />
          <KV rows={{ source_asset_id: g.source_asset_id, route: g.route,
            is_portrait: g.is_portrait, run_id: g.run_id, source_path: g.source_path }} />
        </div>
      </Card>

      {/* ② 候选胶片条 —— signature */}
      <div>
        <SectionTitle n="②">候选渲染 · recall → rerank → QA</SectionTitle>
        <div className="rounded-xl border border-line bg-panel">
          <div className="sprockets h-2 rounded-t-xl" />
          <div className="flex gap-3 overflow-x-auto p-3">
            {candidates.map((c) => <Frame key={c.cand_id} c={c} />)}
            {!candidates.length && <Empty>无候选</Empty>}
          </div>
          <div className="sprockets h-2 rounded-b-xl" />
        </div>
      </div>

      {/* ③ SFT */}
      {!!sft.length && (
        <Card>
          <SectionTitle n="③">SFT 产出</SectionTitle>
          <div className="space-y-4">
            {sft.map((s) => (
              <div key={s.sft_id} className="grid gap-3 md:grid-cols-[260px_1fr]">
                <Thumb path={s.i_tar} w={420} className="max-h-60 object-contain" />
                <div className="space-y-2">
                  <Field label="instruction" v={s.instruction} />
                  <Field label="reasoning" v={s.reasoning} />
                  <div className="grid gap-2 sm:grid-cols-2">
                    <div><div className="mb-1 font-mono text-[11px] text-muted">recipe</div><Json data={s.recipe} /></div>
                    {s.qa && <div><div className="mb-1 font-mono text-[11px] text-muted">QA</div><Json data={s.qa} /></div>}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* ④ DPO */}
      {!!dpo.length && (
        <Card>
          <SectionTitle n="④">DPO 偏好对</SectionTitle>
          <div className="space-y-4">
            {dpo.map((p) => (
              <div key={p.dpo_id} className="rounded-lg border border-line bg-panel2 p-3">
                <div className="mb-2 font-mono text-xs text-muted">margin {fmt(p.margin)}</div>
                <div className="grid gap-4 sm:grid-cols-2">
                  <Side tone="keep" label="chosen" after={p.chosen_after} recipe={p.chosen} />
                  <Side tone="drop" label="rejected" after={p.rejected_after} recipe={p.rejected} />
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  )
}

// 一帧候选
function Frame({ c }) {
  const role = ROLE[c.role]
  return (
    <div className="w-56 shrink-0 overflow-hidden rounded-lg border border-line bg-panel2">
      <div className={cx('h-1', role?.rail || 'bg-line')} />
      <div className="space-y-2 p-2">
        <div className="flex items-center gap-1.5">
          <span className="font-mono text-sm font-bold text-fg">#{c.rank ?? '·'}</span>
          {role?.badge}
          {c.veto && <Badge tone="drop">veto</Badge>}
          {c.reliable === false && <Badge tone="flag">unreliable</Badge>}
        </div>
        <Thumb path={c.after_path} w={320} className="aspect-[4/3] object-cover" />
        <KV rows={{ preset_id: c.preset_id, kind: c.kind, fmt: c.fmt,
          region: c.local ? '有蒙版' : 'global', merit: c.merit_score }} />
        {c.merit_hits && <Json data={c.merit_hits} />}
        {c.qa && <Json data={c.qa} />}
      </div>
    </div>
  )
}

function Side({ tone, label, after, recipe }) {
  return (
    <div className="space-y-2">
      <Badge tone={tone}>{label}</Badge>
      <Thumb path={after} w={420} className="max-h-56 object-contain" />
      <Json data={recipe} />
    </div>
  )
}

function Field({ label, v }) {
  return (
    <div>
      <div className="mb-1 font-mono text-[11px] text-muted">{label}</div>
      <p className="whitespace-pre-wrap rounded-md border border-line bg-ink/50 p-2 text-sm leading-relaxed text-fg/90">{v || '·'}</p>
    </div>
  )
}
const fmt = (v) => (v == null ? '·' : typeof v === 'number' && !Number.isInteger(v) ? v.toFixed(2) : v)
