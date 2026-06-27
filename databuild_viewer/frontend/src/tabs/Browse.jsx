import { useState } from 'react'
import { api, img } from '../api.js'
import {
  useAsync, cx, Badge, Verdict, Spinner, ErrBox, Empty, Pager,
  Workbench, Card, Thumb, KV, Json, SectionTitle,
} from '../lib/ui.jsx'

export default function Browse() {
  const [type, setType] = useState('image')
  const [corpus, setCorpus] = useState(null)
  const [sort, setSort] = useState('aes')
  const [order, setOrder] = useState('desc')
  const [page, setPage] = useState(1)
  const [sel, setSel] = useState(null)

  const corpora = useAsync(() => api.corpora(type), [type])
  const list = useAsync(() => api.list({ type, corpus, sort, order, page }), [type, corpus, sort, order, page])

  const switchType = (t) => { setType(t); setCorpus(null); setPage(1); setSel(null) }

  const rail = (
    <div className="space-y-4">
      <Toggle value={type} onChange={switchType}
        options={[['image', 'source 图'], ['preset', 'preset']]} />
      <div className="space-y-1.5">
        <Label>排序</Label>
        <div className="flex gap-1.5">
          <select value={sort} onChange={(e) => { setSort(e.target.value); setPage(1) }} className={selCls}>
            <option value="aes">{type === 'image' ? 'merit_frac' : 'pro_rate'}</option>
            <option value="created">入库时间</option>
          </select>
          <select value={order} onChange={(e) => { setOrder(e.target.value); setPage(1) }} className={selCls}>
            <option value="desc">↓ 高→低</option>
            <option value="asc">↑ 低→高</option>
          </select>
        </div>
      </div>
      <div>
        <Label>{type === 'image' ? '数据集 corpus' : 'pack'}</Label>
        <div className="mt-1.5 space-y-0.5">
          <CorpusRow active={corpus === null} onClick={() => { setCorpus(null); setPage(1); setSel(null) }}
            name="全部" n={corpora.data?.reduce((a, c) => a + c.n, 0)} />
          {corpora.data?.map((c) => (
            <CorpusRow key={c.corpus} active={corpus === c.corpus} name={c.corpus || '·'} n={c.n}
              onClick={() => { setCorpus(c.corpus); setPage(1); setSel(null) }} />
          ))}
        </div>
      </div>
    </div>
  )

  return (
    <Workbench rail={rail}>
      {sel ? (
        <Detail id={sel} onBack={() => setSel(null)} />
      ) : list.loading ? <Spinner /> : list.err ? <ErrBox err={list.err} /> : (
        <>
          <div className="mb-4 flex items-center justify-between">
            <h2 className="font-display text-base font-semibold">
              {type === 'image' ? 'Source 图' : 'Preset'}
              {corpus && <span className="ml-2 font-mono text-xs text-muted">/ {corpus}</span>}
            </h2>
            <Pager total={list.data.total} page={page} onPage={setPage} />
          </div>
          <div className="grid grid-cols-[repeat(auto-fill,minmax(150px,1fr))] gap-3">
            {list.data.items.map((it) => (
              <GridCard key={it.asset_id} item={it} type={type} onClick={() => setSel(it.asset_id)} />
            ))}
          </div>
          {!list.data.items.length && <Empty>无数据</Empty>}
        </>
      )}
    </Workbench>
  )
}

const selCls = 'min-w-0 flex-1 rounded-md border border-line bg-panel2 px-2 py-1 font-mono text-xs text-fg'
const Label = ({ children }) => <div className="font-mono text-[11px] uppercase tracking-wider text-muted">{children}</div>

function Toggle({ value, onChange, options }) {
  return (
    <div className="flex rounded-lg border border-line bg-panel2 p-0.5">
      {options.map(([v, label]) => (
        <button key={v} onClick={() => onChange(v)}
          className={cx('flex-1 rounded-md px-2 py-1 font-mono text-xs transition',
            value === v ? 'bg-safelight/15 text-safelight' : 'text-muted hover:text-fg')}>
          {label}
        </button>
      ))}
    </div>
  )
}

function CorpusRow({ active, name, n, onClick }) {
  return (
    <button onClick={onClick}
      className={cx('flex w-full items-center justify-between rounded-md px-2 py-1 font-mono text-xs transition',
        active ? 'bg-safelight/15 text-safelight' : 'text-fg/80 hover:bg-panel2')}>
      <span className="truncate">{name}</span>
      {n != null && <span className="text-muted">{n.toLocaleString()}</span>}
    </button>
  )
}

function GridCard({ item, type, onClick }) {
  return (
    <button onClick={onClick}
      className="group overflow-hidden rounded-lg border border-line bg-panel text-left transition hover:border-safelight/50">
      {type === 'image' ? (
        <img src={img(item.path, 280)} loading="lazy" alt=""
          className="aspect-square w-full bg-panel2 object-cover" />
      ) : (
        <div className="grid aspect-square w-full place-items-center bg-panel2 font-mono text-xs text-muted">
          {item.kind || 'preset'}
        </div>
      )}
      <div className="space-y-1 p-2">
        {type === 'image' ? (
          <div className="flex items-center justify-between">
            <span className="font-mono text-xs text-safelight">merit {fmt(item.merit_frac)}</span>
            <Verdict value={item.final_decision} />
          </div>
        ) : (
          <div className="flex flex-wrap items-center gap-1">
            <span className="font-mono text-[11px] text-fg/80">{item.pack_id || '·'}</span>
            {item.pass_c === 1 && <Badge tone="keep">pass_c</Badge>}
            <Badge tone={item.has_embedding ? 'amber' : 'muted'}>{item.has_embedding ? 'emb✓' : 'emb✗'}</Badge>
          </div>
        )}
        <div className="truncate font-mono text-[10px] text-muted">{item.asset_id}</div>
      </div>
    </button>
  )
}
const fmt = (v) => (v == null ? '·' : typeof v === 'number' && !Number.isInteger(v) ? v.toFixed(2) : v)

// ---- 详情 ----
function Detail({ id, onBack }) {
  const { data, loading, err } = useAsync(() => api.asset(id), [id])
  if (loading) return <Spinner />
  if (err) return <ErrBox err={err} />
  const a = data.asset
  const isPreset = a.asset_type === 'preset'
  return (
    <div className="space-y-4">
      <button onClick={onBack} className="font-mono text-xs text-safelight hover:text-ember">‹ 返回列表</button>
      <div className="flex flex-wrap items-center gap-2">
        <h2 className="font-display text-base font-semibold">{id}</h2>
        <Verdict value={a.final_decision} />
        {isPreset && <Badge tone={data.has_embedding ? 'amber' : 'muted'}>{data.has_embedding ? 'emb✓' : 'emb✗'}</Badge>}
      </div>
      {isPreset ? <PresetDetail a={a} data={data} /> : <ImageDetail a={a} data={data} />}
    </div>
  )
}

function ImageDetail({ a, data }) {
  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
      <Card><Thumb path={a.path} w={900} className="max-h-[460px] object-contain" /></Card>
      <div className="space-y-4">
        <Card>
          <SectionTitle>基本</SectionTitle>
          <KV rows={{ corpus: a.corpus, scene: a.scene, size: `${a.width}×${a.height}`, status: a.status,
            aesthetic_vlm: a.aesthetic_vlm, merit_frac: a.merit_frac, musiq: a.musiq,
            pass_a: a.pass_a, pass_b: a.pass_b, auto_verdict: a.auto_verdict }} />
        </Card>
        {data.caption && <Card><SectionTitle>caption</SectionTitle><Json data={data.caption} open /></Card>}
      </div>
      <Card className="lg:col-span-2">
        <SectionTitle n="R1">第 1 轮 QA · 验真 A / 适配 B</SectionTitle>
        <QaTable rows={data.round1} />
      </Card>
      <Card className="lg:col-span-2">
        <SectionTitle n="R2" right={<Badge tone="amber">merit_frac {fmt(a.merit_frac)}</Badge>}>
          第 2 轮 QA · 审美 AES
        </SectionTitle>
        <QaTable rows={data.round2} />
      </Card>
      {!!data.decisions?.length && <Card className="lg:col-span-2"><SectionTitle>决策</SectionTitle><Json data={data.decisions} /></Card>}
    </div>
  )
}

function PresetDetail({ a, data }) {
  return (
    <div className="space-y-4">
      <Card>
        <SectionTitle>preset 元信息</SectionTitle>
        <KV rows={{ pack_id: a.pack_id, kind: a.kind, fmt: a.fmt, status: a.status, pass_c: a.pass_c,
          look_name: a.preset_look_name, grade_family: a.preset_grade_family,
          clean_verdict: a.preset_clean_verdict, pro_rate: a.preset_pro_rate,
          intent_rate: a.preset_intent_rate, coherence: a.preset_coherence }} />
      </Card>
      {a.preset_caption && <Card><SectionTitle>VLM caption</SectionTitle><Json data={a.preset_caption} open /></Card>}
      <div className="grid gap-4 md:grid-cols-2">
        {a.preset_axes && <Card><SectionTitle>axes</SectionTitle><KV rows={a.preset_axes} /></Card>}
        {a.preset_per_probe && <Card><SectionTitle>6 探针指标</SectionTitle><Json data={a.preset_per_probe} /></Card>}
      </div>
      <Card>
        <SectionTitle right={<span className="font-mono text-xs text-muted">{data.previews?.length || 0} 个</span>}>
          探针 before / after
        </SectionTitle>
        <div className="grid grid-cols-[repeat(auto-fill,minmax(220px,1fr))] gap-3">
          {data.previews?.map((p, i) => (
            <div key={i} className="space-y-1.5 rounded-lg border border-line bg-panel2 p-2">
              <div className="flex items-center justify-between font-mono text-[11px] text-muted">
                <span>探针 {i + 1}</span><span>{p.render_engine}</span>
              </div>
              <div className="grid grid-cols-2 gap-1.5">
                <Thumb path={p.before_path} w={260} className="aspect-square object-cover" alt="before" />
                <Thumb path={p.after_path} w={260} className="aspect-square object-cover" alt="after" />
              </div>
              {p.paired_metrics && <Json data={p.paired_metrics} />}
            </div>
          ))}
        </div>
      </Card>
    </div>
  )
}

function QaTable({ rows }) {
  if (!rows?.length) return <Empty>无</Empty>
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse font-mono text-xs">
        <thead>
          <tr className="text-left text-muted">
            <th className="py-1 pr-3 font-medium">问卷</th>
            <th className="py-1 pr-3 font-medium">项</th>
            <th className="py-1 pr-3 font-medium">答</th>
            <th className="py-1 font-medium">理由</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((q, i) => (
            <tr key={i} className="border-t border-line/60 align-top">
              <td className="py-1 pr-3 text-safelight">{q.questionnaire}</td>
              <td className="py-1 pr-3">{q.item}</td>
              <td className="py-1 pr-3">{q.answer == null ? '·' : String(q.answer)}</td>
              <td className="py-1 text-fg/80">{q.rationale || ''}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
