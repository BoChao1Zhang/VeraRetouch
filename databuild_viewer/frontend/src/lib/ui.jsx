import { useEffect, useState } from 'react'
import { img, full } from '../api.js'

// 小的数据钩子：跑一个返回 promise 的 fn，给出 {data,loading,err}
export function useAsync(fn, deps) {
  const [s, setS] = useState({ data: null, loading: true, err: null })
  useEffect(() => {
    let live = true
    setS((p) => ({ ...p, loading: true, err: null }))
    Promise.resolve(fn())
      .then((data) => live && setS({ data, loading: false, err: null }))
      .catch((err) => live && setS({ data: null, loading: false, err: String(err) }))
    return () => { live = false }
  }, deps) // eslint-disable-line
  return s
}

export const cx = (...a) => a.filter(Boolean).join(' ')

const VERDICT = {
  keep: 'text-keep border-keep/40 bg-keep/10',
  drop: 'text-drop border-drop/40 bg-drop/10',
}
export function Badge({ children, tone = 'muted', title }) {
  const tones = {
    muted: 'text-muted border-line bg-panel2',
    keep: VERDICT.keep,
    drop: VERDICT.drop,
    flag: 'text-flag border-flag/40 bg-flag/10',
    amber: 'text-safelight border-safelight/40 bg-safelight/10',
  }
  return (
    <span title={title} className={cx(
      'inline-flex items-center gap-1 rounded-full border px-2 py-px font-mono text-[11px] leading-5',
      tones[tone] || tones.muted)}>
      {children}
    </span>
  )
}

export function Verdict({ value }) {
  if (!value) return null
  const tone = value === 'keep' ? 'keep' : value === 'drop' ? 'drop' : 'flag'
  return <Badge tone={tone}>{value}</Badge>
}

export function Spinner({ label = '载入中' }) {
  return <div className="p-8 text-center font-mono text-sm text-muted">{label}…</div>
}
export function ErrBox({ err }) {
  return <div className="m-4 rounded-lg border border-drop/40 bg-drop/10 p-4 font-mono text-sm text-drop">{err}</div>
}
export function Empty({ children }) {
  return <div className="p-10 text-center font-mono text-sm text-muted">{children}</div>
}

export function SectionTitle({ n, children, right }) {
  return (
    <div className="mb-3 flex items-center gap-2">
      {n != null && <span className="font-mono text-xs text-safelight">{n}</span>}
      <h3 className="font-display text-sm font-semibold tracking-wide text-fg">{children}</h3>
      <div className="ml-auto">{right}</div>
    </div>
  )
}

export function Card({ className, children }) {
  return <div className={cx('rounded-xl border border-line bg-panel p-4', className)}>{children}</div>
}

// 缩略图，点击开原图新标签
export function Thumb({ path, w = 400, className, alt = '' }) {
  if (!path) return <div className={cx('grid place-items-center bg-panel2 text-muted', className)}>无图</div>
  return (
    <a href={full(path)} target="_blank" rel="noreferrer" className="block">
      <img src={img(path, w)} alt={alt} loading="lazy"
        className={cx('w-full rounded-lg border border-line bg-panel2 object-cover', className)} />
    </a>
  )
}

// key/value 表
export function KV({ rows }) {
  const entries = Array.isArray(rows) ? rows : Object.entries(rows || {})
  return (
    <table className="w-full border-collapse font-mono text-xs">
      <tbody>
        {entries.map(([k, v]) => (
          <tr key={k} className="border-b border-line/60 last:border-0">
            <td className="w-40 py-1 pr-3 align-top text-muted">{k}</td>
            <td className="py-1 align-top break-words text-fg">{fmtVal(v)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}
const fmtVal = (v) => {
  if (v == null || v === '') return <span className="text-muted">·</span>
  if (typeof v === 'object') return <Json data={v} />
  if (typeof v === 'number' && !Number.isInteger(v)) return v.toFixed(2)
  return String(v)
}

// 可折叠 JSON 块
export function Json({ data, open = false }) {
  const [show, setShow] = useState(open)
  const txt = typeof data === 'string' ? data : JSON.stringify(data, null, 2)
  const big = txt.length > 160
  return (
    <div>
      {big && (
        <button onClick={() => setShow((s) => !s)}
          className="mb-1 font-mono text-[11px] text-safelight hover:text-ember">
          {show ? '▾ 收起' : `▸ 展开 (${txt.length}b)`}
        </button>
      )}
      {(show || !big) && (
        <pre className="max-h-80 overflow-auto whitespace-pre-wrap break-words rounded-md border border-line bg-ink/60 p-2 font-mono text-[11px] leading-relaxed text-fg/90">{txt}</pre>
      )}
    </div>
  )
}

export function Pager({ total, page, pageSize = 200, onPage, unit = '条' }) {
  const pages = Math.max(1, Math.ceil(total / pageSize))
  return (
    <div className="flex items-center gap-2 font-mono text-xs text-muted">
      <span>{total.toLocaleString()} {unit} · {page}/{pages}</span>
      <button disabled={page <= 1} onClick={() => onPage(page - 1)}
        className="rounded border border-line px-2 py-0.5 enabled:hover:border-safelight disabled:opacity-30">‹</button>
      <button disabled={page >= pages} onClick={() => onPage(page + 1)}
        className="rounded border border-line px-2 py-0.5 enabled:hover:border-safelight disabled:opacity-30">›</button>
    </div>
  )
}

// 左 1/4 + 右 3/4 的工作台布局
export function Workbench({ rail, children }) {
  return (
    <div className="flex h-full min-h-0">
      <aside className="w-1/4 min-w-[240px] max-w-[360px] shrink-0 overflow-y-auto border-r border-line bg-panel/40 p-3">{rail}</aside>
      <main className="min-w-0 flex-1 overflow-y-auto p-5">{children}</main>
    </div>
  )
}
