import { useEffect, useState } from 'react'
import { ExternalLink, ImageOff, LoaderCircle } from 'lucide-react'
import { full, img } from '../api.js'

export function useAsync(fn, deps) {
  const [state, setState] = useState({ data: null, loading: true, err: null })
  useEffect(() => {
    let live = true
    setState({ data: null, loading: true, err: null })
    Promise.resolve(fn())
      .then((data) => live && setState({ data, loading: false, err: null }))
      .catch((error) => live && setState({ data: null, loading: false, err: String(error) }))
    return () => { live = false }
  }, deps) // eslint-disable-line react-hooks/exhaustive-deps
  return state
}

export const cx = (...values) => values.filter(Boolean).join(' ')

export function Badge({ children, tone = 'neutral', title }) {
  return (
    <span title={title} className={cx('badge', `badge-${tone}`)}>
      {children}
    </span>
  )
}

export function IconButton({ label, children, className, ...props }) {
  return (
    <button type="button" aria-label={label} title={label} className={cx('icon-button', className)} {...props}>
      {children}
    </button>
  )
}

export function Spinner({ label = 'Loading' }) {
  return (
    <div className="state-box" role="status">
      <LoaderCircle size={18} className="animate-spin" aria-hidden="true" />
      <span>{label}</span>
    </div>
  )
}

export function ErrorBox({ error }) {
  return (
    <div className="state-box state-error" role="alert">
      <strong>Request failed</strong>
      <span>{String(error || 'Unknown error')}</span>
    </div>
  )
}

export function Empty({ title = 'No data', children }) {
  return (
    <div className="state-box state-empty">
      <ImageOff size={20} aria-hidden="true" />
      <strong>{title}</strong>
      {children ? <span>{children}</span> : null}
    </div>
  )
}

export function AssetImage({ path, width = 960, alt = '', className, link = false }) {
  if (!path) {
    return (
      <div className={cx('asset-placeholder', className)}>
        <ImageOff size={20} aria-hidden="true" />
      </div>
    )
  }
  const image = <img src={img(path, width)} alt={alt} loading="lazy" className={className} />
  if (!link) return image
  return (
    <a className="asset-link" href={full(path)} target="_blank" rel="noreferrer" title="Open original">
      {image}
      <ExternalLink size={14} aria-hidden="true" />
    </a>
  )
}

export function Metric({ label, value, accent = false }) {
  return (
    <div className={cx('metric', accent && 'metric-accent')}>
      <span>{label}</span>
      <strong>{formatValue(value)}</strong>
    </div>
  )
}

export function Field({ label, children, mono = false }) {
  return (
    <div className="field">
      <span className="field-label">{label}</span>
      <div className={cx('field-value', mono && 'font-mono')}>{children ?? 'n/a'}</div>
    </div>
  )
}

export function JsonBlock({ data }) {
  return <pre className="json-block">{JSON.stringify(data ?? {}, null, 2)}</pre>
}

export function formatValue(value, digits = 2) {
  if (value == null || value === '') return 'n/a'
  if (typeof value === 'boolean') return value ? 'yes' : 'no'
  if (typeof value === 'number' && !Number.isInteger(value)) return value.toFixed(digits)
  return String(value)
}
