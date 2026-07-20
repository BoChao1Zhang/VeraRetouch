import { AlertTriangle, Database } from 'lucide-react'
import { api } from './api.js'
import Inspector from './Inspector.jsx'
import { Badge, useAsync } from './lib/ui.jsx'

export default function App() {
  const health = useAsync(() => api.health(), [])
  const status = health.data

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="brand-block">
          <span className="brand-mark" aria-hidden="true" />
          <div>
            <h1>Canonical Databuild</h1>
            <span>inspection console</span>
          </div>
        </div>
        <div className="header-status" aria-live="polite">
          {health.err ? (
            <Badge tone="danger" title={health.err}>
              <AlertTriangle size={12} aria-hidden="true" /> backend unavailable
            </Badge>
          ) : status ? (
            <>
              <Badge tone={status.source === 'postgres' ? 'success' : 'accent'}>
                <Database size={12} aria-hidden="true" /> {status.source}
              </Badge>
              <span>{status.builds ?? 0} builds</span>
              {status.malformed_records > 0 && (
                <Badge tone="warning">{status.malformed_records} malformed</Badge>
              )}
            </>
          ) : (
            <span>connecting</span>
          )}
        </div>
      </header>
      <Inspector />
    </div>
  )
}
