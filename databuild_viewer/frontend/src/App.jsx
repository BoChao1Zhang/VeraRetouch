import { useState } from 'react'
import { cx } from './lib/ui.jsx'
import Browse from './tabs/Browse.jsx'
import Build from './tabs/Build.jsx'
import Review from './tabs/Review.jsx'

const TABS = [
  { id: 'browse', label: '浏览', sub: 'source · preset' },
  { id: 'build', label: '构建过程', sub: 'construct' },
  { id: 'review', label: '人工 review', sub: 'top-2 · 打分' },
]

export default function App() {
  const [tab, setTab] = useState('browse')
  return (
    <div className="flex h-full flex-col">
      <header className="flex shrink-0 items-stretch border-b border-line bg-panel/60">
        <div className="flex items-center gap-2.5 px-5">
          <span className="h-2.5 w-2.5 rounded-full bg-safelight shadow-[0_0_10px_2px] shadow-safelight/50" />
          <span className="font-display text-sm font-bold tracking-[0.2em] text-fg">DATABUILD</span>
          <span className="font-mono text-[11px] text-muted">console</span>
        </div>
        <nav className="flex">
          {TABS.map((t) => (
            <button key={t.id} onClick={() => setTab(t.id)}
              className={cx('group relative px-5 text-left',
                tab === t.id ? 'text-fg' : 'text-muted hover:text-fg')}>
              <div className="font-display text-sm font-medium">{t.label}</div>
              <div className="font-mono text-[10px] opacity-70">{t.sub}</div>
              <span className={cx('absolute inset-x-3 bottom-0 h-0.5 rounded-full transition',
                tab === t.id ? 'bg-safelight' : 'bg-transparent group-hover:bg-line')} />
            </button>
          ))}
        </nav>
      </header>
      <div className="min-h-0 flex-1">
        {tab === 'browse' && <Browse />}
        {tab === 'build' && <Build />}
        {tab === 'review' && <Review />}
      </div>
    </div>
  )
}
