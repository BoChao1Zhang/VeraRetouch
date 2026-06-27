import { useState } from 'react'
import { api } from '../api.js'
import {
  useAsync, cx, Badge, Spinner, ErrBox, Empty, Workbench, Card, Thumb,
} from '../lib/ui.jsx'

export default function Review() {
  const [kind, setKind] = useState('pairwise')
  const [reviewer, setReviewer] = useState('local')
  const [idx, setIdx] = useState(null)
  const [nonce, setNonce] = useState(0) // 提交后刷新列表状态

  const tasks = useAsync(() => api.reviewTasks(kind), [kind, nonce])

  const submit = async (payload) => {
    await api.submitReview({ ...payload, task_kind: kind, reviewer })
    setNonce((n) => n + 1)
  }

  const data = tasks.data
  const rail = (
    <div className="space-y-4">
      <div className="flex rounded-lg border border-line bg-panel2 p-0.5">
        {[['pairwise', 'top-2 选好看'], ['scalar', 'before/after 打分']].map(([v, l]) => (
          <button key={v} onClick={() => { setKind(v); setIdx(null) }}
            className={cx('flex-1 rounded-md px-2 py-1 font-mono text-[11px] transition',
              kind === v ? 'bg-safelight/15 text-safelight' : 'text-muted hover:text-fg')}>{l}</button>
        ))}
      </div>
      <label className="block space-y-1.5">
        <div className="font-mono text-[11px] uppercase tracking-wider text-muted">reviewer</div>
        <input value={reviewer} onChange={(e) => setReviewer(e.target.value)}
          className="w-full rounded-md border border-line bg-panel2 px-2 py-1 font-mono text-xs text-fg" />
      </label>
      {tasks.loading ? <Spinner /> : tasks.err ? <ErrBox err={tasks.err} /> :
        !data.available ? <Empty>无任务，等待生成候选 manifest</Empty> : (
          <div className="space-y-1">
            {data.tasks.map((t, i) => (
              <button key={t.task_id || i} onClick={() => setIdx(i)}
                className={cx('flex w-full items-center justify-between gap-2 rounded-md border px-2 py-1.5 text-left transition',
                  idx === i ? 'border-safelight/50 bg-safelight/10' : 'border-line hover:bg-panel2')}>
                <span className="truncate font-mono text-[11px] text-fg/85">{t.task_id || `#${i}`}</span>
                {t._review ? <Badge tone="keep">✓</Badge> : <Badge>待</Badge>}
              </button>
            ))}
          </div>
        )}
    </div>
  )

  return (
    <Workbench rail={rail}>
      {idx == null || !data?.tasks?.[idx] ? (
        <Empty>选左侧一个任务</Empty>
      ) : kind === 'pairwise' ? (
        <Pairwise task={data.tasks[idx]} onPick={(choice) =>
          submit({ task_id: data.tasks[idx].task_id, subject_id: data.tasks[idx].source_id, choice })} />
      ) : (
        <Scalar task={data.tasks[idx]} onScore={(score) =>
          submit({ task_id: data.tasks[idx].task_id, subject_id: data.tasks[idx].source_id, score })} />
      )}
    </Workbench>
  )
}

function Pairwise({ task, onPick }) {
  const chosen = task._review?.choice
  return (
    <div className="space-y-4">
      <Header title="选更好看的一张" task={task} sub={chosen ? `已选 ${chosen}` : null} />
      <div className="grid gap-4 md:grid-cols-2">
        {(task.candidates || []).map((c) => (
          <button key={c.id} onClick={() => onPick(c.id)}
            className={cx('group rounded-xl border-2 p-2 text-left transition',
              chosen === c.id ? 'border-safelight' : 'border-line hover:border-safelight/50')}>
            <div className="mb-2 flex items-center justify-between font-mono text-xs">
              <span className="text-muted">候选 {c.id}</span>
              {chosen === c.id && <Badge tone="amber">已选</Badge>}
            </div>
            <Thumb path={c.after_path} w={700} className="max-h-[420px] object-contain" />
          </button>
        ))}
      </div>
    </div>
  )
}

function Scalar({ task, onScore }) {
  const [v, setV] = useState(task._review?.score ?? 5)
  return (
    <div className="space-y-4">
      <Header title="给 after 打分" task={task} sub={task._review ? `当前 ${task._review.score}` : null} />
      <div className="grid gap-4 md:grid-cols-2">
        <Card><div className="mb-1.5 font-mono text-[11px] text-muted">before</div>
          <Thumb path={task.before_path} w={700} className="max-h-[420px] object-contain" /></Card>
        <Card><div className="mb-1.5 font-mono text-[11px] text-muted">after</div>
          <Thumb path={task.after_path} w={700} className="max-h-[420px] object-contain" /></Card>
      </div>
      <Card>
        <div className="flex items-center gap-4">
          <input type="range" min="0" max="10" step="0.5" value={v}
            onChange={(e) => setV(parseFloat(e.target.value))}
            className="flex-1 accent-[var(--color-safelight)]" />
          <span className="w-16 text-center font-mono text-2xl font-bold text-safelight">{v}</span>
          <button onClick={() => onScore(v)}
            className="rounded-lg bg-safelight px-4 py-2 font-mono text-sm font-semibold text-ink hover:bg-ember">
            提交分数
          </button>
        </div>
      </Card>
    </div>
  )
}

function Header({ title, task, sub }) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <h2 className="font-display text-base font-semibold">{title}</h2>
      <span className="font-mono text-[11px] text-muted">task {task.task_id}</span>
      {sub && <Badge tone="amber">{sub}</Badge>}
    </div>
  )
}
