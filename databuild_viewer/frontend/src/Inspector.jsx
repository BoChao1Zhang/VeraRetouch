import { useEffect, useMemo, useState } from 'react'
import {
  AlertTriangle,
  Check,
  ChevronLeft,
  ChevronRight,
  Clipboard,
  HardDriveDownload,
  Image as ImageIcon,
  Layers,
  RefreshCw,
  RotateCcw,
  ScanLine,
  SlidersHorizontal,
} from 'lucide-react'
import { api, img } from './api.js'
import {
  AssetImage,
  Badge,
  cx,
  Empty,
  ErrorBox,
  Field,
  formatValue,
  IconButton,
  JsonBlock,
  Metric,
  Spinner,
  useAsync,
} from './lib/ui.jsx'

const PAGE_SIZE = 50
const EMPTY_FILTERS = {
  build_id: '',
  mode: '',
  format: '',
  major: '',
  minor: '',
  queue: '',
  failure: '',
  winner: 'any',
}

const VIEW_MODES = [
  { id: 'rendered', label: 'Rendered', icon: ImageIcon },
  { id: 'mask', label: 'C_GT', icon: ScanLine },
  { id: 'overlay', label: 'Overlay', icon: Layers },
]

export default function Inspector() {
  const [filters, setFilters] = useState(EMPTY_FILTERS)
  const [page, setPage] = useState(1)
  const [selectedGroup, setSelectedGroup] = useState('')
  const builds = useAsync(() => api.builds(), [])

  useEffect(() => {
    if (!filters.build_id && builds.data?.length) {
      setFilters((current) => ({ ...current, build_id: builds.data[0].build_id }))
    }
  }, [builds.data, filters.build_id])

  const facets = useAsync(
    () => filters.build_id ? api.facets(filters.build_id) : Promise.resolve({}),
    [filters.build_id],
  )
  const filterKey = JSON.stringify(filters)
  const groups = useAsync(
    () => filters.build_id
      ? api.groups(filters, page, PAGE_SIZE)
      : Promise.resolve({ total: 0, page: 1, page_size: PAGE_SIZE, items: [] }),
    [filterKey, page],
  )
  const activeGroup = groups.data?.items.some((row) => row.group_id === selectedGroup)
    ? selectedGroup
    : groups.data?.items[0]?.group_id || ''

  useEffect(() => {
    if (!groups.data) return
    const visible = groups.data.items.some((row) => row.group_id === selectedGroup)
    if (!visible) setSelectedGroup(groups.data.items[0]?.group_id || '')
  }, [groups.data, selectedGroup])

  const updateFilter = (name, value) => {
    setFilters((current) => {
      if (name === 'build_id') return { ...EMPTY_FILTERS, build_id: value }
      return { ...current, [name]: value }
    })
    setPage(1)
    setSelectedGroup('')
  }

  const resetFilters = () => {
    setFilters({ ...EMPTY_FILTERS, build_id: filters.build_id })
    setPage(1)
    setSelectedGroup('')
  }

  return (
    <div className="inspector-layout">
      <aside className="inspector-rail">
        <Filters
          builds={builds.data || []}
          facets={facets.data || {}}
          facetError={facets.err}
          filters={filters}
          onChange={updateFilter}
          onReset={resetFilters}
        />
        <GroupRail
          result={groups}
          page={page}
          onPage={setPage}
          selected={activeGroup}
          onSelect={setSelectedGroup}
        />
      </aside>
      <main className="inspector-main">
        {builds.err ? <ErrorBox error={builds.err} /> : !builds.loading && !builds.data?.length ? (
          <Empty title="No canonical builds" />
        ) : groups.err ? <ErrorBox error={groups.err} /> : groups.loading && !groups.data ? (
          <Spinner label="Loading groups" />
        ) : activeGroup ? (
          <GroupInspector groupId={activeGroup} buildId={filters.build_id} />
        ) : (
          <Empty title="No matching groups" />
        )}
      </main>
    </div>
  )
}

function Filters({ builds, facets, facetError, filters, onChange, onReset }) {
  const [expanded, setExpanded] = useState(false)
  return (
    <section className="filter-panel" aria-label="Dataset filters">
      <div className="section-heading compact-heading">
        <div>
          <span className="eyebrow">Dataset</span>
          <h2>Filters</h2>
        </div>
        <div className="filter-actions">
          <IconButton
            label={expanded ? 'Collapse filters' : 'Expand filters'}
            className="mobile-filter-toggle"
            aria-expanded={expanded}
            onClick={() => setExpanded((value) => !value)}
          >
            <SlidersHorizontal size={16} aria-hidden="true" />
          </IconButton>
          <IconButton label="Reset filters" onClick={onReset}>
            <RotateCcw size={16} aria-hidden="true" />
          </IconButton>
        </div>
      </div>

      {facetError ? <ErrorBox error={facetError} /> : null}

      <div className={cx('filter-controls', expanded && 'expanded')}>
        <SelectFilter label="Build" value={filters.build_id} onChange={(value) => onChange('build_id', value)}>
          {!builds.length && <option value="">No builds</option>}
          {builds.map((build) => (
            <option key={build.build_id} value={build.build_id}>
              {build.build_id} · {build.status || 'unknown'}
            </option>
          ))}
        </SelectFilter>

        <div className="filter-field">
          <span>Mode</span>
          <div className="segment-control segment-three">
            {[
              ['', 'All'],
              ['local', 'Local'],
              ['global', 'Global'],
            ].map(([value, label]) => (
              <button
                type="button"
                key={label}
                className={cx(filters.mode === value && 'active')}
                aria-pressed={filters.mode === value}
                onClick={() => onChange('mode', value)}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        <div className="filter-grid">
          <SelectFilter label="Format" value={filters.format} onChange={(value) => onChange('format', value)}>
            <option value="">All</option>
            {(facets.formats || []).map((value) => <option key={value}>{value}</option>)}
          </SelectFilter>
          <SelectFilter label="Winner" value={filters.winner} onChange={(value) => onChange('winner', value)}>
            <option value="any">Any</option>
            <option value="yes">Selected</option>
            <option value="no">Not selected</option>
            <option value="top1">Top 1</option>
            <option value="top2">Top 2</option>
          </SelectFilter>
        </div>

        <SelectFilter label="Major" value={filters.major} onChange={(value) => onChange('major', value)}>
          <option value="">Any</option>
          {(facets.majors || []).map((value) => <option key={value}>{value}</option>)}
        </SelectFilter>
        <SelectFilter label="Minor" value={filters.minor} onChange={(value) => onChange('minor', value)}>
          <option value="">Any</option>
          {(facets.minors || []).map((value) => <option key={value}>{value}</option>)}
        </SelectFilter>

        <div className="filter-grid">
          <SelectFilter label="Queue" value={filters.queue} onChange={(value) => onChange('queue', value)}>
            <option value="">Any</option>
            {(facets.queue_states || []).map((value) => <option key={value}>{value}</option>)}
          </SelectFilter>
          <SelectFilter label="Failure" value={filters.failure} onChange={(value) => onChange('failure', value)}>
            <option value="">Any</option>
            {(facets.failure_states || []).map((value) => <option key={value}>{value}</option>)}
          </SelectFilter>
        </div>
      </div>
    </section>
  )
}

function SelectFilter({ label, value, onChange, children }) {
  return (
    <label className="filter-field">
      <span>{label}</span>
      <select aria-label={label} value={value} onChange={(event) => onChange(event.target.value)}>
        {children}
      </select>
    </label>
  )
}

function GroupRail({ result, page, onPage, selected, onSelect }) {
  if (result.err) return <ErrorBox error={result.err} />
  if (result.loading && !result.data) return <Spinner label="Loading index" />
  const payload = result.data || { total: 0, items: [] }
  const pages = Math.max(1, Math.ceil(payload.total / PAGE_SIZE))
  return (
    <section className="group-index" aria-label="Canonical groups">
      <div className="group-index-header">
        <div>
          <span className="eyebrow">Groups</span>
          <strong>{payload.total.toLocaleString()}</strong>
        </div>
        <div className="pagination">
          <span>{page}/{pages}</span>
          <IconButton label="Previous page" disabled={page <= 1} onClick={() => onPage(page - 1)}>
            <ChevronLeft size={16} aria-hidden="true" />
          </IconButton>
          <IconButton label="Next page" disabled={page >= pages} onClick={() => onPage(page + 1)}>
            <ChevronRight size={16} aria-hidden="true" />
          </IconButton>
        </div>
      </div>
      <div className="group-list">
        {payload.items.map((group) => (
          <button
            type="button"
            key={group.group_id}
            className={cx('group-row', selected === group.group_id && 'selected')}
            onClick={() => onSelect(group.group_id)}
          >
            <span className="group-thumb group-thumb-placeholder" aria-hidden="true">
              <ImageIcon size={18} />
            </span>
            <span className="group-row-body">
              <span className="group-row-top">
                <Badge tone={group.render_mode === 'local' ? 'accent' : 'neutral'}>{group.render_mode}</Badge>
                <QueueBadge value={group.queue_state} />
                {group.failure_state !== 'none' && <FailureBadge value={group.failure_state} />}
              </span>
              <strong>{shortId(group.group_id)}</strong>
              <span>{group.major || 'uncategorized'} · {group.winner_count} winners</span>
            </span>
          </button>
        ))}
        {!payload.items.length && <Empty title="No matching groups" />}
      </div>
    </section>
  )
}

function usePreparation(groupId, buildId, retryToken) {
  const [state, setState] = useState({
    data: null, initialLoading: true, refreshing: false, err: null, lost: false,
  })

  useEffect(() => {
    let live = true
    let timer = null
    let last = null
    let transientFailures = 0
    setState({ data: null, initialLoading: true, refreshing: false, err: null, lost: false })

    const schedule = (start, delay) => {
      timer = window.setTimeout(() => poll(start), delay)
    }
    const poll = async (start) => {
      if (!live) return
      if (!start && last) {
        setState({ data: last, initialLoading: false, refreshing: true, err: null, lost: false })
      }
      try {
        const snapshot = start
          ? await api.prepareGroup(groupId, buildId, retryToken > 0)
          : await api.preparation(groupId, buildId)
        if (!live) return
        last = snapshot
        transientFailures = 0
        setState({ data: snapshot, initialLoading: false, refreshing: false, err: null, lost: false })
        if (['queued', 'locating', 'materializing'].includes(snapshot.state)) {
          schedule(false, 200)
        }
      } catch (error) {
        if (!live) return
        const status = Number(error?.status || 0)
        const jobLost = !start && status === 404 && Boolean(last)
        const transient = status === 0 || status >= 500
        transientFailures += 1
        const exhausted = transientFailures >= 5
        setState({
          data: last,
          initialLoading: false,
          refreshing: false,
          err: error,
          lost: jobLost || exhausted || (!transient && !jobLost),
        })
        if (!jobLost && transient && !exhausted) {
          schedule(start && !last, Math.min(4000, 250 * (2 ** (transientFailures - 1))))
        }
      }
    }

    poll(true)
    return () => {
      live = false
      if (timer != null) window.clearTimeout(timer)
    }
  }, [groupId, buildId, retryToken])

  return state
}

function MaterializationProgress({ snapshot, error, lost, onRetry }) {
  const state = snapshot?.state || 'queued'
  const files = snapshot?.files || { done: 0, total: 0 }
  const bytes = snapshot?.bytes || { done: 0, total: 0 }
  const percent = bytes.total > 0 ? Math.min(100, Math.round((bytes.done / bytes.total) * 100)) : null
  const failed = state === 'failed' || lost || Boolean(error && !snapshot)
  const label = {
    queued: 'Queued',
    locating: 'Locating index entries',
    materializing: 'Materializing assets',
    ready: 'Assets ready',
    failed: 'Materialization failed',
  }[state] || state
  const displayLabel = lost ? 'Preparation job lost' : failed && state !== 'failed' ? 'Preparation unavailable' : label
  const current = snapshot?.current_item?.split('/').pop()

  return (
    <section className="materialization-panel" role={failed ? 'alert' : 'status'} aria-live="polite">
      <div className="materialization-heading">
        <HardDriveDownload size={18} aria-hidden="true" />
        <div>
          <span className="eyebrow">Indexed cache</span>
          <strong>{displayLabel}</strong>
        </div>
        <Badge tone={failed ? 'danger' : state === 'ready' ? 'success' : 'accent'}>{state}</Badge>
      </div>
      <progress value={percent ?? undefined} max="100" aria-label="Asset materialization progress" />
      <div className="materialization-meta">
        <span>{files.done}/{files.total} files</span>
        <span>{formatBytes(bytes.done)} / {formatBytes(bytes.total)}</span>
        {current && <span title={snapshot.current_item}>{current}</span>}
      </div>
      {(snapshot?.message || error) && (
        <p>{snapshot?.message || error?.message || String(error)}</p>
      )}
      {failed && (
        <button type="button" className="retry-button" onClick={onRetry}>
          <RefreshCw size={14} aria-hidden="true" /> Retry
        </button>
      )}
    </section>
  )
}

function GroupInspector({ groupId, buildId }) {
  const detail = useAsync(() => api.group(groupId, buildId), [groupId, buildId])
  const [candidateId, setCandidateId] = useState('')
  const [viewMode, setViewMode] = useState('rendered')
  const [retryToken, setRetryToken] = useState(0)
  const preparation = usePreparation(groupId, buildId, retryToken)

  useEffect(() => {
    setRetryToken(0)
  }, [groupId])

  useEffect(() => {
    if (!detail.data) return
    const candidates = detail.data.candidates || []
    const firstWinner = candidates.find((candidate) => candidate.winner)
    setCandidateId(firstWinner?.candidate_id || candidates[0]?.candidate_id || '')
    setViewMode('rendered')
  }, [detail.data])

  if (preparation.initialLoading && !preparation.data) return <Spinner label="Preparing indexed assets" />
  if (preparation.data?.state !== 'ready') {
    return (
      <MaterializationProgress
        snapshot={preparation.data}
        error={preparation.err}
        lost={preparation.lost}
        onRetry={() => setRetryToken((value) => value + 1)}
      />
    )
  }
  if (detail.loading && !detail.data) return <Spinner label="Loading group" />
  if (detail.err) return <ErrorBox error={detail.err} />
  if (!detail.data) return <Empty title="Group unavailable" />

  const { group, candidates = [], sft = [], failures = [] } = detail.data
  const winnerSft = new Map(sft.map((row) => [row.candidate_id, row]))
  const selected = candidates.find((candidate) => candidate.candidate_id === candidateId) || candidates[0]
  const historicalCgt = winnerSft.get(selected?.candidate_id)?.local?.C_GT
  const visualCandidate = selected && !selected.cgt_path && historicalCgt
    ? { ...selected, cgt_path: historicalCgt }
    : selected
  const localView = group.render_mode === 'local' && Boolean(visualCandidate?.cgt_path)
  const activeView = localView ? viewMode : 'rendered'
  const failedCandidates = new Set(
    failures.filter((row) => row.stage === 'annotation' && row.terminal).map((row) => row.candidate_id),
  )

  return (
    <article className="group-detail">
      <header className="detail-header">
        <div>
          <span className="eyebrow">{group.build_id}</span>
          <h2>{shortId(group.group_id, 24)}</h2>
        </div>
        <div className="detail-badges">
          <Badge tone={group.render_mode === 'local' ? 'accent' : 'neutral'}>{group.render_mode}</Badge>
          <QueueBadge value={group.queue_state} />
          <FailureBadge value={group.failure_state} />
          <Badge>{candidates.length} candidates</Badge>
          <Badge>{sft.length} SFT</Badge>
        </div>
      </header>

      <section className="candidate-section">
        <div className="section-heading">
          <div>
            <span className="eyebrow">Candidate set</span>
            <h3>Eight accepted renders</h3>
          </div>
          <span className="section-meta">{group.major || 'uncategorized'} · {group.scene || 'unknown scene'}</span>
        </div>
        <div className="candidate-strip" role="listbox" aria-label="Eight accepted candidates">
          {candidates.map((candidate, index) => (
            <CandidateTile
              key={candidate.candidate_id}
              candidate={candidate}
              index={index}
              selected={candidate.candidate_id === selected?.candidate_id}
              annotationState={
                winnerSft.has(candidate.candidate_id)
                  ? 'complete'
                  : failedCandidates.has(candidate.candidate_id)
                    ? 'failed'
                    : candidate.winner ? 'pending' : null
              }
              onClick={() => {
                setCandidateId(candidate.candidate_id)
                setViewMode('rendered')
              }}
            />
          ))}
        </div>
      </section>

      {selected ? (
        <>
          <section className="visual-section">
            <div className="section-heading">
              <div>
                <span className="eyebrow">Visual compare</span>
                <h3>Source and selected candidate</h3>
              </div>
              <div className="segment-control view-segments" aria-label="Local candidate view">
                {VIEW_MODES.map(({ id, label, icon: Icon }) => (
                  <button
                    type="button"
                    key={id}
                    className={cx(activeView === id && 'active')}
                    aria-pressed={activeView === id}
                    disabled={id !== 'rendered' && !localView}
                    onClick={() => setViewMode(id)}
                  >
                    <Icon size={15} aria-hidden="true" />
                    <span>{label}</span>
                  </button>
                ))}
              </div>
            </div>
            <div className="compare-grid">
              <Figure label="Before" path={group.source_path} />
              <CandidateFigure label="After" group={group} candidate={visualCandidate} mode={activeView} />
            </div>
          </section>

          <CandidateMetadata candidate={selected} group={group} />
        </>
      ) : (
        <Empty title="Candidate set is empty" />
      )}

      <SftSection rows={sft} onCandidate={setCandidateId} />
      <FailureSection rows={failures} />
    </article>
  )
}

function CandidateTile({ candidate, index, selected, annotationState, onClick }) {
  const qa = candidate.qa || {}
  return (
    <button
      type="button"
      role="option"
      aria-selected={selected}
      className={cx('candidate-tile', selected && 'selected', candidate.winner && 'winner')}
      onClick={onClick}
    >
      <span className="candidate-tile-head">
        <strong>{String(index + 1).padStart(2, '0')}</strong>
        <span>{candidate.slot_mode || 'global'}</span>
        {candidate.winner && <Badge tone="success">top {candidate.rank}</Badge>}
      </span>
      <AssetImage path={candidate.after_path} width={360} alt={`Candidate ${index + 1}`} className="candidate-image" />
      <span className="candidate-tile-meta">
        <strong>{candidate.format || 'unknown'}</strong>
        <span>q {formatValue(qa.q)}</span>
      </span>
      <span className="candidate-preset">{candidate.preset_id || 'missing preset'}</span>
      {annotationState && <span className={cx('annotation-state', `annotation-${annotationState}`)}>{annotationState}</span>}
    </button>
  )
}

function Figure({ label, path }) {
  return (
    <figure className="image-figure">
      <figcaption>{label}</figcaption>
      <div className="image-stage">
        <AssetImage path={path} width={1280} alt={label} className="stage-image" link />
      </div>
    </figure>
  )
}

function CandidateFigure({ label, group, candidate, mode }) {
  let content
  if (mode === 'mask') {
    content = <AssetImage path={candidate.cgt_path} width={1280} alt="C_GT mask" className="stage-image mask-image" link />
  } else if (mode === 'overlay') {
    const maskUrl = img(candidate.cgt_path, 1280)
    content = (
      <div className="overlay-stage">
        <AssetImage path={group.source_path} width={1280} alt="Source with mask overlay" className="stage-image" />
        <span
          className="mask-tint"
          style={{ WebkitMaskImage: `url("${maskUrl}")`, maskImage: `url("${maskUrl}")` }}
          aria-hidden="true"
        />
      </div>
    )
  } else {
    content = <AssetImage path={candidate.after_path} width={1280} alt="Rendered candidate" className="stage-image" link />
  }
  return (
    <figure className="image-figure">
      <figcaption>{label} · {viewLabel(mode)}</figcaption>
      <div className="image-stage">{content}</div>
    </figure>
  )
}

function CandidateMetadata({ candidate, group }) {
  const qa = candidate.qa || {}
  const visibility = candidate.visibility || {}
  const subject = candidate.subject || group.subject
  const subjectName = typeof subject === 'string'
    ? subject
    : subject?.name || subject?.subject_name || subject?.label
  return (
    <section className="metadata-section">
      <div className="section-heading">
        <div>
          <span className="eyebrow">Selected candidate</span>
          <h3>{candidate.preset_id || candidate.candidate_id}</h3>
        </div>
        <IconButton
          label="Copy candidate ID"
          onClick={() => navigator.clipboard?.writeText(candidate.candidate_id || '')}
        >
          <Clipboard size={16} aria-hidden="true" />
        </IconButton>
      </div>
      <div className="metadata-layout">
        <div className="metric-grid">
          <Metric label="Visible dE" value={visibility.visible_de} accent />
          <Metric label="Visible fraction" value={visibility.visible_fraction} />
          <Metric label="OneAlign" value={qa.onealign} accent />
          <Metric label="Quality q" value={qa.q} />
          <Metric label="Improvement" value={qa.improvement} />
          <Metric label="Rank" value={candidate.rank} />
        </div>
        <div className="metadata-fields">
          <Field label="Preset ID" mono>{candidate.preset_id}</Field>
          <Field label="Format">{candidate.format}</Field>
          <Field label="Taxonomy">{candidate.major || 'n/a'} / {candidate.minor || 'n/a'}</Field>
          <Field label="Slot">{candidate.slot_id} · {candidate.slot_mode || 'global'} · pairing {formatValue(candidate.pairing_index)}</Field>
          <Field label="Subject / region">{subjectName || 'n/a'} / {candidate.region || 'n/a'}</Field>
          <Field label="Mask" mono>{candidate.mask_id || 'global'}{candidate.amount != null ? ` · amount ${formatValue(candidate.amount)}` : ''}</Field>
          <Field label="Reliability">{formatValue(qa.reliable)} · veto {formatValue(qa.veto)}</Field>
          <Field label="Render engine" mono>{candidate.render_engine}</Field>
        </div>
      </div>
      <details className="diagnostic-details">
        <summary>Objective hints and diagnostics</summary>
        <div className="diagnostic-grid">
          <JsonBlock data={candidate.objective_hints} />
          <JsonBlock data={{ qa, render_diagnostics: candidate.render_diagnostics, attempt_lineage: candidate.attempt_lineage }} />
        </div>
      </details>
    </section>
  )
}

function SftSection({ rows, onCandidate }) {
  const ordered = useMemo(
    () => [...rows].sort((left, right) => (left.winner_rank || 0) - (right.winner_rank || 0)),
    [rows],
  )
  return (
    <section className="sft-section">
      <div className="section-heading">
        <div>
          <span className="eyebrow">Training output</span>
          <h3>Top-2 SFT</h3>
        </div>
        <span className="section-meta">{ordered.length}/2 annotated</span>
      </div>
      {ordered.length ? (
        <div className="sft-grid">
          {ordered.map((row) => (
            <article className="sft-card" key={row.sft_id}>
              <header>
                <div>
                  <Badge tone="success"><Check size={12} aria-hidden="true" /> top {row.winner_rank}</Badge>
                  <span>{row.task_type}</span>
                </div>
                <button type="button" onClick={() => onCandidate(row.candidate_id)}>{shortId(row.candidate_id)}</button>
              </header>
              <AssetImage path={row.I_tar || row.target_path} width={640} alt={`SFT rank ${row.winner_rank}`} className="sft-image" />
              <Field label="Instruction short">{row.instruction_short}</Field>
              <Field label="Instruction">{row.instruction}</Field>
              <Field label="Reasoning"><pre className="reasoning-text">{row.reasoning}</pre></Field>
              <div className="sft-provenance">
                <span>{row.annot_src}</span>
                <span>{row.qa?.annotation?.returned_model || 'model not recorded'}</span>
              </div>
              <details className="diagnostic-details">
                <summary>Recipe and annotation QA</summary>
                <JsonBlock data={{ recipe: row.recipe, local: row.local, qa: row.qa }} />
              </details>
            </article>
          ))}
        </div>
      ) : (
        <Empty title="No completed SFT annotations" />
      )}
    </section>
  )
}

function FailureSection({ rows }) {
  if (!rows.length) return null
  return (
    <section className="failure-section">
      <div className="section-heading">
        <div>
          <span className="eyebrow">Audit trail</span>
          <h3>Failure events</h3>
        </div>
        <Badge tone="danger">{rows.length}</Badge>
      </div>
      <div className="failure-list">
        {rows.map((row, index) => (
          <article key={row.event_id || `${row.error_code}-${index}`}>
            <AlertTriangle size={16} aria-hidden="true" />
            <div>
              <strong>{row.error_code || 'unknown failure'}</strong>
              <p>{row.message || 'No message recorded'}</p>
              <span>{row.stage || 'unknown stage'} · {row.terminal ? 'terminal' : row.retryable ? 'retryable' : 'recorded'}</span>
            </div>
          </article>
        ))}
      </div>
    </section>
  )
}

function QueueBadge({ value }) {
  const tones = { complete: 'success', pending: 'warning', failed: 'danger', none: 'neutral' }
  return <Badge tone={tones[value] || 'neutral'}>{value || 'unknown queue'}</Badge>
}

function FailureBadge({ value }) {
  if (!value || value === 'none') return <Badge>no failures</Badge>
  const tones = { terminal: 'danger', retryable: 'warning', recorded: 'neutral' }
  return <Badge tone={tones[value] || 'neutral'}>{value}</Badge>
}

const formatBytes = (value) => {
  const bytes = Number(value || 0)
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KiB`
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MiB`
  return `${(bytes / 1024 ** 3).toFixed(1)} GiB`
}

const shortId = (value, length = 16) => {
  const text = String(value || 'unknown')
  return text.length > length ? `${text.slice(0, length)}...` : text
}

const viewLabel = (mode) => ({ rendered: 'Rendered', mask: 'C_GT', overlay: 'Mask overlay' }[mode] || mode)
