import { expect, test } from '@playwright/test'
import fs from 'node:fs'

const SOURCE_IMAGE = '/home/bc/data/datasets/ppr10k/source/ppr10k_000017.png'
const MASK_IMAGE = '/home/bc/data/datasets/vera_directionA_1M/subject_cache/2afbd343514d2e0a/subject.png'
const FALLBACK_PNG = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=',
  'base64',
)

const candidates = Array.from({ length: 8 }, (_, index) => ({
  candidate_id: `candidate-${index + 1}`,
  slot_id: `slot-${index + 1}`,
  slot_index: index,
  slot_mode: ['radial', 'radial', 'semantic', 'semantic', 'band', 'band', 'linear', 'linear'][index],
  mode_index: index % 2,
  pairing_index: index,
  preset_id: `portrait-cinematic-preset-with-a-long-stable-id-${index + 1}`,
  format: index < 3 ? 'xmp' : index < 6 ? 'lrtemplate' : 'lut',
  major: index < 4 ? 'Portrait Color' : 'Cinematic Film',
  minor: index % 2 ? 'Warm editorial highlight rolloff' : 'Controlled cyan shadow separation',
  after_path: `/fixture/candidate-${index + 1}.jpg`,
  render_engine: 'canonical_cuda',
  visibility: { visible_de: 3.4 + index * 0.33, visible_fraction: 0.62 + index * 0.02, accepted: true },
  objective_hints: { brightness: 0.14, warmth: -0.08, chroma: 0.21, contrast: 0.18 },
  qa: {
    onealign: 0.71 + index * 0.01,
    source_onealign: 0.55,
    q: 0.73 + index * 0.02,
    improvement: 0.17 + index * 0.01,
    reliable: true,
    veto: false,
    veto_flags: [],
    qa_mode: 'onealign',
  },
  rank: index + 1,
  winner: index < 2,
  mask_id: index === 2 || index === 3 ? 'shared-semantic-mask' : `mask-${index + 1}`,
  cgt_path: '/fixture/cgt.png',
  subject: { name: 'woman holding white flowers' },
  region: 'center-left',
  raw_alpha_mean: 0.56,
  amount: 0.89,
  effective_alpha_mean: 0.50,
  render_diagnostics: { backend: 'cuda:1', pre_jpeg_shape: [1024, 1536, 3] },
  attempt_lineage: { group_attempt: 1, preset_attempt: 1 },
}))

const group = {
  schema_version: 1,
  build_id: 'canonical-2026-07-20-long-build-id',
  group_id: 'group-canonical-local-00000001',
  source_id: 'ppr10k_0017_a',
  source_path: '/fixture/source.jpg',
  scene: 'studio portrait',
  subject: { name: 'woman holding white flowers' },
  render_mode: 'local',
  preset_filter: 'all',
  major: 'Portrait Color',
  winner_ids: ['candidate-1', 'candidate-2'],
  winner_ranks: [1, 2],
  queue_state: 'complete',
  failure_state: 'recorded',
}

const reasoning = [
  '<problem_lighting>The subject is evenly exposed, but the face lacks directional separation from the background.</problem_lighting>',
  '<plan_lighting>Lift the facial midtones while preserving the white flowers and controlling highlight rolloff.</plan_lighting>',
  '<problem_global_color>The source is neutral and visually flat across skin, foliage, and the studio wall.</problem_global_color>',
  '<plan_global_color>Introduce restrained warmth in skin and a cooler cyan relationship in the surrounding shadows.</plan_global_color>',
  '<problem_specific_color>The green stems compete with the face and the flower whites drift slightly yellow.</problem_specific_color>',
  '<plan_specific_color>Reduce green saturation locally and return the flower highlights toward a clean neutral white.</plan_specific_color>',
].join('\n')

const sft = [1, 2].map((rank) => ({
  build_id: group.build_id,
  sft_id: `sft-row-${rank}`,
  annotation_task_id: `annotation-task-${rank}`,
  group_id: group.group_id,
  candidate_id: `candidate-${rank}`,
  winner_rank: rank,
  I_in: group.source_path,
  I_tar: `/fixture/candidate-${rank}.jpg`,
  recipe: { preset_id: candidates[rank - 1].preset_id, render_mode: 'local' },
  local: { C_GT: '/fixture/cgt.png', subject: 'woman', region: 'center-left' },
  task_type: 'local',
  instruction_short: rank === 1 ? 'Refine the woman with warm facial light.' : 'Separate the subject with controlled color contrast.',
  instruction: `${rank === 1 ? 'Warm' : 'Balance'} the woman holding white flowers while preserving natural skin texture, recover the delicate highlight structure in every petal, keep the studio background subdued, and maintain a believable transition across the softly feathered local region.`,
  reasoning,
  annot_src: rank === 1 ? 'responses:external:relay-a' : 'responses:local',
  qa: { annotation: { status: 'completed', returned_model: rank === 1 ? 'external-model' : 'qwen3_5-35b-a3b' } },
}))

const detail = {
  group,
  candidates,
  sft,
  failures: [{
    event_id: 'render-attempt-replaced-1',
    event_type: 'candidate_attempt_failed',
    stage: 'rendering',
    group_id: group.group_id,
    candidate_id: 'discarded-candidate',
    retryable: true,
    terminal: false,
    error_code: 'visibility_below_threshold',
    message: 'The discarded preset did not meet the weighted visible difference threshold and was replaced.',
  }],
}

const summary = {
  group_id: group.group_id,
  build_id: group.build_id,
  source_id: group.source_id,
  source_path: group.source_path,
  scene: group.scene,
  render_mode: group.render_mode,
  preset_filter: group.preset_filter,
  major: group.major,
  winner_ids: group.winner_ids,
  winner_count: 2,
  candidate_count: 8,
  sft_count: 2,
  queue_state: 'complete',
  failure_state: 'recorded',
  failure_count: 1,
  candidates: candidates.map(({ candidate_id, slot_index, after_path, format, winner, rank }) => (
    { candidate_id, slot_index, after_path, format, winner, rank }
  )),
}

const secondGroup = {
  ...group,
  group_id: 'second-group',
  source_id: 'second-source',
  winner_ids: ['second-candidate-1', 'second-candidate-2'],
  queue_state: 'pending',
  failure_state: 'none',
}
const secondCandidates = candidates.map((candidate, index) => ({
  ...candidate,
  candidate_id: `second-candidate-${index + 1}`,
  winner: index < 2,
}))
const secondDetail = {
  group: secondGroup,
  candidates: secondCandidates,
  sft: [],
  failures: [],
}
const secondSummary = {
  ...summary,
  group_id: secondGroup.group_id,
  source_id: secondGroup.source_id,
  winner_ids: secondGroup.winner_ids,
  queue_state: 'pending',
  failure_state: 'none',
  sft_count: 0,
  failure_count: 0,
  candidates: secondCandidates.map(({ candidate_id, slot_index, after_path, format, winner, rank }) => (
    { candidate_id, slot_index, after_path, format, winner, rank }
  )),
}

const preparation = (groupId, state, done = 0, total = 10, message = null) => ({
  schema_version: 1,
  group_id: groupId,
  state,
  files: { done, total },
  bytes: { done: done * 1024 * 1024, total: total * 1024 * 1024 },
  current_item: state === 'materializing' ? `/retired/archive/member-${done}.jpg` : null,
  message,
  updated_at: new Date().toISOString(),
})

async function installApi(page, state = 'normal') {
  const preparationPolls = new Map()
  const counters = { imageRequests: 0, prepareGets: 0, preparePosts: 0 }
  await page.route('**/img?**', async (route) => {
    counters.imageRequests += 1
    const path = new URL(route.request().url()).searchParams.get('path') || ''
    const filename = path.includes('cgt') ? MASK_IMAGE : SOURCE_IMAGE
    const body = fs.existsSync(filename) ? fs.readFileSync(filename) : FALLBACK_PNG
    await route.fulfill({ status: 200, contentType: 'image/png', body })
  })
  await page.route('**/api/**', async (route) => {
    const url = new URL(route.request().url())
    if (state === 'error' && url.pathname === '/api/groups') {
      await route.fulfill({ status: 503, contentType: 'application/json', body: JSON.stringify({ detail: 'projection unavailable' }) })
      return
    }
    if (state === 'facets-error' && url.pathname === '/api/facets') {
      await route.fulfill({ status: 503, contentType: 'application/json', body: JSON.stringify({ detail: 'facet index unavailable' }) })
      return
    }
    let body
    if (
      url.pathname === '/api/facets' || url.pathname === '/api/groups'
      || /^\/api\/groups\/[^/]+(?:\/prepare)?$/.test(url.pathname)
    ) {
      expect(url.searchParams.get('build_id')).toBe(group.build_id)
    }
    const prepareMatch = url.pathname.match(/^\/api\/groups\/([^/]+)\/prepare$/)
    if (prepareMatch) {
      const groupId = decodeURIComponent(prepareMatch[1])
      const poll = preparationPolls.get(groupId) || 0
      if (route.request().method() === 'POST') {
        counters.preparePosts += 1
        if (state === 'prepare-failure') {
          body = preparation(groupId, 'failed', 1, 10, 'OSError: fixture shard unavailable')
        } else if (state === 'job-lost' && url.searchParams.get('retry') === 'true') {
          body = preparation(groupId, 'ready', 10, 10)
        } else if (
          state === 'progress' || state === 'progress-hold' || state === 'progress-switch'
          || state === 'job-lost' || state === 'transient-503'
        ) {
          body = preparation(groupId, 'queued', 0, 10)
        } else {
          body = preparation(groupId, 'ready', 10, 10)
        }
      } else {
        counters.prepareGets += 1
        if (state === 'job-lost') {
          if (poll === 0) {
            preparationPolls.set(groupId, 1)
            body = preparation(groupId, 'materializing', 3, 10)
          } else {
            await route.fulfill({ status: 404, contentType: 'application/json', body: JSON.stringify({ detail: 'asset preparation not started' }) })
            return
          }
        } else if (state === 'transient-503' && poll === 0) {
          preparationPolls.set(groupId, 1)
          await route.fulfill({ status: 503, contentType: 'application/json', body: JSON.stringify({ detail: 'temporary catalog stall' }) })
          return
        } else if (state === 'transient-503') {
          await new Promise((resolve) => setTimeout(resolve, 500))
          body = preparation(groupId, 'ready', 10, 10)
        } else if (state === 'progress-hold') {
          body = preparation(groupId, 'materializing', 4, 10)
        } else if (state === 'progress-switch' && groupId === group.group_id) {
        await new Promise((resolve) => setTimeout(resolve, 450))
        body = preparation(groupId, 'failed', 3, 10, 'late old-group failure')
        } else if (state === 'progress-switch') {
        body = preparation(groupId, 'ready', 10, 10)
        } else if (state === 'progress') {
        const states = [
          preparation(groupId, 'materializing', 2, 10),
          preparation(groupId, 'materializing', 7, 10),
          preparation(groupId, 'ready', 10, 10),
        ]
        body = states[Math.min(poll, states.length - 1)]
        preparationPolls.set(groupId, poll + 1)
        } else {
          body = preparation(groupId, 'ready', 10, 10)
        }
      }
    } else if (url.pathname === '/api/health') {
      body = { ok: true, source: 'postgres', builds: 1, malformed_records: 0 }
    } else if (url.pathname === '/api/builds') {
      body = [{ build_id: group.build_id, phase: 'complete', status: 'complete_with_failures', completed: { groups: 1, sft: 2 } }]
    } else if (url.pathname === '/api/facets') {
      body = {
        builds: [group.build_id],
        modes: ['global', 'local'],
        formats: ['lrtemplate', 'lut', 'xmp'],
        majors: ['Cinematic Film', 'Portrait Color'],
        minors: ['Controlled cyan shadow separation', 'Warm editorial highlight rolloff'],
        queue_states: ['complete', 'pending', 'failed', 'none'],
        failure_states: ['terminal', 'retryable', 'recorded', 'none'],
        winner_filters: ['any', 'yes', 'no', 'top1', 'top2'],
      }
    } else if (url.pathname === '/api/groups') {
      body = state === 'empty'
        ? { total: 0, page: 1, page_size: 50, items: [] }
        : state === 'switch' || state === 'progress-switch'
          ? { total: 2, page: 1, page_size: 50, items: [summary, secondSummary] }
          : { total: 1, page: 1, page_size: 50, items: [summary] }
    } else if (url.pathname === `/api/groups/${group.group_id}`) {
      body = state === 'historical-mask'
        ? { ...detail, candidates: candidates.map(({ cgt_path: _cgtPath, ...candidate }) => candidate) }
        : detail
    } else if (url.pathname === `/api/groups/${secondGroup.group_id}`) {
      await new Promise((resolve) => setTimeout(resolve, 300))
      body = secondDetail
    } else {
      await route.continue()
      return
    }
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(body) })
  })
  return counters
}

test('desktop candidate strip, filters, overlay, and long text', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 960 })
  await installApi(page)
  await page.goto('/')

  await expect(page.locator('.candidate-strip [role="option"]')).toHaveCount(8)
  await expect(page.getByText('Top-2 SFT')).toBeVisible()
  await expect(page.locator('.sft-card')).toHaveCount(2)
  await page.getByRole('button', { name: 'Overlay' }).click()
  await expect(page.locator('.mask-tint')).toBeVisible()

  const modeRequest = page.waitForRequest((request) => request.url().includes('/api/groups?') && request.url().includes('mode=local'))
  await page.getByRole('button', { name: 'Local', exact: true }).click()
  await modeRequest
  await page.getByRole('combobox', { name: 'Format', exact: true }).selectOption('xmp')
  await page.getByRole('combobox', { name: 'Major', exact: true }).selectOption('Portrait Color')
  await page.getByRole('combobox', { name: 'Minor', exact: true }).selectOption('Warm editorial highlight rolloff')
  await page.getByRole('combobox', { name: 'Queue', exact: true }).selectOption('complete')
  await page.getByRole('combobox', { name: 'Failure', exact: true }).selectOption('recorded')
  await page.getByRole('combobox', { name: 'Winner', exact: true }).selectOption('top1')
  await page.getByRole('button', { name: 'Overlay' }).click()
  await expect(page.locator('.mask-tint')).toBeVisible()

  const layout = await page.evaluate(() => ({
    viewport: window.innerWidth,
    documentWidth: document.documentElement.scrollWidth,
    reasoningOverflow: [...document.querySelectorAll('.reasoning-text')].some((node) => node.scrollWidth > node.clientWidth + 1),
    stripOverflow: document.querySelector('.candidate-strip').scrollWidth > document.querySelector('.candidate-strip').clientWidth,
  }))
  expect(layout.documentWidth).toBeLessThanOrEqual(layout.viewport)
  expect(layout.reasoningOverflow).toBe(false)
  expect(layout.stripOverflow).toBe(true)
  await page.screenshot({ path: '/tmp/databuild-viewer-desktop-overlay.png', fullPage: true })
})

test('mobile candidate strip and C_GT remain contained', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await installApi(page)
  await page.goto('/')
  await expect(page.locator('.candidate-strip [role="option"]')).toHaveCount(8)
  await page.getByRole('button', { name: 'C_GT' }).click()
  await expect(page.getByAltText('C_GT mask')).toBeVisible()

  const layout = await page.evaluate(() => ({
    viewport: window.innerWidth,
    documentWidth: document.documentElement.scrollWidth,
    stripWidth: document.querySelector('.candidate-strip').getBoundingClientRect().width,
    mainWidth: document.querySelector('.inspector-main').getBoundingClientRect().width,
  }))
  expect(layout.documentWidth).toBeLessThanOrEqual(layout.viewport)
  expect(layout.stripWidth).toBeLessThanOrEqual(layout.mainWidth)
  await page.screenshot({ path: '/tmp/databuild-viewer-mobile-mask.png', fullPage: true })
})

test('empty state is stable', async ({ page }) => {
  await page.setViewportSize({ width: 1024, height: 768 })
  await installApi(page, 'empty')
  await page.goto('/')
  await expect(page.getByRole('main').getByText('No matching groups')).toBeVisible()
  await page.screenshot({ path: '/tmp/databuild-viewer-empty.png', fullPage: true })
})

test('backend error state is stable', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await installApi(page, 'error')
  await page.goto('/')
  await expect(page.getByRole('main').getByRole('alert')).toContainText('projection unavailable')
  await page.screenshot({ path: '/tmp/databuild-viewer-error-mobile.png', fullPage: true })
})

test('facet errors are visible without hiding available groups', async ({ page }) => {
  await installApi(page, 'facets-error')
  await page.goto('/')
  await expect(page.locator('.inspector-rail').getByRole('alert')).toContainText('facet index unavailable')
  await expect(page.locator('.candidate-strip [role="option"]')).toHaveCount(8)
})

test('switching groups clears stale detail while the next request loads', async ({ page }) => {
  await installApi(page, 'switch')
  await page.goto('/')
  await expect(page.locator('.candidate-strip [role="option"]')).toHaveCount(8)
  await page.locator('.group-row').nth(1).click()
  await expect(page.getByText('Loading group')).toBeVisible()
  await expect(page.locator('.group-detail')).toHaveCount(0)
  await expect(page.getByRole('heading', { name: 'second-group' })).toBeVisible()
})

test('historical SFT C_GT enables local mask inspection', async ({ page }) => {
  await installApi(page, 'historical-mask')
  await page.goto('/')
  await page.getByRole('button', { name: 'C_GT' }).click()
  await expect(page.getByAltText('C_GT mask')).toBeVisible()
})

test('indexed materialization progress increases and gates asset detail', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 })
  await installApi(page, 'progress')
  await page.goto('/')
  await expect(page.getByText('Materializing assets')).toBeVisible()
  await expect(page.getByText('2/10 files')).toBeVisible()
  await expect(page.locator('.candidate-strip')).toHaveCount(0)
  await expect(page.getByText('7/10 files')).toBeVisible()
  await expect(page.locator('.candidate-strip [role="option"]')).toHaveCount(8)
  await page.screenshot({ path: '/tmp/databuild-viewer-desktop-materialized.png', fullPage: true })
})

test('materialization failure is visible, retryable, and leaves navigation usable', async ({ page }) => {
  await installApi(page, 'prepare-failure')
  await page.goto('/')
  await expect(page.getByRole('alert')).toContainText('fixture shard unavailable')
  await expect(page.getByRole('button', { name: 'Retry' })).toBeVisible()
  await expect(page.locator('.group-row')).toHaveCount(1)
})

test('late progress from a previous group cannot replace the selected group', async ({ page }) => {
  await installApi(page, 'progress-switch')
  await page.goto('/')
  await expect(page.getByText('Queued', { exact: true })).toBeVisible()
  await page.locator('.group-row').nth(1).click()
  await expect(page.getByRole('heading', { name: 'second-group' })).toBeVisible()
  await page.waitForTimeout(550)
  await expect(page.getByRole('heading', { name: 'second-group' })).toBeVisible()
  await expect(page.getByText('late old-group failure')).toHaveCount(0)
})

test('archived image requests stay at zero before selected group is ready', async ({ page }) => {
  const counters = await installApi(page, 'progress-hold')
  await page.goto('/')
  await expect(page.getByText('4/10 files')).toBeVisible()
  await page.waitForTimeout(350)
  expect(counters.imageRequests).toBe(0)
})

test('lost backend job stops polling and explicit retry starts a new prepare', async ({ page }) => {
  const counters = await installApi(page, 'job-lost')
  await page.goto('/')
  await expect(page.getByText('3/10 files')).toBeVisible()
  await expect(page.getByText('Preparation job lost')).toBeVisible()
  const getsAfterLoss = counters.prepareGets
  await page.waitForTimeout(600)
  expect(counters.prepareGets).toBe(getsAfterLoss)
  const postsBeforeRetry = counters.preparePosts
  await page.getByRole('button', { name: 'Retry' }).click()
  await expect(page.locator('.candidate-strip [role="option"]')).toHaveCount(8)
  expect(counters.preparePosts).toBe(postsBeforeRetry + 1)
})

test('temporary 503 retains progress and recovers with bounded backoff', async ({ page }) => {
  const counters = await installApi(page, 'transient-503')
  await page.goto('/')
  await expect(page.getByText('temporary catalog stall')).toBeVisible()
  await expect(page.locator('.candidate-strip [role="option"]')).toHaveCount(8)
  expect(counters.prepareGets).toBe(2)
})

test('unmount clears materialization polling timer', async ({ page }) => {
  const counters = await installApi(page, 'progress-hold')
  await page.goto('/')
  await expect(page.getByText('4/10 files')).toBeVisible()
  await page.goto('about:blank')
  const stoppedAt = counters.prepareGets
  await page.waitForTimeout(500)
  expect(counters.prepareGets).toBe(stoppedAt)
})

test('mobile materialization progress remains contained', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await installApi(page, 'progress-hold')
  await page.goto('/')
  await expect(page.getByText('4/10 files')).toBeVisible()
  const layout = await page.evaluate(() => ({
    viewport: window.innerWidth,
    documentWidth: document.documentElement.scrollWidth,
    progressWidth: document.querySelector('.materialization-panel').getBoundingClientRect().width,
    mainWidth: document.querySelector('.inspector-main').getBoundingClientRect().width,
  }))
  expect(layout.documentWidth).toBeLessThanOrEqual(layout.viewport)
  expect(layout.progressWidth).toBeLessThanOrEqual(layout.mainWidth)
  await page.screenshot({ path: '/tmp/databuild-viewer-mobile-materializing.png', fullPage: true })
})
