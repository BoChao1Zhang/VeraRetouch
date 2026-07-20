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

async function installApi(page, state = 'normal') {
  await page.route('**/img?**', async (route) => {
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
    if (url.pathname === '/api/health') {
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
        : state === 'switch'
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
