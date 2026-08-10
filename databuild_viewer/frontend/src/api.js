export class ApiError extends Error {
  constructor(status, message) {
    super(`${status || 'network'} ${message}`)
    this.name = 'ApiError'
    this.status = status
  }
}

const requestJson = async (url, options) => {
  let response
  try {
    response = await fetch(url, options)
  } catch (error) {
    throw new ApiError(0, error instanceof Error ? error.message : 'request failed')
  }
  if (!response.ok) {
    let message = response.statusText
    try {
      const payload = await response.json()
      message = payload.detail || message
    } catch {
      // Keep the HTTP status text when the backend did not return JSON.
    }
    throw new ApiError(response.status, message)
  }
  return response.json()
}

const query = (values) => {
  const params = new URLSearchParams()
  Object.entries(values).forEach(([key, value]) => {
    if (value != null && value !== '' && value !== 'any') params.set(key, value)
  })
  return params.toString()
}

export const api = {
  health: () => requestJson('/api/health'),
  builds: () => requestJson('/api/builds'),
  facets: (buildId) => requestJson(`/api/facets?${query({ build_id: buildId })}`),
  groups: (filters, page = 1, pageSize = 50) => requestJson(
    `/api/groups?${query({ ...filters, page, page_size: pageSize })}`,
  ),
  group: (id, buildId) => requestJson(
    `/api/groups/${encodeURIComponent(id)}?${query({ build_id: buildId })}`,
  ),
  prepareGroup: (id, buildId, retry = false) => requestJson(
    `/api/groups/${encodeURIComponent(id)}/prepare?${query({ build_id: buildId, retry: retry ? 'true' : '' })}`,
    { method: 'POST' },
  ),
  preparation: (id, buildId) => requestJson(
    `/api/groups/${encodeURIComponent(id)}/prepare?${query({ build_id: buildId })}`,
  ),
}

export const img = (path, width = 768) => (
  path ? `/img?${query({ w: width, path })}` : ''
)

export const full = (path) => (
  path ? `/img?${query({ full: 'true', path })}` : ''
)
