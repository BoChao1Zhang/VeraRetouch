const requestJson = async (url) => {
  const response = await fetch(url)
  if (!response.ok) {
    let message = response.statusText
    try {
      const payload = await response.json()
      message = payload.detail || message
    } catch {
      // Keep the HTTP status text when the backend did not return JSON.
    }
    throw new Error(`${response.status} ${message}`)
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
  group: (id) => requestJson(`/api/groups/${encodeURIComponent(id)}`),
}

export const img = (path, width = 768) => (
  path ? `/img?${query({ w: width, path })}` : ''
)

export const full = (path) => (
  path ? `/img?${query({ full: 'true', path })}` : ''
)
