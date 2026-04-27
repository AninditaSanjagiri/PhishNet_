import axios from 'axios'

const BASE = import.meta.env.VITE_API_BASE || '/api'

const client = axios.create({
  baseURL: BASE,
  timeout: 30_000,   // 30s — tight enough to fail fast, generous enough for BERT
})

function handleErr(err) {
  throw new Error(err.response?.data?.detail || err.message || 'Request failed')
}

export async function analyzeUrl(url) {
  try {
    const { data } = await client.post('/analyze', { url })
    return data
  } catch (err) { handleErr(err) }
}

export async function getHistory(limit = 30) {
  try {
    const { data } = await client.get('/history', { params: { limit } })
    return data
  } catch (err) { handleErr(err) }
}

export async function clearHistory() {
  try {
    const { data } = await client.delete('/history')
    return data
  } catch (err) { handleErr(err) }
}

export async function getEvalSummary() {
  try {
    const { data } = await client.get('/evaluation/summary')
    return data
  } catch (err) { handleErr(err) }
}

export async function checkHealth() {
  try {
    const { data } = await client.get('/health')
    return data.status === 'ok'
  } catch { return false }
}
