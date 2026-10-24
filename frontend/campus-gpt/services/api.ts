export type ChatSource = {
  title: string
  detail: string
}

export type AskMetadata = {
  detected_department?: string | null
  document_type?: string | null
  detected_batch?: string[] | null
  detected_semester?: number | string | null
}

export type ChatResponse = AskMetadata & {
  answer: string
  sources?: ChatSource[]
}

type BackendAskResponse = {
  answer: string
  sources?: { source: string; document_type: string }[] | null
  query_classification?: string
  document_type?: string | null
  detected_department?: string | null
  detected_batch?: string[] | null
  detected_semester?: number | string | null
  candidate_k?: number
  top_k?: number
  latency?: Record<string, number>
  intent_plans?: unknown[]
}

// Keep the backend origin in one place; NEXT_PUBLIC_API_URL can override it.
const API_URL = (process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000').replace(/\/+$/, '')

/** Send a question to the CampusGPT backend. Department and intent are automatic. */
export async function askCampusGPT(question: string): Promise<ChatResponse> {
  const response = await fetch(`${API_URL}/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  })

  if (!response.ok) {
    throw new Error(`CampusGPT API request failed (${response.status})`)
  }

  const data: BackendAskResponse = await response.json()
  return {
    answer: data.answer,
    sources: data.sources?.map((source) => ({
      title: source.source,
      detail: source.document_type,
    })),
    detected_department: data.detected_department,
    document_type: data.document_type,
    detected_batch: data.detected_batch,
    detected_semester: data.detected_semester,
  }
}
