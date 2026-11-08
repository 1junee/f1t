// 이름의 baseline은 기존 명칭이며, 현재는 백엔드의 전체 검색 파이프라인을 호출한다.
const baselineApiBaseUrl = normalizeLocalApiBaseUrl(
  import.meta.env.VITE_API_BASE_URL as string | undefined,
)
const TEXT_SEARCH_TIMEOUT_MS = 75_000
// 아래 제한은 요청 한 번의 대기 시간이다. 사전 준비와 재시도가 더해지면 전체 시간은 길어진다.
const IMAGE_SEARCH_TIMEOUT_MS = 180_000
const API_WARMUP_TIMEOUT_MS = 20_000
const MODEL_INFO_TIMEOUT_MS = 10_000
const RETRYABLE_SEARCH_STATUS_CODES = new Set([502, 503, 504])

let baselineWarmupPromise: Promise<void> | null = null

export type BaselineSearchResult = {
  rank: number
  id: string
  filename?: string
  similarity: number
  image_url?: string
  source_table?: string
  name?: string | null
  brand?: string | null
  category1?: string | null
  category2?: string | null
  price?: number | string | null
  color?: string | null
  sleeve?: string | null
  length?: string | null
  sex?: string | string[] | null
  season?: string | string[] | null
  stretch?: string | string[] | null
  thickness?: string | string[] | null
  fit?: string | string[] | null
  fabric?: string | null
}

export type BaselineReasoningWarning = {
  code: string
  label: string
  message: string
}

export type BaselineReasoningMetadata = {
  provider: 'gemini' | 'qwen3_5_0_8b' | 'qwen3_vl_2b' | 'raw_query' | string
  status: 'ok' | 'fallback' | 'degraded' | string  // fallback = cloud provider failed, Qwen took over
  warnings: BaselineReasoningWarning[]
  pipeline?: SearchPipelineMetadata
}

export type SearchTimings = {
  total_ms: number
  stages: { id: string; label: string; duration_ms: number; parallel_group?: string }[]
}

export type SearchPipelineMetadata = {
  timings?: SearchTimings
  query_type: 'text_only' | 'image_only' | 'text_image' | string
  id: string
  label: string
  baseline: boolean
  retrieval_strategy: string
  intent_type?: string
  parsed_attributes?: Record<string, string>
  analysis_lanes?: Record<string, unknown>
  detail_hint?: string | null
  descriptions?: Record<string, string | null | undefined>
  target_description_request?: string
  retrieval_description?: string
  text_attribute_fields?: string[]
  constrained_text_attributes?: Record<string, string[]>
  exact_filter_fields?: string[]
  semantic_filter_fields?: string[]
  soft_text_rerank_fields?: string[]
  applied_filters?: Record<string, string>
  user_filter_overrides?: Record<string, string>
  prefetch_count?: number
  result_strategy?: Record<string, unknown>
  notes?: string[]
}

export type BaselineSearchResponse = {
  // 서버 준비부터 응답 본문 수신까지 브라우저에서 측정하며 최근 검색에도 저장한다.
  client_elapsed_ms?: number
  // results는 상품 카드에 사용한다. 검색 상세에는 조건·추천 이유만 표시하고 기술 메타데이터는 응답에 보존한다.
  mode: 'baseline' | 'supabase_vector' | string
  provider: string
  pipeline?: SearchPipelineMetadata
  target_description: string
  target_description_ko?: string
  recommendation_reason?: string
  reasoning?: BaselineReasoningMetadata
  results: BaselineSearchResult[]
}

type SearchBaselineOptions = {
  query: string
  image?: File | null
  topK?: number
  table?: string
  category2?: string
  category2Keyword?: string
  provider?: string
}

function normalizeLocalApiBaseUrl(apiBaseUrl?: string) {
  const trimmedApiBaseUrl = apiBaseUrl?.trim()

  if (!trimmedApiBaseUrl) {
    return undefined
  }

  try {
    const url = new URL(trimmedApiBaseUrl)

    if (url.hostname === 'localhost') {
      url.hostname = '127.0.0.1'
    }

    return url.toString().replace(/\/$/, '')
  } catch {
    return trimmedApiBaseUrl.replace(/\/$/, '')
  }
}

export function hasBaselineSearchConfig() {
  // 주소 문자열 유무만 확인한다. 실제 서버 연결 성공 여부를 검사하는 함수는 아니다.
  return Boolean(baselineApiBaseUrl)
}

export function warmBaselineSearch() {
  if (!baselineApiBaseUrl) {
    return Promise.resolve()
  }

  // 진행 중/성공한 /health 요청은 공유한다. 실패 캐시는 지워 다음 호출에서 준비를 재시도한다.
  if (!baselineWarmupPromise) {
    const controller = new AbortController()
    const timeoutId = window.setTimeout(
      () => controller.abort(),
      API_WARMUP_TIMEOUT_MS,
    )

    baselineWarmupPromise = fetch(`${baselineApiBaseUrl}/health`, {
      signal: controller.signal,
    })
      // fetch는 HTTP 404만으로 reject하지 않으므로 상태를 명시적으로 확인한다.
      .then((response) => {
        if (!response.ok) throw new Error(`Server warmup failed. (${response.status})`)
      })
      .catch(() => { baselineWarmupPromise = null })
      .finally(() => window.clearTimeout(timeoutId))
  }

  return baselineWarmupPromise
}

export type ProviderModels = Record<string, string>

export async function fetchProviderModels(): Promise<ProviderModels> {
  // 모델명 조회 실패는 빈 객체로 반환해 App의 기본 표기를 유지한다. 검색 요청과는 별도다.
  if (!baselineApiBaseUrl) return {}

  await warmBaselineSearch()

  const controller = new AbortController()
  const timeoutId = window.setTimeout(() => controller.abort(), MODEL_INFO_TIMEOUT_MS)
  try {
    const res = await fetch(`${baselineApiBaseUrl.replace(/\/$/, '')}/models`, { signal: controller.signal })
    if (!res.ok) return {}
    return (await res.json()) as ProviderModels
  } catch {
    return {}
  } finally {
    // 모델 정보는 보조 표시이므로 응답 본문까지 시간 제한을 적용하고 실패 시 기본값을 유지한다.
    window.clearTimeout(timeoutId)
  }
}

export function getBaselineImageUrl(result: BaselineSearchResult) {
  if (result.image_url) {
    return result.image_url
  }

  if (!baselineApiBaseUrl) {
    return ''
  }

  if (!result.filename) {
    return ''
  }

  return `${baselineApiBaseUrl.replace(/\/$/, '')}/baseline-images/${result.filename}`
}

export async function searchBaseline({
  query,
  image,
  topK = 10,
  table,
  category2,
  category2Keyword,
  provider,
}: SearchBaselineOptions) {
  if (!baselineApiBaseUrl) {
    throw new Error('VITE_API_BASE_URL is not configured.')
  }

  const startedAt = performance.now()
  await warmBaselineSearch()

  const trimmedQuery = query.trim()
  // 이 await가 끝난 뒤에 검색용 타임아웃을 시작한다. 이미지 요청에는 더 긴 시간을 준다.
  const timeoutMs = image ? IMAGE_SEARCH_TIMEOUT_MS : TEXT_SEARCH_TIMEOUT_MS

  // 문장과 이미지 파일을 함께 보내므로 JSON 대신 multipart/form-data를 사용한다.
  function createFormData() {
    const formData = new FormData()
    // Content-Type은 브라우저가 boundary와 함께 지정하므로 fetch에서 직접 설정하지 않는다.

    if (trimmedQuery) formData.append('query', trimmedQuery)
    if (image) formData.append('image', image)

    formData.append('top_k', String(topK))

    if (table) formData.append('table', table)
    if (category2) formData.append('category2', category2)
    if (category2Keyword) {
      formData.append('category2_keyword', category2Keyword)
    }
    if (provider) formData.append('provider', provider)

    return formData
  }

  let response: Response | null = null

  // 네트워크 오류와 502/503/504만 한 번 재시도한다. 시간 초과는 즉시 사용자에게 알린다.
  for (let attempt = 0; attempt < 2; attempt += 1) {
    const controller = new AbortController()
    // 브라우저의 대기를 중단하는 장치이며, 서버의 모델 작업 취소까지 전파하지는 않는다.
    const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs)

    try {
      response = await fetch(`${baselineApiBaseUrl}/search`, {
        method: 'POST',
        body: createFormData(),
        signal: controller.signal,
      })
    } catch (error) {
      // CORS로 차단된 응답은 status/detail을 읽을 Response 없이 이 catch로 들어올 수 있다.
      if (error instanceof DOMException && error.name === 'AbortError') {
        throw new Error(
          '검색 시간이 오래 걸리고 있습니다. 잠시 후 다시 시도해 주세요.',
          { cause: error },
        )
      }

      if (attempt === 0) {
        continue
      }

      throw error
    } finally {
      // 응답 헤더를 받으면 타이머가 해제되므로 아래 JSON 본문 읽기는 이 시간 제한 밖이다.
      window.clearTimeout(timeoutId)
    }

    if (!RETRYABLE_SEARCH_STATUS_CODES.has(response.status) || attempt === 1) {
      break
    }
  }

  if (!response) {
    throw new Error('검색 서버에 연결하지 못했습니다.')
  }

  if (!response.ok) {
    // 404라도 브라우저가 응답을 읽을 수 있으면 여기서 상태 코드와 JSON detail을 보존한다.
    let detail = `Baseline search failed. (${response.status})`

    try {
      const body = (await response.json()) as { detail?: string }

      if (body.detail) {
        detail = body.detail
      }
    } catch {
      // Keep the generic status message.
    }

    throw new Error(detail)
  }

  // as는 TypeScript의 타입 단언이다. 서버 JSON의 필드 구조를 런타임에 검증하지는 않는다.
  const result = (await response.json()) as BaselineSearchResponse
  return { ...result, client_elapsed_ms: performance.now() - startedAt }
}
