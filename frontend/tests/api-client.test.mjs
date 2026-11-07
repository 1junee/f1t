import assert from 'node:assert/strict'
import test from 'node:test'
import { createClock, deferred, flushPromises, loadModule } from './helpers.mjs'

test('warmup은 동시 요청을 공유하고 HTTP 실패 후에는 다시 확인한다', async () => {
  const first = deferred()
  let calls = 0
  const api = loadModule('lib/baselineSearch.ts', { globals: { fetch: () => ++calls === 1 ? first.promise : Promise.resolve(new Response('', { status: 200 })) } })
  const a = api.warmBaselineSearch()
  const b = api.warmBaselineSearch()
  assert.equal(calls, 1)
  first.resolve(new Response('', { status: 404 }))
  await Promise.all([a, b])
  await api.warmBaselineSearch()
  assert.equal(calls, 2, '실패한 준비 요청을 영구 성공 캐시처럼 보관하지 않는다')
  await api.warmBaselineSearch()
  assert.equal(calls, 2, '성공한 준비 결과는 재사용한다')
})

test('warmup 네트워크 실패도 다음 호출에서 재시도한다', async () => {
  let calls = 0
  const api = loadModule('lib/baselineSearch.ts', { globals: { fetch: async () => { calls += 1; if (calls === 1) throw new TypeError('Failed to fetch'); return new Response('') } } })
  await api.warmBaselineSearch()
  await api.warmBaselineSearch()
  assert.equal(calls, 2)
})

test('/models가 멈추면 제한 시간에 요청을 중단하고 기본 모델 표시를 유지한다', async () => {
  const clock = createClock()
  let modelSignal
  const api = loadModule('lib/baselineSearch.ts', { globals: {
    window: clock,
    fetch: async (url, options) => {
      if (url.endsWith('/health')) return new Response('')
      modelSignal = options?.signal
      return new Promise((resolve, reject) => modelSignal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError'))))
    },
  } })
  const request = api.fetchProviderModels()
  await flushPromises()
  assert.ok(modelSignal, '/models에도 취소 신호가 필요하다')
  const timer = [...clock.pending.values()][0]
  assert.ok(timer && timer.delay > 0 && timer.delay <= 20_000)
  clock.runDelay(timer.delay)
  assert.deepEqual(Object.keys(await request), [])
  assert.equal(clock.pending.size, 0)
})

test('카탈로그 REST 요청은 모든 페이지에 안정적인 id 정렬을 사용한다', async () => {
  const urls = []
  const api = loadModule('lib/supabaseRest.ts', { globals: { fetch: async (url) => {
    urls.push(new URL(url))
    return new Response('[]', { headers: { 'content-range': '0-0/0' } })
  } } })
  await api.fetchProducts({ table: 'musinsa_pants', offset: 0, limit: 20 })
  await api.fetchProducts({ table: 'musinsa_pants', offset: 20, limit: 20 })
  assert.deepEqual(urls.map((url) => url.searchParams.get('order')), ['id.asc', 'id.asc'])
})

test('warmup 시간 초과 뒤에도 다음 준비 요청은 다시 실행한다', async () => {
  const clock = createClock()
  let calls = 0
  const api = loadModule('lib/baselineSearch.ts', { globals: {
    window: clock,
    fetch: (url, { signal }) => {
      calls += 1
      if (calls > 1) return Promise.resolve(new Response(''))
      return new Promise((resolve, reject) => signal.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError'))))
    },
  } })
  const first = api.warmBaselineSearch()
  clock.runDelay(20_000)
  await first
  await api.warmBaselineSearch()
  assert.equal(calls, 2)
  assert.equal(clock.pending.size, 0)
})

test('/models는 정상 응답 형식을 유지하고 JSON 본문 대기에도 timeout을 적용한다', async () => {
  const clock = createClock()
  let stuck = false
  const api = loadModule('lib/baselineSearch.ts', { globals: {
    window: clock,
    fetch: async (url, { signal }) => {
      if (url.endsWith('/health')) return new Response('')
      if (!stuck) return new Response(JSON.stringify({ gemini: 'example-model', openai: 'other-model' }))
      return { ok: true, json: () => new Promise((resolve, reject) => signal.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')))) }
    },
  } })
  assert.deepEqual(await api.fetchProviderModels(), { gemini: 'example-model', openai: 'other-model' })
  stuck = true
  const request = api.fetchProviderModels()
  await flushPromises()
  clock.runDelay(10_000)
  assert.deepEqual(Object.keys(await request), [])
  assert.equal(clock.pending.size, 0)
})

test('count 헤더가 없으면 현재 페이지 크기를 전체 상품 수로 오인하지 않는다', async () => {
  const api = loadModule('lib/supabaseRest.ts', { globals: { fetch: async () => new Response('[{"id":1}]') } })
  const result = await api.fetchProducts({ table: 'musinsa_pants', offset: 20, limit: 1 })
  assert.equal(result.totalCount, null)
})
