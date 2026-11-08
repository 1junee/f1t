import assert from 'node:assert/strict'
import test from 'node:test'
import { createAppHarness, deferred, flushPromises, response, text } from './helpers.mjs'

test('전체 탭: 먼저 소진된 그룹이 있어도 모든 상품을 한 번씩 표시한다', async () => {
  const sizes = { musinsa_top_clothes: 2, musinsa_pants: 9, 원피스: 15, 스커트: 0 }
  const calls = []
  const app = createAppHarness({
    fetchProducts: async ({ table, category2Keyword, offset, limit }) => {
      const group = category2Keyword ?? table
      calls.push({ group, offset })
      const all = Array.from({ length: sizes[group] }, (_, index) => ({ id: `${group}-${index}`, name: `${group}-${index}`, _sourceTable: table }))
      return { products: all.slice(offset, offset + limit), totalCount: all.length }
    },
  })
  let loaded = 0
  for (let page = 0; page < 6; page += 1) {
    await app.loadCatalog(loaded, page > 0)
    app.render()
    loaded = app.nodes((node) => node.type === 'article').length
    if (loaded >= 26) break
  }
  const cards = app.nodes((node) => node.type === 'article').map((node) => node.key)
  assert.equal(cards.length, 26)
  assert.equal(new Set(cards).size, 26, '이미 표시한 상품이 반복되면 안 된다')
  assert.equal(calls.filter((call) => call.group === 'musinsa_top_clothes').length, 1, '소진 그룹은 다시 조회하지 않는다')
  assert.equal(calls.filter((call) => call.group === '스커트').length, 1)
})

test('홈 이동은 검색 실패를 지우고 카탈로그를 다시 조회할 수 있게 한다', async () => {
  const app = createAppHarness({ search: async () => { throw new Error('search unavailable') } })
  await app.submit('검은 셔츠')
  assert.equal(app.nodes((node) => node.props.role === 'alert').length, 1)
  app.node((node) => node.props['aria-label'] === 'F1T 홈').props.onClick()
  app.render()
  assert.equal(app.nodes((node) => node.props.role === 'alert').length, 0)
  app.catalogEffect()()
  assert.ok([...app.clock.pending.values()].some((timer) => timer.delay === 0), '홈 카탈로그 조회가 예약되어야 한다')
})

test('무한 스크롤 콜백이 겹쳐도 같은 페이지는 한 번만 요청한다', async () => {
  const pending = deferred()
  let calls = 0
  const app = createAppHarness({ fetchProducts: () => { calls += 1; return pending.promise } })
  const first = app.loadCatalog(0, false)
  await app.loadCatalog(0, false)
  assert.equal(calls, 4)
  pending.resolve({ products: [], totalCount: 0 })
  await first
  app.render()
  await app.loadCatalog(0, true)
  assert.equal(calls, 4, '모든 그룹이 소진된 뒤에는 추가 요청이 없어야 한다')
})

test('카테고리 전환 직후 오래된 append가 먼저 와도 새 목록은 offset 0에서 시작한다', async () => {
  const pendingAppend = deferred()
  const transitionCalls = []
  let transitioning = false
  const makePage = ({ table, category2Keyword, offset, limit }) => ({
    products: Array.from({ length: limit }, (_, index) => ({ id: `${category2Keyword ?? table}-${offset + index}`, name: `${category2Keyword ?? table}-${offset + index}`, _sourceTable: table })),
    totalCount: 100,
  })
  const app = createAppHarness({ fetchProducts: (options) => {
    if (transitioning) {
      transitionCalls.push(options)
      if (options.offset === 20) return pendingAppend.promise
    }
    return Promise.resolve(makePage(options))
  } })
  await app.loadCatalog(0, false)
  app.render()
  assert.equal(app.nodes((node) => node.type === 'article').length, 20)
  transitioning = true
  app.node((node) => node.type === 'button' && text(node) === '바지').props.onClick()
  app.render()
  app.catalogEffect()()
  const staleAppend = app.loadCatalog(20, true)
  app.clock.runDelay(0)
  pendingAppend.resolve(makePage({ table: 'musinsa_pants', offset: 20, limit: 20 }))
  await staleAppend
  await flushPromises()
  app.render()
  assert.deepEqual(transitionCalls.map((call) => call.offset), [0])
  assert.equal(app.nodes((node) => node.type === 'article').length, 20)
  assert.match(text(app.nodes((node) => node.type === 'article')[0]), /musinsa_pants-0/)
})

test('홈 이동 후 늦게 도착한 검색 응답은 화면을 덮지 않는다', async () => {
  const pending = deferred()
  const app = createAppHarness({ search: () => pending.promise })
  await app.submit('이전 검색')
  app.node((node) => node.props['aria-label'] === 'F1T 홈').props.onClick()
  pending.resolve(response('late-result'))
  await flushPromises()
  app.render()
  assert.equal(app.nodes((node) => node.type === 'article').length, 0)
})

test('최근 검색 복원은 진행 요청을 무효화하고 이전 완료 타이머를 정리한다', async () => {
  const pending = deferred()
  let count = 0
  const app = createAppHarness({ search: () => ++count === 1 ? Promise.resolve(response('saved-result')) : pending.promise })
  await app.submit('저장한 검색')
  assert.equal(app.clock.pending.size, 2)
  app.node((node) => node.props['aria-label'] === '최근 검색과 전체 카테고리 열기').props.onClick()
  app.render()
  app.node((node) => node.type === 'button' && text(node) === '저장한 검색↗').props.onClick()
  assert.equal(app.clock.pending.size, 0, '완료 자동 스크롤/표시 타이머를 취소한다')
  app.render()
  await app.submit('새 요청')
  app.node((node) => node.props['aria-label'] === '최근 검색과 전체 카테고리 열기').props.onClick()
  app.render()
  app.node((node) => node.type === 'button' && text(node) === '저장한 검색↗').props.onClick()
  pending.resolve(response('late-result'))
  await flushPromises()
  app.render()
  const cards = app.nodes((node) => node.type === 'article')
  assert.equal(cards.length, 1)
  assert.match(text(cards[0]), /saved-result/)
})

test('소개 창을 열어도 진행 중인 검색 상태를 유지한다', async () => {
  const pending = deferred()
  const app = createAppHarness({ search: () => pending.promise })
  await app.submit('검색 진행 중')
  app.node((node) => node.props.className === 'f1t-about-button').props.onClick()
  app.render()
  assert.equal(app.node((node) => node.props.className === 'f1t-search-button').props.disabled, true)
  assert.match(text(app.node((node) => node.props.role === 'status')), /상품을 찾고 있어요/)
  pending.resolve(response('done'))
  await flushPromises()
})

test('최근 검색을 복원하면 결과와 함께 검색 당시 카테고리 탭도 돌아온다', async () => {
  const app = createAppHarness({ search: async () => response('top-result') })
  app.node((node) => node.type === 'button' && text(node) === '상의').props.onClick()
  app.render()
  await app.submit('검은 셔츠')
  app.node((node) => node.type === 'button' && text(node) === '바지').props.onClick()
  app.render()
  app.node((node) => node.props['aria-label'] === '최근 검색과 전체 카테고리 열기').props.onClick()
  app.render()
  app.node((node) => node.type === 'button' && text(node) === '검은 셔츠↗').props.onClick()
  app.render()
  assert.equal(text(app.node((node) => node.props['aria-current'] === 'page')), '상의')
  assert.match(text(app.node((node) => node.props.className === 'f1t-request-summary')), /상의/)
  assert.match(text(app.node((node) => node.type === 'article')), /top-result/)
})


test('검색 완료 시간과 병렬 단계 시간을 표시하고 홈 이동 시 지운다', async () => {
  const app = createAppHarness({ search: async () => ({
    ...response('timed'), client_elapsed_ms: 1250,
    pipeline: { timings: { total_ms: 1100, stages: [
      { id: 'intent', label: '검색 의도·조건 해석', duration_ms: 800, parallel_group: 'analysis' },
      { id: 'description', label: '목표 상품 설명 생성', duration_ms: 900, parallel_group: 'analysis' },
      { id: 'search_plan', label: '검색 조건 구성', duration_ms: 10 },
      { id: 'translation', label: '번역', duration_ms: 100, parallel_group: 'encoding' },
      { id: 'embedding', label: '설명 임베딩', duration_ms: 120, parallel_group: 'encoding' },
      { id: 'fabric_embedding', label: '소재 임베딩', duration_ms: 110, parallel_group: 'encoding' },
    ] } },
  }) })
  await app.submit('검은 셔츠')
  const timing = app.node((node) => node.props.className === 'f1t-search-timing')
  assert.match(text(timing), /1.25초.*검색 완료/)
  assert.match(text(timing), /서버 처리 1.10초/)
  assert.match(text(timing), /검색 의도·조건 해석1차 묶음 · 동시 실행0.80초/)
  assert.match(text(timing), /목표 상품 설명 생성1차 묶음 · 동시 실행0.90초/)
  assert.match(text(timing), /검색 조건 구성0.01초/)
  assert.match(text(timing), /번역2차 묶음 · 동시 실행0.10초/)
  assert.match(text(timing), /설명 임베딩2차 묶음 · 동시 실행0.12초/)
  assert.match(text(timing), /소재 임베딩2차 묶음 · 동시 실행0.11초/)
  assert.match(text(timing), /1차 묶음.*끝나면 검색 조건을 구성하고, 2차 묶음/)
  assert.match(text(timing), /각 묶음 안에서만 동시에 진행/)
  app.node((node) => node.props['aria-label'] === 'F1T 홈').props.onClick()
  app.render()
  assert.equal(app.nodes((node) => node.props.className === 'f1t-search-timing').length, 0)
})

test('이전 서버의 응답도 전체 시간만 표시하고 정상 처리한다', async () => {
  const app = createAppHarness({ search: async () => ({ ...response('legacy'), client_elapsed_ms: 2200 }) })
  await app.submit('셔츠')
  const timing = text(app.node((node) => node.props.className === 'f1t-search-timing'))
  assert.match(timing, /2.20초/)
  assert.match(timing, /단계별 시간 정보는 제공되지/)
})
