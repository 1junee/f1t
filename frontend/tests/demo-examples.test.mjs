import assert from 'node:assert/strict'
import test from 'node:test'
import { readFileSync } from 'node:fs'
import { loadModule, flushPromises, createAppHarness, response } from './helpers.mjs'

function harness(fetch) {
  let cursor = 0
  const state = [], effects = [], selections = []
  const jsx = (type, props) => ({ type, props })
  const { DemoExamples } = loadModule('DemoExamples.tsx', {
    globals: { fetch, File },
    modules: {
      react: {
        useState(initial) {
          const i = cursor++
          if (!(i in state)) state[i] = initial
          return [state[i], (value) => { state[i] = typeof value === 'function' ? value(state[i]) : value }]
        },
        useEffect(fn) { effects.push(fn) },
      },
      'react/jsx-runtime': { jsx, jsxs: jsx },
    },
  })
  return {
    effects, selections,
    render(disabled = false) {
      cursor = 0
      return DemoExamples({ disabled, onSelect: (...args) => selections.push(args) }).props.children[1].props.children
    },
  }
}

test('바지 → 셔츠 순서로 원본 사진과 요청을 함께 입력하며 선택 자체로 검색하지 않는다', async () => {
  const requests = []
  const demo = harness(async (url) => {
    requests.push(url)
    return new Response(readFileSync(new URL(`../public${url}`, import.meta.url)), { headers: { 'Content-Type': 'image/jpeg' } })
  })
  assert.ok(demo.render().every((button) => button.props.disabled))
  demo.effects[0]()
  await flushPromises()
  for (const button of demo.render()) button.props.onClick()
  assert.equal(demo.selections.length, 2)
  const expected = [
    ['side-stripe', '이 사진의 트레이닝 팬츠와 비슷한데, 옆에 라인이 있는 바지 찾아줘.'],
    ['short-sleeve-shirt', '이 사진의 셔츠와 비슷한데, 소매가 짧은 반팔 셔츠로 찾아줘.'],
  ]
  for (const [i, [id, query]] of expected.entries()) {
    const [actualQuery, file] = demo.selections[i]
    assert.equal(actualQuery, query)
    assert.equal(file.name, `${id}.jpg`)
    assert.deepEqual(Buffer.from(await file.arrayBuffer()), readFileSync(new URL(`../public/demo/${id}.jpg`, import.meta.url)))
  }
  assert.deepEqual(requests, ['/demo/side-stripe.jpg', '/demo/short-sleeve-shirt.jpg'])
  assert.ok(demo.render(true).every((button) => button.props.disabled))
})

test('사진 로드 실패 시 입력을 바꾸지 않고 재시도한다', async () => {
  const demo = harness(async () => new Response('', { status: 404 }))
  demo.render()
  demo.effects[0]()
  await flushPromises()
  const buttons = demo.render()
  assert.ok(buttons.every((button) => !button.props.disabled))
  buttons[0].props.onClick()
  assert.equal(demo.selections.length, 0)
  assert.ok(demo.render().every((button) => button.props.disabled))
})

test('데모 선택 후 스타일 찾기는 선택한 사진·요청을 검색 함수에 전달한다', async () => {
  const calls = []
  const app = createAppHarness({ search: async (args) => { calls.push(args); return response('demo') } })
  const file = new File(['photo'], 'side-stripe.jpg', { type: 'image/jpeg' })
  app.node((node) => typeof node.type === 'function' && node.type.name === 'DemoExamples').props.onSelect('사진의 옆선을 바꿔줘', file)
  app.render()
  assert.equal(calls.length, 0)
  app.node((node) => node.type === 'form').props.onSubmit({ preventDefault() {} })
  await flushPromises()
  assert.equal(calls.length, 1)
  assert.equal(calls[0].image, file)
  assert.equal(calls[0].query, '사진의 옆선을 바꿔줘')
  assert.equal(calls[0].table, undefined)
  assert.equal(calls[0].topK, 10)
})
