import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import vm from 'node:vm'
import ts from 'typescript'

const sourceRoot = fileURLToPath(new URL('../src/', import.meta.url))

// 실제 TS 모듈을 컴파일해 실행한다. 환경값과 네트워크/시간만 테스트용으로 주입한다.
export function loadModule(entry, { globals = {}, modules = {} } = {}) {
  const cache = new Map()
  function load(filename) {
    if (cache.has(filename)) return cache.get(filename).exports
    const module = { exports: {} }
    cache.set(filename, module)
    const source = readFileSync(filename, 'utf8').replaceAll('import.meta.env', JSON.stringify({
      VITE_API_BASE_URL: 'https://api.example.test',
      VITE_SUPABASE_URL: 'https://db.example.test',
      VITE_SUPABASE_ANON_KEY: 'test-public-key',
    }))
    const { outputText } = ts.transpileModule(source, {
      compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.ReactJSX },
      fileName: filename,
    })
    vm.runInNewContext(outputText, {
      module, exports: module.exports, console, URL, Error, TypeError, AbortController, DOMException, FormData, Response,
      setTimeout, clearTimeout, performance, window: { setTimeout, clearTimeout }, ...globals,
      require(specifier) {
        if (specifier in modules) return modules[specifier]
        if (!specifier.startsWith('.')) throw new Error(`Unexpected test import: ${specifier}`)
        const path = resolve(dirname(filename), specifier)
        return load(/\.[cm]?[jt]sx?$/.test(path) ? path : `${path}.ts`)
      },
    }, { filename })
    return module.exports
  }
  return load(resolve(sourceRoot, entry))
}

export function createClock() {
  let nextId = 0
  const pending = new Map()
  return {
    pending,
    setTimeout(callback, delay) { const id = ++nextId; pending.set(id, { callback, delay }); return id },
    clearTimeout(id) { pending.delete(id) },
    runDelay(delay) {
      for (const [id, timer] of [...pending]) {
        if (timer.delay === delay) { pending.delete(id); timer.callback() }
      }
    },
  }
}

export function deferred() {
  let resolvePromise
  let rejectPromise
  const promise = new Promise((resolve, reject) => { resolvePromise = resolve; rejectPromise = reject })
  return { promise, resolve: resolvePromise, reject: rejectPromise }
}

export async function flushPromises() {
  for (let i = 0; i < 8; i += 1) await Promise.resolve()
}

// DOM 레이아웃을 흉내 내지 않고 실제 App의 상태/이벤트 콜백만 실행한다.
// effect는 기록만 하므로 포커스·스크롤·IntersectionObserver 자체는 브라우저 검증 대상이다.
export function createAppHarness({ search, fetchProducts } = {}) {
  const slots = []
  let cursor = 0
  let callbacks = []
  let effects = []
  const clock = createClock()
  const jsx = (type, props, key) => ({ type, props: props ?? {}, key })
  const react = {
    useState(initial) {
      const index = cursor++
      if (!(index in slots)) slots[index] = typeof initial === 'function' ? initial() : initial
      return [slots[index], (next) => { slots[index] = typeof next === 'function' ? next(slots[index]) : next }]
    },
    useRef(initial) {
      const index = cursor++
      if (!(index in slots)) slots[index] = { current: initial }
      return slots[index]
    },
    useMemo(fn) { return fn() },
    useCallback(fn) { callbacks.push(fn); return fn },
    useEffect(fn) { effects.push(fn) },
    useLayoutEffect(fn) { effects.push(fn) },
  }
  const { default: App } = loadModule('App.tsx', {
    globals: { window: { ...clock, scrollTo() {} } },
    modules: {
      react,
      'react/jsx-runtime': { jsx, jsxs: jsx, Fragment: 'fragment' },
      './lib/supabaseRest': {
        hasSupabaseConfig: () => true,
        fetchCategoryRows: async () => [],
        fetchProducts: fetchProducts ?? (async () => ({ products: [], totalCount: 0 })),
        getProductImageUrl: (product) => product.image_url ?? '',
        getProductThumbnailUrl: (product) => product.image_url ?? '',
        getSupabaseImageThumbnailUrl: (url) => url,
      },
      './lib/baselineSearch': {
        hasBaselineSearchConfig: () => true,
        warmBaselineSearch: async () => {},
        fetchProviderModels: async () => ({}),
        searchBaseline: search ?? (async () => response('default')),
        getBaselineImageUrl: (product) => product.image_url ?? '',
      },
    },
  })
  let tree
  const api = {
    clock,
    render() { cursor = 0; callbacks = []; effects = []; tree = App(); return tree },
    nodes(predicate) {
      const found = []
      function visit(node) {
        if (Array.isArray(node)) { node.forEach(visit); return }
        if (!node || typeof node !== 'object') return
        if (predicate(node)) found.push(node)
        visit(node.props?.children)
      }
      visit(tree)
      return found
    },
    node(predicate) {
      const result = api.nodes(predicate)[0]
      if (!result) throw new Error('Expected App node was not rendered')
      return result
    },
    async submit(query) {
      api.node((node) => node.type === 'textarea').props.onChange({ target: { value: query } })
      api.render()
      api.node((node) => node.type === 'form').props.onSubmit({ preventDefault() {} })
      await flushPromises()
      api.render()
    },
    loadCatalog(offset, append) { return callbacks[0](offset, append) },
    catalogEffect() { return effects.find((effect) => effect.toString().includes('loadProductPage(0, false')) },
  }
  api.render()
  return api
}

export function response(id) {
  return {
    mode: 'supabase_vector', provider: 'gemini', target_description: '',
    results: [{ id, rank: 1, similarity: 0.9, name: id, source_table: 'musinsa_top_clothes' }],
  }
}

export function text(node) {
  if (node == null || typeof node === 'boolean') return ''
  if (Array.isArray(node)) return node.map(text).join('')
  return typeof node === 'object' ? text(node.props?.children) : String(node)
}
