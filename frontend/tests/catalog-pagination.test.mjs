import assert from 'node:assert/strict'
import test from 'node:test'
import { loadModule } from './helpers.mjs'

const groups = [
  { id: 'dresses', table: 'musinsa_skirt_dress', category2Keyword: '원피스' },
  { id: 'skirts', table: 'musinsa_skirt_dress', category2Keyword: '스커트' },
]
const product = (id) => ({ id, _sourceTable: 'musinsa_skirt_dress' })

test('그룹 간 같은 상품이 있어도 카드는 한 번만 표시하고 원본 행만큼 커서를 이동한다', async () => {
  const { loadGroupedCatalogPage, appendUniqueProducts } = loadModule('lib/catalogPagination.ts')
  const calls = []
  const fetchPage = async ({ category2Keyword, offset, limit }) => {
    calls.push({ category2Keyword, offset })
    const rows = category2Keyword === '원피스' ? [product(1), product(2), product(3)] : [product(2), product(4), product(5)]
    return { products: rows.slice(offset, offset + limit), totalCount: rows.length }
  }
  const first = await loadGroupedCatalogPage(groups, {}, fetchPage, 4)
  assert.deepEqual(Array.from(first.products, (item) => item.id), [1, 2, 4])
  const second = await loadGroupedCatalogPage(groups, first.pagination, fetchPage, 4)
  assert.deepEqual(calls.slice(2).map((call) => call.offset), [2, 2])
  assert.deepEqual(Array.from(appendUniqueProducts(first.products, second.products), (item) => item.id), [1, 2, 4, 3, 5])
  assert.equal(second.hasMore, false)
})

test('한 그룹 조회가 실패하면 원래 커서를 유지하고 같은 위치에서 다시 읽는다', async () => {
  const { loadGroupedCatalogPage } = loadModule('lib/catalogPagination.ts')
  const pagination = Object.freeze({
    dresses: Object.freeze({ offset: 2, totalCount: 6, exhausted: false }),
    skirts: Object.freeze({ offset: 2, totalCount: 6, exhausted: false }),
  })
  await assert.rejects(loadGroupedCatalogPage(groups, pagination, async ({ category2Keyword }) => {
    if (category2Keyword === '스커트') throw new Error('temporary failure')
    return { products: [product(3), product(4)], totalCount: 6 }
  }, 4), /temporary failure/)
  const offsets = []
  const result = await loadGroupedCatalogPage(groups, pagination, async ({ offset }) => {
    offsets.push(offset)
    return { products: [product(3), product(4)], totalCount: 6 }
  }, 4)
  assert.deepEqual(offsets, [2, 2])
  assert.equal(pagination.dresses.offset, 2)
  assert.equal(result.pagination.dresses.offset, 4)
})

test('전체 수를 모르는 경우 꽉 찬 페이지 뒤에는 계속 조회하고 짧은 페이지에서 멈춘다', async () => {
  const { loadGroupedCatalogPage } = loadModule('lib/catalogPagination.ts')
  const fetchPage = async ({ offset, limit }) => ({ products: [product(1), product(2), product(3)].slice(offset, offset + limit), totalCount: null })
  const first = await loadGroupedCatalogPage(groups.slice(0, 1), {}, fetchPage, 2)
  assert.equal(first.totalCount, null)
  assert.equal(first.hasMore, true)
  const second = await loadGroupedCatalogPage(groups.slice(0, 1), first.pagination, fetchPage, 2)
  assert.equal(second.hasMore, false)
  assert.equal(second.pagination.dresses.offset, 3)
})
