import type { Product, ProductTable } from '../types/product'

type CatalogGroup = {
  id: string
  table: ProductTable
  category1?: string
  category2Keyword?: string
}

type GroupCursor = {
  offset: number
  totalCount: number | null
  exhausted: boolean
}

export type CatalogPagination = Record<string, GroupCursor>
type ProductPage = { products: Product[]; totalCount: number | null }
type FetchGroupPage = (options: Omit<CatalogGroup, 'id'> & {
  offset: number
  limit: number
  searchText: string
}) => Promise<ProductPage>

// 표시한 카드 개수와 DB에서 읽은 행 개수는 다를 수 있다. 중복 제거 후에도 그룹별 offset은 유지한다.
export function appendUniqueProducts(current: Product[], incoming: Product[]) {
  const seen = new Set(current.map((product) => `${product._sourceTable}:${product.id}`))
  return [...current, ...incoming.filter((product) => {
    const key = `${product._sourceTable}:${product.id}`
    if (seen.has(key)) return false
    seen.add(key)
    return true
  })]
}

export async function loadGroupedCatalogPage(
  groups: CatalogGroup[],
  pagination: CatalogPagination,
  fetchPage: FetchGroupPage,
  pageSize: number,
  searchText = '',
) {
  const next: CatalogPagination = Object.fromEntries(groups.map((group) => [
    group.id,
    pagination[group.id] ?? { offset: 0, totalCount: null, exhausted: false },
  ]))
  const activeGroups = groups.filter((group) => !next[group.id].exhausted)
  // 한 그룹이 먼저 소진되면 남은 그룹만 조회한다. 각 그룹이 실제로 읽은 위치에서 이어 간다.
  const pages = await Promise.all(activeGroups.map(async (group, index) => {
    const cursor = next[group.id]
    const limit = Math.floor(pageSize / activeGroups.length) + (index < pageSize % activeGroups.length ? 1 : 0)
    const page = await fetchPage({
      table: group.table,
      category1: group.category1,
      category2Keyword: group.category2Keyword,
      offset: cursor.offset,
      limit,
      searchText,
    })
    return { group, cursor, limit, page }
  }))

  // 모두 성공한 경우에만 새 커서를 반환한다. 일부 조회 실패 시 이전 페이지 위치로 재시도할 수 있다.
  for (const { group, cursor, limit, page } of pages) {
    const offset = cursor.offset + page.products.length
    next[group.id] = {
      offset,
      totalCount: page.totalCount,
      exhausted: page.products.length < limit || (page.totalCount !== null && offset >= page.totalCount),
    }
  }
  const cursors = Object.values(next)
  return {
    products: appendUniqueProducts([], pages.flatMap(({ page }) => page.products)),
    totalCount: cursors.every((cursor) => cursor.totalCount !== null)
      ? cursors.reduce((sum, cursor) => sum + (cursor.totalCount ?? 0), 0)
      : null,
    pagination: next,
    hasMore: cursors.some((cursor) => !cursor.exhausted),
  }
}
