// 카테고리 탐색용 REST 조회다. 자연어·이미지 검색은 baselineSearch.ts에서 백엔드로 보낸다.
import type { Product, ProductTable } from '../types/product'

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL as string | undefined
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY as string | undefined
const supabaseImageTransformEnabled =
  import.meta.env.VITE_SUPABASE_IMAGE_TRANSFORM_ENABLED === 'true'
const baseProductSelect =
  // 목록에 필요한 열만 요청하고, 테이블마다 존재하는 의류 속성 열을 아래에서 덧붙인다.
  'id,name,brand,image_url,category1,category2,price,color'
const productSelectByTable: Record<ProductTable, string> = {
  musinsa_pants: `${baseProductSelect},length,fit,fabric`,
  musinsa_top_clothes: `${baseProductSelect},sleeve,fit,fabric`,
  musinsa_skirt_dress: `${baseProductSelect},sleeve,length,fit,fabric`,
}

type FetchProductsOptions = {
  table: ProductTable
  searchText?: string
  category1?: string
  category2?: string
  category2Keyword?: string
  offset?: number
  limit?: number
}

export type CategoryRow = {
  table: ProductTable
  category1: string
  category2: string
}

type ProductThumbnailOptions = {
  width?: number
  height?: number
  quality?: number
  resize?: 'cover' | 'contain' | 'fill'
}

function normalizeCategoryText(value?: string | null) {
  // 같은 한글/공백 표현을 통일해 화면에서 같은 카테고리가 중복되는 것을 줄인다.
  return (value ?? '')
    .normalize('NFC')
    .replace(/\u00a0/g, ' ')
    .replace(/\s*\/\s*/g, '/')
    .replace(/\s+/g, ' ')
    .trim()
}

function getSearchTerms(searchText: string) {
  // 불용어 제거와 작은 동의어 사전으로 문자열 조건을 확장하는 규칙 기반 처리다.
  const stopWords = new Set([
    '찾아줘',
    '추천해줘',
    '보여줘',
    '옷',
    '상품',
    '아이템',
    '비슷한',
    '있는',
    '없는',
    '으로',
    '으로도',
    '이랑',
    '같은',
  ])
  const synonymMap: Record<string, string[]> = {
    검정: ['검정', '검은', '블랙', 'black'],
    검은: ['검정', '검은', '블랙', 'black'],
    블랙: ['검정', '검은', '블랙', 'black'],
    하얀: ['하얀', '화이트', 'white'],
    흰색: ['흰색', '화이트', 'white'],
    화이트: ['흰색', '화이트', 'white'],
    후드티: ['후드티', '후드'],
    맨투맨: ['맨투맨', '스웨트'],
  }
  const normalizedTerms = searchText
    .normalize('NFC')
    .replace(/[^\p{L}\p{N}\s/]/gu, ' ')
    .split(/\s+/)
    .map((term) => term.trim())
    .filter((term) => term.length > 1 && !stopWords.has(term))

  const expandedTerms = normalizedTerms.flatMap((term) => synonymMap[term] ?? [term])
  const uniqueTerms = [...new Set(expandedTerms)]

  return uniqueTerms.length > 0 ? uniqueTerms : [searchText.trim()]
}

export function hasSupabaseConfig() {
  return Boolean(supabaseUrl && supabaseAnonKey)
}

export function getProductImageUrl(product: Product) {
  if (product.image_url) {
    return product.image_url
  }

  if (!supabaseUrl || !product._sourceTable) {
    return ''
  }

  // 저장된 URL이 없으면 상품 테이블별 Storage 버킷과 상품 ID로 이미지 주소를 만든다.
  const storageBucketByTable: Record<ProductTable, string> = {
    musinsa_pants: 'musinsa_pants',
    musinsa_skirt_dress: 'skirt_dress',
    musinsa_top_clothes: 'test',
  }

  return `${supabaseUrl}/storage/v1/object/public/${
    storageBucketByTable[product._sourceTable]
  }/${product.id}.jpg`
}

export function getSupabaseImageThumbnailUrl(
  imageUrl: string,
  {
    width = 480,
    height = 640,
    quality = 75,
    resize = 'cover',
  }: ProductThumbnailOptions = {},
) {
  if (!supabaseImageTransformEnabled) {
    // 이미지 변환 기능을 켜지 않은 배포에서는 저장된 원본 URL을 그대로 사용한다.
    return imageUrl
  }

  try {
    const url = new URL(imageUrl)
    const objectPublicPath = '/storage/v1/object/public/'

    if (!url.pathname.includes(objectPublicPath)) {
      return imageUrl
    }

    url.pathname = url.pathname.replace(
      objectPublicPath,
      '/storage/v1/render/image/public/',
    )
    url.searchParams.set('width', String(width))
    url.searchParams.set('height', String(height))
    url.searchParams.set('resize', resize)
    url.searchParams.set('quality', String(quality))

    return url.toString()
  } catch {
    return imageUrl
  }
}

export function getProductThumbnailUrl(
  product: Product,
  options: ProductThumbnailOptions = {},
) {
  const imageUrl = getProductImageUrl(product)

  if (!imageUrl) {
    return ''
  }

  return getSupabaseImageThumbnailUrl(imageUrl, options)
}

export async function fetchProducts({
  table,
  searchText = '',
  category1,
  category2,
  category2Keyword,
  offset = 0,
  limit = 20,
}: FetchProductsOptions) {
  if (!supabaseUrl || !supabaseAnonKey) {
    throw new Error(
      'frontend/.env.local에 VITE_SUPABASE_URL과 VITE_SUPABASE_ANON_KEY를 설정해 주세요.',
    )
  }

  const url = new URL(`${supabaseUrl}/rest/v1/${table}`)
  const trimmedSearch = searchText.trim()

  url.searchParams.set('select', productSelectByTable[table])
  url.searchParams.set('offset', String(offset))
  // 서버가 전체 데이터를 보낸 뒤 자르는 것이 아니라 REST 요청 자체에 페이지 범위를 넣는다.
  url.searchParams.set('limit', String(limit))
  // 같은 데이터에서 페이지를 다시 읽어도 순서가 바뀌지 않도록 고유 상품 ID로 정렬한다.
  url.searchParams.set('order', 'id.asc')

  if (category1) {
    url.searchParams.set('category1', `eq.${category1}`)
  }

  if (category2) {
    url.searchParams.set('category2', `eq.${category2}`)
  }

  if (category2Keyword) {
    url.searchParams.set('category2', `ilike.*${category2Keyword}*`)
  }

  // 이 경로의 검색은 문자열 부분 일치이며, 임베딩 유사도 검색이 아니다.
  if (trimmedSearch) {
    const searchTerms = getSearchTerms(trimmedSearch)
    const searchableColumns = [
      'name',
      'brand',
      'category1',
      'category2',
      'color',
      'fabric',
    ]

    url.searchParams.set(
      'or',
      // 여러 단어/열 중 하나만 일치해도 포함한다. 모든 조건을 만족하는 AND 검색은 아니다.
      `(${searchTerms
        .flatMap((term) =>
          searchableColumns.map((column) => `${column}.ilike.*${term}*`),
        )
        .join(',')})`,
    )
  }

  const response = await fetch(url, {
    headers: {
      apikey: supabaseAnonKey,
      Authorization: `Bearer ${supabaseAnonKey}`,
      Prefer: 'count=exact',
    },
  })

  if (response.status === 416) {
    // offset이 전체 개수를 초과 → 더 이상 상품 없음
    const total = response.headers.get('content-range')?.split('/')[1]
    return { products: [], totalCount: total && total !== '*' ? Number(total) : null }
  }

  if (!response.ok) {
    throw new Error(`상품을 불러오지 못했습니다. (${response.status})`)
  }

  const products = (await response.json()) as Product[]
  // 현재 페이지 행 수와 전체 조건 일치 개수를 구분해 무한 스크롤 종료 여부를 판단한다.
  const contentRange = response.headers.get('content-range')
  const totalCountText = contentRange?.split('/')[1]
  const totalCount =
    totalCountText && totalCountText !== '*'
      ? Number(totalCountText)
      : null

  return {
    products: products.map((product) => ({
      // 어느 테이블에서 왔는지 붙여 카드 키와 이미지 버킷 경로를 구분한다.
      ...product,
      _sourceTable: table,
    })),
    totalCount,
  }
}

export async function fetchCategoryRows(table: ProductTable) {
  if (!supabaseUrl || !supabaseAnonKey) {
    throw new Error(
      'frontend/.env.local에 VITE_SUPABASE_URL과 VITE_SUPABASE_ANON_KEY를 설정해 주세요.',
    )
  }

  const url = new URL(`${supabaseUrl}/rest/v1/${table}`)
  url.searchParams.set('select', 'category1,category2')
  url.searchParams.set('limit', '1000')
  // 전체 테이블의 고유 카테고리를 집계하는 API가 아니라 최대 1,000개 행을 읽는 방식이다.

  const response = await fetch(url, {
    headers: {
      apikey: supabaseAnonKey,
      Authorization: `Bearer ${supabaseAnonKey}`,
    },
  })

  if (!response.ok) {
    throw new Error(`카테고리를 불러오지 못했습니다. (${response.status})`)
  }

  const rows = (await response.json()) as Array<{
    category1?: string | null
    category2?: string | null
  }>

  // 조회한 최대 1,000개 행에서 중복을 제거해 사이드바 카테고리 목록을 만든다.
  const seen = new Set<string>()

  return rows.reduce<CategoryRow[]>((categories, row) => {
    const category1 = normalizeCategoryText(row.category1)
    const category2 = normalizeCategoryText(row.category2)

    if (!category1 || !category2) {
      return categories
    }

    const key = `${table}-${category1}-${category2}`.toLowerCase()

    if (seen.has(key)) {
      return categories
    }

    seen.add(key)
    categories.push({ table, category1, category2 })

    return categories
  }, [])
}
