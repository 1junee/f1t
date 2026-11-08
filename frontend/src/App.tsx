import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import type { SyntheticEvent } from 'react'

import {
  fetchCategoryRows,
  fetchProducts,
  getProductImageUrl,
  getProductThumbnailUrl,
  getSupabaseImageThumbnailUrl,
  hasSupabaseConfig,
} from './lib/supabaseRest'
import {
  fetchProviderModels,
  getBaselineImageUrl,
  hasBaselineSearchConfig,
  searchBaseline,
  warmBaselineSearch,
} from './lib/baselineSearch'
import type { BaselineSearchResponse, ProviderModels } from './lib/baselineSearch'
import type { CategoryRow } from './lib/supabaseRest'
import type { Product, ProductTable } from './types/product'
import { appendUniqueProducts, loadGroupedCatalogPage } from './lib/catalogPagination'
import type { CatalogPagination } from './lib/catalogPagination'

type CategoryGroup = {
  id: string
  label: string
  table: ProductTable
  category1?: string
  category2Keyword?: string
}

type ActiveCategory = {
  id: string
  label: string
  isAll?: boolean
  table?: ProductTable
  category2?: string
  category2Keyword?: string
}

type SelectedImage = {
  file: File
  previewUrl: string
}

type RecentSearchItem = {
  id: string
  query: string
  mode?: string
  submittedHadImage?: boolean
  categoryId?: string
  categoryLabel?: string
  products?: Product[]
  totalProductCount?: number | null
  baselineSearchResponse?: BaselineSearchResponse
}

type ModalPageId = 'about'
type NavItemId = 'home' | ModalPageId

const categoryGroups: CategoryGroup[] = [
  { id: 'tops', label: '상의', table: 'musinsa_top_clothes' },
  { id: 'pants', label: '바지', table: 'musinsa_pants' },
  {
    id: 'dresses',
    label: '원피스',
    table: 'musinsa_skirt_dress',
    category2Keyword: '원피스',
  },
  {
    id: 'skirts',
    label: '스커트',
    table: 'musinsa_skirt_dress',
    category2Keyword: '스커트',
  },
]

const PRODUCTS_PER_PAGE = 20
const SEARCH_PROGRESS_INTERVAL_MS = 2600
const allCategory: ActiveCategory = {
  id: 'all',
  label: '전체',
  isAll: true,
}
const imageReferenceQueryPattern =
  /(이\s*(이미지|사진|옷|상품|디자인|스타일|거|것)|이미지처럼|사진처럼|첨부|올린\s*(이미지|사진)|여기서|이거랑|이\s*옷이랑)/
const pageNavItems: { id: NavItemId; label: string }[] = [
  { id: 'home', label: '홈' },
  { id: 'about', label: '소개' },
]

const searchExamples: RecentSearchItem[] = [
  {
    id: 'sample-1',
    query: '이 이미지랑 비슷한 색으로 긴소매 셔츠 찾아줘',
  },
  {
    id: 'sample-2',
    query: '러닝할 때 입기 좋은 가벼운 바지 추천해줘',
  },
  {
    id: 'sample-3',
    query: '패턴 없는 미니멀한 원피스 찾아줘',
  },
]

const textSearchProgressSteps = [
  { title: '스타일에 맞는 상품을 찾고 있어요', description: '색상이나 소재를 함께 적으면 원하는 옷을 찾는 데 도움이 돼요.' },
  { title: '스타일에 맞는 상품을 찾고 있어요', description: '계절이나 입을 상황도 검색 조건으로 사용할 수 있어요.' },
  { title: '스타일에 맞는 상품을 찾고 있어요', description: '검색이 끝나면 아래에서 상품을 확인할 수 있어요.' },
]

const imageSearchProgressSteps = [
  { title: '스타일에 맞는 상품을 찾고 있어요', description: '사진과 원하는 조건을 함께 참고해 상품을 찾아요.' },
  { title: '스타일에 맞는 상품을 찾고 있어요', description: '사진에서 바꾸고 싶은 색상이나 기장을 문장으로 적어도 좋아요.' },
  { title: '스타일에 맞는 상품을 찾고 있어요', description: '검색이 끝나면 아래에서 상품을 확인할 수 있어요.' },
]

function getCategoryKey(category2: string) {
  return category2.toLowerCase()
}

function formatPrice(price: Product['price']) {
  if (price === null || price === undefined || price === '') {
    return ''
  }

  const numberPrice = Number(price)

  if (Number.isNaN(numberPrice)) {
    return String(price)
  }

  return `${numberPrice.toLocaleString('ko-KR')}원`
}

function handleImageFallback(event: SyntheticEvent<HTMLImageElement>) {
  // 썸네일 로드가 실패하면 원본으로 한 번 바꾼다. 속성을 지워 무한 재시도를 막는다.
  const image = event.currentTarget
  const fallbackSrc = image.dataset.fallbackSrc

  if (!fallbackSrc || image.src === fallbackSrc) {
    return
  }

  image.removeAttribute('data-fallback-src')
  image.src = fallbackSrc
}

const attributeDisplayLabels: Record<string, string> = {
  sleeve: '소매',
  length: '기장',
  color: '색상',
  category1: '카테고리',
  sex: '성별',
  season: '시즌',
  stretch: '신축성',
  thickness: '두께',
  fit: '핏',
  fabric: '소재',
}

function getParsedAttributeItems(pipeline?: BaselineSearchResponse['pipeline']) {
  return Object.entries(pipeline?.parsed_attributes ?? {})
    .filter(([, value]) => Boolean(value))
    .map(([key, value]) => ({
      key,
      label: attributeDisplayLabels[key] ?? key,
      value,
    }))
}

function AboutPage() {
  // 소개 내용만 그리는 정적 컴포넌트다. 검색 상태나 API 호출은 갖지 않는다.
  return (
    <section>
      <p className="text-sm font-medium text-slate-500">About F1T</p>
      <h1 className="mt-3 text-2xl font-semibold tracking-tight text-slate-950 md:text-3xl">
        머릿속에 그린 옷을 이미지와 말로 찾아보세요
      </h1>
      <p className="mt-5 text-base leading-7 text-slate-600">
        F1T는 자연어와 참고 이미지를 함께 이해해 사용자의 의도를 실제 패션
        상품으로 연결하는 검색 서비스입니다. 찾고 싶은 옷을 평소 말하듯
        설명하면 카테고리, 색상, 핏, 패턴, 소재와 입을 상황을 반영해 19,833개
        상품 중 요청에 가까운 결과를 보여줍니다.
      </p>

      <div className="mt-10 grid gap-4 md:grid-cols-3">
        <article className="rounded-xl border border-slate-200 bg-slate-50 p-5">
          <h2 className="text-base font-semibold text-slate-950">
            문장으로 바로 검색
          </h2>
          <p className="mt-3 text-sm leading-6 text-slate-500">
            “출근할 때 입을 단정한 검은 셔츠”처럼 상품명을 몰라도 원하는
            분위기와 조건을 자연스럽게 입력할 수 있습니다.
          </p>
        </article>

        <article className="rounded-xl border border-slate-200 bg-slate-50 p-5">
          <h2 className="text-base font-semibold text-slate-950">
            이미지와 조건 결합
          </h2>
          <p className="mt-3 text-sm leading-6 text-slate-500">
            마음에 든 옷 사진을 올리고 “비슷하지만 검은색으로”처럼 바꾸고
            싶은 조건을 덧붙이면, 이미지의 특징과 문장의 요구를 함께
            반영합니다.
          </p>
        </article>

        <article className="rounded-xl border border-slate-200 bg-slate-50 p-5">
          <h2 className="text-base font-semibold text-slate-950">
            가까운 상품과 이유 확인
          </h2>
          <p className="mt-3 text-sm leading-6 text-slate-500">
            상품의 속성과 시각적 의미를 함께 비교해 요청에 가까운 순서로
            정렬하고, 어떤 조건이 추천에 반영됐는지도 결과에서 확인할 수
            있습니다.
          </p>
        </article>
      </div>

      <div className="mt-8 rounded-xl border border-slate-200 bg-slate-50 p-6">
        <h2 className="text-lg font-semibold text-slate-950">
          키워드에 담긴 조건과 원하는 스타일을 함께 살펴봅니다
        </h2>
        <p className="mt-3 text-sm leading-6 text-slate-500">
          “검은색”, “여유 있는 핏”처럼 구체적인 조건과 “출근할 때 입을 옷”처럼
          상황을 설명하는 문장을 함께 참고합니다. 사진을 첨부하면 이미지의
          특징도 반영해 가까운 상품을 찾습니다. 사진과 다른 조건을 문장으로
          요청하면, 말로 지정한 조건을 우선하도록 설계했습니다.
        </p>
      </div>
    </section>
  )
}

function PolicyModal({
  onClose,
}: {
  onClose: () => void
}) {
  // 닫기 동작은 부모가 전달한 onClose로 위임한다. 열림 여부는 App의 activeModal이 결정한다.
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/35 px-4 py-8 backdrop-blur-sm"
      role="dialog"
      aria-modal="true"
      data-f1t-dialog="about"
      tabIndex={-1}
      aria-labelledby="about-modal-title"
      onMouseDown={onClose}
    >
      <div
        className="max-h-[90vh] w-full max-w-6xl overflow-y-auto rounded-2xl border border-slate-200 bg-white p-8 shadow-2xl md:p-12"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="mb-6 flex items-center justify-between gap-4">
          <h2
            id="about-modal-title"
            className="text-sm font-semibold text-slate-400"
          >
            서비스 소개
          </h2>

          <button
            type="button"
            onClick={onClose}
            className="flex h-9 w-9 items-center justify-center rounded-full border border-slate-200 text-lg leading-none text-slate-500 transition hover:border-slate-300 hover:bg-slate-50 hover:text-slate-950"
            aria-label="팝업 닫기"
          >
            ×
          </button>
        </div>

        <AboutPage />
      </div>
    </div>
  )
}

function App() {
  // DOM 위치와 비동기 요청 식별자는 ref, 화면에 표시할 입력·결과는 state로 관리한다.
  const imageInputRef = useRef<HTMLInputElement | null>(null)
  const loadMoreRef = useRef<HTMLDivElement | null>(null)
  const resultsSectionRef = useRef<HTMLDivElement | null>(null)
  const sidebarScrollRef = useRef<HTMLDivElement | null>(null)
  const categoryGroupRefs = useRef<Map<string, HTMLDivElement>>(new Map())
  const sidebarAnchorRestoreTimeoutsRef = useRef<number[]>([])
  const pendingScrollAdjustRef = useRef<{
    groupId: string
    previousTop: number
  } | null>(null)
  const requestIdRef = useRef(0)
  const catalogPaginationRef = useRef<CatalogPagination>({})
  const catalogRequestRef = useRef<number | null>(null)
  const catalogReadyRef = useRef(false)
  // 요청 번호는 응답을 버릴지 판단하는 값이다. fetch 자체를 취소하는 AbortController와 다르다.
  const searchCompleteDelayRef = useRef<number | null>(null)
  const searchCompleteHideDelayRef = useRef<number | null>(null)
  const [activeModal, setActiveModal] = useState<ModalPageId | null>(null)
  const [isSidebarOpen, setIsSidebarOpen] = useState(false)
  const [activeCategoryId, setActiveCategoryId] = useState('all')
  const [openCategoryIds, setOpenCategoryIds] = useState(() => new Set<string>())
  const [categoryRows, setCategoryRows] = useState<CategoryRow[]>([])
  // 입력 중인 문장과 마지막으로 제출한 문장을 분리해 결과 설명이 바뀌지 않게 한다.
  const [query, setQuery] = useState('')
  const [submittedQuery, setSubmittedQuery] = useState('')
  const [submittedHadImage, setSubmittedHadImage] = useState(false)
  const [searchRunId, setSearchRunId] = useState(0)
  // 같은 카테고리라도 홈/재시도에서 값을 올리면 카탈로그 조회 effect를 다시 실행한다.
  const [isUnderstandingRequest, setIsUnderstandingRequest] = useState(false)
  const [isSearchCompleteVisible, setIsSearchCompleteVisible] = useState(false)
  const [searchProgressIndex, setSearchProgressIndex] = useState(0)
  const [searchStartedAt, setSearchStartedAt] = useState<number | null>(null)
  const [searchElapsedSeconds, setSearchElapsedSeconds] = useState(0)
  const [selectedImage, setSelectedImage] = useState<SelectedImage | null>(null)
  const [isImageDragging, setIsImageDragging] = useState(false)
  const [selectedProvider, setSelectedProvider] = useState('gemini')
  // provider는 요청 해석·설명 생성 모델 선택이며, 검색 임베딩 모델을 바꾸는 메뉴는 아니다.
  const [isProviderMenuOpen, setIsProviderMenuOpen] = useState(false)
  const [providerModels, setProviderModels] = useState<ProviderModels>({
    gemini: 'gemini-3.5-flash-lite',
    openai: 'gpt-5.4-nano',
    qwen: 'qwen3.5-0.8B',
  })
  // 모델 검색 응답과 카탈로그 상품 목록은 별도 상태다. 아래 렌더링에서 두 경로를 구분한다.
  const [baselineSearchResponse, setBaselineSearchResponse] =
    useState<BaselineSearchResponse | null>(null)
  const [recentSearches, setRecentSearches] = useState<RecentSearchItem[]>([])
  const [likedItemIds, setLikedItemIds] = useState(() => new Set<string>())
  const [products, setProducts] = useState<Product[]>([])
  const [totalProductCount, setTotalProductCount] = useState<number | null>(null)
  const [hasMoreProducts, setHasMoreProducts] = useState(true)
  const [selectedProductForDetail, setSelectedProductForDetail] = useState<Product | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  // isLoading은 카탈로그 조회, isUnderstandingRequest는 AI 검색 대기 상태다.
  const [errorMessage, setErrorMessage] = useState('')

  // DB의 세부 카테고리 행을 화면의 상의·바지·원피스·스커트 그룹에 연결한다.
  const categoryTree = useMemo(
    () =>
      categoryGroups.map((group) => {
        const seenCategory2 = new Set<string>()
        const children = categoryRows
          .filter((row) => {
            if (row.table !== group.table) {
              return false
            }

            if (group.category1 && row.category1 !== group.category1) {
              return false
            }

            if (
              group.category2Keyword &&
              !row.category2.includes(group.category2Keyword)
            ) {
              return false
            }

            return true
          })
          .filter((row) => {
            const categoryKey = getCategoryKey(row.category2)

            if (seenCategory2.has(categoryKey)) {
              return false
            }

            seenCategory2.add(categoryKey)
            return true
          })
          .sort((a, b) => a.category2.localeCompare(b.category2, 'ko-KR'))

        return { ...group, children }
      }),
    [categoryRows],
  )

  const activeCategory: ActiveCategory = (() => {
    if (activeCategoryId === allCategory.id) {
      return allCategory
    }

    for (const group of categoryTree) {
      if (activeCategoryId === group.id) {
        return group
      }

      const child = group.children.find(
        (row) => `${group.id}-${row.category2}` === activeCategoryId,
      )

      if (child) {
        return {
          id: `${group.id}-${child.category2}`,
          label: child.category2,
          table: child.table,
          category2: child.category2,
        }
      }
    }

    return categoryGroups[0]
  })()

  const searchProgressSteps = selectedImage
    ? imageSearchProgressSteps
    : textSearchProgressSteps
  const currentSearchProgress =
    searchProgressSteps[
      Math.min(searchProgressIndex, searchProgressSteps.length - 1)
    ]
  const requestSummaryItems = useMemo(() => {
    // 입력창의 현재 문장 대신 제출 당시 조건을 결과 위에 표시한다.
    const items: string[] = []

    if (activeCategory.label !== allCategory.label) {
      items.push(activeCategory.label)
    }

    if (submittedHadImage) {
      items.push('이미지 참고')
    }

    if (submittedQuery) {
      items.push(submittedQuery)
    }

    return items
  }, [activeCategory.label, submittedHadImage, submittedQuery])
  const baselineResults = baselineSearchResponse?.results ?? []
  const searchTimings = baselineSearchResponse?.pipeline?.timings
  const elapsedMs = baselineSearchResponse?.client_elapsed_ms
  const isBaselineSearch = Boolean(baselineSearchResponse)
  const hasSubmittedSearch = isBaselineSearch || Boolean(submittedQuery) || submittedHadImage
  // 대화상자 하나를 기준으로 초점 관리 effect가 동작하도록 화면 상태를 이름으로 정리한다.
  const activeDialog = selectedProductForDetail ? 'product' : activeModal ? 'about' : isSidebarOpen ? 'drawer' : null
  const parsedAttributeItems = useMemo(
    () => getParsedAttributeItems(baselineSearchResponse?.pipeline),
    [baselineSearchResponse?.pipeline],
  )
  // 내부 모델명과 경고 코드는 API에 남기되, 고객에게는 검색 품질에 영향을 주는 상태만 안내한다.
  const searchInterpretationLimited = Boolean(
    baselineSearchResponse?.reasoning &&
    (['fallback', 'degraded'].includes(baselineSearchResponse.reasoning.status) ||
      baselineSearchResponse.reasoning.provider === 'raw_query'),
  )

  // 첫 진입 때 서버를 준비하고, 모델 선택 메뉴에 표시할 서버 설정의 모델명을 조회한다.
  useEffect(() => {
    void warmBaselineSearch()

    fetchProviderModels().then((models) => {
      if (Object.keys(models).length > 0) {
        setProviderModels(models)
      }
    })
  }, [])

  useEffect(() => {
    let ignore = false
    // 세 테이블의 카테고리 목록은 서로 의존하지 않아 Promise.all로 함께 가져온다.

    async function loadCategories() {
      if (!hasSupabaseConfig()) {
        return
      }

      try {
        const rows = (
          await Promise.all([
            fetchCategoryRows('musinsa_top_clothes'),
            fetchCategoryRows('musinsa_skirt_dress'),
            fetchCategoryRows('musinsa_pants'),
          ])
        ).flat()

        if (!ignore) {
          setCategoryRows(rows)
        }
      } catch {
        if (!ignore) {
          setCategoryRows([])
        }
      }
    }

    loadCategories()

    return () => {
      // 화면이 사라진 뒤 응답이 와도 setState하지 않는다. 네트워크 요청 취소는 아니다.
      ignore = true
    }
  }, [])

  // 카테고리를 둘러볼 때는 모델 호출 없이 Supabase에서 상품 페이지를 가져온다.
  const loadProductPage = useCallback(
    async (offset: number, append: boolean, requestId = requestIdRef.current) => {
      // offset은 시작 위치, append는 기존 목록에 이어 붙일지 여부다.
      // 같은 화면의 무한 스크롤 콜백이 연속 실행되어도 한 페이지를 중복 요청하지 않는다.
      if (requestId !== requestIdRef.current || catalogRequestRef.current === requestId) return
      // 전환 직후의 오래된 observer가 offset 0 조회보다 먼저 append를 시작하지 못하게 한다.
      if (append && !catalogReadyRef.current) return
      if (!hasSupabaseConfig()) {
        setErrorMessage(
          'Supabase 연결을 위해 frontend/.env.local에 공개용 anon key를 설정해 주세요.',
        )
        return
      }

      catalogRequestRef.current = requestId
      setIsLoading(true)
      setErrorMessage('')

      if (!append) {
        catalogReadyRef.current = false
        setTotalProductCount(null)
      }

      try {
        let result: { products: Product[]; totalCount: number | null }
        let nextPagination = catalogPaginationRef.current
        let hasMore: boolean

        if (activeCategory.isAll) {
          // 전체 탭은 네 카테고리에서 분할 조회한 페이지를 합친다. 추천 순위 정렬은 하지 않는다.
          const page = await loadGroupedCatalogPage(
            categoryGroups,
            append ? catalogPaginationRef.current : {},
            fetchProducts,
            PRODUCTS_PER_PAGE,
            submittedQuery,
          )
          result = page
          nextPagination = page.pagination
          hasMore = page.hasMore
        } else {
          result = await fetchProducts({
            table: activeCategory.table ?? 'musinsa_top_clothes',
            category2: activeCategory.category2,
            category2Keyword: activeCategory.category2Keyword,
            searchText: submittedQuery,
            offset,
            limit: PRODUCTS_PER_PAGE,
          })
          hasMore = result.products.length === PRODUCTS_PER_PAGE &&
            (result.totalCount === null || offset + result.products.length < result.totalCount)
        }

        // 늦게 도착한 이전 요청이 새 화면을 덮어쓰지 않도록 요청 번호를 확인한다.
        if (requestId === requestIdRef.current) {
          catalogReadyRef.current = true
          catalogPaginationRef.current = nextPagination
          setHasMoreProducts(hasMore)
          setTotalProductCount(result.totalCount)
          setProducts((currentProducts) =>
            appendUniqueProducts(append ? currentProducts : [], result.products),
          )

          if (!append && (submittedQuery || submittedHadImage)) {
            saveRecentSearchSnapshot({
              id: `catalog-${Date.now()}`,
              query: getSearchTitle(submittedQuery, submittedHadImage),
              mode: 'catalog',
              categoryId: activeCategoryId,
              categoryLabel: activeCategory.label,
              submittedHadImage,
              products: result.products,
              totalProductCount: result.totalCount,
            })
          }

        }
      } catch (error) {
        if (requestId === requestIdRef.current) {
          setErrorMessage(
            error instanceof Error
              ? error.message
              : '상품을 불러오는 중 문제가 생겼습니다.',
          )
          setIsUnderstandingRequest(false)
          if (!append) {
            setProducts([])
            setTotalProductCount(null)
          }
        }
      } finally {
        if (catalogRequestRef.current === requestId) catalogRequestRef.current = null
        // 이전 요청의 finally가 새 요청의 로딩 표시까지 끄지 않도록 같은 번호인지 확인한다.
        if (requestId === requestIdRef.current) {
          setIsLoading(false)
        }
      }
    },
    [
      activeCategory.category2,
      activeCategory.category2Keyword,
      activeCategory.isAll,
      activeCategory.table,
      activeCategory.label,
      activeCategoryId,
      submittedHadImage,
      submittedQuery,
    ],
  )

  // 모델 검색 중·완료·실패 화면을 일반 카탈로그 조회로 덮어쓰지 않는다.
  useEffect(() => {
    if (baselineSearchResponse || submittedQuery || submittedHadImage || isUnderstandingRequest || errorMessage) {
      // 오류도 여기서 멈춰야 REST 조회가 오류를 지우고 일반 상품을 검색 성공처럼 보여주지 않는다.
      return
    }

    requestIdRef.current += 1
    const requestId = requestIdRef.current
    catalogReadyRef.current = false

    const loadTimer = window.setTimeout(() => {
      // effect 정리 시 아직 시작하지 않은 조회를 취소할 수 있도록 다음 작업으로 예약한다.
      loadProductPage(0, false, requestId)
    }, 0)

    return () => window.clearTimeout(loadTimer)
  }, [
    activeCategoryId,
    baselineSearchResponse,
    categoryTree,
    errorMessage,
    isUnderstandingRequest,
    loadProductPage,
    searchRunId,
    submittedHadImage,
    submittedQuery,
  ])

  useEffect(() => {
    const sentinel = loadMoreRef.current

    if (
      !sentinel ||
      isLoading ||
      errorMessage ||
      products.length === 0 ||
      !hasMoreProducts
    ) {
      return
    }

    // 목록 끝이 가까워지면 다음 카탈로그 페이지를 불러온다.
    // rootMargin 600px은 화면 끝에 도착하기 조금 전에 다음 페이지를 준비하는 여유 거리다.
    const requestId = requestIdRef.current
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          loadProductPage(products.length, true, requestId)
        }
      },
      { rootMargin: '600px 0px' },
    )

    observer.observe(sentinel)

    return () => observer.disconnect()
  }, [
    errorMessage,
    hasMoreProducts,
    isLoading,
    loadProductPage,
    products.length,
  ])

  useEffect(() => {
    return () => {
      if (selectedImage) {
        // 파일 미리보기용 브라우저 메모리를 이미지 교체/화면 종료 때 반환한다.
        URL.revokeObjectURL(selectedImage.previewUrl)
      }
    }
  }, [selectedImage])

  useEffect(
    () => () => {
      // 화면을 떠난 뒤 자동 스크롤이나 완료 알림 타이머가 실행되지 않도록 정리한다.
      if (searchCompleteDelayRef.current) {
        window.clearTimeout(searchCompleteDelayRef.current)
      }

      if (searchCompleteHideDelayRef.current) {
        window.clearTimeout(searchCompleteHideDelayRef.current)
      }

      sidebarAnchorRestoreTimeoutsRef.current.forEach((timeoutId) =>
        window.clearTimeout(timeoutId),
      )
    },
    [],
  )

  // 호출 횟수를 세지 않고 실제 경과 시간을 계산해 백그라운드 탭에서도 시간이 밀리지 않게 한다.
  useEffect(() => {
    if (!isUnderstandingRequest || searchStartedAt === null) return

    const elapsedInterval = window.setInterval(() => {
      setSearchElapsedSeconds(Math.floor((performance.now() - searchStartedAt) / 1000))
    }, 1000)

    // 완료·실패·화면 이동·컴포넌트 제거 시 타이머를 정리한다.
    return () => window.clearInterval(elapsedInterval)
  }, [isUnderstandingRequest, searchStartedAt])

  // 진행 문구는 타이머로 바뀌는 안내이며, 서버의 실제 단계별 진행률을 받는 것은 아니다.
  useEffect(() => {
    if (!isUnderstandingRequest) {
      return
    }

    const progressInterval = window.setInterval(() => {
      setSearchProgressIndex((currentIndex) => {
        if (currentIndex >= searchProgressSteps.length - 1) return currentIndex
        return currentIndex + 1
      })
    }, SEARCH_PROGRESS_INTERVAL_MS)

    return () => window.clearInterval(progressInterval)
  }, [isUnderstandingRequest, searchProgressSteps.length])

  useEffect(() => {
    if (!activeModal && !selectedProductForDetail && !isSidebarOpen && !isProviderMenuOpen) {
      return
    }

    function handleEscapeKey(event: KeyboardEvent) {
      // Escape는 열린 UI만 닫는다. 진행 중인 서버 검색을 취소하지는 않는다.
      if (event.key === 'Escape') {
        setActiveModal(null)
        setSelectedProductForDetail(null)
        setIsSidebarOpen(false)
        setIsProviderMenuOpen(false)
      }
    }

    window.addEventListener('keydown', handleEscapeKey)

    return () => window.removeEventListener('keydown', handleEscapeKey)
  }, [activeModal, selectedProductForDetail, isSidebarOpen, isProviderMenuOpen])

  // 열린 대화상자 안에서 Tab을 순환시키고 닫으면 원래 버튼으로 초점을 돌린다.
  useEffect(() => {
    if (!activeDialog) return
    const dialog = document.querySelector<HTMLElement>(`[data-f1t-dialog="${activeDialog}"]`)
    if (!dialog) return
    const previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null
    // 열기 직전 버튼과 body 스크롤 값을 저장해 닫힐 때 복구한다.
    const previousOverflow = document.body.style.overflow
    const getFocusableElements = () => Array.from(dialog.querySelectorAll<HTMLElement>(
      'button:not([disabled]), a[href], input:not([disabled]):not([type="hidden"]), textarea:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])',
    )).filter((element) => element.tabIndex >= 0 && element.getClientRects().length > 0)
    const focusFirst = () => (getFocusableElements()[0] ?? dialog).focus()

    function trapTab(event: KeyboardEvent) {
      // 메뉴가 펼쳐져 버튼 수가 달라질 수 있으므로 Tab을 누를 때마다 목록을 다시 구한다.
      if (event.key !== 'Tab') return
      const elements = getFocusableElements()
      const first = elements[0]
      const last = elements[elements.length - 1]
      if (!first) {
        event.preventDefault()
        dialog?.focus()
      } else if (!dialog?.contains(document.activeElement) || (event.shiftKey && document.activeElement === first)) {
        event.preventDefault()
        ;(event.shiftKey ? last : first).focus()
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault()
        first.focus()
      }
    }

    document.body.style.overflow = 'hidden'
    focusFirst()
    dialog.addEventListener('keydown', trapTab)
    return () => {
      dialog.removeEventListener('keydown', trapTab)
      document.body.style.overflow = previousOverflow
      if (previousFocus?.isConnected) previousFocus.focus()
    }
  }, [activeDialog])

  // 카테고리 토글 후 헤더가 화면에서 같은 위치에 오도록 scrollTop 보정
  useLayoutEffect(() => {
    const pending = pendingScrollAdjustRef.current
    if (!pending) return
    pendingScrollAdjustRef.current = null

    const headerEl = categoryGroupRefs.current.get(pending.groupId)
    const navEl = sidebarScrollRef.current
    if (!headerEl || !navEl) return

    sidebarAnchorRestoreTimeoutsRef.current.forEach((timeoutId) =>
      window.clearTimeout(timeoutId),
    )

    sidebarAnchorRestoreTimeoutsRef.current = [0, 40, 100, 180, 260].map(
      (delay) =>
        window.setTimeout(() => {
          const currentHeaderEl = categoryGroupRefs.current.get(pending.groupId)
          const currentNavEl = sidebarScrollRef.current

          if (!currentHeaderEl || !currentNavEl) {
            return
          }

          const topDiff =
            currentHeaderEl.getBoundingClientRect().top - pending.previousTop
          currentNavEl.scrollTop += topDiff
        }, delay),
    )
  }, [openCategoryIds])

  function toggleCategory(groupId: string) {
    // 펼치기 전 제목 위치를 저장하고, useLayoutEffect가 DOM 변경 뒤 스크롤 차이를 보정한다.
    const headerEl = categoryGroupRefs.current.get(groupId)

    if (headerEl) {
      pendingScrollAdjustRef.current = {
        groupId,
        previousTop: headerEl.getBoundingClientRect().top,
      }
    }

    setOpenCategoryIds((currentIds) => {
      const nextIds = new Set(currentIds)
      if (nextIds.has(groupId)) nextIds.delete(groupId)
      else nextIds.add(groupId)
      return nextIds
    })
  }

  // 화면 전환 때 진행 응답과 완료 알림을 함께 무효화한다. 서버 작업 자체를 중단하지는 않는다.
  function invalidateActiveRequest() {
    requestIdRef.current += 1
    catalogReadyRef.current = false
    if (searchCompleteDelayRef.current !== null) window.clearTimeout(searchCompleteDelayRef.current)
    if (searchCompleteHideDelayRef.current !== null) window.clearTimeout(searchCompleteHideDelayRef.current)
    searchCompleteDelayRef.current = null
    searchCompleteHideDelayRef.current = null
    return requestIdRef.current
  }

  function resetCatalog() {
    catalogPaginationRef.current = {}
    setProducts([])
    setTotalProductCount(null)
    setHasMoreProducts(false)
    setIsLoading(false)
  }

  function handleCategorySelect(categoryId: string) {
    if (categoryId === activeCategoryId && !hasSubmittedSearch && !isUnderstandingRequest && !errorMessage) return
    // 카테고리 탭은 새 카탈로그 탐색이다. 진행 중인 검색 결과가 다시 덮어쓰지 않게 무효화한다.
    invalidateActiveRequest()
    resetCatalog()
    setBaselineSearchResponse(null)
    setSubmittedQuery('')
    setSubmittedHadImage(false)
    setIsUnderstandingRequest(false)
    setIsSearchCompleteVisible(false)
    setErrorMessage('')
    setActiveCategoryId(categoryId)
  }

  function getSearchTitle(queryText: string, hadImage: boolean) {
    if (queryText) {
      return queryText
    }

    return hadImage ? '이미지로 검색' : ''
  }

  function needsReferenceImage(queryText: string) {
    // 단순한 정규식 검사다. 문장의 실제 의도 분석은 이 뒤의 백엔드가 담당한다.
    return imageReferenceQueryPattern.test(queryText.replace(/\s+/g, ' '))
  }

  // 최근 검색은 현재 React 메모리에 최대 5개 보관하며 새로고침 후에는 유지되지 않는다.
  function saveRecentSearchSnapshot(searchItem: RecentSearchItem) {
    if (!searchItem.query) {
      return
    }

    setRecentSearches((currentSearches) => [
      searchItem,
      ...currentSearches.filter((item) => item.query !== searchItem.query),
    ].slice(0, 5))
  }

  function restoreRecentSearch(searchItem: RecentSearchItem) {
    // 저장된 응답 스냅샷을 화면에 다시 넣는다. 같은 문장을 서버에 재요청하는 동작은 아니다.
    // 이미지 사용 여부만 저장하므로 이 동작에서 원본 File까지 되살리지는 않는다.
    setQuery(searchItem.query)
    setErrorMessage('')

    if (!searchItem.baselineSearchResponse && !searchItem.products) {
      setIsSidebarOpen(false)
      return
    }

    invalidateActiveRequest()
    setIsLoading(false)
    if (searchItem.categoryId) {
      setActiveCategoryId(searchItem.categoryId)
    }

    if (searchItem.baselineSearchResponse) {
      setBaselineSearchResponse(searchItem.baselineSearchResponse)
      setProducts([])
      setTotalProductCount(searchItem.baselineSearchResponse.results.length)
    } else if (searchItem.products) {
      setBaselineSearchResponse(null)
      setProducts(searchItem.products)
      setTotalProductCount(searchItem.totalProductCount ?? searchItem.products.length)
    } else {
      setBaselineSearchResponse(null)
    }

    setSubmittedQuery(searchItem.query)
    setSubmittedHadImage(Boolean(searchItem.submittedHadImage))
    setIsSidebarOpen(false)
    setIsUnderstandingRequest(false)
    setIsSearchCompleteVisible(false)
    resultsSectionRef.current?.scrollIntoView({
      behavior: 'smooth',
      block: 'start',
    })
  }

  function toggleLikedItem(itemId: string) {
    // Set을 새로 만들어 React가 변경을 감지하게 한다. 관심 상품은 서버에 저장하지 않는다.
    setLikedItemIds((currentIds) => {
      const nextIds = new Set(currentIds)

      if (nextIds.has(itemId)) {
        nextIds.delete(itemId)
      } else {
        nextIds.add(itemId)
      }

      return nextIds
    })
  }

  // 검색 실행의 중심: 입력 검증 → /search 호출 → 결과 상태 갱신 → 완료/오류 표시.
  async function runSubmittedSearch(
    queryText: string,
    imageFile: File | null,
    requestId: number,
  ) {
    const searchTitle = getSearchTitle(queryText, Boolean(imageFile))
    // 응답 대기 중의 화면 변화와 무관하게 검색 요청 당시 카테고리를 스냅샷에 남긴다.
    const searchCategoryId = activeCategoryId
    const searchCategoryLabel = activeCategory.label

    if (!queryText && !imageFile) {
      if (requestId !== requestIdRef.current) {
        return
      }

      setBaselineSearchResponse(null)
      setProducts([])
      setTotalProductCount(null)
      setErrorMessage('검색어를 입력하거나 이미지를 추가해 주세요.')
      setIsUnderstandingRequest(false)
      setIsSearchCompleteVisible(false)
      return
    }

    if (!hasBaselineSearchConfig()) {
      if (requestId !== requestIdRef.current) {
        return
      }

      setBaselineSearchResponse(null)
      setProducts([])
      setTotalProductCount(null)
      setErrorMessage(
        'Gemini 검색 API 주소가 설정되지 않았습니다. frontend/.env.local의 VITE_API_BASE_URL을 확인해 주세요.',
      )
      setIsUnderstandingRequest(false)
      setIsSearchCompleteVisible(false)
      return
    }

    try {
      // 선택한 카테고리는 서버의 후보 필터로, provider는 설명 생성 모델 선택으로 전달한다.
      const response = await searchBaseline({
        query: queryText,
        image: imageFile,
        topK: 10,
        table: activeCategory.isAll ? undefined : activeCategory.table,
        category2: activeCategory.isAll ? undefined : activeCategory.category2,
        category2Keyword: activeCategory.isAll
          ? undefined
          : activeCategory.category2Keyword,
        provider: selectedProvider,
      })

      if (requestId !== requestIdRef.current) {
        // 검색 도중 다른 검색/카테고리 이동으로 번호가 바뀌었다면 도착한 결과를 버린다.
        return
      }

      setSubmittedQuery(queryText)
      setSubmittedHadImage(Boolean(imageFile))
      setBaselineSearchResponse(response)
      // 검색 응답을 저장하면 렌더링이 카탈로그 대신 response.results를 선택한다.
      setProducts([])
      setTotalProductCount(response.results.length)
      setErrorMessage('')
      setIsUnderstandingRequest(false)
      setIsSearchCompleteVisible(true)
      saveRecentSearchSnapshot({
        id: `baseline-${Date.now()}`,
        query: searchTitle,
        mode: response.mode,
        categoryId: searchCategoryId,
        categoryLabel: searchCategoryLabel,
        submittedHadImage: Boolean(imageFile),
        baselineSearchResponse: response,
      })

      if (searchCompleteDelayRef.current) {
        window.clearTimeout(searchCompleteDelayRef.current)
      }

      if (searchCompleteHideDelayRef.current) {
        window.clearTimeout(searchCompleteHideDelayRef.current)
      }

      searchCompleteDelayRef.current = window.setTimeout(() => {
        resultsSectionRef.current?.scrollIntoView({
          behavior: 'smooth',
          block: 'start',
        })
      }, 900)

      searchCompleteHideDelayRef.current = window.setTimeout(() => {
        setIsSearchCompleteVisible(false)
      }, 1800)
    } catch (error) {
      if (requestId !== requestIdRef.current) {
        return
      }

      setSubmittedQuery(queryText)
      setSubmittedHadImage(Boolean(imageFile))
      setBaselineSearchResponse(null)
      setProducts([])
      setTotalProductCount(null)
      const message = error instanceof Error ? error.message : ''
      // CORS/네트워크 차단의 Failed to fetch는 서버 본문을 읽을 수 없어 일반 연결 오류로 바꾼다.
      setErrorMessage(
        message && !message.toLowerCase().includes('failed to fetch')
          ? message
          : 'Gemini 검색 서버에 연결하지 못했습니다.',
      )
      setIsUnderstandingRequest(false)
      setIsSearchCompleteVisible(false)
    } finally {
      if (requestId === requestIdRef.current) {
        setIsUnderstandingRequest(false)
      }
    }
  }

  // 폼 제출 시 입력을 고정하고 요청 번호를 갱신한 뒤 비동기 검색을 시작한다.
  function handleSubmit(event: React.SyntheticEvent<HTMLFormElement>) {
    event.preventDefault()
    const nextRequestId = invalidateActiveRequest()

    setIsSearchCompleteVisible(false)
    setIsUnderstandingRequest(true)
    setSearchStartedAt(performance.now())
    setSearchElapsedSeconds(0)
    setSearchProgressIndex(0)
    const queryText = query.trim()
    const imageFile = selectedImage?.file ?? null

    // 제출 시 이전 카드·오류를 비우고 입력을 고정한다. 타이핑 중에는 검색을 호출하지 않는다.
    setSubmittedQuery(queryText)
    setSubmittedHadImage(Boolean(imageFile))
    setBaselineSearchResponse(null)
    setProducts([])
    setTotalProductCount(null)
    setIsLoading(false)
    setErrorMessage('')

    if (queryText && !imageFile && needsReferenceImage(queryText)) {
      setBaselineSearchResponse(null)
      setProducts([])
      setTotalProductCount(null)
      setErrorMessage('이미지를 첨부하세요.')
      setIsUnderstandingRequest(false)
      return
    }

    void runSubmittedSearch(queryText, imageFile, nextRequestId)
  }

  // 미리보기 URL은 브라우저에서만 쓰고, 서버에는 원본 File을 전송한다.
  function setImageFile(file: File) {
    setSelectedImage((currentImage) => {
      if (currentImage) {
        URL.revokeObjectURL(currentImage.previewUrl)
      }

      return {
        file,
        previewUrl: URL.createObjectURL(file),
      }
    })
  }

  function handleImageChange(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]

    if (!file || !file.type.startsWith('image/')) {
      return
    }

    setImageFile(file)
  }

  function handleImageDragOver(event: React.DragEvent<HTMLDivElement>) {
    event.preventDefault()
    setIsImageDragging(true)
  }

  function handleImageDragLeave(event: React.DragEvent<HTMLDivElement>) {
    if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
      setIsImageDragging(false)
    }
  }

  function handleImageDrop(event: React.DragEvent<HTMLDivElement>) {
    // 기본 파일 열기를 막고 첫 이미지 파일을 입력창 선택과 같은 setImageFile 경로에 넣는다.
    event.preventDefault()
    setIsImageDragging(false)

    const file = event.dataTransfer.files?.[0]

    if (!file || !file.type.startsWith('image/')) {
      return
    }

    setImageFile(file)

    if (imageInputRef.current) {
      imageInputRef.current.value = ''
    }
  }

  function handleImageRemove(event: React.MouseEvent<HTMLButtonElement>) {
    event.stopPropagation()
    setSelectedImage((currentImage) => {
      if (currentImage) {
        URL.revokeObjectURL(currentImage.previewUrl)
      }

      return null
    })

    if (imageInputRef.current) {
      imageInputRef.current.value = ''
    }
  }

  function handleHomeClick() {
    // 검색 조건을 해제하고 searchRunId를 올려 기본 카탈로그 조회를 요청한다.
    invalidateActiveRequest()
    resetCatalog()
    setErrorMessage('')

    setActiveModal(null)
    setIsSidebarOpen(false)
    setActiveCategoryId(allCategory.id)
    setBaselineSearchResponse(null)
    setQuery('')
    setSubmittedQuery('')
    setSubmittedHadImage(false)
    setIsUnderstandingRequest(false)
    setIsSearchCompleteVisible(false)
    setSearchRunId((currentId) => currentId + 1)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  function handlePageClick(pageId: NavItemId) {
    // home은 목록 초기화, about은 소개 대화상자 열기로 분기한다.
    if (pageId === 'home') {
      handleHomeClick()
      return
    }

    // 소개 창은 검색 위에 열리는 UI다. 진행 요청이나 완료 표시 타이머는 그대로 유지한다.
    setActiveModal(pageId)
    setIsSidebarOpen(false)
  }

  // 카탈로그와 검색 응답을 같은 카드 형태로 맞추되, 검색 순위와 이미지 주소는 유지한다.
  const displayItems = isBaselineSearch
    ? baselineResults.map((result) => ({
        product: { ...result, image_url: getBaselineImageUrl(result), _sourceTable: result.source_table as ProductTable | undefined } as Product,
        imageUrl: getBaselineImageUrl(result),
        key: `search-${result.source_table ?? 'baseline'}-${result.id}`,
        rank: result.rank,
      }))
    : products.map((product, index) => ({
        product,
        imageUrl: getProductImageUrl(product),
        key: `${product._sourceTable}-${product.id}`,
        rank: index + 1,
      }))

  // 아래는 화면 구성: 사이드바 → 입력 폼 → 검색 메타데이터/상품 목록 → 상세창.
  const sidebarContent = (
    <div className="f1t-drawer-content">
      <div className="f1t-drawer-heading">
        <span className="f1t-wordmark">F1T<span>.</span></span>
        <button type="button" className="f1t-icon-button" onClick={() => setIsSidebarOpen(false)} aria-label="메뉴 닫기">×</button>
      </div>
      <button type="button" className="f1t-new-search" onClick={handleHomeClick}>새로운 스타일 찾기 <span aria-hidden="true">↗</span></button>
      <div ref={sidebarScrollRef} className="f1t-drawer-scroll">
        <section className="f1t-drawer-section">
          <div className="f1t-section-eyebrow">최근 검색 <span>이 페이지에서</span></div>
          {recentSearches.length === 0 ? (
            <p className="f1t-empty-history">아직 검색한 내용이 없어요.<br />마음에 드는 스타일을 찾아보세요.</p>
          ) : (
            <div className="f1t-history-list">
              {recentSearches.map((item) => (
                <button key={item.id} type="button" onClick={() => restoreRecentSearch(item)}>
                  <span>{item.query}</span><span aria-hidden="true">↗</span>
                </button>
              ))}
            </div>
          )}
        </section>
        <section className="f1t-drawer-section">
          <p className="f1t-section-eyebrow">카테고리</p>
          <button type="button" className="f1t-category-all" onClick={() => { handleCategorySelect(allCategory.id); setIsSidebarOpen(false) }}>전체 상품</button>
          {categoryTree.map((group) => (
            <div key={group.id} ref={(element) => { if (element) categoryGroupRefs.current.set(group.id, element); else categoryGroupRefs.current.delete(group.id) }}>
              <div className="f1t-drawer-category">
                <button type="button" onClick={() => { handleCategorySelect(group.id); setIsSidebarOpen(false) }}>{group.label}</button>
                <button type="button" onClick={() => toggleCategory(group.id)} aria-expanded={openCategoryIds.has(group.id)} aria-label={`${group.label} 세부 카테고리`}><span aria-hidden="true">{openCategoryIds.has(group.id) ? '−' : '+'}</span></button>
              </div>
              {openCategoryIds.has(group.id) && (
                <div className="f1t-subcategories">
                  {group.children.length ? group.children.map((child) => (
                    <button key={child.category2} type="button" onClick={() => { handleCategorySelect(`${group.id}-${child.category2}`); setIsSidebarOpen(false) }}>{child.category2}</button>
                  )) : <p>세부 카테고리가 없습니다.</p>}
                </div>
              )}
            </div>
          ))}
        </section>
      </div>
    </div>
  )

  return (
    <main className="f1t-app">
      <header className="f1t-header">
        <div className="f1t-header-inner">
          <button type="button" onClick={handleHomeClick} className="f1t-brand" aria-label="F1T 홈">
            <span className="f1t-wordmark">F1T<span>.</span></span>
            <span className="f1t-brand-caption">FASHION INTENTION TRANSLATOR</span>
          </button>
          <div className="f1t-header-actions">
            {pageNavItems.filter((item) => item.id === 'about').map((item) => <button key={item.id} type="button" className="f1t-about-button" onClick={() => handlePageClick(item.id)}>F1T 소개</button>)}
            <button type="button" className="f1t-menu-button" onClick={() => setIsSidebarOpen(true)} aria-label="최근 검색과 전체 카테고리 열기" aria-expanded={isSidebarOpen}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true"><path d="M4 7h16M4 12h16M4 17h16" /></svg>
            </button>
          </div>
        </div>
      </header>

      {isSidebarOpen && (
        <div className="f1t-drawer-layer">
          <button type="button" className="f1t-drawer-backdrop" aria-label="메뉴 닫기" onClick={() => setIsSidebarOpen(false)} />
          <aside className="f1t-drawer" role="dialog" aria-modal="true" aria-label="최근 검색과 카테고리" data-f1t-dialog="drawer" tabIndex={-1}>{sidebarContent}</aside>
        </div>
      )}

      <div className="f1t-content">
        <section className="f1t-search-section" aria-labelledby="search-heading">
          <div className="f1t-search-intro">
            <p className="f1t-eyebrow"><span /> YOUR NEXT FIND</p>
            <h1 id="search-heading">찾는 옷을,<br />말해보세요.</h1>
            <p>색감부터 분위기까지.<br className="f1t-desktop-break" /> 머릿속 스타일을 상품으로 만나보세요.</p>
          </div>
          <div className="f1t-search-workspace">
            <form id="fashion-search" onSubmit={handleSubmit} className={`f1t-search-form ${isImageDragging ? 'is-dragging' : ''}`}>
              <div className="f1t-search-main">
                <input ref={imageInputRef} type="file" accept="image/*" hidden onChange={handleImageChange} />
                <div className="f1t-upload-wrap" onDragOver={handleImageDragOver} onDragLeave={handleImageDragLeave} onDrop={handleImageDrop}>
                  <button type="button" className={`f1t-upload ${selectedImage ? 'has-image' : ''}`} onClick={() => imageInputRef.current?.click()} aria-label={selectedImage ? '참고 이미지 바꾸기' : '참고 이미지 추가'}>
                    {selectedImage ? <img src={selectedImage.previewUrl} alt="첨부한 참고 이미지" /> : <>
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.25" aria-hidden="true"><rect x="3" y="3" width="18" height="18" rx="2" /><circle cx="8" cy="8" r="1.5" /><path d="m4 17 5-5 4 4 3-3 5 5" /></svg>
                      <span>사진 추가</span>
                    </>}
                  </button>
                  {selectedImage && <button type="button" className="f1t-remove-image" onClick={handleImageRemove} aria-label="참고 이미지 삭제">×</button>}
                </div>
                <div className="f1t-query-field">
                  <label htmlFor="fashion-query">어떤 옷을 찾고 있나요?</label>
                  <textarea id="fashion-query" value={query} onChange={(event) => setQuery(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); if (!isUnderstandingRequest && !isSearchCompleteVisible) event.currentTarget.form?.requestSubmit() } }} placeholder="예: 출근할 때 입을 단정한 검은색 셔츠" rows={2} />
                </div>
              </div>
              <div className="f1t-search-bottom">
                <span className="f1t-search-hint">사진을 더하면 원하는 느낌에 더 가까이</span>
                <button type="submit" className="f1t-search-button" disabled={isUnderstandingRequest || isSearchCompleteVisible}>{isUnderstandingRequest ? '찾고 있어요' : isSearchCompleteVisible ? '검색 완료' : '스타일 찾기'}<span aria-hidden="true">↗</span></button>
              </div>
            </form>

            <div className="f1t-search-options">
              <span>이렇게 찾아보세요</span>
              <button type="button" onClick={() => setQuery('러닝할 때 입기 좋은 가벼운 바지 추천해줘')}>가벼운 러닝 팬츠</button>
              <button type="button" onClick={() => setQuery(searchExamples[2].query)}>미니멀한 원피스</button>
              <button type="button" className="f1t-advanced-toggle" onClick={() => setIsProviderMenuOpen((open) => !open)} aria-expanded={isProviderMenuOpen}>고급 설정 <span aria-hidden="true">{isProviderMenuOpen ? '−' : '+'}</span></button>
            </div>
            {isProviderMenuOpen && (
              <div className="f1t-advanced-panel">
                {/* 모델 선택 */}
                <label htmlFor="description-provider">요청 해석 모델</label>
                <select id="description-provider" value={selectedProvider} onChange={(event) => setSelectedProvider(event.target.value)}>
                  {['gemini', 'openai', 'qwen'].map((provider) => <option key={provider} value={provider}>{providerModels[provider] ?? provider}</option>)}
                </select>
              </div>
            )}
            {(isUnderstandingRequest || isSearchCompleteVisible) && (
              <div className="f1t-search-status">
                <span className={isUnderstandingRequest ? 'f1t-status-dot is-loading' : 'f1t-status-dot'} aria-hidden="true" />
                <div className="f1t-search-status-copy">
                  <div className="f1t-search-status-heading">
                    <strong role="status" aria-live="polite">{isSearchCompleteVisible ? '검색이 완료됐어요' : currentSearchProgress.title}</strong>
                    {isUnderstandingRequest && <span className="f1t-search-elapsed" role="timer" aria-label="검색 경과 시간" aria-live="off"><span aria-hidden="true">· </span>{searchElapsedSeconds}초</span>}
                  </div>
                  <p>{isSearchCompleteVisible ? '아래에서 검색 결과를 확인해보세요.' : currentSearchProgress.description}</p>
                </div>
              </div>
            )}
          </div>
        </section>

        <nav className="f1t-category-tabs" aria-label="상품 카테고리">
          {[allCategory, ...categoryGroups].map((category) => <button key={category.id} type="button" className={activeCategoryId === category.id ? 'is-active' : ''} aria-current={activeCategoryId === category.id ? 'page' : undefined} onClick={() => handleCategorySelect(category.id)}>{category.label}</button>)}
          <button type="button" className="f1t-all-categories" onClick={() => setIsSidebarOpen(true)}>세부 카테고리 <span aria-hidden="true">+</span></button>
        </nav>

        <section ref={resultsSectionRef} className="f1t-results" aria-labelledby="results-heading">
          <div className="f1t-results-heading">
            {/* 상품 수와 실제 검색 소요 시간을 보여주고, 보조 기기에는 목록 제목을 제공한다. */}
            <h2 id="results-heading" className="f1t-sr-only">{hasSubmittedSearch ? '검색 결과' : '상품 목록'}</h2>
            <span className="f1t-result-count">{isBaselineSearch ? `${baselineResults.length.toLocaleString('ko-KR')}개 상품` : totalProductCount !== null ? `${totalProductCount.toLocaleString('ko-KR')}개 상품` : isLoading ? '불러오는 중' : ''}</span>
          </div>

          {isBaselineSearch && (elapsedMs !== undefined || searchTimings) && (
            <details className="f1t-search-timing">
              <summary>
                <span><strong>{((elapsedMs ?? searchTimings!.total_ms) / 1000).toFixed(2)}초</strong> 만에 검색 완료</span>
                <span className="f1t-timing-toggle">단계별 시간 <span aria-hidden="true">+</span></span>
              </summary>
              <div className="f1t-timing-body">
                {searchTimings ? <>
                  <p className="f1t-timing-note">서버 처리 {(searchTimings.total_ms / 1000).toFixed(2)}초 · 전체 시간에는 서버 준비와 네트워크 왕복이 포함됩니다.</p>
                  <dl className="f1t-timing-stages">
                    {searchTimings.stages.map((stage) => <div key={stage.id}>
                      <dt>{stage.label}{stage.parallel_group && <span className="f1t-timing-parallel">{stage.parallel_group === 'analysis' ? '1차 묶음 · 동시 실행' : stage.parallel_group === 'encoding' ? '2차 묶음 · 동시 실행' : '동시 실행'}</span>}</dt>
                      <dd>{(stage.duration_ms / 1000).toFixed(2)}초</dd>
                    </div>)}
                  </dl>
                  <p className="f1t-timing-note">1차 묶음(의도 추출·목표 설명 생성)이 끝나면 검색 조건을 구성하고, 2차 묶음(설명 임베딩·소재 조건이 있을 때 소재 임베딩)을 실행합니다. 각 묶음 안에서만 동시에 진행하므로 단계별 시간의 합은 전체 시간과 다릅니다.</p>
                </> : <p className="f1t-timing-note">전체 응답 시간입니다. 이 검색의 단계별 시간 정보는 제공되지 않았어요.</p>}
              </div>
            </details>
          )}

          {requestSummaryItems.length > 0 && <div className="f1t-request-summary">{requestSummaryItems.map((item) => <span key={item}>{item}</span>)}</div>}

          {isBaselineSearch && (baselineSearchResponse?.recommendation_reason || parsedAttributeItems.length > 0 || searchInterpretationLimited) && (
            <details className="f1t-search-details">
              <summary>검색 조건과 추천 이유 살펴보기 <span aria-hidden="true">+</span></summary>
              <div className="f1t-details-body">
                {baselineSearchResponse?.recommendation_reason && <p className="f1t-recommendation-copy">{baselineSearchResponse.recommendation_reason}</p>}
                {parsedAttributeItems.length > 0 && <div className="f1t-attribute-list">{parsedAttributeItems.map((item) => <span key={item.key}><b>{item.label}</b> {item.value}</span>)}</div>}
                {searchInterpretationLimited && <p className="f1t-detail-copy" role="status">요청을 해석하는 일부 기능을 이용하지 못해 대체 방식으로 검색했어요. 원하는 조건이 결과에 반영됐는지 확인해주세요.</p>}
              </div>
            </details>
          )}

          {errorMessage && !isBaselineSearch && (
            <div className="f1t-error" role="alert">
              <strong>{hasSubmittedSearch ? '검색 요청을 처리하지 못했어요.' : '상품을 불러오지 못했어요.'}</strong>
              <p>{errorMessage}</p>
              {hasSubmittedSearch ? (
                <button type="submit" form="fashion-search" disabled={isUnderstandingRequest}>다시 검색 <span aria-hidden="true">↗</span></button>
              ) : (
                <button type="button" onClick={() => { setErrorMessage(''); setSearchRunId((currentId) => currentId + 1) }}>다시 시도 <span aria-hidden="true">↗</span></button>
              )}
            </div>
          )}

          {!isBaselineSearch && isLoading && products.length === 0 && <div className="f1t-product-grid" aria-label="상품 불러오는 중">{Array.from({ length: 12 }).map((_, index) => <div className="f1t-product-skeleton" key={index}><div /><span /><span /></div>)}</div>}

          {displayItems.length > 0 && <div className={`f1t-product-grid ${isLoading && !isBaselineSearch ? 'is-refreshing' : ''}`}>
            {displayItems.map(({ product, imageUrl, key, rank }) => {
              const thumbnailUrl = isBaselineSearch ? getSupabaseImageThumbnailUrl(imageUrl) : getProductThumbnailUrl(product)
              const liked = likedItemIds.has(key)
              return <article className="f1t-product-card" key={key}>
                <button type="button" className="f1t-product-open" onClick={() => setSelectedProductForDetail(product)} aria-label={`${product.name ?? '상품'} 상세보기`}>
                  <div className="f1t-product-image">
                    {imageUrl ? <img src={thumbnailUrl || imageUrl} alt={product.name ?? '상품 이미지'} data-fallback-src={imageUrl} onError={handleImageFallback} loading={rank <= 4 ? 'eager' : 'lazy'} fetchPriority={rank <= 4 ? 'high' : 'low'} decoding="async" /> : <span className="f1t-missing-image">이미지 준비 중</span>}
                    {isBaselineSearch && <span className="f1t-product-rank">{String(rank).padStart(2, '0')}</span>}
                  </div>
                  <div className="f1t-product-info">
                    {product.brand && <p className="f1t-product-brand">{product.brand}</p>}
                    <h3>{product.name ?? product.id}</h3>
                    {formatPrice(product.price) && <p className="f1t-product-price">{formatPrice(product.price)}</p>}
                  </div>
                </button>
                <button type="button" className={`f1t-like ${liked ? 'is-liked' : ''}`} aria-label={liked ? '관심 상품 해제' : '관심 상품 선택'} aria-pressed={liked} onClick={(event) => { event.stopPropagation(); toggleLikedItem(key) }}>
                  <svg viewBox="0 0 24 24" fill={liked ? 'currentColor' : 'none'} stroke="currentColor" strokeWidth="1.4" aria-hidden="true"><path d="M20.8 4.6a5.5 5.5 0 0 0-7.8 0L12 5.7l-1.1-1.1a5.5 5.5 0 0 0-7.8 7.8L12 21l8.8-8.6a5.5 5.5 0 0 0 0-7.8Z" /></svg>
                </button>
              </article>
            })}
          </div>}

          {!isLoading && !errorMessage && displayItems.length === 0 && !isUnderstandingRequest && <div className="f1t-empty-results"><span aria-hidden="true">↗</span><h3>{hasSubmittedSearch ? '조금 다른 조건으로 찾아볼까요?' : '아직 표시할 상품이 없어요.'}</h3><p>{hasSubmittedSearch ? '조건을 줄이거나 다른 표현으로 검색해보세요.' : '카테고리를 선택하거나 찾는 스타일을 입력해보세요.'}</p></div>}

          {!isBaselineSearch && products.length > 0 && <div ref={loadMoreRef} className="f1t-load-more" role="status">{isLoading ? '상품을 더 불러오고 있어요' : hasMoreProducts ? '' : '모든 상품을 살펴봤어요'}</div>}
        </section>

        <footer className="f1t-footer"><span className="f1t-wordmark">F1T<span>.</span></span><p>취향을 말하면, 스타일이 보입니다.</p><button type="button" onClick={() => handlePageClick('about')}>서비스 소개 ↗</button></footer>
      </div>

      {activeModal && <PolicyModal onClose={() => setActiveModal(null)} />}

      {selectedProductForDetail && (
        <div className="f1t-product-modal-backdrop" onClick={() => setSelectedProductForDetail(null)}>
          <section className="f1t-product-modal" role="dialog" aria-modal="true" aria-labelledby="product-detail-title" data-f1t-dialog="product" tabIndex={-1} onClick={(event) => event.stopPropagation()}>
            <button type="button" className="f1t-modal-close" onClick={() => setSelectedProductForDetail(null)} aria-label="상품 상세 닫기">×</button>
            <div className="f1t-modal-image"><img src={getProductImageUrl(selectedProductForDetail)} alt={selectedProductForDetail.name ?? '상품 이미지'} /></div>
            <div className="f1t-modal-info">
              {selectedProductForDetail.brand && <p className="f1t-eyebrow">{selectedProductForDetail.brand}</p>}
              <h2 id="product-detail-title">{selectedProductForDetail.name ?? selectedProductForDetail.id}</h2>
              {formatPrice(selectedProductForDetail.price) && <p className="f1t-modal-price">{formatPrice(selectedProductForDetail.price)}</p>}
              <dl>{[['카테고리', selectedProductForDetail.category2 || selectedProductForDetail.category1], ['색상', selectedProductForDetail.color], ['소매', selectedProductForDetail.sleeve], ['기장', selectedProductForDetail.length], ['핏', selectedProductForDetail.fit], ['소재', selectedProductForDetail.fabric]].filter(([, value]) => value).map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>
              <a className="f1t-product-link" href={`https://www.musinsa.com/products/${selectedProductForDetail.id}`} target="_blank" rel="noopener noreferrer">상품 보러 가기 <span aria-hidden="true">↗</span></a>
            </div>
          </section>
        </div>
      )}
    </main>
  )
}

export default App
