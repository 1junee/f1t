export type ProductTable =
  | 'musinsa_pants'
  | 'musinsa_top_clothes'
  | 'musinsa_skirt_dress'

// 카탈로그 카드와 상세창의 공통 형태다. _sourceTable은 조회 후 프론트에서 덧붙인다.
// ?와 null을 허용하는 필드는 DB에 값이 없을 수 있으므로 화면에서 표시 여부를 확인한다.
export type Product = {
  id: string | number
  name?: string | null
  brand?: string | null
  image_url?: string | null
  category1?: string | null
  category2?: string | null
  price?: number | string | null
  color?: string | null
  sleeve?: string | null
  length?: string | null
  fit?: string | null
  fabric?: string | null
  _sourceTable?: ProductTable
}
