from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable


# 검색 조건 정리 → DB 후보 조회 → 검색 결과 후처리를 담당한다.
# VLM 호출과 벡터 점수 계산은 각각 intent 모듈과 recommendation_service에 있다.
VALID_TABLES = {
    "musinsa_pants",
    "musinsa_top_clothes",
    "musinsa_skirt_dress",
}

# 카테고리 → 테이블 라우팅
TABLE_ROUTING_MAP = {
    "top": "musinsa_top_clothes",
    "pants": "musinsa_pants",
    "skirt": "musinsa_skirt_dress",
    "dress": "musinsa_skirt_dress",
    "one-piece": "musinsa_skirt_dress",
}

# 영어 속성값 → DB 한글값 변환
DB_TRANSLATION_MAP = {
    "sex": {"male": "남성", "female": "여성", "unisex": ["남성", "여성"]},
    "fit": {
        "slim": "슬림",
        "regular": "레귤러",
        "loose": "루즈",
        "oversized": "오버사이즈",
        "wide": ["루즈", "오버사이즈"],
    },
    "stretch": {"no": "없음", "yes": "있음"},
    "thickness": {
        "thin": ["얇음", "약간 얇음"],
        "medium": "보통",
        "thick": ["두꺼움", "약간두꺼움", "약간 두꺼움"],
    },
    "season": {"spring": "봄", "summer": "여름", "fall": "가을", "winter": "겨울"},
}

ARRAY_COLUMNS = ["sex", "fit", "stretch", "thickness", "season"]
TEXT_COLUMNS = ["sleeve", "length"]

# 값이 명확한 속성만 필터로 쓰고, 색상·소재는 이 목록에서 제외한다.
EXACT_FILTER_FIELDS = (
    "sleeve",
    "length",
    "category1",
    "sex",
    "season",
    "stretch",
    "thickness",
    "fit",
)
# 문자열 일치 보너스는 현재 비활성화돼 있다. 소재 벡터 점수는 별도 경로에서 계산한다.
SOFT_TEXT_RERANK_FIELDS = ()
INTENT_EXTRACTION_STATUS = "gemini_intent_analysis"
# 호환용 /retrieve·기존 방식 벤치마크의 REST 상한이다. 운영 /search는 filtered RPC를 쓴다.
METADATA_PREFETCH_LIMIT = 1000
SUPABASE_REST_TIMEOUT_SECONDS = 8

# 여러 종류가 언급되면 아래 규칙 순서에서 처음 찾은 카테고리 하나를 선택한다.
CATEGORY_RULES = (
    ("one-piece", "musinsa_skirt_dress", "원피스", ("원피스", "드레스", "one piece", "one-piece")),
    ("skirt", "musinsa_skirt_dress", "스커트", ("스커트", "치마", "skirt")),
    ("pants", "musinsa_pants", None, ("바지", "팬츠", "청바지", "데님", "슬랙스", "조거", "쇼츠", "반바지", "pants", "jeans")),
    ("top", "musinsa_top_clothes", None, ("상의", "티셔츠", "티 ", "셔츠", "후드", "후드티", "맨투맨", "니트", "가디건", "블라우스", "top", "shirt", "hoodie")),
)

ATTRIBUTE_PATTERNS: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "sleeve": (
        ("sleeveless", ("민소매", "나시", "sleeveless")),
        ("long", ("긴팔", "긴 소매", "소매가 긴", "롱슬리브", "long sleeve", "long-sleeve")),
        ("short", ("반팔", "short sleeve", "short-sleeve")),
    ),
    "length": (
        ("half", ("하프", "미디", "중간 기장", "half", "midi")),
        ("long", ("긴 기장", "롱", "맥시", "long", "maxi")),
        ("short", ("짧은 기장", "숏", "미니", "short", "mini")),
    ),
    "sex": (
        ("unisex", ("유니섹스", "남녀공용", "공용", "unisex")),
        ("male", ("남성", "남자", "남성용", "male", "men")),
        ("female", ("여성", "여자", "여성용", "female", "women")),
    ),
    "season": (
        ("spring", ("봄", "spring")),
        ("summer", ("여름", "summer")),
        ("fall", ("가을", "autumn", "fall")),
        ("winter", ("겨울", "winter")),
    ),
    "stretch": (
        ("no", ("신축성 없는", "늘어나지 않는", "non-stretch", "no stretch")),
        ("yes", ("신축성", "스판", "스트레치", "stretch")),
    ),
    "thickness": (
        ("thin", ("얇은", "가벼운", "thin", "lightweight")),
        ("medium", ("중간 두께", "보통 두께", "medium")),
        ("thick", ("두꺼운", "도톰한", "thick", "heavyweight")),
    ),
    "fit": (
        ("oversized", ("오버사이즈", "오버핏", "oversized", "over fit")),
        ("wide", ("와이드 핏", "와이드핏", "wide fit", "wide-fit")),
        ("loose", ("루즈핏", "여유로운", "loose")),
        ("regular", ("레귤러핏", "정핏", "regular")),
        ("slim", ("슬림핏", "슬림", "slim")),
    ),
}

COLOR_TERMS = (
    "black", "white", "ivory", "gray", "grey", "charcoal", "navy", "blue",
    "denim", "brown", "beige", "cream", "red", "pink", "purple", "green",
    "khaki", "olive", "yellow", "orange", "블랙", "검정", "검은", "화이트",
    "흰색", "아이보리", "회색", "그레이", "차콜", "네이비", "파랑", "블루",
    "데님", "브라운", "갈색", "베이지", "크림", "빨강", "레드", "핑크",
    "보라", "퍼플", "초록", "그린", "카키", "올리브", "노랑", "옐로우",
    "오렌지",
)
FABRIC_TERMS = (
    "cotton", "linen", "denim", "wool", "polyester", "nylon", "leather",
    "suede", "corduroy", "knit", "fleece", "jersey", "mesh", "코튼",
    "면 소재", "면 원단", "순면", "면티", "면 티셔츠", "면 셔츠", "린넨",
    "데님", "울", "양모", "폴리", "폴리에스터", "나일론", "가죽", "레더",
    "스웨이드", "코듀로이", "니트", "플리스", "저지", "메쉬",
)
ABSTRACT_TPO_TERMS = (
    "tpo", "면접", "출근", "회사", "오피스", "비즈니스", "하객", "결혼식",
    "데이트", "소개팅", "여행", "휴가", "캠핑", "운동", "러닝", "헬스",
    "등산", "학교", "졸업식", "입학식", "장례식", "파티", "클럽", "격식",
    "단정", "꾸안꾸", "interview", "office", "business", "wedding",
    "date", "travel", "workout", "running", "formal", "casual",
)

VALUE_VARIANTS: dict[str, dict[str, tuple[str, ...]]] = {
    field: dict(values)
    for field, values in ATTRIBUTE_PATTERNS.items()
}
DB_EXACT_MATCH_VARIANTS = {
    "sleeve": {
        "long": ("long",),
        "short": ("short",),
        "sleeveless": ("sleeveless",),
    },
    "sex": {
        "male": ("남성",),
        "female": ("여성",),
        "unisex": ("남성", "여성"),
    },
    "fit": {
        "slim": ("슬림",),
        "regular": ("레귤러",),
        "loose": ("루즈",),
        "oversized": ("오버사이즈",),
        "wide": ("루즈", "오버사이즈"),
    },
    "stretch": {
        "no": ("없음", "거의 없음"),
        "yes": ("있음", "약간 있음"),
    },
    "thickness": {
        "thin": ("얇음", "약간 얇음"),
        "medium": ("보통",),
        "thick": ("두꺼움", "약간두꺼움", "약간 두꺼움"),
    },
    "season": {
        "spring": ("봄",),
        "summer": ("여름",),
        "fall": ("가을",),
        "winter": ("겨울",),
    },
}
# 후처리에서는 사용자 표현과 DB 한글 표기를 같은 속성의 동의어로 비교한다.
for field, values in DB_EXACT_MATCH_VARIANTS.items():
    for value, variants in values.items():
        VALUE_VARIANTS.setdefault(field, {})[value] = (
            *VALUE_VARIANTS.get(field, {}).get(value, (value,)),
            *variants,
        )


@dataclass(frozen=True)
class ParsedTextAttributes:
    # None은 조건을 찾지 못했다는 뜻이며, 해당 속성을 임의로 제한하지 않는다.
    sleeve: str | None = None
    length: str | None = None
    color: str | None = None
    category1: str | None = None
    sex: str | None = None
    season: str | None = None
    stretch: str | None = None
    thickness: str | None = None
    fit: str | None = None
    fabric: str | None = None

    def to_dict(self) -> dict[str, str]:
        # 비어 있는 필드는 제외해 필터에 실제 요청 조건만 전달한다.
        return {
            key: value
            for key, value in self.__dict__.items()
            if value is not None
        }

    def with_missing_from(self, fallback: "ParsedTextAttributes") -> "ParsedTextAttributes":
        # VLM이 비운 값은 키워드 파서로 채운다. 명시된 wide 핏은 예외적으로 우선한다.
        return ParsedTextAttributes(
            **{
                field: (
                    getattr(fallback, field)
                    if field == "fit" and getattr(fallback, field) == "wide"
                    else getattr(self, field) or getattr(fallback, field)
                )
                for field in self.__dataclass_fields__
            }
        )


@dataclass(frozen=True)
class TextAnalysis:
    # 규칙으로 읽은 검색 속성과 요청 유형을 보관한다.
    structured_attributes: ParsedTextAttributes
    intent_type: str
    lane_metadata: dict[str, Any]

    @property
    def parsed_attributes(self) -> ParsedTextAttributes:
        return self.structured_attributes


@dataclass(frozen=True)
class AttributeTextSearchPlan:
    # 검색 계획을 담는 자료형이다. 이 객체를 만드는 것만으로 DB를 조회하지 않는다.
    parsed_attributes: ParsedTextAttributes
    has_image: bool
    table_filter: str | None
    category2_filter: str | None
    category2_keyword_filter: str | None
    prefetch_count: int
    user_filter_overrides: dict[str, str]
    intent_type: str
    analysis_lanes: dict[str, Any]
    extraction_info: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        # 응답에서 검색 과정을 설명하기 위한 메타데이터이며 검색 실행 함수는 아니다.
        exact_fields = [
            field
            for field in EXACT_FILTER_FIELDS
            if getattr(self.parsed_attributes, field) is not None
        ]
        semantic_fields = [
            field
            for field in SOFT_TEXT_RERANK_FIELDS
            if getattr(self.parsed_attributes, field) is not None
        ]

        return {
            "query_type": "text_image" if self.has_image else "text_only",
            "id": (
                "intent_text_image_table_narrowing"
                if self.has_image
                else "intent_text_table_narrowing"
            ),
            "label": (
                "보수적 의도 추출 기반 텍스트+이미지 파이프라인"
                if self.has_image
                else "보수적 의도 추출 기반 텍스트 파이프라인"
            ),
            "baseline": False,
            "intent_type": self.intent_type,
            "retrieval_strategy": (
                "conservative_metadata_narrowing_then_target_description_gemini_embedding_search"
            ),
            "parsed_attributes": self.parsed_attributes.to_dict(),
            "analysis_lanes": self.analysis_lanes,
            "intent_extraction": {
                "status": (self.extraction_info or {}).get("status", INTENT_EXTRACTION_STATUS),
                "method": (self.extraction_info or {}).get("method", "rule_based_keyword_parser"),
                "prompt_id": (self.extraction_info or {}).get("prompt_id"),
                "prompt": None,
                "notes": [
                    "Conservative intent extraction retains only explicitly mentioned attributes.",
                    "Text+image requests use image evidence only when the query explicitly asks for a visible image attribute.",
                ],
            },
            "exact_filter_fields": exact_fields,
            "soft_text_rerank_fields": semantic_fields,
            "applied_filters": {
                key: value
                for key, value in {
                    "table": self.table_filter,
                    "category2": self.category2_filter,
                    "category2_keyword": self.category2_keyword_filter,
                }.items()
                if value
            },
            "user_filter_overrides": self.user_filter_overrides,
            "prefetch_count": self.prefetch_count,
            "notes": [
                "Metadata extraction uses the conservative prompt from the evaluation experiment, with rule-based parsing only as support metadata.",
                "category1 is used to narrow the Supabase table before vector search only when explicit or conservatively extracted.",
                "Exact metadata filters and cosine ranking run inside the filtered RPC over all matching products.",
                "The filtered RPC returns only the requested top results; fallback vector search uses a wider candidate set.",
                "Color stays in the target description; an explicit fabric attribute is embedded separately for weighted similarity.",
                "Detail terms are handled by the target-description generation path, not as database columns.",
                "Abstract TPO queries are marked but not hard-filtered unless explicit attributes are present.",
            ],
        }


def parse_text_attributes(query: str) -> ParsedTextAttributes:
    # 기존 호출부에는 분석 전체 대신 구조화된 속성만 돌려준다.
    return analyze_text_query(query).parsed_attributes


def analyze_text_query(query: str) -> TextAnalysis:
    # 외부 모델 호출 없이 속성 키워드와 요청 유형을 분석한다.
    normalized = _normalize_query(query)
    structured_attributes = _parse_structured_attributes(normalized)

    intent_type = _detect_intent_type(normalized, structured_attributes)
    structured_fields = structured_attributes.to_dict()
    lane_metadata = {
        "structured_attributes": {
            "status": "filled" if structured_fields else "empty",
            "fields": sorted(structured_fields),
            "implementation": INTENT_EXTRACTION_STATUS,
            "scope": (
                "sleeve, length, category1, sex, season, stretch, thickness, "
                "fit, color, fabric"
            ),
        },
        "intent": {
            "type": intent_type,
            "tpo_terms_detected": _matched_terms(normalized, ABSTRACT_TPO_TERMS),
        },
    }

    return TextAnalysis(
        structured_attributes=structured_attributes,
        intent_type=intent_type,
        lane_metadata=lane_metadata,
    )


def build_attribute_text_search_plan(
    query: str,
    top_k: int,
    has_image: bool = False,
    extracted_attributes: ParsedTextAttributes | None = None,
    extraction_info: dict[str, Any] | None = None,
    table_filter: str | None = None,
    category2_filter: str | None = None,
    category2_keyword_filter: str | None = None,
) -> AttributeTextSearchPlan:

    # 성공한 VLM 결과는 빈 속성까지 존중한다. 규칙으로 제외된 조건을 되살리지 않는다.
    analysis = analyze_text_query(query)
    parsed = (
        extracted_attributes
        if extracted_attributes is not None
        and (extraction_info or {}).get("status", "ok") == "ok"
        else analysis.parsed_attributes
    )
    # 실패·미사용 시의 규칙 결과도 parsed에 들어 있으므로 여기서 다시 원문을 해석하지 않는다.
    category = parsed.category1
    inferred_table, inferred_category2_keyword = _route_category(category)
    user_overrides = {
        key: value
        for key, value in {
            "table": table_filter,
            "category2": category2_filter,
            "category2_keyword": category2_keyword_filter,
        }.items()
        if value
    }

    # 화면에서 선택한 카테고리처럼 요청에 직접 전달된 필터를 추론값보다 우선한다.
    effective_table = table_filter or inferred_table
    if effective_table not in VALID_TABLES:
        effective_table = table_filter

    # 명시적으로 범위를 선택했다면 원문의 추론 키워드로 다시 좁히지 않는다.
    effective_category2_keyword = category2_keyword_filter
    if effective_category2_keyword is None and table_filter is None and category2_filter is None:
        effective_category2_keyword = inferred_category2_keyword
    # 후처리할 여유 후보를 요청한다. 실제 RPC 반환 수에는 SQL의 별도 상한이 있다.
    has_parsed_attributes = bool(parsed.to_dict() or category)
    if has_parsed_attributes:
        prefetch_count = max(top_k, METADATA_PREFETCH_LIMIT)
    elif analysis.intent_type == "abstract_tpo" and not effective_table:
        prefetch_count = top_k
    else:
        prefetch_count = top_k

    return AttributeTextSearchPlan(
        parsed_attributes=parsed,
        has_image=has_image,
        table_filter=effective_table,
        category2_filter=category2_filter,
        category2_keyword_filter=effective_category2_keyword,
        prefetch_count=prefetch_count,
        user_filter_overrides=user_overrides,
        intent_type=analysis.intent_type,
        analysis_lanes=analysis.lane_metadata,
        extraction_info=extraction_info,
    )


def _route_category(category: str | None) -> tuple[str | None, str | None]:
    # 스커트와 원피스는 같은 테이블을 쓰므로 구분용 키워드도 함께 반환한다.
    if not category:
        return None, None
    table = TABLE_ROUTING_MAP.get(category)
    if not table:
        return None, None
    category2_keyword = None
    if category == "skirt":
        category2_keyword = "스커트"
    elif category in {"one-piece", "dress"}:
        category2_keyword = "원피스"
    return table, category2_keyword


def get_target_table(category1: str | None) -> str | None:
    """Compatibility helper for the old retrieval prototype API."""
    return _route_category((category1 or "").lower())[0]


def _normalize_query(text: str) -> str:
    # 한글 조합 방식과 영문 대소문자 차이가 키워드 탐지에 영향을 주지 않게 맞춘다.
    return f" {unicodedata.normalize('NFC', text.lower())} "


def _parse_category(normalized: str) -> tuple[str | None, str | None, str | None]:
    for category, table, category2_keyword, terms in CATEGORY_RULES:
        if _contains_any(normalized, terms):
            return category, table, category2_keyword
    return None, None, None


def _parse_structured_attributes(normalized: str) -> ParsedTextAttributes:
    # 문맥을 추론하는 모델이 아니라, 미리 정의된 단어·패턴을 찾는 보완 파서다.
    values: dict[str, str | None] = {
        "category1": _parse_category(normalized)[0],
        "color": _first_term(normalized, COLOR_TERMS),
        "fabric": _first_term(normalized, FABRIC_TERMS),
    }

    for field, patterns in ATTRIBUTE_PATTERNS.items():
        values[field] = _first_pattern_value(normalized, patterns)

    return ParsedTextAttributes(**values)




def _detect_intent_type(
    normalized: str,
    structured_attributes: ParsedTextAttributes,
) -> str:
    # '면접' 같은 상황 표현을 분류할 뿐, 계절·두께 등의 필터를 새로 추론하지 않는다.
    if not _contains_any(normalized, ABSTRACT_TPO_TERMS):
        return "attribute"
    if structured_attributes.category1:
        return "attribute_with_tpo_context"
    return "abstract_tpo"


def _first_pattern_value(
    normalized: str,
    patterns: Iterable[tuple[str, tuple[str, ...]]],
) -> str | None:
    # 문장 속 등장 순서가 아니라 패턴 목록의 순서가 우선순위를 결정한다.
    for value, terms in patterns:
        if _contains_any(normalized, terms):
            return value
    return None


def _first_term(normalized: str, terms: Iterable[str]) -> str | None:
    for term in terms:
        if _contains_any(normalized, (term,)):
            return term
    return None


def _join_terms(normalized: str, terms: Iterable[str]) -> str | None:
    found = [term for term in terms if _contains_any(normalized, (term,))]
    return ", ".join(dict.fromkeys(found)) or None


def _matched_terms(normalized: str, terms: Iterable[str]) -> list[str]:
    return [term for term in terms if _contains_any(normalized, (term,))]


def _contains_any(normalized: str, terms: Iterable[str]) -> bool:
    return any(_contains_term(normalized, term) for term in terms)


def _contains_term(normalized: str, term: str) -> bool:
    term = term.lower().strip()
    if not term:
        return False
    if re.search(r"[a-z0-9]", term):
        # 영문은 단어 경계를 확인해 'male'이 'female' 안에서 잘못 매칭되지 않게 한다.
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", normalized))
    return term in normalized


def _row_has_value(row: dict[str, Any], field: str) -> bool:
    value = row.get(field)
    return value not in (None, "", [], {})


def _row_matches_attribute(row: dict[str, Any], field: str, value: str) -> bool:
    # DB 필터와 달리, 여기서는 정규화한 상품 문자열에 동의어가 포함되는지 검사한다.
    row_text = _normalize_row_value(row.get(field))
    if not row_text:
        return False

    variants = VALUE_VARIANTS.get(field, {}).get(value, (value,))
    return any(_normalize_row_value(variant) in row_text for variant in variants)


def _exact_match_count(row: dict[str, Any], parsed: ParsedTextAttributes) -> int:
    # 카테고리는 검색 범위를 정할 때 쓰므로 상품별 일치 보너스에서는 제외한다.
    return sum(
        1
        for field in EXACT_FILTER_FIELDS
        if field != "category1"
        and getattr(parsed, field) is not None
        and _row_matches_attribute(row, field, getattr(parsed, field) or "")
    )


def _semantic_match_count(row: dict[str, Any], parsed: ParsedTextAttributes) -> int:
    count = 0
    for field in SOFT_TEXT_RERANK_FIELDS:
        expected = getattr(parsed, field)
        row_text = _normalize_row_value(row.get(field))
        if expected and row_text and any(
            _normalize_row_value(term) in row_text
            for term in _split_semantic_terms(expected)
        ):
            count += 1
    return count


def _split_semantic_terms(value: str) -> list[str]:
    return [term.strip() for term in value.split(",") if term.strip()]


def _normalize_row_value(value: Any) -> str:
    # DB 배열과 문자열을 같은 방식으로 비교할 수 있도록 소문자 한 줄로 맞춘다.
    if isinstance(value, list):
        value = " ".join(str(item) for item in value)
    return re.sub(r"\s+", " ", str(value or "").lower()).strip()


def refine_attribute_text_results(
    rows: list[dict[str, Any]],
    parsed: ParsedTextAttributes,
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # 반환 행에 실제로 존재하는 속성만 재검사한다. 없는 컬럼 때문에 전부 탈락시키지 않는다.
    exact_fields = [
        field
        for field in EXACT_FILTER_FIELDS
        if field != "category1" and getattr(parsed, field) is not None
    ]
    supported_exact_fields = [
        # 한 행이라도 값이 있으면 지원 필드다. 다른 행의 누락값은 불일치로 처리된다.
        field for field in exact_fields if any(_row_has_value(row, field) for row in rows)
    ]
    unsupported_exact_fields = [
        field for field in exact_fields if field not in supported_exact_fields
    ]

    filtered_rows = [
        row for row in rows
        if all(
            _row_matches_attribute(row, field, getattr(parsed, field) or "")
            for field in supported_exact_fields
        )
    ]
    # 모든 후보가 탈락하면 원래 후보를 유지하고, 응답 메타데이터에 조건 완화를 기록한다.
    used_exact_filter = bool(supported_exact_fields and filtered_rows)
    working_rows = filtered_rows if used_exact_filter else rows

    # 정렬 때만 속성 일치 보너스를 더한다. 행의 similarity 값 자체는 바꾸지 않는다.
    reranked_rows = sorted(
        working_rows,
        key=lambda row: (
            float(row.get("similarity") or 0)
            + 0.02 * _exact_match_count(row, parsed)
            + 0.01 * _semantic_match_count(row, parsed)
        ),
        reverse=True,
    )

    return reranked_rows[:top_k], {
        "post_filter_status": (
            "applied"
            if used_exact_filter
            else "relaxed_no_exact_matches"
            if supported_exact_fields
            else "not_supported_by_rpc_result"
        ),
        "supported_exact_fields": supported_exact_fields,
        "unsupported_exact_fields": unsupported_exact_fields,
        "soft_text_rerank_fields": [
            field
            for field in SOFT_TEXT_RERANK_FIELDS
            if getattr(parsed, field) is not None
        ],
        "input_count": len(rows),
        "output_count": min(len(reranked_rows), top_k),
    }


# ---------------------------------------------------------------------------
# SQL-based candidate pre-filtering (Supabase direct query)
# ---------------------------------------------------------------------------

def translate_to_db_value(key: str, value: Any) -> Any:
    """영어 속성값을 DB 저장값(한글)으로 변환."""
    # 예: wide 하나가 DB의 루즈·오버사이즈 두 값으로 확장될 수 있다.
    if key not in DB_TRANSLATION_MAP:
        return value
    mapping = DB_TRANSLATION_MAP[key]
    if isinstance(value, list):
        result = []
        for v in value:
            mapped = mapping.get(v, v)
            result.extend(mapped if isinstance(mapped, list) else [mapped])
        return list(set(result))
    mapped = mapping.get(value, value)
    return mapped


def retrieve_candidates(
    inferred_attributes: dict[str, Any],
    include_embeddings: bool = False,
    page_size: int = 100,
    max_rows: int = METADATA_PREFETCH_LIMIT,
    *,
    table_filter: str | None = None,
    category2_filter: str | None = None,
    category2_keyword_filter: str | None = None,
) -> list[dict[str, Any]]:
    """
    호환용 상품 조회/기존 방식 벤치마크에 쓰는 REST 조회. 운영 검색 순위 계산에는 쓰지 않는다.

    Args:
        inferred_attributes: {"category1": "top", "sleeve": "long", "color": "blue", ...}
        include_embeddings: 저장된 Gemini 이미지 임베딩을 응답에 포함할지 여부

    Returns:
        SQL 필터 조건을 만족하는 상품 리스트
    """
    from supabase import ClientOptions, create_client

    supabase_url = os.environ.get("SUPABASE_URL", "")
    supabase_key = os.environ.get("SUPABASE_KEY", "")
    if not supabase_url or not supabase_key:
        raise RuntimeError("SUPABASE_URL / SUPABASE_KEY 환경변수가 설정되지 않았습니다.")

    timeout_seconds = float(
        os.getenv("SUPABASE_REST_TIMEOUT_SECONDS", str(SUPABASE_REST_TIMEOUT_SECONDS))
    )
    supabase = create_client(
        supabase_url,
        supabase_key,
        options=ClientOptions(postgrest_client_timeout=timeout_seconds),
    )

    # 검색 계획의 테이블을 우선한다. 범위가 없으면 호출자가 전체 테이블 RPC로 이어간다.
    category1 = (inferred_attributes.get("category1") or "").lower()
    target_table = table_filter or TABLE_ROUTING_MAP.get(category1)
    if not target_table:
        return []
    if target_table not in VALID_TABLES:
        raise ValueError("Invalid table_filter.")

    # 임베딩이 필요 없는 /retrieve는 DB에서부터 벡터를 제외한다.
    fields = ["id", "name", "brand", "image_url", "category1", "category2", "price",
              "color", "sex", "season", "stretch", "thickness", "fit", "fabric"]
    if target_table != "musinsa_pants":
        fields.append("sleeve")
    if target_table != "musinsa_top_clothes":
        fields.append("length")
    if include_embeddings:
        fields.extend(["gemini_image_embedding_768", "gemini_fabric_text_embedding_768"])
    select_fields = ",".join(fields)
    rows: list[dict[str, Any]] = []
    offset = 0
    # 한 번에 모두 전송하지 않고 페이지로 읽되, 기본 1,000개에서 멈춘다.
    while True:
        page_end = min(offset + page_size - 1, max_rows - 1)
        # range의 마지막 위치도 포함되므로 페이지 끝에서 1을 뺀다.
        db_query = supabase.table(target_table).select(select_fields)
        # RPC와 동일하게 세부 분류는 정확 일치와 대소문자 무시 부분 일치를 함께 적용한다.
        if category2_filter is not None:
            db_query = db_query.eq("category2", category2_filter)
        if category2_keyword_filter is not None:
            db_query = db_query.ilike("category2", f"%{category2_keyword_filter}%")
        db_query = _apply_candidate_filters(db_query, inferred_attributes)
        # 페이지마다 같은 순서를 유지해 중복·누락 가능성을 줄인다.
        response = db_query.order("id").range(offset, page_end).execute()
        page = response.data or []
        normalized_page = [
            _normalize_candidate_row(row, target_table, supabase_url)
            for row in page
        ]
        if not include_embeddings:
            for row in normalized_page:
                row.pop("gemini_image_embedding_768", None)
                row.pop("gemini_fabric_text_embedding_768", None)
        rows.extend(normalized_page)
        if len(page) < page_size or len(rows) >= max_rows:
            return rows
        offset += page_size


def retrieve_products(inferred_attributes: dict[str, Any]) -> list[dict[str, Any]]:
    """Compatibility API for /retrieve and the old retrieval prototype.

    Uses env-based Supabase configuration and the current candidate narrowing
    implementation. Stored Gemini embeddings are not returned for this endpoint.
    """
    return retrieve_candidates(inferred_attributes, include_embeddings=False)


def _apply_candidate_filters(db_query: Any, inferred_attributes: dict[str, Any]) -> Any:
    # 여기서는 쿼리 객체에 조건만 붙인다. 실제 요청은 retrieve_candidates의 execute다.
    # 색상·소재는 SQL 완전 일치 조건으로 제한하지 않고 의미 유사도 검색에 맡긴다.
    for key, val in inferred_attributes.items():
        if not val or key in {"category1", "color", "fabric", "visual_detail", "reasoning"}:
            continue

        translated = translate_to_db_value(key, val)
        if key in ARRAY_COLUMNS:
            v_list = translated if isinstance(translated, list) else [translated]
            if key == "sex" and len(v_list) > 1:
                # 공용은 두 성별을 모두 포함해야 하고, 나머지 배열 속성은 하나만 겹쳐도 된다.
                db_query = db_query.contains(key, v_list)
            else:
                db_query = db_query.overlaps(key, v_list)
        elif key in TEXT_COLUMNS:
            if isinstance(translated, list):
                db_query = db_query.or_(",".join(f"{key}.eq.{v}" for v in translated))
            else:
                db_query = db_query.eq(key, translated)
    return db_query


def _normalize_candidate_row(
    row: dict[str, Any],
    table: str,
    supabase_url: str,
) -> dict[str, Any]:
    # 여러 테이블의 결과를 같은 카드 형식으로 쓰도록 출처와 이미지 주소를 보완한다.
    normalized = {**row, "source_table": table}
    if not normalized.get("image_url"):
        bucket = {
            "musinsa_top_clothes": "test",
            "musinsa_pants": "musinsa_pants",
            "musinsa_skirt_dress": "skirt_dress",
        }[table]
        normalized["image_url"] = (
            f"{supabase_url.rstrip('/')}/storage/v1/object/public/"
            f"{bucket}/{normalized['id']}.jpg"
        )
    return normalized


def filter_candidates_by_metadata(
    rows: list[dict[str, Any]],
    parsed: ParsedTextAttributes,
) -> list[dict[str, Any]]:
    # 이 보조 함수는 조건을 엄격히 적용한다. refine 함수처럼 빈 결과를 완화하지 않는다.
    exact_fields = [
        field
        for field in EXACT_FILTER_FIELDS
        if field != "category1" and getattr(parsed, field) is not None
    ]
    if not exact_fields:
        return rows
    return [
        row
        for row in rows
        if all(
            _row_matches_attribute(row, field, getattr(parsed, field) or "")
            for field in exact_fields
        )
    ]
