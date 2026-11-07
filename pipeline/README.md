# F1T 패션 추천 파이프라인

패션 검색과 추천의 핵심 로직이 담긴 폴더입니다. HTTP API는 `../backend/`에 있습니다. 이 배포용 스냅샷에는 런타임에 필요하지 않은 실험 및 데이터 구축 자산을 포함하지 않습니다.

---

## 파이프라인 구조

진입점은 `recommendation_service.py`이며, 아래 단계를 조율합니다.

| 파일 | 역할 |
|---|---|
| `intent/intent_extraction.py` | 사용자가 언급한 속성을 추출하고, 의류 종류가 명확할 때만 카테고리 도출 |
| `retrieval/candidate_selection.py` | 속성 기반 Supabase 테이블 라우팅 + SQL 후보 필터링 |
| `target_description/target_description_generation.py` | Gemini로 검색용 영문 target description 생성 |
| `retrieval/target_description_retrieval.py` | gemini-embedding-2로 텍스트 → 768차원 벡터 인코딩 |
| `target_description/recommendation_explanation.py` | 매칭된 속성 기반 한국어 추천 이유 생성 |

오케스트레이터:

| 파일 | 역할 |
|---|---|
| `recommendation_service.py` | 파이프라인 전체 조율 (병렬 실행 포함) |
| `vector_db_client.py` | Supabase pgvector RPC 호출 클라이언트 |

---

## 전체 흐름

```
사용자 입력 (텍스트 + 이미지?)
    ↓
[1단계 + 2단계 — 병렬]
  ├→ 메타데이터 추출 — Gemini VLM (명시된 속성만)
  └→ Target Description 생성 — Gemini gemini-3.5-flash
    ↓
[3단계] 테이블·조건·후보 수를 정하는 검색 계획 생성 (DB 조회 전)
    ↓
[4단계 — 병렬]
  ├→ 목표 설명문 한국어 번역
  ├→ 목표 설명문 Gemini Embedding 인코딩
  └→ (fabric 속성 있을 시) fabric 임베딩 인코딩
    ↓
[5단계] SQL 후보 조회 + 로컬 재정렬 / RPC 벡터 검색 → 속성 후처리
    ↓
[6단계] 한국어 추천 이유 생성
    ↓
API 응답 반환
```

---

## 유사도 스코어 계산

벡터 검색 결과의 최종 유사도는 두 임베딩의 가중 합으로 계산합니다.

```
최종 점수 = 0.8 × 이미지 유사도 + 0.2 × fabric 유사도
```

- **이미지 유사도**: `gemini_image_embedding_768` — 상품 이미지 vs. target description 텍스트
- **fabric 유사도**: `gemini_fabric_text_embedding_768` — 상품 소재 설명 vs. 사용자 요청 소재
- 요청에 fabric 속성이 없으면 이미지 유사도만 사용
- 요청에 fabric 속성이 있는데 상품의 소재 벡터가 없으면 소재 점수는 0으로 계산

가중치는 `recommendation_service.py`의 `_rank_stored_gemini_vectors(fabric_weight=0.2)` 및 Supabase RPC 함수 `match_fashion_items_768`에서 적용합니다.

### 실제 검색 경로

`recommendation_service.py`는 모든 요청을 같은 방식으로 검색하지 않습니다.

1. 소매·기장·계절·핏처럼 정확한 속성이 있으면 `retrieve_candidates()`가 해당 카테고리 테이블을 REST로 조회합니다.
2. 후보가 있으면 DB에 저장된 `gemini_image_embedding_768`과 `gemini_fabric_text_embedding_768`을 사용해 Python에서 최종 순위를 계산합니다.
3. 정확한 후보가 없거나 REST 조회가 실패하면 `match_fashion_items_768` RPC로 전환합니다.
4. 테이블 조건이 없는 RPC 검색은 Supabase 연결 수를 제한하기 위해 세 상품 테이블을 순차 검색한 뒤 결과를 합칩니다.

메타데이터 필터 이후의 후보 수와 HNSW의 후보 수는 서로 다른 값입니다. 아래의 50~100개는 **RPC가 이미지 벡터로 고른 뒤 재정렬하는 후보 수**입니다.

---

## 로컬 실행

```bash
# 1. 환경변수 설정
cp pipeline/.env.example pipeline/.env
# GEMINI_API_KEY, SUPABASE_URL, SUPABASE_KEY, FRONTEND_ORIGINS 입력

# 2. 의존성 설치
conda create -n f1t python=3.11 -y
conda activate f1t
pip install -r pipeline/requirements.txt
cd frontend && npm install && cd ..

# 3. 백엔드 실행 (f1t_clean/ 루트에서)
uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload

# 4. 프론트엔드 실행 (다른 터미널)
cd frontend && npm run dev
```

프론트엔드: `http://localhost:5173`

---

## API

`POST /search` — `multipart/form-data`

| 파라미터 | 필수 | 설명 |
|---|---|---|
| `query` | 둘 중 하나 | 텍스트 검색어 |
| `image` | 둘 중 하나 | 참조 이미지 파일 |
| `top_k` | 선택 | 결과 수 (기본값 10) |
| `table` | 선택 | `musinsa_top_clothes` / `musinsa_pants` / `musinsa_skirt_dress` |
| `category2_keyword` | 선택 | 세부 카테고리 키워드 (예: `원피스`, `스커트`) |
| `pipeline_method` | 선택 | `intent` (기본값) |
| `provider` | 선택 | `gemini` (기본값) |

응답 예시:
```json
{
  "provider": "gemini",
  "target_description": "Oversized black long-sleeve hoodie",
  "target_description_ko": "오버사이즈 블랙 긴팔 후드티",
  "recommendation_reason": "요청하신 긴 소매와 오버사이즈 핏 조건에 맞는 후드 상의를 중심으로 찾았습니다. 여유 있는 실루엣과 후드 디테일이 캐주얼한 분위기를 살려줘 데일리로 입기 좋아 추천했습니다.",
  "pipeline": {
    "id": "intent_text_table_narrowing",
    "parsed_attributes": { "sleeve": "long", "fit": "oversized" },
    "parallel_execution": { "intent_and_target_description": true }
  },
  "results": [
    {
      "rank": 1,
      "name": "...",
      "similarity": 0.712
    }
  ]
}
```

---

## 환경변수 (.env)

| 변수 | 설명 |
|---|---|
| `GEMINI_API_KEY` | Gemini API 키 |
| `SUPABASE_URL` | Supabase 프로젝트 URL |
| `SUPABASE_KEY` | Supabase 서비스 키 |
| `FRONTEND_ORIGINS` | CORS 허용 출처 |

---

## Supabase 설정

벡터 검색을 위해 다음 세 항목을 Supabase에 설정해야 합니다.

**1. 이미지 HNSW 인덱스** (3개 테이블 모두):
```sql
CREATE INDEX ON musinsa_top_clothes USING hnsw (gemini_image_embedding_768 vector_cosine_ops);
CREATE INDEX ON musinsa_pants USING hnsw (gemini_image_embedding_768 vector_cosine_ops);
CREATE INDEX ON musinsa_skirt_dress USING hnsw (gemini_image_embedding_768 vector_cosine_ops);
```

### HNSW 후보 검색과 재정렬

임베딩은 이미지나 문장을 768개의 숫자로 바꾼 좌표입니다. 의미나 모습이 비슷하면 좌표도 가까워집니다. 코사인 거리 연산자 `<=>`는 두 좌표가 향하는 방향이 얼마나 다른지 측정하고, 코사인 유사도는 `1 - 코사인 거리`로 계산합니다.

인덱스가 없다면 검색할 때마다 모든 상품의 좌표를 하나씩 비교해야 합니다. HNSW(Hierarchical Navigable Small World)는 이 좌표들을 다음과 같은 다층 그래프로 정리합니다.

```text
상위층:  ●────────────●          멀리 건너뛰는 드문 연결
          ╲            ╲
중간층:  ●────●────●────●        검색 범위를 빠르게 좁힘
          │  ╱│   ╱│   ╱│
하위층:  ●─●─●─●─●─●─●─●─●      비슷한 상품끼리 촘촘한 연결
```

검색은 상위층의 한 점에서 시작합니다. 현재 점보다 질문 벡터에 가까운 이웃이 있으면 그쪽으로 이동하고, 더 가까운 이웃을 찾기 어려워지면 아래층으로 내려갑니다. 이 과정을 가장 촘촘한 하위층까지 반복해 가까운 후보를 찾습니다. 모든 상품을 확인하지 않기 때문에 빠르지만, 가장 가까운 상품을 100% 보장하는 정확 검색이 아니라 **근사 최근접 검색**입니다.

이 프로젝트는 속도와 재정렬 품질을 맞추기 위해 다음 두 값을 구분합니다.

```sql
result_count := least(greatest(coalesce(p_match_count, 10), 1), 100);
candidate_count := least(greatest(result_count * 5, 50), 100);
perform set_config('hnsw.ef_search', '100', true);
```

| 값 | 뜻 | 기본 Top 10 요청 |
|---|---|---:|
| `result_count` | 최종 반환할 상품 수 | 10 |
| `candidate_count` | 이미지 HNSW에서 받아 재정렬할 후보 수 | 50 |
| `hnsw.ef_search` | HNSW가 내부에서 유지하며 탐색하는 후보 폭 | 100 |

즉, “가까운 후보 50~100개”는 HNSW의 보편적인 고정값이 아니라 **`match_fashion_items_768` 함수에 명시한 프로젝트 정책**입니다. 요청 결과 수의 5배를 후보로 가져오되 최소 50개, 최대 100개로 제한합니다. 그 후보에 소재 조건이 있으면 이미지 80%와 소재 20%로 다시 점수를 매기고, 최종 `result_count`개만 반환합니다.

소재 HNSW를 따로 두지 않는 이유는 소재 벡터가 전체 상품을 처음부터 검색하는 기준이 아니라, 이미지 HNSW가 고른 작은 후보 집합을 재정렬하는 기준이기 때문입니다.

> `candidate_count`와 `ef_search`는 다른 값입니다. 전자는 SQL이 재정렬할 행 수이고, 후자는 HNSW 내부 탐색 폭입니다. `candidate_count`만 크게 잡고 `ef_search`가 작으면 실제 후보가 기대보다 적을 수 있으므로, RPC는 요청 안에서 `ef_search=100`을 함께 설정합니다.

- 라이브 함수와 동일한 SQL: [`sql/match_fashion_items_768.sql`](sql/match_fashion_items_768.sql)
- 공식 설명: [Supabase HNSW indexes](https://supabase.com/docs/guides/ai/vector-indexes/hnsw-indexes)

**2. RPC 함수** `match_fashion_items_768`:  
백엔드 전용 PostgreSQL 함수입니다. 이미지 HNSW로 후보를 고른 뒤 소재가
지정된 검색에는 `0.8 × 이미지 + 0.2 × fabric` 점수를 적용합니다.

함수 실행 권한은 `service_role`에만 있으며 브라우저의 `anon` 키로는 호출할 수 없습니다. 서비스 키는 프론트엔드 환경변수에 넣지 않고 백엔드에서만 사용합니다.

**3. 임베딩 빌드**:  
배포 환경에서는 이미 구축된 Supabase 테이블과 RPC를 사용합니다. 임베딩 생성 스크립트와 실험 산출물은 이 배포용 스냅샷에서 제외했습니다.
