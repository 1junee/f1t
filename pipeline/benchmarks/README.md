# 검색 지연 벤치마크

현재 정확 필터 RPC의 DB·HTTP 계측과 기존 REST 방식 비교는 [`retrieval_diagnostics.py`](retrieval_diagnostics.py)를 사용합니다. 측정 방법, 결과, 응답 필드 설명은 [상품 조회 진단](RETRIEVAL_DIAGNOSTICS.md)에 정리했습니다. 아래 `search_latency.py`는 기존 방식 비교 실험입니다.

`search_latency.py`는 동일한 검색용 설명문 임베딩으로 전수 비교와 **이전 REST/Python 검색 방식**의 지연을 비교하는 벤치마크입니다. 현재 운영의 정확 메타데이터 검색은 `filtered_exact_vector_search_768` RPC를 사용하므로, 이 스크립트의 `hybrid` 결과를 현재 운영 성능으로 해석하면 안 됩니다.

- `full_scan`: HNSW와 bitmap scan을 끄고 세 상품 테이블의 모든 이미지 임베딩을 정확 코사인 비교
- `hybrid`: 이전 방식의 비교군. 명시 속성이 있으면 REST로 후보·벡터를 받아 Python에서 정확 코사인 재정렬하고, 후보가 없으면 HNSW 검색

VLM 설명 생성과 Gemini 쿼리 임베딩 생성 시간은 측정에서 제외합니다. 쿼리 임베딩은 최초 한 번 생성해 캐시에 저장하며 두 방법이 같은 벡터를 사용합니다. **전체 `/search` 응답시간이나 병렬화 전후 성능을 측정하는 도구는 아닙니다.**

`hybrid`는 저장된 명시 속성을 성공한 추출 결과로 재생하고 검색 계획·속성 후처리 함수를 공유합니다. 기본 후보 상한 1,000개, REST 페이지 100개는 이전 검색 방식의 설정입니다. 후보 조회/정렬 실패도 RPC로 전환하고 그 표본의 strategy는 `hnsw_fallback_prefilter_error`로 기록합니다. 현재 운영은 RPC 장애를 오류로 반환하므로 장애 처리도 다릅니다. `--max-candidates` 또는 `--page-size`를 바꾸면 이전 기본 설정과도 다른 실험이므로 결과의 configuration을 함께 표시하세요.

현재 운영은 정확 속성으로 필터링한 전체 후보를 DB 안에서 비교하고 상위 상품만 반환합니다. ID순 후보 상한을 적용하지 않으며, `matched_count`는 전체 일치 후보 수이고 반환 수가 아닙니다. 색상·소재는 정확 필터에서 제외하고 설명문·유사도 점수에 반영합니다. 이 문서의 과거 벤치마크 결과는 새 RPC의 측정 결과로 갱신하지 않았습니다.

측정에는 저장된 속성으로 계획을 만드는 로컬 계산, DB 왕복, 재정렬·후처리가 포함됩니다. 모델 추출·설명 생성, 임베딩 생성, 한국어 추천 이유 생성은 포함하지 않습니다. 실제 사용자가 선택한 카테고리 override를 재생하는 입력 필드는 현재 없으므로, 그 상호작용은 서비스 회귀 테스트에서 따로 확인합니다.

소재 조건이 있는 질의는 운영 검색과 동일하게 fabric 텍스트도 Gemini 임베딩으로 변환해 `hybrid` 경로에 전달합니다. SQL 후보 재정렬과 HNSW RPC 모두 `0.8 × 이미지 + 0.2 × fabric` 점수 정책을 그대로 사용합니다.

## 1. 전수 비교 RPC 준비

운영 DB가 아닌 동일 스펙의 벤치마크 DB에서 `pipeline/sql/benchmark_full_scan_fashion_items_768.sql`을 적용합니다. 이 함수의 scan 설정 변경은 해당 트랜잭션에만 적용됩니다.

## 2. 입력 준비

JSON 배열 또는 JSONL을 사용할 수 있습니다. 각 항목은 다음 형태입니다.

```json
{
  "id": "q001",
  "query": "긴팔 검은색 상의를 찾아줘",
  "target_description": "A black long-sleeve top",
  "inferred_attributes": {
    "category1": "top",
    "sleeve": "long"
  }
}
```

입력 개수 기본값은 기존 벤치마크의 108개입니다. 다른 질의 집합은 실제 고유 개수에 맞춰 `--expected-count`를 지정합니다.

`target_description`이 없으면 `query`를 임베딩하지만, 최종 실험에서는 실제 VLM이 생성했던 검색용 설명문을 넣어야 합니다. 입력 수를 임의 복제로 맞추지 않도록 `--strict-count`를 사용합니다.

```bash
python -m pipeline.benchmarks.search_latency \
  path/to/queries_108.json \
  --expected-count 108 \
  --strict-count \
  --dry-run
```

현재 저장소 스냅샷에 포함된 고유 쿼리는 54개입니다. 입력 파서만 확인하려면 다음 명령을 사용합니다.

```bash
python -m pipeline.benchmarks.search_latency \
  pipeline/tests/intent_analysis_results.json \
  pipeline/tests/simple_test/simple_test_results_gemini.json \
  pipeline/tests/simple_test/image_test_results_gemini.json \
  --expected-count 54 \
  --strict-count \
  --dry-run
```

[`results/preliminary_cpu_only_54.json`](results/preliminary_cpu_only_54.json)은 54개 고유 질의로 CPU 환경에서 실행한 예비 결과입니다. 운영 DB와 동일한 배포 환경, 반복 수, 108개 strict 입력을 갖춘 최종 운영 수치가 아니므로 포트폴리오의 성능 근거로는 사용하지 않습니다.

## 3. 측정 실행

```bash
python -m pipeline.benchmarks.search_latency \
  path/to/queries_108.json \
  --expected-count 108 \
  --strict-count \
  --repetitions 3 \
  --warmup-queries 3 \
  --output .omx/benchmarks/search_latency_108.json
```

한 쿼리의 두 방법은 번갈아 실행하고 전체 순서는 고정 seed로 섞습니다. 결과에는 다음 값이 저장됩니다.

- 방법별 평균, 중앙값, p95, 최솟값, 최댓값
- 같은 쿼리·반복에서 **두 방식이 모두 성공한 쌍만** 사용한 평균 절감 시간과 평균 지연 비율(배속)
- 평균 비교 후보 수와 후보 축소율
- SQL 후보 축소와 HNSW 폴백 사용 횟수
- 개별 실행의 지연, 후보 수, 오류

JSON은 요약과 원본 표본을 함께 저장하고 같은 경로에 CSV도 생성합니다. 최종 보고에는 평균만 쓰지 말고 중앙값과 p95를 함께 제시합니다.

## 해석 시 주의점

- 두 방식 모두 클라이언트 요청 시작부터 결과 수신까지의 검색 지연을 측정합니다.
- `hybrid`의 기존 SQL 후보 경로는 후보 임베딩을 페이지 단위로 받아 로컬에서 정확 코사인을 계산하므로 DB 왕복과 전송 시간도 포함합니다. 현재 운영의 정확 필터 RPC 경로와 구분해야 합니다.
- HNSW의 `candidate_count`는 RPC 정책상 테이블당 50~100개라는 **후보 상한의 추정치**입니다. 실제 반환 후보 수나 HNSW 내부에서 방문한 노드 수가 아닙니다. REST 경로는 실제 받은 행 수입니다. 서로 의미가 달라 후보 축소율을 실제 계산량 절감률로 해석하면 안 됩니다.
- 실험 중 다른 DB 부하, 리전, 네트워크가 결과에 영향을 줄 수 있으므로 동일 시간대·동일 DB에서 paired 방식으로 실행하고 최소 3회 반복합니다.
- 정확한 결과 비교가 목적이면 전수 비교 Top-K와 hybrid Top-K의 recall@K도 별도 측정해야 합니다. 이 스크립트의 주 지표는 검색 지연과 후보 수입니다.

## 캐시와 결과 해석

임베딩 캐시 각 항목에는 모델 ID·출력 차원·입력 접두어 형식이 저장됩니다. 같은 문장이라도 계약이 달라지거나 계약 정보가 없는 이전 캐시라면 다시 생성합니다. 출력 차원이 계약과 다르면 저장하지 않습니다. 서로 다른 모델의 실험은 파일도 분리해 두는 편이 결과 추적에 좋습니다.

방법별 평균·p95는 각각의 성공 표본 전체를 요약하고, `paired`의 절감 시간·배속·감소율은 양쪽이 모두 성공한 동일 표본만 사용합니다. 실패 개수도 함께 보고하세요. 후보 수 요약은 각 방법의 성공 표본 전체로 계산되므로 실패가 비대칭이면 paired 비교와 표본 집합이 다를 수 있습니다.

`full_scan`은 이미지 코사인만, 소재 조건이 있는 `hybrid`는 이미지·소재 가중 점수를 사용합니다. 두 방식의 순위 차이는 인덱스 근사 오차만을 뜻하지 않습니다. 순수 검색 recall 실험에는 같은 후보 범위·같은 점수 함수의 exact 기준이 별도로 필요합니다.


## 근사 검색 요청·병합 구조 비교

현재 운영 `SupabaseVectorSearchClient.search_approximate()`는 범위가 없어도 DB에 한 번 요청하며, DB가 테이블별 후보를 병합·정렬합니다. 이전 구현은 `retrieval_request_comparison.py`의 `search_three_requests_legacy()`로 보존했습니다. 운영 코드에서는 이 실험 함수를 호출하지 않습니다.

```python
from pipeline.benchmarks.retrieval_request_comparison import search_three_requests_legacy

# pipeline은 초기화된 FashionRecommendationPipeline, kwargs는 동일한 고정 입력이다.
# kwargs: embedding, match_count, table_filter, category2_filter,
#         category2_keyword_filter, fabric_embedding
old_rows = search_three_requests_legacy(pipeline.vector_search, **kwargs)
new_rows = pipeline.vector_search.search_approximate(**kwargs)
```

범위 없는 입력으로 기존 3회/현재 1회를 비교합니다. 위 예시는 호출법이며 반복 측정기는 아닙니다. 실제 paired 실험에서는 순서를 섞고 같은 벡터·필터·개수·가중치를 유지해 시간과 결과를 기록해야 합니다. 이전 함수의 전체 빈 결과 오류도 재현 목적으로 유지했으므로 정상 0건과 실제 장애는 별도 집계합니다. 단위 테스트는 호출 횟수·인자·병합·오류 계약을 검증하며 실제 성능 개선을 입증하지 않습니다. 상세 설계는 로컬 `experiments/06_retrieval_execution/PLAN.md`에 있습니다.
