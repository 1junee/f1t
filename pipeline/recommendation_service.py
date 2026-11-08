# 검색의 실행 순서를 조율한다. 개별 모델 호출·속성 추출·DB 통신은 각 모듈에 맡긴다.
from __future__ import annotations

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from time import perf_counter
from typing import Any

from .env import load_pipeline_env
from .timing import SearchTimings
from .intent.intent_extraction import extract_conservative_metadata
from .retrieval.candidate_selection import (
    build_attribute_text_search_plan,
    refine_attribute_text_results,
)
from .retrieval.target_description_retrieval import RetrieveModule
from .target_description.recommendation_explanation import (
    SearchStatus,
    generate_query_recommendation_explanation,
)
from .target_description.target_description_generation import (
    DEFAULT_IMAGE_ONLY_QUERY,
    GeminiReasoningClient,
    InvalidSearchInput,
    OPENAI_MODEL_NAME,
    OpenAIReasoningClient,
    QWEN_FALLBACK_MODEL_NAME,
    Qwen3VLReasoningClient,
    ReasoningProviderError,
    ReasoningWarning,
)
from .vector_db_client import SupabaseVectorSearchClient


logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 3
DEFAULT_RECOMMENDATION_COUNT = 10
MAX_TOP_K = 100
VALID_PROVIDERS = {"gemini", "openai", "qwen"}
VALID_TABLES = {
    "musinsa_pants",
    "musinsa_top_clothes",
    "musinsa_skirt_dress",
}
IMAGE_REFERENCE_QUERY_PATTERN = re.compile(
    r"(이\s*(이미지|사진|옷|상품|디자인|스타일|거|것)|이미지처럼|사진처럼|첨부|올린\s*(이미지|사진)|여기서|이거랑|이\s*옷이랑)"
)

load_pipeline_env()

# .env를 읽은 뒤 모듈 로딩 시 결정한다. 실행 중 환경변수를 바꿔도 이 상수는 유지된다.
PIPELINE_MODE = os.getenv("PIPELINE_MODE", "supabase_vector")

# 응답에서 보여줄 단계 정보다. 실제 실행 순서는 아래 search()의 함수 호출을 따른다.
INTENT_RECOMMENDATION_STAGES = [
    {
        "id": "intent_extraction",
        "label": "VLM 의도 추출",
        "implementation": "pipeline.intent.intent_extraction.extract_conservative_metadata",
    },
    {
        "id": "candidate_selection",
        "label": "의도 기반 DB 후보 축소",
        "implementation": "pipeline.retrieval.candidate_selection.build_attribute_text_search_plan",
    },
    {
        "id": "target_description_vector_search",
        "label": "타겟 디스크립션 기반 최종 추천",
        "implementation": (
            "pipeline.retrieval.target_description_retrieval.RetrieveModule + "
            "pipeline.vector_db_client.SupabaseVectorSearchClient"
        ),
    },
    {
        "id": "recommendation_explanation",
        "label": "한국어 추천 이유 생성",
        "implementation": (
            "pipeline.target_description.recommendation_explanation."
            "build_query_recommendation_explanation"
        ),
    },
]


class FashionRecommendationPipeline:
    """Application-level fashion recommendation workflow."""

    def __init__(self) -> None:
        # 서버 시작 시 임베딩·DB 클라이언트를 준비한다. 여기서 상품 검색을 실행하지는 않는다.
        if PIPELINE_MODE != "supabase_vector":
            raise RuntimeError("PIPELINE_MODE=supabase_vector is required.")

        self._default_provider = os.getenv("VLM_PROVIDER", "gemini").lower()
        self._client_cache: dict[str, Any] = {}
        self._client_cache_lock = Lock()
        self.retriever = RetrieveModule()
        self.vector_search = SupabaseVectorSearchClient()

    def _get_reasoning_client(self, provider: str) -> Any:
        # 설명 생성용 클라이언트는 해당 제공자를 처음 사용할 때 만들고 재사용한다.
        if provider not in VALID_PROVIDERS:
            raise ValueError("provider must be gemini, openai, or qwen.")
        # 확인·생성·저장을 함께 잠근다. 클라이언트의 run()은 잠금 밖에서 실행한다.
        with self._client_cache_lock:
            if provider not in self._client_cache:
                if provider == "openai":
                    self._client_cache[provider] = OpenAIReasoningClient()
                elif provider == "qwen":
                    self._client_cache[provider] = Qwen3VLReasoningClient()
                else:
                    self._client_cache[provider] = GeminiReasoningClient()
            return self._client_cache[provider]

    # 사용자 입력과 옵션 → (영문 목표 설명, 실행 메타데이터, 상품 목록).
    # API는 이 세 값을 화면용 JSON으로 포장한다. 이 함수가 검색 단계들의 연결 지점이다.
    def search(
        self,
        query: str,
        top_k: int = DEFAULT_RECOMMENDATION_COUNT,
        image_bytes: bytes | None = None,
        image_mime_type: str | None = None,
        table_filter: str | None = None,
        category2_filter: str | None = None,
        category2_keyword_filter: str | None = None,
        provider: str | None = None,
        intent_provider: str | None = None,
        target_description_provider: str | None = None,
    ) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
        timings = SearchTimings()

        # 공통 제공자를 기본으로 쓰고 역할별 지정이 있으면 우선한다. 임베딩은 별도 설정이다.
        selected_provider = (provider or self._default_provider).lower().strip()
        selected_intent_provider = (intent_provider or selected_provider).lower().strip()
        selected_description_provider = (
            (target_description_provider or selected_provider).lower().strip()
        )
        if any(
            value not in VALID_PROVIDERS
            for value in (
                selected_provider,
                selected_intent_provider,
                selected_description_provider,
            )
        ):
            raise ValueError("provider must be gemini, openai, or qwen.")
        raw_fallback = query.strip() or DEFAULT_IMAGE_ONLY_QUERY

        # 두 작업은 같은 원본 입력을 받고 서로의 출력을 쓰지 않아 API 대기를 겹칠 수 있다.
        # 요청을 처리하는 흐름 안에서 별도 스레드 두 개를 사용한다. 응답을 미리 보내지는 않는다.
        # 외부 API의 응답 대기를 겹치는 구조이며, CPU 계산의 속도가 자동으로 두 배가 되는 것은 아니다.
        with ThreadPoolExecutor(max_workers=2) as executor:
            intent_future = executor.submit(
                timings.run,
                "intent_extraction",
                "검색 의도·조건 해석",
                "analysis",
                extract_conservative_metadata,
                query=query,
                image_bytes=image_bytes,
                image_mime_type=image_mime_type,
                provider=selected_intent_provider,
            )
            reasoning_future = executor.submit(
                timings.run,
                "target_description",
                "목표 상품 설명 생성",
                "analysis",
                self._run_target_description_reasoning,
                query=query,
                image_bytes=image_bytes,
                image_mime_type=image_mime_type,
                provider=selected_description_provider,
                raw_fallback=raw_fallback,
            )
            # 두 submit이 끝난 뒤 결과를 기다린다. result()의 순서가 실행을 순차화하지는 않는다.
            intent_extraction = intent_future.result()
            reasoning_result, reasoning_provider, reasoning_status, warnings = (
                reasoning_future.result()
            )

        # 추출 속성으로 검색할 테이블·필터·후보 수를 정한다. 아직 DB를 조회하지 않는다.
        with timings.stage("search_plan", "검색 조건 구성"):
            attribute_plan = build_attribute_text_search_plan(
                query=query,
                top_k=top_k,
                has_image=bool(image_bytes),
                extracted_attributes=(
                    intent_extraction.attributes
                    if intent_extraction and intent_extraction.status == "ok"
                    else None
                ),
                extraction_info=(
                    intent_extraction.to_metadata() if intent_extraction else None
                ),
                table_filter=table_filter,
                category2_filter=category2_filter,
                category2_keyword_filter=category2_keyword_filter,
            )

        parsed_attributes = attribute_plan.parsed_attributes.to_dict()
        pipeline_metadata = attribute_plan.to_dict()
        pipeline_metadata["providers"] = {
            "intent": intent_extraction.to_metadata() if intent_extraction else None,
            "target_description": {
                "requested_provider": selected_description_provider,
                "provider": reasoning_provider,
                "status": reasoning_status,
            },
        }
        pipeline_metadata["parallel_execution"] = {
            "intent_and_target_description": True,
            "description_and_fabric_embedding": bool(attribute_plan.parsed_attributes.fabric),
        }

        # 참고 이미지를 그대로 검색 벡터로 만들지 않고, 생성된 설명문을 검색 입력으로 사용한다.
        target_description = reasoning_result["Target Image Description"]
        pipeline_metadata["descriptions"] = {
            "target_description_reasoning": target_description,
        }
        pipeline_metadata["target_description_reasoning"] = reasoning_result
        pipeline_metadata["target_description_request"] = query

        # fabric 속성이 있으면 gemini 임베딩으로 인코딩 (target description 임베딩과 병렬)
        fabric_text = (
            attribute_plan.parsed_attributes.fabric
            if attribute_plan.parsed_attributes.fabric
            else None
        )

        # 검색용 설명 임베딩과 선택적 소재 임베딩은 서로 독립적인 작업이다.
        with ThreadPoolExecutor(max_workers=2) as executor:
            embedding_future = executor.submit(
                timings.run,
                "embedding",
                "검색 문장 임베딩",
                "encoding",
                self.retriever.encode_text,
                [target_description],
            )
            fabric_future = (
                executor.submit(
                    timings.run,
                    "fabric_embedding",
                    "소재 조건 임베딩",
                    "encoding",
                    self.retriever.encode_text,
                    [fabric_text],
                )
                if fabric_text
                else None
            )
            # encode_text는 문장 목록을 받으므로 첫 결과 [0]을 꺼낸다. DB의 상품 벡터와 비교할 질의 벡터다.
            query_embedding = embedding_future.result()[0].tolist()
            fabric_embedding = (
                fabric_future.result()[0].tolist() if fabric_future else None
            )

        with timings.stage("retrieval", "상품 조회·유사도 검색"):
            active_table_filter = attribute_plan.table_filter
            active_category2_filter = attribute_plan.category2_filter
            active_category2_keyword_filter = attribute_plan.category2_keyword_filter
            match_count = attribute_plan.prefetch_count

            # None은 아직 유효한 검색 경로의 결과를 얻지 못했다는 표식으로 RPC 전환 여부를 결정한다.
            results = None

            # 정확한 속성 조건은 DB 안에서 전체 일치 후보를 정렬한다. 벡터를 내려받지 않는다.
            if _has_exact_metadata_filters(parsed_attributes):
                strict_results, retrieval_diagnostics = (
                    self.vector_search.search_filtered_exact(
                        embedding=query_embedding,
                        # DB에서 이미 정확 필터와 전체 순위를 계산했으므로 요청한 수만 받는다.
                        match_count=top_k,
                        metadata_filters=parsed_attributes,
                        table_filter=active_table_filter,
                        category2_filter=active_category2_filter,
                        category2_keyword_filter=active_category2_keyword_filter,
                        fabric_embedding=fabric_embedding,
                    )
                )
                pipeline_metadata["retrieval_diagnostics"] = retrieval_diagnostics
                if strict_results:
                    results = strict_results
                    pipeline_metadata["prefetch_count"] = top_k
                    pipeline_metadata["candidate_selection"] = {
                        "status": "applied",
                        "strategy": "intent_db_filter_then_exact_vector_rank",
                        "strict_candidate_count": retrieval_diagnostics["matched_count"],
                        "uses_stored_image_embeddings": True,
                        "fabric_weighted": fabric_embedding is not None,
                    }
                else:
                    pipeline_metadata["candidate_selection"] = {
                        "status": "fallback_no_strict_candidates",
                        "strategy": "rpc_vector_search",
                        "strict_candidate_count": 0,
                    }
                # RPC 설치 누락·장애는 예외를 그대로 올린다. 조건을 조용히 무시해 성공 처리하지 않는다.

            # 정확 조건이 없거나 일치 후보가 0개일 때 기존 완화 검색을 사용한다.
            if results is None:
                fallback_started = perf_counter()
                results = self.vector_search.search_approximate(
                    embedding=query_embedding,
                    match_count=match_count,
                    table_filter=active_table_filter,
                    category2_filter=active_category2_filter,
                    category2_keyword_filter=active_category2_keyword_filter,
                    fabric_embedding=fabric_embedding,
                )
                if "retrieval_diagnostics" in pipeline_metadata:
                    # request_count=1은 위 정확 필터 RPC만의 계측이다. 완화 검색은 별도로 표시한다.
                    pipeline_metadata["retrieval_diagnostics"]["scope"] = "strict_filtered_rpc"
                    pipeline_metadata["retrieval_diagnostics"]["fallback"] = {
                        "reason": "no_strict_candidates",
                        "strategy": "rpc_vector_search",
                        "duration_ms": round((perf_counter() - fallback_started) * 1000, 2),
                        "scope": "approximate_rpc",
                        # 성공한 완화 검색의 단일 RPC만 센다. 위 정확 검색 1회는 별도다.
                        "request_count": 1,
                    }

        # 두 검색 경로의 결과에 동일한 후처리를 적용하고 최종 top_k개를 정한다.
        with timings.stage("postprocess", "결과 정렬·선별"):
            results, result_strategy = refine_attribute_text_results(
                results,
                attribute_plan.parsed_attributes,
                top_k,
            )

        # 후처리까지 끝난 최종 결과를 기준으로 안내한다. 0건은 완화 여부보다 우선한다.
        candidate_status = pipeline_metadata.get("candidate_selection", {}).get("status")
        search_status: SearchStatus
        if not results:
            search_status = "no_results"
        elif (
            candidate_status == "fallback_no_strict_candidates"
            or result_strategy.get("post_filter_status") == "relaxed_no_exact_matches"
        ):
            search_status = "relaxed"
        elif candidate_status == "applied":
            search_status = "exact"
        else:
            search_status = "similarity"
        applied_filters = (
            {
                field: value
                for field, value in parsed_attributes.items()
                if field not in {"category1", "color", "fabric"}
            }
            if search_status == "exact"
            else {}
        )
        pipeline_metadata["search_status"] = search_status

        pipeline_metadata["result_strategy"] = result_strategy

        pipeline_metadata["stages"] = INTENT_RECOMMENDATION_STAGES
        pipeline_metadata["recommendation_count"] = top_k
        # 현재 설명은 요청·추출 속성·목표 설명을 요약한다. 개별 반환 상품 행은 전달하지 않는다.
        with timings.stage("recommendation_explanation", "추천 이유 작성"):
            recommendation_explanation = generate_query_recommendation_explanation(
                query=query,
                parsed_attributes=pipeline_metadata.get("parsed_attributes"),
                search_status=search_status,
                result_count=len(results),
                applied_filters=applied_filters,
                target_description_reasoning=reasoning_result,
                has_reference_image=bool(image_bytes),
            )

        recommendation_reason = recommendation_explanation.recommendation_reason

        metadata = {
            "provider": reasoning_provider,
            "status": reasoning_status,
            "warnings": [warning.to_dict() for warning in warnings],
            "pipeline": pipeline_metadata,
            "recommendation_reason": recommendation_reason,
            "recommendation_explanation": recommendation_explanation.to_metadata(),
        }

        pipeline_metadata["timings"] = timings.snapshot()
        return target_description, metadata, results

    # 원본 문장·이미지 → 목표 설명 딕셔너리, 실제 사용 제공자, 상태, 경고 목록.
    # 대체 경로를 사용해도 반환 형태를 유지하므로 search()가 같은 방식으로 이어서 처리한다.
    def _run_target_description_reasoning(
        self,
        *,
        query: str,
        raw_fallback: str,
        image_bytes: bytes | None,
        image_mime_type: str | None,
        provider: str,
    ) -> tuple[dict[str, Any], str, str, list[ReasoningWarning]]:
        if provider not in VALID_PROVIDERS:
            raise ValueError("provider must be gemini, openai, or qwen.")

        is_cloud_provider = provider not in ("qwen",)
        warnings: list[ReasoningWarning] = []
        reasoning_provider = provider
        reasoning_status = "ok"

        try:
            # 초기화 실패도 제공자 장애로 처리하되, 잘못된 제공자 이름은 위에서 차단한다.
            primary_client = self._get_reasoning_client(provider)
            reasoning_result = primary_client.run(
                manipulation_text=query,
                image_bytes=image_bytes,
                image_mime_type=image_mime_type,
            )
        except InvalidSearchInput:
            # 비패션 입력은 API가 400으로 응답해야 하므로 원문 검색으로 전환하지 않는다.
            raise
        except ReasoningProviderError as exc:
            logger.exception("목표 상품 설명 생성 제공자 오류: provider=%s", provider)
            warnings.append(exc.to_warning())

            # 알려진 클라우드 추론 오류는 Qwen으로 재시도하고, 그것도 실패하면 원문을 사용한다.
            if is_cloud_provider:
                try:
                    qwen_client = self._get_reasoning_client("qwen")
                    reasoning_result = qwen_client.run(
                        manipulation_text=query,
                        image_bytes=image_bytes,
                        image_mime_type=image_mime_type,
                    )
                    reasoning_provider = "qwen"
                    reasoning_status = "fallback"
                except InvalidSearchInput:
                    raise
                except Exception:
                    logger.exception("Qwen 대체 설명 생성 실패: 사용자 원문으로 검색합니다.")
                    reasoning_result, reasoning_provider, reasoning_status = (
                        _raw_query_fallback(raw_fallback)
                    )
            else:
                reasoning_result, reasoning_provider, reasoning_status = (
                    _raw_query_fallback(raw_fallback)
                )
        except Exception:
            logger.exception(
                "목표 상품 설명 생성 실패: provider=%s, 사용자 원문으로 검색합니다.",
                provider,
            )
            # 기타 예외는 Qwen 재시도 없이 원문 검색으로 낮추고 경고를 응답에 남긴다.
            warnings.append(
                ReasoningWarning(
                    code="provider_inference_failed",
                    label="추론 오류",
                    message="추론에 실패해 원문으로 검색했습니다.",
                )
            )
            reasoning_result, reasoning_provider, reasoning_status = (
                _raw_query_fallback(raw_fallback)
            )

        return reasoning_result, reasoning_provider, reasoning_status, warnings


def _raw_query_fallback(text: str) -> tuple[dict[str, Any], str, str]:
    """원문 검색으로 전환할 때 설명·제공자·상태를 함께 정한다."""
    return {"Target Image Description": text}, "raw_query", "degraded"


# '이 사진처럼' 같은 표현의 존재를 정규식으로 검사한다. 모델이 의미를 판정하는 단계는 아니다.
def needs_reference_image(query: str) -> bool:
    return bool(IMAGE_REFERENCE_QUERY_PATTERN.search(re.sub(r"\s+", " ", query)))


def _has_exact_metadata_filters(attributes: dict[str, Any]) -> bool:
    # 카테고리는 검색 범위, 색상·소재는 의미 유사도로 다루므로 여기의 정확 필터에서 제외한다.
    return any(
        value
        for field, value in attributes.items()
        if field not in {"category1", "color", "fabric"}
    )


__all__ = [
    "DEFAULT_RECOMMENDATION_COUNT",
    "DEFAULT_TOP_K",
    "FashionRecommendationPipeline",
    "MAX_TOP_K",
    "PIPELINE_MODE",
    "VALID_TABLES",
    "needs_reference_image",
]
