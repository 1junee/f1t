from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Literal

from .target_description_generation import (
    GEMINI_MODEL_NAME,
    LEGACY_GEMINI_MODEL_ENV,
    TARGET_DESCRIPTION_MODEL_ENV,
)


RECOMMENDATION_REASON_MODEL_ENV = "GEMINI_RECOMMENDATION_REASON_MODEL"
RECOMMENDATION_REASON_TIMEOUT_ENV = "GEMINI_RECOMMENDATION_REASON_TIMEOUT_MS"
DEFAULT_RECOMMENDATION_REASON_TIMEOUT_MS = 10000

FORBIDDEN_REASON_TERMS = (
    "Target Description",
    "target description",
    "메타데이터",
    "벡터",
    "DB",
    "후보 상품",
    "이미지 특징",
)


ATTRIBUTE_VALUE_LABELS = {
    "sex": {"male": "남성", "female": "여성", "unisex": "남녀공용"},
    "season": {"spring": "봄", "summer": "여름", "fall": "가을", "winter": "겨울"},
    "stretch": {"yes": "신축성 있음", "no": "신축성 없음"},
    "category1": {
        "top": "상의",
        "pants": "바지",
        "skirt": "스커트",
        "dress": "원피스",
        "one-piece": "원피스",
    },
    "sleeve": {
        "long": "긴 소매",
        "short": "짧은 소매",
        "sleeveless": "민소매",
    },
    "length": {
        "long": "긴 기장",
        "half": "중간 기장",
        "short": "짧은 기장",
    },
    "fit": {
        "slim": "슬림 핏",
        "regular": "레귤러 핏",
        "loose": "루즈 핏",
        "oversized": "오버사이즈 핏",
        "wide": "와이드 핏",
    },
    "thickness": {
        "thin": "얇은 두께",
        "medium": "보통 두께",
        "thick": "두꺼운 두께",
    },
}

SearchStatus = Literal["exact", "relaxed", "similarity", "no_results"]


@dataclass(frozen=True)
class RecommendationExplanationResult:
    # 최종 문구뿐 아니라 모델 호출·대체 문구 사용 여부도 응답 메타데이터로 남긴다.
    recommendation_reason: str
    method: str
    uses_additional_model_call: bool
    fallback_used: bool
    model: str | None = None
    error: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "language": "ko",
            "method": self.method,
            "uses_additional_model_call": self.uses_additional_model_call,
            "fallback_used": self.fallback_used,
        }
        if self.model:
            metadata["model"] = self.model
        if self.error:
            metadata["error"] = self.error
        return metadata


def generate_query_recommendation_explanation(
    *,
    query: str,
    parsed_attributes: dict[str, str] | None,
    search_status: SearchStatus,
    result_count: int,
    applied_filters: dict[str, str] | None = None,
    target_description_reasoning: dict[str, Any] | None = None,
    has_reference_image: bool = False,
    model_name: str | None = None,
    allow_model_call: bool = True,
) -> RecommendationExplanationResult:
    # 영문 목표 설명과 검색 상태를 사용한다. 개별 상품 적합성을 검증하는 기능은 아니다.
    # 실제 적용한 정확 필터만 대체 문구에 넣고, 결과 없음/완화는 고정 안내를 사용한다.
    fallback = build_query_recommendation_explanation(
        search_status=search_status,
        result_count=result_count,
        applied_filters=applied_filters,
    )
    if result_count == 0 or search_status in {"no_results", "relaxed"}:
        return RecommendationExplanationResult(
            recommendation_reason=fallback,
            method="deterministic_search_status",
            uses_additional_model_call=False,
            fallback_used=False,
        )
    model_name = resolve_recommendation_reason_model(model_name)
    # 모델 호출을 끈 경우와 키가 없는 경우에는 네트워크 요청 없이 대체 문구를 반환한다.
    if not allow_model_call:
        return RecommendationExplanationResult(
            recommendation_reason=fallback,
            method="deterministic_query_summary",
            uses_additional_model_call=False,
            fallback_used=True,
            model=model_name,
            error="Recommendation reason model call is disabled.",
        )
    if not os.getenv("GEMINI_API_KEY"):
        return RecommendationExplanationResult(
            recommendation_reason=fallback,
            method="deterministic_query_summary",
            uses_additional_model_call=False,
            fallback_used=True,
            model=model_name,
            error="GEMINI_API_KEY is not set.",
        )

    try:
        prompt = _build_gemini_recommendation_reason_prompt(
            query=query,
            parsed_attributes=parsed_attributes or {},
            search_status=search_status,
            result_count=result_count,
            applied_filters=applied_filters or {},
            target_description_reasoning=target_description_reasoning or {},
            has_reference_image=has_reference_image,
        )
        generated = _generate_gemini_recommendation_reason(
            prompt=prompt,
            model_name=model_name,
        )
        # 한국어·문장 수·금지 표현을 확인한다. 상품과의 사실 일치 여부를 검증하는 단계는 아니다.
        reason = _validate_generated_reason(generated)
        return RecommendationExplanationResult(
            recommendation_reason=reason,
            method="gemini_query_reason",
            uses_additional_model_call=True,
            fallback_used=False,
            model=model_name,
        )
    except Exception as exc:
        # 설명 생성 실패로 검색 결과 전체를 실패시키지 않고 준비한 문구를 반환한다.
        return RecommendationExplanationResult(
            recommendation_reason=fallback,
            method="deterministic_query_summary",
            uses_additional_model_call=True,
            fallback_used=True,
            model=model_name,
            error=str(exc),
        )


def build_query_recommendation_explanation(
    *,
    search_status: SearchStatus,
    result_count: int,
    applied_filters: dict[str, str] | None = None,
) -> str:
    """번역이나 키워드 추측 없이 실제 검색 상태를 한국어로 안내한다."""
    if result_count == 0 or search_status == "no_results":
        return "이번 검색에서 추천할 상품을 찾지 못했습니다."
    if search_status == "relaxed":
        return (
            "요청 조건에 정확히 일치하는 상품이 없어, "
            "일부 조건을 완화해 유사한 아이템을 검색했습니다."
        )
    if search_status == "exact":
        # 색상·소재는 의미 유사도 대상이다. 실제 DB에 적용한 정확 필터만 이름을 붙인다.
        conditions = [
            ATTRIBUTE_VALUE_LABELS[field][value]
            for field, value in (applied_filters or {}).items()
            if value in ATTRIBUTE_VALUE_LABELS.get(field, {})
        ]
        if conditions:
            return f"{', '.join(conditions)} 조건을 적용해 검색했습니다."
        return "설정한 속성 조건을 적용해 검색했습니다."
    return "입력하신 요청을 바탕으로 유사한 아이템을 검색했습니다."


def resolve_recommendation_reason_model(model_name: str | None = None) -> str:
    # 추천 문구 전용 모델을 지정하지 않으면 검색용 설명문 모델 설정을 따른다.
    return (
        model_name
        or os.getenv(RECOMMENDATION_REASON_MODEL_ENV)
        or os.getenv(TARGET_DESCRIPTION_MODEL_ENV)
        or os.getenv(LEGACY_GEMINI_MODEL_ENV)
        or GEMINI_MODEL_NAME
    )


def _build_gemini_recommendation_reason_prompt(
    *,
    query: str,
    parsed_attributes: dict[str, str],
    search_status: SearchStatus,
    result_count: int,
    applied_filters: dict[str, str],
    target_description_reasoning: dict[str, Any],
    has_reference_image: bool,
) -> str:
    reasoning_context = {
        "original_image_description": target_description_reasoning.get(
            "Original Image Description", ""
        ),
        "thoughts": target_description_reasoning.get("Thoughts", ""),
        "reflections": target_description_reasoning.get("Reflections", ""),
        "target_image_description": target_description_reasoning.get(
            "Target Image Description", ""
        ),
    }
    payload = {
        # 상품 행 대신 요청과 해석 결과를 전달하므로, 이 문구는 요청 전체에 대한 설명이다.
        "user_query": query,
        "has_reference_image": has_reference_image,
        "extracted_attributes": parsed_attributes,
        "search_status": search_status,
        "result_count": result_count,
        "applied_filters": applied_filters,
        "reasoning_context": reasoning_context,
    }
    return (
        "You are a Korean UX copywriter for a fashion recommendation service.\n"
        "Write one user-facing recommendation reason from the input context.\n"
        "Return only the final Korean copy. No JSON. No markdown.\n\n"
        "Rules:\n"
        "1. Korean only, exactly two natural sentences.\n"
        "2. Sentence 1 summarizes the user's requested condition or occasion.\n"
        "3. Sentence 2 describes the search approach, not unverified product suitability.\n"
        "4. Product rows are not provided. Do not claim returned products satisfy all requested conditions.\n"
        "Only applied_filters were enforced exactly; other attributes describe search intent.\n"
        "5. Do not mention: Target Description, metadata, vector, DB, candidate products, image features.\n\n"
        f"Input context:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _generate_gemini_recommendation_reason(
    *,
    prompt: str,
    model_name: str,
) -> str:
    # 검색용 설명문 생성과는 별도의 Gemini 호출이며, 자체 타임아웃을 적용한다.
    timeout_ms = int(
        os.getenv(
            RECOMMENDATION_REASON_TIMEOUT_ENV,
            str(DEFAULT_RECOMMENDATION_REASON_TIMEOUT_MS),
        )
    )
    request_body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 2048,
            # Flash-Lite 3.5는 thinkingBudget=0을 거부하므로 지원하는 최소 추론 수준을 사용한다.
            "thinkingConfig": (
                {"thinkingLevel": "minimal"}
                if model_name.removeprefix("models/") == "gemini-3.5-flash-lite"
                else {"thinkingBudget": 0}
            ),
        },
    }
    request_data = json.dumps(request_body).encode("utf-8")
    api_key = os.getenv("GEMINI_API_KEY")
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_name}:generateContent?key={api_key}"
    )
    request = urllib.request.Request(
        url,
        data=request_data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        # 환경변수는 밀리초로 받고 urllib에는 초로 변환해 전달한다.
        with urllib.request.urlopen(request, timeout=timeout_ms / 1000) as response:
            response_json = json.loads(response.read().decode("utf-8"))
        text = response_json["candidates"][0]["content"]["parts"][0]["text"].strip()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini recommendation reason failed: {detail[:400]}") from exc
    except Exception as exc:
        raise RuntimeError(f"Gemini recommendation reason failed: {exc}") from exc

    return text


def _validate_generated_reason(text: str) -> str:
    reason = _clean_generated_reason(text)
    if not reason:
        raise RuntimeError("Recommendation reason response is empty.")
    if not re.search(r"[가-힣]", reason):
        raise RuntimeError("Recommendation reason response is not Korean.")
    # 프롬프트는 두 문장을 요청하지만, 현재 검사는 두 문장 미만인지만 확인한다.
    if _sentence_count(reason) < 2:
        raise RuntimeError("Recommendation reason response is not two sentences.")
    if re.search(r"[*#`{}\\[\\]]", reason):
        raise RuntimeError("Recommendation reason response contains markup.")
    forbidden = [term for term in FORBIDDEN_REASON_TERMS if term in reason]
    if forbidden:
        raise RuntimeError(
            "Recommendation reason response contains internal terms: "
            + ", ".join(forbidden)
        )
    return reason


def _clean_generated_reason(text: str) -> str:
    # 모델이 붙인 코드 블록·바깥따옴표·중복 공백을 제거해 화면용 한 문단으로 만든다.
    reason = str(text or "").strip()
    reason = re.sub(r"^```(?:text)?\s*|\s*```$", "", reason).strip()
    reason = reason.strip("\"'“”‘’")
    reason = re.sub(r"\s+", " ", reason).strip()
    return reason


def _sentence_count(text: str) -> int:
    return len(re.findall(r"[.!?。]|[다요죠군요습니다]\.", text))
