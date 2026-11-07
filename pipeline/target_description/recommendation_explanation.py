from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

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


ATTRIBUTE_LABELS = {
    "category1": "카테고리",
    "color": "색상",
    "fabric": "소재",
    "sleeve": "소매",
    "length": "기장",
    "sex": "성별",
    "season": "계절",
    "stretch": "신축성",
    "thickness": "두께",
    "fit": "핏",
}

ATTRIBUTE_VALUE_LABELS = {
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

DETAIL_LABELS = {
    "hood": "후드",
    "character": "캐릭터 그래픽",
    "graphic": "그래픽 프린트",
    "print": "프린트",
    "pattern": "패턴",
    "logo": "로고",
}

VALUE_ALIASES = {
    "category1": {
        "top": ("top", "상의"),
        "pants": ("pants", "바지", "팬츠"),
        "skirt": ("skirt", "스커트", "치마"),
        "dress": ("dress", "one-piece", "원피스", "드레스"),
        "one-piece": ("dress", "one-piece", "원피스", "드레스"),
    },
    "sleeve": {
        "long": ("long", "긴소매", "긴팔"),
        "short": ("short", "반소매", "반팔"),
        "sleeveless": ("sleeveless", "민소매", "나시"),
    },
    "length": {
        "long": ("long", "롱", "긴"),
        "half": ("half", "미디", "중간"),
        "short": ("short", "숏", "짧은", "미니"),
    },
    "fit": {
        "slim": ("slim", "슬림"),
        "regular": ("regular", "레귤러", "정핏"),
        "loose": ("loose", "루즈", "여유"),
        "oversized": ("oversized", "oversize", "오버사이즈", "오버핏"),
    },
    "thickness": {
        "thin": ("thin", "얇음", "약간 얇음"),
        "medium": ("medium", "보통"),
        "thick": ("thick", "두꺼움", "약간두꺼움", "약간 두꺼움"),
    },
}


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


def attach_recommendation_explanations(
    rows: list[dict[str, Any]],
    parsed_attributes: dict[str, str] | None,
    target_description_ko: str,
) -> list[dict[str, Any]]:
    # 상품별 속성을 대조하는 보조 경로다. 현재 /search는 아래의 요청 단위 설명을 사용한다.
    return [
        {
            **row,
            **build_recommendation_explanation(
                row=row,
                parsed_attributes=parsed_attributes or {},
                target_description_ko=target_description_ko,
            ),
        }
        for row in rows
    ]


def generate_query_recommendation_explanation(
    *,
    query: str,
    parsed_attributes: dict[str, str] | None,
    target_description_ko: str,
    detail_target_description: str | None = None,
    target_description_reasoning: dict[str, Any] | None = None,
    has_reference_image: bool = False,
    model_name: str | None = None,
    allow_model_call: bool = True,
) -> RecommendationExplanationResult:
    # 실제 검색 결과 목록을 받지 않고, 요청 조건과 목표 설명으로 공통 추천 문구를 만든다.
    # 모델 호출을 못 하더라도 응답할 수 있도록 규칙 기반 문구를 먼저 준비한다.
    fallback = build_query_recommendation_explanation(
        parsed_attributes=parsed_attributes,
        target_description_ko=target_description_ko,
        detail_target_description=detail_target_description,
        has_reference_image=has_reference_image,
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
            target_description_ko=target_description_ko,
            detail_target_description=detail_target_description,
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
    parsed_attributes: dict[str, str] | None,
    target_description_ko: str,
    detail_target_description: str | None = None,
    has_reference_image: bool = False,
) -> str:
    # API 없이 속성과 문장 템플릿만으로 만드는 대체 문구다.
    attributes = parsed_attributes or {}
    target = target_description_ko.strip()
    conditions = _condition_phrases(attributes, target, detail_target_description)
    condition_text = _join_korean_phrases(conditions)

    if has_reference_image:
        # 이미지에서 유지하려는 특징을 앞 문장에 넣고, 같은 조건을 뒤에서 중복하지 않게 한다.
        image_focus = _reference_image_focus(attributes)
        condition_text = _join_korean_phrases(
            _conditions_without_reference_focus(conditions, image_focus)
        )
        if condition_text:
            opening = (
                f"기준 이미지의 {image_focus} 느낌은 유지하면서, "
                f"요청하신 {condition_text} 조건에 맞는 아이템을 찾았습니다."
            )
        else:
            opening = (
                f"기준 이미지의 {image_focus} 느낌을 바탕으로 "
                "요청에 맞는 아이템을 찾았습니다."
            )
    elif condition_text:
        opening = (
            f"요청하신 {condition_text} 조건에 맞는 아이템을 중심으로 "
            "찾았습니다."
        )
    elif target:
        opening = f"요청하신 '{target}' 스타일에 맞는 아이템을 중심으로 찾았습니다."
    else:
        opening = "요청하신 스타일에 맞는 아이템을 중심으로 찾았습니다."

    closing = _style_reason_sentence(
        attributes,
        target,
        detail_target_description,
        has_reference_image,
    )
    return f"{opening} {closing}"


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
    target_description_ko: str,
    detail_target_description: str | None,
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
        "target_description_ko": target_description_ko,
        "detail_target_description": detail_target_description or "",
        "reasoning_context": reasoning_context,
    }
    return (
        "You are a Korean UX copywriter for a fashion recommendation service.\n"
        "Write one user-facing recommendation reason from the input context.\n"
        "Return only the final Korean copy. No JSON. No markdown.\n\n"
        "Rules:\n"
        "1. Korean only, exactly two natural sentences.\n"
        "2. Sentence 1 summarizes the user's requested condition or occasion.\n"
        "3. Sentence 2 explains why the results fit, using mood, details, and use case.\n"
        "4. Do not explain individual products.\n"
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
            "thinkingConfig": {"thinkingBudget": 0},
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


def build_recommendation_explanation(
    row: dict[str, Any],
    parsed_attributes: dict[str, str],
    target_description_ko: str,
) -> dict[str, Any]:
    # 개별 상품 데이터가 주어지는 경우, 요청 속성과 일치한 항목을 설명 근거로 모은다.
    matched_attributes = [
        {
            "field": field,
            "label": ATTRIBUTE_LABELS.get(field, field),
            "value": _display_value(row.get(field), expected),
        }
        for field, expected in parsed_attributes.items()
        if expected and _matches_attribute(row.get(field), field, expected)
    ]

    if matched_attributes:
        matched_text = ", ".join(
            f"{attribute['label']} {attribute['value']}"
            for attribute in matched_attributes[:3]
        )
        reason = (
            f"요청하신 {matched_text} 조건과 잘 맞고, "
            "전체적인 스타일도 요청한 분위기와 잘 어울려 추천했어요."
        )
    elif target_description_ko.strip():
        reason = (
            "요청하신 스타일과 분위기가 잘 맞는 아이템이라 추천했어요."
        )
    else:
        reason = "입력한 요청과 전체적인 분위기가 잘 맞는 아이템이라 추천했어요."

    return {
        "recommendation_reason": reason,
        "recommendation_reason_details": matched_attributes,
    }


def _matches_attribute(row_value: Any, field: str, expected: str) -> bool:
    row_text = _normalize(row_value)
    if not row_text:
        return False

    variants = VALUE_ALIASES.get(field, {}).get(expected.lower(), (expected,))
    # 영어 속성명과 한국어 저장값 등의 별칭 중 하나가 상품 값에 포함되는지 확인한다.
    return any(_normalize(variant) in row_text for variant in variants)


def _display_value(row_value: Any, fallback: str) -> str:
    if isinstance(row_value, list):
        values = [str(value).strip() for value in row_value if str(value).strip()]
        return ", ".join(values) if values else fallback
    return str(row_value or fallback).strip()


def _attribute_phrase(field: str, value: str) -> str:
    label = ATTRIBUTE_LABELS.get(field, field)
    mapped_value = ATTRIBUTE_VALUE_LABELS.get(field, {}).get(value)
    if mapped_value and field != "category1":
        return mapped_value
    display_value = mapped_value or value
    return f"{label} {display_value}"


def _condition_phrases(
    attributes: dict[str, str],
    target_description_ko: str,
    detail_target_description: str | None,
) -> list[str]:
    phrases: list[str] = []
    category = _normalize(attributes.get("category1"))
    sleeve = _normalize(attributes.get("sleeve"))

    color = attributes.get("color")
    if color:
        phrases.append(_color_phrase(color))

    if attributes.get("fabric"):
        phrases.append(f"{attributes['fabric']} 소재")

    if category == "top":
        # 상의는 소매, 하의·원피스는 기장 중심으로 한국어 조건을 조합한다.
        phrases.append(_top_phrase(sleeve, target_description_ko))
        fit = _attribute_condition_phrase("fit", attributes.get("fit"))
        if fit:
            phrases.append(fit)
    elif category in {"pants", "skirt", "dress", "one-piece"}:
        category_phrase = _category_condition_phrase(category)
        length = _attribute_condition_phrase("length", attributes.get("length"))
        thickness = _attribute_condition_phrase("thickness", attributes.get("thickness"))
        fit = _attribute_condition_phrase("fit", attributes.get("fit"))
        phrases.extend(
            phrase
            for phrase in (length, thickness, fit, category_phrase)
            if phrase
        )
    else:
        phrases.extend(
            phrase
            for phrase in (
                _attribute_condition_phrase("sleeve", attributes.get("sleeve")),
                _attribute_condition_phrase("length", attributes.get("length")),
                _attribute_condition_phrase("thickness", attributes.get("thickness")),
                _attribute_condition_phrase("fit", attributes.get("fit")),
            )
            if phrase
        )
        if category:
            phrases.append(_category_condition_phrase(category))

    phrases.extend(_condition_detail_phrases(detail_target_description or target_description_ko))
    return _dedupe_phrases(phrases)


def _attribute_condition_phrase(field: str, value: str | None) -> str:
    if not value:
        return ""
    mapped_value = ATTRIBUTE_VALUE_LABELS.get(field, {}).get(value, value)
    if field == "sleeve" and value == "short":
        return "반팔"
    if field == "thickness" and value == "thin":
        return "얇은 두께"
    return mapped_value


def _category_condition_phrase(category: str) -> str:
    return {
        "top": "상의",
        "pants": "바지",
        "skirt": "스커트",
        "dress": "원피스",
        "one-piece": "원피스",
    }.get(category, category)


def _top_phrase(sleeve: str, target_description_ko: str) -> str:
    target = target_description_ko.lower()
    garment = "티셔츠" if "티셔츠" in target or "t-shirt" in target else "상의"
    if sleeve == "short":
        return f"반팔 {garment}"
    if sleeve == "long":
        return f"긴 소매 {garment}"
    if sleeve == "sleeveless":
        return f"민소매 {garment}"
    return garment


def _color_phrase(value: str) -> str:
    text = str(value).strip()
    normalized = _normalize(text)
    if normalized in {"bright", "light", "밝은색", "밝은 색"} or "밝" in normalized:
        return "밝은 색감"
    return f"{text} 색감"


def _detail_phrases(detail_target_description: str | None) -> list[str]:
    # 모델 호출 없이 미리 정한 디테일 단어를 찾아 설명에 쓸 한국어 표현으로 바꾼다.
    text = str(detail_target_description or "").lower()
    details: list[str] = []
    if "character" in text or "캐릭터" in text:
        details.append("캐릭터 그래픽")
    elif "graphic" in text or "그래픽" in text:
        details.append("그래픽 프린트")

    for keyword, label in DETAIL_LABELS.items():
        if keyword in {"character", "graphic"}:
            continue
        if keyword in text or label in text:
            details.append(label)
    return _dedupe_phrases(details)


def _condition_detail_phrases(detail_target_description: str | None) -> list[str]:
    details = _detail_phrases(detail_target_description)
    return ["후드 디테일" if detail == "후드" else detail for detail in details if detail == "후드"]


def _style_reason_sentence(
    attributes: dict[str, str],
    target_description_ko: str,
    detail_target_description: str | None,
    has_reference_image: bool,
) -> str:
    category = _normalize(attributes.get("category1"))
    details = _detail_phrases(detail_target_description or target_description_ko)
    fit = _normalize(attributes.get("fit"))
    thickness = _normalize(attributes.get("thickness"))

    if "캐릭터 그래픽" in details:
        return (
            "전면 그래픽이 있어 캐릭터 있는 티셔츠라는 의도와 잘 어울리고, "
            "캐주얼하고 귀여운 분위기의 데일리 아이템으로 활용하기 좋아 추천했습니다."
        )
    if "후드" in details and fit in {"wide", "loose", "oversized"}:
        return (
            "여유 있는 실루엣과 후드 디자인이 캐주얼한 분위기를 살려줘 "
            "데일리로 입기 좋아 추천했습니다."
        )
    if category == "pants" and thickness == "thin":
        if has_reference_image:
            return (
                "비슷한 실루엣을 바탕으로 더 가볍게 입기 좋은 팬츠를 "
                "우선해서 추천했습니다."
            )
        return "가볍게 입기 좋은 두께감이라 편한 데일리 팬츠로 활용하기 좋아 추천했습니다."
    if category in {"dress", "one-piece"}:
        return "원피스 특유의 단정한 분위기를 살리면서도 요청한 디테일과 잘 맞아 추천했습니다."
    if category == "top":
        return "상체 실루엣과 디테일이 요청한 분위기와 잘 맞아 데일리 상의로 활용하기 좋아 추천했습니다."
    return "요청한 조건과 전체적인 분위기가 잘 맞아 데일리 아이템으로 활용하기 좋아 추천했습니다."


def _reference_image_focus(attributes: dict[str, str]) -> str:
    length = _normalize(attributes.get("length"))
    if length == "long":
        return "긴 기장"
    if length == "short":
        return "짧은 기장"
    if attributes.get("color"):
        return "색감"
    return "전체적인"


def _conditions_without_reference_focus(
    conditions: list[str],
    image_focus: str,
) -> list[str]:
    if image_focus in {"긴 기장", "짧은 기장"}:
        return [condition for condition in conditions if condition != image_focus]
    return conditions


def _join_korean_phrases(phrases: list[str]) -> str:
    if not phrases:
        return ""
    if len(phrases) == 1:
        return phrases[0]
    return ", ".join(phrases[:-1]) + f", {phrases[-1]}"


def _dedupe_phrases(phrases: list[str]) -> list[str]:
    # 같은 표현은 한 번만 남기되 먼저 나타난 순서를 유지한다.
    seen: set[str] = set()
    deduped: list[str] = []
    for phrase in phrases:
        text = str(phrase or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


def _normalize(value: Any) -> str:
    if isinstance(value, list):
        value = " ".join(str(item) for item in value)
    return re.sub(r"\s+", " ", str(value or "").lower()).strip()
