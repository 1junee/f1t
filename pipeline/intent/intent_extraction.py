from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from threading import Lock
from dataclasses import dataclass
from typing import Any

import requests

from ..retrieval.candidate_selection import ParsedTextAttributes, parse_text_attributes

if __package__:
    from ..env import load_pipeline_env
else:
    from env import load_pipeline_env

load_pipeline_env()

DEFAULT_INTENT_MODEL = "gemini-3.5-flash-lite"
INTENT_MODEL_ENV = "GEMINI_INTENT_MODEL"
LEGACY_METADATA_MODEL_ENV = "GEMINI_METADATA_MODEL"
LEGACY_GEMINI_MODEL_ENV = "GEMINI_MODEL"

# 한 번의 모델 호출 안에서 '명시된 속성 선택 → 값 추출'을 수행하도록 지시한다.
# 검색 후보를 과도하게 줄이지 않도록 사용자가 말하지 않은 조건은 추가하지 않는다.
INFERENCE_PROMPT_TEMPLATE = """You are a fashion search intent analyzer.

User query: {user_query}

{image_instruction}

---

Step 1 — Identify mentioned attributes (think first):
Read the query carefully. List ONLY the attributes the user explicitly stated. Do NOT add anything the user did not say.
Ask yourself: "Did the user actually say this?" If not, exclude it.
Keep only positively requested values. Omit negated or excluded values; never turn them into positive search conditions.
If a replacement is stated ("검정 말고 흰색", "면 제외하고 린넨"), keep only the requested replacement.
If an attribute is only excluded or unspecified, omit it or return null. Do not invent an alternative.

CRITICAL — Do NOT infer attributes from the garment type name itself.
Examples of forbidden inference:
- User says "티셔츠" → do NOT add sleeve:short (user didn't mention sleeve)
- User says "치마" or "원피스" → do NOT add sex:female (user didn't mention sex)
- User says "슬렉스" → do NOT add length:long (user didn't mention length)
- User says "후드" → do NOT add sex:unisex (user didn't mention sex)
Only category1 may be derived from the garment type name. Nothing else.

Step 2 — Map to schema:
For each attribute identified in Step 1, assign a value using the schema below.

Fixed attributes (use predefined values only):
- category1: top / pants / dress / skirt
- sleeve: long / short / sleeveless  → only for tops
- length: long / half / short  → only for pants / skirt / dress
- sex: male / female / unisex
- season: spring / summer / fall / winter
- stretch: yes / no
- thickness: thin / medium / thick
- fit: slim / regular / loose / oversize / wide

Explicit semantic attributes (short strings, not predefined enums):
- color: the positively requested color, preferably a concise lowercase English name (e.g. navy, ivory)
- fabric: the positively requested material, preferably a concise lowercase English name (e.g. cotton, linen, wool)
These two attributes are similarity inputs, not exact database filters.
Do not infer color or fabric from garment type, style, occasion, or a reference image unless the user explicitly asks for that image attribute.
Examples:
- "네이비 면 소재 셔츠" → category1:top, color:navy, fabric:cotton
- "검정과 면은 제외한 셔츠" → category1:top; omit color and fabric
- "검정과 면 말고 아이보리 린넨 셔츠" → category1:top, color:ivory, fabric:linen
- "와이드 핏 바지" → category1:pants, fit:wide
Return color and fabric as strings or null, never arrays or objects.

Step 3 — Output:
Respond with valid JSON only. Include ONLY the attributes from Step 1. No extras.

{{
  "<attribute>": "<value>",
  "reasoning": "한국어로, 각 속성에 대해 사용자가 실제로 말한 표현을 인용하여 설명. 사용자가 직접 말하지 않은 속성은 절대 포함하지 말 것."
}}"""

_VALUE_ALIASES = {
    ("category1", "dress"): "one-piece",
    ("fit", "oversize"): "oversized",
}

_ALLOWED_VALUES: dict[str, set[str]] = {
    "category1": {"top", "pants", "skirt", "dress", "one-piece"},
    "sex": {"male", "female", "unisex"},
    "sleeve": {"short", "long", "sleeveless"},
    "length": {"short", "half", "long"},
    "season": {"spring", "summer", "fall", "winter"},
    "stretch": {"yes", "no"},
    "thickness": {"thin", "medium", "thick"},
    "fit": {"slim", "regular", "loose", "oversized", "oversize", "wide"},
}


@dataclass(frozen=True)
class ConservativeMetadataExtraction:
    # 속성값과 처리 상태를 함께 반환해, 호출자가 모델 결과와 대체 결과를 구분할 수 있게 한다.
    attributes: ParsedTextAttributes
    status: str
    method: str
    prompt_id: str
    raw_output: str
    error: str | None = None
    model_name: str | None = None
    requested_provider: str = "gemini"
    provider: str = "gemini"

    def to_metadata(self) -> dict[str, Any]:
        # API 설명용 요약에는 원문 응답 전체 대신 사용한 필드·모델·오류 정보를 담는다.
        return {
            "status": self.status,
            "method": self.method,
            "prompt_id": self.prompt_id,
            "fields": sorted(self.attributes.to_dict()),
            "error": self.error,
            "model": self.model_name,
            "requested_provider": self.requested_provider,
            "provider": self.provider,
        }


def resolve_intent_model(model_name: str | None = None) -> str:
    # 함수 인자가 최우선이며, 없으면 전용 설정 → 기존 공통 설정 → 기본 모델 순서로 정한다.
    return (
        model_name
        or os.getenv(INTENT_MODEL_ENV)
        or os.getenv(LEGACY_METADATA_MODEL_ENV)
        or os.getenv(LEGACY_GEMINI_MODEL_ENV)
        or DEFAULT_INTENT_MODEL
    )


class IntentExtractionModule:
    def __init__(self, api_key: str | None = None, model_name: str | None = None):
        from google import genai
        from google.genai import types

        self._types = types
        # 이 SDK의 타임아웃 단위는 밀리초다. 설명 생성 REST 클라이언트의 초 단위와 구분한다.
        timeout_ms = int(os.getenv("GEMINI_REQUEST_TIMEOUT_MS", "60000"))
        self.client = genai.Client(
            api_key=api_key or os.getenv("GEMINI_API_KEY"),
            http_options=types.HttpOptions(timeout=timeout_ms),
        )
        self.model_name = resolve_intent_model(model_name)
        # JSON 형식 출력을 요청하지만, 실제 필드와 값의 검사는 응답을 받은 뒤 다시 수행한다.
        self.config = types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=1024,
            response_mime_type="application/json",
        )

    def infer(
        self,
        query: str,
        image_bytes: bytes | None = None,
        image_mime_type: str | None = None,
        image_url: str | None = None,
    ) -> dict:
        # 파일 바이트가 있으면 URL보다 우선한다. URL만 주어졌을 때는 직접 내려받는다.
        image_part = None
        if image_bytes:
            image_part = self._types.Part.from_bytes(
                data=image_bytes,
                mime_type=image_mime_type or "image/jpeg",
            )
        elif image_url:
            try:
                resp = requests.get(image_url, timeout=10)
                resp.raise_for_status()
                image_part = self._types.Part.from_bytes(
                    data=resp.content,
                    mime_type=resp.headers.get("content-type") or "image/jpeg",
                )
            except Exception:
                # 이미지 다운로드 실패는 텍스트만 해석하는 경로로 이어진다.
                image_part = None

        if image_part:
            # 이미지는 요청한 속성의 값을 확인하는 용도이며, 새 필터를 만드는 근거는 아니다.
            image_instruction = (
                "Reference image provided: use the image ONLY to determine the value of "
                "attributes the user explicitly mentioned in the query. Do NOT extract or add any "
                "attribute the user did not mention, even if it is clearly visible in the image."
            )
        else:
            image_instruction = "No image provided: extract attributes from the text query only."

        prompt = INFERENCE_PROMPT_TEMPLATE.format(
            user_query=query, image_instruction=image_instruction
        )

        parts = []
        if image_part:
            parts.append(image_part)
        parts.append(self._types.Part.from_text(text=prompt))
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[self._types.Content(role="user", parts=parts)],
            config=self.config,
        )
        parsed = _parse_json(str(response.text or "").strip())
        # 파싱 실패를 예외로 올려 공통 대체 경로가 처리하게 한다.
        if parsed is None:
            raise RuntimeError("Metadata response JSON parse failed.")
        return parsed


# 같은 모델의 API 클라이언트를 요청마다 다시 만들지 않도록 재사용한다.
_module_cache: dict[str, IntentExtractionModule] = {}


def _get_module(model_name: str | None = None) -> IntentExtractionModule:
    resolved_model = resolve_intent_model(model_name)
    if resolved_model not in _module_cache:
        _module_cache[resolved_model] = IntentExtractionModule(model_name=resolved_model)
    return _module_cache[resolved_model]


class ProviderIntentExtractionModule:
    """기존 제공자 클라이언트에 동일한 보수적 속성 추출 프롬프트를 전달한다."""

    def __init__(self, provider: str, model_name: str | None = None):
        from ..target_description.target_description_generation import (
            OpenAIReasoningClient, Qwen3VLReasoningClient,
        )
        client_type = OpenAIReasoningClient if provider == "openai" else Qwen3VLReasoningClient
        self.client = client_type(model_name=model_name)
        self.model_name = self.client.model_name
        self._inference_lock = Lock()

    def infer(self, query, image_bytes=None, image_mime_type=None, image_url=None):
        if not image_bytes and image_url:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            image_bytes = response.content
            image_mime_type = response.headers.get("content-type") or "image/jpeg"
        image_instruction = (
            "Reference image provided: use it ONLY for attributes explicitly requested by the user. "
            "Do NOT add attributes merely because they are visible."
            if image_bytes else "No image provided: extract attributes from the text query only."
        )
        # 캐시한 로컬 모델을 여러 요청이 동시에 초기화·실행하지 않도록 보호한다.
        with self._inference_lock:
            output = self.client.generate_with_prompt(
                prompt=INFERENCE_PROMPT_TEMPLATE.format(user_query=query, image_instruction=image_instruction),
                image_bytes=image_bytes,
                image_mime_type=image_mime_type,
            )
        parsed = _parse_json(output)
        if parsed is None:
            raise RuntimeError("Metadata response JSON parse failed.")
        return parsed


@lru_cache(maxsize=8)
def _get_provider_module(provider: str, model_name: str) -> ProviderIntentExtractionModule:
    return ProviderIntentExtractionModule(provider, model_name)


def extract_conservative_metadata(
    query: str,
    image_bytes: bytes | None = None,
    image_mime_type: str | None = None,
    image_url: str | None = None,
    # kept for call-site compat, ignored
    client: Any = None,
    model_name: str | None = None,
    provider: str = "gemini",
) -> ConservativeMetadataExtraction:
    # 외부에서 호출하는 진입점: 요청 문장·선택적 이미지를 받아 정규화된 속성을 반환한다.
    provider = provider.strip().lower()
    if provider not in {"gemini", "openai", "qwen"}:
        raise ValueError("provider must be gemini, openai, or qwen.")
    from ..target_description.target_description_generation import OPENAI_MODEL_NAME, QWEN_FALLBACK_MODEL_NAME
    if provider == "gemini":
        resolved_model = resolve_intent_model(model_name)
    elif provider == "openai":
        resolved_model = model_name or os.getenv("OPENAI_MODEL") or OPENAI_MODEL_NAME
    else:
        resolved_model = (model_name or os.getenv("QWEN_FALLBACK_MODEL")
                          or os.getenv("QWEN3_VL_MODEL") or QWEN_FALLBACK_MODEL_NAME)
    try:
        if provider == "gemini":
            module = _get_module(resolved_model)
        else:
            module = _get_provider_module(provider, resolved_model)
            resolved_model = module.model_name
        raw = module.infer(
            query=query,
            image_bytes=image_bytes,
            image_mime_type=image_mime_type,
            image_url=image_url,
        )
        raw_str = json.dumps(raw, ensure_ascii=False)
        # 모델의 자유로운 출력값을 검색 모듈이 사용하는 속성 구조로 정규화한다.
        attrs = _map_to_parsed_attributes(raw)
        return ConservativeMetadataExtraction(
            attributes=attrs,
            status="ok",
            method=f"{provider}_intent_analysis",
            prompt_id="intent_analysis_v1",
            raw_output=raw_str,
            model_name=resolved_model,
            requested_provider=provider,
            provider=provider,
        )
    except Exception as exc:
        # 모델 호출·파싱이 실패하면 텍스트 규칙 파서의 결과와 degraded 상태를 반환한다.
        return ConservativeMetadataExtraction(
            attributes=parse_text_attributes(query),
            status="degraded",
            method="rule_based_keyword_parser_fallback",
            prompt_id="intent_analysis_v1",
            raw_output="",
            error=str(exc),
            model_name=resolved_model,
            requested_provider=provider,
            provider="rule_based",
        )


def analyze_fashion_intent(text_input: str, image_input: str | None = None) -> dict[str, str]:
    """파일 경로·URL 입력을 지원하는 보조 래퍼. HTTP /analyze는 추출 함수를 직접 호출한다."""
    image_url = image_input if image_input and image_input.startswith(("http://", "https://")) else None
    image_bytes = None
    if image_input and not image_url and os.path.exists(image_input):
        with open(image_input, "rb") as image_file:
            image_bytes = image_file.read()

    extraction = extract_conservative_metadata(
        query=text_input,
        image_bytes=image_bytes,
        image_url=image_url,
    )
    return extraction.attributes.to_dict()


def _map_to_parsed_attributes(raw: dict) -> ParsedTextAttributes:
    # reasoning 같은 부가 설명은 제외하고 ParsedTextAttributes에 정의된 필드만 읽는다.
    values: dict[str, str | None] = {}
    for field in ParsedTextAttributes.__dataclass_fields__:
        values[field] = _normalize(field, raw.get(field))
    return ParsedTextAttributes(**values)


def _normalize(field: str, value: Any) -> str | None:
    # 자유문자열 속성도 계약을 검사한다. 누락·잘못된 형식을 원문 키워드로 보충하지 않는다.
    if field in {"color", "fabric"}:
        if not isinstance(value, str):
            return None
        normalized = re.sub(r"\s+", " ", value).strip().lower()
        return None if normalized in {"", "null", "none", "-"} else normalized
    # 모델이 출력한 여러 '값 없음' 표현을 None 하나로 통일한다.
    if value in (None, "", "null", "none", "None", "-", []):
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    lowered = normalized.lower()
    lowered = _VALUE_ALIASES.get((field, lowered), lowered)
    # 고정 속성은 허용된 값만 남기고, 알 수 없는 값은 필터에 넣지 않는다.
    if field in _ALLOWED_VALUES and lowered not in _ALLOWED_VALUES[field]:
        return None
    return lowered if field in _ALLOWED_VALUES else normalized


def _parse_json(text: str) -> dict | None:
    # 코드 블록 또는 중괄호 부분을 파싱하고, 사전(dict)으로 읽힌 결과만 남긴다.
    try:
        m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if m:
            data = json.loads(m.group(1))
            return data if isinstance(data, dict) else None
        m = re.search(r"(\{.*\})", text, re.DOTALL)
        if m:
            data = json.loads(m.group(1))
            return data if isinstance(data, dict) else None
    except Exception:
        return None
    return None
