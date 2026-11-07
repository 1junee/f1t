"""Target-description reasoning clients.

These clients turn text,
image, or text+image input into a single target description for Gemini embedding retrieval.
"""
from __future__ import annotations

import base64
import os
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from io import BytesIO
from typing import Any

from PIL import Image

if __package__:
    from ..env import load_pipeline_env
else:  # pragma: no cover - allows direct imports from pipeline/
    from env import load_pipeline_env

load_pipeline_env()

# --- Prompts (inlined from target_description_prompts) ---

TARGETDESCRIPTION_FASHION_SYSTEM_PROMPT = (
    "You are a professional fashion stylist and visual search expert. "
    "You have deep expertise in clothing categories, silhouettes, materials, "
    "patterns, color styling, seasonal occasions, and fashion item description "
    "for image-based retrieval."
)

# 참고 이미지 경로에서는 이미지의 특징을 유지하되, 충돌하는 사용자 문장을 우선하도록 한다.
TARGETDESCRIPTION_REFERENCE_IMAGE_PROMPT = """You are a fashion search description expert and professional fashion stylist. You are given a reference fashion image and optional manipulation text. Your goal is to generate a target image description for retrieval by combining the visual evidence from the image with the user's text.

## Guidelines on generating the Original Image Description
- Ensure the original image description is thorough, capturing all visible objects, attributes, and elements.
- The original image description should be as accurate as possible, reflecting the content of the image.
- If the original image does not clearly show a clothing or fashion item, set "Is Fashion Image" to false.
- A valid fashion image should clearly show clothing, shoes, bags, or wearable fashion accessories.

## Guidelines for text + image search
- Treat the image as the reference item and the text as a refinement, edit, or override.
- If the text conflicts with the image, the explicit text request wins.
- Preserve visible item structure unless the text changes it: category, sleeve, length, silhouette, and notable visual details.
- When the query clearly specifies sleeve or length, reflect those details in the target item description.
- Color, fabric, and notable visible details may be freely described in natural language because exact candidate restriction is unsafe.
- Generate one final target item description, not multiple alternatives.
- If the request is clearly unrelated to fashion (e.g. food, travel, technology), set "Is Fashion Query" to false.

## Guidelines on generating the Thoughts
- In your Thoughts, explain your understanding of the manipulation intents and how you formulated the target
image description.
- Provide insight into how you interpreted the manipulation intent in detail in the manipulation text.
- Discuss how the manipulation intent influenced which elements of the original image you focused.

## Guidelines on generating the Reflections
- In your Reflections, summarize how the manipulation intent influenced your approach to transforming the
original image description.
- Explain how the changes made reflect the specific semantic, Highlight key decisions that were made to
preserve the coherence and context of the original image while meeting the manipulation intent.
- Reflect on the impact these changes have on the overall appearance or narrative of the image.
- Ensure that your reflections provide a concise yet insightful summary of the considerations and strategies
applied in crafting the target description, offering a logical connection between the original and final content.

## Guidelines on generating Target Image Description
- The target image description you generate should be complete and can cover various semantic aspects.
- The target image description only contains the target image content and needs to be as simple as possible.
Minimize aesthetic descriptions as much as possible.

## Output Format
Return ONLY a valid JSON object with the following structure:
{
    "Is Fashion Image": true or false,
    "Is Fashion Query": true or false,
    "Original Image Description": "...",
    "Thoughts": "...",
    "Reflections": "...",
    "Target Image Description": "..."
}"""

# 텍스트만 있을 때는 참조 이미지의 특징을 가정하지 않고, 요청한 조건 중심으로 설명하게 한다.
TARGETDESCRIPTION_TEXT_ONLY_PROMPT = """You are a fashion search description expert and professional fashion stylist. You are given a user's fashion search request. Your goal is to generate a target image description that can be used for image retrieval.

## Guidelines on generating the Original Image Description
- No reference image is provided for text-only search.
- Set "Original Image Description" to an empty string.

## Guidelines for text-only search
- Only describe attributes the user explicitly mentioned. Do not infer or add visual details not stated in the request.
- If the request is occasion-based with no specific attributes, keep the description generic and avoid adding assumed silhouette or style details.
- Focus on visible fashion attributes such as category, color, length, sleeve, fabric, and pattern.
- Keep the target image description concise and retrieval-friendly.
- Do not mention unavailable alternatives or multiple options.
- If the request is clearly unrelated to fashion (e.g. food, travel, technology), set "Is Fashion Query" to false.

## Guidelines on generating the Thoughts
- In your Thoughts, explain how you interpreted the user's search request.
- Mention which visible fashion attributes you inferred directly or practically from the request.
- Keep the reasoning focused on retrieval-relevant details.

## Guidelines on generating the Reflections
- In your Reflections, summarize how the request shaped the final target description.
- Highlight key decisions made to preserve coherence while keeping the description useful for visual search.

## Guidelines on generating Target Image Description
- The target image description you generate should be complete and can cover various semantic aspects.
- The target image description only contains the target image content and needs to be as simple as possible.
- Minimize aesthetic descriptions as much as possible.

## Output Format
Return ONLY a valid JSON object with the following structure:
{
    "Is Fashion Query": true or false,
    "Original Image Description": "",
    "Thoughts": "...",
    "Reflections": "...",
    "Target Image Description": "..."
}"""


def build_target_description_prompt(
    request_text: str,
    has_image: bool,
) -> str:
    # 이미지가 있으면 유지할 특징과 수정할 특징을 합치고, 없으면 요청 문장만 해석한다.
    if has_image:
        return (
            f"{TARGETDESCRIPTION_REFERENCE_IMAGE_PROMPT}\n\n"
            f"<Input>\n"
            f"Original Image: [Image 1]\n"
            f"Manipulation text: {request_text}\n"
            f"<Response>\n"
        )

    return (
        f"{TARGETDESCRIPTION_TEXT_ONLY_PROMPT}\n\n"
        f"<Input>\n"
        f"User request: {request_text}\n"
        f"<Response>\n"
    )


# 아래 클라이언트는 검색용 설명문을 생성한다. 임베딩은 retrieval 모듈에서 별도로 만든다.
# 어떤 provider를 쓰고 실패 시 무엇으로 전환할지는 recommendation_service.py가 결정한다.

MODEL_NAME = "Qwen/Qwen3-VL-32B-Instruct"
GEMINI_MODEL_NAME = "gemini-3.5-flash"
TARGET_DESCRIPTION_MODEL_ENV = "GEMINI_TARGET_DESCRIPTION_MODEL"
LEGACY_GEMINI_MODEL_ENV = "GEMINI_MODEL"
QWEN_FALLBACK_MODEL_NAME = "Qwen/Qwen3.5-0.8B"
QWEN3_VL_MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
OPENAI_MODEL_NAME = "gpt-5.4-nano"
DEFAULT_IMAGE_ONLY_QUERY = "Find a fashion item visually similar to the reference image."


def resolve_target_description_model(model_name: str | None = None) -> str:
    # 메타데이터 모델과 별도로 설명문 생성 모델만 바꿀 수 있는 설정 경로다.
    return (
        model_name
        or os.getenv(TARGET_DESCRIPTION_MODEL_ENV)
        or os.getenv(LEGACY_GEMINI_MODEL_ENV)
        or GEMINI_MODEL_NAME
    )


def parse_json_from_text(text):
    """LLM 출력에서 JSON 추출"""
    # 본문 전체를 그대로 믿지 않고 JSON 부분만 읽는다. 실패하면 호출자가 원문 사용 여부를 정한다.
    try:
        match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        match = re.search(r'(\{.*\})', text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        return None
    except Exception:
        return None


@dataclass(frozen=True)
class ReasoningWarning:
    code: str
    label: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "label": self.label,
            "message": self.message,
        }


class InvalidSearchInput(ValueError):
    """패션 검색에 맞지 않는 입력. 제공자 장애와 달리 원문 검색으로 대체하지 않는다."""


class ReasoningProviderError(RuntimeError):
    # 사용자에게 보여줄 경고와 디버깅용 상세 정보를 분리해 보관한다.
    def __init__(
        self,
        code: str,
        label: str,
        message: str,
        detail: str | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.label = label
        self.message = message
        self.detail = detail or ""

    def to_warning(self) -> ReasoningWarning:
        # 경고 응답에는 원격 서버의 상세 오류 본문을 포함하지 않는다.
        return ReasoningWarning(
            code=self.code,
            label=self.label,
            message=self.message,
        )


def normalize_reasoning_result(parsed: dict[str, Any], fallback: str) -> dict[str, Any]:
    # 제공자별 응답을 공통 키로 맞춘다. 검색 벡터의 입력은 Target Image Description이다.
    return {
        "Original Image Description": str(parsed.get("Original Image Description", "")),
        "Thoughts": str(parsed.get("Thoughts", "")),
        "Reflections": str(parsed.get("Reflections", "")),
        "Target Image Description": str(
            parsed.get("Target Image Description") or fallback
        ),
    }


class GeminiReasoningClient:
    # 기본 설명 생성 경로: 텍스트와 선택적 이미지를 Gemini REST API에 전달한다.
    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        timeout: int = 60,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = resolve_target_description_model(model_name)
        self.timeout = timeout

        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is required for target-description search.")

    def run(
        self,
        manipulation_text: str,
        image_bytes: bytes | None = None,
        image_mime_type: str | None = None,
        image_path: str | None = None,
    ) -> dict[str, str]:
        normalized_text = manipulation_text.strip()

        if image_path:
            # 로컬 실행용 경로 입력도 HTTP 업로드와 같은 바이트 입력으로 맞춘다.
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")

            with open(image_path, "rb") as image_file:
                image_bytes = image_file.read()

            suffix = os.path.splitext(image_path)[1].lower()
            image_mime_type = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".webp": "image/webp",
            }.get(suffix, image_mime_type or "image/jpeg")

        if not normalized_text:
            # 이미지만 올린 요청도 모델에는 '비슷한 패션 아이템'이라는 기본 요청을 전달한다.
            normalized_text = DEFAULT_IMAGE_ONLY_QUERY

        prompt = self._build_prompt(
            manipulation_text=normalized_text,
            has_image=bool(image_bytes),
        )
        response_text = self._generate(
            prompt=prompt,
            image_bytes=image_bytes,
            image_mime_type=image_mime_type,
        )
        # 이미지 묘사·해석·최종 설명을 한 응답에서 받으며, 필드마다 모델을 다시 호출하지 않는다.
        parsed = parse_json_from_text(response_text)

        # 모델이 명시적으로 부적절하다고 답했을 때만 입력 오류를 발생시킨다.
        if parsed and image_bytes and self._is_explicit_false(parsed.get("Is Fashion Image")):
            raise InvalidSearchInput("옷이 정확히 보이는 사진을 첨부하세요.")

        if parsed and self._is_explicit_false(parsed.get("Is Fashion Query")):
            raise InvalidSearchInput("패션 관련 검색어를 입력해주세요.")

        if parsed and parsed.get("Target Image Description"):
            return normalize_reasoning_result(parsed, fallback=normalized_text)

        # 응답에 설명문이 없으면 원래 요청 문장을 검색용 설명으로 사용한다.
        return {
            "Original Image Description": "",
            "Thoughts": "",
            "Reflections": "",
            "Target Image Description": normalized_text,
        }

    # 아래 번역·키워드 메서드는 보조 기능이다. 현재 검색의 번역은 서비스 모듈에서 수행한다.
    def translate_to_korean(self, text: str) -> str:
        normalized_text = text.strip()
        if not normalized_text:
            return ""

        return self._generate(
            prompt=(
                "You are a fashion translator. Translate the following English fashion item description "
                "into natural, fluent Korean as if written by a Korean fashion editor. "
                "Keep brand names, English loan words commonly used in Korean fashion (e.g. 오버사이즈, 크루넥) as-is. "
                "Do not summarize or omit any detail. Return ONLY the Korean translation.\n\n"
                f"{normalized_text}"
            ),
        ).strip()

    def translate_to_english(self, text: str) -> str:
        normalized_text = text.strip()
        if not normalized_text:
            return ""

        return self._generate(
            prompt=(
                "You are a fashion translator. Translate the following Korean fashion search query "
                "into natural English suitable for a CLIP image retrieval model. "
                "Return ONLY the English translation.\n\n"
                f"{normalized_text}"
            ),
        ).strip()

    def extract_keywords(self, text: str) -> list[str]:
        normalized_text = text.strip()
        if not normalized_text:
            return []

        response = self._generate(
            prompt=(
                "Extract key fashion attributes from the following description as short Korean tags. "
                "Categories to extract (if present): 색상, 카테고리, 소재, 핏, 소매, 넥라인, 패턴, 기장, 무드. "
                "Format each tag as '속성: 값' (e.g. '색상: 블랙', '핏: 오버사이즈'). "
                "Return ONLY a JSON array of strings, max 8 tags.\n\n"
                f"{normalized_text}"
            ),
        ).strip()

        parsed = parse_json_from_text(response)
        if isinstance(parsed, list):
            return [str(k) for k in parsed if k]
        return []

    def generate_with_prompt(
        self,
        prompt: str,
        image_bytes: bytes | None = None,
        image_mime_type: str | None = None,
    ) -> str:
        return self._generate(prompt, image_bytes, image_mime_type)

    def _build_prompt(self, manipulation_text: str, has_image: bool) -> str:
        return build_target_description_prompt(manipulation_text, has_image)

    def _generate(
        self,
        prompt: str,
        image_bytes: bytes | None = None,
        image_mime_type: str | None = None,
    ) -> str:
        parts: list[dict[str, Any]] = []

        if image_bytes:
            # 별도 업로드 없이 이미지 바이트를 base64로 넣어 텍스트와 함께 보낸다.
            parts.append(
                {
                    "inline_data": {
                        "mime_type": image_mime_type or "image/jpeg",
                        "data": base64.b64encode(image_bytes).decode("ascii"),
                    }
                }
            )

        parts.append({"text": prompt})

        request_body = {
            # 공통 역할 지시와 사용자별 입력을 나눠 API가 요구하는 요청 구조를 만든다.
            "systemInstruction": {"parts": [{"text": TARGETDESCRIPTION_FASHION_SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": 8192,
            },
        }
        request_data = json.dumps(request_body).encode("utf-8")
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_name}:generateContent?key={self.api_key}"
        )
        request = urllib.request.Request(
            url,
            data=request_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_json = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # 제공자 오류를 공통 예외로 바꾸면 상위 서비스가 대체 경로를 선택할 수 있다.
            detail = exc.read().decode("utf-8", errors="replace")
            code, label, message = self._classify_http_error(exc.code, detail)
            raise ReasoningProviderError(code, label, message, detail=detail) from exc
        except urllib.error.URLError as exc:
            raise ReasoningProviderError(
                "gemini_unavailable",
                "Gemini 연결 오류",
                "Gemini 서버에 연결하지 못해 fallback으로 검색했습니다.",
                detail=str(exc.reason),
            ) from exc
        except json.JSONDecodeError as exc:
            raise ReasoningProviderError(
                "gemini_bad_response",
                "Gemini 응답 오류",
                "Gemini 응답을 해석하지 못해 fallback으로 검색했습니다.",
                detail=str(exc),
            ) from exc

        try:
            return response_json["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise ReasoningProviderError(
                "gemini_bad_response",
                "Gemini 응답 오류",
                "Gemini 응답 형식이 예상과 달라 fallback으로 검색했습니다.",
                detail=json.dumps(response_json, ensure_ascii=False)[:1000],
            ) from exc

    def _classify_http_error(self, status_code: int, detail: str) -> tuple[str, str, str]:
        # 한도 초과를 일반 연결 오류와 구분해 화면에서 원인을 안내할 수 있게 한다.
        normalized_detail = detail.lower()

        if (
            status_code == 429
            or "resource_exhausted" in normalized_detail
            or "quota" in normalized_detail
        ):
            return (
                "gemini_quota_exceeded",
                "Gemini 한도 초과",
                "Gemini 요청 한도를 초과해 fallback으로 검색했습니다.",
            )

        return (
            "gemini_unavailable",
            "Gemini 오류",
            "Gemini 요청을 처리하지 못해 fallback으로 검색했습니다.",
        )

    def _is_explicit_false(self, value: Any) -> bool:
        # JSON 불리언뿐 아니라 모델이 문자열로 반환한 false도 같은 의미로 처리한다.
        if value is False:
            return True

        if isinstance(value, str):
            return value.strip().lower() in {"false", "no", "0"}

        return False


class OpenAIReasoningClient:
    # provider=openai로 선택했을 때 사용하는 대안 경로이며, 결과 키는 Gemini와 같다.
    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        timeout: int = 60,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name or os.getenv("OPENAI_MODEL", OPENAI_MODEL_NAME)
        self.timeout = timeout

        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI reasoning.")

    def run(
        self,
        manipulation_text: str,
        image_bytes: bytes | None = None,
        image_mime_type: str | None = None,
        image_path: str | None = None,
    ) -> dict[str, str]:
        normalized_text = manipulation_text.strip()

        if image_path:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")

            with open(image_path, "rb") as f:
                image_bytes = f.read()

            suffix = os.path.splitext(image_path)[1].lower()
            image_mime_type = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".webp": "image/webp",
            }.get(suffix, image_mime_type or "image/jpeg")

        if not normalized_text:
            normalized_text = DEFAULT_IMAGE_ONLY_QUERY

        prompt = self._build_prompt(manipulation_text=normalized_text, has_image=bool(image_bytes))
        response_text = self._generate(prompt=prompt, image_bytes=image_bytes, image_mime_type=image_mime_type)
        parsed = parse_json_from_text(response_text)

        if parsed and image_bytes and self._is_explicit_false(parsed.get("Is Fashion Image")):
            raise InvalidSearchInput("옷이 정확히 보이는 사진을 첨부하세요.")

        if parsed and self._is_explicit_false(parsed.get("Is Fashion Query")):
            raise InvalidSearchInput("패션 관련 검색어를 입력해주세요.")

        if parsed and parsed.get("Target Image Description"):
            return normalize_reasoning_result(parsed, fallback=normalized_text)

        return {
            "Original Image Description": "",
            "Thoughts": "",
            "Reflections": "",
            "Target Image Description": normalized_text,
        }

    def generate_with_prompt(
        self,
        prompt: str,
        image_bytes: bytes | None = None,
        image_mime_type: str | None = None,
    ) -> str:
        return self._generate(prompt, image_bytes, image_mime_type)

    def _build_prompt(self, manipulation_text: str, has_image: bool) -> str:
        return build_target_description_prompt(manipulation_text, has_image)

    def _generate(
        self,
        prompt: str,
        image_bytes: bytes | None = None,
        image_mime_type: str | None = None,
    ) -> str:
        # 같은 프롬프트라도 제공자마다 이미지와 메시지를 담는 API 형식은 다르다.
        content: list[dict[str, Any]] = []

        if image_bytes:
            mime = image_mime_type or "image/jpeg"
            b64 = base64.b64encode(image_bytes).decode("ascii")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })

        content.append({"type": "text", "text": prompt})

        request_body = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": TARGETDESCRIPTION_FASHION_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            "temperature": 0,
            "max_tokens": 8192,
        }
        request_data = json.dumps(request_body).encode("utf-8")
        request = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=request_data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_json = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            code, label, message = self._classify_http_error(exc.code, detail)
            raise ReasoningProviderError(code, label, message, detail=detail) from exc
        except urllib.error.URLError as exc:
            raise ReasoningProviderError(
                "openai_unavailable",
                "OpenAI 연결 오류",
                "OpenAI 서버에 연결하지 못해 fallback으로 검색했습니다.",
                detail=str(exc.reason),
            ) from exc
        except json.JSONDecodeError as exc:
            raise ReasoningProviderError(
                "openai_bad_response",
                "OpenAI 응답 오류",
                "OpenAI 응답을 해석하지 못해 fallback으로 검색했습니다.",
                detail=str(exc),
            ) from exc

        try:
            return response_json["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise ReasoningProviderError(
                "openai_bad_response",
                "OpenAI 응답 오류",
                "OpenAI 응답 형식이 예상과 달라 fallback으로 검색했습니다.",
                detail=json.dumps(response_json, ensure_ascii=False)[:1000],
            ) from exc

    def translate_to_korean(self, text: str) -> str:
        normalized_text = text.strip()
        if not normalized_text:
            return ""

        return self._generate(
            prompt=(
                "You are a fashion translator. Translate the following English fashion item description "
                "into natural, fluent Korean as if written by a Korean fashion editor. "
                "Keep brand names, English loan words commonly used in Korean fashion (e.g. 오버사이즈, 크루넥) as-is. "
                "Do not summarize or omit any detail. Return ONLY the Korean translation.\n\n"
                f"{normalized_text}"
            ),
        ).strip()

    def translate_to_english(self, text: str) -> str:
        normalized_text = text.strip()
        if not normalized_text:
            return ""

        return self._generate(
            prompt=(
                "You are a fashion translator. Translate the following Korean fashion search query "
                "into natural English suitable for a CLIP image retrieval model. "
                "Return ONLY the English translation.\n\n"
                f"{normalized_text}"
            ),
        ).strip()

    def extract_keywords(self, text: str) -> list[str]:
        normalized_text = text.strip()
        if not normalized_text:
            return []

        response = self._generate(
            prompt=(
                "Extract key fashion attributes from the following description as short Korean tags. "
                "Categories to extract (if present): 색상, 카테고리, 소재, 핏, 소매, 넥라인, 패턴, 기장, 무드. "
                "Format each tag as '속성: 값' (e.g. '색상: 블랙', '핏: 오버사이즈'). "
                "Return ONLY a JSON array of strings, max 8 tags.\n\n"
                f"{normalized_text}"
            ),
        ).strip()

        parsed = parse_json_from_text(response)
        if isinstance(parsed, list):
            return [str(k) for k in parsed if k]
        return []

    def _classify_http_error(self, status_code: int, detail: str) -> tuple[str, str, str]:
        normalized_detail = detail.lower()

        if status_code == 429 or "rate_limit" in normalized_detail:
            return (
                "openai_quota_exceeded",
                "OpenAI 한도 초과",
                "OpenAI 요청 한도를 초과해 fallback으로 검색했습니다.",
            )

        if status_code == 401:
            return (
                "openai_auth_error",
                "OpenAI 인증 오류",
                "OpenAI API 키가 유효하지 않습니다.",
            )

        return (
            "openai_unavailable",
            "OpenAI 오류",
            "OpenAI 요청을 처리하지 못해 fallback으로 검색했습니다.",
        )

    def _is_explicit_false(self, value: Any) -> bool:
        if value is False:
            return True

        if isinstance(value, str):
            return value.strip().lower() in {"false", "no", "0"}

        return False


class Qwen3VLReasoningClient:
    # 로컬 설명 생성 경로다. 임베딩 전용 Qwen3-VL-Embedding과는 다른 모델이다.
    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        max_new_tokens: int = 2048,
    ):
        self.model_name = (
            model_name
            or os.getenv("QWEN_FALLBACK_MODEL")
            or os.getenv("QWEN3_VL_MODEL")
            or QWEN_FALLBACK_MODEL_NAME
        )
        self.device = (
            device
            or os.getenv("QWEN_FALLBACK_DEVICE")
            or os.getenv("QWEN3_VL_DEVICE")
            or "auto"
        )
        self.max_new_tokens = int(
            os.getenv(
                "QWEN_FALLBACK_MAX_NEW_TOKENS",
                os.getenv("QWEN3_VL_MAX_NEW_TOKENS", str(max_new_tokens)),
            )
        )
        self.processor = None
        self.model = None

    @property
    def provider_id(self) -> str:
        # 모델 표시용 이름을 일정한 문자열로 만든다. 이 함수가 실제 모델을 선택하지는 않는다.
        normalized = re.sub(r"[^a-z0-9]+", "_", self.model_name.lower()).strip("_")

        if "qwen3_5" in normalized and "0_8b" in normalized:
            return "qwen3_5_0_8b"

        if "qwen3_vl_2b" in normalized:
            return "qwen3_vl_2b"

        return normalized or "qwen_fallback"

    def run(
        self,
        manipulation_text: str,
        image_bytes: bytes | None = None,
        image_mime_type: str | None = None,
        image_path: str | None = None,
    ) -> dict[str, str]:
        normalized_text = manipulation_text.strip() or DEFAULT_IMAGE_ONLY_QUERY

        if image_path:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")

            with open(image_path, "rb") as image_file:
                image_bytes = image_file.read()

        image = None
        if image_bytes:
            # 로컬 모델의 전처리기가 받는 PIL RGB 이미지로 변환한다.
            image = Image.open(BytesIO(image_bytes)).convert("RGB")

        prompt = self._build_prompt(
            manipulation_text=normalized_text,
            has_image=bool(image),
        )
        response_text = self._generate(prompt=prompt, image=image)
        parsed = parse_json_from_text(response_text)

        if parsed and image and self._is_explicit_false(parsed.get("Is Fashion Image")):
            raise InvalidSearchInput("옷이 정확히 보이는 사진을 첨부하세요.")

        if parsed and self._is_explicit_false(parsed.get("Is Fashion Query")):
            raise InvalidSearchInput("패션 관련 검색어를 입력해주세요.")

        if parsed and parsed.get("Target Image Description"):
            return normalize_reasoning_result(parsed, fallback=normalized_text)

        return {
            "Original Image Description": "",
            "Thoughts": "",
            "Reflections": "",
            "Target Image Description": normalized_text,
        }

    def translate_to_korean(self, text: str) -> str:
        normalized_text = text.strip()
        if not normalized_text:
            return ""

        return self._generate(
            prompt=(
                "Translate the following fashion search description fully and faithfully "
                "to natural Korean. Do not summarize, omit details, or extract keywords. "
                "Return ONLY the Korean translation, no explanations.\n\n"
                f"{normalized_text}"
            ),
            image=None,
        ).strip()

    def translate_to_english(self, text: str) -> str:
        normalized_text = text.strip()
        if not normalized_text:
            return ""

        return self._generate(
            prompt=(
                "Translate the following Korean fashion search query fully and faithfully "
                "to natural English. Return ONLY the English translation, no explanations.\n\n"
                f"{normalized_text}"
            ),
            image=None,
        ).strip()

    def generate_with_prompt(
        self,
        prompt: str,
        image_bytes: bytes | None = None,
        image_mime_type: str | None = None,
    ) -> str:
        image = None
        if image_bytes:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        return self._generate(prompt, image)

    def _build_prompt(self, manipulation_text: str, has_image: bool) -> str:
        return build_target_description_prompt(manipulation_text, has_image)

    def _load(self) -> None:
        # 실제 추론이 필요할 때 한 번 로드한다. 모델 파일은 로컬 캐시에 있어야 한다.
        if self.processor is not None and self.model is not None:
            return

        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        try:
            import torch
            from huggingface_hub import snapshot_download
            from transformers import AutoProcessor
        except Exception as exc:
            raise ReasoningProviderError(
                "qwen_unavailable",
                "Qwen fallback 오류",
                "Qwen fallback을 실행하지 못해 원문으로 검색했습니다.",
                detail=str(exc),
            ) from exc

        try:
            model_path = snapshot_download(
                repo_id=self.model_name,
                local_files_only=True,
            )
            model_cls = self._resolve_model_class(model_path)
            kwargs: dict[str, Any] = {
                "device_map": self.device,
                "trust_remote_code": True,
            }

            if self.device == "cpu" or not torch.cuda.is_available():
                # 현재 구현은 CPU 또는 CUDA가 없는 환경에서 float32로 모델을 읽는다.
                kwargs["dtype"] = torch.float32
            else:
                kwargs["dtype"] = torch.bfloat16

            self.processor = AutoProcessor.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True,
            )
            self.model = model_cls.from_pretrained(
                model_path,
                local_files_only=True,
                **kwargs,
            )
            self.model.eval()
        except Exception as exc:
            self.processor = None
            self.model = None
            raise ReasoningProviderError(
                "qwen_unavailable",
                "Qwen fallback 오류",
                "Qwen fallback 모델을 불러오지 못해 원문으로 검색했습니다.",
                detail=str(exc),
            ) from exc

    def _resolve_model_class(self, model_path: str):
        try:
            import transformers
        except Exception as exc:
            raise ReasoningProviderError(
                "qwen_unavailable",
                "Qwen fallback 오류",
                "Qwen fallback 모델을 불러오지 못해 원문으로 검색했습니다.",
                detail=str(exc),
            ) from exc

        config_path = os.path.join(model_path, "config.json")
        try:
            with open(config_path, "r", encoding="utf-8") as config_file:
                model_config = json.load(config_file)
        except Exception as exc:
            raise ReasoningProviderError(
                "qwen_bad_config",
                "Qwen fallback 설정 오류",
                "Qwen fallback 모델 설정을 읽지 못해 원문으로 검색했습니다.",
                detail=str(exc),
            ) from exc

        architectures = model_config.get("architectures") or []
        model_type = str(model_config.get("model_type") or "")

        # 모델 설정에 적힌 클래스를 먼저 찾고, 없으면 범용 이미지-텍스트 생성 클래스를 시도한다.
        for architecture in architectures:
            model_cls = getattr(transformers, str(architecture), None)
            if model_cls is not None:
                return model_cls

        if model_type == "qwen3_5":
            raise ReasoningProviderError(
                "qwen_unsupported_transformers",
                "Qwen fallback 버전 오류",
                "Qwen3.5 모델을 로드하려면 qwen3_5를 지원하는 최신 Transformers가 필요해 원문으로 검색했습니다.",
                detail=(
                    f"Missing architecture {architectures!r}. "
                    "Install transformers from the Hugging Face main branch."
                ),
            )

        try:
            from transformers import AutoModelForImageTextToText

            return AutoModelForImageTextToText
        except Exception:
            try:
                from transformers import AutoModelForVision2Seq

                return AutoModelForVision2Seq
            except Exception as exc:
                raise ReasoningProviderError(
                    "qwen_unavailable",
                    "Qwen fallback 오류",
                    "Qwen fallback 모델을 불러오지 못해 원문으로 검색했습니다.",
                    detail=str(exc),
                ) from exc

    def _generate(self, prompt: str, image: Image.Image | None) -> str:
        self._load()

        try:
            import torch

            assert self.processor is not None
            assert self.model is not None

            content: list[dict[str, Any]] = []
            if image is not None:
                content.append({"type": "image", "image": image})
            content.append({"type": "text", "text": prompt})
            messages = [
                {"role": "system", "content": TARGETDESCRIPTION_FASHION_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)

            # 학습이 아닌 추론이므로 기울기를 저장하지 않아 불필요한 메모리 사용을 줄인다.
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                )

            # 생성 결과에는 입력 토큰도 들어 있으므로, 새로 생성된 답변 부분만 디코딩한다.
            generated_ids = [
                output_ids_item[len(input_ids_item):]
                for input_ids_item, output_ids_item in zip(inputs.input_ids, output_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return output_text[0].strip()
        except ReasoningProviderError:
            raise
        except Exception as exc:
            raise ReasoningProviderError(
                "qwen_inference_failed",
                "Qwen fallback 오류",
                "Qwen fallback 추론에 실패해 원문으로 검색했습니다.",
                detail=str(exc),
            ) from exc

    def _is_explicit_false(self, value: Any) -> bool:
        if value is False:
            return True

        if isinstance(value, str):
            return value.strip().lower() in {"false", "no", "0"}

        return False
