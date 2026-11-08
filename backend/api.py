# HTTP 입력 검증과 응답 형식을 담당한다. 실제 검색 순서는 recommendation_service.py에 있다.
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.request_limits import RequestLimiter
from pipeline.env import load_pipeline_env
from pipeline.recommendation_service import (
    DEFAULT_RECOMMENDATION_COUNT,
    FashionRecommendationPipeline,
    MAX_TOP_K,
    PIPELINE_MODE,
    VALID_TABLES,
    needs_reference_image,
)
from pipeline.intent.intent_extraction import (
    extract_conservative_metadata,
    resolve_intent_model,
)
from pipeline.retrieval.candidate_selection import retrieve_products
from pipeline.retrieval.target_description_retrieval import GEMINI_EMBEDDING_MODEL
from pipeline.target_description.target_description_generation import (
    InvalidSearchInput,
    OPENAI_MODEL_NAME,
    QWEN_FALLBACK_MODEL_NAME,
    resolve_target_description_model,
)


load_pipeline_env()

logger = logging.getLogger(__name__)
DEFAULT_MAX_IMAGE_BYTES = 8 * 1024 * 1024


# 환경변수의 문자열 → 업로드 제한 바이트 수. 설정이 잘못됐어도 기본 제한으로 실행한다.
def max_image_bytes_from_env() -> int:
    raw_value = os.getenv("MAX_IMAGE_BYTES")
    if raw_value is None:
        return DEFAULT_MAX_IMAGE_BYTES

    try:
        value = int(raw_value)
    except ValueError:
        logger.warning("Ignoring invalid MAX_IMAGE_BYTES value")
        return DEFAULT_MAX_IMAGE_BYTES

    if value < 1:
        logger.warning("Ignoring non-positive MAX_IMAGE_BYTES value")
        return DEFAULT_MAX_IMAGE_BYTES
    return value


# 모듈을 불러올 때 계산하는 설정이다. 요청마다 환경변수를 다시 읽는 값은 아니다.
MAX_IMAGE_BYTES = max_image_bytes_from_env()
# 환경변수를 읽은 뒤 한 번 생성하며 /search와 /analyze가 동일한 한도를 공유한다.
request_limiter = RequestLimiter.from_env()
SUPPORTED_IMAGE_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
}


# 업로드 객체 → (이미지 바이트, MIME 형식). 이미지가 없으면 둘 다 None이다.
# 여기서는 신고된 MIME과 크기를 검사한다. 실제 이미지 해석은 이후 모델 처리 단계다.
def read_image_upload(upload: UploadFile | None) -> tuple[bytes | None, str | None]:
    if upload is None:
        return None, None

    image_mime_type = upload.content_type or ""
    if image_mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail="image must be a supported image file.",
        )

    # 제한보다 1바이트 더 읽어 초과 여부를 확인하고, 큰 파일 전체를 메모리에 올리지 않는다.
    image_bytes = upload.file.read(MAX_IMAGE_BYTES + 1)
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="image file is too large.")

    return image_bytes, image_mime_type


# 쉼표로 나열한 프론트 주소를 CORS 허용 목록으로 바꾼다.
def parse_origins() -> list[str]:
    raw = os.getenv("FRONTEND_ORIGINS", "*").strip()
    if raw == "*":
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 시작 시 검색 서비스를 한 번 만들고 각 요청에서 재사용한다.
    app.state.search_service = FashionRecommendationPipeline()
    # yield 이전은 서버 시작 준비, 이후는 종료 시 정리 코드를 둘 자리다.
    yield


app = FastAPI(title="F1T Supabase Vector Search API", lifespan=lifespan)
# CORS는 브라우저가 다른 출처의 응답을 읽는 규칙이며, 사용자 인증 기능은 아니다.
_origins = parse_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    # 서버 응답 여부만 확인한다. DB나 외부 모델의 정상 동작까지 검사하지 않는다.
    return {
        "status": "ok",
        "mode": PIPELINE_MODE,
        "provider": os.getenv("VLM_PROVIDER", "gemini"),
    }


@app.get("/status")
def status_check(request: Request) -> dict:
    # 구성값만 확인한다. 외부 모델/DB 연결 성공 여부를 검증하거나 과금 호출하지 않는다.
    result: dict = {
        "embedding_model": os.getenv("GEMINI_EMBEDDING_MODEL", GEMINI_EMBEDDING_MODEL),
        "embedding": "not_checked",
        "check_mode": "configuration_only",
    }

    result["gemini_key_set"] = bool(os.getenv("GEMINI_API_KEY"))
    result["supabase_url_set"] = bool(os.getenv("SUPABASE_URL"))
    result["supabase_key_set"] = bool(os.getenv("SUPABASE_KEY"))
    return result


@app.get("/models")
def models_info() -> dict[str, str]:
    # 설명 생성 제공자들의 모델 이름이다. provider를 바꿔도 임베딩 모델이 바뀌지는 않는다.
    qwen_model = (
        os.getenv("QWEN_FALLBACK_MODEL")
        or os.getenv("QWEN3_VL_MODEL")
        or QWEN_FALLBACK_MODEL_NAME
    )
    return {
        "gemini": resolve_target_description_model(),
        "gemini_intent": resolve_intent_model(),
        "gemini_target_description": resolve_target_description_model(),
        "openai": os.getenv("OPENAI_MODEL", OPENAI_MODEL_NAME),
        "qwen": qwen_model.split("/")[-1],
    }


# 문장·이미지·검색 옵션 → 설명문과 상품 목록을 담은 JSON.
# 동기 요청 핸들러이며 내부 병렬 작업도 끝까지 기다린 뒤 한 번에 응답한다.
@app.post("/search")
def search(
    request: Request,
    query: str | None = Form(default=None),
    image: UploadFile | None = File(default=None),
    top_k: int = Form(default=DEFAULT_RECOMMENDATION_COUNT),
    table: str | None = Form(default=None),
    category2: str | None = Form(default=None),
    category2_keyword: str | None = Form(default=None),
    provider: str | None = Form(default=None),
    intent_provider: str | None = Form(default=None),
    target_description_provider: str | None = Form(default=None),
) -> dict[str, Any]:
    # multipart 폼의 빈 문자열을 정리하고, 모델 호출 전에 입력 범위를 검증한다.
    normalized_query = (query or "").strip()
    table_filter = (table or "").strip() or None
    category2_filter = (category2 or "").strip() or None
    category2_keyword_filter = (category2_keyword or "").strip() or None
    if top_k < 1 or top_k > MAX_TOP_K:
        raise HTTPException(
            status_code=400,
            detail=f"top_k must be between 1 and {MAX_TOP_K}.",
        )

    if table_filter and table_filter not in VALID_TABLES:
        raise HTTPException(status_code=400, detail="Invalid table filter.")

    normalized_provider = (provider or os.getenv("VLM_PROVIDER", "gemini")).strip().lower()
    if normalized_provider not in {"gemini", "openai", "qwen"}:
        raise HTTPException(status_code=400, detail="Invalid provider. Use gemini, openai, or qwen.")

    role_providers = {}
    for name, value in (("intent_provider", intent_provider),
                        ("target_description_provider", target_description_provider)):
        normalized = value.strip().lower() if isinstance(value, str) else None
        if normalized and normalized not in {"gemini", "openai", "qwen"}:
            raise HTTPException(status_code=400, detail=f"Invalid {name}.")
        role_providers[name] = normalized or None

    image_bytes, image_mime_type = read_image_upload(image)

    if not normalized_query and not image_bytes:
        raise HTTPException(status_code=400, detail="query or image is required.")

    if normalized_query and not image_bytes and needs_reference_image(normalized_query):
        raise HTTPException(status_code=400, detail="이미지를 첨부하세요.")

    # 서비스가 설명문·검색 결과를 모두 완성할 때까지 이 요청의 응답은 기다린다.
    service: FashionRecommendationPipeline = request.app.state.search_service
    try:
        with request_limiter.acquire():
            target_description, target_description_ko, reasoning_metadata, raw_results = (
                service.search(
                    query=normalized_query,
                    top_k=top_k,
                    image_bytes=image_bytes,
                    image_mime_type=image_mime_type,
                    table_filter=table_filter,
                    category2_filter=category2_filter,
                    category2_keyword_filter=category2_keyword_filter,
                    provider=normalized_provider,
                    **role_providers,
                )
            )
    # 서비스 내부 예외를 HTTP 상태 코드로 바꾼다. 상세 원인은 서버 로그에 남긴다.
    except HTTPException:
        raise
    except InvalidSearchInput as exc:
        # 서비스가 사용자 안내용으로 분리한 오류만 원문을 전달한다.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        logger.info("Invalid recommendation request: %s", exc)
        raise HTTPException(status_code=400, detail="Invalid search request.") from exc
    except RuntimeError as exc:
        logger.exception("Search pipeline runtime failure")
        raise HTTPException(
            status_code=502,
            detail="Search service is temporarily unavailable.",
        ) from exc
    except Exception as exc:
        logger.exception("Unhandled search failure")
        raise HTTPException(status_code=500, detail="Internal server error.") from exc

    # 내부 DB 행 전체 대신 화면에서 사용하는 상품 정보와 검색 설명만 반환한다.
    return {
        "mode": PIPELINE_MODE,
        "provider": reasoning_metadata["provider"],
        "pipeline": reasoning_metadata.get("pipeline"),
        "target_description": target_description,
        "target_description_ko": target_description_ko,
        "recommendation_reason": reasoning_metadata.get("recommendation_reason"),
        "reasoning": reasoning_metadata,
        "results": [
            {
                "rank": rank,
                "id": str(row.get("id") or ""),
                "source_table": row.get("source_table"),
                "name": row.get("name"),
                "brand": row.get("brand"),
                "image_url": row.get("image_url"),
                "category1": row.get("category1"),
                "category2": row.get("category2"),
                "price": row.get("price"),
                "color": row.get("color"),
                "sleeve": row.get("sleeve"),
                "length": row.get("length"),
                "sex": row.get("sex"),
                "season": row.get("season"),
                "stretch": row.get("stretch"),
                "thickness": row.get("thickness"),
                "fit": row.get("fit"),
                "fabric": row.get("fabric"),
                "similarity": float(row.get("similarity") or 0),
            }
            for rank, row in enumerate(raw_results, start=1)
        ],
    }


@app.post("/analyze")
# 전체 검색 없이 속성 추출 단계만 호출하는 보조 API다.
def analyze(
    text_input: str = Form(...),
    image_input: UploadFile | None = File(None),
) -> dict[str, Any]:
    try:
        image_bytes, image_mime_type = read_image_upload(image_input)
        with request_limiter.acquire():
            extraction = extract_conservative_metadata(
                query=text_input,
                image_bytes=image_bytes,
                image_mime_type=image_mime_type,
            )
        return {
            "status": "success",
            "inferred_attributes": extraction.attributes.to_dict(),
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unhandled analysis failure")
        raise HTTPException(status_code=500, detail="Internal server error.") from exc


@app.post("/retrieve")
# 추출된 속성을 입력받아 후보 조회 단계만 실행하는 보조 API다.
def retrieve(request_body: dict) -> dict[str, Any]:
    try:
        inferred_attributes = request_body.get("inferred_attributes", {})
        products = retrieve_products(inferred_attributes)
        return {
            "status": "success",
            "products": products,
            "count": len(products),
        }

    except Exception as exc:
        logger.exception("Unhandled retrieval failure")
        raise HTTPException(status_code=500, detail="Internal server error.") from exc
