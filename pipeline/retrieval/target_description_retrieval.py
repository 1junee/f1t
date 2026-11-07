"""Target-description Gemini embedding retrieval module.

Uses gemini-embedding-2 to encode target descriptions into 768-dim vectors
for pgvector similarity search in Supabase.
"""
from __future__ import annotations

import os
from typing import Any


# 설명문과 소재 문장을 벡터로 바꾸는 현재 기본 모델이다. 상품 벡터와 같은 공간을 써야 한다.
GEMINI_EMBEDDING_MODEL = "gemini-embedding-2"
GEMINI_EMBEDDING_DIM = 768


class GeminiEmbeddingClient:
    """Text encoder via Gemini Embedding API (google-genai SDK)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        dimensionality: int | None = None,
    ):
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "google-genai is required. Install with: pip install google-genai"
            ) from exc

        self._types = types
        self._client = genai.Client(
            api_key=api_key or os.getenv("GEMINI_API_KEY")
        )
        # 호출 인자가 있으면 우선하고, 없으면 환경설정 또는 기본 모델명을 사용한다.
        self._model = (
            model
            or os.getenv("GEMINI_EMBEDDING_MODEL", GEMINI_EMBEDDING_MODEL)
        )
        # 출력 차원을 바꾸려면 DB에 저장된 상품 벡터와 검색 스키마도 맞춰야 한다.
        dim = dimensionality or int(
            os.getenv("GEMINI_EMBEDDING_DIM", str(GEMINI_EMBEDDING_DIM))
        )
        self._config = types.EmbedContentConfig(output_dimensionality=dim)

    def encode_text(self, captions: list[str]) -> Any:
        import numpy as np

        # 문장마다 API를 호출하고, 반환값을 [문장 수, 벡터 차원] 배열로 모은다.
        vectors = []
        for text in captions:
            # 접두어는 임베딩 모델에 함께 전달하는 입력이다. 설명문 생성 요청이 아니다.
            response = self._client.models.embed_content(
                model=self._model,
                contents=f"task: search result | query: {text}",
                config=self._config,
            )
            # 입력 순서를 유지한다. API 실패는 여기서 숨기지 않고 호출자에게 전달된다.
            vectors.append(response.embeddings[0].values)
        # 자료형만 float32로 통일한다. 여기서 벡터 길이를 1로 정규화하지는 않는다.
        return np.array(vectors, dtype=np.float32)


class RetrieveModule:
    """Query encoder using gemini-embedding-2 for Supabase pgvector search."""

    def __init__(
        self,
        clip_model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        hf_api_key: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        dimensionality: int | None = None,
        **_legacy_options: Any,
    ):
        # 옛 호출부와의 호환용 인자다. clip_model_name을 바꿔도 CLIP을 실행하지 않는다.
        _ = (clip_model_name, pretrained, hf_api_key, _legacy_options)
        self._backend = GeminiEmbeddingClient(
            api_key=api_key,
            model=model,
            dimensionality=dimensionality,
        )

    def encode_text(self, captions: list[str]) -> Any:
        # 이 모듈은 벡터 생성까지만 맡고, 상품 조회·유사도 계산은 상위 서비스가 맡는다.
        return self._backend.encode_text(captions)
