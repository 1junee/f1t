# 잘못된 입력과 내부 오류가 외부 API에서 어떻게 처리되는지 검사한다.
from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, UploadFile
from starlette.datastructures import Headers

from backend import api
from backend.request_limits import RequestLimiter
from pipeline.recommendation_service import FashionRecommendationPipeline


@pytest.fixture(autouse=True)
def fresh_model_request_limits(monkeypatch):
    monkeypatch.setattr(api, "request_limiter", RequestLimiter())


def make_upload(content: bytes, content_type: str = "image/jpeg") -> UploadFile:
    return UploadFile(
        file=BytesIO(content),
        filename="sample.jpg",
        headers=Headers({"content-type": content_type}),
    )


def test_image_upload_rejects_non_image_content() -> None:
    with pytest.raises(HTTPException) as exc_info:
        api.read_image_upload(make_upload(b"plain text", "text/plain"))

    assert exc_info.value.status_code == 400


def test_invalid_upload_limit_falls_back_to_safe_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAX_IMAGE_BYTES", "8mb")

    assert api.max_image_bytes_from_env() == api.DEFAULT_MAX_IMAGE_BYTES


def test_image_upload_rejects_oversized_content(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "MAX_IMAGE_BYTES", 4)

    with pytest.raises(HTTPException) as exc_info:
        api.read_image_upload(make_upload(b"12345"))

    assert exc_info.value.status_code == 413


def test_analyze_passes_image_bytes_without_a_temporary_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_extract_conservative_metadata(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        attributes = SimpleNamespace(to_dict=lambda: {"category": "outer"})
        return SimpleNamespace(attributes=attributes)

    monkeypatch.setattr(
        api, "extract_conservative_metadata", fake_extract_conservative_metadata
    )

    response = api.analyze("find a jacket", make_upload(b"image-bytes"))

    assert response == {
        "status": "success",
        "inferred_attributes": {"category": "outer"},
    }
    assert captured == {
        "query": "find a jacket",
        "image_bytes": b"image-bytes",
        "image_mime_type": "image/jpeg",
    }


def test_analyze_hides_internal_error_details(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_analysis(**_kwargs: object) -> None:
        raise RuntimeError("private provider credential detail")

    monkeypatch.setattr(api, "extract_conservative_metadata", fail_analysis)

    with pytest.raises(HTTPException) as exc_info:
        api.analyze("find a jacket", None)

    assert exc_info.value.status_code == 500
    assert "private provider" not in str(exc_info.value.detail)


def test_legacy_multi_table_search_fails_when_any_required_table_is_unavailable() -> None:
    class PartiallyFailingSearch:
        def search_approximate(self, *, table_filter: str, **_kwargs: object) -> list[dict[str, object]]:
            if table_filter == "musinsa_pants":
                raise RuntimeError("database detail")
            return [{"similarity": 0.9, "source_table": table_filter}]

    from pipeline.benchmarks.retrieval_request_comparison import search_three_requests_legacy

    with pytest.raises(RuntimeError, match="일부 상품 카테고리"):
        search_three_requests_legacy(
            vector_search=PartiallyFailingSearch(),
            embedding=[0.1],
            match_count=3,
            table_filter=None,
            category2_filter=None,
            category2_keyword_filter=None,
        )
