# 모델 호출을 대체해 업로드 전달과 API 응답 형식을 검사한다.
from io import BytesIO
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, UploadFile
from starlette.datastructures import Headers

from backend import api
from backend.request_limits import RequestLimiter


@pytest.fixture(autouse=True)
def fresh_model_request_limits(monkeypatch):
    monkeypatch.setattr(api, "request_limiter", RequestLimiter())


def test_analyze_passes_uploaded_image_without_temporary_file(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_extract_conservative_metadata(**kwargs):
        captured.update(kwargs)
        attributes = SimpleNamespace(to_dict=lambda: {"color": "black"})
        return SimpleNamespace(attributes=attributes)

    monkeypatch.setattr(
        api,
        "extract_conservative_metadata",
        fake_extract_conservative_metadata,
        raising=False,
    )
    upload = UploadFile(
        filename="reference.png",
        file=BytesIO(b"image-bytes"),
        headers=Headers({"content-type": "image/png"}),
    )

    result = api.analyze(text_input="검은색으로", image_input=upload)

    assert result == {
        "status": "success",
        "inferred_attributes": {"color": "black"},
    }
    assert captured == {
        "query": "검은색으로",
        "image_bytes": b"image-bytes",
        "image_mime_type": "image/png",
    }


def test_status_reports_the_active_embedding_model(monkeypatch) -> None:
    class FakeRetriever:
        @staticmethod
        def encode_text(_captions: list[str]):
            raise AssertionError("Status must not call the embedding model")

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                search_service=SimpleNamespace(retriever=FakeRetriever()),
            )
        )
    )
    monkeypatch.setenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-test")

    result = api.status_check(request)

    assert result["embedding_model"] == "gemini-embedding-test"
    assert result["embedding"] == "not_checked"
    assert result["check_mode"] == "configuration_only"
    assert "clip" not in result
    assert "clip_model" not in result


def test_read_image_upload_rejects_non_image_mime_type() -> None:
    upload = UploadFile(
        filename="payload.txt",
        file=BytesIO(b"not-an-image"),
        headers=Headers({"content-type": "text/plain"}),
    )

    try:
        api.read_image_upload(upload)
    except HTTPException as exc:
        assert exc.status_code == 400
        assert exc.detail == "image must be a supported image file."
    else:
        raise AssertionError("Expected HTTPException")


def test_read_image_upload_rejects_oversized_images() -> None:
    upload = UploadFile(
        filename="large.png",
        file=BytesIO(b"0" * (api.MAX_IMAGE_BYTES + 1)),
        headers=Headers({"content-type": "image/png"}),
    )

    try:
        api.read_image_upload(upload)
    except HTTPException as exc:
        assert exc.status_code == 413
        assert exc.detail == "image file is too large."
    else:
        raise AssertionError("Expected HTTPException")


def test_search_hides_runtime_error_details(monkeypatch) -> None:
    class FailingSearchService:
        @staticmethod
        def search(**_kwargs):
            raise RuntimeError("secret provider token leaked in stack")

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(search_service=FailingSearchService())
        )
    )

    with pytest.raises(api.HTTPException) as exc_info:
        api.search(
            request,
            query="검은 셔츠",
            image=None,
            top_k=10,
            table=None,
            category2=None,
            category2_keyword=None,
            provider=None,
            pipeline_method=api.DEFAULT_PIPELINE_METHOD,
        )

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == "Search service is temporarily unavailable."


def test_analyze_rejects_non_image_upload(monkeypatch) -> None:
    upload = UploadFile(
        filename="reference.txt",
        file=BytesIO(b"not-image"),
        headers=Headers({"content-type": "text/plain"}),
    )

    with pytest.raises(HTTPException) as exc_info:
        api.analyze(text_input="검은색으로", image_input=upload)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "image must be a supported image file."


def test_analyze_hides_internal_error(monkeypatch) -> None:
    def fail_extract_conservative_metadata(**_kwargs):
        raise RuntimeError("secret provider stack trace")

    monkeypatch.setattr(
        api,
        "extract_conservative_metadata",
        fail_extract_conservative_metadata,
        raising=False,
    )

    with pytest.raises(HTTPException) as exc_info:
        api.analyze(text_input="검은색으로", image_input=None)

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Internal server error."
