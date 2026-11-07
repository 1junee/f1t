# 저장 벡터 차원, 정확 필터 RPC 계약, DB/HTTP 계측값을 검사한다.
import json
from io import BytesIO
import urllib.error
from unittest.mock import Mock

import pytest

from pipeline import vector_db_client as db
from pipeline.vector_db_client import SupabaseVectorSearchClient


def test_rpc_for_gemini_embedding_dimension() -> None:
    assert SupabaseVectorSearchClient._rpc_for_dim(768) == "match_fashion_items_768"


def test_rpc_rejects_unsupported_embedding_dimension() -> None:
    with pytest.raises(ValueError, match="768-dimensional"):
        SupabaseVectorSearchClient._rpc_for_dim(512)

def filtered_response(results=None, **diagnostics):
    return {
        "results": results or [],
        "diagnostics": {"strategy": "filtered_exact_cosine", "matched_count": 2889,
                        "db_filter_ms": 1.0, "db_rank_ms": 20.0, "db_total_ms": 22.0,
                        **diagnostics},
    }


@pytest.mark.parametrize("table,bucket", [
    ("musinsa_top_clothes", "test"), ("musinsa_pants", "musinsa_pants"),
    ("musinsa_skirt_dress", "skirt_dress"),
])
@pytest.mark.parametrize("image_url", [None, ""])
def test_filtered_rpc_preserves_missing_image_url_fallback(monkeypatch, table, bucket, image_url):
    body = json.dumps(filtered_response([{"id": "123", "source_table": table, "image_url": image_url}])).encode()
    monkeypatch.setattr(db.urllib.request, "urlopen", Mock(return_value=BytesIO(body)))
    rows, _ = SupabaseVectorSearchClient("https://example.invalid", "key").search_filtered(
        [0.1] * 768, 1, metadata_filters={"season": "summer"},
    )
    assert rows[0]["image_url"] == f"https://example.invalid/storage/v1/object/public/{bucket}/123.jpg"


def test_filtered_rpc_translates_filters_keeps_scope_and_measures_transport(monkeypatch):
    payload = filtered_response([{"id": "9000", "similarity": 0.9}])
    body = json.dumps(payload).encode()
    open_request = Mock(return_value=BytesIO(body))
    monkeypatch.setattr(db.urllib.request, "urlopen", open_request)
    client = db.SupabaseVectorSearchClient("https://example.invalid", "private-key")
    rows, info = client.search_filtered(
        [0.1] * 768, 10, table_filter="musinsa_skirt_dress",
        category2_filter="미니 스커트", category2_keyword_filter="스커트",
        fabric_embedding=[0.2] * 768,
        metadata_filters={"sex": "unisex", "fit": "wide", "season": "summer", "sleeve": "short",
                          "color": "white", "fabric": "cotton", "category1": "top"},
    )
    request = open_request.call_args.args[0]
    sent = json.loads(request.data)
    assert request.full_url.endswith("/rpc/match_fashion_items_filtered_768")
    assert sent["p_metadata_filters"] == {"sex": ["남성", "여성"], "fit": ["루즈", "오버사이즈"],
                                            "season": ["여름"], "sleeve": ["short"]}
    assert sent["p_table_filter"] == "musinsa_skirt_dress"
    assert sent["p_category2_filter"] == "미니 스커트"
    assert sent["p_category2_keyword_filter"] == "스커트"
    assert len(json.loads(sent["p_fabric_embedding"])) == 768
    assert rows[0]["id"] == "9000"
    assert info["request_count"] == 1
    assert info["response_bytes"] == len(body)
    assert info["matched_count"] == 2889
    for key in ("headers_wait_ms", "body_read_ms", "json_decode_ms", "http_total_ms", "transport_overhead_estimate_ms"):
        assert info[key] >= 0
    open_request.assert_called_once()


@pytest.mark.parametrize("payload", [[], {"results": {}, "diagnostics": {}},
    {"results": ["invalid"], "diagnostics": {}}, {"results": [], "diagnostics": {}},
    filtered_response(db_total_ms="oops"), filtered_response(matched_count=-1)])
def test_filtered_rpc_rejects_malformed_response(monkeypatch, payload):
    monkeypatch.setattr(db.urllib.request, "urlopen", Mock(return_value=BytesIO(json.dumps(payload).encode())))
    client = db.SupabaseVectorSearchClient("https://example.invalid", "private-key")
    with pytest.raises(RuntimeError, match="response"):
        client.search_filtered([0.1] * 768, 3, metadata_filters={"sleeve": "short"})


def test_filtered_rpc_missing_function_is_actionable(monkeypatch):
    error = urllib.error.HTTPError("https://example.invalid", 404, "missing", {}, BytesIO(b'{"code":"PGRST202"}'))
    monkeypatch.setattr(db.urllib.request, "urlopen", Mock(side_effect=error))
    client = db.SupabaseVectorSearchClient("https://example.invalid", "private-key")
    with pytest.raises(RuntimeError, match="match_fashion_items_filtered_768.sql"):
        client.search_filtered([0.1] * 768, 3, metadata_filters={"sleeve": "short"})


def test_filtered_rpc_does_not_expose_upstream_body(monkeypatch):
    error = urllib.error.HTTPError("https://example.invalid", 500, "bad", {}, BytesIO(b'private-key and sensitive SQL'))
    monkeypatch.setattr(db.urllib.request, "urlopen", Mock(side_effect=error))
    client = db.SupabaseVectorSearchClient("https://example.invalid", "private-key")
    with pytest.raises(RuntimeError) as caught:
        client.search_filtered([0.1] * 768, 3, metadata_filters={})
    assert "private-key" not in str(caught.value)
    assert "500" in str(caught.value)


def test_filtered_rpc_rejects_wrong_fabric_dimension_before_http(monkeypatch):
    request = Mock()
    monkeypatch.setattr(db.urllib.request, "urlopen", request)
    client = db.SupabaseVectorSearchClient("https://example.invalid", "private-key")
    with pytest.raises(ValueError, match="768-dimensional"):
        client.search_filtered([0.1] * 768, 3, fabric_embedding=[0.2] * 12, metadata_filters={})
    request.assert_not_called()


def test_filtered_rpc_omits_vectors_even_if_server_contract_regresses(monkeypatch):
    payload = filtered_response([{"id": "9000", "similarity": 0.9,
                                 "gemini_image_embedding_768": [0.1],
                                 "gemini_fabric_text_embedding_768": [0.2]}])
    monkeypatch.setattr(db.urllib.request, "urlopen", Mock(return_value=BytesIO(json.dumps(payload).encode())))
    client = db.SupabaseVectorSearchClient("https://example.invalid", "private-key")
    rows, _ = client.search_filtered([0.1] * 768, 3, metadata_filters={})
    assert rows == [{"id": "9000", "similarity": 0.9}]


def test_filtered_rpc_connection_error_is_distinct_from_empty_results(monkeypatch):
    monkeypatch.setattr(db.urllib.request, "urlopen", Mock(side_effect=TimeoutError("network")))
    client = db.SupabaseVectorSearchClient("https://example.invalid", "private-key")
    with pytest.raises(RuntimeError, match="timed out"):
        client.search_filtered([0.1] * 768, 3, metadata_filters={})
