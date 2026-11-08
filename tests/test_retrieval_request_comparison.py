"""운영의 단일 요청과 실험용 이전 3회 요청의 계약을 확인한다."""
import json
import urllib.request
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from pipeline.benchmarks.retrieval_request_comparison import search_three_requests_legacy
from pipeline.recommendation_service import VALID_TABLES
from pipeline.vector_db_client import SupabaseVectorSearchClient


def inputs():
    return dict(embedding=[0.1] * 768, match_count=2, table_filter=None,
                category2_filter=None, category2_keyword_filter=None,
                fabric_embedding=[0.2] * 768)


@pytest.mark.parametrize('scope', [{}, {'table_filter': 'musinsa_top_clothes'},
    {'category2_filter': '반소매 티셔츠'}, {'category2_keyword_filter': '셔츠'}])
def test_production_client_calls_rpc_once_and_preserves_all_arguments(monkeypatch, scope):
    rows = [{'id': 'a', 'similarity': 0.9}, {'id': 'b', 'similarity': 0.8}]
    request = Mock(return_value=BytesIO(json.dumps(rows).encode()))
    monkeypatch.setattr(urllib.request, 'urlopen', request)
    client = SupabaseVectorSearchClient('https://example.invalid', 'test-key')
    kwargs = inputs()
    kwargs.update(scope)
    assert client.search_approximate(**kwargs) == rows
    request.assert_called_once()
    sent = request.call_args.args[0]
    assert sent.full_url.endswith('/rpc/approximate_vector_search_768')
    payload = json.loads(sent.data)
    assert json.loads(payload['p_query_embedding']) == kwargs['embedding']
    assert json.loads(payload['p_fabric_embedding']) == kwargs['fabric_embedding']
    for key in ('match_count', 'table_filter', 'category2_filter', 'category2_keyword_filter'):
        assert payload[f'p_{key}'] == kwargs[key]


def test_production_empty_results_are_not_database_failure(monkeypatch):
    request = Mock(return_value=BytesIO(b'[]'))
    monkeypatch.setattr(urllib.request, 'urlopen', request)
    client = SupabaseVectorSearchClient('https://example.invalid', 'test-key')
    assert client.search_approximate(**inputs()) == []
    request.assert_called_once()


def test_production_database_error_propagates_without_retry_or_partial_results(monkeypatch):
    request = Mock(side_effect=urllib.error.URLError('RPC unavailable'))
    monkeypatch.setattr(urllib.request, 'urlopen', request)
    client = SupabaseVectorSearchClient('https://example.invalid', 'test-key')
    with pytest.raises(RuntimeError, match='RPC unavailable'):
        client.search_approximate(**inputs())
    request.assert_called_once()


def test_legacy_preserves_three_requests_global_ranking_and_limit():
    request = Mock(side_effect=[
        [{'id': 'a', 'similarity': 0.8}],
        [{'id': 'b', 'similarity': 0.9}],
        [{'id': 'c', 'similarity': 0.7}],
    ])
    result = search_three_requests_legacy(SimpleNamespace(search_approximate=request), **inputs())
    assert [row['id'] for row in result] == ['b', 'a']
    assert [call.kwargs['table_filter'] for call in request.call_args_list] == sorted(VALID_TABLES)
    assert all(call.kwargs['fabric_embedding'] == [0.2] * 768 for call in request.call_args_list)


def test_legacy_scoped_search_still_uses_one_request():
    request = Mock(return_value=[])
    kwargs = inputs()
    kwargs['table_filter'] = 'musinsa_top_clothes'
    assert search_three_requests_legacy(SimpleNamespace(search_approximate=request), **kwargs) == []
    request.assert_called_once_with(**kwargs)


def test_legacy_preserves_original_empty_result_error_for_reproduction():
    request = Mock(return_value=[])
    with pytest.raises(RuntimeError, match='모든 테이블 검색이 실패'):
        search_three_requests_legacy(SimpleNamespace(search_approximate=request), **inputs())
    assert request.call_count == 3
