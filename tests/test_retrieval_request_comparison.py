"""운영의 단일 요청과 실험용 이전 3회 요청의 계약을 확인한다."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from pipeline.benchmarks.retrieval_request_comparison import search_three_requests_legacy
from pipeline.recommendation_service import FashionRecommendationPipeline, VALID_TABLES


def inputs():
    return dict(embedding=[0.1] * 768, match_count=2, table_filter=None,
                category2_filter=None, category2_keyword_filter=None,
                fabric_embedding=[0.2] * 768)


@pytest.mark.parametrize('scope', [{}, {'table_filter': 'musinsa_top_clothes'},
    {'category2_filter': '반소매 티셔츠'}, {'category2_keyword_filter': '셔츠'}])
def test_production_calls_once_and_preserves_all_arguments(scope):
    rows = [{'id': 'a', 'similarity': 0.9}, {'id': 'b', 'similarity': 0.8}]
    request = Mock(return_value=rows)
    pipeline = FashionRecommendationPipeline.__new__(FashionRecommendationPipeline)
    pipeline.vector_search = SimpleNamespace(search_approximate=request)
    kwargs = inputs()
    kwargs.update(scope)
    assert pipeline._search_vectors(**kwargs) is rows
    request.assert_called_once_with(**kwargs)


def test_production_empty_results_are_not_database_failure():
    pipeline = FashionRecommendationPipeline.__new__(FashionRecommendationPipeline)
    request = Mock(return_value=[])
    pipeline.vector_search = SimpleNamespace(search_approximate=request)
    assert pipeline._search_vectors(**inputs()) == []
    request.assert_called_once()


def test_production_database_error_propagates_without_retry_or_partial_results():
    pipeline = FashionRecommendationPipeline.__new__(FashionRecommendationPipeline)
    request = Mock(side_effect=RuntimeError('RPC unavailable'))
    pipeline.vector_search = SimpleNamespace(search_approximate=request)
    with pytest.raises(RuntimeError, match='RPC unavailable'):
        pipeline._search_vectors(**inputs())
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
