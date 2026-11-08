"""번역 없는 추천 설명과 실제 검색 결과 상태의 일치를 검증한다."""
from types import SimpleNamespace
from unittest.mock import Mock
import urllib.request

import numpy as np
import pytest

from pipeline import recommendation_service as service
from pipeline.retrieval.candidate_selection import ParsedTextAttributes
from pipeline.target_description import recommendation_explanation as explanation


@pytest.mark.parametrize('status,count,expected', [
    ('exact', 2, '긴 소매 조건을 적용해 검색했습니다.'),
    ('relaxed', 2, '일부 조건을 완화'),
    ('similarity', 2, '요청을 바탕으로 유사한'),
    ('exact', 0, '추천할 상품을 찾지 못했습니다'),
    ('relaxed', 0, '추천할 상품을 찾지 못했습니다'),
])
def test_fallback_uses_search_state_not_description_keywords(status, count, expected):
    text = explanation.build_query_recommendation_explanation(
        search_status=status, result_count=count,
        applied_filters={'sleeve': 'long', 'color': 'white', 'fabric': 'cotton'},
    )
    assert expected in text
    assert 'white' not in text and 'cotton' not in text
    assert '잘 맞' not in text


@pytest.mark.parametrize('status,count', [('no_results', 0), ('relaxed', 2)])
def test_empty_and_relaxed_results_use_fixed_notice_without_llm(monkeypatch, status, count):
    model = Mock(side_effect=AssertionError('must not call model'))
    monkeypatch.setattr(explanation, '_generate_gemini_recommendation_reason', model)
    result = explanation.generate_query_recommendation_explanation(
        query='리본 없는 셔츠', parsed_attributes={}, search_status=status, result_count=count,
        target_description_reasoning={'Target Image Description': 'shirt without a ribbon'},
    )
    model.assert_not_called()
    assert not result.uses_additional_model_call
    assert result.method == 'deterministic_search_status'


@pytest.mark.parametrize('response', [RuntimeError('model failed'), 'English only', ''])
def test_model_failure_or_invalid_output_keeps_honest_fallback(monkeypatch, response):
    monkeypatch.setenv('GEMINI_API_KEY', 'test-key')
    model = Mock(side_effect=response) if isinstance(response, Exception) else Mock(return_value=response)
    monkeypatch.setattr(explanation, '_generate_gemini_recommendation_reason', model)
    result = explanation.generate_query_recommendation_explanation(
        query='긴팔 셔츠', parsed_attributes={'sleeve': 'long'},
        search_status='exact', result_count=2, applied_filters={'sleeve': 'long'},
        target_description_reasoning={'Target Image Description': 'long-sleeve shirt'},
    )
    assert result.fallback_used
    assert result.recommendation_reason == '긴 소매 조건을 적용해 검색했습니다.'


def test_llm_gets_english_description_and_actual_status_without_translation(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'test-key')
    model = Mock(return_value='긴팔 셔츠를 요청하셨습니다. 소매 조건을 적용해 검색했습니다.')
    monkeypatch.setattr(explanation, '_generate_gemini_recommendation_reason', model)
    result = explanation.generate_query_recommendation_explanation(
        query='긴팔 셔츠', parsed_attributes={'sleeve': 'long'},
        search_status='exact', result_count=2, applied_filters={'sleeve': 'long'},
        target_description_reasoning={'Target Image Description': 'long-sleeve shirt without a logo'},
    )
    prompt = model.call_args.kwargs['prompt']
    assert 'long-sleeve shirt without a logo' in prompt
    assert 'target_description_ko' not in prompt
    assert '"search_status": "exact"' in prompt
    assert '"result_count": 2' in prompt
    assert not result.fallback_used


@pytest.mark.parametrize('has_filters,strict_rows,approx_rows,expected', [
    (True, [{'id': 'a', 'season': ['여름']}], [], 'exact'),
    (True, [], [{'id': 'b', 'season': ['겨울']}], 'relaxed'),
    (True, [], [], 'no_results'),
    (False, [], [{'id': 'c'}], 'similarity'),
    (False, [], [], 'no_results'),
])
def test_pipeline_reports_final_state_and_never_translates(monkeypatch, has_filters, strict_rows, approx_rows, expected):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    http = Mock(side_effect=AssertionError('No translation or other HTTP allowed'))
    monkeypatch.setattr(urllib.request, 'urlopen', http)
    attributes = ParsedTextAttributes(season='summer') if has_filters else ParsedTextAttributes()
    monkeypatch.setattr(service, 'extract_conservative_metadata', lambda **kwargs: SimpleNamespace(
        status='ok', attributes=attributes, to_metadata=lambda: {'status': 'ok'},
    ))
    pipeline = service.FashionRecommendationPipeline.__new__(service.FashionRecommendationPipeline)
    pipeline._default_provider = 'gemini'
    pipeline._run_target_description_reasoning = Mock(return_value=(
        {'Target Image Description': 'summer clothing'}, 'gemini', 'ok', [],
    ))
    pipeline.retriever = SimpleNamespace(encode_text=Mock(return_value=np.array([[1.0, 0.0]])))
    pipeline.vector_search = SimpleNamespace(
        search_filtered_exact=Mock(return_value=(strict_rows, {'matched_count': len(strict_rows)})),
        search_approximate=Mock(return_value=approx_rows),
    )
    target, metadata, rows = pipeline.search('여름 옷')
    assert target == 'summer clothing'
    assert metadata['pipeline']['search_status'] == expected
    assert all(stage['id'] != 'translation' for stage in metadata['pipeline']['timings']['stages'])
    assert 'translation_and_embedding' not in metadata['pipeline']['parallel_execution']
    assert bool(rows) == (expected != 'no_results')
    if expected == 'exact':
        assert '여름 조건을 적용' in metadata['recommendation_reason']
    elif expected == 'relaxed':
        assert '일부 조건을 완화' in metadata['recommendation_reason']
    elif expected == 'no_results':
        assert '찾지 못했습니다' in metadata['recommendation_reason']
    http.assert_not_called()
