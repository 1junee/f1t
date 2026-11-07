"""검색 범위와 장애 대체 경로를 실제 API 호출 없이 검증한다."""
from types import SimpleNamespace
from unittest.mock import Mock
from io import BytesIO
import sys

import numpy as np
import pytest
from PIL import Image

from pipeline import recommendation_service as service
from pipeline.retrieval import candidate_selection as candidates
from pipeline.target_description import target_description_generation as generation


@pytest.fixture(autouse=True)
def no_http_calls(monkeypatch):
    monkeypatch.setattr(service.urllib.request, "urlopen", Mock(side_effect=AssertionError("HTTP is forbidden in regression tests")))


@pytest.mark.parametrize("attributes", [candidates.ParsedTextAttributes(), candidates.ParsedTextAttributes(category1="top")])
def test_successful_intent_does_not_restore_rejected_keyword_filters(attributes):
    plan = candidates.build_attribute_text_search_plan(
        "긴팔 말고 반팔, 바지 말고 상의", top_k=3,
        extracted_attributes=attributes, extraction_metadata={"status": "ok"},
    )
    assert plan.parsed_attributes == attributes
    assert plan.table_filter == ("musinsa_top_clothes" if attributes.category1 else None)


def test_failed_intent_still_uses_rule_fallback():
    plan = candidates.build_attribute_text_search_plan(
        "긴팔 상의", top_k=3, extraction_metadata={"status": "degraded"},
    )
    assert plan.parsed_attributes.sleeve == "long"
    assert plan.table_filter == "musinsa_top_clothes"


def test_failed_intent_does_not_trust_stale_extracted_attributes():
    plan = candidates.build_attribute_text_search_plan(
        "긴팔 상의", top_k=3, extracted_attributes=candidates.ParsedTextAttributes(category1="pants"),
        extraction_metadata={"status": "degraded"},
    )
    assert plan.parsed_attributes.sleeve == "long"
    assert plan.table_filter == "musinsa_top_clothes"


@pytest.mark.parametrize("scope", [
    {"table_filter": "musinsa_top_clothes"},
    {"category2_filter": "미니 스커트"},
    {"table_filter": "musinsa_skirt_dress"},
])
def test_explicit_category_scope_does_not_inherit_inferred_keyword(scope):
    plan = candidates.build_attribute_text_search_plan(
        "원피스", top_k=3,
        extracted_attributes=candidates.ParsedTextAttributes(category1="one-piece"),
        **scope,
    )
    assert plan.category2_keyword_filter is None


def test_explicit_category_keyword_wins_over_inference():
    plan = candidates.build_attribute_text_search_plan(
        "원피스", top_k=3,
        extracted_attributes=candidates.ParsedTextAttributes(category1="one-piece"),
        table_filter="musinsa_skirt_dress", category2_keyword_filter="스커트",
    )
    assert plan.category2_keyword_filter == "스커트"


class QueryRecorder:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.append((name, args))
            if name == "execute":
                return SimpleNamespace(data=[])
            return self
        return call


@pytest.fixture
def db_query(monkeypatch):
    query = QueryRecorder()
    monkeypatch.setenv("SUPABASE_URL", "https://example.invalid")
    monkeypatch.setenv("SUPABASE_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "supabase", SimpleNamespace(
        ClientOptions=lambda **kwargs: kwargs, create_client=lambda *args, **kwargs: query,
    ))
    return query


def test_candidate_query_honors_explicit_table_and_both_category_filters(db_query):
    candidates.retrieve_candidates(
        {"category1": "top", "season": "summer"},
        table_filter="musinsa_pants", category2_filter="데님 팬츠",
        category2_keyword_filter="팬츠", include_embeddings=True,
    )
    assert ("table", ("musinsa_pants",)) in db_query.calls
    assert ("eq", ("category2", "데님 팬츠")) in db_query.calls
    assert ("ilike", ("category2", "%팬츠%")) in db_query.calls
    assert db_query.calls.index(("order", ("id",))) < next(
        i for i, (method, _) in enumerate(db_query.calls) if method == "range"
    )


@pytest.mark.parametrize("table,absent_column", [
    ("musinsa_top_clothes", "length"), ("musinsa_pants", "sleeve"),
])
def test_product_lookup_does_not_download_vectors_or_missing_columns(db_query, table, absent_column):
    candidates.retrieve_candidates({}, table_filter=table)
    selection = next(args[0] for method, args in db_query.calls if method == "select")
    assert selection != "*"
    assert "embedding" not in selection
    assert absent_column not in selection.split(",")


@pytest.mark.parametrize("category, keyword", [("skirt", "스커트"), ("one-piece", "원피스")])
def test_candidate_query_separates_shared_skirt_dress_table(db_query, category, keyword):
    plan = candidates.build_attribute_text_search_plan(
        "", 3, extracted_attributes=candidates.ParsedTextAttributes(category1=category, season="summer"),
    )
    candidates.retrieve_candidates(
        plan.parsed_attributes.to_dict(), table_filter=plan.table_filter,
        category2_keyword_filter=plan.category2_keyword_filter,
    )
    assert ("table", ("musinsa_skirt_dress",)) in db_query.calls
    assert ("ilike", ("category2", f"%{keyword}%")) in db_query.calls


def pipeline_without_io():
    pipeline = service.FashionRecommendationPipeline.__new__(service.FashionRecommendationPipeline)
    pipeline._default_provider = "gemini"
    pipeline._client_cache = {}
    return pipeline


@pytest.mark.parametrize("strict_rows", [[], [{"id": "p1", "season": ["여름"], "similarity": 0.9}], RuntimeError("filtered RPC unavailable")])
def test_search_passes_same_plan_scope_to_strict_and_rpc(monkeypatch, strict_rows):
    pipeline = pipeline_without_io()
    extraction = SimpleNamespace(
        status="ok", attributes=candidates.ParsedTextAttributes(category1="top", season="summer"),
        to_metadata=lambda: {"status": "ok"},
    )
    monkeypatch.setattr(service, "extract_conservative_metadata", lambda **kwargs: extraction)
    pipeline._run_target_description_reasoning = Mock(return_value=({"Target Image Description": "summer pants"}, "gemini", "ok", []))
    pipeline._translate = Mock(return_value="여름 바지")
    pipeline.retriever = SimpleNamespace(encode_text=lambda texts: np.array([[1.0, 0.0]]))
    pipeline._search_vectors = Mock(return_value=[{"id": "p2", "similarity": 0.5}])
    fetch = (Mock(side_effect=strict_rows) if isinstance(strict_rows, Exception) else
             Mock(return_value=(strict_rows, {"matched_count": len(strict_rows), "request_count": 1})))
    pipeline.vector_search = SimpleNamespace(search_filtered=fetch)
    monkeypatch.setattr(service, "generate_query_recommendation_explanation", lambda **kwargs: SimpleNamespace(
        recommendation_reason="추천 설명", to_metadata=lambda: {},
    ))
    scope = dict(table_filter="musinsa_pants", category2_filter="데님 팬츠", category2_keyword_filter="팬츠")
    if isinstance(strict_rows, Exception):
        with pytest.raises(RuntimeError, match="filtered RPC unavailable"):
            pipeline.search("여름 바지", **scope)
        pipeline._search_vectors.assert_not_called()
        return
    _, _, metadata, _ = pipeline.search("여름 바지", **scope)
    timing = metadata["pipeline"]["timings"]
    assert timing["total_ms"] >= 0
    stages = {stage["id"]: stage for stage in timing["stages"]}
    assert set(stages) == {
        "intent_extraction", "target_description", "search_plan", "translation",
        "embedding", "retrieval", "postprocess", "recommendation_explanation",
    }
    assert stages["intent_extraction"]["parallel_group"] == stages["target_description"]["parallel_group"]
    assert stages["translation"]["parallel_group"] == stages["embedding"]["parallel_group"]
    assert all(stage["duration_ms"] >= 0 for stage in stages.values())
    for key, expected in scope.items():
        assert fetch.call_args.kwargs.get(key) == expected
    assert fetch.call_args.kwargs["match_count"] == service.DEFAULT_RECOMMENDATION_COUNT
    assert metadata["pipeline"]["retrieval_diagnostics"]["request_count"] == 1
    if not strict_rows:
        assert metadata["pipeline"]["retrieval_diagnostics"]["fallback"]["reason"] == "no_strict_candidates"
        for key, expected in scope.items():
            assert pipeline._search_vectors.call_args.kwargs[key] == expected
    else:
        pipeline._search_vectors.assert_not_called()
        assert metadata["pipeline"]["prefetch_count"] == service.DEFAULT_RECOMMENDATION_COUNT


def run_reasoning(pipeline, provider="gemini"):
    return pipeline._run_target_description_reasoning(
        query="pizza", raw_fallback="pizza", image_bytes=None,
        image_mime_type=None, provider=provider,
    )


@pytest.mark.parametrize("provider_type", [generation.GeminiReasoningClient, generation.OpenAIReasoningClient, generation.Qwen3VLReasoningClient])
@pytest.mark.parametrize("payload, image", [('{"Is Fashion Query": false}', None), ('{"Is Fashion Image": false}', b"image")])
def test_nonfashion_response_is_invalid_search_input(provider_type, payload, image):
    if image:
        image_buffer = BytesIO()
        Image.new("RGB", (1, 1)).save(image_buffer, format="PNG")
        image = image_buffer.getvalue()
    client = provider_type.__new__(provider_type)
    client._generate = Mock(return_value=payload)
    invalid = getattr(generation, "InvalidSearchInput", None)
    assert invalid is not None
    with pytest.raises(invalid):
        client.run(manipulation_text="pizza", image_bytes=image)


def test_invalid_input_is_not_converted_to_raw_fallback():
    invalid = getattr(generation, "InvalidSearchInput", None)
    assert invalid is not None
    pipeline = pipeline_without_io()
    pipeline._get_reasoning_client = Mock(return_value=SimpleNamespace(run=Mock(side_effect=invalid("패션 관련 검색어를 입력해주세요."))))
    with pytest.raises(invalid):
        run_reasoning(pipeline)


def test_invalid_input_from_qwen_fallback_is_preserved():
    invalid = getattr(generation, "InvalidSearchInput", None)
    assert invalid is not None
    pipeline = pipeline_without_io()
    primary = SimpleNamespace(run=Mock(side_effect=generation.ReasoningProviderError("limit", "한도", "한도")))
    secondary = SimpleNamespace(run=Mock(side_effect=invalid("패션 관련 검색어를 입력해주세요.")))
    pipeline._get_reasoning_client = Mock(side_effect=[primary, secondary])
    with pytest.raises(invalid):
        run_reasoning(pipeline)


def test_provider_initialization_failure_uses_fallback():
    pipeline = pipeline_without_io()
    pipeline._get_reasoning_client = Mock(side_effect=RuntimeError("missing provider configuration"))
    result, provider, status, warnings = run_reasoning(pipeline)
    assert result["Target Image Description"] == "pizza"
    assert (provider, status) == ("raw_query", "degraded")
    assert warnings


def test_known_provider_initialization_failure_can_use_qwen():
    pipeline = pipeline_without_io()
    qwen = SimpleNamespace(run=Mock(return_value={"Target Image Description": "a top"}))
    pipeline._get_reasoning_client = Mock(side_effect=[
        generation.ReasoningProviderError("provider_unavailable", "연결 오류", "초기화 실패"), qwen,
    ])
    result, provider, status, warnings = run_reasoning(pipeline)
    assert result["Target Image Description"] == "a top"
    assert (provider, status) == ("qwen", "fallback")
    assert warnings


def test_unknown_provider_is_rejected_before_any_model_calls(monkeypatch):
    pipeline = pipeline_without_io()
    pipeline._run_target_description_reasoning = Mock(side_effect=AssertionError("model must not be called"))
    extraction = Mock(side_effect=AssertionError("model must not be called"))
    monkeypatch.setattr(service, "extract_conservative_metadata", extraction)
    with pytest.raises(ValueError, match="provider"):
        pipeline.search("상의", provider="typo")
    extraction.assert_not_called()


def test_unknown_provider_is_rejected_by_reasoning_entrypoint():
    pipeline = pipeline_without_io()
    pipeline._get_reasoning_client = Mock(side_effect=AssertionError("must not initialize"))
    with pytest.raises(ValueError, match="provider"):
        run_reasoning(pipeline, "typo")
    pipeline._get_reasoning_client.assert_not_called()


def test_unknown_provider_cannot_silently_initialize_gemini(monkeypatch):
    pipeline = pipeline_without_io()
    factory = Mock(side_effect=AssertionError("must not initialize Gemini"))
    monkeypatch.setattr(service, "GeminiReasoningClient", factory)
    with pytest.raises(ValueError, match="provider"):
        pipeline._get_reasoning_client("typo")
    factory.assert_not_called()


def mock_intent_model(monkeypatch, payload):
    import json
    from pipeline.intent import intent_extraction as intent

    module = intent.IntentExtractionModule.__new__(intent.IntentExtractionModule)
    generate = Mock(return_value=SimpleNamespace(text=json.dumps(payload)))
    module.client = SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    module._types = SimpleNamespace(
        Part=SimpleNamespace(from_text=lambda **kwargs: kwargs),
        Content=lambda **kwargs: kwargs,
    )
    module.model_name = "test-model"
    module.config = None
    monkeypatch.setattr(intent, "_get_module", lambda *args: module)
    return intent, generate


@pytest.mark.parametrize("query, payload, expected", [
    ("네이비 면 소재 셔츠", {"category1": "top", "fabric": "  Cotton  ", "color": " NAVY "}, {"category1": "top", "fabric": "cotton", "color": "navy"}),
    ("검정과 면 말고 아이보리 린넨 셔츠", {"category1": "top", "fabric": "Linen", "color": "Ivory"}, {"category1": "top", "fabric": "linen", "color": "ivory"}),
    ("검정과 면은 제외한 셔츠", {"category1": "top"}, {"category1": "top"}),
    ("파란 면 소재 셔츠", {"category1": "top"}, {"category1": "top"}),
    ("셔츠", {"category1": "top", "fabric": " NONE ", "color": " Null "}, {"category1": "top"}),
    ("셔츠", {"category1": "top", "fabric": ["cotton"], "color": {"value": "blue"}}, {"category1": "top"}),
    ("와이드 핏 바지", {"category1": "pants", "fit": "wide"}, {"category1": "pants", "fit": "wide"}),
])
def test_actual_intent_extraction_preserves_explicit_semantic_attributes(monkeypatch, query, payload, expected):
    intent, generate = mock_intent_model(monkeypatch, payload)
    extracted = intent.extract_conservative_metadata(query)
    assert extracted.status == "ok"
    assert extracted.attributes.to_dict() == expected
    plan = candidates.build_attribute_text_search_plan(
        query, top_k=3, extracted_attributes=extracted.attributes,
        extraction_metadata=extracted.to_metadata(),
    )
    assert plan.parsed_attributes.to_dict() == expected
    prompt = generate.call_args.kwargs["contents"][0]["parts"][0]["text"]
    assert "- color:" in prompt
    assert "- fabric:" in prompt
    assert "wide" in prompt
    assert "negated" in prompt


@pytest.mark.parametrize("payload, expected_fabric", [
    ({"category1": "top", "fabric": "Cotton"}, "cotton"),
    ({"category1": "top"}, None),
    ({"category1": "top", "fabric": ""}, None),
])
def test_search_embeds_only_successfully_extracted_fabric(monkeypatch, payload, expected_fabric):
    mock_intent_model(monkeypatch, payload)
    pipeline = pipeline_without_io()
    pipeline._run_target_description_reasoning = Mock(return_value=({"Target Image Description": "a cotton shirt"}, "gemini", "ok", []))
    pipeline._translate = Mock(return_value="면 셔츠")
    encode = Mock(return_value=np.array([[1.0, 0.0]]))
    pipeline.retriever = SimpleNamespace(encode_text=encode)
    pipeline._search_vectors = Mock(return_value=[{"id": "p2", "similarity": 0.5}])
    monkeypatch.setattr(service, "generate_query_recommendation_explanation", lambda **kwargs: SimpleNamespace(
        recommendation_reason="추천 설명", to_metadata=lambda: {},
    ))
    pipeline.search("면 소재 셔츠")
    encoded_texts = [call.args[0] for call in encode.call_args_list]
    assert ["a cotton shirt"] in encoded_texts
    if expected_fabric:
        assert [expected_fabric] in encoded_texts
        assert encode.call_count == 2
        assert pipeline._search_vectors.call_args.kwargs["fabric_embedding"] == [1.0, 0.0]
    else:
        assert encode.call_count == 1
        assert pipeline._search_vectors.call_args.kwargs["fabric_embedding"] is None
