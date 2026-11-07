# 실제 DB 대신 가짜 검색기를 사용해 벤치마크 분기·소재 전달·집계를 검사한다.
from __future__ import annotations

import json

import pytest

from pipeline.benchmarks.search_latency import (
    HybridSearchRunner,
    QueryCase,
    load_query_cases,
    summarize_samples,
)


def test_load_query_cases_supports_description_and_query_fallback(tmp_path):
    path = tmp_path / "queries.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "q1",
                    "query": "긴팔 상의",
                    "target_description": "a long-sleeve top",
                    "inferred_attributes": {"category1": "top", "sleeve": "long"},
                },
                {"id": "q2", "query": "출근룩 추천", "inferred_attributes": {}},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    cases = load_query_cases([path])

    assert [case.query_id for case in cases] == ["q1", "q2"]
    assert cases[0].target_description == "a long-sleeve top"
    assert cases[1].target_description == "출근룩 추천"
    assert cases[0].attributes["sleeve"] == "long"


def test_load_query_cases_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "queries.json"
    path.write_text(
        json.dumps([{"id": "same", "query": "a"}, {"id": "same", "query": "b"}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate query id"):
        load_query_cases([path])


def test_hybrid_runner_uses_filtered_candidates_before_hnsw():
    candidates = [
        {"id": "a", "gemini_image_embedding_768": [1.0, 0.0]},
        {"id": "b", "gemini_image_embedding_768": [0.0, 1.0]},
    ]
    fetch_calls = []

    def fetcher(attributes, **kwargs):
        fetch_calls.append((attributes, kwargs))
        return candidates

    class VectorClient:
        def search(self, **kwargs):
            raise AssertionError("HNSW should not be used when SQL candidates exist")

    runner = HybridSearchRunner(
        VectorClient(),
        top_k=1,
        max_candidates=20_000,
        page_size=1_000,
        candidate_fetcher=fetcher,
    )
    case = QueryCase(
        "q",
        "긴팔 상의",
        "long sleeve top",
        {"category1": "top", "sleeve": "long"},
        "test",
    )

    results, count, strategy = runner.search(case, [1.0, 0.0])

    assert results[0]["id"] == "a"
    assert count == 2
    assert strategy == "sql_prefilter_exact_cosine"
    assert fetch_calls[0][1]["max_rows"] == 20_000
    assert fetch_calls[0][1]["table_filter"] == "musinsa_top_clothes"
    assert fetch_calls[0][1]["category2_filter"] is None


def test_hybrid_runner_passes_fabric_embedding_to_hnsw_fallback():
    captured = {}

    class VectorClient:
        def search(self, **kwargs):
            captured.update(kwargs)
            return [{"id": "a", "similarity": 0.9}]

    runner = HybridSearchRunner(
        VectorClient(),
        top_k=1,
        max_candidates=20_000,
        page_size=1_000,
        candidate_fetcher=lambda *_args, **_kwargs: [],
    )
    case = QueryCase(
        "q",
        "울 재킷",
        "wool jacket",
        {"category1": "outer", "fabric": "wool"},
        "test",
    )

    results, count, strategy = runner.search(
        case,
        [1.0, 0.0],
        fabric_embedding=[0.0, 1.0],
    )

    assert results[0]["id"] == "a"
    assert count == 300
    assert strategy == "hnsw_fallback"
    assert captured["fabric_embedding"] == [0.0, 1.0]


def test_summarize_samples_reports_paired_latency_and_candidate_reduction():
    samples = [
        {
            "query_id": "q1",
            "repetition": 1,
            "method": "full_scan",
            "duration_ms": 100.0,
            "candidate_count": 100,
            "strategy": "database_full_scan",
            "error": None,
        },
        {
            "query_id": "q1",
            "repetition": 1,
            "method": "hybrid",
            "duration_ms": 25.0,
            "candidate_count": 20,
            "strategy": "sql_prefilter_exact_cosine",
            "error": None,
        },
        {
            "query_id": "q2",
            "repetition": 1,
            "method": "full_scan",
            "duration_ms": 120.0,
            "candidate_count": 100,
            "strategy": "database_full_scan",
            "error": None,
        },
        {
            "query_id": "q2",
            "repetition": 1,
            "method": "hybrid",
            "duration_ms": 30.0,
            "candidate_count": 10,
            "strategy": "hnsw_fallback",
            "error": None,
        },
    ]

    summary = summarize_samples(samples)

    assert summary["paired"]["pair_count"] == 2
    assert summary["paired"]["mean_saved_ms"] == 82.5
    assert summary["paired"]["mean_speedup"] == 4.0
    assert summary["candidates"]["reduction_pct"] == 85.0


def test_defaults_match_production_candidate_limits():
    from pipeline.benchmarks.search_latency import build_parser
    args = build_parser().parse_args(["queries.json"])
    assert (args.max_candidates, args.page_size) == (1000, 100)


def test_candidate_error_falls_back_with_production_prefetch_and_filters():
    calls = []
    def fail_candidates(*_args, **_kwargs):
        raise RuntimeError("temporary REST error")
    class VectorClient:
        def search(self, **kwargs):
            calls.append(kwargs)
            return [{"id": "a", "similarity": 0.8, "sleeve": "long"}]
    runner = HybridSearchRunner(VectorClient(), top_k=1, max_candidates=1000,
                                page_size=100, candidate_fetcher=fail_candidates)
    case = QueryCase("q", "긴팔 상의", "long sleeve top",
                     {"category1": "top", "sleeve": "long"}, "test")
    results, count, strategy = runner.search(case, [1.0, 0.0])
    assert results[0]["id"] == "a"
    assert calls[0]["match_count"] == 1000
    assert calls[0]["table_filter"] == "musinsa_top_clothes"
    assert count == 100
    assert strategy == "hnsw_fallback_prefilter_error"


def test_hybrid_applies_production_attribute_postfilter():
    class VectorClient:
        def search(self, **_kwargs):
            return [
                {"id": "wrong", "similarity": .99, "sleeve": "short"},
                {"id": "right", "similarity": .5, "sleeve": "long"},
            ]
    runner = HybridSearchRunner(VectorClient(), top_k=1, max_candidates=1000,
                                page_size=100, candidate_fetcher=lambda *a, **kw: [])
    case = QueryCase("q", "긴팔 상의", "long sleeve top",
                     {"category1": "top", "sleeve": "long"}, "test")
    results, _, _ = runner.search(case, [1.0, 0.0])
    assert [row["id"] for row in results] == ["right"]


def test_paired_speedup_excludes_unpaired_successes():
    samples = [
        {"query_id": "paired", "repetition": 1, "method": method,
         "duration_ms": duration, "candidate_count": 1, "strategy": "test", "error": None}
        for method, duration in [("full_scan", 100), ("hybrid", 25)]
    ]
    samples.append({**samples[0], "query_id": "unpaired", "duration_ms": 10000})
    samples.append({**samples[1], "query_id": "unpaired", "error": "failure"})
    summary = summarize_samples(samples)
    assert summary["paired"]["mean_speedup"] == 4
    assert summary["paired"]["mean_latency_reduction_pct"] == 75
    assert summary["failed_sample_count"] == 1


def test_cache_invalidates_on_embedding_contract_change(tmp_path):
    import numpy as np
    from pipeline.benchmarks.search_latency import EmbeddingCache
    class Encoder:
        calls = 0
        def encode_text(self, texts):
            self.calls += 1
            return np.ones((len(texts), 768))
    path = tmp_path / "embeddings.json"
    encoder = Encoder()
    items = [("target:q", "same text", "q")]
    EmbeddingCache(path, model="model-a").prepare_texts(items, encoder)
    EmbeddingCache(path, model="model-a").prepare_texts(items, encoder)
    assert encoder.calls == 1
    EmbeddingCache(path, model="model-b").prepare_texts(items, encoder)
    assert encoder.calls == 2
    EmbeddingCache(path, model="model-b", preprocessing="different-prefix").prepare_texts(items, encoder)
    assert encoder.calls == 3


def test_cache_rejects_wrong_output_dimensions(tmp_path):
    import numpy as np
    from pipeline.benchmarks.search_latency import EmbeddingCache
    class Encoder:
        def encode_text(self, _texts):
            return np.ones((1, 2))
    path = tmp_path / "embeddings.json"
    with pytest.raises(ValueError, match="dimension"):
        EmbeddingCache(path).prepare_texts([("q", "query", "q")], Encoder())
    assert not path.exists()


def test_hybrid_does_not_fill_successful_empty_attributes_from_negated_query():
    calls = []
    class VectorClient:
        def search(self, **kwargs):
            calls.append(kwargs)
            return [{"id": "a", "similarity": .8}]
    def reject_candidates(*_args, **_kwargs):
        raise AssertionError("No exact filters were extracted")
    runner = HybridSearchRunner(VectorClient(), top_k=1, max_candidates=1000,
                                page_size=100, candidate_fetcher=reject_candidates)
    case = QueryCase("q", "긴팔은 제외한 셔츠", "shirt without long sleeves", {}, "test")
    runner.search(case, [1., 0.])
    assert all(call["match_count"] == 1 for call in calls)
