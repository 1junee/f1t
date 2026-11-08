"""Compare exhaustive vector search with the production hybrid retrieval path.

The measured interval intentionally excludes VLM target-description generation and
Gemini query embedding. Query embeddings are prepared once, cached, and reused by
both retrieval methods.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import statistics
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from pipeline.env import load_pipeline_env
from pipeline.recommendation_service import (
    VALID_TABLES,
    _has_exact_metadata_filters,
    _rank_stored_gemini_vectors,
)
from pipeline.retrieval.candidate_selection import (
    METADATA_PREFETCH_LIMIT,
    ParsedTextAttributes,
    build_attribute_text_search_plan,
    refine_attribute_text_results,
    retrieve_candidates,
)
from pipeline.retrieval.target_description_retrieval import (
    GEMINI_EMBEDDING_DIM, GEMINI_EMBEDDING_MODEL, RetrieveModule,
)
from pipeline.vector_db_client import SupabaseVectorSearchClient


BASELINE_RPC = "benchmark_full_scan_fashion_items_768"
DEFAULT_TOP_K = 10
DEFAULT_REPETITIONS = 3
DEFAULT_EXPECTED_QUERY_COUNT = 108
DEFAULT_MAX_CANDIDATES = METADATA_PREFETCH_LIMIT
DEFAULT_PAGE_SIZE = 100
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueryCase:
    query_id: str
    query: str
    target_description: str
    attributes: dict[str, Any]
    source: str


@dataclass(frozen=True)
class SearchObservation:
    method: str
    duration_ms: float
    candidate_count: int
    strategy: str
    result_count: int


def _records_from_path(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("queries", "cases", "data", "results"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError(
        f"{path}: expected a JSON array, JSONL, or an object with a query list"
    )


def _target_description(record: dict[str, Any], query: str) -> str:
    direct = record.get("target_description") or record.get("retrieval_description")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    for container_key in (
        "target_description_reasoning",
        "reasoning_result",
        "reasoning",
    ):
        container = record.get(container_key)
        if isinstance(container, dict):
            value = container.get("Target Image Description")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return query


def load_query_cases(paths: Sequence[Path]) -> list[QueryCase]:
    # 입력 형식을 통일하고 중복 ID를 막는다. 설명문이 없으면 원래 질의를 사용한다.
    cases: list[QueryCase] = []
    seen_ids: set[str] = set()
    for path in paths:
        for index, record in enumerate(_records_from_path(path), start=1):
            if not isinstance(record, dict):
                raise ValueError(f"{path}: record {index} is not an object")
            query = str(record.get("query") or "").strip()
            if not query:
                raise ValueError(f"{path}: record {index} has no query")
            query_id = str(record.get("id") or f"{path.stem}-{index}")
            if query_id in seen_ids:
                raise ValueError(f"duplicate query id: {query_id}")
            seen_ids.add(query_id)
            raw_attributes = record.get("inferred_attributes") or record.get(
                "attributes"
            )
            attributes = (
                dict(raw_attributes) if isinstance(raw_attributes, dict) else {}
            )
            attributes = {
                key: value
                for key, value in attributes.items()
                if key not in {"reasoning", "error", "raw_output"}
            }
            cases.append(
                QueryCase(
                    query_id=query_id,
                    query=query,
                    target_description=_target_description(record, query),
                    attributes=attributes,
                    source=str(path),
                )
            )
    return cases


class EmbeddingCache:
    # 문장뿐 아니라 모델·차원·입력 전처리가 같을 때만 벡터를 재사용한다.
    def __init__(
        self, path: Path, *, model: str | None = None,
        dimensions: int | None = None,
        preprocessing: str = "task: search result | query: {text}",
    ) -> None:
        self.path = path
        self.contract = {
            "model": model or os.getenv("GEMINI_EMBEDDING_MODEL", GEMINI_EMBEDDING_MODEL),
            "dimensions": dimensions or int(os.getenv("GEMINI_EMBEDDING_DIM", str(GEMINI_EMBEDDING_DIM))),
            "preprocessing": preprocessing,
        }
        self._entries: dict[str, dict[str, Any]] = {}
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                self._entries = payload

    def prepare(
        self,
        cases: Sequence[QueryCase],
        encoder: RetrieveModule,
    ) -> dict[str, list[float]]:
        return self.prepare_texts(
            [
                (
                    f"target:{case.query_id}",
                    case.target_description,
                    case.query_id,
                )
                for case in cases
            ],
            encoder,
        )

    def prepare_fabric(
        self,
        cases: Sequence[QueryCase],
        encoder: RetrieveModule,
    ) -> dict[str, list[float]]:
        return self.prepare_texts(
            [
                (f"fabric:{case.query_id}", str(fabric).strip(), str(fabric).strip())
                for case in cases
                if (fabric := case.attributes.get("fabric"))
                and str(fabric).strip()
            ],
            encoder,
        )

    def prepare_texts(
        self,
        items: Sequence[tuple[str, str, str]],
        encoder: RetrieveModule,
    ) -> dict[str, list[float]]:
        changed = False
        result: dict[str, list[float]] = {}
        for cache_key, text, result_key in items:
            # 계약 정보가 없는 예전 캐시도 다시 계산한다. 같은 차원만으로 같은 벡터 공간이 되지는 않는다.
            cached = self._entries.get(cache_key)
            if (
                isinstance(cached, dict)
                and cached.get("text") == text
                and isinstance(cached.get("embedding"), list)
                and len(cached["embedding"]) == self.contract["dimensions"]
                and cached.get("contract") == self.contract
            ):
                embedding = [float(value) for value in cached["embedding"]]
            else:
                embedding = encoder.encode_text([text])[0].tolist()
                if len(embedding) != self.contract["dimensions"]:
                    raise ValueError("Embedding output dimension does not match cache contract")
                self._entries[cache_key] = {
                    "text": text,
                    "embedding": embedding,
                    "contract": self.contract,
                }
                changed = True
            result[result_key] = embedding

        if changed or not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(self._entries, ensure_ascii=False),
                encoding="utf-8",
            )
            temporary.replace(self.path)
        return result


class ExhaustiveSearchClient:
    def __init__(
        self,
        supabase_url: str,
        supabase_key: str,
        timeout_seconds: int = 60,
    ) -> None:
        self.supabase_url = supabase_url.rstrip("/")
        self.supabase_key = supabase_key
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _to_pgvector(vector: Sequence[float]) -> str:
        return "[" + ",".join(f"{float(value):.8f}" for value in vector) + "]"

    def search(self, embedding: Sequence[float], top_k: int) -> list[dict[str, Any]]:
        request = urllib.request.Request(
            f"{self.supabase_url}/rest/v1/rpc/{BASELINE_RPC}",
            data=json.dumps(
                {
                    "p_query_embedding": self._to_pgvector(embedding),
                    "p_match_count": top_k,
                }
            ).encode("utf-8"),
            headers={
                "apikey": self.supabase_key,
                "Authorization": f"Bearer {self.supabase_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 404 or "PGRST202" in body:
                sql_path = (
                    Path(__file__).parents[1]
                    / "sql"
                    / "benchmark_full_scan_fashion_items_768.sql"
                )
                raise RuntimeError(
                    f"Benchmark RPC {BASELINE_RPC} is not installed. Apply {sql_path} "
                    "to the benchmark database first."
                ) from exc
            raise RuntimeError(
                f"Full-scan RPC failed ({exc.code}): {body[:400]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Full-scan RPC failed: {exc.reason}") from exc

        if not isinstance(payload, list):
            raise RuntimeError(f"Unexpected full-scan response: {payload}")
        return payload


class HybridSearchRunner:
    # 후보가 있으면 로컬 재정렬, 없으면 RPC 검색이라는 운영 검색의 분기를 따라간다.
    # 운영과 같은 계획·후처리 함수를 쓰지만 생성/임베딩 시간은 제외한 검색 구간 실험이다.
    def __init__(
        self,
        vector_client: SupabaseVectorSearchClient,
        *,
        top_k: int,
        max_candidates: int,
        page_size: int,
        candidate_fetcher: Callable[..., list[dict[str, Any]]] = retrieve_candidates,
    ) -> None:
        self.vector_client = vector_client
        self.top_k = top_k
        self.max_candidates = max_candidates
        self.page_size = page_size
        self.candidate_fetcher = candidate_fetcher

    def search(
        self,
        case: QueryCase,
        embedding: list[float],
        fabric_embedding: list[float] | None = None,
    ) -> tuple[list[dict[str, Any]], int, str]:
        # 저장된 명시 속성을 성공한 추출 결과로 재생한다. 모델 호출은 측정 밖이다.
        extracted = ParsedTextAttributes(**{
            key: value for key, value in case.attributes.items()
            if key in ParsedTextAttributes.__dataclass_fields__
        })
        plan = build_attribute_text_search_plan(
            case.query, self.top_k, extracted_attributes=extracted,
            extraction_info={"status": "ok"},
        )
        attributes = plan.parsed_attributes.to_dict()
        prefilter_failed = False
        if _has_exact_metadata_filters(attributes):
            try:
                candidates = self.candidate_fetcher(
                    attributes, include_embeddings=True,
                    page_size=self.page_size, max_rows=self.max_candidates,
                    table_filter=plan.table_filter,
                    category2_filter=plan.category2_filter,
                    category2_keyword_filter=plan.category2_keyword_filter,
                )
                if candidates:
                    ranked = _rank_stored_gemini_vectors(
                        embedding=embedding, rows=candidates, top_k=self.top_k,
                        fabric_embedding=fabric_embedding,
                    )
                    results, _ = refine_attribute_text_results(
                        ranked, plan.parsed_attributes, self.top_k,
                    )
                    return results, len(candidates), "sql_prefilter_exact_cosine"
            except Exception:
                # 운영과 동일하게 REST 후보 조회/정렬 실패도 RPC 경로로 전환한다.
                logger.exception("Benchmark candidate prefilter failed")
                prefilter_failed = True

        filtered = bool(plan.table_filter or plan.category2_filter or plan.category2_keyword_filter)
        tables = [plan.table_filter] if filtered else sorted(VALID_TABLES)
        rows: list[dict[str, Any]] = []
        for table in tables:
            # 테이블 일부만 성공한 결과는 정상 표본으로 집계하지 않는다.
            rows.extend(self.vector_client.search(
                embedding=embedding, match_count=plan.prefetch_count,
                table_filter=table,
                category2_filter=plan.category2_filter,
                category2_keyword_filter=plan.category2_keyword_filter,
                fabric_embedding=fabric_embedding,
            ))
        if not filtered and not rows:
            raise RuntimeError("모든 테이블 검색이 실패했습니다. 잠시 후 다시 시도해주세요.")
        rows.sort(key=lambda row: float(row.get("similarity") or 0), reverse=True)
        results, _ = refine_attribute_text_results(
            rows[:plan.prefetch_count], plan.parsed_attributes, self.top_k,
        )
        candidates_per_table = min(max(plan.prefetch_count * 5, 50), 100)
        return (
            results, candidates_per_table * len(tables),
            "hnsw_fallback_prefilter_error" if prefilter_failed else "hnsw_fallback",
        )


def _measure(
    method: str,
    action: Callable[[], tuple[list[dict[str, Any]], int, str]],
) -> SearchObservation:
    # DB 요청과 결과 수신을 포함한 검색 구간만 잰다. 임베딩은 호출 전에 준비된다.
    started = time.perf_counter_ns()
    results, candidate_count, strategy = action()
    duration_ms = (time.perf_counter_ns() - started) / 1_000_000
    return SearchObservation(
        method=method,
        duration_ms=duration_ms,
        candidate_count=candidate_count,
        strategy=strategy,
        result_count=len(results),
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_samples(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    # 성공 표본의 지연을 집계하고, 같은 질의·반복끼리 두 방식의 차이를 비교한다.
    # 실패 표본은 지연 집계에서 제외되므로 평균·p95와 실패 개수를 함께 읽어야 한다.
    successful = [sample for sample in samples if sample.get("error") is None]
    durations: dict[str, list[float]] = {
        method: [
            float(sample["duration_ms"])
            for sample in successful
            if sample["method"] == method
        ]
        for method in ("full_scan", "hybrid")
    }

    method_summary: dict[str, Any] = {}
    for method, values in durations.items():
        method_summary[method] = {
            "sample_count": len(values),
            "mean_ms": statistics.fmean(values) if values else math.nan,
            "median_ms": statistics.median(values) if values else math.nan,
            "p95_ms": _percentile(values, 0.95),
            "min_ms": min(values) if values else math.nan,
            "max_ms": max(values) if values else math.nan,
        }

    paired: dict[tuple[str, int], dict[str, float]] = {}
    for sample in successful:
        paired.setdefault((sample["query_id"], sample["repetition"]), {})[
            sample["method"]
        ] = float(sample["duration_ms"])
    paired_deltas = [
        values["full_scan"] - values["hybrid"]
        for values in paired.values()
        if {"full_scan", "hybrid"}.issubset(values)
    ]
    # 비교 비율도 두 방식이 모두 성공한 같은 질의·반복만 사용한다.
    complete_pairs = [values for values in paired.values() if {"full_scan", "hybrid"}.issubset(values)]
    full_mean = statistics.fmean(v["full_scan"] for v in complete_pairs) if complete_pairs else math.nan
    hybrid_mean = statistics.fmean(v["hybrid"] for v in complete_pairs) if complete_pairs else math.nan

    hybrid_samples = [sample for sample in successful if sample["method"] == "hybrid"]
    candidate_counts = [int(sample["candidate_count"]) for sample in hybrid_samples]
    baseline_counts = [
        int(sample["candidate_count"])
        for sample in successful
        if sample["method"] == "full_scan"
    ]
    baseline_candidate_mean = (
        statistics.fmean(baseline_counts) if baseline_counts else math.nan
    )
    hybrid_candidate_mean = (
        statistics.fmean(candidate_counts) if candidate_counts else math.nan
    )

    return {
        "methods": method_summary,
        "paired": {
            "pair_count": len(paired_deltas),
            "mean_saved_ms": (
                statistics.fmean(paired_deltas) if paired_deltas else math.nan
            ),
            "median_saved_ms": (
                statistics.median(paired_deltas) if paired_deltas else math.nan
            ),
            "mean_speedup": (
                full_mean / hybrid_mean
                if hybrid_mean and not math.isnan(hybrid_mean)
                else math.nan
            ),
            "mean_latency_reduction_pct": (
                (1 - hybrid_mean / full_mean) * 100
                if full_mean and not math.isnan(full_mean)
                else math.nan
            ),
        },
        "candidates": {
            "full_scan_mean": baseline_candidate_mean,
            "hybrid_mean": hybrid_candidate_mean,
            "reduction_pct": (
                (1 - hybrid_candidate_mean / baseline_candidate_mean) * 100
                if baseline_candidate_mean and not math.isnan(baseline_candidate_mean)
                else math.nan
            ),
            "strategy_counts": dict(
                Counter(sample["strategy"] for sample in hybrid_samples)
            ),
        },
        "failed_sample_count": len(samples) - len(successful),
    }


def run_benchmark(
    cases: Sequence[QueryCase],
    embeddings: dict[str, list[float]],
    fabric_embeddings: dict[str, list[float]],
    exhaustive: ExhaustiveSearchClient,
    hybrid: HybridSearchRunner,
    *,
    repetitions: int,
    warmup_queries: int,
    seed: int,
) -> list[dict[str, Any]]:
    # 워밍업은 연결·캐시 준비용이며 측정 표본에 넣지 않는다. 여기서 실패하면 본 측정에 진입하지 않는다.
    for case in cases[:warmup_queries]:
        embedding = embeddings[case.query_id]
        fabric_embedding = fabric_embeddings.get(case.query_id)
        exhaustive.search(embedding, hybrid.top_k)
        hybrid.search(case, embedding, fabric_embedding)

    schedule = [
        (case, repetition) for repetition in range(1, repetitions + 1) for case in cases
    ]
    # 질의 순서와 방식 실행 순서를 섞어 실행 순서에 따른 영향을 줄인다.
    random.Random(seed).shuffle(schedule)

    samples: list[dict[str, Any]] = []
    for case, repetition in schedule:
        embedding = embeddings[case.query_id]
        fabric_embedding = fabric_embeddings.get(case.query_id)
        methods = ["full_scan", "hybrid"]
        stable_id_sum = sum(case.query_id.encode("utf-8"))
        if (stable_id_sum + repetition) % 2:
            methods.reverse()
        for method in methods:
            try:
                if method == "full_scan":
                    observation = _measure(
                        method,
                        lambda: _baseline_action(exhaustive, embedding, hybrid.top_k),
                    )
                else:
                    observation = _measure(
                        method,
                        lambda: hybrid.search(case, embedding, fabric_embedding),
                    )
                samples.append(
                    {
                        "query_id": case.query_id,
                        "query": case.query,
                        "source": case.source,
                        "repetition": repetition,
                        **observation.__dict__,
                        "error": None,
                    }
                )
            except Exception as exc:
                samples.append(
                    {
                        "query_id": case.query_id,
                        "query": case.query,
                        "source": case.source,
                        "repetition": repetition,
                        "method": method,
                        "duration_ms": math.nan,
                        "candidate_count": 0,
                        "strategy": "error",
                        "result_count": 0,
                        "error": str(exc),
                    }
                )
    return samples


def _baseline_action(
    exhaustive: ExhaustiveSearchClient,
    embedding: list[float],
    top_k: int,
) -> tuple[list[dict[str, Any]], int, str]:
    results = exhaustive.search(embedding, top_k)
    candidate_count = int(results[0].get("candidate_count") or 0) if results else 0
    return results, candidate_count, "database_full_scan"


def _write_results(output: Path, payload: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    csv_path = output.with_suffix(".csv")
    fieldnames = [
        "query_id",
        "query",
        "source",
        "repetition",
        "method",
        "duration_ms",
        "candidate_count",
        "strategy",
        "result_count",
        "error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(payload["samples"])


def _safe_json_value(value: Any) -> Any:
    # 표본이 없을 때 생긴 NaN 등을 JSON의 null로 바꿔 표준 JSON으로 저장한다.
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {key: _safe_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe_json_value(item) for item in value]
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query_files", nargs="+", type=Path)
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=Path(".omx/benchmarks/query_embeddings.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path(".omx/benchmarks/search_latency.json")
    )
    parser.add_argument(
        "--expected-count", type=int, default=DEFAULT_EXPECTED_QUERY_COUNT
    )
    parser.add_argument("--strict-count", action="store_true")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--warmup-queries", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cases = load_query_cases(args.query_files)
    fallback_count = sum(case.target_description == case.query for case in cases)
    validation = {
        "query_count": len(cases),
        "expected_query_count": args.expected_count,
        "query_count_matches": len(cases) == args.expected_count,
        "raw_query_description_fallback_count": fallback_count,
        "exact_filter_query_count": sum(
            _has_exact_metadata_filters(case.attributes) for case in cases
        ),
    }
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    if args.strict_count and not validation["query_count_matches"]:
        raise SystemExit(
            f"Expected {args.expected_count} queries, loaded {len(cases)}. "
            "Do not duplicate queries to satisfy the count."
        )
    if args.dry_run:
        # 입력 확인만 할 때는 API 클라이언트를 만들거나 실험을 실행하지 않는다.
        return 0

    load_pipeline_env()
    supabase_url = os.getenv("SUPABASE_URL", "")
    supabase_key = os.getenv("SUPABASE_KEY", "")
    if not supabase_url or not supabase_key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY are required")

    embedding_cache = EmbeddingCache(args.embedding_cache)
    encoder = RetrieveModule(
        model=embedding_cache.contract["model"],
        dimensionality=embedding_cache.contract["dimensions"],
    )
    embeddings = embedding_cache.prepare(cases, encoder)
    fabric_embeddings = embedding_cache.prepare_fabric(cases, encoder)
    exhaustive = ExhaustiveSearchClient(supabase_url, supabase_key)
    hybrid = HybridSearchRunner(
        SupabaseVectorSearchClient(supabase_url, supabase_key),
        top_k=args.top_k,
        max_candidates=args.max_candidates,
        page_size=args.page_size,
    )
    samples = run_benchmark(
        cases,
        embeddings,
        fabric_embeddings,
        exhaustive,
        hybrid,
        repetitions=args.repetitions,
        warmup_queries=args.warmup_queries,
        seed=args.seed,
    )
    payload = _safe_json_value(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "measurement_scope": (
                "retrieval only; excludes VLM target-description and query embedding generation"
            ),
            "configuration": {
                "query_files": [str(path) for path in args.query_files],
                "query_count": len(cases),
                "top_k": args.top_k,
                "repetitions": args.repetitions,
                "warmup_queries": args.warmup_queries,
                "max_candidates": args.max_candidates,
                "page_size": args.page_size,
                "embedding_contract": embedding_cache.contract,
                "postfilter": "refine_attribute_text_results",
                "candidate_count_scope": "REST rows or RPC candidate ceiling, not visited HNSW nodes",
                "seed": args.seed,
                "raw_query_description_fallback_count": fallback_count,
            },
            "summary": summarize_samples(samples),
            "samples": samples,
        }
    )
    _write_results(args.output, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"JSON: {args.output}")
    print(f"CSV:  {args.output.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
