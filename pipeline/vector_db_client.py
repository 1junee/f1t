from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from time import perf_counter
from typing import Any, Sequence

if __package__:
    from .env import load_pipeline_env
else:  # pragma: no cover - allows direct imports from pipeline/
    from env import load_pipeline_env

# 설정 파일은 모듈 로딩 시 읽고, 실제 URL·키는 클라이언트를 생성할 때 확정한다.
load_pipeline_env()


class SupabaseVectorSearchClient:
    """Supabase pgvector RPC client for 768-dimensional Gemini retrieval."""

    def __init__(
        self,
        supabase_url: str | None = None,
        supabase_key: str | None = None,
        timeout: int = 30,
    ) -> None:
        # 명시적으로 전달한 설정을 우선한다. 필수 설정이 없으면 검색 전에 생성 단계에서 실패한다.
        self.supabase_url = (supabase_url or os.getenv("SUPABASE_URL") or "").rstrip("/")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_KEY") or ""
        self.timeout = timeout

        if not self.supabase_url or not self.supabase_key:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_KEY are required for pgvector search."
            )

    @staticmethod
    def _to_pgvector(vector: Sequence[float]) -> str:
        # REST 요청 본문에 넣을 pgvector 문자열로 직렬화한다.
        # 숫자 목록 [0.1, 0.2, ...]을 DB가 읽을 문자열로 바꾸며 새 임베딩을 계산하지는 않는다.
        return "[" + ",".join(f"{float(value):.8f}" for value in vector) + "]"

    @staticmethod
    def _rpc_for_dim(dim: int) -> str:
        # 현재 배포된 컬럼·RPC는 768차원 전용이므로 다른 차원은 DB 호출 전에 막는다.
        if dim != 768:
            raise ValueError(
                f"Only 768-dimensional Gemini embeddings are supported; got {dim}."
            )
        return "match_fashion_items_768"

    # 목표 설명문 벡터·선택적 소재 벡터·범위 조건 → DB가 계산한 후보 상품 목록.
    # 상품 이미지/소재 벡터는 DB에 이미 저장돼 있으므로 이 요청에 모두 싣지 않는다.
    def search(
        self,
        embedding: Sequence[float],
        match_count: int,
        table_filter: str | None = None,
        category2_filter: str | None = None,
        category2_keyword_filter: str | None = None,
        fabric_embedding: Sequence[float] | None = None,
    ) -> list[dict[str, Any]]:
        # 벡터 검색 계산은 DB 함수가 수행한다. 이 클래스는 입력과 응답을 전달한다.
        rpc = self._rpc_for_dim(len(embedding))
        payload = {
            "p_query_embedding": self._to_pgvector(embedding),
            "p_match_count": match_count,
            "p_table_filter": table_filter,
            "p_category2_filter": category2_filter,
            "p_category2_keyword_filter": category2_keyword_filter,
        }
        if fabric_embedding:
            payload["p_fabric_embedding"] = self._to_pgvector(fabric_embedding)

        # /rpc/함수이름은 저장된 SQL 함수를 호출하는 HTTP 경로다. payload의 p_*는 SQL 인자와 대응한다.
        request = urllib.request.Request(
            f"{self.supabase_url}/rest/v1/rpc/{rpc}",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "apikey": self.supabase_key,
                "Authorization": f"Bearer {self.supabase_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            # 검색 결과 없음과 RPC 설치 누락은 다르다. 후자는 명시적인 오류로 올린다.
            if exc.code == 404 or "PGRST202" in body:
                raise RuntimeError(
                    f"Supabase RPC {rpc}를 찾지 못했습니다. "
                    "배포 전에 Supabase SQL Editor에서 match_fashion_items_768 RPC를 "
                    "생성했는지 확인해 주세요."
                ) from exc

            raise RuntimeError(
                f"Supabase vector search failed. ({exc.code}) {body[:400]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Supabase vector search failed: {exc.reason}") from exc

        # HTTP 요청이 성공해도 응답 형식이 틀릴 수 있어 JSON과 목록 타입을 각각 확인한다.
        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Supabase vector search returned invalid JSON.") from exc

        if not isinstance(data, list):
            raise RuntimeError(f"Unexpected Supabase vector search response: {data}")

        return data

    def search_filtered(
        self,
        embedding: Sequence[float],
        match_count: int,
        table_filter: str | None = None,
        category2_filter: str | None = None,
        category2_keyword_filter: str | None = None,
        fabric_embedding: Sequence[float] | None = None,
        *,
        metadata_filters: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """속성 필터와 정확 코사인 순위를 DB에서 계산하고 작은 결과만 받는다."""
        if __package__:
            from .retrieval.candidate_selection import (
                ARRAY_COLUMNS, TEXT_COLUMNS, VALID_TABLES, _normalize_candidate_row, translate_to_db_value,
            )
        else:  # pragma: no cover
            from retrieval.candidate_selection import (
                ARRAY_COLUMNS, TEXT_COLUMNS, VALID_TABLES, _normalize_candidate_row, translate_to_db_value,
            )

        self._rpc_for_dim(len(embedding))
        if fabric_embedding is not None:
            self._rpc_for_dim(len(fabric_embedding))
        # 색상·소재는 의미 검색 대상이다. 구조화된 정확 필터만 DB 저장값으로 번역한다.
        filters = {}
        for key in ARRAY_COLUMNS + TEXT_COLUMNS:
            value = metadata_filters.get(key)
            if value:
                translated = translate_to_db_value(key, value)
                filters[key] = translated if isinstance(translated, list) else [translated]

        rpc = "match_fashion_items_filtered_768"
        payload = {
            "p_query_embedding": self._to_pgvector(embedding),
            "p_match_count": match_count,
            "p_table_filter": table_filter,
            "p_category2_filter": category2_filter,
            "p_category2_keyword_filter": category2_keyword_filter,
            "p_metadata_filters": filters,
            "p_fabric_embedding": (
                self._to_pgvector(fabric_embedding) if fabric_embedding is not None else None
            ),
        }
        request = urllib.request.Request(
            f"{self.supabase_url}/rest/v1/rpc/{rpc}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"apikey": self.supabase_key,
                     "Authorization": f"Bearer {self.supabase_key}",
                     "Content-Type": "application/json"},
            method="POST",
        )
        started = perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                headers_received = perf_counter()
                body = response.read()
                body_received = perf_counter()
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 404 or "PGRST202" in error_body:
                raise RuntimeError(
                    f"Supabase RPC {rpc}를 찾지 못했습니다. "
                    "pipeline/sql/match_fashion_items_filtered_768.sql 배포를 확인해 주세요."
                ) from exc
            # SQL 오류 본문에는 내부 정보가 있을 수 있으므로 사용자 응답에 그대로 싣지 않는다.
            raise RuntimeError(f"Supabase filtered vector search failed. HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError("Supabase filtered vector search connection failed or timed out.") from exc

        try:
            data = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RuntimeError("Supabase filtered vector search response is invalid JSON.") from exc
        decoded = perf_counter()
        if not isinstance(data, dict):
            raise RuntimeError("Unexpected Supabase filtered vector search response.")
        rows, db_info = data.get("results"), data.get("diagnostics")
        if (not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows)
                or not isinstance(db_info, dict)):
            raise RuntimeError("Unexpected Supabase filtered vector search response shape.")
        # 타이머를 숫자로 검증해 잘못된 서버 응답을 성공/0초로 오인하지 않는다.
        for field in ("matched_count", "db_filter_ms", "db_rank_ms", "db_total_ms"):
            value = db_info.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise RuntimeError("Unexpected Supabase filtered vector search response diagnostics.")

        http_ms = (body_received - started) * 1000
        diagnostics = {
            **db_info,
            "request_count": 1,
            "response_bytes": len(body),
            # 헤더 대기는 연결·게이트웨이·DB 처리가 모두 포함된 값이며 순수 SQL 시간이 아니다.
            "headers_wait_ms": round((headers_received - started) * 1000, 2),
            "body_read_ms": round((body_received - headers_received) * 1000, 2),
            "json_decode_ms": round((decoded - body_received) * 1000, 2),
            "http_total_ms": round(http_ms, 2),
            # DB 내부 시간과 HTTP 시간은 중첩된다. 차이는 네트워크만의 정확한 측정값이 아니다.
            "transport_overhead_estimate_ms": round(max(http_ms - db_info["db_total_ms"], 0), 2),
        }
        # DB 계약이 달라져도 내부 벡터는 클라이언트 응답으로 전달하지 않는다.
        products = [{key: value for key, value in row.items()
                     if key not in {"gemini_image_embedding_768", "gemini_fabric_text_embedding_768"}}
                    for row in rows]
        # 기존 REST 경로처럼 이미지 URL이 비어 있으면 테이블·ID로 Storage 주소를 보완한다.
        products = [
            _normalize_candidate_row(row, row["source_table"], self.supabase_url)
            if row.get("source_table") in VALID_TABLES and row.get("id") else row
            for row in products
        ]
        return products, diagnostics
