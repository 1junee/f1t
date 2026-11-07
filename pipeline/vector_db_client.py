from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
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
