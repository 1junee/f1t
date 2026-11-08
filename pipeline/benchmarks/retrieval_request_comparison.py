"""3회 순차 RPC + 서버 병합의 이전 구현. 운영에서는 호출하지 않는다.

동일 입력으로 SupabaseVectorSearchClient.search_approximate()와 비교한다.
범위가 있으면 기존처럼 1회, 범위가 없으면 3회 호출한다.
기존의 전체 빈 결과를 오류로 처리하는 동작도 재현 목적으로 보존했다.
성능 비교에서는 성공/빈 결과/오류를 구분해 기록해야 한다.
"""
from __future__ import annotations

import logging
from typing import Any

from pipeline.recommendation_service import VALID_TABLES
from pipeline.vector_db_client import SupabaseVectorSearchClient

logger = logging.getLogger(__name__)


def search_three_requests_legacy(
    vector_search: SupabaseVectorSearchClient,
    embedding: list[float],
    match_count: int,
    table_filter: str | None,
    category2_filter: str | None,
    category2_keyword_filter: str | None,
    fabric_embedding: list[float] | None = None,
) -> list[dict[str, Any]]:

    if table_filter or category2_filter or category2_keyword_filter:
        return vector_search.search_approximate(
            embedding=embedding,
            match_count=match_count,
            table_filter=table_filter,
            category2_filter=category2_filter,
            category2_keyword_filter=category2_keyword_filter,
            fabric_embedding=fabric_embedding,
        )

    # 과거 구현 재현: 테이블별 순차 호출. 연결 한도가 원인이었다는 근거는 확인되지 않았다.
    table_results = []
    failed_tables: list[str] = []
    for table in sorted(VALID_TABLES):
        try:
            rows = vector_search.search_approximate(
                embedding=embedding,
                match_count=match_count,
                table_filter=table,
                fabric_embedding=fabric_embedding,
            )
            table_results.append(rows)
        except Exception:
            logger.exception("Vector search failed for table %s", table)
            failed_tables.append(table)

    # 일부 카테고리가 누락된 결과를 정상 응답처럼 보여주지 않고 실패로 처리한다.
    if failed_tables:
        raise RuntimeError(
            "일부 상품 카테고리 검색에 실패했습니다. 잠시 후 다시 시도해주세요."
        )

    merged = [row for rows in table_results for row in rows]
    if not merged:
        raise RuntimeError("모든 테이블 검색이 실패했습니다. 잠시 후 다시 시도해주세요.")
    # 테이블별 상위 후보를 합쳐 다시 정렬해야 전체 카테고리의 상위 후보를 고를 수 있다.
    merged.sort(key=lambda row: float(row.get("similarity") or 0), reverse=True)
    return merged[:match_count]
