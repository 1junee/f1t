"""기존 Python 벡터 계산을 재현하는 비교 실험용 코드. 운영 검색에서는 사용하지 않는다."""
from __future__ import annotations

import json
from typing import Any

import numpy as np


# [이전 방식 비교 실험용] 현재 운영 검색에서는 호출하지 않는다.
# DB에서 받은 상품 벡터를 Python에서 계산하던 경로를 재현한다.
# 현재 운영 검색의 점수 계산·정렬은 DB의 exact/approximate RPC가 담당한다.
def _rank_stored_gemini_vectors(
    embedding: list[float],
    rows: list[dict[str, Any]],
    top_k: int,
    fabric_embedding: list[float] | None = None,
    fabric_weight: float = 0.2,  # 최종 점수 = 0.8 × 이미지 유사도 + 0.2 × fabric 유사도
) -> list[dict[str, Any]]:
    # 각 행의 상품 이미지 벡터와 질의 벡터를 비교한다. 여기서는 HNSW를 사용하지 않는다.
    image_weight = 1.0 - fabric_weight if fabric_embedding is not None else 1.0
    query = np.asarray(embedding, dtype=np.float32)

    # query는 길이 D인 벡터, vectors는 상품 수 N × 차원 D의 행렬이다.
    vectors = np.asarray(
        [_parse_stored_vector(row.get("gemini_image_embedding_768")) for row in rows],
        dtype=np.float32,
    )

    # 벡터의 길이 차이를 제거한 코사인 유사도: 내적 / (두 벡터의 크기 곱).
    query_norm = float(np.linalg.norm(query))
    row_norms = np.linalg.norm(vectors, axis=1)
    denom = row_norms * query_norm

    # 영벡터가 있을 때 0으로 나누지 않도록 분모를 보호한다. 영벡터의 내적은 0으로 남는다.
    denom[denom == 0] = 1.0
    image_scores = (vectors @ query) / denom

    if fabric_embedding is not None:
        fq = np.asarray(fabric_embedding, dtype=np.float32)
        fq_norm = float(np.linalg.norm(fq))
        fabric_vectors = []
        for row in rows:
            raw = row.get("gemini_fabric_text_embedding_768")
            try:
                fabric_vectors.append(_parse_stored_vector(raw))
            except Exception:
                # 소재 벡터가 없으면 소재 점수는 0이며 이미지 가중치를 1로 되돌리지는 않는다.
                fabric_vectors.append([0.0] * len(fq))
        fv = np.asarray(fabric_vectors, dtype=np.float32)
        fv_norms = np.linalg.norm(fv, axis=1)
        fdenom = fv_norms * fq_norm
        fdenom[fdenom == 0] = 1.0
        fabric_scores = (fv @ fq) / fdenom
        final_scores = image_weight * image_scores + fabric_weight * fabric_scores
    else:
        final_scores = image_scores

    # argsort는 작은 점수부터 정렬하므로 뒤집어 높은 점수의 상품 인덱스만 선택한다.
    ranked_indices = np.argsort(final_scores)[::-1][:top_k]
    # 순위 계산이 끝난 큰 벡터는 결과에서 빼고 상품 정보와 최종 점수만 남긴다.
    exclude = {"gemini_image_embedding_768", "gemini_fabric_text_embedding_768"}
    return [
        {
            **{k: v for k, v in rows[int(i)].items() if k not in exclude},
            "similarity": float(final_scores[int(i)]),
        }
        for i in ranked_indices
    ]


def _parse_stored_vector(value: Any) -> list[float]:
    # DB 응답의 배열과 JSON 문자열 두 형태를 같은 숫자 리스트로 맞춘다.
    if isinstance(value, list):
        return [float(item) for item in value]
    if isinstance(value, str) and value.strip():
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [float(item) for item in parsed]
    raise ValueError("gemini_image_embedding_768 is missing or invalid.")


