"""비교 실험으로 옮긴 기존 점수·응답 형태를 검증한다."""
import pytest

from pipeline.benchmarks.legacy_vector_ranking import _rank_stored_gemini_vectors


@pytest.mark.parametrize("fabric_embedding, expected", [(None, [1.0, 0.0]), ([1.0, 0.0], [0.8, 0.2])])
def test_legacy_ranking_preserves_weights_and_removes_vectors(fabric_embedding, expected):
    rows = [
        {"id": "image", "gemini_image_embedding_768": "[1, 0]"},
        {
            "id": "fabric",
            "gemini_image_embedding_768": [0, 1],
            "gemini_fabric_text_embedding_768": [1, 0],
        },
    ]
    result = _rank_stored_gemini_vectors([1, 0], rows, 2, fabric_embedding)
    assert [row["id"] for row in result] == ["image", "fabric"]
    assert [row["similarity"] for row in result] == pytest.approx(expected)
    assert all(set(row) == {"id", "similarity"} for row in result)
    assert "gemini_image_embedding_768" in rows[0]
