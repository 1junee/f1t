# 저장 벡터와 같은 차원만 RPC 검색에 전달할 수 있는지 검사한다.
import pytest

from pipeline.vector_db_client import SupabaseVectorSearchClient


def test_rpc_for_gemini_embedding_dimension() -> None:
    assert SupabaseVectorSearchClient._rpc_for_dim(768) == "match_fashion_items_768"


def test_rpc_rejects_unsupported_embedding_dimension() -> None:
    with pytest.raises(ValueError, match="768-dimensional"):
        SupabaseVectorSearchClient._rpc_for_dim(512)
