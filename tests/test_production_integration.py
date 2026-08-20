import os
from pathlib import Path

import pytest

from hybrid_rag.production_engine import ProductionRAGEngine
from hybrid_rag.settings import ProductionSettings


pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def production_engine():
    if os.getenv("RUN_PRODUCTION_INTEGRATION") != "1":
        pytest.skip("set RUN_PRODUCTION_INTEGRATION=1 to use local services")
    root = Path(__file__).resolve().parents[1]
    engine = ProductionRAGEngine.from_settings(ProductionSettings.from_env(root))
    yield engine
    engine.graph.close()


def test_real_backends_are_populated(production_engine):
    health = production_engine.health()
    assert health["embedding_model"] == "BAAI/bge-small-zh-v1.5"
    assert health["embedding_dimension"] == 512
    assert health["milvus"] is True
    assert health["milvus_rows"] == len(production_engine.chunks)
    assert health["neo4j"] is True
    assert health["neo4j_counts"]["relationships"] >= 20


def test_real_hybrid_graph_query_returns_distinct_citations(production_engine):
    result = production_engine.query(
        "BGE 与 Milvus 和 RRF 的多跳关系是什么？",
        top_k=5,
        strategy="hybrid_graph",
        include_debug=True,
    )
    assert result.status == "ok"
    assert len(result.citations) == 5
    assert len({hit.chunk_id for hit in result.citations}) == 5
    assert result.debug["channel_counts"]["bge_milvus"] >= 5
    assert result.debug["channel_counts"]["neo4j"] >= 1
    assert any(hit.channel == "rrf" for hit in result.citations)
