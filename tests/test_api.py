from fastapi.testclient import TestClient

from hybrid_rag.api import app


client = TestClient(app)


def test_health_reports_verified_backends():
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "bm25" in payload["verified_backends"]
    assert payload["chunks"] > 0


def test_query_returns_citations_and_trace_id():
    response = client.post(
        "/query", json={"query": "RRF 的公式是什么？", "top_k": 3, "use_llm": False}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert len(payload["trace_id"]) == 32
    assert payload["citations"][0]["source"] == "data/corpus/05_rrf_fusion.md"


def test_request_validation_rejects_invalid_top_k():
    response = client.post("/query", json={"query": "test", "top_k": 0})
    assert response.status_code == 422

