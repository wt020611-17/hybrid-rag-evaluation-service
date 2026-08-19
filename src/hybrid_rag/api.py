from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .engine import HybridRAGEngine


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    use_llm: bool = False
    route: Optional[str] = Field(default=None, pattern="^(keyword|vector|graph|hybrid)$")


@lru_cache(maxsize=1)
def get_engine() -> HybridRAGEngine:
    root = Path(__file__).resolve().parents[2]
    return HybridRAGEngine.from_project(root)


def create_app() -> FastAPI:
    application = FastAPI(
        title="Hybrid RAG Evaluation Service",
        version="0.1.0",
        description="Evidence-first hybrid retrieval, citations, and evaluation.",
    )

    @application.get("/health")
    def health() -> Dict[str, Any]:
        engine = get_engine()
        return {
            "status": "ok",
            "version": "0.1.0",
            "chunks": len(engine.chunks),
            "llm_configured": bool(engine.generator and engine.generator.configured),
            "verified_backends": ["bm25", "tfidf", "in_memory_graph", "rrf"],
        }

    @application.post("/query")
    def query(request: QueryRequest) -> Dict[str, Any]:
        result = get_engine().query(
            request.query,
            top_k=request.top_k,
            use_llm=request.use_llm,
            route=request.route,
        )
        if result.status == "blocked":
            raise HTTPException(status_code=422, detail=result.to_dict())
        return result.to_dict()

    @application.post("/debug/retrieval")
    def debug_retrieval(request: QueryRequest) -> Dict[str, Any]:
        result = get_engine().query(
            request.query,
            top_k=request.top_k,
            use_llm=False,
            include_debug=True,
            route=request.route,
        )
        return result.to_dict()

    return application


app = create_app()

