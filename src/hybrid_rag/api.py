import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .engine import HybridRAGEngine
from .production_engine import PRODUCTION_STRATEGIES, ProductionRAGEngine
from .settings import ProductionSettings


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    use_llm: bool = False
    route: Optional[str] = Field(default=None, pattern="^(keyword|vector|graph|hybrid)$")
    strategy: Optional[str] = Field(
        default=None, pattern="^(bm25|bge|hybrid|hybrid_graph|routed)$"
    )


@lru_cache(maxsize=1)
def get_engine() -> object:
    root = Path(os.getenv("RAG_PROJECT_ROOT", str(Path.cwd()))).resolve()
    if os.getenv("RAG_MODE", "baseline").lower() == "production":
        return ProductionRAGEngine.from_settings(ProductionSettings.from_env(root))
    return HybridRAGEngine.from_project(root)


def create_app() -> FastAPI:
    application = FastAPI(
        title="Hybrid RAG Evaluation Service",
        version="1.0.0",
        description="Hybrid RAG with reproducible baseline and production backends.",
    )

    @application.get("/health")
    def health() -> Dict[str, Any]:
        engine = get_engine()
        if isinstance(engine, ProductionRAGEngine):
            backend_health = engine.health()
            return {
                "status": "ok",
                "mode": "production",
                "version": "1.0.0",
                "chunks": len(engine.chunks),
                "llm_configured": bool(engine.generator and engine.generator.configured),
                "verified_backends": ["bm25", "bge", "milvus", "neo4j", "rrf"],
                "backend_health": backend_health,
            }
        return {
            "status": "ok",
            "mode": "baseline",
            "version": "1.0.0",
            "chunks": len(engine.chunks),
            "llm_configured": bool(engine.generator and engine.generator.configured),
            "verified_backends": ["bm25", "tfidf", "in_memory_graph", "rrf"],
        }

    @application.post("/query")
    def query(request: QueryRequest) -> Dict[str, Any]:
        engine = get_engine()
        if isinstance(engine, ProductionRAGEngine):
            strategy = request.strategy or _route_to_strategy(request.route)
            result = engine.query(
                request.query,
                top_k=request.top_k,
                use_llm=request.use_llm,
                strategy=strategy,
            )
        else:
            result = engine.query(
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
        engine = get_engine()
        if isinstance(engine, ProductionRAGEngine):
            result = engine.query(
                request.query,
                top_k=request.top_k,
                use_llm=False,
                include_debug=True,
                strategy=request.strategy or _route_to_strategy(request.route),
            )
        else:
            result = engine.query(
                request.query,
                top_k=request.top_k,
                use_llm=False,
                include_debug=True,
                route=request.route,
            )
        return result.to_dict()

    return application


app = create_app()


def _route_to_strategy(route: Optional[str]) -> str:
    return {
        None: "routed",
        "keyword": "bm25",
        "vector": "bge",
        "graph": "hybrid_graph",
        "hybrid": "hybrid",
    }[route]

