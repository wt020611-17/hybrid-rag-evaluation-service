import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .bm25 import BM25Retriever
from .chunking import chunk_documents
from .corpus import load_documents
from .generator import OpenAICompatibleGenerator
from .graph import GraphRetriever
from .models import Chunk, QueryResult, SearchHit
from .router import QueryRouter
from .rrf import reciprocal_rank_fusion
from .vector import TfidfVectorRetriever


class HybridRAGEngine:
    def __init__(
        self,
        chunks: Sequence[Chunk],
        graph: Optional[GraphRetriever] = None,
        generator: Optional[OpenAICompatibleGenerator] = None,
    ) -> None:
        self.chunks = list(chunks)
        self.bm25 = BM25Retriever().fit(self.chunks)
        self.vector = TfidfVectorRetriever().fit(self.chunks)
        self.graph = graph
        self.generator = generator
        self.router = QueryRouter()

    @classmethod
    def from_project(cls, project_root: Path) -> "HybridRAGEngine":
        project_root = Path(project_root).resolve()
        documents = load_documents(project_root / "data" / "corpus", base_dir=project_root)
        chunks = chunk_documents(documents)
        graph_path = project_root / "data" / "graph" / "relations.json"
        graph = GraphRetriever.from_json(graph_path) if graph_path.exists() else None
        generator = OpenAICompatibleGenerator()
        return cls(chunks=chunks, graph=graph, generator=generator)

    def retrieve(self, query: str, top_k: int = 5, route: Optional[str] = None) -> Dict[str, object]:
        decision = self.router.route(query)
        selected = route or decision.route
        pool_size = max(top_k * 3, 10)
        rankings: List[List[SearchHit]] = []

        bm25_hits = self.bm25.search(query, pool_size)
        vector_hits = self.vector.search(query, pool_size)
        graph_hits = self.graph.search(query, pool_size) if self.graph else []

        if selected == "keyword":
            rankings = [bm25_hits]
        elif selected == "vector":
            rankings = [vector_hits]
        elif selected == "graph":
            rankings = [graph_hits, bm25_hits]
        elif selected == "hybrid":
            rankings = [bm25_hits, vector_hits]
            if self.graph and self.graph.matched_entities(query):
                rankings.append(graph_hits)
        else:
            raise ValueError(f"unsupported route: {selected}")

        hits = reciprocal_rank_fusion(rankings, top_k=top_k)
        return {
            "route": selected,
            "route_reason": decision.reason if route is None else "route explicitly requested",
            "hits": hits,
            "channel_counts": {
                "bm25": len(bm25_hits),
                "vector": len(vector_hits),
                "graph": len(graph_hits),
            },
        }

    def query(
        self,
        query: str,
        top_k: int = 5,
        use_llm: bool = False,
        include_debug: bool = False,
        route: Optional[str] = None,
    ) -> QueryResult:
        trace_id = uuid.uuid4().hex
        if not query or not query.strip():
            return QueryResult(
                status="blocked",
                trace_id=trace_id,
                query=query,
                route="none",
                answer="query must not be empty",
                citations=[],
            )
        retrieval = self.retrieve(query=query, top_k=top_k, route=route)
        hits = retrieval["hits"]
        if not hits:
            return QueryResult(
                status="empty",
                trace_id=trace_id,
                query=query,
                route=str(retrieval["route"]),
                answer="未找到足够证据。",
                citations=[],
                debug=retrieval if include_debug else None,
            )

        answer = self._extractive_answer(hits)
        status = "ok"
        if use_llm:
            if self.generator and self.generator.configured:
                try:
                    answer = self.generator.generate(query, hits)
                except RuntimeError:
                    status = "degraded"
            else:
                status = "degraded"
        return QueryResult(
            status=status,
            trace_id=trace_id,
            query=query,
            route=str(retrieval["route"]),
            answer=answer,
            citations=list(hits),
            debug=retrieval if include_debug else None,
        )

    @staticmethod
    def _extractive_answer(hits: Sequence[SearchHit]) -> str:
        sentences: List[str] = []
        for index, hit in enumerate(hits[:3], start=1):
            text = re.sub(r"\s+", " ", hit.text).strip()
            if len(text) > 220:
                text = text[:220].rstrip() + "…"
            sentences.append(f"[{index}] {text}")
        return "\n".join(sentences)
