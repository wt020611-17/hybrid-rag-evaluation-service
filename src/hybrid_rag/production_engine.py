import re
import uuid
from typing import Dict, List, Optional, Sequence

from .bm25 import BM25Retriever
from .chunking import chunk_documents
from .corpus import load_documents
from .embeddings import BGEEmbedder
from .generator import OpenAICompatibleGenerator
from .milvus_store import MilvusBGERetriever, MilvusVectorStore
from .models import Chunk, QueryResult, SearchHit
from .neo4j_store import Neo4jGraphRetriever
from .router import QueryRouter
from .rrf import reciprocal_rank_fusion
from .settings import ProductionSettings


PRODUCTION_STRATEGIES = ("bm25", "bge", "hybrid", "hybrid_graph", "routed")


class ProductionRAGEngine:
    def __init__(
        self,
        chunks: Sequence[Chunk],
        vector: MilvusBGERetriever,
        graph: Neo4jGraphRetriever,
        generator: Optional[OpenAICompatibleGenerator] = None,
    ) -> None:
        self.chunks = list(chunks)
        self.bm25 = BM25Retriever().fit(self.chunks)
        self.vector = vector
        self.graph = graph
        self.generator = generator
        self.router = QueryRouter()

    @classmethod
    def from_settings(cls, settings: ProductionSettings) -> "ProductionRAGEngine":
        documents = load_documents(settings.corpus_dir, base_dir=settings.project_root)
        chunks = chunk_documents(documents)
        embedder = BGEEmbedder(
            model_name=settings.bge_model,
            model_path=settings.bge_model_path,
            device=settings.bge_device,
            batch_size=settings.bge_batch_size,
        )
        store = MilvusVectorStore(
            settings.milvus_uri,
            settings.milvus_collection,
            embedder.dimension,
        )
        vector = MilvusBGERetriever(embedder, store)
        graph = Neo4jGraphRetriever.from_json(
            settings.graph_path,
            settings.neo4j_uri,
            settings.neo4j_user,
            settings.neo4j_password,
            settings.neo4j_namespace,
        )
        return cls(chunks, vector, graph, OpenAICompatibleGenerator())

    @classmethod
    def sync(cls, settings: ProductionSettings) -> Dict[str, object]:
        engine = cls.from_settings(settings)
        vector_rows = engine.vector.index(engine.chunks)
        graph_counts = engine.graph.rebuild(settings.graph_path)
        return {
            "documents": len({chunk.document_id for chunk in engine.chunks}),
            "chunks": len(engine.chunks),
            "milvus_rows": vector_rows,
            "milvus_collection": settings.milvus_collection,
            "embedding_model": engine.vector.embedder.model_name,
            "embedding_dimension": engine.vector.embedder.dimension,
            "neo4j_namespace": settings.neo4j_namespace,
            "neo4j": graph_counts,
        }

    def health(self) -> Dict[str, object]:
        return {
            "milvus": self.vector.store.health(),
            "milvus_rows": self.vector.store.count(),
            "neo4j": self.graph.health(),
            "neo4j_counts": self.graph.count(),
            "embedding_model": self.vector.embedder.model_name,
            "embedding_dimension": self.vector.embedder.dimension,
        }

    def retrieve(
        self, query: str, top_k: int = 5, strategy: str = "routed"
    ) -> Dict[str, object]:
        if strategy not in PRODUCTION_STRATEGIES:
            raise ValueError("unsupported production strategy: {}".format(strategy))
        decision = self.router.route(query)
        selected = strategy
        if strategy == "routed":
            selected = {
                "keyword": "bm25",
                "vector": "bge",
                "graph": "hybrid_graph",
                "hybrid": "hybrid",
            }[decision.route]

        pool_size = max(top_k * 3, 10)
        bm25_hits = self.bm25.search(query, pool_size)
        vector_hits = self.vector.search(query, pool_size)
        graph_hits = self.graph.search(query, pool_size)
        if selected == "bm25":
            rankings = [bm25_hits]
        elif selected == "bge":
            rankings = [vector_hits]
        elif selected == "hybrid":
            rankings = [bm25_hits, vector_hits]
            weights = [1.0, 1.0]
        else:
            rankings = [bm25_hits, vector_hits]
            weights = [1.0, 1.0]
            if graph_hits:
                rankings.append(graph_hits)
                weights.append(2.0)
        if selected in ("bm25", "bge"):
            weights = [1.0]
        return {
            "strategy": strategy,
            "route": selected,
            "router_decision": decision.route,
            "route_reason": decision.reason,
            "hits": reciprocal_rank_fusion(rankings, top_k=top_k, weights=weights),
            "channel_counts": {
                "bm25": len(bm25_hits),
                "bge_milvus": len(vector_hits),
                "neo4j": len(graph_hits),
            },
        }

    def query(
        self,
        query: str,
        top_k: int = 5,
        use_llm: bool = False,
        include_debug: bool = False,
        strategy: str = "routed",
    ) -> QueryResult:
        trace_id = uuid.uuid4().hex
        if not query or not query.strip():
            return QueryResult("blocked", trace_id, query, "none", "query must not be empty", [])
        retrieval = self.retrieve(query, top_k=top_k, strategy=strategy)
        hits = list(retrieval["hits"])
        if not hits:
            return QueryResult(
                "empty",
                trace_id,
                query,
                str(retrieval["route"]),
                "未找到足够证据。",
                [],
                retrieval if include_debug else None,
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
            status,
            trace_id,
            query,
            str(retrieval["route"]),
            answer,
            hits,
            retrieval if include_debug else None,
        )

    @staticmethod
    def _extractive_answer(hits: Sequence[SearchHit]) -> str:
        sentences: List[str] = []
        for index, hit in enumerate(hits[:3], start=1):
            text = re.sub(r"\s+", " ", hit.text).strip()
            if len(text) > 220:
                text = text[:220].rstrip() + "…"
            sentences.append("[{}] {}".format(index, text))
        return "\n".join(sentences)
