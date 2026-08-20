import json
import tempfile
import unittest
from pathlib import Path

from hybrid_rag.bm25 import BM25Retriever
from hybrid_rag.chunking import chunk_document
from hybrid_rag.graph import GraphRetriever
from hybrid_rag.models import Chunk, Document, SearchHit
from hybrid_rag.router import QueryRouter
from hybrid_rag.rrf import reciprocal_rank_fusion
from hybrid_rag.tokenization import tokenize
from hybrid_rag.vector import TfidfVectorRetriever


def chunk(chunk_id: str, text: str, source: str = "source.md") -> Chunk:
    return Chunk(chunk_id, "doc", source, text, 0)


class TokenizationTests(unittest.TestCase):
    def test_mixed_tokens_include_identifier_and_chinese_bigram(self):
        tokens = tokenize("BM25 支持 trace_id 与知识检索")
        self.assertIn("bm25", tokens)
        self.assertIn("trace_id", tokens)
        self.assertIn("知识", tokens)


class ChunkingTests(unittest.TestCase):
    def test_chunks_keep_source_and_stable_ids(self):
        document = Document("doc-1", "data/example.md", "第一段内容。" * 80)
        first = chunk_document(document, chunk_size=100, overlap=20)
        second = chunk_document(document, chunk_size=100, overlap=20)
        self.assertGreater(len(first), 1)
        self.assertEqual([item.chunk_id for item in first], [item.chunk_id for item in second])
        self.assertTrue(all(item.source == "data/example.md" for item in first))

    def test_invalid_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            chunk_document(Document("d", "s", "text"), chunk_size=10, overlap=10)


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            chunk("a", "BM25 擅长编号 ERR-1042 和精确关键词", "bm25.md"),
            chunk("b", "向量检索适合含义相近的自然语言", "vector.md"),
            chunk("c", "RRF 根据多路排名进行融合", "rrf.md"),
        ]

    def test_bm25_ranks_exact_identifier(self):
        hits = BM25Retriever().fit(self.chunks).search("ERR-1042", top_k=2)
        self.assertEqual(hits[0].source, "bm25.md")

    def test_tfidf_vector_ranks_related_text(self):
        hits = TfidfVectorRetriever().fit(self.chunks).search("多路排名融合", top_k=2)
        self.assertEqual(hits[0].source, "rrf.md")

    def test_rrf_rewards_multi_channel_hit(self):
        a1 = SearchHit("a", "a.md", "a", 10, "bm25", rank=1)
        b1 = SearchHit("b", "b.md", "b", 9, "bm25", rank=2)
        a2 = SearchHit("a", "a.md", "a", 0.8, "vector", rank=2)
        b2 = SearchHit("b", "b.md", "b", 0.9, "vector", rank=1)
        fused = reciprocal_rank_fusion([[a1, b1], [a2, b2]], top_k=2)
        self.assertEqual({hit.chunk_id for hit in fused}, {"a", "b"})
        self.assertEqual(fused[0].score, fused[1].score)

    def test_rrf_accepts_explicit_channel_weights(self):
        lexical = SearchHit("a", "a.md", "a", 1, "bm25", rank=1)
        graph = SearchHit("b", "b.md", "b", 1, "neo4j", rank=1)
        fused = reciprocal_rank_fusion(
            [[lexical], [graph]], top_k=2, weights=[1.0, 2.0]
        )
        self.assertEqual(fused[0].chunk_id, "b")


class GraphTests(unittest.TestCase):
    def test_two_hop_path_is_returned(self):
        graph = GraphRetriever(
            [
                {"source": "Router", "relation": "选择", "target": "Retriever", "source_document": "router.md"},
                {"source": "Retriever", "relation": "包含", "target": "BM25", "source_document": "bm25.md"},
            ],
            max_depth=2,
        )
        hits = graph.search("Router 与 BM25 的多跳关系", top_k=5)
        self.assertTrue(any("Router --选择--> Retriever；Retriever --包含--> BM25" in hit.text for hit in hits))


class RouterTests(unittest.TestCase):
    def test_route_rules(self):
        router = QueryRouter()
        self.assertEqual(router.route("RRF 的公式").route, "keyword")
        self.assertEqual(router.route("为什么需要切分？").route, "vector")
        self.assertEqual(router.route("RRF 和 BM25 的关系").route, "graph")
        self.assertEqual(router.route("如何选择切分参数").route, "hybrid")


if __name__ == "__main__":
    unittest.main()

