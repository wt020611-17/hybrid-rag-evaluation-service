import unittest
from pathlib import Path

from hybrid_rag.engine import HybridRAGEngine
from hybrid_rag.evaluation import evaluate, load_cases


ROOT = Path(__file__).resolve().parents[1]


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = HybridRAGEngine.from_project(ROOT)

    def test_query_returns_trace_and_citations(self):
        result = self.engine.query("RRF 的公式是什么？", top_k=3)
        self.assertEqual(result.status, "ok")
        self.assertEqual(len(result.trace_id), 32)
        self.assertTrue(result.citations)
        self.assertEqual(result.citations[0].source, "data/corpus/05_rrf_fusion.md")

    def test_empty_query_is_blocked(self):
        result = self.engine.query("   ")
        self.assertEqual(result.status, "blocked")
        self.assertFalse(result.citations)

    def test_unconfigured_llm_degrades_to_retrieval(self):
        result = self.engine.query("RAG 是什么？", use_llm=True)
        self.assertEqual(result.status, "degraded")
        self.assertTrue(result.citations)

    def test_fixed_evaluation_set_has_expected_size_and_quality(self):
        cases = load_cases(ROOT / "data" / "eval" / "queries.jsonl")
        report = evaluate(self.engine, cases, top_k=5)
        self.assertEqual(report["metrics"]["query_count"], 30)
        self.assertGreaterEqual(report["metrics"]["recall_at_5"], 0.85)
        self.assertGreaterEqual(report["metrics"]["route_accuracy"], 0.95)


if __name__ == "__main__":
    unittest.main()

