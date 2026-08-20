import time
from statistics import mean
from typing import Dict, Iterable, List

from .production_engine import PRODUCTION_STRATEGIES, ProductionRAGEngine


def evaluate_ablation(
    engine: ProductionRAGEngine,
    cases: Iterable[Dict[str, object]],
    top_k: int = 5,
) -> Dict[str, object]:
    materialized = list(cases)
    reports = {}
    for strategy in PRODUCTION_STRATEGIES:
        rows: List[Dict[str, object]] = []
        for case in materialized:
            started = time.perf_counter()
            result = engine.query(str(case["query"]), top_k=top_k, strategy=strategy)
            latency_ms = (time.perf_counter() - started) * 1000.0
            relevant = {str(value) for value in case["relevant_sources"]}
            sources = [hit.source for hit in result.citations]
            first_rank = next(
                (rank for rank, source in enumerate(sources, start=1) if source in relevant),
                None,
            )
            rows.append(
                {
                    "id": case["id"],
                    "query": case["query"],
                    "expected_route": case.get("expected_route"),
                    "actual_route": result.route,
                    "sources": sources,
                    "first_relevant_rank": first_rank,
                    "reciprocal_rank": 0.0 if first_rank is None else 1.0 / first_rank,
                    "latency_ms": round(latency_ms, 3),
                }
            )
        latencies = sorted(float(row["latency_ms"]) for row in rows)
        count = len(rows)
        reports[strategy] = {
            "metrics": {
                "query_count": count,
                "recall_at_{}".format(top_k): _average(
                    row["first_relevant_rank"] is not None for row in rows
                ),
                "mrr": _average(float(row["reciprocal_rank"]) for row in rows),
                "top1_citation_accuracy": _average(
                    row["first_relevant_rank"] == 1 for row in rows
                ),
                "route_accuracy": _average(
                    _route_matches(str(row["actual_route"]), row["expected_route"])
                    for row in rows
                ) if strategy == "routed" else None,
                "mean_latency_ms": round(mean(latencies), 3) if latencies else 0.0,
                "p50_latency_ms": _percentile(latencies, 0.50),
                "p95_latency_ms": _percentile(latencies, 0.95),
            },
            "failures": [row for row in rows if row["first_relevant_rank"] is None],
            "cases": rows,
        }
    return {
        "schema_version": "1.0",
        "dataset": "curated public primary-source summaries",
        "query_count": len(materialized),
        "top_k": top_k,
        "backends": {
            "keyword": "BM25",
            "dense": "BAAI/bge-small-zh-v1.5 + Milvus HNSW/COSINE",
            "graph": "Neo4j explicit 1-2 hop paths",
            "fusion": "RRF k=60",
        },
        "strategies": reports,
        "limitations": [
            "The corpus is curated and small; results do not represent web-scale retrieval.",
            "Latency is a local sequential CPU measurement, not a concurrency load test.",
            "Labels were manually authored and should be independently reviewed before publication claims.",
        ],
    }


def _route_matches(actual: str, expected: object) -> bool:
    mapping = {
        "keyword": "bm25",
        "vector": "bge",
        "graph": "hybrid_graph",
        "hybrid": "hybrid",
    }
    return actual == mapping.get(str(expected), str(expected))


def _average(values: Iterable[object]) -> float:
    materialized = [float(value) for value in values]
    return round(sum(materialized) / len(materialized), 4) if materialized else 0.0


def _percentile(values: List[float], fraction: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, int(round((len(values) - 1) * fraction))))
    return round(values[index], 3)
