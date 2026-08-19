import json
import time
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List

from .engine import HybridRAGEngine


def load_cases(path: Path) -> List[Dict[str, object]]:
    cases: List[Dict[str, object]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            cases.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at line {line_number}") from exc
    return cases


def evaluate(
    engine: HybridRAGEngine, cases: Iterable[Dict[str, object]], top_k: int = 5
) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    for case in cases:
        started = time.perf_counter()
        result = engine.query(str(case["query"]), top_k=top_k)
        latency_ms = (time.perf_counter() - started) * 1000.0
        relevant = set(str(item) for item in case["relevant_sources"])
        sources = [hit.source for hit in result.citations]
        first_rank = next(
            (index for index, source in enumerate(sources, start=1) if source in relevant),
            None,
        )
        rows.append(
            {
                "id": case["id"],
                "query": case["query"],
                "expected_route": case.get("expected_route"),
                "actual_route": result.route,
                "sources": sources,
                "hit": first_rank is not None,
                "first_relevant_rank": first_rank,
                "reciprocal_rank": 0.0 if first_rank is None else 1.0 / first_rank,
                "top1_citation_correct": bool(sources and sources[0] in relevant),
                "latency_ms": round(latency_ms, 3),
            }
        )
    count = len(rows)
    metrics = {
        "query_count": count,
        f"recall_at_{top_k}": _average(bool(row["hit"]) for row in rows),
        "mrr": _average(float(row["reciprocal_rank"]) for row in rows),
        "top1_citation_accuracy": _average(
            bool(row["top1_citation_correct"]) for row in rows
        ),
        "route_accuracy": _average(
            row["actual_route"] == row["expected_route"] for row in rows
        ),
        "mean_latency_ms": 0.0 if not rows else round(mean(float(row["latency_ms"]) for row in rows), 3),
    }
    return {
        "schema_version": "1.0",
        "backend": {
            "keyword": "bm25",
            "vector": "tfidf",
            "graph": "in_memory_explicit_relations",
            "fusion": "rrf_k60",
        },
        "top_k": top_k,
        "metrics": metrics,
        "cases": rows,
        "limitations": [
            "The corpus and labels are synthetic demonstration data.",
            "TF-IDF is the verified offline vector baseline; BGE is not claimed in this report.",
            "Latency is a local single-process measurement, not a load test.",
        ],
    }


def write_report(report: Dict[str, object], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _average(values: Iterable[object]) -> float:
    materialized = [float(value) for value in values]
    return 0.0 if not materialized else round(sum(materialized) / len(materialized), 4)

