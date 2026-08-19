import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from .engine import HybridRAGEngine
from .evaluation import evaluate, load_cases, write_report


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hybrid RAG evaluation service")
    subparsers = parser.add_subparsers(dest="command", required=True)

    query_parser = subparsers.add_parser("query", help="query the demonstration corpus")
    query_parser.add_argument("query")
    query_parser.add_argument("--top-k", type=int, default=5)
    query_parser.add_argument("--route", choices=["keyword", "vector", "graph", "hybrid"])
    query_parser.add_argument("--debug", action="store_true")
    query_parser.add_argument("--use-llm", action="store_true")

    eval_parser = subparsers.add_parser("eval", help="run the fixed evaluation set")
    eval_parser.add_argument(
        "--dataset", type=Path, default=project_root() / "data" / "eval" / "queries.jsonl"
    )
    eval_parser.add_argument("--output", type=Path, default=project_root() / "reports" / "baseline.json")
    eval_parser.add_argument("--top-k", type=int, default=5)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    engine = HybridRAGEngine.from_project(project_root())
    if args.command == "query":
        result = engine.query(
            args.query,
            top_k=args.top_k,
            use_llm=args.use_llm,
            include_debug=args.debug,
            route=args.route,
        )
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0
    report = evaluate(engine, load_cases(args.dataset), top_k=args.top_k)
    write_report(report, args.output)
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"report={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

