import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ProductionSettings:
    project_root: Path
    corpus_dir: Path
    eval_path: Path
    graph_path: Path
    bge_model: str
    bge_model_path: Optional[str]
    bge_device: str
    bge_batch_size: int
    milvus_uri: str
    milvus_collection: str
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    neo4j_namespace: str

    @classmethod
    def from_env(cls, project_root: Path) -> "ProductionSettings":
        project_root = Path(project_root).resolve()
        _load_project_env(project_root)
        password = os.getenv("NEO4J_PASSWORD", "")
        if not password or password.startswith("replace_"):
            raise ValueError("NEO4J_PASSWORD must be configured for production mode")
        return cls(
            project_root=project_root,
            corpus_dir=_resolve(project_root, os.getenv("PUBLIC_CORPUS_DIR", "data/public_corpus")),
            eval_path=_resolve(project_root, os.getenv("PUBLIC_EVAL_PATH", "data/eval/public_queries.jsonl")),
            graph_path=_resolve(project_root, os.getenv("PUBLIC_GRAPH_PATH", "data/graph/public_relations.json")),
            bge_model=os.getenv("BGE_MODEL", "BAAI/bge-small-zh-v1.5"),
            bge_model_path=os.getenv("BGE_MODEL_PATH") or None,
            bge_device=os.getenv("BGE_DEVICE", "cpu"),
            bge_batch_size=int(os.getenv("BGE_BATCH_SIZE", "16")),
            milvus_uri=os.getenv("MILVUS_URI", "http://127.0.0.1:19531"),
            milvus_collection=os.getenv("MILVUS_COLLECTION", "hybrid_rag_public_docs_v1"),
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://127.0.0.1:7688"),
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_password=password,
            neo4j_namespace=os.getenv("NEO4J_NAMESPACE", "hybrid-rag-v1"),
        )


def _resolve(root: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def _load_project_env(project_root: Path) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(project_root / ".env", override=False)

