import hashlib
from pathlib import Path
from typing import Iterable, List, Optional

from .models import Document


SUPPORTED_SUFFIXES = {".md", ".txt"}


def load_documents(corpus_dir: Path, base_dir: Optional[Path] = None) -> List[Document]:
    corpus_dir = Path(corpus_dir).resolve()
    base_dir = Path(base_dir).resolve() if base_dir else corpus_dir
    documents: List[Document] = []
    for path in sorted(corpus_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8")
        source = path.relative_to(base_dir).as_posix()
        digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:10]
        title = _extract_title(text) or path.stem
        documents.append(
            Document(document_id=f"doc-{digest}", source=source, text=text, title=title)
        )
    return documents


def _extract_title(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return ""


def iter_text_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        path = Path(path)
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path
        elif path.is_dir():
            for candidate in path.rglob("*"):
                if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_SUFFIXES:
                    yield candidate

