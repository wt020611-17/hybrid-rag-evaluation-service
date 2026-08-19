import hashlib
import re
from typing import Iterable, List

from .models import Chunk, Document


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_document(document: Document, chunk_size: int = 360, overlap: int = 60) -> List[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    text = normalize_text(document.text)
    if not text:
        return []

    chunks: List[Chunk] = []
    start = 0
    position = 0
    while start < len(text):
        hard_end = min(start + chunk_size, len(text))
        end = hard_end
        if hard_end < len(text):
            window = text[start:hard_end]
            candidates = [window.rfind(marker) for marker in ("\n\n", "。", "！", "？", ". ")]
            boundary = max(candidates)
            if boundary >= int(chunk_size * 0.55):
                end = start + boundary + (2 if window[boundary:boundary + 2] == "\n\n" else 1)

        content = text[start:end].strip()
        if content:
            digest = hashlib.sha1(
                f"{document.document_id}:{position}:{content}".encode("utf-8")
            ).hexdigest()[:12]
            chunks.append(
                Chunk(
                    chunk_id=f"{document.document_id}:{digest}",
                    document_id=document.document_id,
                    source=document.source,
                    text=content,
                    position=position,
                    metadata=dict(document.metadata),
                )
            )
            position += 1
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_documents(
    documents: Iterable[Document], chunk_size: int = 360, overlap: int = 60
) -> List[Chunk]:
    chunks: List[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, chunk_size=chunk_size, overlap=overlap))
    return chunks

