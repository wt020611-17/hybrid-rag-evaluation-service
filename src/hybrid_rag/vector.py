import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple

from .models import Chunk, SearchHit
from .tokenization import tokenize


SparseVector = Dict[str, float]


class TfidfVectorRetriever:
    """Deterministic vector baseline used for offline verification."""

    def __init__(self) -> None:
        self._chunks: List[Chunk] = []
        self._idf: Dict[str, float] = {}
        self._vectors: List[SparseVector] = []

    def fit(self, chunks: Iterable[Chunk]) -> "TfidfVectorRetriever":
        self._chunks = list(chunks)
        tokenized = [tokenize(chunk.text) for chunk in self._chunks]
        frequency: Dict[str, int] = defaultdict(int)
        for tokens in tokenized:
            for token in set(tokens):
                frequency[token] += 1
        count = len(tokenized)
        self._idf = {
            token: math.log((count + 1.0) / (document_count + 1.0)) + 1.0
            for token, document_count in frequency.items()
        }
        self._vectors = [self._transform(tokens) for tokens in tokenized]
        return self

    def _transform(self, tokens: List[str]) -> SparseVector:
        counts = Counter(tokens)
        vector = {
            token: count * self._idf[token]
            for token, count in counts.items()
            if token in self._idf
        }
        norm = math.sqrt(sum(value * value for value in vector.values()))
        if norm == 0:
            return {}
        return {token: value / norm for token, value in vector.items()}

    def search(self, query: str, top_k: int = 5) -> List[SearchHit]:
        if not self._chunks or top_k <= 0:
            return []
        query_vector = self._transform(tokenize(query))
        scores: List[Tuple[int, float]] = []
        for index, vector in enumerate(self._vectors):
            score = sum(value * vector.get(token, 0.0) for token, value in query_vector.items())
            if score > 0:
                scores.append((index, score))
        scores.sort(key=lambda item: (-item[1], self._chunks[item[0]].chunk_id))
        hits: List[SearchHit] = []
        for rank, (index, score) in enumerate(scores[:top_k], start=1):
            chunk = self._chunks[index]
            hits.append(
                SearchHit(
                    chunk_id=chunk.chunk_id,
                    source=chunk.source,
                    text=chunk.text,
                    score=score,
                    channel="vector",
                    rank=rank,
                )
            )
        return hits

