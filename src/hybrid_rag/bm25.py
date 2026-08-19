import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple

from .models import Chunk, SearchHit
from .tokenization import tokenize


class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: List[Chunk] = []
        self._term_frequencies: List[Counter] = []
        self._document_frequency: Dict[str, int] = {}
        self._lengths: List[int] = []
        self._average_length = 0.0

    def fit(self, chunks: Iterable[Chunk]) -> "BM25Retriever":
        self._chunks = list(chunks)
        self._term_frequencies = []
        self._lengths = []
        document_frequency: Dict[str, int] = defaultdict(int)
        for chunk in self._chunks:
            counter = Counter(tokenize(chunk.text))
            self._term_frequencies.append(counter)
            self._lengths.append(sum(counter.values()))
            for term in counter:
                document_frequency[term] += 1
        self._document_frequency = dict(document_frequency)
        self._average_length = (
            sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        )
        return self

    def search(self, query: str, top_k: int = 5) -> List[SearchHit]:
        if not self._chunks or top_k <= 0:
            return []
        query_terms = tokenize(query)
        scores: List[Tuple[int, float]] = []
        count = len(self._chunks)
        for index, frequencies in enumerate(self._term_frequencies):
            score = 0.0
            length = self._lengths[index]
            for term in query_terms:
                term_frequency = frequencies.get(term, 0)
                if term_frequency == 0:
                    continue
                frequency = self._document_frequency.get(term, 0)
                inverse_document_frequency = math.log(
                    1.0 + (count - frequency + 0.5) / (frequency + 0.5)
                )
                normalizer = term_frequency + self.k1 * (
                    1.0
                    - self.b
                    + self.b * length / max(self._average_length, 1.0)
                )
                score += inverse_document_frequency * (
                    term_frequency * (self.k1 + 1.0) / normalizer
                )
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
                    channel="bm25",
                    rank=rank,
                )
            )
        return hits

