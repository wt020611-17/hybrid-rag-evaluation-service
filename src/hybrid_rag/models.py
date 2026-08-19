from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Document:
    document_id: str
    source: str
    text: str
    title: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    source: str
    text: str
    position: int
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    source: str
    text: str
    score: float
    channel: str
    rank: int = 0
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class QueryResult:
    status: str
    trace_id: str
    query: str
    route: str
    answer: str
    citations: List[SearchHit]
    debug: Optional[Dict[str, object]] = None

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        if self.debug is None:
            payload.pop("debug", None)
        return payload

