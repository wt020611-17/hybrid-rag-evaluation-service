from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence

from .models import SearchHit


def reciprocal_rank_fusion(
    rankings: Sequence[Iterable[SearchHit]],
    top_k: int = 5,
    rank_constant: int = 60,
    weights: Optional[Sequence[float]] = None,
) -> List[SearchHit]:
    if rank_constant < 0:
        raise ValueError("rank_constant must be non-negative")
    if weights is not None and len(weights) != len(rankings):
        raise ValueError("weights must align with rankings")
    channel_weights = list(weights) if weights is not None else [1.0] * len(rankings)
    scores: Dict[str, float] = defaultdict(float)
    representative: Dict[str, SearchHit] = {}
    channels: Dict[str, List[str]] = defaultdict(list)

    for ranking, weight in zip(rankings, channel_weights):
        for fallback_rank, hit in enumerate(ranking, start=1):
            rank = hit.rank or fallback_rank
            scores[hit.chunk_id] += weight / (rank_constant + rank)
            representative.setdefault(hit.chunk_id, hit)
            if hit.channel not in channels[hit.chunk_id]:
                channels[hit.chunk_id].append(hit.channel)

    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    fused: List[SearchHit] = []
    for rank, chunk_id in enumerate(ordered[:top_k], start=1):
        hit = representative[chunk_id]
        metadata = dict(hit.metadata)
        metadata["channels"] = ",".join(channels[chunk_id])
        fused.append(
            SearchHit(
                chunk_id=hit.chunk_id,
                source=hit.source,
                text=hit.text,
                score=scores[chunk_id],
                channel="rrf",
                rank=rank,
                metadata=metadata,
            )
        )
    return fused

