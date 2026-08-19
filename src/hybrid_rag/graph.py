import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .models import SearchHit


class GraphRetriever:
    def __init__(self, edges: Iterable[Dict[str, str]], max_depth: int = 2) -> None:
        self.max_depth = max_depth
        self._edges = list(edges)
        self._adjacency: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        self._display_names: Dict[str, str] = {}
        for edge in self._edges:
            source_key = edge["source"].lower()
            target_key = edge["target"].lower()
            self._display_names[source_key] = edge["source"]
            self._display_names[target_key] = edge["target"]
            self._adjacency[source_key].append(edge)
            reverse = {
                "source": edge["target"],
                "relation": f"被{edge['relation']}",
                "target": edge["source"],
                "source_document": edge.get("source_document", "data/graph/relations.json"),
            }
            self._adjacency[target_key].append(reverse)

    @classmethod
    def from_json(cls, path: Path, max_depth: int = 2) -> "GraphRetriever":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        edges = payload["edges"] if isinstance(payload, dict) else payload
        return cls(edges=edges, max_depth=max_depth)

    def matched_entities(self, query: str) -> List[str]:
        lowered = query.lower()
        matches = [entity for entity in self._display_names if entity in lowered]
        return sorted(matches, key=lambda value: (-len(value), value))

    def search(self, query: str, top_k: int = 5) -> List[SearchHit]:
        seeds = self.matched_entities(query)
        if not seeds:
            return []
        hits: List[SearchHit] = []
        seen_edges: Set[Tuple[str, str, str]] = set()
        queue = deque((seed, 0, []) for seed in seeds)
        visited: Set[Tuple[str, int]] = set()
        while queue and len(hits) < top_k * 3:
            entity, depth, path = queue.popleft()
            if (entity, depth) in visited or depth >= self.max_depth:
                continue
            visited.add((entity, depth))
            for edge in self._adjacency.get(entity, []):
                key = (edge["source"], edge["relation"], edge["target"])
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                current_path = path + [key]
                text = "；".join(
                    f"{source} --{relation}--> {target}"
                    for source, relation, target in current_path
                )
                digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
                hits.append(
                    SearchHit(
                        chunk_id=f"graph:{digest}",
                        source=edge.get("source_document", "data/graph/relations.json"),
                        text=text,
                        score=1.0 / (depth + 1),
                        channel="graph",
                        rank=0,
                        metadata={"depth": str(depth + 1)},
                    )
                )
                queue.append((edge["target"].lower(), depth + 1, current_path))
        hits.sort(key=lambda hit: (-hit.score, hit.chunk_id))
        return [
            SearchHit(
                chunk_id=hit.chunk_id,
                source=hit.source,
                text=hit.text,
                score=hit.score,
                channel=hit.channel,
                rank=rank,
                metadata=hit.metadata,
            )
            for rank, hit in enumerate(hits[:top_k], start=1)
        ]

