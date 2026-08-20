import hashlib
import json
from pathlib import Path
from typing import Dict, List

from .models import SearchHit


class Neo4jGraphRetriever:
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        namespace: str,
        aliases: Dict[str, str],
    ) -> None:
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise RuntimeError("Neo4j driver is required for production graph retrieval") from exc
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.namespace = namespace
        self.aliases = {key.lower(): value for key, value in aliases.items()}

    @classmethod
    def from_json(
        cls, path: Path, uri: str, user: str, password: str, namespace: str
    ) -> "Neo4jGraphRetriever":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        aliases: Dict[str, str] = {}
        for canonical, values in payload.get("aliases", {}).items():
            aliases[canonical.lower()] = canonical
            for value in values:
                aliases[str(value).lower()] = canonical
        for edge in payload["edges"]:
            aliases.setdefault(edge["source"].lower(), edge["source"])
            aliases.setdefault(edge["target"].lower(), edge["target"])
        return cls(uri, user, password, namespace, aliases)

    def health(self) -> bool:
        self.driver.verify_connectivity()
        return True

    def close(self) -> None:
        self.driver.close()

    def rebuild(self, path: Path) -> Dict[str, int]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        edges = payload["edges"]
        with self.driver.session() as session:
            session.run(
                "CREATE CONSTRAINT rag_entity_identity IF NOT EXISTS "
                "FOR (n:RAGEntity) REQUIRE (n.namespace, n.name) IS UNIQUE"
            ).consume()
            session.run(
                "MATCH (n:RAGEntity {namespace: $namespace}) DETACH DELETE n",
                namespace=self.namespace,
            ).consume()
            session.run(
                "UNWIND $edges AS edge "
                "MERGE (source:RAGEntity {namespace: $namespace, name: edge.source}) "
                "MERGE (target:RAGEntity {namespace: $namespace, name: edge.target}) "
                "MERGE (source)-[relation:RELATED {namespace: $namespace, kind: edge.relation, "
                "source_document: edge.source_document}]->(target)",
                namespace=self.namespace,
                edges=edges,
            ).consume()
        return self.count()

    def count(self) -> Dict[str, int]:
        with self.driver.session() as session:
            record = session.run(
                "MATCH (n:RAGEntity {namespace: $namespace}) "
                "OPTIONAL MATCH (n)-[r:RELATED {namespace: $namespace}]->() "
                "RETURN count(DISTINCT n) AS nodes, count(DISTINCT r) AS relationships",
                namespace=self.namespace,
            ).single()
        return {"nodes": int(record["nodes"]), "relationships": int(record["relationships"])}

    def matched_entities(self, query: str) -> List[str]:
        lowered = query.lower()
        matches = {canonical for alias, canonical in self.aliases.items() if alias in lowered}
        return sorted(matches, key=lambda value: (-len(value), value))

    def search(self, query: str, top_k: int = 5) -> List[SearchHit]:
        seeds = self.matched_entities(query)
        if not seeds or top_k <= 0:
            return []
        cypher = (
            "MATCH (seed:RAGEntity {namespace: $namespace}) WHERE seed.name IN $seeds "
            "MATCH path=(seed)-[:RELATED*1..2]-(other:RAGEntity {namespace: $namespace}) "
            "WITH path, relationships(path) AS rels, nodes(path) AS nodes "
            "RETURN [node IN nodes | node.name] AS names, "
            "[rel IN rels | rel.kind] AS kinds, "
            "[rel IN rels | rel.source_document] AS sources, length(path) AS depth "
            "ORDER BY depth ASC, names ASC LIMIT $limit"
        )
        with self.driver.session() as session:
            rows = list(
                session.run(
                    cypher,
                    namespace=self.namespace,
                    seeds=seeds,
                    limit=max(top_k * 3, 10),
                )
            )
        hits: List[SearchHit] = []
        seen = set()
        for row in rows:
            names = list(row["names"])
            kinds = list(row["kinds"])
            depth = int(row["depth"])
            text = "；".join(
                "{} --{}--> {}".format(names[index], kinds[index], names[index + 1])
                for index in range(len(kinds))
            )
            if text in seen:
                continue
            seen.add(text)
            digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
            sources = [value for value in row["sources"] if value]
            hits.append(
                SearchHit(
                    chunk_id="neo4j:{}".format(digest),
                    source=sources[-1] if sources else "data/graph/public_relations.json",
                    text=text,
                    score=1.0 / depth,
                    channel="neo4j",
                    rank=len(hits) + 1,
                    metadata={"depth": str(depth), "seed": ",".join(seeds)},
                )
            )
            if len(hits) >= top_k:
                break
        return hits
