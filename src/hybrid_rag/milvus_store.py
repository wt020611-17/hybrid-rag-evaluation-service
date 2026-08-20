from typing import Dict, List, Sequence

from .models import Chunk, SearchHit


class MilvusVectorStore:
    def __init__(self, uri: str, collection_name: str, dimension: int) -> None:
        try:
            from pymilvus import DataType, MilvusClient
        except ImportError as exc:
            raise RuntimeError("PyMilvus is required for the production vector store") from exc
        self._DataType = DataType
        self._MilvusClient = MilvusClient
        self.client = MilvusClient(uri=uri)
        self.collection_name = collection_name
        self.dimension = dimension

    def health(self) -> bool:
        self.client.list_collections()
        return True

    def rebuild(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunk/vector counts do not match")
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)

        schema = self._MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(
            field_name="chunk_id",
            datatype=self._DataType.VARCHAR,
            is_primary=True,
            max_length=128,
        )
        schema.add_field(field_name="document_id", datatype=self._DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="source", datatype=self._DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="text", datatype=self._DataType.VARCHAR, max_length=4096)
        schema.add_field(field_name="position", datatype=self._DataType.INT64)
        schema.add_field(
            field_name="embedding",
            datatype=self._DataType.FLOAT_VECTOR,
            dim=self.dimension,
        )
        indexes = self._MilvusClient.prepare_index_params()
        indexes.add_index(
            field_name="embedding",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 128},
        )
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=indexes,
            consistency_level="Strong",
        )
        rows = [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "source": chunk.source,
                "text": chunk.text,
                "position": chunk.position,
                "embedding": list(vector),
            }
            for chunk, vector in zip(chunks, vectors)
        ]
        if rows:
            self.client.insert(collection_name=self.collection_name, data=rows)
            self.client.flush(collection_name=self.collection_name)
        self.client.load_collection(collection_name=self.collection_name)
        return len(rows)

    def search(self, vector: Sequence[float], top_k: int = 5) -> List[SearchHit]:
        if top_k <= 0:
            return []
        results = self.client.search(
            collection_name=self.collection_name,
            data=[list(vector)],
            anns_field="embedding",
            limit=top_k,
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
            output_fields=["chunk_id", "document_id", "source", "text", "position"],
        )
        hits: List[SearchHit] = []
        for rank, item in enumerate(results[0] if results else [], start=1):
            entity = item.get("entity", {})
            chunk_id = entity.get("chunk_id") or item.get("id")
            if not chunk_id:
                raise RuntimeError("Milvus search result is missing its primary chunk_id")
            hits.append(
                SearchHit(
                    chunk_id=str(chunk_id),
                    source=str(entity.get("source", "")),
                    text=str(entity.get("text", "")),
                    score=float(item.get("distance", 0.0)),
                    channel="bge_milvus",
                    rank=rank,
                    metadata={
                        "document_id": str(entity.get("document_id", "")),
                        "position": str(entity.get("position", "")),
                    },
                )
            )
        return hits

    def count(self) -> int:
        rows = self.client.query(
            collection_name=self.collection_name,
            filter="",
            output_fields=["count(*)"],
        )
        return int(rows[0]["count(*)"]) if rows else 0

    def delete_source(self, source: str) -> int:
        escaped = source.replace("\\", "\\\\").replace('"', '\\"')
        result = self.client.delete(
            collection_name=self.collection_name,
            filter='source == "{}"'.format(escaped),
        )
        return int(result.get("delete_count", 0))


class MilvusBGERetriever:
    def __init__(self, embedder: object, store: MilvusVectorStore) -> None:
        self.embedder = embedder
        self.store = store
        self._query_cache: Dict[str, List[float]] = {}

    def index(self, chunks: Sequence[Chunk]) -> int:
        vectors = self.embedder.embed_documents(chunk.text for chunk in chunks)
        return self.store.rebuild(chunks, vectors)

    def search(self, query: str, top_k: int = 5) -> List[SearchHit]:
        if query not in self._query_cache:
            self._query_cache[query] = self.embedder.embed_query(query)
        return self.store.search(self._query_cache[query], top_k=top_k)
