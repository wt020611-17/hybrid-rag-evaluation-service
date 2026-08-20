# Architecture and evidence boundary

## Two reproducible paths

The repository keeps two implementations: `HybridRAGEngine` is the dependency-light BM25 + TF-IDF + in-memory graph baseline; `ProductionRAGEngine` is the BM25 + local BGE + Milvus + Neo4j + weighted RRF path used by the public-corpus ablation.

```text
ingest -> stable 360/60 chunks
       |-> BM25 ranking --------------------- 1.0x --|
       |-> BGE CLS/L2 -> Milvus HNSW ------- 1.0x --|-> RRF(k=60)
relations -> namespaced Neo4j -> 1..2 hop --- 2.0x --|
query -> rule router -> citations -> extractive/optional LLM answer
```

The graph weight prevents a valid path present in one channel from always losing to a document duplicated across BM25 and BGE. It is measured on the same 60 labels: hybrid graph reaches Recall@5 1.0000 and MRR 0.8528 versus 0.9500/0.8394 for two-channel hybrid.

## Storage isolation

- Milvus collection `hybrid_rag_public_docs_v1` has explicit chunk/document/source/text/position fields and a 512-D vector.
- Neo4j uses `RAGEntity`, uniqueness on `(namespace, name)`, and rebuild deletes only `hybrid-rag-v1` nodes.
- Compose ports are Milvus 19531, Neo4j 7475/7688 and MinIO 9010/9011; volumes are project-named.

## Safety and observability

Cypher receives parameterized seeds and a fixed `1..2` bound. API responses include trace ID, source, chunk ID, rank, channel and fusion score. `/health` checks Milvus and Neo4j connectivity/count plus embedding dimension. If the optional LLM is unavailable, citations remain and degradation is explicit.

## Evidence and limitations

Verified: local normalized 512-D BGE inference, 17 Milvus rows, 17 Neo4j nodes/24 relationships, 16 default tests, 2 real-backend tests, 60-query ablation and Docker image export.

Limitations: small manually curated corpus; labels need independent review before external benchmarking; no concurrency/load test; no-answer confidence calibration and learned routing are future work.
