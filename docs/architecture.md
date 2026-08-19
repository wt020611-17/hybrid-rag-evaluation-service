# Architecture and evidence boundary

The v0.1.0 request path is:

```text
query -> QueryRouter -> BM25 / TF-IDF Vector / Explicit Graph
      -> RRF -> top-k chunks/paths -> extractive answer + citations
      -> optional OpenAI-compatible generator with retrieval-only fallback
```

Every citation carries `source`, `chunk_id`, rank, score, channel, and optional
path metadata. The evaluation uses source-level relevance labels.

## Verified versus planned

| Capability | v0.1.0 state | Evidence |
|---|---|---|
| Markdown/TXT ingestion and chunking | verified | unit tests |
| BM25 | verified | unit tests + 30-query report |
| Vector retrieval | TF-IDF baseline verified | report identifies backend |
| Graph retrieval | in-memory 1–2 hop verified | path test + report |
| RRF | verified | deterministic unit test |
| Query Router | verified rule baseline | route accuracy |
| FastAPI | implemented; requires declared dependencies | endpoint contract |
| Docker | image definition present; build must be verified separately | Dockerfile |
| BGE | planned | no completion claim |
| Milvus | planned | no completion claim |
| Neo4j | planned | no completion claim |

