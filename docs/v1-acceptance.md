# v1.0 completion gate

The project is complete only when all of the following are evidenced on the
same commit:

- [x] BAAI/bge-small-zh-v1.5 runs locally on CPU and produces normalized 512-D vectors.
- [x] A dedicated Milvus collection is created, populated and searched.
- [x] A dedicated Neo4j namespace answers constrained one/two-hop queries.
- [x] The corpus contains curated summaries backed by public primary sources.
- [x] 60 labelled queries cover exact, semantic, graph and hybrid retrieval.
- [x] Five strategies share one reproducible ablation report.
- [x] API health/query/debug code reports real backends, trace IDs and fallback behavior.
- [x] Unit, API, real-backend integration and evaluation tests pass.
- [x] A production Docker image was exported; Compose documents low-memory staged startup.
- [x] README and resume wording distinguish measured facts from limitations.

Deferred rather than claimed: no-answer confidence calibration, learned routing, concurrent load testing and production SLA.
