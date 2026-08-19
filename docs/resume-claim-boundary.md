# Resume claim boundary

The current resume text promises Neo4j, BGE + Milvus, BM25 + vector + graph +
RRF, Query Router, LLM integration, FastAPI, and Docker. This repository must
not be used to support the full sentence until every external backend is run
and evaluated.

Safe v0.1.0 wording:

> 独立实现可评测的混合检索 RAG 服务，完成文档切分、BM25/TF-IDF 向量召回、受限图路径检索与 RRF 融合；设计 30 条固定查询并输出 Recall@5、MRR、引用命中率和路由准确率，同时提供 FastAPI 契约、Trace ID 与无 LLM 降级。

Only after evidence exists may TF-IDF be replaced with BGE, in-memory storage
with Milvus, and explicit relations with Neo4j in the wording.

