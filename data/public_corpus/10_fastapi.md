# FastAPI 请求校验与服务接口

来源：FastAPI 官方教程（https://fastapi.tiangolo.com/tutorial/body/）。

FastAPI 使用 Pydantic 模型声明请求体并自动执行类型和范围校验。本项目的 POST /query 接受 query、top_k、strategy 和 use_llm，top_k 限制在 1 到 20；POST /debug/retrieval 返回路由原因和各通道候选数量；GET /health 检查实际 Milvus、Neo4j、模型维度和索引数量。每次查询生成 trace_id，响应引用保留 source、chunk_id、通道与排名，便于复盘。
