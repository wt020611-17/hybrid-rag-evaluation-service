# FastAPI 服务可靠性

RAG API 的请求模型应限制 query 长度和 top_k 范围。响应除 answer 外，还要返回 status、trace_id、route 和 citations，使调用方能区分成功、空结果、输入阻断、降级和内部错误。

/health 用于验证进程和已加载索引状态；/query 面向正常问答；/debug/retrieval 返回路由原因和各通道信息，只应在受控环境开放。稳定错误模型比直接把 Python 异常文本暴露给用户更安全。

外部模型调用需要超时、有限重试与降级。LLM 不可用时，系统仍可返回检索证据。Dockerfile 负责封装应用依赖；数据库容器与应用容器是否都已运行，必须分别验证，不能把只有数据库 Compose 描述成应用已部署。

