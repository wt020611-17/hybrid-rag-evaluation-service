# Docker Compose 启动顺序与健康检查

来源：Docker Docs, startup order（https://docs.docker.com/compose/how-tos/startup-order/）。

Compose 的 depends_on 可以结合 service_healthy 和 service_completed_successfully 表达依赖条件。本项目先等待 etcd、MinIO、Milvus、Neo4j 健康，再运行一次性 indexer；只有 indexer 成功，API 才启动。健康检查不能代替业务级验证，因此验收还会实际插入向量、执行查询、检查图节点关系并调用 HTTP 接口。
