# Milvus Standalone 的 Docker Compose 依赖

来源：Milvus 官方 Docker Compose 安装文档（https://milvus.io/docs/install_standalone-docker-compose.md）。

Milvus standalone 的 Compose 部署通常包含 Milvus、etcd 和 MinIO：etcd 保存元数据，MinIO 保存对象数据，Milvus 提供向量写入和搜索服务。服务健康检查与 depends_on 的健康条件可以减少启动竞态。本项目使用独立端口 19531 和独立命名卷，避免碰触机器上其他 Milvus 实例；API 只有在索引任务成功完成后才启动。
