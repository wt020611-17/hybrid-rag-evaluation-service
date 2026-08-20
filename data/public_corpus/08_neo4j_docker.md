# Neo4j Docker 配置与数据隔离

来源：Neo4j Operations Manual, Docker（https://neo4j.com/docs/operations-manual/current/docker/）。

Neo4j 官方镜像可通过 NEO4J_AUTH 设置初始账号密码，并将 /data 与 /logs 挂载到持久卷。本项目把 HTTP/Bolt 映射到 7475/7688，并为项目使用单独卷。图节点带 namespace 属性，重建操作只删除该 namespace 下的 RAGEntity，避免误删同一数据库中的其他数据。密码只放在被 Git 忽略的 .env 中。
