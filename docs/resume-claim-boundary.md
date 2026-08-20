# Resume wording and interview boundary

## 推荐简历表述

**可评测混合检索 RAG 服务｜Python、FastAPI、BGE、Milvus、Neo4j、Docker**

- 独立搭建中文混合检索链路：以 `bge-small-zh-v1.5` 生成 512 维归一化向量并接入 Milvus HNSW/COSINE，结合自实现 BM25、Neo4j 一至二跳实体路径和可解释 Query Router，通过加权 RRF 返回可追溯引用。
- 整理 15 篇公开一手资料摘要并人工标注 60 条查询，构建 BM25/BGE/双路融合/图增强/路由五组消融；图增强方案在本地固定集取得 Recall@5 1.0000、MRR 0.8528，相比双路融合提升 5 个百分点和 0.0134。
- 使用 FastAPI 暴露 health/query/debug 接口，记录 trace ID、chunk 来源与通道；通过 16 项默认测试、2 项真实 Milvus/Neo4j 集成测试，并提供 Docker Compose 隔离部署和无 LLM 降级路径。

## 面试时主动说明

- 指标来自 60 条小规模人工标注查询，不是行业 benchmark；重点是评测框架、失败分析和消融方法。
- BGE 单路没有超过 BM25：Recall@5 为 0.90 对 0.95。价值在于发现互补性、修复真实主键/池化问题，并用图增强补到 1.00。
- 图数据是显式整理的 24 条关系，不是自动知识图谱抽取；Cypher 限制一至二跳并使用 namespace 隔离。
- Docker 镜像约 565MB；16GB Windows 机器要分步运行索引和 API，未做并发压测，不声称生产 SLA。
- LLM 是可选生成层；没有密钥时使用抽取式答案，不能描述成已完成生成模型微调。

## 可深入追问的真实问题

1. 为什么 BGE 使用 CLS pooling，最初的 mean pooling 如何通过消融暴露？
2. 为什么 PyMilvus 主键解析错误会让 RRF 把 15 个候选去重成 1 个？
3. 为什么图通道权重为 2.0，如何通过消融验证？
4. collection schema、HNSW 参数、namespace 删除边界和路径爆炸如何控制？
5. Recall@5、MRR、Top-1 citation accuracy 分别说明什么？
