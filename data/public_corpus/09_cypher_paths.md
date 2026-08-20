# Cypher 一至二跳路径检索

来源：Neo4j Cypher Manual, variable-length patterns（https://neo4j.com/docs/cypher-manual/current/patterns/variable-length-patterns/）。

Cypher 可以用可变长度关系模式表达有限多跳查询。本项目先从别名表识别查询中的实体，再以参数化 seeds 和 namespace 匹配 RAGEntity，限制 RELATED 路径为 1..2 跳。返回节点名、关系 kind、source_document 与 depth，并用 1/depth 作为图通道排序信号。固定最大深度和 limit 可以避免小问题扩张成无界遍历；所有用户文本均作为参数或在本地匹配，不拼接进 Cypher。
