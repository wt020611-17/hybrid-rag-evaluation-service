# Milvus HNSW 与 COSINE 搜索

来源：Milvus 官方 HNSW 文档（https://milvus.io/docs/hnsw.md）。

HNSW 是基于多层近邻图的近似最近邻索引。M 影响每个节点维护的连接数量，efConstruction 影响建索引时的搜索宽度，查询阶段 ef 越大通常召回越高但延迟也更高。本项目使用 HNSW、COSINE、M=16、efConstruction=128、查询 ef=64。参数选择必须通过相同数据集做召回和延迟对比，不能只根据默认值宣称性能优势。
