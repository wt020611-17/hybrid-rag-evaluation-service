# Milvus Collection Schema

来源：Milvus 官方 Schema 文档（https://milvus.io/docs/schema.md）。

Milvus collection 需要定义主键、向量字段和用于回传的标量字段。本项目使用 chunk_id 作为 VARCHAR 主键，embedding 为 512 维 FLOAT_VECTOR，并保存 document_id、source、text、position。关闭 dynamic field 可以让写入错误更早暴露。索引重建时必须保证每条向量与 chunk 元数据一一对应；删除文档时按 source 过滤，而不是清空其他项目的数据。
