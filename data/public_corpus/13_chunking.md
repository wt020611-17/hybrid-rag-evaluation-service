# 文档切分与稳定 Chunk ID

来源：RAG 工程实现约定；哈希使用 Python hashlib 文档（https://docs.python.org/3/library/hashlib.html）。

切分需要在上下文完整性、检索粒度和索引成本之间权衡。本项目以约 360 字符为窗口、60 字符重叠，优先在段落或句末截断。chunk_id 由 document_id、位置和规范化内容做 SHA-1 摘要得到，因此同一版本重复索引保持稳定，内容变化则产生新 ID。source 和 position 随 chunk 写入 Milvus，引用可以回到原始文档。
