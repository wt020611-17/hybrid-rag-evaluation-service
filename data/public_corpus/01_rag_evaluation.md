# RAG 检索评测的基本指标

来源：Stanford Introduction to Information Retrieval（https://nlp.stanford.edu/IR-book/）与 TREC 评测方法。

检索评测必须把查询、相关文档标签和固定的 top-k 配置保存下来。Recall@k 表示前 k 个结果是否覆盖相关证据，MRR 使用第一个相关结果排名的倒数，Top-1 citation accuracy 检查第一条引用是否正确。离线指标只反映给定语料与标签，不能替代并发压测、人工答案质量评审或线上反馈。为了防止只展示成功案例，报告还应保存逐题结果、失败样本、数据版本和局限性。
