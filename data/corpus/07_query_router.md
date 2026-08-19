# Query Router 与降级

Query Router 根据问题特征选择检索路线。本项目提供四类路由：编号和接口等精确标识走 keyword，概念解释走 vector，关系和多跳问题走 graph，其余问题走 hybrid。

Router 本身必须评测。固定查询集为每个问题标注 expected_route，再计算 route_accuracy。只展示几个成功案例不能证明路由可靠，因为错误路由可能让后续最强的检索器也拿不到证据。

路由失败需要有限降级：graph 没有命中实体时仍补充 BM25；外部 LLM 未配置或超时时返回检索片段与引用，并把状态标记为 degraded，而不是让整个请求无响应。

