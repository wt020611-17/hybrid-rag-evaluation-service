# 引用与可观测性

来源：OpenTelemetry Trace 规范（https://opentelemetry.io/docs/concepts/signals/traces/）及本项目工程设计。

可追溯 RAG 响应至少需要 trace_id、实际路由、检索通道、chunk_id、source、rank 和分数。引用不是装饰：它让使用者定位证据，也让开发者检查错误来自检索、融合还是生成。离线报告保存每题来源列表和首个相关结果排名；服务健康接口报告后端连通性和索引数量。单机延迟只作为回归基线，不能包装成生产 SLA。
