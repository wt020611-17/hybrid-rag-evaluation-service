import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RouteDecision:
    route: str
    reason: str


class QueryRouter:
    GRAPH_HINTS = ("关系", "路径", "关联", "依赖", "多跳", "图谱", "图检索", "连接")
    VECTOR_HINTS = ("含义", "概念", "如何理解", "为什么", "区别", "作用")

    def route(self, query: str) -> RouteDecision:
        stripped = query.strip()
        if not stripped:
            return RouteDecision("hybrid", "empty query falls back to hybrid")
        if any(hint in stripped for hint in self.GRAPH_HINTS):
            return RouteDecision("graph", "graph relationship hint matched")
        if re.search(r"(?:/[a-z][a-z0-9/_-]*|[A-Z]{2,}[A-Za-z0-9@._-]*|\w+@\w+|\w+_\w+|\d+\.\d+)", stripped):
            return RouteDecision("keyword", "identifier or exact token matched")
        if any(hint in stripped for hint in self.VECTOR_HINTS):
            return RouteDecision("vector", "conceptual question hint matched")
        return RouteDecision("hybrid", "default multi-channel retrieval")
