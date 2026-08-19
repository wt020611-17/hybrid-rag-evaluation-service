# Hybrid RAG Evaluation Service

一个“证据优先”的个人 RAG 项目：用独立实现证明文档入库、BM25、向量检索、图路径检索、RRF 融合、引用返回、Query Router、FastAPI 契约和固定评测集，而不是把教程仓库换名后当作原创。

## 当前已验证范围（v0.1.0）

- 读取 Markdown/TXT，按字符窗口切分并保留来源元数据。
- 纯 Python BM25 与 TF-IDF 向量检索，支持离线运行。
- 基于显式实体关系的 1～2 跳图路径检索。
- Reciprocal Rank Fusion（RRF）融合多路排名。
- 规则 Query Router，提供 `keyword`、`vector`、`graph`、`hybrid` 路由。
- 返回答案片段、引用、分数、检索通道和 `trace_id`。
- 30 条固定查询，复现 Recall@K、MRR、首条引用命中率和路由准确率。
- FastAPI `/health`、`/query`、`/debug/retrieval` 接口与 Dockerfile。

## 诚实边界

当前可复现基线使用 TF-IDF 稀疏向量，不把它宣传成 BGE Embedding；图检索使用项目内的显式关系表，不把它宣传成 Neo4j；向量索引在进程内运行，不把它宣传成 Milvus。BGE、Milvus、Neo4j 和外部 LLM 是下一阶段基础设施接入项，只有真实跑通并留下测试/评测证据后才进入简历完成时态。

学习基线与许可边界见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 架构

```text
Markdown/TXT -> Document -> Chunk
                          |-> BM25 ---------|
                          |-> TF-IDF Vector-|-> RRF -> Citations -> Answer
relations.json ---------->|-> Graph Path --|
Query --------------------> Router ---------^
```

## 快速开始

### 仅运行离线核心与评测（无第三方依赖）

```powershell
$env:PYTHONPATH="src"
python -m hybrid_rag.cli query "为什么不同检索器的原始分数不能直接相加？"
python -m hybrid_rag.cli eval --output reports/baseline.json
python -m unittest discover -s tests -v
```

### 启动 API

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
uvicorn hybrid_rag.api:app --host 0.0.0.0 --port 8000
```

验证：

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod -Method Post http://localhost:8000/query `
  -ContentType 'application/json' `
  -Body '{"query":"RRF 如何融合排名？","top_k":5,"use_llm":false}'
```

### Docker

```powershell
docker build -t hybrid-rag-evaluation-service:0.1.0 .
docker run --rm -p 8000:8000 hybrid-rag-evaluation-service:0.1.0
```

## 目录

```text
src/hybrid_rag/    核心实现与 API
data/corpus/       自编写的合成演示知识库
data/graph/        显式实体关系
data/eval/         30 条固定评测查询
scripts/           一键评测脚本
tests/             单元与端到端测试
reports/           可复现结果
docs/              架构、验收和简历边界
```

## 评测口径

- `Recall@K`：前 K 个结果是否命中任一标注来源。
- `MRR`：第一个命中来源的排名倒数。
- `top1_citation_accuracy`：首条引用是否来自标注来源。
- `route_accuracy`：规则 Router 是否命中标注路由。

演示语料和查询是合成数据，只用于验证工程链路，不代表通用中文 RAG 的行业基准。所有结果必须通过命令重新生成，README 不手写“提升 xx%”。

2026-08-19 在 Windows / Python 3.9.7 上复现的 v0.1.0 基线：15 项测试全部通过；30 条查询的 Recall@5 为 1.0000、MRR 为 0.9111、首条引用命中率为 0.8333、路由准确率为 1.0000。逐条结果和限制见 `reports/baseline.json`。这些数字只描述仓库内的合成回归集。

## API 响应示例

```json
{
  "status": "ok",
  "trace_id": "f54f...",
  "route": "keyword",
  "answer": "RRF 只使用各检索器中的名次...",
  "citations": [
    {
      "source": "data/corpus/05_rrf_fusion.md",
      "chunk_id": "...",
      "score": 0.0325,
      "channel": "rrf"
    }
  ]
}
```

## 下一阶段

1. 用 `BAAI/bge-small-zh-v1.5` 替换 TF-IDF 基线并保存同一评测集对比。
2. 接入 Milvus，补 collection schema、索引参数、空结果和连接失败测试。
3. 接入 Neo4j，补导入脚本、受限跳数 Cypher 和路径爆炸保护。
4. 将规则 Router 与可评测的分类 Router 对比。
5. 接入 OpenAI-compatible LLM，保留无 LLM 时的引用降级。

完整前置条件和发布流程见 [PRECONDITIONS.md](PRECONDITIONS.md)。

