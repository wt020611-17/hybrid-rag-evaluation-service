# Hybrid RAG Evaluation Service

一个可复现、可评测的中文混合检索 RAG 项目。项目同时保留零外部服务的 v0.1 基线，以及真实 BGE + Milvus + Neo4j 的 v1.0 生产路径；每项简历表述都能对应到代码、测试或逐题评测报告。

## v1.0 已实现

- `BAAI/bge-small-zh-v1.5` 本地 CPU 推理，按模型卡使用 CLS pooling 与 L2 归一化，输出 512 维向量。
- Milvus 2.5 独立 collection：显式 schema、HNSW/COSINE、M=16、efConstruction=128、ef=64。
- Neo4j 5 独立 namespace：参数化 Cypher、实体别名、一至二跳受限路径。
- 自实现 BM25、可解释 Query Router 与加权 RRF；图通道权重只作用于显式关系查询。
- FastAPI `/health`、`/query`、`/debug/retrieval`，返回 trace ID、chunk、来源、通道和排名。
- 15 篇公开一手资料的中文摘要、17 个 chunk、60 条人工标注查询、五策略消融报告。
- Docker Compose 隔离 Milvus/etcd/MinIO/Neo4j 的端口和命名卷；索引与 API 支持分步启动。
- 原 v0.1 的 TF-IDF/内存图实现仍可离线运行，用于解释技术演进和回归测试。

## 实测结果

2026-08-19，Windows 11、16GB RAM、CPU 推理，固定 60 条查询，`top_k=5`：

| 策略 | Recall@5 | MRR | Top-1 引用准确率 |
|---|---:|---:|---:|
| BM25 | 0.9500 | 0.8292 | 0.7500 |
| BGE + Milvus | 0.9000 | 0.7575 | 0.6667 |
| BM25 + BGE + RRF | 0.9500 | 0.8394 | 0.7667 |
| BM25 + BGE + Neo4j + 加权 RRF | **1.0000** | **0.8528** | 0.7500 |
| Query Router | 0.9667 | 0.7736 | 0.6500 |

Router 标注准确率为 1.0000。完整逐题结果、失败样本、延迟和限制见 [`reports/ablation.json`](reports/ablation.json)。这些数字只代表小规模人工整理语料，不代表通用中文检索基准；本机顺序延迟也不是生产 SLA。

## 架构

```text
public Markdown -> stable chunks -> BM25 -------------------|
                              -> BGE -> Milvus HNSW --------|-> weighted RRF -> citations
explicit relations -----------------> Neo4j 1..2 hop -------|
query -> explainable router --------------------------------^
                                                   -> FastAPI / optional LLM
```

详细数据流和证据边界见 [`docs/architecture.md`](docs/architecture.md)。

## 快速开始

### 离线基线

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
hybrid-rag eval --output reports/baseline.json
pytest
```

### 生产检索（推荐低内存分步运行）

```powershell
Copy-Item .env.example .env
# 修改 .env 中的 NEO4J_PASSWORD 和 BGE_MODEL_PATH
python scripts/download_model.py --output models/bge-small-zh-v1.5
python -m pip install -e ".[dev,production]"

docker compose up -d etcd minio milvus neo4j
hybrid-rag sync-production
hybrid-rag eval-production

$env:RAG_MODE="production"
uvicorn hybrid_rag.api:app --host 0.0.0.0 --port 8000
```

16GB 机器建议索引结束后再启动 API，不要同时运行 Docker build、indexer 和 API 模型。容器化 API 可在本地模型已下载后构建：

```powershell
docker compose build api
docker compose --profile app up -d --no-deps api
```

## API 示例

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
$body = @{
  query = "BGE 与 Milvus、Neo4j 和 RRF 的多跳关系是什么？"
  top_k = 5
  strategy = "hybrid_graph"
  use_llm = $false
} | ConvertTo-Json
Invoke-RestMethod -Method Post http://127.0.0.1:8000/query `
  -ContentType "application/json; charset=utf-8" -Body $body
```

生产策略为 `bm25`、`bge`、`hybrid`、`hybrid_graph`、`routed`。外部 LLM 未配置时仍返回可引用的抽取式答案，不伪造生成能力。

## 验证证据

- `16 passed, 2 skipped`：默认单元/API 测试；两项真实后端测试需显式开启。
- `RUN_PRODUCTION_INTEGRATION=1 pytest tests/test_production_integration.py`：2 项 Milvus/Neo4j 集成测试通过。
- 真实索引：15 documents / 17 chunks / 17 Milvus rows / 17 Neo4j nodes / 24 relationships。
- Docker 镜像成功导出，大小约 565MB；低内存机器采用分步启动。

## 目录

```text
src/hybrid_rag/       基线与生产检索、API、评测
data/public_corpus/   公开一手资料的中文摘要
data/graph/           Neo4j 实体关系与别名
data/eval/            60 条生产评测 + 30 条基线评测
reports/              可复现逐题报告
tests/                单元、API、真实后端集成测试
docs/                 架构、简历边界、验收标准
```

第三方来源与许可证边界见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。
