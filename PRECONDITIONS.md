# Preconditions and release checklist

## v0.1.0 本地复现

- Python 3.9+（推荐 3.11 或 3.12）。
- 创建新的 `.venv`；不要复制其他项目的虚拟环境。
- 安装 `.[dev]` 后运行 `pytest -q`。
- 离线评测不需要 API Key、Milvus、Neo4j 或外部模型。
- 启动 API 后分别请求 `/health` 与 `/query`。

## Docker

- Docker Desktop/Engine 必须处于运行状态。
- `docker build` 成功后再写“应用容器化验证完成”。
- 只有数据库容器启动不等于应用已经容器化部署。

## BGE + Milvus

- 固定 BGE 模型名称、版本、向量维度、归一化方式和距离度量。
- Milvus collection schema、index type、metric type 与模型输出一致。
- 补充写入、检索、空结果、重复入库、删除与连接失败测试。
- 在同一 30 条查询上生成与 TF-IDF 基线可对比的新报告。

## Neo4j

- 固定节点/关系类型、方向、唯一键和最大跳数。
- 提供可重复导入脚本和至少 1 跳、2 跳、带过滤的 Cypher。
- 测试实体未命中、路径过多、连接失败与超时。
- 返回路径证据，不能只返回大模型总结。

## LLM

- `.env` 只保存在本机，仓库只提交 `.env.example`。
- 使用 OpenAI-compatible endpoint 时配置超时与检索降级。
- 评测检索与生成分开，避免用生成流畅度掩盖召回失败。

## GitHub

本项目应建立一个新的个人 GitHub 仓库，例如
`hybrid-rag-evaluation-service`。不要把 Datawhale 上游副本的内容和作者
历史伪装成个人历史。推荐流程：

1. 在 GitHub 创建同名空仓库，不自动生成 README/LICENSE/.gitignore。
2. 本地保留当前独立 Git 历史。
3. 添加自己的远端：`git remote add origin <your-repo-url>`。
4. 推送前再次确认 `git status`、密钥扫描和 `reports/baseline.json`。
5. 首次推送：`git push -u origin main`。

GitHub 仓库可以先设为 Private；确认许可、隐私、README 和演示材料后再
转 Public。上游学习来源继续保留在 `THIRD_PARTY_NOTICES.md`。

