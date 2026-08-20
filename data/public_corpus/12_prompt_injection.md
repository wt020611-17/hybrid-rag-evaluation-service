# RAG 与 Prompt Injection 边界

来源：OWASP GenAI Security Project, Prompt Injection（https://genai.owasp.org/llmrisk/llm01-prompt-injection/）。

检索到的文档属于不可信数据，可能包含要求模型忽略系统规则的间接提示注入。RAG 不能因为内容来自知识库就默认可信。降低风险的方法包括数据来源控制、把证据与指令分隔、最小化工具权限、输出引用、敏感操作二次授权和记录审计轨迹。检索指标高并不等于生成答案安全，安全测试应单独设计对抗样本。
