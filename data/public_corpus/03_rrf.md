# Reciprocal Rank Fusion

来源：Microsoft Research, Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods（https://www.microsoft.com/en-us/research/publication/reciprocal-rank-fusion-outperforms-condorcet-and-individual-rank-learning-methods/）。

RRF 用排名而不是不同检索器的原始分数做融合。文档 d 的融合分数为各通道 1/(k+rank(d)) 的总和，常用 rank constant k=60。这样无需校准 BM25 分数与余弦相似度的量纲，同一文档若在多个通道靠前会累积分数。RRF 简单稳定，但它不会自动判断证据真实性；候选池大小、去重键和通道质量仍会影响结果。
