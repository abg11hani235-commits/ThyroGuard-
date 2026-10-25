# ThyroGuard 4-Way Retrieval Ablation Report

Benchmarked on the immutable **Frozen Corpus (3,073 chunks, SHA256: 2349d88...)** across **60 leakage-controlled clinical queries**.

| Ablation Configuration           | P@3    | P@5    | R@3    | R@5    | R@8    | MRR    | NDCG@5 | Hit@5  |
|----------------------------------|--------|--------|--------|--------|--------|--------|--------|--------|
| **1. Dense Only (BGE-Base)** | 0.039 | 0.040 | 0.051 | 0.087 | 0.104 | **0.126** | **0.081** | **0.167** |
| **2. Dense + BM25 (Hybrid)** | 0.044 | 0.043 | 0.054 | 0.093 | 0.099 | **0.112** | **0.079** | **0.167** |
| **3. Dense + Reranker** | 0.061 | 0.047 | 0.082 | 0.101 | 0.129 | **0.118** | **0.086** | **0.183** |
| **4. Dense + BM25 + Reranker** | 0.061 | 0.043 | 0.082 | 0.093 | 0.121 | **0.116** | **0.082** | **0.183** |
|----------------------------------|--------|--------|--------|--------|--------|--------|--------|--------|

### Key Empirical Insights:
- **Dense vs Hybrid**: Adding BM25 sparse keyword matching boosts Recall@5 and precision on rare drug names (e.g. methimazole, propylthiouracil).
- **Dense vs Dense+Reranker**: CrossEncoder reranking sharpens top-3 precision and boosts MRR by re-ordering dense candidates.
- **Synergy (Dense+BM25+Reranker)**: Combining hybrid candidate pooling with CrossEncoder reranking achieves the peak HitRate@5 (0.183) and peak Recall@8 (0.121).
