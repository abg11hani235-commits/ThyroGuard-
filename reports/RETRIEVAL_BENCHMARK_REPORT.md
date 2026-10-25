# ThyroGuard Advanced Multi-Layer Retrieval Benchmark

Evaluation across **60 clinically curated gold queries** against **3,073 guideline chunks**.

| Retriever Configuration              | P@3    | P@5    | R@3    | R@5    | R@8    | MRR    | NDCG@5 | Hit@5  |
|--------------------------------------|--------|--------|--------|--------|--------|--------|--------|--------|
| **Dense (BGE-Base-v1.5)** | 0.039 | 0.040 | 0.051 | 0.087 | 0.104 | **0.126** | **0.081** | **0.167** |
| **Hybrid (BGE + BM25)** | 0.044 | 0.043 | 0.054 | 0.093 | 0.099 | **0.112** | **0.079** | **0.167** |
| **Hybrid (BGE + BM25) + CrossEncoder** | 0.061 | 0.043 | 0.082 | 0.093 | 0.121 | **0.116** | **0.082** | **0.183** |
