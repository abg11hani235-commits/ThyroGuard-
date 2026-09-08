from typing import List, Tuple
from .dense_retriever import DenseRetriever
from .bm25_retriever import BM25Retriever

class HybridRetriever:
    """Fuses Dense and Sparse retrieval candidates using weighted Reciprocal Rank Fusion (RRF)."""
    def __init__(self, dense_retriever: DenseRetriever, bm25_retriever: BM25Retriever, dense_weight: float = 0.60, bm25_weight: float = 0.40, rrf_k: int = 40):
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k

    def retrieve(self, query: str = None, pool_size: int = 40, dense_query: str = None, sparse_query: str = None) -> List[Tuple[str, float]]:
        d_q = dense_query or query
        s_q = sparse_query or query
        dense_results = self.dense.retrieve(d_q, top_k=pool_size)
        bm25_results = self.bm25.retrieve(s_q, top_k=pool_size)

        rrf_scores = {}
        for rank, (doc_id, _) in enumerate(dense_results, 1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + self.dense_weight * (1.0 / (self.rrf_k + rank))

        for rank, (doc_id, _) in enumerate(bm25_results, 1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + self.bm25_weight * (1.0 / (self.rrf_k + rank))

        sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_candidates[:pool_size]
