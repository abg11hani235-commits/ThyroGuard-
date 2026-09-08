import math
from typing import List, Dict, Set

class RetrievalEvaluator:
    """Computes exact Information Retrieval (IR) metrics against gold evidence chunks."""
    @staticmethod
    def evaluate_query(retrieved_ids: List[str], gold_ids: List[str]) -> Dict[str, float]:
        gold_set = set(gold_ids)
        if not gold_set:
            return {}

        metrics = {}
        for k in [3, 5, 8]:
            top_k = retrieved_ids[:k]
            hits = sum(1 for cid in top_k if cid in gold_set)
            metrics[f'precision@{k}'] = hits / k
            metrics[f'recall@{k}'] = hits / len(gold_set)
            metrics[f'hit_rate@{k}'] = 1.0 if hits > 0 else 0.0

        # Mean Reciprocal Rank (MRR)
        mrr = 0.0
        for rank, cid in enumerate(retrieved_ids, 1):
            if cid in gold_set:
                mrr = 1.0 / rank
                break
        metrics['mrr'] = mrr

        # Normalized Discounted Cumulative Gain (NDCG@5)
        dcg = sum(1.0 / math.log2(i + 2) for i, cid in enumerate(retrieved_ids[:5]) if cid in gold_set)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(5, len(gold_set))))
        metrics['ndcg@5'] = (dcg / idcg) if idcg > 0 else 0.0

        return metrics
