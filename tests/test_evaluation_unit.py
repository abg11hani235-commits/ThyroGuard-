import math
import numpy as np
from typing import List, Dict, Set

class IRMetrics:
    @staticmethod
    def precision_at_k(retrieved: List[str], gold: Set[str], k: int) -> float:
        if k <= 0:
            return 0.0
        top_k = retrieved[:k]
        hits = sum(1 for doc in top_k if doc in gold)
        return hits / k

    @staticmethod
    def recall_at_k(retrieved: List[str], gold: Set[str], k: int) -> float:
        if not gold:
            return 0.0
        top_k = retrieved[:k]
        hits = sum(1 for doc in top_k if doc in gold)
        return hits / len(gold)

    @staticmethod
    def hit_rate_at_k(retrieved: List[str], gold: Set[str], k: int) -> float:
        top_k = retrieved[:k]
        hits = sum(1 for doc in top_k if doc in gold)
        return 1.0 if hits > 0 else 0.0

    @staticmethod
    def mrr(retrieved: List[str], gold: Set[str]) -> float:
        for rank, doc in enumerate(retrieved, 1):
            if doc in gold:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def ndcg_at_k(retrieved: List[str], gold: Set[str], k: int) -> float:
        if not gold or k <= 0:
            return 0.0
        dcg = sum(1.0 / math.log2(i + 2) for i, doc in enumerate(retrieved[:k]) if doc in gold)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(gold))))
        return dcg / idcg if idcg > 0 else 0.0

def test_toy_examples():
    print("Testing IRMetrics on Toy Examples...")
    
    # Toy Case 1: Gold = {A, B}. Retrieved = [A, X, Y, B, Z]
    gold_1 = {"A", "B"}
    ret_1 = ["A", "X", "Y", "B", "Z"]
    
    # P@3: hits in top 3 is [A] -> 1/3 = 0.3333...
    assert abs(IRMetrics.precision_at_k(ret_1, gold_1, 3) - 1/3) < 1e-5
    # R@3: hits in top 3 is 1 / 2 = 0.5
    assert abs(IRMetrics.recall_at_k(ret_1, gold_1, 3) - 0.5) < 1e-5
    # R@5: hits in top 5 is [A, B] -> 2 / 2 = 1.0
    assert abs(IRMetrics.recall_at_k(ret_1, gold_1, 5) - 1.0) < 1e-5
    # Hit@3: 1.0
    assert IRMetrics.hit_rate_at_k(ret_1, gold_1, 3) == 1.0
    # MRR: First hit is rank 1 ('A') -> 1.0
    assert IRMetrics.mrr(ret_1, gold_1) == 1.0
    # NDCG@3: DCG = 1/log2(2) = 1.0. IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.6309 = 1.6309. NDCG = 1 / 1.6309 = 0.6131
    idcg_2 = (1.0 / math.log2(2)) + (1.0 / math.log2(3))
    assert abs(IRMetrics.ndcg_at_k(ret_1, gold_1, 3) - (1.0 / idcg_2)) < 1e-5
    
    # Toy Case 2: Gold = {G}. Retrieved = [X, Y, G, Z]
    gold_2 = {"G"}
    ret_2 = ["X", "Y", "G", "Z"]
    assert IRMetrics.precision_at_k(ret_2, gold_2, 2) == 0.0
    assert IRMetrics.precision_at_k(ret_2, gold_2, 3) == 1/3
    assert IRMetrics.recall_at_k(ret_2, gold_2, 2) == 0.0
    assert IRMetrics.recall_at_k(ret_2, gold_2, 3) == 1.0
    assert IRMetrics.mrr(ret_2, gold_2) == 1/3  # Rank 3
    
    # Toy Case 3: Zero matches
    gold_3 = {"A"}
    ret_3 = ["X", "Y", "Z"]
    assert IRMetrics.precision_at_k(ret_3, gold_3, 3) == 0.0
    assert IRMetrics.recall_at_k(ret_3, gold_3, 3) == 0.0
    assert IRMetrics.hit_rate_at_k(ret_3, gold_3, 3) == 0.0
    assert IRMetrics.mrr(ret_3, gold_3) == 0.0
    assert IRMetrics.ndcg_at_k(ret_3, gold_3, 3) == 0.0
    
    print("ALL TOY METRIC TESTS PASSED SUCCESSFULLY!")

if __name__ == '__main__':
    test_toy_examples()
