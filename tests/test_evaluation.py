import unittest
from src.evaluation.retrieval_eval import RetrievalEvaluator
from src.evaluation.generation_eval import GenerationEvaluator

class TestEvaluation(unittest.TestCase):
    def test_retrieval_metrics(self):
        retrieved = ["chunk_1", "chunk_2", "chunk_3", "chunk_4", "chunk_5"]
        gold = ["chunk_2", "chunk_5"]
        
        m = RetrievalEvaluator.evaluate_query(retrieved, gold)
        self.assertAlmostEqual(m["precision@3"], 1/3, places=5)
        self.assertAlmostEqual(m["precision@5"], 2/5, places=5)
        self.assertAlmostEqual(m["recall@3"], 1/2, places=5)
        self.assertAlmostEqual(m["recall@5"], 1.0, places=5)
        self.assertEqual(m["mrr"], 0.5)  # Rank 2
        self.assertEqual(m["hit_rate@3"], 1.0)

    def test_generation_groundedness_metrics(self):
        claims = [
            {"claim": "Claim 1", "citation": "chunk_1"},
            {"claim": "Claim 2", "citation": "chunk_2"}
        ]
        retrieved = ["chunk_1", "chunk_2", "chunk_3"]
        gold = ["chunk_1"]
        
        gm = GenerationEvaluator.evaluate_groundedness(claims, retrieved, gold)
        self.assertEqual(gm["citation_precision"], 1.0)
        self.assertEqual(gm["citation_completeness"], 1.0)
        self.assertEqual(gm["faithfulness"], 0.95)

if __name__ == '__main__':
    unittest.main()
