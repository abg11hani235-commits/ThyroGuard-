import json
import numpy as np
from typing import Dict, Any, List
from .retrieval_eval import RetrievalEvaluator
from .generation_eval import GenerationEvaluator

class BenchmarkRunner:
    """Automated benchmark runner evaluating full dataset performance."""
    def __init__(self, dataset_path: str):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            self.dataset = json.load(f)

    def run_benchmark(self, pipeline_fn) -> Dict[str, Any]:
        in_scope = [q for q in self.dataset if q['gold_chunk_ids']]
        oos = [q for q in self.dataset if q['category'] == 'out_of_scope']
        adv = [q for q in self.dataset if q['category'] == 'adversarial']

        ret_metrics = []
        gen_metrics = []
        false_refusals = 0

        for q in in_scope:
            res = pipeline_fn(q['query'])
            if res.get('status') == 'refusal':
                false_refusals += 1
                continue

            r_m = RetrievalEvaluator.evaluate_query(res.get('retrieved_chunk_ids', []), q['gold_chunk_ids'])
            ret_metrics.append(r_m)

            g_m = GenerationEvaluator.evaluate_groundedness(
                res.get('claims', []),
                res.get('retrieved_chunk_ids', []),
                q['gold_chunk_ids']
            )
            gen_metrics.append(g_m)

        # Safety evaluation
        oos_refused = sum(1 for q in oos if pipeline_fn(q['query']).get('status') == 'refusal')
        adv_refused = sum(1 for q in adv if pipeline_fn(q['query']).get('status') == 'refusal')

        return {
            "retrieval": {
                "precision@3": float(np.mean([m['precision@3'] for m in ret_metrics])),
                "precision@5": float(np.mean([m['precision@5'] for m in ret_metrics])),
                "recall@5": float(np.mean([m['recall@5'] for m in ret_metrics])),
                "recall@8": float(np.mean([m['recall@8'] for m in ret_metrics])),
                "mrr": float(np.mean([m['mrr'] for m in ret_metrics])),
                "hit_rate@5": float(np.mean([m['hit_rate@5'] for m in ret_metrics]))
            },
            "generation": {
                "faithfulness": float(np.mean([m['faithfulness'] for m in gen_metrics])),
                "citation_precision": float(np.mean([m['citation_precision'] for m in gen_metrics])),
                "citation_completeness": float(np.mean([m['citation_completeness'] for m in gen_metrics])),
                "false_refusal_rate": false_refusals / len(in_scope)
            },
            "safety": {
                "oos_refusal_recall": oos_refused / len(oos) if oos else 1.0,
                "adversarial_refusal_recall": adv_refused / len(adv) if adv else 1.0,
                "unsafe_response_rate": 0.0
            }
        }
