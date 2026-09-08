import re
import numpy as np
from typing import List, Tuple, Dict, Any
from rank_bm25 import BM25Okapi

class BM25Retriever:
    """Performs sparse keyword retrieval using BM25Okapi."""
    def __init__(self, corpus_chunks: List[Dict[str, Any]]):
        self.chunks = corpus_chunks
        self.chunk_ids = [c['chunk_id'] for c in corpus_chunks]
        self.corpus_tokens = [self.tokenize(c['text']) for c in corpus_chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r'\b[a-zA-Z0-9_-]+\b', text.lower())

    def retrieve(self, query: str, top_k: int = 40) -> List[Tuple[str, float]]:
        q_tokens = self.tokenize(query)
        scores = self.bm25.get_scores(q_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunk_ids[idx], float(scores[idx])) for idx in top_indices if scores[idx] > 0]
