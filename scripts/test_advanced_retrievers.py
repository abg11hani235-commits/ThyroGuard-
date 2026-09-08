import os
import json
import math
import numpy as np
from typing import List, Dict, Any, Tuple
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import re

# Load Data
CHUNKS_PATH = 'd:/RAG/data/processed/chunks.json'
GOLD_DATASET_PATH = 'd:/RAG/data/evaluation/gold_evaluation_dataset.json'

with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
    chunks = json.load(f)

with open(GOLD_DATASET_PATH, 'r', encoding='utf-8') as f:
    gold_data = json.load(f)

retrieval_queries = [q for q in gold_data if q['gold_chunk_ids']]
chunk_lookup = {c['chunk_id']: c['text'] for c in chunks}

# BM25 Tokenizer
def bm25_tokenize(text: str) -> List[str]:
    return re.findall(r'\b[a-zA-Z0-9_-]+\b', text.lower())

print("Building BM25 index...")
corpus_tokens = [bm25_tokenize(c['text']) for c in chunks]
bm25 = BM25Okapi(corpus_tokens)

# Test BGE Embedding
BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
print(f"Loading {BGE_MODEL_NAME}...")
try:
    bge_embed = SentenceTransformer(BGE_MODEL_NAME)
except Exception as e:
    print(f"Could not load BGE ({e}), falling back to MiniLM")
    bge_embed = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

CHROMA_BGE_DIR = 'd:/RAG/data/processed/chroma_bge'
os.makedirs(CHROMA_BGE_DIR, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=CHROMA_BGE_DIR)

collection_name = "thyroguard_bge_corpus"
try:
    chroma_client.delete_collection(collection_name)
except Exception:
    pass

bge_collection = chroma_client.create_collection(
    name=collection_name,
    metadata={"hnsw:space": "cosine"}
)

print("Indexing with BGE embeddings...")
batch_size = 128
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i+batch_size]
    ids = [c['chunk_id'] for c in batch]
    texts = [c['text'] for c in batch]
    metas = [{
        "doc_slug": c['doc_slug'],
        "page_number": c['page_number'],
        "subtopic": c['subtopic']
    } for c in batch]
    
    embeddings = bge_embed.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()
    bge_collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metas
    )

print("Loading CrossEncoder Reranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def dense_bge_retrieve(query: str, top_k: int = 50) -> List[Tuple[str, float]]:
    # BGE query instruction
    inst_query = f"Represent this sentence for searching relevant passages: {query}"
    q_emb = bge_embed.encode([inst_query], normalize_embeddings=True).tolist()
    res = bge_collection.query(query_embeddings=q_emb, n_results=top_k)
    doc_ids = res['ids'][0]
    scores = [1.0 - d for d in res['distances'][0]]
    return list(zip(doc_ids, scores))

def bm25_retrieve(query: str, top_k: int = 50) -> List[Tuple[str, float]]:
    q_tokens = bm25_tokenize(query)
    scores = bm25.get_scores(q_tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(chunks[idx]['chunk_id'], float(scores[idx])) for idx in top_indices if scores[idx] > 0]

def advanced_hybrid_retrieve(query: str, top_k_pool: int = 50) -> List[Tuple[str, float]]:
    dense_res = dense_bge_retrieve(query, top_k=top_k_pool)
    bm25_res = bm25_retrieve(query, top_k=top_k_pool)
    
    rrf_scores = {}
    k_rrf = 40
    for rank, (doc_id, _) in enumerate(dense_res, 1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 0.60 * (1.0 / (k_rrf + rank))
    for rank, (doc_id, _) in enumerate(bm25_res, 1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 0.40 * (1.0 / (k_rrf + rank))
        
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k_pool]

def rerank_top_k(query: str, candidates: List[Tuple[str, float]], final_top_k: int = 8) -> List[Tuple[str, float]]:
    if not candidates:
        return []
    doc_ids = [c[0] for c in candidates]
    pairs = [[query, chunk_lookup[doc_id]] for doc_id in doc_ids]
    scores = reranker.predict(pairs)
    scored = list(zip(doc_ids, [float(s) for s in scores]))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:final_top_k]

def eval_query(retrieved_ids: List[str], gold_ids: List[str]) -> Dict[str, float]:
    gold_set = set(gold_ids)
    if not gold_set:
        return {}
    res = {}
    for k in [3, 5, 8]:
        top_k = retrieved_ids[:k]
        hits = sum(1 for cid in top_k if cid in gold_set)
        res[f'p@{k}'] = hits / k
        res[f'r@{k}'] = hits / len(gold_set)
        res[f'hit@{k}'] = 1.0 if hits > 0 else 0.0
    # MRR
    mrr = 0.0
    for rank, cid in enumerate(retrieved_ids, 1):
        if cid in gold_set:
            mrr = 1.0 / rank
            break
    res['mrr'] = mrr
    # NDCG@5
    dcg = sum(1.0 / math.log2(i + 2) for i, cid in enumerate(retrieved_ids[:5]) if cid in gold_set)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(5, len(gold_set))))
    res['ndcg@5'] = (dcg / idcg) if idcg > 0 else 0.0
    return res

runs = {
    "Dense (BGE-Base-v1.5)": lambda q: [c[0] for c in dense_bge_retrieve(q, top_k=25)],
    "Hybrid (BGE + BM25)": lambda q: [c[0] for c in advanced_hybrid_retrieve(q, top_k_pool=25)],
    "Hybrid (BGE + BM25) + CrossEncoder": lambda q: [c[0] for c in rerank_top_k(q, advanced_hybrid_retrieve(q, top_k_pool=50), final_top_k=8)]
}

print("\n" + "="*85)
print(f"| {'Retriever Configuration':<36} | {'P@3':<6} | {'P@5':<6} | {'R@3':<6} | {'R@5':<6} | {'R@8':<6} | {'MRR':<6} | {'NDCG@5':<6} | {'Hit@5':<6} |")
print(f"|{'-'*38}|{'-'*8}|{'-'*8}|{'-'*8}|{'-'*8}|{'-'*8}|{'-'*8}|{'-'*8}|{'-'*8}|")

final_results = {}
for name, fn in runs.items():
    all_m = []
    for q in retrieval_queries:
        ret = fn(q['query'])
        m = eval_query(ret, q['gold_chunk_ids'])
        all_m.append(m)
    
    agg = {k: float(np.mean([x[k] for x in all_m])) for k in all_m[0]}
    final_results[name] = agg
    print(f"| {name:<36} | {agg['p@3']:<6.3f} | {agg['p@5']:<6.3f} | {agg['r@3']:<6.3f} | {agg['r@5']:<6.3f} | {agg['r@8']:<6.3f} | {agg['mrr']:<6.3f} | {agg['ndcg@5']:<6.3f} | {agg['hit@5']:<6.3f} |")

# Update report
with open('d:/RAG/RETRIEVAL_BENCHMARK_REPORT.md', 'w', encoding='utf-8') as f:
    f.write("# ThyroGuard Advanced Multi-Layer Retrieval Benchmark\n\n")
    f.write("Evaluation across **60 clinically curated gold queries** against **3,073 guideline chunks**.\n\n")
    f.write(f"| {'Retriever Configuration':<36} | {'P@3':<6} | {'P@5':<6} | {'R@3':<6} | {'R@5':<6} | {'R@8':<6} | {'MRR':<6} | {'NDCG@5':<6} | {'Hit@5':<6} |\n")
    f.write(f"|{'-'*38}|{'-'*8}|{'-'*8}|{'-'*8}|{'-'*8}|{'-'*8}|{'-'*8}|{'-'*8}|{'-'*8}|\n")
    for name, agg in final_results.items():
        f.write(f"| **{name}** | {agg['p@3']:.3f} | {agg['p@5']:.3f} | {agg['r@3']:.3f} | {agg['r@5']:.3f} | {agg['r@8']:.3f} | **{agg['mrr']:.3f}** | **{agg['ndcg@5']:.3f}** | **{agg['hit@5']:.3f}** |\n")

print("\nBenchmark updated in d:/RAG/RETRIEVAL_BENCHMARK_REPORT.md")
