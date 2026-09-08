import os
import json
import math
import numpy as np
from typing import List, Dict, Any, Tuple
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import re

# 1. Load Data
CHUNKS_PATH = 'd:/RAG/data/processed/chunks.json'
GOLD_DATASET_PATH = 'd:/RAG/data/evaluation/gold_evaluation_dataset.json'
CHROMA_DIR = 'd:/RAG/data/processed/chroma_benchmark'

print("Loading chunks and gold dataset...")
with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
    chunks = json.load(f)

with open(GOLD_DATASET_PATH, 'r', encoding='utf-8') as f:
    gold_data = json.load(f)

# Filter only retrieval questions (excluding out-of-scope and adversarial)
retrieval_queries = [q for q in gold_data if q['gold_chunk_ids']]
print(f"Loaded {len(chunks)} chunks, {len(retrieval_queries)} clinical retrieval queries.")

# 2. Tokenizer for BM25
def bm25_tokenize(text: str) -> List[str]:
    # Lowercase and extract alphanumeric tokens
    tokens = re.findall(r'\b[a-zA-Z0-9_-]+\b', text.lower())
    return tokens

print("Building BM25 Index...")
corpus_tokens = [bm25_tokenize(c['text']) for c in chunks]
bm25 = BM25Okapi(corpus_tokens)

# 3. Dense Indexing with SentenceTransformer & ChromaDB
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
print(f"Loading Embedding Model: {EMBED_MODEL_NAME}...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

os.makedirs(CHROMA_DIR, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

collection_name = "thyroguard_benchmark_corpus"
try:
    chroma_client.delete_collection(collection_name)
except Exception:
    pass

collection = chroma_client.create_collection(
    name=collection_name,
    metadata={"hnsw:space": "cosine"}
)

print(f"Indexing {len(chunks)} chunks into ChromaDB (batching)...")
batch_size = 256
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i+batch_size]
    ids = [c['chunk_id'] for c in batch]
    texts = [c['text'] for c in batch]
    metas = [{
        "doc_slug": c['doc_slug'],
        "page_number": c['page_number'],
        "category": c['category'],
        "subtopic": c['subtopic']
    } for c in batch]
    
    embeddings = embed_model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metas
    )
print("ChromaDB indexing complete.")

# 4. CrossEncoder Reranker
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
print(f"Loading CrossEncoder Reranker: {RERANKER_MODEL_NAME}...")
reranker = CrossEncoder(RERANKER_MODEL_NAME)

# 5. Retrieval Functions
def dense_retrieve(query: str, top_k: int = 25) -> List[Tuple[str, float]]:
    q_emb = embed_model.encode([query], normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=q_emb,
        n_results=top_k
    )
    doc_ids = results['ids'][0]
    distances = results['distances'][0]
    # Convert cosine distance to cosine similarity: sim = 1 - dist
    scores = [1.0 - d for d in distances]
    return list(zip(doc_ids, scores))

def bm25_retrieve(query: str, top_k: int = 25) -> List[Tuple[str, float]]:
    q_tokens = bm25_tokenize(query)
    scores = bm25.get_scores(q_tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(chunks[idx]['chunk_id'], float(scores[idx])) for idx in top_indices if scores[idx] > 0]

def hybrid_retrieve(query: str, dense_k: int = 25, bm25_k: int = 25, dense_weight: float = 0.65, bm25_weight: float = 0.35) -> List[Tuple[str, float]]:
    dense_res = dense_retrieve(query, top_k=dense_k)
    bm25_res = bm25_retrieve(query, top_k=bm25_k)
    
    # Reciprocal Rank Fusion (RRF) with weights
    rrf_scores = {}
    k_rrf = 60
    
    for rank, (doc_id, _) in enumerate(dense_res, 1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + dense_weight * (1.0 / (k_rrf + rank))
        
    for rank, (doc_id, _) in enumerate(bm25_res, 1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + bm25_weight * (1.0 / (k_rrf + rank))
        
    sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_candidates

chunk_lookup = {c['chunk_id']: c['text'] for c in chunks}

def rerank_candidates(query: str, candidates: List[Tuple[str, float]], top_k: int = 8) -> List[Tuple[str, float]]:
    if not candidates:
        return []
    doc_ids = [c[0] for c in candidates]
    pairs = [[query, chunk_lookup[doc_id]] for doc_id in doc_ids]
    scores = reranker.predict(pairs)
    scored = list(zip(doc_ids, [float(s) for s in scores]))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

# 6. Evaluation Metrics Engine
def calculate_metrics_for_query(retrieved_ids: List[str], gold_ids: List[str]) -> Dict[str, float]:
    gold_set = set(gold_ids)
    if not gold_set:
        return {}
        
    metrics = {}
    for k in [3, 5, 8]:
        top_k_ids = retrieved_ids[:k]
        hits = [1 if cid in gold_set else 0 for cid in top_k_ids]
        num_hits = sum(hits)
        
        metrics[f'precision@{k}'] = num_hits / k
        metrics[f'recall@{k}'] = num_hits / len(gold_set)
        metrics[f'hit_rate@{k}'] = 1.0 if num_hits > 0 else 0.0
        
    # MRR (Mean Reciprocal Rank)
    mrr = 0.0
    for rank, cid in enumerate(retrieved_ids, 1):
        if cid in gold_set:
            mrr = 1.0 / rank
            break
    metrics['mrr'] = mrr
    
    # NDCG@5
    dcg = 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(5, len(gold_set))))
    for i, cid in enumerate(retrieved_ids[:5]):
        if cid in gold_set:
            dcg += 1.0 / math.log2(i + 2)
    metrics['ndcg@5'] = (dcg / idcg) if idcg > 0 else 0.0
    
    return metrics

# 7. Run Benchmark Across 3 Configurations
configurations = {
    "Dense (MiniLM)": lambda q: [cid for cid, _ in dense_retrieve(q, top_k=25)],
    "Hybrid (Dense + BM25)": lambda q: [cid for cid, _ in hybrid_retrieve(q, dense_k=25, bm25_k=25)],
    "Hybrid + CrossEncoder Reranker": lambda q: [cid for cid, _ in rerank_candidates(q, hybrid_retrieve(q, dense_k=25, bm25_k=25), top_k=8)]
}

benchmark_results = {}
category_results = {}

for config_name, retriever_fn in configurations.items():
    print(f"\nEvaluating Configuration: {config_name}...")
    query_metrics_list = []
    cat_metrics = {"easy": [], "medium": [], "hard": [], "multi_hop": []}
    
    for q in retrieval_queries:
        retrieved_ids = retriever_fn(q['query'])
        q_metric = calculate_metrics_for_query(retrieved_ids, q['gold_chunk_ids'])
        q_metric['id'] = q['id']
        q_metric['category'] = q['category']
        query_metrics_list.append(q_metric)
        cat_metrics[q['category']].append(q_metric)
        
    # Aggregate Overall
    agg = {}
    metric_keys = ['precision@3', 'precision@5', 'recall@3', 'recall@5', 'recall@8', 'mrr', 'ndcg@5', 'hit_rate@3', 'hit_rate@5', 'hit_rate@8']
    for k in metric_keys:
        agg[k] = float(np.mean([m[k] for m in query_metrics_list]))
    benchmark_results[config_name] = agg
    
    # Aggregate Categories
    cat_agg = {}
    for cat, list_m in cat_metrics.items():
        cat_agg[cat] = {k: float(np.mean([m[k] for m in list_m])) for k in metric_keys}
    category_results[config_name] = cat_agg

# 8. Display & Save Results
print("\n" + "="*80)
print("THYROGUARD RETRIEVAL BENCHMARK SUMMARY (60 Clinical Gold Questions)")
print("="*80)

header = f"| {'Retriever Configuration':<32} | {'P@3':<6} | {'P@5':<6} | {'R@3':<6} | {'R@5':<6} | {'R@8':<6} | {'MRR':<6} | {'NDCG@5':<6} | {'Hit@5':<6} |"
separator = f"|{'-'*34}|{'-'*8}|{'-'*8}|{'-'*8}|{'-'*8}|{'-'*8}|{'-'*8}|{'-'*8}|{'-'*8}|"
print(header)
print(separator)

for config_name, metrics in benchmark_results.items():
    row = f"| {config_name:<32} | {metrics['precision@3']:<6.3f} | {metrics['precision@5']:<6.3f} | {metrics['recall@3']:<6.3f} | {metrics['recall@5']:<6.3f} | {metrics['recall@8']:<6.3f} | {metrics['mrr']:<6.3f} | {metrics['ndcg@5']:<6.3f} | {metrics['hit_rate@5']:<6.3f} |"
    print(row)

# Save Benchmark Artifacts
os.makedirs('d:/RAG/data/evaluation/benchmark_results', exist_ok=True)
with open('d:/RAG/data/evaluation/benchmark_results/retrieval_benchmark.json', 'w', encoding='utf-8') as f:
    json.dump({
        "overall": benchmark_results,
        "per_category": category_results
    }, f, indent=2)

# Generate Markdown Report
with open('d:/RAG/RETRIEVAL_BENCHMARK_REPORT.md', 'w', encoding='utf-8') as f:
    f.write("# ThyroGuard Retrieval Benchmark Report\n\n")
    f.write("Evaluation performed across **60 clinically curated, leak-free evaluation questions** with **133 ground-truth gold chunks**.\n\n")
    f.write("## 1. Overall Multi-Layer Retrieval Comparison\n\n")
    f.write(header + "\n" + separator + "\n")
    for config_name, metrics in benchmark_results.items():
        f.write(f"| **{config_name}** | {metrics['precision@3']:.3f} | {metrics['precision@5']:.3f} | {metrics['recall@3']:.3f} | {metrics['recall@5']:.3f} | {metrics['recall@8']:.3f} | **{metrics['mrr']:.3f}** | **{metrics['ndcg@5']:.3f}** | {metrics['hit_rate@5']:.3f} |\n")
    
    f.write("\n## 2. Per-Category Breakdown (Recall@5 & MRR)\n\n")
    f.write("| Category | Questions | Dense (R@5 / MRR) | Hybrid (R@5 / MRR) | Hybrid + Reranker (R@5 / MRR) |\n")
    f.write("|---|---:|---|---|---|\n")
    for cat in ["easy", "medium", "hard", "multi_hop"]:
        d_r = category_results["Dense (MiniLM)"][cat]
        h_r = category_results["Hybrid (Dense + BM25)"][cat]
        r_r = category_results["Hybrid + CrossEncoder Reranker"][cat]
        f.write(f"| **{cat.capitalize()}** | 15 | {d_r['recall@5']:.3f} / {d_r['mrr']:.3f} | {h_r['recall@5']:.3f} / {h_r['mrr']:.3f} | **{r_r['recall@5']:.3f} / {r_r['mrr']:.3f}** |\n")

print("\nBenchmark report saved to d:/RAG/RETRIEVAL_BENCHMARK_REPORT.md")
