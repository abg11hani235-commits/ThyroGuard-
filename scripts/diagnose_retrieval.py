import json
import chromadb
from sentence_transformers import SentenceTransformer

with open('d:/RAG/data/evaluation/gold_evaluation_dataset.json', 'r', encoding='utf-8') as f:
    gold = json.load(f)

with open('d:/RAG/data/processed/chunks.json', 'r', encoding='utf-8') as f:
    chunks = json.load(f)
chunks_map = {c['chunk_id']: c for c in chunks}

embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path='d:/RAG/data/processed/chroma_benchmark')
collection = chroma_client.get_collection('thyroguard_benchmark_corpus')

for idx in [0, 1, 2, 3]:
    q = gold[idx]
    print(f"\n==========================================")
    print(f"[{q['id']}] {q['query']}")
    print(f"Gold Chunks: {q['gold_chunk_ids']}")
    for g_id in q['gold_chunk_ids']:
        if g_id in chunks_map:
            print(f"  * Gold Chunk Content ({g_id}):\n    {chunks_map[g_id]['text'][:200]}...")
    
    q_emb = embed_model.encode([q['query']], normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=10)
    print("Retrieved Chunks:")
    for rank, (cid, dist) in enumerate(zip(res['ids'][0], res['distances'][0]), 1):
        is_gold = " [MATCH!]" if cid in q['gold_chunk_ids'] else ""
        print(f"  {rank}. {cid} (dist: {dist:.4f}){is_gold} -> {chunks_map[cid]['text'][:100]}...")
