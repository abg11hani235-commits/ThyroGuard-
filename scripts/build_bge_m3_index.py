"""
Build ChromaDB Dense Index for BAAI/bge-m3 (1024-D).
Outputs to: data/processed/chroma_bge_m3
Collection: thyroguard_bge_m3_corpus
"""

import os
import json
import time
import torch
import chromadb
from sentence_transformers import SentenceTransformer

def build_m3_index():
    corpus_file = "data/processed/FROZEN_CORPUS.json"
    chroma_dir = "data/processed/chroma_bge_m3"
    collection_name = "thyroguard_bge_m3_corpus"
    batch_size = 64

    if not os.path.exists(corpus_file):
        raise FileNotFoundError(f"Corpus file not found at {corpus_file}")

    print(f"[BGE-M3 Indexer] Loading corpus from {corpus_file}...")
    with open(corpus_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    total_chunks = len(chunks)
    print(f"[BGE-M3 Indexer] Total chunks to index: {total_chunks}")

    # Set threads
    torch.set_num_threads(8)

    print("[BGE-M3 Indexer] Initializing ChromaDB client...")
    os.makedirs(chroma_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    # Check already indexed
    existing_ids = set(collection.get(include=[])["ids"])
    print(f"[BGE-M3 Indexer] Already indexed chunks: {len(existing_ids)} / {total_chunks}")

    chunks_to_index = [c for c in chunks if c["chunk_id"] not in existing_ids]
    if not chunks_to_index:
        print("[BGE-M3 Indexer] All chunks are already indexed! Index is complete.")
        return

    print(f"[BGE-M3 Indexer] Loading embedding model BAAI/bge-m3...")
    t0 = time.time()
    model = SentenceTransformer("BAAI/bge-m3")
    print(f"[BGE-M3 Indexer] Model loaded in {time.time() - t0:.2f}s")

    start_time = time.time()
    processed_count = 0
    total_to_process = len(chunks_to_index)

    for i in range(0, total_to_process, batch_size):
        batch = chunks_to_index[i:i + batch_size]
        batch_ids = [c["chunk_id"] for c in batch]
        batch_texts = [c["text"] for c in batch]
        batch_metas = []
        for c in batch:
            batch_metas.append({
                "doc_slug": c.get("doc_slug", ""),
                "page_number": int(c.get("page_number", 1)),
                "subtopic": c.get("subtopic", "General")
            })

        # Encode with BGE-M3 (1024-D, normalized)
        embeddings = model.encode(
            batch_texts,
            batch_size=len(batch),
            normalize_embeddings=True,
            show_progress_bar=False
        ).tolist()

        collection.add(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=batch_metas
        )

        processed_count += len(batch)
        elapsed = time.time() - start_time
        speed = processed_count / elapsed
        remaining_chunks = total_to_process - processed_count
        eta_minutes = (remaining_chunks / speed) / 60 if speed > 0 else 0

        pct = (processed_count / total_to_process) * 100
    print(f"[BGE-M3 Indexer] Indexing completed in {(time.time() - start_time)/60:.1f} minutes!")
    print(f"[BGE-M3 Indexer] Total chunks in collection: {collection.count()}")
    complete_marker = os.path.join(chroma_dir, "INDEX_COMPLETE")
    with open(complete_marker, "w", encoding="utf-8") as f:
        f.write(f"count={collection.count()}\ncompleted_at={time.time()}\n")
    print(f"[BGE-M3 Indexer] Created completion marker at {complete_marker}")

if __name__ == "__main__":
    build_m3_index()
