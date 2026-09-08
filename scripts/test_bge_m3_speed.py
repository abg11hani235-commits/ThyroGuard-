import torch
import time
import json
from sentence_transformers import SentenceTransformer

torch.set_num_threads(8)
model = SentenceTransformer('BAAI/bge-m3')

with open('data/processed/FROZEN_CORPUS.json', 'r', encoding='utf-8') as f:
    chunks = json.load(f)

texts = [c['text'] for c in chunks[:64]]
t0 = time.time()
embs = model.encode(texts, batch_size=64, normalize_embeddings=True)
dt = time.time() - t0
speed = 64 / dt
est_min = (len(chunks) / speed) / 60
print(f"64 chunks in {dt:.2f}s ({speed:.1f} chunks/sec).")
print(f"Estimated time for all {len(chunks)} chunks: {est_min:.1f} minutes.")
