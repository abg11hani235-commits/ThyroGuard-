import sys
import os
sys.path.insert(0, os.path.abspath("."))
import time
from src.pipeline.rag_pipeline import ThyroGuardPipeline

p = ThyroGuardPipeline('data/processed/FROZEN_CORPUS.json', 'data/processed/chroma_bge')

q = "ما هي جرعة الالتروكسين للبالغين الأصحاء؟"

# 1st call (Cold execution)
t0 = time.time()
r1 = p.query(q)
ms1 = round((time.time() - t0) * 1000)

# 2nd call (Warm cache hit)
t1 = time.time()
r2 = p.query(q)
ms2 = round((time.time() - t1) * 1000)

print("=" * 60)
print(f"1st Call (Full Pipeline) : {ms1}ms | Cached: {r1.get('cached', False)}")
print(f"2nd Call (Semantic Cache): {ms2}ms | Cached: {r2.get('cached', False)} (Type: {r2.get('cache_hit_type')})")
print(f"Latency Reduction        : {round((1 - ms2/ms1)*100, 1)}% faster")
print("=" * 60)
