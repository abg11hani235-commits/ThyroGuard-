import json
import re

with open('d:/RAG/data/processed/chunks.json', 'r', encoding='utf-8') as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks.")

def find_evidence(doc_slug=None, keywords=[], min_matches=1):
    results = []
    for c in chunks:
        if doc_slug and c['doc_slug'] != doc_slug:
            continue
        text_lower = c['text'].lower()
        matches = [kw for kw in keywords if kw.lower() in text_lower]
        if len(matches) >= min_matches:
            results.append((len(matches), c['chunk_id'], c['page_number'], c['text'][:250]))
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:5]

# Test queries
print("Search: Starting dose levothyroxine in healthy adult")
for r in find_evidence('ata_2014_hypo', ['1.6', 'mcg/kg', 'dose', 'levothyroxine'], 3):
    print(r[1], f"p.{r[2]}", r[3].replace('\n', ' '))
