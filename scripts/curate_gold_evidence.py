import json
import sys
import io

# Fix encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

with open('d:/RAG/data/processed/chunks.json', 'r', encoding='utf-8') as f:
    chunks = json.load(f)
chunks_map = {c['chunk_id']: c for c in chunks}

def search_text(keywords, doc_filter=None, max_res=5):
    res = []
    for c in chunks:
        if doc_filter and c['doc_slug'] != doc_filter:
            continue
        text_lower = c['text'].lower()
        score = sum(1 for kw in keywords if kw.lower() in text_lower)
        if score > 0:
            res.append((score, c['chunk_id'], c['page_number'], c['text']))
    res.sort(key=lambda x: (x[0], -len(x[3])), reverse=True)
    return res[:max_res]

queries_to_curate = [
    # EASY
    ("EASY_01", ["starting dose", "levothyroxine", "1.6", "kg"], "ata_2014_hypo"),
    ("EASY_02", ["serum thyrotropin", "single best", "screening test", "tsh"], "ata_aace_2012_hypo"),
    ("EASY_03", ["methimazole", "first-line", "graves", "preferred"], "ata_2016_hyper"),
    ("EASY_04", ["high suspicion", "fna", "1 cm", "nodule"], "ata_2015_nodules_dtc"),
    ("EASY_05", ["4-8 weeks", "6-8 weeks", "adjusting", "tsh", "re-evaluated", "levothyroxine"], "ata_2014_hypo"),
    ("EASY_06", ["subclinical hypothyroidism", "normal free", "elevated tsh"], "ata_aace_2012_hypo"),
    ("EASY_07", ["first trimester", "propylthiouracil", "methimazole", "embryopathy"], "ata_2016_hyper"),
    ("EASY_08", ["empty stomach", "breakfast", "bedtime", "levothyroxine", "water"], "ata_2014_hypo"),
    ("EASY_09", ["anaplastic", "stage iv", "staging", "atc"], "ata_2020_atc"),
    ("EASY_10", ["papillary", "most common", "differentiated", "80%"], "review_2021_thyroid_cancer"),
    ("EASY_11", ["agranulocytosis", "fever", "sore throat", "complete blood count"], "ata_2016_hyper"),
    ("EASY_12", ["purely cystic", "fna", "biopsy", "benign"], "ata_2015_nodules_dtc"),
    ("EASY_13", ["thyroglobulin", "surveillance", "recurrence", "tumor marker"], "ata_2015_nodules_dtc"),
    ("EASY_14", ["beta-adrenergic", "beta blocker", "propranolol", "palpitations"], "ata_2016_hyper"),
    ("EASY_15", ["10.0", "10 mIU/L", "subclinical", "levothyroxine", "indicated"], "ata_aace_2012_hypo"),
]

for qid, kws, doc in queries_to_curate:
    print(f"\n========================================================")
    print(f"[{qid}] Keywords: {kws} in {doc}")
    results = search_text(kws, doc_filter=doc, max_res=3)
    for score, cid, page, text in results:
        print(f"--- MATCH {cid} (p.{page}, score={score}) ---")
        print(text[:300].replace('\n', ' '))
