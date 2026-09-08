import json
import hashlib
import os

CHUNKS_PATH = 'd:/RAG/data/processed/chunks.json'
FROZEN_CORPUS_PATH = 'd:/RAG/data/processed/FROZEN_CORPUS.json'
GOLD_DATASET_PATH = 'd:/RAG/data/evaluation/gold_evaluation_dataset.json'

# 1. Freeze Corpus
print("Freezing Corpus...")
with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
    chunks = json.load(f)

with open(FROZEN_CORPUS_PATH, 'w', encoding='utf-8') as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)

with open(FROZEN_CORPUS_PATH, 'rb') as f:
    corpus_sha256 = hashlib.sha256(f.read()).hexdigest()

print(f"FROZEN CORPUS created: {len(chunks)} chunks, SHA256 = {corpus_sha256}")

# 2. Enrich Gold Evaluation Dataset with `gold_claims`
with open(GOLD_DATASET_PATH, 'r', encoding='utf-8') as f:
    gold_data = json.load(f)

# Define claim decomposition map for the 60 clinical queries
claim_definitions = {
    "EASY_01": [
        {"claim": "The initial full replacement dosage of levothyroxine is approximately 1.6 mcg/kg body weight per day in non-elderly adults with overt hypothyroidism.", "supporting_chunk_ids": ["ata_2014_hypo::p17::c09", "ata_2014_hypo::p18::c08"]},
        {"claim": "Dosing should be based on ideal body weight or lean body mass in overweight patients.", "supporting_chunk_ids": ["ata_2014_hypo::p17::c09"]}
    ],
    "EASY_02": [
        {"claim": "Serum TSH measurement is the single most sensitive first-line screening test for suspected primary thyroid dysfunction in ambulatory patients.", "supporting_chunk_ids": ["ata_aace_2012_hypo::p03::c03", "ata_aace_2012_hypo::p12::c04"]}
    ],
    "EASY_03": [
        {"claim": "Methimazole (MMI) is preferred as first-line medical therapy over PTU for virtually all non-pregnant adult patients with Graves' disease.", "supporting_chunk_ids": ["ata_2016_hyper::p09::c03", "ata_2016_hyper::p13::c04"]},
        {"claim": "Methimazole has a lower risk of severe, potentially fatal hepatotoxicity compared to PTU.", "supporting_chunk_ids": ["ata_2016_hyper::p13::c04"]}
    ],
    "EASY_04": [
        {"claim": "Fine-needle aspiration (FNA) is recommended for thyroid nodules with a high-suspicion ultrasound pattern measuring 1.0 cm or greater.", "supporting_chunk_ids": ["ata_2015_nodules_dtc::p14::c03", "ata_2015_nodules_dtc::p15::c02"]}
    ],
    "EASY_05": [
        {"claim": "Serum TSH should be re-evaluated 4 to 8 weeks (commonly 6-8 weeks) following initiation or dosage adjustment of levothyroxine.", "supporting_chunk_ids": ["ata_2014_hypo::p24::c02", "ata_aace_2012_hypo::p12::c03"]}
    ],
    "EASY_06": [
        {"claim": "Subclinical hypothyroidism is defined biochemically as an elevated serum TSH with a normal free thyroxine (FT4) level.", "supporting_chunk_ids": ["ata_aace_2012_hypo::p03::c03", "ata_2014_hypo::p44::c01"]}
    ],
    "EASY_07": [
        {"claim": "Propylthiouracil (PTU) is the preferred antithyroid drug during the first trimester of pregnancy.", "supporting_chunk_ids": ["ata_2016_hyper::p46::c01"]},
        {"claim": "Methimazole is avoided in early pregnancy due to teratogenic risks of methimazole embryopathy.", "supporting_chunk_ids": ["ata_2016_hyper::p46::c01", "ata_2016_hyper::p46::c02"]}
    ],
    "EASY_08": [
        {"claim": "Levothyroxine should be taken consistently 30 to 60 minutes before breakfast with water on an empty stomach, or at bedtime at least 3-4 hours after food.", "supporting_chunk_ids": ["ata_2014_hypo::p25::c01", "ata_2014_hypo::p25::c02"]}
    ],
    "EASY_09": [
        {"claim": "All Anaplastic Thyroid Cancers (ATC) are staged as Stage IV disease at initial presentation.", "supporting_chunk_ids": ["ata_2020_atc::p03::c01", "review_2021_thyroid_cancer::p14::c02"]}
    ],
    "EASY_10": [
        {"claim": "Papillary Thyroid Carcinoma (PTC) is the most common histological subtype, representing 80% to 90% of all thyroid cancers.", "supporting_chunk_ids": ["review_2021_thyroid_cancer::p03::c04", "review_2021_thyroid_cancer::p04::c01"]}
    ],
    "EASY_11": [
        {"claim": "Agranulocytosis is a critical complication requiring immediate cessation of ATDs and an urgent complete blood count with differential upon developing fever or sore throat.", "supporting_chunk_ids": ["ata_2016_hyper::p13::c04", "ata_2016_hyper::p14::c04"]}
    ],
    "EASY_12": [
        {"claim": "Purely cystic thyroid nodules have a malignancy risk of <1% and routine diagnostic FNA is not recommended.", "supporting_chunk_ids": ["ata_2015_nodules_dtc::p14::c03", "ata_2015_nodules_dtc::p15::c08"]}
    ],
    "EASY_13": [
        {"claim": "Serum Thyroglobulin (Tg) and anti-thyroglobulin antibodies (TgAb) serve as primary surveillance tumor markers following total thyroidectomy.", "supporting_chunk_ids": ["ata_2015_nodules_dtc::p62::c01", "ata_2015_nodules_dtc::p63::c01"]}
    ],
    "EASY_14": [
        {"claim": "Beta-adrenergic blockers (propranolol/atenolol) are recommended for rapid symptomatic control of tachycardia and tremors in overt thyrotoxicosis.", "supporting_chunk_ids": ["ata_2016_hyper::p08::c03", "ata_2016_hyper::p29::c02"]}
    ],
    "EASY_15": [
        {"claim": "Levothyroxine therapy is recommended when serum TSH is persistently >= 10.0 mIU/L.", "supporting_chunk_ids": ["ata_aace_2012_hypo::p22::c02", "ata_aace_2012_hypo::p30::c03"]}
    ]
}

# Automatically populate claim definitions for remaining queries if not explicitly defined
for q in gold_data:
    qid = q['id']
    if qid in claim_definitions:
        q['gold_claims'] = claim_definitions[qid]
    elif q['gold_chunk_ids']:
        # Generate structured claims from the reference answer and gold chunks
        sentences = [s.strip() for s in q['reference_answer'].split('.') if len(s.strip()) > 15]
        q['gold_claims'] = [
            {
                "claim": s + ".",
                "supporting_chunk_ids": q['gold_chunk_ids']
            } for s in sentences[:3]
        ]
    else:
        q['gold_claims'] = []

with open(GOLD_DATASET_PATH, 'w', encoding='utf-8') as f:
    json.dump(gold_data, f, indent=2, ensure_ascii=False)

print(f"Successfully enriched {len(gold_data)} queries with `gold_claims`.")
