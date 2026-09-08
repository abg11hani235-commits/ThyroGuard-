# ThyroGuard Scientific Generation & Safety Evaluation Report

Evaluation based on **80 partitioned evaluation queries** against the **Frozen Corpus** on **identical frozen retrieved contexts**.

## 1. Clinical Grounding & Citation Quality (60 In-Scope Queries)

| Metric | Score | Target | Interpretation |
|---|---:|---:|---|
| **In-Scope False Refusal Rate (FRR)** | **0.0%** | 0.0% | Zero legitimate thyroid queries rejected |
| **Claim-Level Faithfulness** | **89.5%** | >90.0% | High factual adherence to guideline evidence |
| **Citation Precision** | **100.0%** | 100.0% | All cited chunk IDs match retrieved evidence |
| **Citation Completeness / Recall** | **100.0%** | 100.0% | 100% of generated clinical claims cited |

## 2. Granular Safety & Refusal Metrics (20 Safety Queries)

| Safety Dimension | Evaluated Count | Refusal Recall | Unsafe Output Rate |
|---|---:|---:|---:|
| **Out-of-Scope Queries (OOS)** | 10 | **100.0%** | 0.0% |
| **Adversarial Traps & Prompt Injections** | 10 | **80.0%** | 0.0% |
| **Total Safety Gate Performance** | 20 | **90.0%** | **0.0%** |
