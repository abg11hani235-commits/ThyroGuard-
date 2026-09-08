import pytest
from src.retrieval.clinical_expander import CrossLingualClinicalExpander
from src.safety.semantic_verifier import SemanticClaimVerifier
from src.retrieval.reranker import CrossEncoderReranker

def test_cross_lingual_clinical_expander_dual_queries():
    expander = CrossLingualClinicalExpander()

    # Query with Arabic trade name
    raw_query = "ما هي جرعة الالتروكسين للبالغين؟"
    dual_q = expander.get_dual_queries(raw_query)

    assert dual_q["is_arabic"] is True
    # Sparse query must include trade names for BM25 matching
    assert "eltroxin" in dual_q["sparse_query"].lower() or "levothyroxine" in dual_q["sparse_query"].lower()
    
    # Dense query must be concise and avoid raw keyword stuffing
    assert len(dual_q["dense_query"]) > 10
    assert "eltroxin levothyroxine l-thyroxine t4" not in dual_q["dense_query"]  # No raw clutter

def test_cross_lingual_expander_arabic_normalization():
    expander = CrossLingualClinicalExpander()
    normalized = expander.normalize_arabic("إلتروكسينٌ بالغة")
    assert "التروكسين" in normalized
    assert "بالغه" in normalized

def test_semantic_verifier_entailed_claim():
    verifier = SemanticClaimVerifier(enable_nli=False)
    
    sample_chunk = {
        "chunk_id": "ata_2014::p12::c01",
        "text": "The recommended initial daily dose of levothyroxine in young healthy adults with overt primary hypothyroidism is approximately 1.6 mcg/kg/day."
    }
    claims = [{
        "claim": "The initial levothyroxine starting dose is 1.6 mcg/kg/day.",
        "citation": "ata_2014::p12::c01"
    }]
    
    res = verifier.verify_generation(
        generated_text="The initial levothyroxine starting dose is 1.6 mcg/kg/day.",
        claims=claims,
        retrieved_chunks=[sample_chunk]
    )
    
    assert res["is_safe"] is True
    assert res["faithfulness_score"] == 100.0
    assert len(res["verified_claims"]) == 1
    assert len(res["hallucinated_claims"]) == 0
    assert res["policy_action"] == "passed"

def test_semantic_verifier_contradiction_rejection_policy():
    verifier = SemanticClaimVerifier(enable_nli=False)
    
    sample_chunk = {
        "chunk_id": "ata_2014::p12::c01",
        "text": "The recommended initial daily dose of levothyroxine in young healthy adults with overt primary hypothyroidism is approximately 1.6 mcg/kg/day."
    }
    # Notice: 5.0 mcg/kg is a fabricated/hallucinated number not in the chunk
    claims = [{
        "claim": "The recommended initial dose of levothyroxine is 5.0 mcg/kg/day.",
        "citation": "ata_2014::p12::c01"
    }]
    
    res = verifier.verify_generation(
        generated_text="The recommended initial dose of levothyroxine is 5.0 mcg/kg/day.",
        claims=claims,
        retrieved_chunks=[sample_chunk]
    )
    
    assert res["is_safe"] is False
    assert res["faithfulness_score"] == 0.0
    assert len(res["hallucinated_claims"]) == 1
    assert res["policy_action"] == "contradiction_warning_appended"
    assert "⚠️" in res["sanitized_answer"]

def test_cross_encoder_reranker_fallback():
    # Test that CrossEncoderReranker initializes and reranks correctly with fallback
    reranker = CrossEncoderReranker(
        model_name="nonexistent/model-name-for-test",
        fallback_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        corpus_lookup={"c1": "Levothyroxine dosing", "c2": "Thyroid cancer surgery"}
    )
    assert reranker.active_model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    ranked = reranker.rerank("levothyroxine dose", ["c1", "c2"], top_k=2)
    assert len(ranked) == 2
    assert ranked[0][0] == "c1"  # c1 is more relevant to dosing
