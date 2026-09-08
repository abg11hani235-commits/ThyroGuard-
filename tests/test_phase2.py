import pytest
import numpy as np
from src.pipeline.semantic_cache import ClinicalSemanticCache
from src.ingestion.loader import SmartClinicalLoader, PDFDocumentLoader

def test_semantic_cache_exact_hit():
    cache = ClinicalSemanticCache(similarity_threshold=0.95)
    sample_response = {"answer": "Starting dose is 1.6 mcg/kg", "status": "success"}
    
    # Store in cache
    cache.set("ما هي جرعة الالتروكسين للبالغين؟", sample_response)
    
    # Exact hit with slightly different spacing or punctuation
    hit = cache.get("ما هي جرعة الالتروكسين للبالغين؟")
    assert hit is not None
    assert hit["cached"] is True
    assert hit["cache_hit_type"] == "exact"
    assert hit["answer"] == "Starting dose is 1.6 mcg/kg"

def test_semantic_cache_cosine_similarity_hit():
    cache = ClinicalSemanticCache(similarity_threshold=0.95)
    sample_response = {"answer": "TSH cutoff is 4.5 mIU/L", "status": "success"}
    
    # Simulated 4-dimensional normalized vectors
    base_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    # 0.995 cosine similarity vector
    similar_vec = np.array([0.995, 0.01, 0.0, 0.0], dtype=np.float32)
    
    cache.set("ما هو الحد الطبيعي لـ TSH؟", sample_response, query_embedding=base_vec.tolist())
    
    # Query with similar embedding vector
    hit = cache.get("ما هي النسبة الطبيعية للـ TSH؟", query_embedding=similar_vec.tolist())
    assert hit is not None
    assert hit["cached"] is True
    assert hit["cache_hit_type"] == "semantic"
    assert hit["cache_similarity"] >= 0.95

def test_semantic_cache_miss_below_threshold():
    cache = ClinicalSemanticCache(similarity_threshold=0.95)
    sample_response = {"answer": "Thyroid storm emergency protocol", "status": "emergency"}
    
    base_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    # Orthogonal/low similarity vector (similarity = 0.0)
    different_vec = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    
    cache.set("عاصفة درقية", sample_response, query_embedding=base_vec.tolist())
    
    hit = cache.get("سرطان الغدة", query_embedding=different_vec.tolist())
    assert hit is None

def test_semantic_cache_lru_eviction():
    cache = ClinicalSemanticCache(max_entries=2)
    cache.set("q1", {"answer": "a1", "status": "success"})
    cache.set("q2", {"answer": "a2", "status": "success"})
    assert cache.size() == 2
    
    # Adding third entry should evict oldest
    cache.set("q3", {"answer": "a3", "status": "success"})
    assert cache.size() == 2
    assert cache.get("q1") is None
    assert cache.get("q3") is not None

def test_smart_clinical_loader_fallback():
    # Test that loader handles existing PDF or non-existent file cleanly
    with pytest.raises(FileNotFoundError):
        SmartClinicalLoader("nonexistent_path.pdf")
