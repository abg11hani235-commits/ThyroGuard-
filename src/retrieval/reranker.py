"""
CrossEncoderReranker — Neural precision reranker for ThyroGuard.

Primary model  : BAAI/bge-reranker-v2-m3 (multilingual, clinically superior)
Fallback model : cross-encoder/ms-marco-MiniLM-L-6-v2 (CPU-friendly)

The reranker reads its target model from config/models.yaml, attempts to
load the primary model, and falls back transparently.

Dosage intent enrichment: when the incoming query contains dose/prescription
vocabulary (Arabic or English), the scoring query is enriched to help the
cross-encoder surface chunks containing numeric dosage recommendations.
"""

import os
import re
import yaml
import logging
from typing import List, Tuple, Dict, Optional
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dosage-intent keyword sets for scoring query enrichment
# ---------------------------------------------------------------------------
_DOSAGE_TERMS_AR = {"جرعة", "جرعه", "بدء", "مقدار", "كمية", "يومي", "يومية", "اليومي"}
_DOSAGE_TERMS_EN = {"dose", "dosage", "initiating", "starting", "initial", "daily", "mcg"}


def _load_reranker_config() -> Dict[str, str]:
    """Reads reranker model settings from config/models.yaml."""
    try:
        cfg_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "config", "models.yaml")
        )
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            retrieval = cfg.get("retrieval", {})
            models = cfg.get("models", {}).get("reranker", {})
            return {
                "primary": retrieval.get("reranker_model")
                           or models.get("primary", "BAAI/bge-reranker-v2-m3"),
                "fallback": retrieval.get("reranker_fallback")
                            or models.get("fallback", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
                "max_length": str(models.get("max_length", 1024)),
            }
    except Exception as e:
        logger.warning(f"Could not read reranker config: {e}")
    return {
        "primary": "BAAI/bge-reranker-v2-m3",
        "fallback": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "max_length": "1024",
    }


def _enrich_dosage_query(query: str) -> str:
    """
    Appends standard dosage vocabulary to the scoring query when the user
    query contains dose/prescription intent keywords.  This guides the
    cross-encoder to favour chunks containing specific numeric dosage values.
    """
    q_lower = query.lower()
    tokens = set(re.findall(r"[\w\u0600-\u06FF]+", q_lower))
    if tokens & _DOSAGE_TERMS_AR or tokens & _DOSAGE_TERMS_EN:
        return (
            f"{query} recommended starting dose dosage mcg kg daily "
            "levothyroxine initial replacement"
        )
    return query


class CrossEncoderReranker:
    """
    Neural cross-encoder reranker for precision candidate scoring.
    Supports SOTA multilingual clinical rerankers (BAAI/bge-reranker-v2-m3)
    with graceful fallback to lightweight cross-encoders (ms-marco-MiniLM-L-6-v2).
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        fallback_model: Optional[str] = None,
        corpus_lookup: Optional[Dict[str, str]] = None,
    ):
        self.corpus_lookup = corpus_lookup or {}

        cfg = _load_reranker_config()
        primary = model_name or cfg["primary"]
        fallback = fallback_model or cfg["fallback"]
        max_length = int(cfg["max_length"])

        # Device selection
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

        self.device = device
        self.active_model_name = primary
        print(f"[Reranker] Enforcing Primary SOTA Model: {self.active_model_name} on device: {self.device}")
        
        # Load reranker with graceful fallback and pass-through protection
        self.model = None
        # 1. Try loading primary model locally first to prevent hanging on blocked/throttled networks
        try:
            self.model = CrossEncoder(
                self.active_model_name,
                max_length=max_length,
                device=self.device,
                local_files_only=True
            )
            print(f"[Reranker] {self.active_model_name} loaded successfully from local cache.")
        except Exception:
            # 2. Try loading primary online if possible
            try:
                self.model = CrossEncoder(
                    self.active_model_name,
                    max_length=max_length,
                    device=self.device
                )
                print(f"[Reranker] {self.active_model_name} loaded successfully.")
            except Exception as e:
                logger.warning(f"[Reranker] Could not load primary model '{self.active_model_name}': {e}")
                if fallback and fallback != self.active_model_name:
                    try:
                        self.model = CrossEncoder(fallback, max_length=max_length, device=self.device, local_files_only=True)
                        print(f"[Reranker] Fallback model '{fallback}' loaded from local cache.")
                    except Exception:
                        try:
                            self.model = CrossEncoder(fallback, max_length=max_length, device=self.device)
                            print(f"[Reranker] Fallback model '{fallback}' loaded successfully.")
                        except Exception as e2:
                            logger.warning(f"[Reranker] Fallback also failed: {e2}. Operating in hybrid pass-through mode.")
                            self.model = None
                else:
                    print("[Reranker] Operating in hybrid pass-through mode (BM25 + Dense RRF ranking).")

    def set_corpus_lookup(self, lookup: Dict[str, str]) -> None:
        self.corpus_lookup = lookup

    def rerank(
        self,
        query: str,
        candidate_ids: List[str],
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Reranks a list of chunk IDs by cross-encoder score.

        Args:
            query:         The user clinical query (Arabic or English).
            candidate_ids: Ordered list of chunk IDs from hybrid retrieval.
            top_k:         Number of top-scored chunks to return.

        Returns:
            List of (chunk_id, score) tuples, sorted descending by score.
        """
        if not candidate_ids:
            return []

        # Cap candidates to top-12 to maintain high precision without CPU latency spikes
        candidates_to_score = candidate_ids[:12]

        if self.model is None:
            # Fallback pass-through: preserve hybrid RRF ordering with synthetic score
            return [(cid, float(1.0 - idx * 0.05)) for idx, cid in enumerate(candidates_to_score[:top_k])]

        try:
            # Enrich query for dosage queries
            scoring_query = _enrich_dosage_query(query)

            pairs = [
                [scoring_query, (self.corpus_lookup.get(cid, "") or "")[:1500]]
                for cid in candidates_to_score
            ]
            scores = self.model.predict(pairs)
            scored = list(zip(candidates_to_score, [float(s) for s in scores]))
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]
        except Exception as e:
            logger.warning(f"[Reranker] Scoring error: {e}. Falling back to candidate order.")
            return [(cid, float(1.0 - idx * 0.05)) for idx, cid in enumerate(candidates_to_score[:top_k])]
