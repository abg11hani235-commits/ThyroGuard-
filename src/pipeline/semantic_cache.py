import time
import os
import re
import json
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from threading import Lock

logger = logging.getLogger(__name__)

class ClinicalSemanticCache:
    """
    High-performance semantic cache for clinical RAG queries.
    Prevents redundant LLM and retrieval executions on identical or semantically
    equivalent clinical queries.
    
    Features:
      - Tier 1: Instant O(1) exact normalized match (<1ms) with in-memory + Redis lookup.
      - Tier 2: Dense cosine similarity semantic match (threshold >= 0.92) (<15ms).
      - Thread-safe LRU in-memory storage synced with Redis backend connectivity.
    """
    def __init__(
        self,
        similarity_threshold: float = 0.92,
        max_entries: int = 500,
        redis_url: Optional[str] = None
    ):
        self.threshold = similarity_threshold
        self.max_entries = max_entries
        self.lock = Lock()
        
        # In-memory storage: query_text -> (response_payload, embedding_vector, timestamp)
        self.cache: Dict[str, Tuple[Dict[str, Any], Optional[np.ndarray], float]] = {}
        
        # Optional Redis client
        self.redis_client = None
        target_redis = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        if target_redis:
            try:
                import redis
                self.redis_client = redis.from_url(target_redis, decode_responses=True, socket_timeout=2)
                self.redis_client.ping()
                logger.info(f"[SemanticCache] Connected to Redis at {target_redis}")
            except Exception as e:
                logger.warning(f"[SemanticCache] Redis connection skipped/failed: {e}. Running in-memory only.")
                self.redis_client = None

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    @staticmethod
    def _normalize_key(text: str) -> str:
        """Normalizes clinical query for exact matching."""
        # Strip tashkeel, excess spaces, and punctuation
        t = re.sub(r'[\u0617-\u061A\u064B-\u0652\u0640]', '', text.strip().lower())
        t = re.sub(r'[!?,;؟،.]', '', t)
        return " ".join(t.split())

    def get(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieves cached response if an exact match or cosine similarity >= threshold is found.
        """
        t0 = time.time()
        norm_key = self._normalize_key(query)

        with self.lock:
            # 1. Tier 1: Exact normalized in-memory match (O(1))
            if norm_key in self.cache:
                resp, _, _ = self.cache[norm_key]
                hit_latency = max(1, round((time.time() - t0) * 1000))
                cached_copy = dict(resp)
                cached_copy["cached"] = True
                cached_copy["cache_hit_type"] = "exact_memory"
                cached_copy["cache_similarity"] = 1.0
                cached_copy["cache_latency_ms"] = hit_latency
                return cached_copy

            # 1b. Tier 1b: Exact match from Redis
            if self.redis_client is not None:
                try:
                    redis_key = f"thyroguard:cache:{norm_key}"
                    raw_val = self.redis_client.get(redis_key)
                    if raw_val:
                        data = json.loads(raw_val)
                        resp = data.get("response", {})
                        emb = np.array(data["embedding"], dtype=np.float32) if data.get("embedding") else None
                        self.cache[norm_key] = (resp, emb, time.time())
                        hit_latency = max(1, round((time.time() - t0) * 1000))
                        cached_copy = dict(resp)
                        cached_copy["cached"] = True
                        cached_copy["cache_hit_type"] = "exact_redis"
                        cached_copy["cache_similarity"] = 1.0
                        cached_copy["cache_latency_ms"] = hit_latency
                        return cached_copy
                except Exception as e:
                    logger.debug(f"[SemanticCache] Redis get error: {e}")

            # 2. Tier 2: Semantic vector match (Cosine >= threshold)
            if query_embedding is not None and len(self.cache) > 0:
                q_vec = np.array(query_embedding, dtype=np.float32)
                best_sim = -1.0
                best_resp = None

                for k, (cached_resp, emb, _) in self.cache.items():
                    if emb is not None:
                        sim = self._cosine_similarity(q_vec, emb)
                        if sim > best_sim:
                            best_sim = sim
                            best_resp = cached_resp

                if best_sim >= self.threshold and best_resp is not None:
                    hit_latency = max(1, round((time.time() - t0) * 1000))
                    cached_copy = dict(best_resp)
                    cached_copy["cached"] = True
                    cached_copy["cache_hit_type"] = "semantic"
                    cached_copy["cache_similarity"] = round(float(best_sim), 3)
                    cached_copy["cache_latency_ms"] = hit_latency
                    return cached_copy

        return None

    def set(
        self,
        query: str,
        response: Dict[str, Any],
        query_embedding: Optional[List[float]] = None
    ) -> None:
        """Stores a clinical query and verified response into the cache."""
        norm_key = self._normalize_key(query)
        emb_array = np.array(query_embedding, dtype=np.float32) if query_embedding is not None else None

        with self.lock:
            # LRU eviction if capacity exceeded
            if len(self.cache) >= self.max_entries:
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][2])
                del self.cache[oldest_key]

            # Don't cache transient errors
            if response.get("status") in ["success", "emergency", "refusal", "greeting"]:
                self.cache[norm_key] = (dict(response), emb_array, time.time())

                # Persist to Redis
                if self.redis_client is not None:
                    try:
                        redis_key = f"thyroguard:cache:{norm_key}"
                        payload = {
                            "response": response,
                            "embedding": query_embedding,
                            "timestamp": time.time()
                        }
                        self.redis_client.setex(redis_key, 86400, json.dumps(payload))
                    except Exception as e:
                        logger.debug(f"[SemanticCache] Redis set error: {e}")

    def clear(self) -> None:
        with self.lock:
            self.cache.clear()
            if self.redis_client is not None:
                try:
                    self.redis_client.flushdb()
                except Exception as e:
                    logger.debug(f"[SemanticCache] Redis flush error: {e}")

    def size(self) -> int:
        with self.lock:
            return len(self.cache)
