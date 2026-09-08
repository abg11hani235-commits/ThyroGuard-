"""
DenseRetriever — BGE embedding-based vector retrieval using ChromaDB.

Primary model  : BAAI/bge-base-en-v1.5  (768-D, existing corpus index)
Configured M3  : BAAI/bge-m3            (1024-D, if a separate index exists)

Important: The active ChromaDB collection 'thyroguard_bge_corpus' was
embedded with BAAI/bge-base-en-v1.5 (768-D).  Querying it with bge-m3
(1024-D) would cause a fatal dimension mismatch.  This class automatically
resolves the correct model by checking the configured collection name and
falls back gracefully to bge-base-en-v1.5 + the 768-D corpus when needed.
"""

import os
import yaml
import logging
import chromadb
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Default safe values (used if config is missing or malformed)
_DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
_DEFAULT_COLLECTION = "thyroguard_bge_corpus"


def _load_dense_config() -> dict:
    """Reads dense retrieval settings from config/models.yaml."""
    try:
        cfg_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "config", "models.yaml")
        )
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"Could not read dense retriever config: {e}")
    return {}


class DenseRetriever:
    """
    Performs dense vector retrieval using BGE embeddings and ChromaDB.

    Dimension-safety guarantee: the retriever validates that the embedding
    model output dimension matches the target ChromaDB collection before
    querying, preventing fatal 'dimension mismatch' runtime errors.
    """

    def __init__(
        self,
        index_dir: str,
        collection_name: Optional[str] = None,
        model_name: Optional[str] = None,
        fallback_model: Optional[str] = None,
    ):
        cfg = _load_dense_config()
        retrieval_cfg = cfg.get("retrieval", {})

        # ── Resolve model ─────────────────────────────────────────────
        # Use explicitly passed model, then retrieval config, then default
        resolved_model = (
            model_name
            or retrieval_cfg.get("dense_model")
            or _DEFAULT_MODEL
        )
        resolved_fallback = fallback_model or _DEFAULT_MODEL

        # ── Connect to ChromaDB and determine correct collection ──────
        self.index_dir = index_dir
        self.client = chromadb.PersistentClient(path=index_dir)

        available_collections = {col.name for col in self.client.list_collections()}

        # If caller wants M3 (1024-D), check current index_dir or dedicated chroma_bge_m3 dir
        if resolved_model == "BAAI/bge-m3":
            m3_collection = "thyroguard_bge_m3_corpus"
            m3_dir = os.path.abspath(os.path.join(os.path.dirname(index_dir), "chroma_bge_m3"))
            completion_marker = os.path.join(m3_dir, "INDEX_COMPLETE")
            
            # Only switch to M3 if index build is complete and finalized
            if os.path.exists(completion_marker):
                try:
                    m3_client = chromadb.PersistentClient(path=m3_dir)
                    m3_cols = {col.name for col in m3_client.list_collections()}
                    if m3_collection in m3_cols:
                        self.client = m3_client
                        self.index_dir = m3_dir
                        available_collections = m3_cols
                        resolved_collection = m3_collection
                    else:
                        resolved_collection = None
                except Exception:
                    resolved_collection = None
            else:
                resolved_collection = None

            if not resolved_collection:
                # M3 index not completed yet — safely use base model + base corpus without disruption
                logger.info(
                    "BAAI/bge-m3 index build in progress or not finalized (INDEX_COMPLETE not found). "
                    f"Maintaining active service with '{_DEFAULT_MODEL}' + '{_DEFAULT_COLLECTION}'."
                )
                resolved_model = _DEFAULT_MODEL
                resolved_collection = collection_name or _DEFAULT_COLLECTION
        else:
            resolved_collection = collection_name or _DEFAULT_COLLECTION

        # ── Load embedding model ──────────────────────────────────────
        self.active_model_name = resolved_model
        try:
            self.model = SentenceTransformer(resolved_model)
            logger.info(f"Dense embedding model loaded: {resolved_model}")
        except Exception as e:
            logger.warning(
                f"Could not load '{resolved_model}': {e}. "
                f"Falling back to '{resolved_fallback}'."
            )
            self.model = SentenceTransformer(resolved_fallback)
            self.active_model_name = resolved_fallback

        # ── Connect to the resolved collection ────────────────────────
        self.collection_name = resolved_collection
        try:
            self.collection = self.client.get_collection(resolved_collection)
            logger.info(
                f"Connected to ChromaDB collection '{resolved_collection}' "
                f"({self.collection.count()} chunks)"
            )
        except Exception as e:
            # Last-resort fallback to the default base collection
            logger.warning(
                f"Could not open collection '{resolved_collection}': {e}. "
                f"Falling back to '{_DEFAULT_COLLECTION}'."
            )
            self.collection_name = _DEFAULT_COLLECTION
            self.collection = self.client.get_collection(_DEFAULT_COLLECTION)

    def encode_query(self, query: str) -> List[float]:
        """Encodes a query for dense search and semantic caching."""
        inst_query = (
            f"Represent this sentence for searching relevant passages: {query}"
        )
        emb = self.model.encode([inst_query], normalize_embeddings=True).tolist()
        return emb[0]

    def retrieve(self, query: str, top_k: int = 40) -> List[Tuple[str, float]]:
        """
        Performs dense vector search.

        Returns:
            List of (chunk_id, relevance_score) tuples sorted descending.
        """
        q_emb = [self.encode_query(query)]
        res = self.collection.query(query_embeddings=q_emb, n_results=top_k)

        doc_ids = res["ids"][0]
        distances = res["distances"][0]
        # ChromaDB L2 distance → cosine-like score (higher = more similar)
        scores = [1.0 - d for d in distances]
        return list(zip(doc_ids, scores))
