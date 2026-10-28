import re
from typing import List, Dict, Set, Any

class CitationValidator:
    """Validates that all cited chunk IDs exist within the retrieved context chunks."""
    @staticmethod
    def extract_citations_from_text(text: str) -> List[str]:
        return re.findall(r'\[([a-zA-Z0-9_-]+::p\d+::c\d+)\]', text)

    @classmethod
    def validate_citations(cls, text: str, retrieved_chunk_ids: List[str]) -> Dict[str, Any]:
        found_citations = cls.extract_citations_from_text(text)
        retrieved_set = set(retrieved_chunk_ids)

        if not found_citations:
            return {
                "valid": True,
                "citation_count": 0,
                "valid_citations": [],
                "hallucinated_citations": [],
                "precision": 1.0
            }

        valid_citations = [c for c in found_citations if c in retrieved_set]
        invalid_citations = [c for c in found_citations if c not in retrieved_set]

        precision = len(valid_citations) / len(found_citations) if found_citations else 1.0

        return {
            "valid": len(invalid_citations) == 0,
            "citation_count": len(found_citations),
            "valid_citations": valid_citations,
            "hallucinated_citations": invalid_citations,
            "precision": precision
        }
