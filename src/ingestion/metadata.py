import hashlib
import os
from typing import Dict, Any

class MetadataNormalizer:
    """Normalizes document metadata and computes cryptographic verification hashes."""
    @staticmethod
    def compute_sha256(file_path: str) -> str:
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()

    @staticmethod
    def build_doc_meta(doc_slug: str, title: str, authoring_body: str, year: int, category: str, subtopic: str, relative_path: str, page_count: int, file_size_kb: float, sha256_hash: str) -> Dict[str, Any]:
        return {
            "doc_slug": doc_slug,
            "filename": os.path.basename(relative_path),
            "relative_path": relative_path,
            "title": title,
            "authoring_body": authoring_body,
            "year": year,
            "category": category,
            "subtopic": subtopic,
            "page_count": page_count,
            "file_size_kb": file_size_kb,
            "sha256": sha256_hash
        }
