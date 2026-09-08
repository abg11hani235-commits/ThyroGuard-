import re
from typing import List, Dict, Any

class StructureAwareChunker:
    """Performs deterministic structure-aware chunking preserving sentences and section flow."""
    def __init__(self, max_chars: int = 900, overlap_chars: int = 120):
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = text.replace('\xa0', ' ').replace('\u200b', ' ')
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def chunk_page(self, text: str, doc_slug: str, page_number: int, base_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        cleaned = self.clean_text(text)
        if len(cleaned) < 50:
            return []

        sentences = re.split(r'(?<=[.!?])\s+|\n\n+', cleaned)
        chunks = []
        current_chunk = []
        current_len = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sent_len = len(sentence)
            if current_len + sent_len > self.max_chars and current_chunk:
                chunk_str = " ".join(current_chunk).strip()
                if chunk_str:
                    chunks.append(chunk_str)

                # Overlap
                overlap_list = []
                overlap_len = 0
                for prev_sent in reversed(current_chunk):
                    if overlap_len + len(prev_sent) <= self.overlap_chars:
                        overlap_list.insert(0, prev_sent)
                        overlap_len += len(prev_sent)
                    else:
                        break
                current_chunk = overlap_list + [sentence]
                current_len = sum(len(s) for s in current_chunk) + len(current_chunk)
            else:
                current_chunk.append(sentence)
                current_len += sent_len + 1

        if current_chunk:
            chunk_str = " ".join(current_chunk).strip()
            if chunk_str:
                chunks.append(chunk_str)

        # Build structured chunk objects
        chunk_objects = []
        base_meta = base_metadata or {}
        for idx, chunk_text in enumerate(chunks):
            chunk_id = f"{doc_slug}::p{page_number:02d}::c{idx+1:02d}"
            chunk_obj = {
                "chunk_id": chunk_id,
                "doc_slug": doc_slug,
                "page_number": page_number,
                "chunk_index": idx + 1,
                "char_count": len(chunk_text),
                "word_count": len(chunk_text.split()),
                "text": chunk_text,
                **base_meta
            }
            chunk_objects.append(chunk_obj)

        return chunk_objects
