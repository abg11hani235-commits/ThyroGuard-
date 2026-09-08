import unittest
from src.ingestion.chunker import StructureAwareChunker

class TestChunker(unittest.TestCase):
    def test_chunker_deterministic_ids(self):
        chunker = StructureAwareChunker(max_chars=300, overlap_chars=50)
        sample_text = (
            "Thyroid nodules are common clinical findings. Most nodules are benign colloid nodules or adenomas. "
            "However, high-suspicion sonographic patterns warrant fine-needle aspiration biopsy when measuring 1.0 cm or larger. "
            "Diagnostic ultrasound remains the primary initial imaging modality."
        )
        chunks = chunker.chunk_page(sample_text, doc_slug="test_guideline", page_number=12)
        self.assertGreaterEqual(len(chunks), 1)
        for idx, c in enumerate(chunks):
            self.assertEqual(c["chunk_id"], f"test_guideline::p12::c{idx+1:02d}")
            self.assertEqual(c["page_number"], 12)
            self.assertGreater(len(c["text"]), 0)

    def test_chunker_empty_handling(self):
        chunker = StructureAwareChunker()
        chunks = chunker.chunk_page("", doc_slug="empty", page_number=1)
        self.assertEqual(len(chunks), 0)

if __name__ == '__main__':
    unittest.main()
