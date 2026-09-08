import unittest
from src.safety.citation_validator import CitationValidator

class TestCitations(unittest.TestCase):
    def test_valid_citations(self):
        text = "Initial dose is 1.6 mcg/kg [ata_2014_hypo::p17::c09] and monitoring is required at 6 weeks [ata_2014_hypo::p24::c02]."
        retrieved = ["ata_2014_hypo::p17::c09", "ata_2014_hypo::p24::c02", "other_chunk::p01::c01"]
        
        audit = CitationValidator.validate_citations(text, retrieved)
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["citation_count"], 2)
        self.assertEqual(len(audit["hallucinated_citations"]), 0)
        self.assertEqual(audit["precision"], 1.0)

    def test_hallucinated_citations(self):
        text = "Treatment is recommended [fake_guideline::p99::c01] based on studies."
        retrieved = ["ata_2014_hypo::p17::c09"]
        
        audit = CitationValidator.validate_citations(text, retrieved)
        self.assertFalse(audit["valid"])
        self.assertEqual(audit["citation_count"], 1)
        self.assertIn("fake_guideline::p99::c01", audit["hallucinated_citations"])
        self.assertEqual(audit["precision"], 0.0)

if __name__ == '__main__':
    unittest.main()
