import unittest
from fastapi.testclient import TestClient
from app.api import app

class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_api_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["corpus_chunks"], 3073)
        self.assertEqual(len(data["corpus_sha256"]), 64)

    def test_api_inventory(self):
        response = self.client.get("/inventory")
        self.assertEqual(response.status_code, 200)
        inventory = response.json()
        self.assertEqual(len(inventory), 8)

    def test_api_query_out_of_scope(self):
        response = self.client.post("/query", json={"query": "How do I treat severe bacterial pneumonia?"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "refusal")
        self.assertIn("REFUSAL", data["answer"])

if __name__ == '__main__':
    unittest.main()
