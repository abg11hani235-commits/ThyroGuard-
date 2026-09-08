import os
import json
import hashlib
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from .schemas import QueryRequest, QueryResponse, HealthResponse, InventoryItem, CitationAudit
from src.pipeline.rag_pipeline import ThyroGuardPipeline

# Load Settings
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "settings.yaml")
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

CORPUS_PATH = os.path.join(PROJECT_ROOT, config["paths"]["corpus"])
INVENTORY_PATH = os.path.join(PROJECT_ROOT, config["paths"]["inventory"])
CHROMA_DIR = os.path.join(PROJECT_ROOT, config["paths"]["chroma_index_dir"])

# Initialize FastAPI App
app = FastAPI(
    title=config["app"]["name"],
    version=config["app"]["version"],
    description="Evidence-Grounded Clinical RAG for Thyroid Endocrinology"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

if os.path.exists(FRONTEND_DIR):
    app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

@app.get("/", include_in_schema=False)
def root_redirect():
    if os.path.exists(FRONTEND_DIR):
        return RedirectResponse(url="/app/")
    return {"message": "ThyroGuard Clinical RAG API is active. Go to /docs for Swagger UI."}

# Global Pipeline Singleton
pipeline = None

@app.on_event("startup")
def startup_event():
    global pipeline
    print(f"Initializing {config['app']['name']} Pipeline...")
    pipeline = ThyroGuardPipeline(
        corpus_path=CORPUS_PATH,
        chroma_dir=CHROMA_DIR
    )
    print("ThyroGuard Pipeline ready for queries.")

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    if not os.path.exists(CORPUS_PATH):
        raise HTTPException(status_code=500, detail="Corpus not found")
    with open(CORPUS_PATH, "rb") as f:
        corpus_hash = hashlib.sha256(f.read()).hexdigest()
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return {
        "status": "healthy",
        "service": config["app"]["name"],
        "version": config["app"]["version"],
        "corpus_chunks": len(chunks),
        "corpus_sha256": corpus_hash
    }

@app.post("/query", response_model=QueryResponse, tags=["Clinical RAG"])
def clinical_query(request: QueryRequest):
    global pipeline
    if pipeline is None:
        pipeline = ThyroGuardPipeline(corpus_path=CORPUS_PATH, chroma_dir=CHROMA_DIR)

    res = pipeline.query(request.query, top_k=request.top_k)
    cit_audit = res.get("citation_audit", {
        "valid": True, "citation_count": 0, "valid_citations": [], "hallucinated_citations": [], "precision": 1.0
    })

    return {
        "query": request.query,
        "answer": res["answer"],
        "citations": res["citations"],
        "claims": res.get("claims", []),
        "retrieved_chunk_ids": res["retrieved_chunk_ids"],
        "retrieved_chunks": res.get("retrieved_chunks", []),
        "citation_audit": CitationAudit(**cit_audit),
        "safety_status": res["safety_status"],
        "status": res["status"],
        "pipeline_trace": res.get("pipeline_trace", None),
        "faithfulness_score": res.get("faithfulness_score", None),
        "semantic_audit": res.get("semantic_audit", None)
    }

@app.get("/inventory", response_model=list[InventoryItem], tags=["Data Foundation"])
def get_inventory():
    if not os.path.exists(INVENTORY_PATH):
        raise HTTPException(status_code=404, detail="Inventory file not found")
    with open(INVENTORY_PATH, "r", encoding="utf-8") as f:
        inventory = json.load(f)
    return inventory

@app.get("/evaluate", tags=["Benchmark"])
def get_benchmark_results():
    manifest_path = os.path.join(PROJECT_ROOT, "reports", "benchmark_manifest.json")
    retrieval_path = os.path.join(PROJECT_ROOT, "reports", "retrieval_ablation_final.json")
    generation_path = os.path.join(PROJECT_ROOT, "reports", "generation_evaluation_final.json")
    safety_path = os.path.join(PROJECT_ROOT, "reports", "safety_evaluation_final.json")

    results = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            results["manifest"] = json.load(f)
    if os.path.exists(retrieval_path):
        with open(retrieval_path, "r", encoding="utf-8") as f:
            results["retrieval_ablation"] = json.load(f)
    if os.path.exists(generation_path):
        with open(generation_path, "r", encoding="utf-8") as f:
            results["generation_evaluation"] = json.load(f)
    if os.path.exists(safety_path):
        with open(safety_path, "r", encoding="utf-8") as f:
            results["safety_evaluation"] = json.load(f)

    return results
