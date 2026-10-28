from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        description="Clinical question text",
        json_schema_extra={"example": "What is the recommended initial dose of levothyroxine in primary hypothyroidism?"}
    )
    top_k: Optional[int] = Field(5, description="Number of context chunks to retrieve", ge=1, le=15)


class CitationAudit(BaseModel):
    valid: bool
    citation_count: int
    valid_citations: List[str] = []
    hallucinated_citations: List[str] = []
    precision: float = 1.0


class DosageAudit(BaseModel):
    verified: bool
    detected_dosages: List[str] = []
    discrepant_dosages: List[str] = []
    is_safe: bool = True


class EnrichedChunk(BaseModel):
    chunk_id: str
    doc_slug: str
    page_number: int
    chunk_index: Optional[int] = None
    section_title: Optional[str] = "General Clinical Guidance"
    has_table: Optional[bool] = False
    char_offset_start: Optional[int] = 0
    char_offset_end: Optional[int] = 0
    text: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[str]
    retrieved_chunk_ids: List[str]
    citation_audit: CitationAudit
    safety_status: str
    status: str
    sub_queries: Optional[List[Dict[str, Any]]] = None
    dosage_audit: Optional[DosageAudit] = None
    claims: Optional[List[Dict[str, Any]]] = None
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None
    pipeline_trace: Optional[Dict[str, Any]] = None
    faithfulness_score: Optional[float] = None
    semantic_audit: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    corpus_chunks: int
    corpus_sha256: str
    cache_mode: Optional[str] = "in_memory"


class InventoryItem(BaseModel):
    doc_slug: str
    title: str
    authoring_body: str
    year: int
    category: str
    subtopic: str
    page_count: int
    sha256: str


class EvaluationSummary(BaseModel):
    benchmark_version: str
    metrics: Dict[str, Any]
