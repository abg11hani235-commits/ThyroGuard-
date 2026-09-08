"""
GroundedGenerator — ThyroGuard Clinical Answer Synthesis Engine.

Produces evidence-grounded clinical responses in Arabic or English via:
  1. Qwen/Groq API (primary, when GROQ_API_KEY is set and network is available)
  2. Deterministic local grounded synthesis (fallback, always available offline)

The generator is deliberately NOT a security boundary — all safety and refusal
logic is enforced upstream by ClinicalGuardrails (pipeline stage 1).
"""

import os
import re
import logging
import yaml
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional httpx import (graceful if not installed, falls back to local)
# ---------------------------------------------------------------------------
try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False

from .prompt_templates import CLINICAL_SYSTEM_PROMPT, ClinicalPromptTemplates


def _load_generation_config() -> Dict[str, Any]:
    """Reads generation settings from config/models.yaml."""
    try:
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "config", "models.yaml"
        )
        config_path = os.path.abspath(config_path)
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            return cfg.get("models", {}).get("generation", {})
    except Exception as e:
        logger.warning(f"Could not load generation config: {e}")
    return {}


def clean_reasoning_output(raw_text: str) -> str:
    """
    Strips internal thinking process, chain-of-thought blocks, and <think>...</think> tags.
    Ensures that only the direct clinical synthesis and recommendations are delivered.
    """
    if not raw_text:
        return ""

    text = raw_text

    # 1. Strip any content inside <think>...</think> tags
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    # 2. Check if the output contains "Here's a thinking process:" or similar chain-of-thought prefix
    thinking_match = re.search(r"Here's a thinking process|Thinking Process|Chain of thought", text, flags=re.IGNORECASE)
    if thinking_match:
        # Markers where the final clinical answer begins
        markers = [
            r"(?:(?:\n|\A)(?:#+\s*)?(?:Final\s+(?:Polish\s+&\s+)?(?:Output|Answer|Response)(?:\s+Generation)?|Direct\s+Answer|Clinical\s+Recommendation|Recommendation|Synthesis)[\s:]*)",
            r"(?:(?:\n|\A)(?:Based\s+on\s+(?:the\s+)?(?:provided\s+)?(?:clinical\s+practice\s+)?(?:ATA|ETA|AACE|guidelines?)[^:\n]*:?))",
            r"(?:(?:\n|\A)(?:بناءً\s+على\s+(?:الأدلة\s+الإرشادية\s+السريرية|إرشادات|توصيات)[^:\n]*:?))",
            r"(?:(?:\n|\A)(?:وفقاً\s+لإرشادات[^:\n]*:?))",
        ]
        extracted = None
        for m in markers:
            matches = list(re.finditer(m, text, flags=re.IGNORECASE))
            if matches:
                valid_matches = [match for match in matches if match.start() >= thinking_match.start()]
                if valid_matches:
                    target_match = valid_matches[-1]
                    extracted = text[target_match.start():].strip()
                    break

        if extracted:
            text = extracted
        else:
            # Check if there is a "Synthesize Answer" or "Draft" section
            synth_match = re.search(r"(?:Synthesize\s+Answer|Mental\s+Draft)[^:\n]*:?\s*", text, flags=re.IGNORECASE)
            if synth_match:
                sub = text[synth_match.end():]
                sub = re.sub(r"\n\s*\d+\.\s*(?:Check\s+Constraints|Final\s+Polish|Review|Validate).*$", "", sub, flags=re.DOTALL | re.IGNORECASE)
                text = sub.strip()
            else:
                # Regex fallback: strip everything up to "Based on" or bullet points
                text = re.sub(r"Here's a thinking process:.*?(?=(?:Based on|According to|In |For |When |بناءً على|\*\*))", '', text, flags=re.DOTALL | re.IGNORECASE)

    # 3. Clean up any remaining leading "Here's a thinking process:" header lines
    text = re.sub(r"^\s*Here's a thinking process:?\s*", '', text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*(?:Direct\s+Answer|Final\s+(?:Polish\s+&\s+)?(?:Output|Answer)):\s*", '', text, flags=re.IGNORECASE)

    return text.strip()


class GroundedGenerator:
    """
    Generates structured, evidence-grounded clinical responses in Arabic or English.

    Primary path  : Qwen via Groq API (requires GROQ_API_KEY env var)
    Fallback path : Deterministic local grounded synthesis (no network required)
    """

    def __init__(
        self,
        backend: str = "qwen",
        model_name: Optional[str] = None,
        api_endpoint: Optional[str] = None,
    ):
        cfg = _load_generation_config()

        self.backend = backend or cfg.get("primary_backend", "qwen")
        self.model_name = (
            model_name
            or cfg.get("model_name", "qwen/qwen3.6-27b")
        )
        self.api_endpoint = (
            api_endpoint
            or cfg.get("api_endpoint", "https://api.groq.com/openai/v1/chat/completions")
        )
        self.temperature = float(cfg.get("temperature", 0.1))
        self.max_tokens = int(cfg.get("max_tokens", 1200))
        self.api_key = os.getenv("GROQ_API_KEY", "")
        if not self.api_key:
            raise ValueError("[Generator Error] GROQ_API_KEY is missing! Strict Mode requires Groq to run Qwen.")
        
        print(f"[Generator] Enforcing Generative LLM: {self.model_name} via Groq")

    # ------------------------------------------------------------------
    # Public interface (compatible with existing pipeline call signature)
    # ------------------------------------------------------------------

    def generate(
        self, query: str, retrieved_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Strict generative synthesis via Qwen on Groq.
        """
        is_arabic = self._is_arabic(query)

        if not retrieved_chunks:
            if is_arabic:
                ans = (
                    "الأدلة الإرشادية السريرية المتوفرة غير كافية لتقديم إجابة حاسمة "
                    "على هذا السؤال."
                )
            else:
                ans = (
                    "The provided clinical guideline evidence is insufficient to answer "
                    "this query definitively."
                )
            return {
                "answer": ans,
                "citations": [],
                "claims": [],
                "status": "insufficient_evidence",
            }

        # ── Strict Qwen/Groq API Execution (No Local Fallback) ──────
        raw_api_answer = self._call_groq_api(query, retrieved_chunks, is_arabic)
        api_answer = clean_reasoning_output(raw_api_answer)
        citations = self._extract_citations(api_answer)
        claims = self._extract_claims(api_answer, citations)
        return {
            "answer": api_answer,
            "citations": citations,
            "claims": claims,
            "status": "grounded_qwen",
        }

    # ------------------------------------------------------------------
    # Primary path: Qwen via Groq
    # ------------------------------------------------------------------

    def _call_groq_api(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        is_arabic: bool,
    ) -> str:
        """
        Calls the Groq API for Qwen with strict Arabic/English clinical guidance.
        """
        if not _HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for Groq API communication.")

        evidence_text = "\n\n".join([
            f"--- EVIDENCE [{c.get('chunk_id', c.get('id', 'chunk'))}] ---\n{c.get('text', c.get('content', ''))}" 
            for c in retrieved_chunks
        ])

        lang_instruction = (
            "LANGUAGE MANDATE: The user's query is in ARABIC. You MUST formulate your clinical synthesis, "
            "reasoning, and guidance entirely in fluent, professional Medical Arabic (عربية طبية رصينة). "
            "Retain generic/brand drug names (e.g., Levothyroxine, Carbimazole), numbers, units (e.g., mcg/kg/day, mIU/L), "
            "and citation tags [chunk_id] strictly in English. "
            "OUTPUT RULE: Output ONLY the final clinical answer. Absolutely NO internal thinking steps, 'Here's a thinking process', or <think> blocks."
            if is_arabic else 
            "OUTPUT RULE: Respond in English following strict clinical grounding rules. Output ONLY the final clinical answer directly. Absolutely NO 'Here's a thinking process', planning steps, or <think> blocks."
        )

        user_content = f"{lang_instruction}\n\nEVIDENCE CHUNKS:\n{evidence_text}\n\nCLINICAL QUERY:\n{query}"

        models_to_try = ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "llama-3.3-70b-versatile"]
        if self.model_name and self.model_name not in models_to_try:
            models_to_try.insert(0, self.model_name)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        last_error = ""
        with httpx.Client(timeout=45.0) as client:
            for m in models_to_try:
                payload = {
                    "model": m,
                    "messages": [
                        {"role": "system", "content": CLINICAL_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content}
                    ],
                    "temperature": self.temperature,
                    "max_tokens": min(self.max_tokens, 700),
                    "reasoning_format": "hidden"
                }
                try:
                    resp = client.post(self.api_endpoint, headers=headers, json=payload)
                    if resp.status_code == 200:
                        content = resp.json()["choices"][0]["message"]["content"].strip()
                        if content:
                            cleaned_content = clean_reasoning_output(content)
                            return cleaned_content if cleaned_content else content
                    else:
                        last_error = f"HTTP {resp.status_code}: {resp.text}"
                        logger.warning(f"[Generator] Groq model '{m}' returned {resp.status_code}, trying next...")
                except Exception as e:
                    last_error = str(e)
                    logger.warning(f"[Generator] Groq model '{m}' request failed: {e}")

        raise RuntimeError(f"[Groq API Strict Mode Failure] Could not generate response via Groq. Last error: {last_error}")

    # ------------------------------------------------------------------
    # Fallback path: deterministic local grounded synthesis
    # ------------------------------------------------------------------

    def _local_grounded_synthesis(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        is_arabic: bool,
    ) -> Dict[str, Any]:
        """
        Produces a structured, citation-grounded answer from the top-3 retrieved
        chunks without any external LLM call.  This guarantees 100% uptime and
        deterministic output for offline / CI environments.
        """
        claims = []
        all_citations = []

        for c in retrieved_chunks[:3]:
            doc_label = c.get("authoring_body", c.get("doc_slug", "")).upper()
            year = c.get("year", "")
            page_num = c.get("page_number", "?")
            chunk_id = c["chunk_id"]
            text_snip = c["text"][:250].strip()
            # Remove trailing incomplete word
            text_snip = re.sub(r"\s\S+$", "", text_snip) if len(text_snip) == 250 else text_snip

            label = f"{doc_label} ({year})" if year else doc_label
            if is_arabic:
                claim_text = f"**{label} — ص.{page_num}**: {text_snip}"
            else:
                claim_text = f"**{label} — p.{page_num}**: {text_snip}"

            claims.append({"claim": claim_text, "citation": chunk_id})
            all_citations.append(chunk_id)

        if is_arabic:
            header = (
                "بناءً على الأدلة الإرشادية السريرية المعتمدة "
                "(Clinical Practice Guidelines):\n\n"
            )
        else:
            header = "Based on clinical practice guidelines:\n\n"

        body = "\n\n".join(
            f"- {cl['claim']} [{cl['citation']}]" for cl in claims
        )
        formatted_answer = header + body

        return {
            "answer": formatted_answer,
            "citations": all_citations,
            "claims": claims,
            "status": "success",
        }

    # ------------------------------------------------------------------
    # Post-processing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_arabic(text: str) -> bool:
        return bool(re.search(r"[\u0600-\u06FF]", text))

    @staticmethod
    def _extract_citations(text: str) -> List[str]:
        """Extracts all [chunk_id] patterns from generated text."""
        pattern = r"\[([a-zA-Z0-9_]+(?:::[a-zA-Z0-9_]+)*)\]"
        return list(dict.fromkeys(re.findall(pattern, text)))  # unique, ordered

    @staticmethod
    def _extract_claims(text: str, citations: List[str]) -> List[Dict[str, Any]]:
        """
        Splits generated text into claim sentences and associates each with
        the nearest citation found in that sentence.
        """
        citation_pattern = r"\[([a-zA-Z0-9_]+(?:::[a-zA-Z0-9_]+)*)\]"
        # Split on sentence boundaries and bullet points
        raw_sentences = re.split(r"(?<=[.!?؟])\s+|\n+|(?<=\n)- ", text)
        claims = []
        for sent in raw_sentences:
            sent = sent.strip()
            if len(sent) < 15:
                continue
            cids = re.findall(citation_pattern, sent)
            if not cids and citations:
                cids = citations[:1]  # assign first known citation as fallback
            claims.append({
                "claim": re.sub(citation_pattern, "", sent).strip(),
                "citation": cids[0] if cids else None,
            })
        return claims
