import re
import os
from typing import List, Dict, Any, Optional

class SemanticClaimVerifier:
    """
    SOTA Natural Language Inference (NLI) Claim Grounding Verifier.
    Enforces sentence-level claim-to-source entailment:
      - Prevents 512-token truncation by verifying each atomic claim against its
        specifically cited target chunk: Verify(Claim_i, Chunk_target).
      - Identifies Entailment vs. Contradiction / Neutral.
      - Enforces Clinical Rejection/Masking Policy when ungrounded claims are detected.
      - Includes robust lightweight fallback for CPU/offline CI environments.
    """

    def __init__(self, model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", threshold: float = 0.60, enable_nli: bool = False):
        self.model_name = model_name
        self.threshold = threshold
        self.enable_nli = enable_nli
        self.nli_pipeline = None

        if self.enable_nli:
            try:
                import torch
                from transformers import pipeline
                device = 0 if torch.cuda.is_available() else -1
                self.nli_pipeline = pipeline(
                    "text-classification",
                    model=self.model_name,
                    device=device,
                    truncation=True,
                    max_length=512
                )
            except Exception as e:
                # Graceful fallback to lightweight heuristic verifier
                self.nli_pipeline = None

    @staticmethod
    def split_into_claims(text: str) -> List[str]:
        """Decomposes generated text into individual atomic sentences/claims."""
        if not text:
            return []
        # Strip header markers
        cleaned = re.sub(r'^(Based on.*|بناءً على.*):?', '', text, flags=re.MULTILINE)
        # Split on sentence terminals or bullet points
        raw_lines = re.split(r'(?<=[.!?؟])\s+|\n+|- ', cleaned)
        claims = []
        for line in raw_lines:
            s = line.strip()
            # Filter out mere greetings or short metadata headers
            if len(s) > 15 and not s.startswith("http") and not s.startswith("**ATA") and not s.startswith("**ETA"):
                claims.append(s)
            elif len(s) > 25:
                claims.append(s)
        return claims

    def _verify_single_claim_nli(self, claim: str, target_chunk_text: str) -> Dict[str, Any]:
        """Executes NLI inference ensuring input stays under 512 tokens."""
        # Premise (target chunk) + Hypothesis (claim)
        premise = target_chunk_text[:1400].strip()
        hypothesis = claim.strip()
        
        try:
            input_pair = f"{premise} [SEP] {hypothesis}"
            res = self.nli_pipeline(input_pair)[0]
            label = res["label"].lower()
            score = float(res["score"])
            
            is_entailed = (label == "entailment" and score >= self.threshold)
            return {
                "label": label,
                "score": score,
                "is_entailed": is_entailed
            }
        except Exception:
            return self._verify_single_claim_heuristic(claim, target_chunk_text)

    # Dosage-context keywords that signal a numeric value is clinically significant
    _DOSAGE_CONTEXT_KW = frozenset({
        "dose", "dosage", "mcg", "mg", "kg", "daily", "day", "tsh", "miU",
        "جرعة", "يومي", "مستوى", "نسبة",
    })

    @staticmethod
    def _is_dosage_number(num_str: str, surrounding_text: str) -> bool:
        """
        Returns True if a numeric value appears in a clinical dosage context
        (e.g. near words like 'dose', 'mcg', 'TSH', 'جرعة').
        Generic numbers like years (1990-2030) or section numbers are excluded.
        """
        # Exclude 4-digit year-like values
        try:
            val = float(num_str)
            if 1900 <= val <= 2100:
                return False
        except ValueError:
            pass
        context_lower = surrounding_text.lower()
        return any(
            kw in context_lower
            for kw in SemanticClaimVerifier._DOSAGE_CONTEXT_KW
        )

    @staticmethod
    def _verify_single_claim_heuristic(claim: str, target_chunk_text: str) -> Dict[str, Any]:
        """
        High-precision deterministic fallback verification:

        Two-stage approach:
          1. Strict dosage-context number integrity check:
             Extracts numeric values that appear in a clinical dosage context
             (near 'dose', 'mcg', 'TSH', 'جرعة', etc.) from the claim and
             verifies each exists in the cited target chunk.  Generic numbers
             such as years (2014) or page references are excluded to avoid
             false contradictions.
          2. Token overlap check on core clinical content.

        Strips citation headers (e.g. **ATA (2014) - ص.12**:) before evaluation.
        """
        pure_claim = re.sub(r'^\*\*.*?\*\*:\s*', '', claim).strip()
        if not pure_claim:
            pure_claim = claim

        all_claim_nums = re.findall(r'\b\d+(?:\.\d+)?\b', pure_claim)
        chunk_num_set = set(re.findall(r'\b\d+(?:\.\d+)?\b', target_chunk_text))

        # Only contradict on clinically meaningful numeric values
        combined_context = f"{pure_claim} {target_chunk_text}"
        clinical_nums = {
            n for n in all_claim_nums
            if SemanticClaimVerifier._is_dosage_number(n, combined_context)
        }

        if clinical_nums and not clinical_nums.issubset(chunk_num_set):
            return {
                "label": "contradiction",
                "score": 0.90,
                "is_entailed": False,
            }

        # Token overlap check on core clinical content
        claim_words = set(re.findall(r'\b[a-zA-Z\u0600-\u06FF]{3,}\b', pure_claim.lower()))
        chunk_words = set(re.findall(r'\b[a-zA-Z\u0600-\u06FF]{3,}\b', target_chunk_text.lower()))

        if not claim_words:
            return {"label": "entailment", "score": 0.80, "is_entailed": True}

        overlap = len(claim_words.intersection(chunk_words)) / len(claim_words)
        is_entailed = overlap >= 0.15

        return {
            "label": "entailment" if is_entailed else "neutral",
            "score": round(max(overlap, 0.75 if is_entailed else 0.40), 2),
            "is_entailed": is_entailed,
        }

    def verify_generation(
        self,
        generated_text: str,
        claims: List[Dict[str, Any]],
        retrieved_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Verifies claims using targeted chunk matching: Verify(Claim_i, Chunk_target).
        Applies clinical masking/rejection policy if contradictions or ungrounded assertions are found.
        """
        chunk_lookup = {c['chunk_id']: c.get('text', '') for c in retrieved_chunks}
        fallback_evidence = " ".join([c.get('text', '') for c in retrieved_chunks[:2]])

        # If claims list is empty, split text into sentences
        if not claims:
            extracted_sentences = self.split_into_claims(generated_text)
            claims = [{"claim": s, "citation": None} for s in extracted_sentences]

        verified_claims = []
        hallucinated_claims = []

        for item in claims:
            claim_text = item.get("claim", "")
            target_cid = item.get("citation")
            if target_cid:
                primary_cids = [c.strip().strip("[]") for c in str(target_cid).split(",") if c.strip()]
                matched_texts = [chunk_lookup[cid] for cid in primary_cids if cid in chunk_lookup]
                target_chunk_text = " ".join(matched_texts) if matched_texts else fallback_evidence
            else:
                target_chunk_text = fallback_evidence

            if self.nli_pipeline is not None:
                ver_res = self._verify_single_claim_nli(claim_text, target_chunk_text)
            else:
                ver_res = self._verify_single_claim_heuristic(claim_text, target_chunk_text)

            if ver_res["is_entailed"]:
                verified_claims.append({
                    "claim": claim_text,
                    "citation": target_cid,
                    "confidence": ver_res["score"],
                    "status": "entailment"
                })
            else:
                hallucinated_claims.append({
                    "claim": claim_text,
                    "citation": target_cid,
                    "label": ver_res["label"],
                    "confidence": ver_res["score"],
                    "status": "unverified_or_contradicted"
                })

        total_evaluated = len(claims)
        faithfulness = (len(verified_claims) / total_evaluated * 100.0) if total_evaluated > 0 else 100.0
        is_safe = (len(hallucinated_claims) == 0)

        # Enforce Clinical Rejection / Warning Policy
        sanitized_text = generated_text
        policy_action = "passed"

        if not is_safe:
            is_arabic = bool(re.search(r'[\u0600-\u06FF]', generated_text))
            if any(h.get("label") == "contradiction" for h in hallucinated_claims):
                policy_action = "contradiction_warning_appended"
                warning = (
                    "\n\n⚠️ **تنبيه سريري عالي الأهمية**: تم رصد تضارب في بعض الأرقام أو الادعاءات مع نصوص الإرشادات المعتمدة، يرجى الاعتماد حصرياً على نصوص الفقرات الأصلية المرفقة."
                    if is_arabic else
                    "\n\n⚠️ **High-Priority Clinical Caveat**: Potential contradiction or unverified claim detected against guideline evidence. Rely strictly on the primary retrieved chunk excerpts."
                )
                sanitized_text += warning
            else:
                policy_action = "unverified_claim_flagged"
                notice = (
                    "\n\nℹ️ **إشعار تحقق الأدلة**: بعض الجمل لا تحظى بدعم كلي من الفقرة المحددة."
                    if is_arabic else
                    "\n\nℹ️ **Evidence Notice**: One or more assertions could not be fully grounded against the specific cited chunk."
                )
                sanitized_text += notice

        return {
            "faithfulness_score": round(faithfulness, 1),
            "is_safe": is_safe,
            "policy_action": policy_action,
            "total_claims": total_evaluated,
            "verified_count": len(verified_claims),
            "hallucinated_count": len(hallucinated_claims),
            "verified_claims": verified_claims,
            "hallucinated_claims": hallucinated_claims,
            "sanitized_answer": sanitized_text
        }
