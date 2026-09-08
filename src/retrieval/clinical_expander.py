import re
from typing import Dict, List, Any

class CrossLingualClinicalExpander:
    """
    Cross-lingual clinical query expander for thyroid endocrinology.
    Provides dual-query generation:
      - 'sparse_query': Rich keyword and trade-name expansion for BM25 lexical matching.
      - 'dense_query': Clean, focused semantic representation without keyword cluttering
                       to prevent embedding drift in dense vector representations.
    """

    CLINICAL_MAP: Dict[str, str] = {
        # Medications & Trade Names
        "التروكسين": "eltroxin levothyroxine l-thyroxine t4",
        "يوثيروكس": "euthyrox levothyroxine l-thyroxine t4",
        "ليفوثيروكسين": "levothyroxine l-thyroxine t4 synthroid",
        "ثيروكسين": "thyroxine levothyroxine t4",
        "ميثيمازول": "methimazole thiamazole mmi",
        "كاربيمازول": "carbimazole cbz antithyroid",
        "بروبيل ثيويوراسيل": "propylthiouracil ptu",
        "يود مشع": "radioactive iodine rai 131i ablation",
        "حاصرات بيتا": "beta blockers propranolol atenolol",

        # Conditions & Syndromes
        "قصور الغدة": "hypothyroidism primary overt subclinical",
        "خمول الغدة": "hypothyroidism elevated tsh",
        "كسل الغدة": "hypothyroidism thyroid hormone deficiency",
        "فرط نشاط الغدة": "hyperthyroidism thyrotoxicosis",
        "نشاط الغدة": "hyperthyroidism graves disease",
        "تسمم درقي": "thyrotoxicosis hyperthyroidism",
        "جريفز": "graves disease hyperthyroidism trab",
        "هاشيموتو": "hashimoto thyroiditis autoimmune",
        "التهاب الغدة": "thyroiditis subacute de quervain silent",

        # Crises & Emergencies
        "عاصفة درقية": "thyroid storm thyrotoxic crisis burch-wartofsky",
        "غيبوبة الوذمة المخاطية": "myxedema coma hypothyroid crisis",

        # Anatomy & Pathology
        "عقد درقية": "thyroid nodules fna biopsy ultrasound sonography",
        "عقيدات": "thyroid nodules solid cystic hypoechoic",
        "بيثيسدا": "bethesda system thyroid cytopathology category",
        "خزعة": "fine needle aspiration fna biopsy cytology",
        "سرطان الغدة": "differentiated thyroid cancer papillary follicular dtc",
        "ورم حليمي": "papillary thyroid cancer ptc",
        "ورم نخاعي": "medullary thyroid cancer mtc calcitonin",
        "ورم كشمي": "anaplastic thyroid cancer atc",

        # Labs & Parameters
        "tsh": "thyroid stimulating hormone thyrotropin reference range",
        "تحليل tsh": "serum tsh upper reference limit 4.5",
        "ft4": "free thyroxine ft4 normal range",
        "ft3": "free triiodothyronine ft3",
        "أجسام مضادة": "tpo antibodies trab anti-thyroglobulin tgab",

        # Demographics & Special Populations
        "حمل": "pregnancy pregnant trimester first second third",
        "حامل": "pregnancy pregnant obstetric maternal fetal",
        "كبار السن": "elderly older adults cardiac coronary artery disease",
        "مسن": "elderly cardiac ischemia slow titration starting dose",
        "جرعة": "dosage dose mcg microgram kg daily titration",
        "بدء العلاج": "starting replacement dose initial therapy",
    }

    # Intent templates for clean dense query representation
    DENSE_INTENT_PATTERNS = [
        (
            r"(التروكسين|يوثيروكس|ليفوثيروكسين|خمول|قصور).*(مسن|كبار السن|قلب|شريان)",
            "Levothyroxine starting replacement dose in elderly patients and coronary artery disease"
        ),
        (
            r"(التروكسين|يوثيروكس|ليفوثيروكسين|خمول|قصور).*(جرعة|بدء|علاج)",
            "Recommended initial daily starting dose of levothyroxine in healthy adults with overt primary hypothyroidism 1.6 mcg/kg"
        ),
        (
            r"(بيثيسدا|خزعة|عقد|عقيدات|fna)",
            "Bethesda System for Reporting Thyroid Cytopathology FNA biopsy criteria for thyroid nodules"
        ),
        (
            r"(جريفز|نشاط|فرط|تسمم درقي).*(يود|جراحة|ميثيمازول|كاربيمازول|علاج)",
            "Graves disease hyperthyroidism treatment options radioactive iodine antithyroid drugs thionamides surgery"
        ),
        (
            r"(tsh|4\.5).*(تحليل|فحص|طبيعي|علاج)",
            "Serum TSH upper reference limit 4.5 mIU/L subclinical hypothyroidism treatment indications"
        ),
        (
            r"(حمل|حامل).*(خمول|قصور|tsh|جرعة)",
            "Management of hypothyroidism in pregnancy and trimester-specific TSH reference ranges"
        ),
    ]

    @staticmethod
    def normalize_arabic(text: str) -> str:
        """Strips Arabic diacritics and unifies variant characters."""
        if not text:
            return ""
        # Remove tashkeel & tatweel
        text = re.sub(r'[\u0617-\u061A\u064B-\u0652\u0640]', '', text)
        # Normalize alef variants
        text = re.sub(r'[إأآا]', 'ا', text)
        # Normalize ya and ta marbuta
        text = re.sub(r'ى', 'ي', text)
        text = re.sub(r'ة\b', 'ه', text)
        return text.strip()

    @classmethod
    def get_dual_queries(cls, raw_query: str) -> Dict[str, str]:
        """
        Produces dual representations of the query:
        1. 'dense_query': Semantic translation and focused intent without keyword cluttering.
        2. 'sparse_query': Lexically enriched with synonyms and trade names for BM25.
        """
        is_ar = bool(re.search(r'[\u0600-\u06FF]', raw_query))
        if not is_ar:
            # English queries: check for trade names or shorthand
            q_lower = raw_query.lower()
            sparse_terms = []
            if "eltroxin" in q_lower or "euthyrox" in q_lower or "synthroid" in q_lower:
                sparse_terms.append("levothyroxine l-thyroxine")
            if "carbimazole" in q_lower or "cbz" in q_lower:
                sparse_terms.append("methimazole antithyroid")
            if "ptu" in q_lower:
                sparse_terms.append("propylthiouracil")
            
            sparse_query = f"{raw_query} {' '.join(sparse_terms)}" if sparse_terms else raw_query
            return {
                "dense_query": raw_query,
                "sparse_query": sparse_query,
                "is_arabic": False
            }

        norm_q = cls.normalize_arabic(raw_query).lower()

        # 1. Build Clean Dense Query (Avoids Keyword Stuffing & Embedding Drift)
        matched_dense = None
        for pattern, clinical_repr in cls.DENSE_INTENT_PATTERNS:
            if re.search(pattern, norm_q):
                matched_dense = clinical_repr
                break

        if matched_dense:
            dense_query = matched_dense
        else:
            # Fallback dense query: map core concepts into concise clinical english phrase
            english_concepts = []
            for ar_term, en_expansion in cls.CLINICAL_MAP.items():
                norm_term = cls.normalize_arabic(ar_term).lower()
                if norm_term in norm_q:
                    # Pick the primary representative token (first 1-2 words)
                    primary_token = " ".join(en_expansion.split()[:2])
                    if primary_token not in english_concepts:
                        english_concepts.append(primary_token)
            
            dense_query = " ".join(english_concepts) if english_concepts else raw_query

        # 2. Build Sparse Query (Enriched with Brand Names & Lexical Synonyms for BM25)
        sparse_additions = []
        for ar_term, en_expansion in cls.CLINICAL_MAP.items():
            norm_term = cls.normalize_arabic(ar_term).lower()
            if norm_term in norm_q:
                sparse_additions.append(en_expansion)

        sparse_query = f"{raw_query} {' '.join(sparse_additions)}" if sparse_additions else raw_query

        return {
            "dense_query": dense_query,
            "sparse_query": sparse_query,
            "is_arabic": True
        }
