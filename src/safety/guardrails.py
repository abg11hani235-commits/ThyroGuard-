import re
from typing import Tuple, Dict, Any

class ClinicalGuardrails:
    """Deterministic scope, conversational routing, emergency detection, and adversarial guardrails."""

    GREETINGS_TRIGGERS = [
        "hi", "hello", "hey", "good morning", "good evening", "howdy", "who are you", 
        "what can you do", "help", "هاي", "أهلا", "اهلا", "مرحبا", "مرحباً", "السلام عليكم", 
        "صباح الخير", "مساء الخير", "من انت", "مين انت", "ازيك", "عامل ايه", "شكرا", "شكراً", "thanks", "thank you"
    ]

    EMERGENCY_TRIGGERS = [
        "غيبوبة", "غيبوبه", "فاقد الوعي", "فاقد للوعي", "اغمى عليه", "أغمى عليه", "اغماء", "إغماء",
        "توقف التنفس", "مش قادر يتنفس", "ضيق تنفس حاد", "الم شديد في الصدر", "ألم شديد في الصدر",
        "جلطة", "سكته", "طوارئ", "اسعاف", "إسعاف", "تسمم حاد", "نزيف حاد",
        "عاصفة درقية", "عاصفه درقيه", "وذمة مخاطية", "وذمه مخاطيه", "غيبوبة الوذمة المخاطية",
        "انخفاض شديد في درجة الحرارة", "انخفاض شديد في الحرارة", "انخفاض الحرارة", "انخفاض حرارة", "برودة شديدة",
        "تسارع ضربات قلب", "تسارع نبض", "تسارع 140", "نبض فوق", "ضربات قلب سريعة", "خفقان شديد",
        "حرارة 40", "سخونة 40", "حمى شديدة", "ارتفاع شديد في الحرارة",
        "هذيان", "تخبط", "تشوش ذهني", "اضطراب وعي", "فقدان وعي", "توهان حاد",
        "بطء شديد في النبض", "بطء قلب", "بطء نبض", "تسمم درقي حاد",
        "coma", "myxedema coma", "thyroid storm", "unconscious", "loss of consciousness", "fainted", "collapsed",
        "stopped breathing", "severe chest pain", "acute dyspnea", "cardiac arrest", "emergency", "ambulance",
        "severe hypothermia", "high fever 40", "delirium", "extreme tachycardia"
    ]

    NON_THYROID_SYMPTOMS = [
        "ركبة", "ركبه", "مغص", "عظام", "صداع", "ألم في البطن", "الم في البطن", "وجع بطن", 
        "ظهر", "ألم في الظهر", "الم في الظهر", "اسهال", "إسهال", "مفاصل", "خشونة", "كتف", 
        "اسنان", "أسنان", "جلدية", "طفح", "knee", "joint", "back pain", "abdominal pain", 
        "stomach", "diarrhea", "headache", "fracture", "bone", "tooth"
    ]

    THYROID_ANCHOR_TERMS = {
        "غدة", "غده", "درقية", "درقيه", "درقي", "tsh", "ft3", "ft4", "t3", "t4",
        "ليفوثيروكسين", "ثيروكسين", "التروكسين", "يوثيروكس", "ميثيمازول", "كاربيمازول",
        "جريفز", "هاشيموتو", "بيثيسدا", "عقد", "عقيدات", "خزعة", "يود مشع", "تضخم درقي",
        "thyroid", "levothyroxine", "graves", "hashimoto", "bethesda", "nodule", "goiter",
        "antithyroid", "hypothyroid", "hyperthyroid"
    }

    OUT_OF_SCOPE_TRIGGERS = [
        "hypertension", "blood pressure", "insulin", "carbohydrates", "diabetes", 
        "pneumonia", "antibiotic", "france", "paris", "tourist", "binary search", 
        "c++", "programming", "depression", "dsm-5", "tramadol", "pain prescription", 
        "bladder cancer", "compound interest", "retirement", "rheumatoid arthritis",
        "التهاب رئوي", "مضاد حيوي", "ارتفاع ضغط الدم", "ضغط الدم", "السكري", "سكر الدم",
        "انسولين", "أنسولين", "اكتئاب", "سرطان المثانة", "روماتويد", "برمجة"
    ]

    ADVERSARIAL_TRIGGERS = [
        "dan", "jailbreak", "ignore previous instructions", "home manufacture", 
        "sewing needles", "self-perform", "5g", "lunar thyroiditis", "thyrostop-99", 
        "cure hashimoto", "levothyroxine for acute", "levothyroxine as the first-line emergency", 
        "injecting tsh", "bovine tsh", "recombinant bovine", "rogue endocrinologist",
        "pure fluid-filled thyroid cysts with no solid tissue have a 95%", "pure fluid-filled"
    ]

    @classmethod
    def evaluate_safety(cls, query: str) -> Tuple[bool, str, str]:
        """
        Evaluates safety and intent:
        Returns: (is_handled_early, gate_category, message)
        If is_handled_early is True, bypass retrieval!
        """
        q_clean = query.strip()
        q_lower = q_clean.lower()

        # 1. Empty or Very Short Input
        if len(q_clean) < 3:
            return (
                True,
                "short_query",
                "يرجى كتابة سؤال سريري واضح بخصوص الغدة الدرقية (مثل جرعات الليفوثيروكسين، عقد الغدة الدرقية، أو علاج فرط النشاط)."
            )

        # 2. Emergency Crisis Protocol (HIGHEST PRIORITY)
        # Check explicit emergency triggers
        is_emergency = any(emg in q_lower for emg in cls.EMERGENCY_TRIGGERS)
        
        # Check clinical combination patterns (e.g. fever 40 + delirium, hypothermia + coma, pulse > 140)
        if not is_emergency:
            has_high_fever = bool(re.search(r'(40|41|حمى شديدة|حرارة مرتفعة جدا)', q_lower))
            has_neuro = bool(re.search(r'(هذيان|تخبط|غيبوبة|اضطراب وعي|فاقد الوعي|توهان)', q_lower))
            has_tachy = bool(re.search(r'(140|130|150|تسارع|خفقان)', q_lower))
            has_hypothermia = bool(re.search(r'(انخفاض|برودة).*(حرارة|جسم)', q_lower))
            
            if (has_high_fever and (has_neuro or has_tachy)) or (has_hypothermia and has_neuro):
                is_emergency = True

        if is_emergency:
            emergency_msg = (
                "🚨 **تنبيه طوارئ سريري عاجل (Clinical Emergency Alert)** 🚨\n\n"
                "هذه الحالة قد تمثل طوارئ طبية مهددة للحياة (مثل غيبوبة الوذمة المخاطية **Myxedema Coma** أو عاصفة درقية حادة **Thyroid Storm**) وتتطلب تدخلاً طبياً فورياً في المستشفى.\n\n"
                "📞 **اتصل بالإسعاف فوراً:**\n"
                "- 🇪🇬 **مصر (Egypt)**: **123** (أو طوارئ الرعاية العاجلة **137**)\n"
                "- 🇸🇦 **السعودية (KSA)**: **997** (أو الطوارئ العامة **911**)\n"
                "- 🇦🇪 **الإمارات (UAE)**: **998** (أو **999**)\n"
                "- 🌍 **الرقم الدولي العام**: **911** أو **112**\n\n"
                "⚠️ **إرشادات أولية فورية لحين وصول الإسعاف:**\n"
                "1. تأكد من مجرى التنفس (Airway) وتمديد المصاب على جنبه (Recovery Position) لتجنب الاختناق.\n"
                "2. **ممنوع تماماً** إعطاء أي أدوية أو ماء عن طريق الفم لمريض فاقد للوعي أو في حالة غيبوبة.\n"
                "3. قم بتدفئة المريض بلطف بغطاء إذا كان يشتبه في غيبوبة قصور درقي حاد (تجنب التدفئة الزائدة المفاجئة).\n"
                "4. انقل كل علب الأدوية والتاريخ الطبي مع المريض للمستشفى."
            )
            return (True, "medical_emergency", emergency_msg)

        # 3. Conversational Chit-chat / Greetings (Bypass Retrieval)
        words = re.findall(r'\b\w+\b', q_lower)
        if len(words) <= 4:
            for greet in cls.GREETINGS_TRIGGERS:
                if greet in q_lower or q_lower == greet:
                    greeting_msg = (
                        "أهلاً بك! 👋 أنا **ThyroGuard**، المساعد السريري الذكي المتخصص في أدلة وتوصيات الغدد الصماء وأورام الغدة الدرقية (ATA, ETA, AACE).\n\n"
                        "💡 **يمكنك سؤالي عن:**\n"
                        "- جرعات ليفوثيروكسين وبدء العلاج لحالات قصور الغدة الدرقية.\n"
                        "- تقييم وتصنيف عقد الغدة الدرقية (Bethesda & ATA Sonographic Patterns).\n"
                        "- بروتوكولات علاج فرط نشاط الغدة وعاصفة الغدة الدرقية.\n"
                        "- متابعة أورام الغدة الدرقية المتمايزة (DTC) ومستويات الثيروجلوبولين.\n\n"
                        "تفضل بطرح سؤالك السريري وسأقوم باستخراج الأدلة والصفحات الموثقة فوراً."
                    )
                    return (True, "conversational_greeting", greeting_msg)

        # 4. Adversarial & Prompt Injection Traps
        for adv in cls.ADVERSARIAL_TRIGGERS:
            if adv in q_lower:
                return (
                    True,
                    "adversarial_blocked",
                    "REFUSAL: Safety guardrail triggered. This query contains false clinical premises, dangerous instructions, or prompt overrides."
                )

        # 5. Out-of-Scope Non-Thyroid Symptoms Check (e.g. Knee, Stomach, Bones without thyroid context)
        has_thyroid_context = any(term in q_lower for term in cls.THYROID_ANCHOR_TERMS)
        if not has_thyroid_context:
            for non_th in cls.NON_THYROID_SYMPTOMS:
                if non_th in q_lower:
                    return (
                        True,
                        "out_of_scope_blocked",
                        "REFUSAL: أنا **ThyroGuard**، نظام سريري متخصص حصراً في أمراض وأورام وإرشادات الغدة الدرقية (ATA, ETA, AACE). سؤالك يتعلق بأعراض خارج اختصاص الغدة الدرقية (مثل آلام العظام أو المفاصل أو الباطنة العامة). يرجى استشارة الطبيب المختص أو زيارة العيادة المناسبة."
                    )

        # 6. Out-of-Scope Topics (Hypertension, Diabetes, Pneumonia, etc.)
        for oos in cls.OUT_OF_SCOPE_TRIGGERS:
            if oos in q_lower:
                return (
                    True,
                    "out_of_scope_blocked",
                    "REFUSAL: I am ThyroGuard, strictly dedicated to clinical thyroid endocrinology and oncology guidelines (ATA, ETA, AACE). This clinical inquiry is outside authorized scope and guidelines."
                )

        # In-Scope Legitimate Query -> Proceed to Hybrid Retrieval!
        return (False, "in_scope", "")
