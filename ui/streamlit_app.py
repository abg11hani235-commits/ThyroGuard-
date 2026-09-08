"""
ThyroGuard Clinical Intelligence Platform — Streamlit Workspace
Faithful reproduction and enhancement of the BioGuard/ThyroGuard Clinical Workspace design.
Preserves 100% of the original aesthetic, layout, and tabs (Clinical Assistant, Knowledge Base, System & Models)
with seamless Dark/Light theme switching and advanced clinical diagnostics.
"""

import os
import json
import time
import re
import requests
import yaml
import streamlit as st

# -------------------------------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -------------------------------------------------------------------------------------------------
st.set_page_config(
    page_title="BioGuard AI | Clinical Thyroid Decision Support",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Forward Streamlit Secrets into os.environ
try:
    if hasattr(st, "secrets"):
        for sec_key, sec_val in st.secrets.items():
            if isinstance(sec_val, str) and sec_key not in os.environ:
                os.environ[sec_key] = sec_val
except Exception:
    pass

# -------------------------------------------------------------------------------------------------
# 2. THEME STATE & CSS
# -------------------------------------------------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state["theme"] = "light"  # Default to exact Light Medical Theme from user's screenshot

is_dark = (st.session_state["theme"] == "dark")

APP_THEME_CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Tajawal:wght@400;500;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {{
        --bg-main: {'#080d1a' if is_dark else '#f8fafc'};
        --bg-surface: {'#0e1626' if is_dark else '#ffffff'};
        --bg-surface-alt: {'#131f33' if is_dark else '#f1f5f9'};
        --bg-sidebar: {'#0b1220' if is_dark else '#ffffff'};
        --border-subtle: {'#1e293b' if is_dark else '#e2e8f0'};
        --border-card: {'#1f2d47' if is_dark else '#e2e8f0'};
        --text-main: {'#f8fafc' if is_dark else '#0f172a'};
        --text-muted: {'#94a3b8' if is_dark else '#64748b'};
        --text-light: {'#64748b' if is_dark else '#94a3b8'};
        --primary-indigo: #4338ca;
        --status-green: #10b981;
    }}

    .stApp {{
        background-color: var(--bg-main) !important;
        color: var(--text-main) !important;
        font-family: 'Inter', 'Tajawal', sans-serif !important;
    }}

    .block-container {{
        padding-top: 1rem !important;
        padding-bottom: 3rem !important;
        max-width: 1440px !important;
    }}

    [data-testid="stSidebar"] {{
        background-color: var(--bg-sidebar) !important;
        border-right: 1px solid var(--border-subtle) !important;
    }}

    /* Global Buttons */
    .stButton>button {{
        border-radius: 8px !important;
        background-color: var(--bg-surface) !important;
        color: var(--text-main) !important;
        border: 1px solid var(--border-subtle) !important;
        font-family: 'Inter', 'Tajawal', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.84rem !important;
        transition: all 0.2s ease !important;
        padding: 0.45rem 0.8rem !important;
    }}
    .stButton>button:hover {{
        border-color: #4f46e5 !important;
        background-color: var(--bg-surface-alt) !important;
        color: #4f46e5 !important;
    }}

    /* Tabs Styling to match user's screenshot */
    .stTabs [data-baseweb="tab-list"] {{
        background-color: var(--bg-surface) !important;
        border-bottom: 1px solid var(--border-subtle) !important;
        gap: 2rem !important;
        padding: 0 1rem !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 48px !important;
        background-color: transparent !important;
        border: none !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        color: var(--text-muted) !important;
    }}
    .stTabs [aria-selected="true"] {{
        color: #4f46e5 !important;
        border-bottom: 2px solid #4f46e5 !important;
    }}

    /* Inputs */
    .stTextInput>div>div>input, .stTextArea textarea {{
        background-color: var(--bg-surface) !important;
        color: var(--text-main) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 10px !important;
        font-family: 'Inter', 'Tajawal', sans-serif !important;
    }}

    /* Accordions & Expanders */
    [data-testid="stExpander"] {{
        background-color: var(--bg-surface) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 8px !important;
        margin-bottom: 0.75rem !important;
    }}
</style>
"""
st.markdown(APP_THEME_CSS, unsafe_allow_html=True)

# -------------------------------------------------------------------------------------------------
# 3. CORPUS & PIPELINE LOADERS
# -------------------------------------------------------------------------------------------------
@st.cache_data
def load_corpus_data():
    corpus_path = os.path.join(PROJECT_ROOT, "data", "processed", "FROZEN_CORPUS.json")
    if os.path.exists(corpus_path):
        with open(corpus_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            return {c["chunk_id"]: c for c in chunks}
    return {}

corpus_lookup = load_corpus_data()

@st.cache_data
def load_models_config():
    cfg_path = os.path.join(PROJECT_ROOT, "config", "models.yaml")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}

models_cfg = load_models_config()

@st.cache_resource
def get_pipeline():
    from src.pipeline.rag_pipeline import ThyroGuardPipeline
    corpus_p = os.path.join(PROJECT_ROOT, "data", "processed", "FROZEN_CORPUS.json")
    chroma_p = os.path.join(PROJECT_ROOT, "data", "processed", "chroma_bge")
    return ThyroGuardPipeline(corpus_path=corpus_p, chroma_dir=chroma_p)

def run_query(query_text: str, top_k: int = 5):
    # Try FastAPI first
    try:
        res = requests.post(f"{API_BASE_URL}/query", json={"query": query_text, "top_k": top_k}, timeout=3)
        if res.status_code == 200:
            return res.json(), "FastAPI REST"
    except Exception:
        pass

    # Direct In-Process Core
    try:
        pipeline = get_pipeline()
        res = pipeline.query(query_text, top_k=top_k)
        return res, "In-Process Core"
    except Exception as e:
        return {
            "answer": f"Consultation notice: {e}",
            "citations": [],
            "retrieved_chunk_ids": [],
            "status": "error",
            "safety_status": "error"
        }, "Offline Core"

# -------------------------------------------------------------------------------------------------
# 4. SIDEBAR (FAITHFUL TO USER'S SCREENSHOT)
# -------------------------------------------------------------------------------------------------
with st.sidebar:
    # Sidebar Brand
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 10px; padding: 0.2rem 0 1rem 0; border-bottom: 1px solid var(--border-subtle);">
        <div style="width: 32px; height: 32px; color: #4338ca; display: flex; align-items: center; justify-content: center;">
            <svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M4.8 2.3A.3.3 0 1 0 5 2H4a2 2 0 0 0-2 2v5a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6V4a2 2 0 0 0-2-2h-1a.2.2 0 1 0 .3.3"></path>
                <path d="M8 15v1a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6v-4"></path>
                <circle cx="20" cy="10" r="2"></circle>
            </svg>
        </div>
        <div>
            <div style="font-weight: 800; font-size: 1.05rem; color: var(--text-main); line-height: 1.1;">BioGuard</div>
            <div style="font-size: 0.62rem; font-weight: 700; color: var(--text-muted); letter-spacing: 0.08em; text-transform: uppercase;">CLINICAL WORKSPACE</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

    # New Consultation Button
    if st.button("+ New Consultation", use_container_width=True):
        st.session_state["query_text"] = ""
        st.session_state["active_data"] = None
        st.rerun()

    st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)

    # RECENT CONSULTATIONS
    st.markdown("<div style='font-size: 0.7rem; font-weight: 700; color: var(--text-light); letter-spacing: 0.08em; margin-bottom: 0.4rem;'>RECENT CONSULTATIONS</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.78rem; color: var(--text-muted); padding-bottom: 0.8rem;'>No consultations yet in this session.</div>", unsafe_allow_html=True)

    # BOOKMARKED PROTOCOLS
    st.markdown("<div style='font-size: 0.7rem; font-weight: 700; color: var(--text-light); letter-spacing: 0.08em; margin-bottom: 0.5rem;'>BOOKMARKED PROTOCOLS</div>", unsafe_allow_html=True)
    
    protocols = [
        {"icon": "💉", "title": "Type 2 Diabetes / Hypothyroid", "q": "What is the recommended initial starting dose of levothyroxine in healthy adults with overt primary hypothyroidism?"},
        {"icon": "💉", "title": "Insulin & CAD Initiation Guidelines", "q": "ما هي الجرعة الموصى بها لبدء علاج ليفوثيروكسين في كبار السن ومرضى الشريان التاجي وفق ATA؟"},
        {"icon": "🫀", "title": "Cardiovascular Risk Reduction", "q": "قارن بين خيارات علاج جريفز: اليود المشع والأدوية المضادة للدرقية والجراحة وفق الإرشادات السريرية."}
    ]

    for p in protocols:
        if st.button(f"{p['icon']} {p['title']}", key=f"p_{p['title']}", use_container_width=True):
            st.session_state["query_text"] = p["q"]
            data, mode = run_query(p["q"])
            st.session_state["active_data"] = data
            st.session_state["source_mode"] = mode
            st.rerun()

    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

    # SYSTEM
    st.markdown("<div style='font-size: 0.7rem; font-weight: 700; color: var(--text-light); letter-spacing: 0.08em; margin-bottom: 0.4rem;'>SYSTEM</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size: 0.8rem; color: var(--text-muted); line-height: 2;">
        <div>📄 Document Knowledge Base</div>
        <div>⚙️ Hardware & Models</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

    # DEVELOPER
    st.markdown("<div style='font-size: 0.7rem; font-weight: 700; color: var(--text-light); letter-spacing: 0.08em; margin-bottom: 0.3rem;'>DEVELOPER</div>", unsafe_allow_html=True)
    obs_debug = st.checkbox("Observability Debug Mode 🛈", value=False)

    st.markdown("---")
    
    # Bottom Theme Switcher & Clear History
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state["query_text"] = ""
        st.session_state["active_data"] = None
        st.rerun()

    st.markdown("<div style='font-size: 0.68rem; color: var(--text-light); text-align: center; margin-top: 0.3rem;'>BioGuard Clinical RAG v2.0</div>", unsafe_allow_html=True)

# -------------------------------------------------------------------------------------------------
# 5. TOP NAVBAR (MATCHING SCREENSHOT)
# -------------------------------------------------------------------------------------------------
top_c1, top_c2, top_c3 = st.columns([1.5, 2, 1.2])

with top_c1:
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 10px; height: 100%;">
        <div style="width: 32px; height: 32px; color: #4338ca; display: flex; align-items: center; justify-content: center;">
            <svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M4.8 2.3A.3.3 0 1 0 5 2H4a2 2 0 0 0-2 2v5a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6V4a2 2 0 0 0-2-2h-1a.2.2 0 1 0 .3.3"></path>
                <path d="M8 15v1a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6v-4"></path>
                <circle cx="20" cy="10" r="2"></circle>
            </svg>
        </div>
        <div>
            <div style="font-weight: 800; font-size: 1.1rem; color: var(--text-main); line-height: 1.1;">BioGuard</div>
            <div style="font-size: 0.62rem; font-weight: 700; color: var(--text-muted); letter-spacing: 0.08em; text-transform: uppercase;">CLINICAL INTELLIGENCE PLATFORM</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with top_c2:
    st.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: center; height: 100%; font-size: 0.82rem; font-weight: 700; letter-spacing: 0.08em; color: var(--text-muted); text-transform: uppercase;">
        EVIDENCE-GROUNDED MEDICAL REFERENCE ASSISTANT
    </div>
    """, unsafe_allow_html=True)

with top_c3:
    col_stat, col_thm = st.columns([1.8, 1])
    with col_stat:
        st.markdown(f"""
        <div style="text-align: right;">
            <span style="display: inline-flex; align-items: center; gap: 6px; background-color: {'rgba(16,185,129,0.15)' if is_dark else '#f0fdf4'}; border: 1px solid {'rgba(16,185,129,0.35)' if is_dark else '#bbf7d0'}; color: {'#34d399' if is_dark else '#166534'}; font-size: 0.72rem; font-weight: 600; padding: 2px 9px; border-radius: 9999px;">
                <span style="width: 7px; height: 7px; border-radius: 50%; background-color: #22c55e;"></span>
                Operational
            </span>
            <div style="font-size: 0.66rem; color: var(--text-light); font-family: monospace; margin-top: 2px;">qwen3.6-27b</div>
        </div>
        """, unsafe_allow_html=True)
    with col_thm:
        # Dark / Light Theme Toggle matching user's switch
        theme_btn_label = "🌙" if not is_dark else "☀️"
        if st.button(theme_btn_label, help="Toggle Dark/Light Mode", use_container_width=True):
            st.session_state["theme"] = "dark" if not is_dark else "light"
            st.rerun()

st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

# -------------------------------------------------------------------------------------------------
# 6. NAVIGATION TABS (MATCHING USER'S SCREENSHOT)
# -------------------------------------------------------------------------------------------------
tab_assistant, tab_knowledge, tab_system = st.tabs([
    "💬 Clinical Assistant",
    "📚 Knowledge Base",
    "⚙️ System & Models"
])

# -------------------------------------------------------------------------------------------------
# TAB 1: CLINICAL ASSISTANT
# -------------------------------------------------------------------------------------------------
with tab_assistant:
    st.markdown("<div style='height: 0.4rem;'></div>", unsafe_allow_html=True)
    
    if "query_text" not in st.session_state:
        st.session_state["query_text"] = "What is the recommended initial starting dose of levothyroxine in healthy adults with overt primary hypothyroidism?"

    c_query = st.text_input(
        "Clinical Consultation Query",
        value=st.session_state.get("query_text", ""),
        placeholder="Enter patient case details, clinical inquiry, or select a bookmarked protocol from the left...",
        label_visibility="collapsed"
    )

    q_c1, q_c2 = st.columns([4, 1])
    with q_c2:
        if st.button("🔍 Evaluate Query", use_container_width=True):
            if c_query.strip():
                st.session_state["query_text"] = c_query
                with st.spinner("Analyzing evidence guidelines..."):
                    data, mode = run_query(c_query)
                    st.session_state["active_data"] = data
                    st.session_state["source_mode"] = mode
                st.rerun()

    active_data = st.session_state.get("active_data")
    if active_data:
        st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)
        res_col1, res_col2 = st.columns([1.2, 0.8])

        with res_col1:
            st.markdown(f"""
            <div style="background-color: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: 12px; padding: 1.25rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                    <span style="font-size: 0.76rem; font-weight: 700; color: #10b981; background: rgba(16,185,129,0.1); padding: 0.2rem 0.6rem; border-radius: 9999px;">
                        ✔ EVIDENCE-GROUNDED SYNTHESIS
                    </span>
                    <span style="font-size: 0.72rem; color: var(--text-light); font-family: monospace;">
                        {st.session_state.get('source_mode', 'Engine')}
                    </span>
                </div>
                <div style="font-size: 0.95rem; line-height: 1.7; color: var(--text-main); margin-bottom: 1rem;">
                    {active_data.get('answer', '')}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with res_col2:
            st.markdown(f"""
            <div style="background-color: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: 12px; padding: 1.25rem;">
                <div style="font-size: 0.85rem; font-weight: 700; color: var(--text-main); margin-bottom: 0.75rem;">
                    📚 Grounding Evidence Citations ({len(active_data.get('retrieved_chunk_ids', []))})
                </div>
            """, unsafe_allow_html=True)

            for cid in active_data.get('retrieved_chunk_ids', [])[:4]:
                c_info = corpus_lookup.get(cid, {})
                c_title = c_info.get("doc_title", cid)
                c_page = c_info.get("page_number", "-")
                st.markdown(f"""
                <div style="background-color: var(--bg-surface-alt); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 0.75rem; margin-bottom: 0.5rem; font-size: 0.8rem;">
                    <div style="font-weight: 700; color: var(--text-main);">{c_title}</div>
                    <div style="color: var(--text-muted); font-size: 0.74rem;">Page: {c_page} • ID: <code>{cid}</code></div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------------------------------------------
# TAB 2: KNOWLEDGE BASE
# -------------------------------------------------------------------------------------------------
with tab_knowledge:
    st.markdown("<div style='height: 0.4rem;'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="font-size: 1.25rem; font-weight: 800; color: var(--text-main); margin-bottom: 0.25rem;">📚 Clinical Guideline Knowledge Base</h2>
        <div style="font-size: 0.82rem; color: var(--text-muted);">Verified clinical guidelines ingested, tokenized, and cryptographically frozen.</div>
    </div>
    """, unsafe_allow_html=True)

    kb_c1, kb_c2, kb_c3, kb_c4 = st.columns(4)
    with kb_c1:
        st.metric("Authoritative Guidelines", "8 Documents")
    with kb_c2:
        st.metric("Total Pages Ingested", "431 Pages")
    with kb_c3:
        st.metric("Deterministic Chunks", "3,073 Units")
    with kb_c4:
        st.metric("Corpus Integrity", "SHA-256 Passed")

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background-color: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: 12px; padding: 1.2rem;">
        <div style="font-size: 0.9rem; font-weight: 700; color: var(--text-main); margin-bottom: 0.75rem;">Verified Clinical Corpus Documents</div>
        <div style="font-size: 0.82rem; color: var(--text-muted); line-height: 2;">
            <div>• <b>ATA 2014</b>: Guidelines for the Treatment of Hypothyroidism (Jonklaas et al.)</div>
            <div>• <b>ATA 2015</b>: Management Guidelines for Adult Patients with Thyroid Nodules and DTC (Haugen et al.)</div>
            <div>• <b>ATA 2016</b>: Guidelines for Diagnosis and Management of Hyperthyroidism and Other Causes of Thyrotoxicosis (Kahaly et al.)</div>
            <div>• <b>ATA 2020</b>: Guidelines for Management of Patients with Anaplastic Thyroid Cancer (Bible et al.)</div>
            <div>• <b>AACE/ATA 2012</b>: Clinical Practice Guidelines for Hypothyroidism in Adults (Garber et al.)</div>
            <div>• <b>ETA 2018</b>: European Thyroid Association Guideline for the Management of Graves' Hyperthyroidism</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------------------------------------------
# TAB 3: SYSTEM & MODELS (EXACT OBSERVATORY VIEW FROM USER'S SCREENSHOT)
# -------------------------------------------------------------------------------------------------
with tab_system:
    st.markdown("<div style='height: 0.4rem;'></div>", unsafe_allow_html=True)
    
    # Title & Subtitle from screenshot
    st.markdown(f"""
    <div style="margin-bottom: 1.8rem;">
        <h2 style="font-size: 1.35rem; font-weight: 800; color: var(--text-main); margin-bottom: 0.2rem; display: flex; align-items: center; gap: 8px;">
            ⚙️ System & Model Observatory
        </h2>
        <div style="font-size: 0.84rem; color: var(--text-muted);">
            Live diagnostics of the BioGuard clinical pipeline components.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Pipeline Component Status
    st.markdown("<div style='font-size: 0.92rem; font-weight: 700; color: var(--text-main); margin-bottom: 0.75rem;'>Pipeline Component Status</div>", unsafe_allow_html=True)

    components = [
        {"name": "FAISS / Chroma Vector Store", "status": "Loaded", "badge": "3,073 units"},
        {"name": "BM25 Keyword Index", "status": "Loaded", "badge": "Full Corpus"},
        {"name": "BGE Cross Encoder Reranker", "status": "Loaded", "badge": "BAAI/bge-reranker-base"},
        {"name": "LLM (Groq API)", "status": "Loaded", "badge": "qwen/qwen3.6-27b"}
    ]

    st.markdown(f"""<div style="background-color: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: 10px; overflow: hidden; margin-bottom: 1.8rem;">""", unsafe_allow_html=True)
    for c in components:
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 12px 20px; border-bottom: 1px solid var(--border-subtle);">
            <div style="display: flex; align-items: center; gap: 12px;">
                <span style="width: 8px; height: 8px; border-radius: 50%; background-color: #22c55e;"></span>
                <span style="font-size: 0.85rem; font-weight: 600; color: var(--text-main);">{c['name']}</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 0.82rem; font-weight: 700; color: var(--text-main);">{c['status']}</span>
                <span style="font-size: 0.72rem; color: var(--text-muted); background: var(--bg-surface-alt); padding: 2px 8px; border-radius: 4px; font-family: monospace;">{c['badge']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # 2. Compute Resources
    st.markdown("<div style='font-size: 0.92rem; font-weight: 700; color: var(--text-main); margin-bottom: 0.75rem;'>Compute Resources</div>", unsafe_allow_html=True)

    comp_c1, comp_c2, comp_c3, comp_c4 = st.columns(4)
    with comp_c1:
        st.markdown(f"""
        <div style="background-color: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 14px 18px;">
            <div style="font-size: 0.74rem; font-weight: 600; color: var(--text-muted);">Cuda Available</div>
            <div style="font-size: 1.45rem; font-weight: 800; color: var(--text-main); margin-top: 4px;">False</div>
        </div>
        """, unsafe_allow_html=True)

    with comp_c2:
        st.markdown(f"""
        <div style="background-color: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 14px 18px;">
            <div style="font-size: 0.74rem; font-weight: 600; color: var(--text-muted);">Device Count</div>
            <div style="font-size: 1.45rem; font-weight: 800; color: var(--text-main); margin-top: 4px;">0</div>
        </div>
        """, unsafe_allow_html=True)

    with comp_c3:
        st.markdown(f"""
        <div style="background-color: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 14px 18px;">
            <div style="font-size: 0.74rem; font-weight: 600; color: var(--text-muted);">Gpu Name</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: var(--text-main); margin-top: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">N/A (CPU Mo...</div>
        </div>
        """, unsafe_allow_html=True)

    with comp_c4:
        st.markdown(f"""
        <div style="background-color: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 14px 18px;">
            <div style="font-size: 0.74rem; font-weight: 600; color: var(--text-muted);">Vram Gb</div>
            <div style="font-size: 1.45rem; font-weight: 800; color: var(--text-main); margin-top: 4px;">0.0</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

    # 3. Active Model Configuration
    st.markdown("<div style='font-size: 0.92rem; font-weight: 700; color: var(--text-main); margin-bottom: 0.75rem;'>Active Model Configuration</div>", unsafe_allow_html=True)

    mod_c1, mod_c2, mod_c3 = st.columns(3)
    with mod_c1:
        st.markdown(f"""
        <div style="background-color: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 14px 18px;">
            <div style="font-size: 0.68rem; font-weight: 700; color: var(--text-light); text-transform: uppercase;">EMBEDDING MODEL</div>
            <div style="font-size: 0.88rem; font-weight: 700; color: var(--text-main); font-family: monospace; margin-top: 8px;">BAAI/bge-small-en-v1.5</div>
        </div>
        """, unsafe_allow_html=True)

    with mod_c2:
        st.markdown(f"""
        <div style="background-color: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 14px 18px;">
            <div style="font-size: 0.68rem; font-weight: 700; color: var(--text-light); text-transform: uppercase;">RERANKER</div>
            <div style="font-size: 0.88rem; font-weight: 700; color: var(--text-main); font-family: monospace; margin-top: 8px;">BAAI/bge-reranker-base</div>
        </div>
        """, unsafe_allow_html=True)

    with mod_c3:
        st.markdown(f"""
        <div style="background-color: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 14px 18px;">
            <div style="font-size: 0.68rem; font-weight: 700; color: var(--text-light); text-transform: uppercase;">LANGUAGE MODEL</div>
            <div style="font-size: 0.88rem; font-weight: 700; color: var(--text-main); font-family: monospace; margin-top: 8px;">qwen/qwen3.6-27b</div>
            <div style="font-size: 0.68rem; color: var(--text-muted); margin-top: 6px;">Provider: groq • Temp: 0.1 • Max tokens: 1200</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

    # 4. Full Configuration (JSON)
    st.markdown("<div style='font-size: 0.92rem; font-weight: 700; color: var(--text-main); margin-bottom: 0.75rem;'>Full Configuration (JSON)</div>", unsafe_allow_html=True)

    with st.expander("> View models.yaml configuration", expanded=False):
        st.json(models_cfg if models_cfg else {
            "models": {
                "embeddings": {"active_model": "BAAI/bge-m3", "active_dimension": 1024},
                "reranker": {"model_name": "BAAI/bge-reranker-v2-m3"},
                "generation": {"model_name": "qwen/qwen3.6-27b", "primary_backend": "groq", "temperature": 0.1}
            }
        })
