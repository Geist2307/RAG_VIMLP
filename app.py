import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from dotenv import load_dotenv
load_dotenv()

import logging
from datetime import datetime

import streamlit as st

from src.document_loader import FinancialDocumentLoader
from src.rag_chain import FinancialRAGChain
from src.vector_store import FinancialVectorStore
from src.query_agent import fetch_and_upsert
from src.chart import build_trend_chart

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("financial_rag")

VECTOR_STORE_DIR = "vector_store"
DATA_PATH        = "data/ecb_reports.json"

st.set_page_config(
    page_title="ECB Financial RAG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
.stApp { background-color: #0d1117; color: #e6edf3; }
.ecb-header {
    display: flex; align-items: center; gap: 14px;
    padding: 2rem 0 0.5rem 0;
    border-bottom: 1px solid #21262d; margin-bottom: 1rem;
}
.ecb-header h1 { font-size: 1.8rem; font-weight: 700; color: #e6edf3; margin: 0; letter-spacing: -0.5px; }
.ecb-header .badge {
    background: #1f6feb22; border: 1px solid #1f6feb55; color: #58a6ff;
    font-size: 0.7rem; font-weight: 600; padding: 2px 8px;
    border-radius: 20px; text-transform: uppercase; letter-spacing: 1px;
}
.ecb-subtitle { color: #8b949e; font-size: 0.95rem; margin-bottom: 1rem; }
.status-card { background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 1rem 1.25rem; margin-bottom: 1rem; font-size: 0.9rem; }
.status-card.success { border-left: 3px solid #3fb950; }
.status-card.error   { border-left: 3px solid #f85149; }
.status-card.info    { border-left: 3px solid #58a6ff; }
.status-card.warning { border-left: 3px solid #d29922; }
.stTextInput > div > div > input {
    background-color: #161b22 !important; border: 1px solid #30363d !important;
    color: #e6edf3 !important; border-radius: 8px !important;
    font-size: 0.95rem !important; padding: 0.75rem 1rem !important;
}
.stTextInput > div > div > input:focus { border-color: #1f6feb !important; box-shadow: 0 0 0 3px #1f6feb22 !important; }
.stButton > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd); color: white;
    border: none; border-radius: 8px; padding: 0.6rem 1.8rem;
    font-weight: 600; font-size: 0.9rem; transition: all 0.2s ease; width: 100%;
}
.stButton > button:hover { background: linear-gradient(135deg, #388bfd, #58a6ff); transform: translateY(-1px); box-shadow: 0 4px 15px #1f6feb44; }
.analysis-box {
    background: #161b22; border: 1px solid #21262d; border-left: 3px solid #3fb950;
    border-radius: 8px; padding: 1.5rem; margin: 1rem 0;
    line-height: 1.7; font-size: 0.95rem; color: #e6edf3;
}
.source-card { background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 1rem 1.25rem; margin-bottom: 0.75rem; font-size: 0.875rem; }
.source-card .source-title { font-weight: 600; color: #58a6ff; margin-bottom: 0.4rem; }
.source-card .source-meta { color: #8b949e; font-size: 0.8rem; margin-bottom: 0.5rem; }
.source-card .source-preview { color: #c9d1d9; line-height: 1.6; border-top: 1px solid #21262d; padding-top: 0.5rem; margin-top: 0.5rem; }
.chip { display: inline-block; background: #21262d; border: 1px solid #30363d; color: #8b949e; font-size: 0.75rem; padding: 2px 10px; border-radius: 20px; margin-right: 6px; }
.error-box { background: #1a0a0a; border: 1px solid #f8514944; border-left: 3px solid #f85149; border-radius: 8px; padding: 1rem 1.25rem; color: #ffa198; font-size: 0.9rem; font-family: 'JetBrains Mono', 'Fira Code', monospace; }
.trend-stat { background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 0.5rem; font-size: 0.85rem; }
.trend-label { color: #8b949e; font-size: 0.75rem; margin-bottom: 2px; }
.trend-value { color: #e6edf3; font-weight: 600; }
.control-card { background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 0.75rem 1.25rem; margin-bottom: 1.5rem; }
hr { border-color: #21262d !important; margin: 1.5rem 0 !important; }
.stSpinner > div { border-top-color: #58a6ff !important; }
</style>
""", unsafe_allow_html=True)


def log_error(e: Exception) -> str:
    logger.error(e, exc_info=True)
    return str(e)


def get_documents():
    loader = FinancialDocumentLoader(DATA_PATH)
    return loader.create_documents()


def create_new_vector_store(progress):
    vector_store = FinancialVectorStore()
    progress.progress(20, "Creating vector store...")
    documents = get_documents()
    if not documents:
        raise ValueError("No documents loaded from data file.")
    progress.progress(50, "Generating embeddings...")
    vector_store.create_vector_store(documents)
    progress.progress(80, "Saving to disk...")
    vector_store.save_local(VECTOR_STORE_DIR)
    progress.progress(100, "Done.")
    return vector_store


def load_existing_vector_store(progress):
    progress.progress(30, "Loading vector store...")
    store = FinancialVectorStore.load_local(VECTOR_STORE_DIR)
    progress.progress(100, "Done.")
    return store


def initialize_rag_system():
    progress = st.progress(0, "Initialising RAG system...")
    try:
        try:
            vector_store = load_existing_vector_store(progress)
            logger.info("Loaded existing vector store.")
        except Exception as e:
            logger.warning(f"Could not load existing store ({e}), creating new one.")
            progress.progress(0, "Building vector store from scratch...")
            vector_store = create_new_vector_store(progress)

        if vector_store is None:
            raise ValueError("Vector store initialisation returned None.")

        rag_chain = FinancialRAGChain(vector_store)
        progress.empty()
        return rag_chain, vector_store

    except Exception as e:
        progress.empty()
        err = log_error(e)
        st.markdown(f'<div class="error-box">Initialisation failed<br><br>{err}</div>', unsafe_allow_html=True)
        return None, None


def render_trend_stats(report: dict):
    trend      = report.get("trend", {})
    model_info = report.get("model_info", {})
    trained_on = trend.get("trained_on", "N/A")
    n_future   = trend.get("n_future", 30)
    n_samples  = trend.get("n_samples", 200)

    st.markdown(f"""
    <div class="trend-stat">
        <div class="trend-label">Model trained on</div>
        <div class="trend-value">{trained_on}</div>
    </div>
    <div class="trend-stat">
        <div class="trend-label">Forecast horizon</div>
        <div class="trend-value">{n_future} days</div>
    </div>
    <div class="trend-stat">
        <div class="trend-label">Posterior samples</div>
        <div class="trend-value">⚠ {n_samples} samples used</div>
    </div>
    <div class="trend-stat">
        <div class="trend-label">Architecture</div>
        <div class="trend-value">[1 → {model_info.get('hidden', 64)} → 1] · {model_info.get('activation', 'sin')}</div>
    </div>
    <div class="trend-stat">
        <div class="trend-label">Training schedule</div>
        <div class="trend-value">{model_info.get('warmup_epochs', 150)} warmup · {model_info.get('anneal_epochs', 150)} anneal</div>
    </div>
    <div class="trend-stat">
        <div class="trend-label">Learning rates</div>
        <div class="trend-value">{model_info.get('lr_warmup', 0.1)} → {model_info.get('lr_anneal', 0.01)}</div>
    </div>
    <div class="trend-stat">
        <div class="trend-label">Final ELBO loss</div>
        <div class="trend-value">{model_info.get('final_loss', 'N/A')}</div>
    </div>
    <div class="trend-stat">
        <div class="trend-label">Trained on N obs</div>
        <div class="trend-value">{model_info.get('n_obs', 'N/A')} days</div>
    </div>
    """, unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
if "rag_chain" not in st.session_state or "vector_store" not in st.session_state:
    rag_chain, vector_store = initialize_rag_system()
    st.session_state.rag_chain    = rag_chain
    st.session_state.vector_store = vector_store

if "history" not in st.session_state:
    st.session_state.history = []


# ── Main UI ────────────────────────────────────────────────────────────────────
def main():

    # header
    st.markdown("""
    <div class="ecb-header">
        <span style="font-size:2rem">📊</span>
        <div><h1>ECB Financial Insights</h1></div>
        <span class="badge">RAG · Bayesian · GPT-5.5</span>
    </div>
    <p class="ecb-subtitle">
        Query ECB exchange rate data with AI-powered retrieval and Bayesian trend analysis.
        Sources are cited from official ECB publications.
    </p>
    """, unsafe_allow_html=True)

    # ── controls under title ───────────────────────────────────────
    ctrl_col1, ctrl_col2 = st.columns([2, 6])
    with ctrl_col1:
        st.markdown('<div class="control-card">', unsafe_allow_html=True)
        forecast_days = st.slider(
            "Forecast horizon (days)",
            min_value=7,
            max_value=90,
            value=30,
            step=7,
            help="How many days ahead the Bayesian MLP will predict"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    with ctrl_col2:
        st.markdown(
            '<div class="control-card" style="color:#8b949e; font-size:0.85rem; margin-top:4px">'
            '🧠 Variational Dropout MLP (Molchanov et al. 2017) · '
            '200 posterior samples · '
            'Pre-trained on 365 days of ECB daily FX data · '
            'Zoom and pan the chart to inspect any window'
            '</div>',
            unsafe_allow_html=True
        )

    # ── search bar ─────────────────────────────────────────────────
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        query = st.text_input(
            label="query",
            label_visibility="collapsed",
            placeholder="e.g. What is the current USD/EUR exchange rate?",
        )
    with col_btn:
        search = st.button("Search", use_container_width=True)

    if not st.session_state.rag_chain:
        st.markdown("""
        <div class="status-card error">
        RAG system failed to initialise. Check the error above and verify your
        <code>.env</code> contains valid API keys.
        </div>
        """, unsafe_allow_html=True)
        return

    if search and query:
        try:
            # 1. intent extraction + live fetch + upsert
            with st.spinner("Extracting intent and fetching fresh ECB data..."):
                agent_result = fetch_and_upsert(
                    query,
                    st.session_state.vector_store,
                    save_path=VECTOR_STORE_DIR,
                    forecast_days=forecast_days,
                )

            days    = agent_result["intent"].days
            added   = agent_result["added"]
            reports = agent_result["reports"]

            if added > 0:
                st.markdown(
                    f'<div class="status-card success">✓ Fetched {days} days of data — '
                    f'{added} new series added to vector store.</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="status-card info">ℹ Fetched {days} days — '
                    f'vector store up to date.</div>',
                    unsafe_allow_html=True
                )

            # 2. RAG query — pass live reports for context
            with st.spinner("Retrieving relevant ECB documents..."):
                relevant_docs = st.session_state.rag_chain.get_relevant_documents(query)

            with st.spinner("Generating analysis..."):
                response = st.session_state.rag_chain.query(query, reports=reports)

            st.session_state.history.append({
                "query":    query,
                "response": response,
                "docs":     relevant_docs,
                "reports":  reports,
                "time":     datetime.now().strftime("%H:%M:%S"),
            })

            # 3. Bayesian charts FIRST
            if reports:
                st.markdown("### Bayesian Trend Visualisation")
                tabs = st.tabs([r.get("currency_pair") or r["reportId"] for r in reports])
                for tab, report in zip(tabs, reports):
                    with tab:
                        fig = build_trend_chart(report)
                        if fig:
                            chart_col, stats_col = st.columns([3, 1])
                            with chart_col:
                                st.plotly_chart(fig, use_container_width=True)
                            with stats_col:
                                render_trend_stats(report)
                        else:
                            st.markdown(
                                '<div class="status-card warning">⚠ Insufficient data to render chart.</div>',
                                unsafe_allow_html=True
                            )

            # 4. Analysis AFTER chart
            st.markdown("### Analysis")
            st.markdown(f'<div class="analysis-box">{response}</div>', unsafe_allow_html=True)

            # 5. Source documents
            st.markdown("### Source Documents")
            for i, doc in enumerate(relevant_docs, 1):
                meta = doc["metadata"]
                st.markdown(f"""
                <div class="source-card">
                    <div class="source-title">{i}. {meta.get('title', 'Unknown')}</div>
                    <div class="source-meta">
                        <span class="chip">📅 {meta.get('publication_date', 'N/A')}</span>
                        <span class="chip">🏷 {meta.get('report_id', 'N/A')}</span>
                        <span class="chip">✍️ {meta.get('author', 'ECB')}</span>
                    </div>
                    <div class="source-preview">{doc['content'][:350]}...</div>
                </div>
                """, unsafe_allow_html=True)

        except ValueError as e:
            st.markdown(f'<div class="status-card warning">⚠️ {str(e)}</div>', unsafe_allow_html=True)
        except Exception as e:
            err = log_error(e)
            st.markdown(f'<div class="error-box">Unexpected error<br><br>{err}</div>', unsafe_allow_html=True)

    # ── query history ──────────────────────────────────────────────
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### Query History")
        for item in reversed(st.session_state.history[-5:]):
            with st.expander(f"🕐 {item['time']} — {item['query'][:80]}"):
                if item.get("reports"):
                    for report in item["reports"]:
                        if report is None:
                            continue
                        fig = build_trend_chart(report)
                        if fig:
                            st.plotly_chart(
                                fig,
                                use_container_width=True,
                                key=f"{item['time']}_{report['reportId']}"
                            )
                st.markdown(f'<div class="analysis-box">{item["response"]}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()