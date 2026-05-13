# ECB Foreign Exchange Analyst — Bayesian Deep Learning + RAG Chain

A Streamlit application that combines live ECB API data, pre-trained Variational Dropout MLPs (Molchanov et al. 2017), and a Retrieval-Augmented Generation chain to deliver grounded FX analysis. The system synthesises quantitative model predictions with qualitative context from official ECB speeches.

---

## Overview

This project demonstrates how LLM-powered text generation and Bayesian deep learning can be combined into a coherent analyst tool. The RAG chain retrieves semantically relevant ECB speeches and merges them with live model predictions to produce structured, cited financial reports.

**Example queries:**
- *"What is the current USD/EUR exchange rate and outlook?"*
- *"Show me the GBP/EUR Bayesian forecast for the next 30 days."*
- *"Give me a technical analysis of the JPY/EUR model."*
- *"What have ECB officials said recently about the euro exchange rate?"*

**Not currently supported:**
- Comparing multiple currency pairs in a single chart
- Macro or fundamental analysis beyond what ECB speeches contain
- Investment advice
- Currency pairs beyond USD, GBP, JPY, CHF vs EUR

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit UI (app.py)                         │
│   Forecast slider · Style selector · Chart · Analysis · History  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              QueryAgent (query_agent.py)                         │
│  Series classification (GPT-5.5 + Pydantic structured output)   │
│  ECB SDMX API fetch · load pretrained model · build live context │
└──────────────┬──────────────────────────┬───────────────────────┘
               │                          │
               ▼                          ▼
┌──────────────────────────┐  ┌──────────────────────────────────┐
│  FinancialVectorStore    │  │  BayesMolchanov.py               │
│  FAISS · OpenAI Embeds   │  │  VarMLP · load_and_predict       │
│  Source: ECB speeches    │  │  Posterior predictive (200x)     │
└──────────────┬───────────┘  └──────────────────────────────────┘
               │                          │
               ▼                          ▼
┌─────────────────────────────────────────────────────────────────┐
│           FinancialRAGChain (rag_chain.py)                       │
│                                                                  │
│  Retrieved ECB speeches  +  Live model predictions               │
│  ── ECB SPEECHES ──          ── LIVE MODEL & MARKET DATA ──     │
│  Semantic similarity         Posterior mean + σ checkpoints      │
│  Speaker · Date · Title      Registry metadata · ELBO loss       │
│                                                                  │
│  GPT-5.5 · temperature=0 · Audience-aware prompt                 │
│  → Structured analyst report (5 sections)                        │
└─────────────────────────────────────────────────────────────────┘
```

### Component breakdown

**`get_data.py`** fetches daily ECB FX reference rates via the ECB SDMX API for the last 365 trading days. Stored in `data/ecb_reports.json`. Used exclusively by the Bayesian MLP pipeline — does not feed the vector store.

**`get_reports.py`** downloads the ECB speeches dataset directly from the ECB website (no scraping — one HTTP request returns a clean pipe-separated CSV). Filters to the last 2 years, converts to structured JSON, and saves to `data/ecb_speeches.json`. This feeds the vector store.

**`BayesMolchanov.py`** implements the Variational Dropout MLP (Molchanov et al. 2017) using PyTorch autograd. Architecture: `[1 → 64 → 1]` with `sin` hidden activation. Models are **trained offline** in `notebooks/` and saved as `.pt` files with a `models/registry.json` index. At query time, `load_and_predict` loads pretrained weights and runs 200 stochastic forward passes to produce a posterior predictive mean and uncertainty band. No training at query time.

**`query_agent.py`** orchestrates the query pipeline: classifies which FX series the user is asking about via a single GPT-5.5 structured output call (Pydantic schema), fetches 365 days of fresh ECB data, loads the correct pretrained model, and returns live reports for chart rendering and RAG context. The vector store is not updated at query time — it holds speeches, managed separately.

**`FinancialVectorStore`** wraps FAISS with OpenAI's `text-embedding-3-small` (1536 dimensions). Built from ECB speeches and persisted to disk. Retrieval is semantic similarity — the most relevant speeches are returned regardless of date. This is a known limitation; date-aware agentic retrieval is a planned v2 improvement.

**`FinancialRAGChain`** builds hybrid context from two sources: retrieved ECB speeches (qualitative, official) and live model data via `TrendEnricher` (quantitative, model-generated). GPT-5.5 synthesises both into a five-section analyst report. The audience style (technical / balanced / non-technical) adapts the depth of model explanation and speech citation.

**`TrendEnricher`** reads model metadata from `models/registry.json` by series ID and extracts forecast checkpoints (mean ± σ) at day 1, midpoint, and end of horizon from the live posterior predictive. Injected into the RAG prompt as structured context.

**`chart.py`** renders a Plotly dark-theme chart: scattered observed data, posterior predictive mean (violet), ±1σ uncertainty ribbon, and a dashed forecast section beyond the last observation.

**`prompt.py`** externalised system prompt. Instructs the LLM to synthesise ECB speeches and model predictions into a structured report with strict grounding rules — no external knowledge, all claims attributed to source.

---

## Bayesian Model — Key Details

| Property | Value |
|---|---|
| Architecture | `[1 → 64 → 1]` |
| Hidden activation | `sin` |
| Training data | 365 days of daily ECB FX rates |
| Training schedule | 150 warmup epochs (KL=0, lr=0.1) + 150 annealing epochs (KL: 0→1, lr=0.01) |
| Inference | 200 stochastic forward passes (reparameterisation trick) |
| Output | Posterior predictive mean + ±1σ uncertainty ribbon + forecast checkpoints |
| Optimiser | Adam |
| Loss | ELBO (NLL + KL divergence) |

**Training is offline.** Models are trained in `notebooks/` and saved to `models/`. The app performs inference only — no training at query time. Hyperparameter tuning is done independently in the notebook environment and requires human judgment.

**Normalisation is fixed at training time.** The `x_mean`, `x_std`, `y_mean`, `y_std` computed on the training set are stored in `models/registry.json` and reused at inference. The app always fetches exactly 365 days to match the training window.

---

## RAG Chain — Design

The RAG chain combines two heterogeneous sources:

```
ECB Speeches (vector store)        Bayesian MLP (live)
─────────────────────────          ──────────────────────
What officials say                 What the data says
Qualitative · Attributed           Quantitative · Uncertain
Lagarde on EUR safe-haven flows    mean=1.1800, σ=0.0332 at day 7
Lane on EUR/USD financial conds    σ widens to 0.0344 at day 30
```

GPT-5.5 synthesises both into a five-section report:
1. Current rate
2. ECB policy context (from speeches, with attribution)
3. Forecast summary (from model checkpoints)
4. Model provenance
5. Limitations + follow-up

**Known limitation:** vector store retrieval is purely semantic — no recency weighting. A highly relevant speech from 6 months ago will rank above a less relevant recent one. Date-aware agentic retrieval is the natural v2 improvement.

---

## Project Structure

```
RAG_Financial_Analyst/
├── src/
│   ├── __init__.py
│   ├── BayesMolchanov.py    # VDM MLP, training utilities, load_and_predict
│   ├── document_loader.py   # ECB speech ingestion → LangChain Documents
│   ├── vector_store.py      # FAISS vector store creation, search, persistence
│   ├── rag_chain.py         # LCEL RAG chain, hybrid context, query validation
│   ├── query_agent.py       # Series classification, ECB fetch, model inference
│   ├── chart.py             # Plotly posterior predictive chart
│   ├── trend_enricher.py    # Live context formatter (model + market data)
│   └── prompt.py            # Externalised system prompt
├── models/
│   ├── registry.json        # Model index: architecture + normalisation constants
│   ├── ECB-FX-001.pt        # USD/EUR pretrained weights
│   ├── ECB-FX-002.pt        # GBP/EUR pretrained weights
│   ├── ECB-FX-003.pt        # JPY/EUR pretrained weights
│   └── ECB-FX-004.pt        # CHF/EUR pretrained weights
├── notebooks/               # Model training and experimentation (not committed)
├── tests/
│   ├── conftest.py
│   ├── test_document_loader.py
│   ├── test_vector_store.py
│   ├── test_rag_chain.py
│   └── test_integration.py
├── data/
│   ├── ecb_reports.json     # Generated by get_data.py — feeds MLP pipeline only
│   └── ecb_speeches.json    # Generated by get_reports.py — feeds vector store
├── vector_store/            # Persisted FAISS index (created on first run)
├── app.py                   # Streamlit interface
├── get_data.py              # ECB SDMX API — daily FX time series
├── get_reports.py           # ECB speeches CSV — qualitative text for RAG
├── requirements.txt
└── .env                     # OPENAI_API_KEY (not committed)
```

---

## Setup

### Prerequisites
- Python 3.11+
- OpenAI API key

### Installation

```bash
git clone https://github.com/Geist2307/RAG_VIMLP.git
cd RAG_VIMLP

python3 -m venv FinancePy
source FinancePy/bin/activate

pip install -r requirements.txt
```

### Environment

```
OPENAI_API_KEY=your_openai_api_key_here
```

### Data

```bash
# fetch time series data (feeds Bayesian MLP)
python get_data.py

# fetch ECB speeches (feeds vector store)
python get_reports.py
```

### Models

Pretrained weights are in `models/`. Trained offline on 365 days of daily ECB FX data. To retrain, open the notebook in `notebooks/` and use `save_model()` to update weights and `registry.json`.

### Run

```bash
python -m streamlit run app.py
```

On first run the vector store is built from `data/ecb_speeches.json` and saved to `vector_store/`. Subsequent runs load the persisted index.

---

## Usage

1. Set the **Forecast horizon** slider (7–90 days)
2. Select **Response style** (technical / balanced / non-technical)
3. Type a natural language query about ECB FX rates
4. The system:
   - Classifies which FX series you are asking about
   - Fetches 365 days of fresh data from the ECB SDMX API
   - Loads the pretrained Bayesian MLP and runs posterior predictive inference
   - Retrieves semantically relevant ECB speeches from the vector store
   - Synthesises both into a structured analyst report

---

## Dependencies

| Package | Purpose |
|---|---|
| `torch` | Variational Dropout MLP (autograd, Adam) |
| `langchain` / `langchain-community` | RAG chain orchestration, LCEL |
| `langchain-openai` | OpenAI embeddings and GPT-5.5 |
| `faiss-cpu` | Local vector similarity search |
| `plotly` | Interactive posterior predictive chart |
| `streamlit` | Web interface |
| `pydantic` | Structured intent extraction schema |
| `scipy` | KL divergence utilities |
| `markdown` | Render LLM markdown output in Streamlit |
| `python-dotenv` | Environment variable management |

---

## Limitations

- **Offline training.** Models do not adapt to very recent data. Retraining requires the notebook and human judgment on hyperparameters.
- **Fixed 365-day training window.** The app always fetches 365 days to match the training normalisation. Fetching fewer days would cause scale mismatch.
- **Four FX pairs only.** USD, GBP, JPY, CHF vs EUR. Other pairs and asset classes not modelled.
- **Semantic retrieval, no recency weighting.** A highly relevant older speech may rank above a less relevant recent one. Date-aware agentic retrieval is the planned v2 improvement.
- **200 posterior samples.** More samples improve uncertainty estimates but increase latency.
- **No streaming.** LLM responses render after full generation completes.

---

## Design Decisions

**Why offline training?** Training a Bayesian MLP reliably requires visual inspection of the loss curve and posterior predictive. Doing this at query time would be slow, unpredictable, and wasteful. Offline training decouples model quality from app latency.

**Why ECB speeches for RAG?** The time series data is already handled by the Bayesian MLP — there is no value in retrieving it again from a vector store. ECB speeches provide genuinely complementary qualitative context: what officials say about EUR dynamics, monetary policy stance, and forward guidance. One HTTP request returns a clean CSV with full speech text — no scraping required.

**Why separate `get_data.py` and `get_reports.py`?** Clean separation of concerns. Time series data feeds the MLP pipeline; speech data feeds the RAG pipeline. Each script has one job.

**Why `sin` activation?** Exchange rate series exhibit oscillatory, non-linear patterns that `sin` captures naturally. `tanh` and `ReLU` were tested and produced less stable posterior predictives on these series.

**Why GPT as explainer, not analyst?** The Bayesian MLP does the quantitative reasoning. GPT structures and communicates the output. This is a more honest and defensible architecture — the model does the work, the LLM makes it readable.

**Why `temperature=0`?** Financial analysis demands consistency. Non-zero temperature introduces unnecessary variance when citing specific rates and dates.

---

## Author

Built as a portfolio project demonstrating the combination of Bayesian deep learning, RAG chain design, and production Streamlit applications for financial analysis.