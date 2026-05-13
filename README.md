# ECB Foreign Exchange Forecasting — Bayesian Deep Learning + RAG

A Streamlit application that combines live ECB API data, pre-trained Variational Dropout MLPs (Molchanov et al. 2017), and a Retrieval-Augmented Generation pipeline to deliver real-time FX analysis and posterior predictive forecasts with uncertainty quantification.

---

## Overview

This project implements a full RAG pipeline over ECB foreign exchange data, enabling natural language queries over structured financial time-series. The system retrieves semantically relevant documents and generates grounded, cited responses. Alongside the text analysis, a Bayesian MLP produces a posterior predictive curve with a ±1σ uncertainty ribbon and a configurable forward forecast.

**Example queries:**
- *"What is the current USD/EUR exchange rate?"*
- *"Show me the GBP/EUR forecast for the next 30 days."*
- *"Give me a technical analysis of the JPY/EUR Bayesian model."*
- *"What does the CHF/EUR posterior predictive show for the next two weeks?"*

**Not currently supported:**
- Comparing multiple currency pairs in a single query
- Macro or fundamental analysis ("why is EUR strengthening?")
- Investment advice
- Currency pairs beyond USD, GBP, JPY, CHF vs EUR

---

## Foreign exchange rates

Currently, only 4 exchange rates are retrieved via the ECB API:

1. USD/EUR
2. GBP/EUR
3. JPY/EUR
4. CHF/EUR

Custom Variational(Bayesian) MLP models were trained offline and have been saved with the same unique identifier as the extracted datasets. Based on the query the agent selects the relevant models, and a forecast + uncertainty margin is presented asa visualisation. The unique advantage of the bayesian model is its probabilisistic weights, which allow naturally to derive confidence intervals. 

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit UI (app.py)                     │
│   Forecast slider · Chart · Analysis · Query history         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              QueryAgent (query_agent.py)                     │
│  Intent extraction (GPT-5.5 + Pydantic structured output)   │
│  ECB API fetch · load pretrained model · upsert vector store │
└──────────────┬──────────────────────────┬───────────────────┘
               │                          │
               ▼                          ▼
┌──────────────────────────┐  ┌──────────────────────────────┐
│  FinancialVectorStore    │  │  BayesMolchanov.py            │
│  FAISS · OpenAI Embeds   │  │  VarMLP · load_and_predict    │
│  (vector_store.py)       │  │  Posterior predictive (200x)  │
└──────────────┬───────────┘  └──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│           FinancialRAGChain (rag_chain.py)                   │
│  LCEL chain · Hybrid context (docs + live model data)        │
│  GPT-5.5 · temperature=0                                     │
└─────────────────────────────────────────────────────────────┘
```

### Component breakdown

**`get_data.py`** fetches daily ECB FX reference rates via the ECB SDMX API for the last 365 trading days. Data is stored in `data/ecb_reports.json` and used to bootstrap the vector store. No modelling happens here — this is pure data extraction via an API.

**`BayesMolchanov.py`** implements the Variational Dropout MLP from Molchanov et al. (2017) using PyTorch autograd. The architecture is `[1 → 64 → 1]` with a `sin` hidden activation. Models are **trained offline** (see `notebooks/`) and saved as `.pt` files with a `models/registry.json` index. At query time, `load_and_predict` loads the pretrained weights and runs 200 stochastic forward passes to produce a posterior predictive mean and uncertainty band — no training happens in the app. 

**`query_agent.py`** orchestrates the query pipeline: extracts intent via a single GPT-5.5 structured output call (Pydantic schema), fetches fresh ECB data, loads the correct pretrained model, and upserts new documents into the vector store. The agent always fetches 365 days to match the model's training window. 

**`FinancialVectorStore`** wraps FAISS with OpenAI's `text-embedding-3-small` model (1536 dimensions). The store is persisted to disk on first run and updated incrementally via `upsert_documents`.

**`FinancialRAGChain`** builds a hybrid context from two sources: retrieved vector store documents and live report data (model metadata + market facts). The system prompt instructs the LLM to cite model architecture, training schedule, ELBO loss, and latest observed values — and to refer the user to the chart for trend direction rather than speculating.

**`chart.py`** renders a Plotly dark-theme chart showing: scattered observed data, posterior predictive mean (violet), ±1σ uncertainty ribbon, and a dashed forecast section beyond the last observation, separated by a red dotted vertical line.

**`TrendEnricher`** (`trend_enricher.py`) formats live report data as a structured string injected into the RAG context. It maps query keywords to relevant series IDs so only pertinent FX pairs are included.

---

## Bayesian Model — Key Details

The forecasting backbone is a **Variational Dropout MLP** (Molchanov et al. 2017):

| Property | Value |
|---|---|
| Architecture | `[1 → 64 → 1]` |
| Hidden activation | `sin` |
| Training data | 365 days of daily ECB FX rates |
| Training schedule | 150 warmup epochs (KL=0, lr=0.1) + 150 annealing epochs (KL: 0→1, lr=0.01) |
| Inference | 200 stochastic forward passes (reparameterisation trick) |
| Output | Posterior predictive mean + ±1σ uncertainty ribbon |
| Optimiser | Adam |
| Loss | ELBO (negative log-likelihood + KL divergence) |

**Training is offline.** Models are trained in `notebooks/` and saved to `models/`. The app only performs inference — no training at query time. This keeps the app fast and predictions consistent.

**Normalisation** is fixed at training time. The `x_mean`, `x_std`, `y_mean`, `y_std` computed on the training set are stored in `models/registry.json` and reused at inference time. The app always fetches 365 days to match the training window, ensuring the model receives inputs in the same normalised space it was trained on.

---

## Project Structure

```
RAG_Financial_Analyst/
├── src/
│   ├── __init__.py
│   ├── BayesMolchanov.py    # VDM MLP, training, inference, load_and_predict
│   ├── document_loader.py   # ECB report ingestion and Document conversion
│   ├── vector_store.py      # FAISS vector store creation, search, persistence
│   ├── rag_chain.py         # LCEL RAG chain, hybrid context, query validation
│   ├── query_agent.py       # Intent extraction, ECB fetch, model inference
│   ├── chart.py             # Plotly posterior predictive chart
│   └── trend_enricher.py    # Live context formatter for RAG prompt
├── models/
│   ├── registry.json        # Model index with architecture + normalisation constants
│   ├── ECB-FX-001.pt        # USD/EUR pretrained weights
│   ├── ECB-FX-002.pt        # GBP/EUR pretrained weights
│   ├── ECB-FX-003.pt        # JPY/EUR pretrained weights
│   └── ECB-FX-004.pt        # CHF/EUR pretrained weights
├── tests/
│   ├── conftest.py
│   ├── test_document_loader.py
│   ├── test_vector_store.py
│   ├── test_rag_chain.py
│   └── test_integration.py
├── data/
│   └── ecb_reports.json     # Generated by get_data.py
├── vector_store/            # Persisted FAISS index (created on first run)
├── app.py                   # Streamlit interface
├── get_data.py              # ECB API data fetcher
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
git clone https://github.com/yourusername/RAG_Financial_Analyst.git
cd RAG_Financial_Analyst

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### Environment

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_openai_api_key_here
```

### Data

Fetch the latest 365 days of ECB daily FX rates:

```bash
python get_data.py
```

This writes to `data/ecb_reports.json` and bootstraps the vector store on first app launch.

### Models

Pretrained model weights are stored in `models/`. They were trained offline on 365 days of daily ECB FX data using the notebooks in `notebooks/` (not included in this repository). The `models/registry.json` file contains architecture parameters and normalisation constants required for inference.


### Run

```bash
python -m streamlit run app.py
```

On first run the vector store is built from `data/ecb_reports.json` and saved to `vector_store/`. Subsequent runs load the persisted index.

---

## Usage

1. Use the **Forecast horizon** slider to set how many days ahead the model predicts (7–90 days)
2. Type a natural language query about ECB FX rates
3. The system fetches fresh data from the ECB API, loads the pretrained model, and returns:
   - A Plotly chart with observed data, posterior mean, ±1σ ribbon, and forecast
   - A text analysis citing model metadata and latest observed values
   - Source documents from the vector store

---

## Dependencies

| Package | Purpose |
|---|---|
| `torch` | Variational Dropout MLP (autograd, Adam) |
| `langchain` / `langchain-community` | RAG orchestration, LCEL chain |
| `langchain-openai` | OpenAI embeddings and chat model |
| `faiss-cpu` | Local vector similarity search |
| `plotly` | Interactive posterior predictive chart |
| `streamlit` | Web interface |
| `pydantic` | Structured intent extraction schema |
| `scipy` | KL divergence utilities |
| `python-dotenv` | Environment variable management |

---

## Limitations

- **Models are trained offline.** The app does not retrain models at query time. If ECB rates shift significantly beyond the training distribution, prediction quality may degrade. Retraining should be done separately.
- **Fixed training window.** Models are trained on exactly 365 days. The app always fetches 365 days to match this window — using fewer days would cause normalisation mismatch.
- **Four FX pairs only.** Currently covers USD/EUR, GBP/EUR, JPY/EUR, CHF/EUR. Other ECB series (inflation, interest rates) are not modelled.
- **200 posterior samples.** The uncertainty estimate is based on 200 stochastic forward passes. More samples improve the estimate but increase latency.
- **No streaming.** LLM responses render after full generation completes.
- **FAISS is local.** The vector store runs in-process and is not shared across sessions or deployments.

---

## Design Decisions

**Why offline training?** Training a Bayesian MLP reliably requires careful hyperparameter tuning and visual inspection of the loss curve and posterior predictive. Doing this at query time would be slow, unpredictable, and wasteful. Offline training decouples model quality from app latency.

**Why `sin` activation?** Exchange rate series exhibit oscillatory, non-linear patterns that `sin` captures naturally. `tanh` and `ReLU` were tested and produced less stable posterior predictives on these series.

**Why FAISS over a hosted vector DB?** For a dataset of this scale, a hosted vector database adds operational overhead without meaningful benefit. FAISS runs in-process, persists to disk, and loads in milliseconds.

**Why `temperature=0`?** Financial analysis demands consistency and factual grounding. Non-zero temperature introduces unnecessary variance when citing specific rates and dates.

**Why structured output for intent extraction?** Pydantic + `with_structured_output` is safer and more maintainable than prompting the LLM to return JSON and parsing it manually. The schema is self-documenting and validated automatically.

---

## Author

Built as a portfolio exercise in Bayesian deep learning, RAG system design, LangChain LCEL patterns, and production Streamlit applications.