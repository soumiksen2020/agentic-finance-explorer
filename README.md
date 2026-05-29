# 🏛️ Multi-Agent Financial Intelligence Committee (NSE/BSE)

[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![CrewAI](https://img.shields.io/badge/Framework-CrewAI-ff69b4?style=flat)](https://www.crewai.com/)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Cloud-Native](https://img.shields.io/badge/Cloud-Render%20%7C%20Streamlit-663399?style=flat)](https://render.com)

An autonomous investment-research assistant for Indian (NSE/BSE) stocks. A crew of three specialized AI agents performs technical and news-driven analysis, a Chief Risk Officer agent audits their findings, and the result is scored by an automated evaluation layer and traced in Langfuse. A Streamlit dashboard presents live prices, fundamentals, an interactive chart, and the committee's verdict.

🔗 **[Live Dashboard](https://agentic-finance-explorer-zrkwkgnuyidyfgbqc8jb4a.streamlit.app/)** | 📖 **[API Documentation](https://agentic-finance-explorer.onrender.com/docs)**

---

## 🚀 Architecture

The system uses a **decoupled pattern** that separates the reasoning engine from the presentation layer:

- **Reasoning Engine (`app.py`):** A FastAPI server that orchestrates the CrewAI agents in a background task, caches results in SQLite, exposes fundamentals, and traces every run in Langfuse.
- **Agent Crew (`main.py`):** Three CrewAI agents wired with deterministic and search tools, producing a Pydantic-validated JSON report.
- **Evaluation Layer (`evaluator.py`):** A rule-based consistency check plus an LLM-as-judge that scores each report; scores are pushed to Langfuse.
- **Frontend (`frontend.py`):** A Streamlit dashboard with a multi-source live-price engine, fundamentals snapshot, and interactive Plotly chart.
- **Deterministic Tool (`tools.py`):** A CrewAI tool that computes RSI and the 20-day moving average from yfinance data.

### Request flow

1. The Streamlit frontend `POST`s a ticker to `/analyze`.
2. The backend fetches a live price and checks the SQLite cache. If the price has moved less than **0.5%** and the last run was **within 1 hour**, the cached report is returned immediately.
3. Otherwise a background task kicks off the CrewAI crew; the frontend polls `/status/{job_id}` (up to 25 times, every 5s).
4. When the crew finishes, the report is saved to SQLite, scored by the evaluator, and the full run (timing + scores) is logged to Langfuse.

## 🧠 The "Committee" (Agents)

Defined in `main.py`, all running on `openai/gpt-4o-mini` (temperature 0):

1. **Senior Quant Researcher** — Calls the `stock_price_analyzer` tool (`tools.py`) to fetch price, RSI(14), and MA20 from yfinance + `pandas_ta`. Deterministic, not LLM guesswork.
2. **Financial News Correspondent** — Uses `SerperDevTool` (Serper Google-search API) to gather recent news and sentiment. (Its prompt directs it toward Moneycontrol / Economic Times / LiveMint, but search is performed via Serper, not direct scraping.)
3. **Chief Risk Officer (adversarial)** — Audits the Quant and News findings to surface concrete risks (regulatory, promoter, macro), assigns a sentiment score (0–10), and emits the final structured JSON.

The final report conforms to the `FinancialAnalysisOutput` Pydantic schema:

| Field | Type | Meaning |
| :--- | :--- | :--- |
| `ticker` | str | Ticker analyzed |
| `technical_signal` | str | `Bullish` / `Bearish` / `Neutral` |
| `sentiment_score` | float | 1.0 (extreme fear) – 10.0 (extreme greed) |
| `key_catalysts` | list[str] | 3 bullets of upside drivers |
| `risk_summary` | list[str] | 3 bullets of critical risks |

## 📊 Evaluation Layer

After each analysis (`evaluator.py`), two evaluators score the report and the results are attached to the Langfuse trace:

- **Signal consistency (rule-based, 0.0/1.0):** Checks that `technical_signal` and `sentiment_score` agree (e.g. a `Bullish` call should have a score > 5.5). No LLM, no cost.
- **LLM-as-judge (`gpt-4o-mini`):** Scores `risk_specificity` (1–5), `catalyst_specificity` (1–5), and `overall_quality` (1–10), penalizing generic boilerplate. Falls back to neutral scores on failure so it never blocks the user's result.

## 🔌 API Endpoints

| Method | Path | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Health check (`{"status": "AI Agents Online", "version": "2.0"}`) |
| `GET` | `/fundamentals/{ticker}` | Market cap, P/E, 52w high/low, EPS, book value, dividend yield, ROCE, ROE, D/E — derived in 3 layers from yfinance `fast_info`, `get_info()`, and financial statements, with NSE→BSE fallback |
| `POST` | `/analyze` | Body `{"ticker": "..."}`. Returns cached result or `{"job_id", "status": "started"}` |
| `GET` | `/status/{job_id}` | Poll job status: `pending` / `completed` / `failed` / `not_found` |

## 🖥️ Frontend Features

- **Live-price engine (`get_current_price`):** Tries Groww's live-price API first, then scrapes Google Finance, then falls back to yfinance history. Auto-routes NSE vs BSE based on the ticker suffix.
- **Fundamental Snapshot:** Pulls the backend `/fundamentals` endpoint (cached 30 min) into a collapsible panel.
- **Interactive chart:** Plotly line/candlestick toggle across 1D / 1M / 3M / 6M timeframes.
- **Committee verdict:** Technical signal, sentiment score, color-coded catalysts and risk audit.

## 🛠️ Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Agent Framework** | CrewAI + crewai-tools (SerperDevTool) |
| **LLM** | GPT-4o-mini (OpenAI), used by both the crew and the judge |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | Streamlit + Plotly |
| **Data** | yfinance, pandas, pandas-ta, BeautifulSoup |
| **Validation** | Pydantic |
| **Observability** | Langfuse |
| **Persistence** | SQLite (`market_data.db`) |
| **Cloud** | Render (API), Streamlit Cloud (UI) |

## 🌟 Engineering Notes

- **Defensive parsing:** If the crew's output lacks a valid `json_dict`, the backend builds a fallback report instead of crashing (`app.py`).
- **Async background tasks:** FastAPI `BackgroundTasks` run the long (~45s+) agentic loop without blocking the request.
- **Smart caching:** SQLite-backed; a cached report is reused only if price moved < 0.5% **and** the last run was < 1 hour ago.
- **Restricted CORS:** Origins come from the `ALLOWED_ORIGINS` env var (default: the Streamlit app URL), limited to `GET`/`POST` and the `Content-Type` header. To allow another frontend, set `ALLOWED_ORIGINS` to a comma-separated list.

---

## ⚙️ Local Setup

### 1. Clone & install (using [`uv`](https://github.com/astral-sh/uv))

```bash
git clone https://github.com/merchantkevin/agentic-finance-explorer.git
cd agentic-finance-explorer
uv sync
```

### 2. Environment variables

Create a `.env` file in the project root:

```bash
# Required
OPENAI_API_KEY=sk-...
SERPER_API_KEY=...

# Optional — Langfuse observability (tracing/scoring still runs without keys but won't be persisted)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com

# Optional — restrict CORS to your frontend (comma-separated)
ALLOWED_ORIGINS=http://localhost:8501
```

### 3. Run the backend (FastAPI)

```bash
uv run uvicorn app:app --reload --port 8000
```

API docs are then available at `http://localhost:8000/docs`.

### 4. Run the frontend (Streamlit)

```bash
uv run streamlit run frontend.py
```

> **Note:** `frontend.py` currently points its `backend_url` at the hosted Render API (`https://agentic-finance-explorer.onrender.com`). To use your local backend, update those URLs in `frontend.py`.

---

## ☁️ Deployment

- **Backend:** Deployed on Render via `render.yml` (`uvicorn app:app`). It declares `OPENAI_API_KEY` and `SERPER_API_KEY`; add the Langfuse keys in the Render dashboard if you want tracing in production. The build command expects a `requirements.txt` — generate one with `uv export --no-hashes -o requirements.txt` (the repo ships `pyproject.toml` / `uv.lock`).
- **Frontend:** Deployed on Streamlit Cloud from `frontend.py`.

> **Disclaimer:** This application uses Large Language Models to synthesize public financial data for informational purposes only. It is **not** financial advice.
