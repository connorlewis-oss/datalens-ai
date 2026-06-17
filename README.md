# DataLens AI

**🔗 Live Demo → [connor-ai-analyst.streamlit.app](https://connor-ai-analyst.streamlit.app)**

An AI-powered business analytics tool that lets you upload any CSV dataset and instantly generate executive summaries, driver analysis, anomaly detection, forecasts, and natural language Q&A — no SQL or coding required.

Built by **Connor Lewis** | BS Business Analytics, Fairfield University | MS AI & Business Analytics (in progress)

![Python](https://img.shields.io/badge/Python-3.9+-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red) ![Claude](https://img.shields.io/badge/Claude-Haiku-orange)

---

## What It Does

Upload a spreadsheet. Ask a question. Get a consultant-level answer with a chart.

The app pre-calculates group summaries, correlations, anomaly scores, and driver rankings before sending anything to Claude — so answers are grounded in real numbers, not hallucinated estimates.

---

## Features

- **Plain English Q&A** — ask any business question and get a structured, data-grounded answer
- **Executive Summary** — AI-generated overview of key trends, outliers, and business insights on upload
- **Insight Cards** — top 4 highlights surfaced automatically from the data
- **Driver Analysis** — identifies which variables most strongly influence your key metric using Pearson correlation and eta-squared
- **Anomaly Detection** — flags statistical outliers using Z-score analysis
- **Trend & Forecast** — linear regression-based forecasting on time series data
- **What-If Simulator** — adjust key drivers and see projected impact on your target metric
- **Auto Charts** — bar and line charts generated automatically when relevant
- **Advanced AI Questions** — 8 consultant-level questions generated dynamically from your actual column names
- **Smart suggested questions** — auto-generated from column headers, or choose from industry templates
- **Data quality alerts** — flags missing values, duplicate rows, and numbers stored as text
- **Compare Two Files mode** — upload two datasets and get pre-calculated diffs (absolute and % change)
- **Downloadable HTML report** — export your full analysis session as a formatted report
- **Rate limiting** — 100 questions per session, 3-second cooldown, 10 MB file size cap

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| AI | Anthropic Claude API (claude-haiku-4-5-20251001) |
| Data | Pandas |
| Charts | Plotly Express |
| Environment | python-dotenv |

---

## How It Works

The core insight behind this app is that LLMs give inaccurate answers when asked to calculate from raw data rows. Instead, the app pre-calculates all the key statistics before sending anything to Claude:

1. **Column detection** — identifies categorical columns (Category, Segment, Region, etc.) and numeric columns (Sales, Profit, etc.), with a priority-first ordering so important business dimensions are always included
2. **Group summaries** — calculates sum, average, std deviation, min, max, count, and % of total for every useful column combination; grand totals are included in headers so Claude reads figures rather than computing them
3. **Time-series summaries** — detects date columns and builds monthly totals, yearly totals, and year-over-year growth rates
4. **Comparison diffs** — when two files are uploaded, pre-calculates absolute and percentage change by category so comparisons are accurate
5. **Structured prompt** — sends pre-calculated statistics to Claude with explicit instructions to copy numbers directly from the data, never estimate, and end with a McKinsey-level strategic insight

---

## Setup

### Prerequisites
- Python 3.9+
- An Anthropic API key ([get one here](https://console.anthropic.com))

### Install

```bash
git clone https://github.com/connorlewis-oss/datalens-ai.git
cd datalens-ai
pip install -r requirements.txt
```

### Configure

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### Run

```bash
python3 -m streamlit run app.py
```

The app opens at `http://localhost:8501`

---

## Sample Data

The app works with any CSV or Excel file. To test it, download the [Superstore Sales Dataset](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting) from Kaggle and try questions like:

- *"What is the total sales by category?"*
- *"Which customers have high sales but low profitability?"*
- *"Show me the monthly sales trend"*
- *"What is our revenue concentration risk across customers?"*

---

## Project Structure

```
ai-data-analyst/
├── app.py              # Main application
├── requirements.txt    # Dependencies
├── .env                # API key (not committed)
└── .gitignore          # Excludes .env
```

---

## About

Built as a resume project by Connor Lewis — BS in Business Analytics (Fairfield University), pursuing MS in AI and Business Analytics.

The goal was to demonstrate how AI can make data analysis accessible to non-technical business users, while showing technical proficiency in Python, API integration, prompt engineering, and data processing.

---

## License

MIT
