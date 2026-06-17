import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import os
import base64
import anthropic
import plotly.express as px
import plotly.graph_objects as go
import json
import re
import html as html_lib
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ── API key (check happens after set_page_config below) ───────────────────────
_api_key = os.getenv("ANTHROPIC_API_KEY")

# ── Rate limiting config ───────────────────────────────────────────────────────
MAX_REQUESTS_PER_SESSION = 100  # max AI calls per browser session
COOLDOWN_SECONDS = 3            # minimum seconds between requests
MAX_FILE_SIZE_MB = 10           # max upload size in MB

# ── Column priority list (shared by build_df_info and metrics) ─────────────────
PRIORITY_NUM_COLS = ["sales", "revenue", "profit", "amount", "total",
                     "price", "cost", "spend", "income", "earnings", "value"]
SKIP_NUM_COLS = ["id", "row", "postal", "code", "zip"]

# ── Session state ──────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "question_input" not in st.session_state:
    st.session_state.question_input = ""
if "request_count" not in st.session_state:
    st.session_state.request_count = 0
if "use_sample" not in st.session_state:
    st.session_state.use_sample = False
if "last_request_time" not in st.session_state:
    st.session_state.last_request_time = None

# ── Industry templates ─────────────────────────────────────────────────────────
INDUSTRY_TEMPLATES = {
    "Auto-detect from data": [],
    "Sales & Revenue": [
        "What are the top 5 products by total sales?",
        "Which region had the highest profit margin?",
        "Which customer segment is most profitable?",
        "Are there products selling well but losing money?",
        "What is the total sales by category?",
        "What is the average order value by region?"
    ],
    "Human Resources": [
        "What is the headcount by department?",
        "Which department has the highest turnover?",
        "What is the average salary by job level?",
        "What is the gender breakdown by department?",
        "Which department has the most vacancies?",
        "What is the average employee tenure?"
    ],
    "Finance": [
        "What is total revenue vs total expenses?",
        "Which cost category is the largest?",
        "What is the profit margin by month?",
        "Which department has the highest budget variance?",
        "What are the top 5 expense categories?",
        "What is the year-over-year growth rate?"
    ],
    "Marketing": [
        "Which campaign had the highest conversion rate?",
        "What is the cost per acquisition by channel?",
        "Which channel drives the most revenue?",
        "What is the ROI by campaign?",
        "Which audience segment converts best?",
        "What is the average click-through rate?"
    ],
    "Inventory": [
        "Which products are below reorder point?",
        "What is the inventory turnover by category?",
        "Which items have the highest carrying cost?",
        "What is the average days to sell by product?",
        "Which supplier has the most delays?",
        "What is the stockout rate by category?"
    ]
}

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataLens AI",
    page_icon="📊",
    layout="wide"
)

# ── Anthropic client (must be after set_page_config so st.error renders) ──────
if not _api_key:
    st.error(
        "**ANTHROPIC_API_KEY not found.** "
        "Create a `.env` file in the project folder with:\n\n"
        "```\nANTHROPIC_API_KEY=sk-ant-your-key-here\n```\n\n"
        "Get a key at [console.anthropic.com](https://console.anthropic.com)."
    )
    st.stop()
client = anthropic.Anthropic(api_key=_api_key)

st.markdown("""
<style>
    /* ── Global — dot-grid background ── */
    .stApp {
        background-color: #edf1fa;
        background-image: radial-gradient(#c2d0ea 1px, transparent 1px);
        background-size: 24px 24px;
    }
    .main { padding: 1.5rem 2.5rem; }

    /* ── Typography ── */
    h1 { font-size: 1.8rem !important; font-weight: 800 !important; color: #0d1b2a !important;
         letter-spacing: -0.02em !important; }
    h3 { font-size: 1rem !important; font-weight: 700 !important;
         color: #0d1b2a !important; margin-top: 1.5rem !important;
         text-transform: uppercase; letter-spacing: 0.04em !important; }

    /* ── Suggestion pill buttons ── */
    .stButton > button {
        background: linear-gradient(white, white) padding-box,
                    linear-gradient(135deg, #c8d8f5, #ddd0fb) border-box;
        border: 1.5px solid transparent;
        color: #1a4a8a;
        border-radius: 9px;
        padding: 0.5rem 1rem;
        font-size: 0.82rem;
        font-weight: 500;
        transition: all 0.18s;
        width: 100%;
        text-align: left;
        white-space: normal;
        height: auto;
        line-height: 1.4;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #0066cc, #6366f1) !important;
        color: white !important;
        border-color: transparent;
        box-shadow: 0 4px 14px rgba(0,102,204,0.28);
        transform: translateY(-2px);
    }

    /* ── Primary Analyze button ── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0066cc 0%, #6366f1 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 2.4rem;
        font-size: 0.95rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        box-shadow: 0 6px 20px rgba(0,102,204,0.4), 0 2px 6px rgba(99,102,241,0.25);
        text-align: center;
        width: 100%;
        height: auto;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0052a3 0%, #4f52d4 100%);
        color: white !important;
        border: none;
        box-shadow: 0 8px 28px rgba(0,102,204,0.5), 0 4px 10px rgba(99,102,241,0.3);
        transform: translateY(-2px);
    }

    /* ── Answer box (legacy) ── */
    .answer-box {
        background: #ffffff;
        border-left: 4px solid #0066cc;
        border-radius: 0 10px 10px 0;
        padding: 1.3rem 1.6rem;
        margin-top: 0.75rem;
        line-height: 1.75;
        color: #1a1a2e !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .answer-box p, .answer-box li, .answer-box span,
    .answer-box strong, .answer-box em, .answer-box code {
        color: #1a1a2e !important;
    }

    /* ── Executive summary ── */
    .summary-box {
        background: linear-gradient(135deg, #e8f0fe 0%, #ede9fe 100%);
        border-radius: 14px;
        padding: 1.2rem 1.4rem 1.2rem 1.6rem;
        border: 1px solid #c7d7f9;
        border-left: none;
        position: relative;
        overflow: hidden;
        font-size: 0.95rem;
        line-height: 1.75;
        color: #1a1a2e;
        box-shadow: 0 4px 18px rgba(0,102,204,0.11), 0 1px 4px rgba(0,0,0,0.05);
    }
    .summary-box::before {
        content: "";
        position: absolute;
        top: 0; left: 0; bottom: 0;
        width: 5px;
        background: linear-gradient(180deg, #0066cc, #6366f1);
    }

    /* ── Metric cards ── */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffffff 0%, #f5f8ff 100%);
        border: 1px solid #dce8f8;
        border-radius: 16px;
        padding: 1.2rem 1.3rem 1.1rem 1.3rem;
        box-shadow: 0 6px 20px rgba(0,102,204,0.08), 0 1px 4px rgba(0,0,0,0.04);
        position: relative;
        overflow: hidden;
    }
    [data-testid="metric-container"]::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 4px;
        background: linear-gradient(90deg, #0066cc, #6366f1);
        border-radius: 16px 16px 0 0;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 1.75rem !important;
        font-weight: 900 !important;
        color: #0d1b2a !important;
        letter-spacing: -0.03em !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricLabel"] {
        font-size: 0.68rem !important;
        font-weight: 700 !important;
        color: #6a80a0 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.09em !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricDelta"] {
        color: #0066cc !important;
    }

    /* ── Hero landing card ── */
    .hero-card {
        background: linear-gradient(135deg, #060e1a 0%, #0d2a56 55%, #1a3a70 100%);
        border-radius: 20px;
        padding: 3.25rem 3.5rem 2.75rem 3.5rem;
        border: 1px solid rgba(255,255,255,0.08);
        text-align: center;
        margin: 0.5rem 0 1.5rem 0;
        box-shadow: 0 20px 60px rgba(0,0,0,0.28), 0 4px 16px rgba(0,102,204,0.2);
        position: relative;
        overflow: hidden;
    }
    .hero-card::before {
        content: "";
        position: absolute;
        top: -80px; right: -60px;
        width: 340px; height: 340px;
        background: radial-gradient(circle, rgba(99,102,241,0.38) 0%, transparent 68%);
        border-radius: 50%;
        pointer-events: none;
    }
    .hero-card::after {
        content: "";
        position: absolute;
        bottom: -90px; left: -50px;
        width: 280px; height: 280px;
        background: radial-gradient(circle, rgba(0,102,204,0.32) 0%, transparent 68%);
        border-radius: 50%;
        pointer-events: none;
    }
    .hero-card h2 {
        color: #ffffff !important;
        font-size: 2.1rem !important;
        margin-bottom: 0.6rem !important;
        font-weight: 900 !important;
        letter-spacing: -0.03em !important;
        position: relative; z-index: 1;
        text-shadow: 0 2px 20px rgba(0,0,0,0.3);
    }
    .hero-card p {
        color: rgba(255,255,255,0.68);
        font-size: 1.05rem;
        margin: 0.3rem 0;
        position: relative; z-index: 1;
    }
    .hero-features {
        display: flex;
        justify-content: center;
        gap: 0.75rem;
        margin-top: 2rem;
        flex-wrap: wrap;
        position: relative; z-index: 1;
    }
    .hero-feature {
        background: rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 0.55rem 1.1rem;
        font-size: 0.82rem;
        color: rgba(255,255,255,0.88);
        border: 1px solid rgba(255,255,255,0.18);
        backdrop-filter: blur(6px);
        font-weight: 500;
    }

    /* ── Section headers ── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        margin: 1.8rem 0 0.85rem 0;
    }
    .section-header-icon {
        width: 28px; height: 28px;
        background: linear-gradient(135deg, #0066cc, #6366f1);
        border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.8rem; flex-shrink: 0;
        box-shadow: 0 4px 12px rgba(0,102,204,0.42);
    }
    .section-header-text {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #3a4f70;
    }

    /* ── Question label ── */
    .question-label {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #7a8aaa;
        margin-bottom: 0.6rem;
    }

    /* ── AI answer card ── */
    .answer-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.8rem;
        padding-bottom: 0.7rem;
        border-bottom: 1px solid #eef2fa;
    }
    .answer-badge {
        background: linear-gradient(135deg, #0066cc, #6366f1);
        color: white;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        padding: 0.22rem 0.65rem;
        border-radius: 20px;
        box-shadow: 0 2px 8px rgba(0,102,204,0.3);
    }
    .answer-card {
        background:
            linear-gradient(#ffffff, #ffffff) padding-box,
            linear-gradient(135deg, #0066cc 0%, #6366f1 100%) border-box;
        border: 2px solid transparent;
        border-radius: 16px;
        padding: 1.4rem 1.6rem 1.2rem 1.6rem;
        box-shadow: 0 8px 32px rgba(0,102,204,0.13), 0 2px 8px rgba(0,0,0,0.05);
        margin: 0.5rem 0 1rem 0;
    }
    .answer-card p, .answer-card li, .answer-card h1,
    .answer-card h2, .answer-card h3, .answer-card strong,
    .answer-card em, .answer-card code, .answer-card td,
    .answer-card th, .answer-card span {
        color: #1a1a2e !important;
    }
    .answer-card ul, .answer-card ol {
        padding-left: 1.5rem;
        margin: 0.5rem 0;
    }
    .answer-card li { margin-bottom: 0.3rem; }
    .answer-card strong { font-weight: 700 !important; }
    .answer-card p { margin-bottom: 0.6rem; }

    /* ── Q&A history ── */
    .history-item {
        background: #ffffff;
        border-radius: 12px;
        border-right: 1px solid #e8eef8;
        border-top: 1px solid #e8eef8;
        border-bottom: 1px solid #e8eef8;
        border-left: 4px solid #0066cc;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        transition: box-shadow 0.15s, transform 0.15s;
    }
    .history-item:hover {
        box-shadow: 0 4px 18px rgba(0,102,204,0.12);
        transform: translateX(2px);
    }
    .history-q {
        font-size: 0.84rem;
        font-weight: 600;
        color: #0052a3;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: flex-start;
        gap: 0.4rem;
    }
    .history-a {
        font-size: 0.84rem;
        color: #3a4a65;
        line-height: 1.65;
        border-top: 1px solid #f0f4fa;
        padding-top: 0.65rem;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #060e1a 0%, #0d1b2a 100%);
        border-right: none;
        box-shadow: 4px 0 24px rgba(0,0,0,0.18);
    }
    [data-testid="stSidebar"] * { color: #c8d8f0 !important; }
    [data-testid="stSidebar"] h1 { color: #ffffff !important; font-size: 1.1rem !important;
        font-weight: 800 !important; letter-spacing: -0.01em !important; }
    [data-testid="stSidebar"] strong { color: #e8edf5 !important; }
    [data-testid="stSidebar"] hr { border-color: #1e3a5f !important; }
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] caption { color: #6a80a0 !important; }

    /* Sidebar selectbox — dark themed */
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: #1e3a5f !important;
        border-color: #2a4f7a !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] * { color: #e8edf5 !important; }
    [data-testid="stSidebar"] [data-baseweb="select"] span { color: #e8edf5 !important; }
    [data-testid="stSidebar"] [data-baseweb="select"] [data-testid="stMarkdownContainer"] p { color: #e8edf5 !important; }
    [data-testid="stSidebar"] div[class*="ValueContainer"] { color: #e8edf5 !important; }
    [data-testid="stSidebar"] div[class*="singleValue"] { color: #e8edf5 !important; }
    [data-testid="stSidebar"] input { color: #e8edf5 !important; }

    /* Sidebar file uploader */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
        background-color: #1e3a5f;
        border-color: #2a4f7a !important;
        border-radius: 8px;
    }

    /* ── Sidebar brand block ── */
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        padding: 0.25rem 0 0.5rem 0;
    }
    .sidebar-brand-icon {
        width: 36px; height: 36px;
        background: linear-gradient(135deg, #0066cc, #6366f1);
        border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.15rem; line-height: 1; flex-shrink: 0;
        box-shadow: 0 4px 14px rgba(0,102,204,0.45);
    }
    .sidebar-brand-text {
        font-size: 1.05rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff, #a5c8ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.01em;
    }
    .sidebar-brand-sub {
        font-size: 0.68rem;
        color: #5a72a0;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }

    /* ── Sidebar credit card ── */
    .sidebar-credit {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 0.9rem 1rem;
        border: 1px solid rgba(255,255,255,0.08);
        margin-top: 0.25rem;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.07);
    }
    .sidebar-credit .name {
        font-size: 0.92rem;
        font-weight: 600;
        color: #e8edf5 !important;
    }
    .sidebar-credit .detail {
        font-size: 0.76rem;
        color: #6a80a0 !important;
        margin-top: 0.2rem;
        line-height: 1.5;
    }
    .sidebar-credit a { color: #60a5fa !important; text-decoration: none; }
    .sidebar-credit a:hover { text-decoration: underline; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-icon">📊</div>
        <div>
            <div class="sidebar-brand-text">DataLens AI</div>
            <div class="sidebar-brand-sub">Business Analytics</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.write("**Industry Template**")
    industry = st.selectbox(
        "Select your data type:",
        list(INDUSTRY_TEMPLATES.keys()),
        label_visibility="collapsed"
    )

    st.divider()
    st.write("**Analysis Mode**")
    mode = st.radio(
        "Mode:",
        ["Single File", "Compare Two Files"],
        label_visibility="collapsed"
    )

    st.divider()
    st.write("**How to Use**")
    st.write("""
    1. Select your industry template
    2. Upload your data file(s)
    3. Review the auto-generated summary
    4. Click a suggested question or type your own
    5. Download your report when done
    """)
    st.divider()
    st.caption("🔒 Your data is sent to the Anthropic API for analysis and is not stored or used for model training.")
    st.divider()
    st.markdown("""
    <div class="sidebar-credit">
        <div class="name">Connor Lewis</div>
        <div class="detail">
            BS Business Analytics · Fairfield University<br>
            MS AI &amp; Business Analytics (in progress)<br>
            <br>
            <strong style="color:#b0bcd4;">Stack:</strong> Claude API · Streamlit · Plotly
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Main heading ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 0.5rem 0 1.5rem 0; border-bottom: 2px solid #d8e4f4; margin-bottom: 1.5rem;">
    <div style="font-size:2rem; font-weight:900; letter-spacing:-0.03em; line-height:1.15; margin-bottom:0.3rem;">
        <span style="background: linear-gradient(135deg, #0d1b2a 0%, #0052a3 60%, #6366f1 100%);
                     -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                     background-clip: text;">AI Business Data Analyst</span>
    </div>
    <p style="color: #5a6a85; font-size: 1rem; margin: 0; font-weight: 400;">
        Upload any spreadsheet and ask questions in plain English.
        Powered by <strong style="color: #0052a3;">Anthropic Claude</strong>.
    </p>
</div>
""", unsafe_allow_html=True)


# ── Helper functions ───────────────────────────────────────────────────────────

def load_file(uploaded_file):
    """Load a CSV or Excel file into a Pandas dataframe."""
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            return pd.read_csv(uploaded_file)
        else:
            return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(
            f"**Could not read '{uploaded_file.name}'.**  \n"
            "Make sure the file is a valid CSV or Excel file and is not "
            "password-protected or corrupted.  \n"
            f"Detail: {e}"
        )
        st.stop()


@st.cache_data
def get_sample_data():
    """Generate a realistic 500-row retail sales dataset for demo purposes."""
    import numpy as np
    rng = np.random.default_rng(42)
    n = 500

    regions   = ["East", "West", "Central", "South"]
    categories = ["Technology", "Furniture", "Office Supplies"]
    sub_cats   = {
        "Technology":      ["Phones", "Laptops", "Accessories", "Copiers"],
        "Furniture":       ["Chairs", "Tables", "Bookcases", "Storage"],
        "Office Supplies": ["Binders", "Paper", "Pens", "Labels"],
    }
    segments   = ["Consumer", "Corporate", "Home Office"]
    ship_modes = ["Standard Class", "Second Class", "First Class", "Same Day"]
    ship_days  = {"Standard Class": (4, 7), "Second Class": (2, 4),
                  "First Class": (1, 2), "Same Day": (0, 0)}
    sales_lo_hi = {"Technology": (120, 2000), "Furniture": (80, 900),
                   "Office Supplies": (10, 200)}
    base_margin = {"Technology": 0.14, "Furniture": 0.10, "Office Supplies": 0.28}

    all_dates   = pd.date_range("2022-01-01", "2024-12-31")
    order_dates = pd.DatetimeIndex(rng.choice(all_dates, n))
    cat_idx     = rng.choice(3, n, p=[0.32, 0.30, 0.38])
    cat_names   = [categories[c] for c in cat_idx]

    firsts = ["Alex","Jordan","Sam","Taylor","Morgan","Riley","Casey","Drew","Pat","Dana"]
    lasts  = ["Smith","Jones","Chen","Patel","Kim","Garcia","Brown","Wilson","Moore","Clark"]

    rows = []
    for i in range(n):
        cat      = cat_names[i]
        sub      = rng.choice(sub_cats[cat])
        seg      = rng.choice(segments,   p=[0.50, 0.30, 0.20])
        region   = rng.choice(regions,    p=[0.30, 0.35, 0.20, 0.15])
        ship     = rng.choice(ship_modes, p=[0.58, 0.20, 0.17, 0.05])
        lo, hi   = sales_lo_hi[cat]
        sales    = round(float(rng.uniform(lo, hi)), 2)
        discount = float(rng.choice([0, 0, 0.1, 0.2, 0.3, 0.5],
                                     p=[0.40, 0.20, 0.18, 0.12, 0.07, 0.03]))
        margin   = base_margin[cat] - discount * 0.85 + float(rng.uniform(-0.03, 0.03))
        profit   = round(sales * margin, 2)
        qty      = int(rng.integers(1, 8))
        order_dt = order_dates[i]
        lo_d, hi_d = ship_days[ship]
        ship_dt  = order_dt + pd.Timedelta(days=int(rng.integers(lo_d, hi_d + 1)))
        customer = f"{rng.choice(firsts)} {rng.choice(lasts)}"
        rows.append({
            "Order ID":     f"ORD-{order_dt.year}-{i+1:04d}",
            "Order Date":   order_dt.strftime("%Y-%m-%d"),
            "Ship Date":    ship_dt.strftime("%Y-%m-%d"),
            "Customer Name": customer,
            "Segment":      seg,
            "Region":       region,
            "Category":     cat,
            "Sub-Category": sub,
            "Ship Mode":    ship,
            "Quantity":     qty,
            "Discount":     discount,
            "Sales":        sales,
            "Profit":       profit,
        })
    return pd.DataFrame(rows)


@st.cache_data
def build_df_info(df, label="Dataset"):
    """
    Build a concise summary with pre-calculated group totals.

    Priority columns (Category, Segment, Region, etc.) are always included
    first regardless of their position in the dataframe, so Claude always
    has the most business-relevant breakdowns available.
    """

    # ── Numeric columns to aggregate (skip ID-like fields) ────────────────────
    all_num = [
        c for c in df.select_dtypes(include="number").columns
        if not any(kw in c.lower() for kw in SKIP_NUM_COLS)
    ]
    # Sort so priority columns come first, then the rest in original order
    numeric_cols = (
        [c for c in all_num if any(p in c.lower() for p in PRIORITY_NUM_COLS)] +
        [c for c in all_num if not any(p in c.lower() for p in PRIORITY_NUM_COLS)]
    )

    # ── Categorical columns: priority-first ordering ───────────────────────────
    # These names are checked with exact case-insensitive matching first,
    # so "Category" is never buried behind less-useful columns.
    priority_names = [
        "category", "segment", "region", "sub-category", "subcategory",
        "ship mode", "state", "country", "department", "product name",
        "customer name", "channel", "brand", "division", "team"
    ]

    # Use word-boundary matching so e.g. "OverTime" isn't caught by "time",
    # "OrderDate" is still caught by "order", etc.
    _skip_cat_patterns = [
        re.compile(r'(^|[^a-z])' + kw + r'([^a-z]|$)')
        for kw in ["id", "date", "time", "postal", "code", "zip",
                   "phone", "address", "row", "order", "name"]
    ]

    def _should_skip_cat(col: str) -> bool:
        low = col.lower()
        return any(p.search(low) for p in _skip_cat_patterns)

    useful_cats = []

    # Pass 1 — add priority columns in priority order (exact case-insensitive match)
    for name in priority_names:
        match = next(
            (c for c in df.columns if c.lower() == name.lower()), None
        )
        if match and match not in useful_cats:
            useful_cats.append(match)

    # Pass 2 — fill remaining slots with other categorical columns
    for c in df.select_dtypes(include="object").columns:
        if c not in useful_cats:
            if not _should_skip_cat(c):
                useful_cats.append(c)

    # ── Unique value counts ───────────────────────────────────────────────────
    unique_info = "\nUnique value counts (use for 'how many distinct X' questions):\n"
    counted = set()
    for col in useful_cats[:5]:
        unique_info += f"  {col}: {df[col].nunique():,} unique values\n"
        counted.add(col)
    for col in df.columns:
        if col not in counted and any(
            kw in col.lower() for kw in
            ["customer", "product", "order", "employee", "user",
             "item", "sku", "store", "location", "invoice"]
        ):
            unique_info += f"  {col}: {df[col].nunique():,} unique values\n"

    # ── Build group summaries (categorical × numeric — all combinations) ────────
    # Prioritise score/satisfaction/rating/balance columns so they're always
    # included even in wide datasets where they'd fall outside the top-5 cap.
    _score_kws = ["satisfaction", "balance", "rating", "score", "level",
                  "involvement", "performance", "engagement", "survey"]
    _priority_nums = [c for c in numeric_cols
                      if any(kw in c.lower() for kw in _score_kws)]
    _other_nums = [c for c in numeric_cols if c not in _priority_nums]
    _group_num_cols = (_priority_nums + _other_nums)[:10]  # up to 10 total

    group_summaries = ""
    for cat_col in useful_cats:               # every categorical column
        for num_col in _group_num_cols:       # priority + top numeric columns
            try:
                grand_total = df[num_col].sum()
                summary = (
                    df.groupby(cat_col)[num_col]
                    .agg(Total="sum", Average="mean", Std_Dev="std",
                         Min="min", Max="max", Count="count")
                    .round(2)
                    .sort_values("Total", ascending=False)
                    .head(15)
                )
                summary["Pct_of_Total"] = (
                    (summary["Total"] / grand_total * 100).round(1)
                    .astype(str) + "%"
                )
                group_summaries += (
                    f"\n{num_col} by {cat_col} "
                    f"(Grand Total = {grand_total:,.2f}):\n"
                    f"{summary.to_string()}\n"
                )
            except Exception:
                pass

    # ── Percentile distributions ───────────────────────────────────────────────
    pct_stats = "\nNumeric column distributions (for spread/range/outlier questions):\n"
    for col in numeric_cols[:4]:
        try:
            s = df[col].describe(percentiles=[.25, .75, .90])
            pct_stats += (
                f"  {col}: mean={s['mean']:.2f}, 25th pct={s['25%']:.2f}, "
                f"75th={s['75%']:.2f}, 90th={s['90%']:.2f}, max={s['max']:.2f}\n"
            )
        except Exception:
            pass

    # ── Top / bottom individual rows ───────────────────────────────────────────
    top_rows_summary = ""
    if numeric_cols:
        key_num = numeric_cols[0]
        display_cols = [c for c in useful_cats[:3] + [key_num] if c in df.columns]
        try:
            top5 = df.nlargest(5, key_num)[display_cols]
            bot5 = df.nsmallest(5, key_num)[display_cols]
            top_rows_summary += (
                f"\nTop 5 rows by {key_num}:\n{top5.to_string(index=False)}\n"
                f"\nBottom 5 rows by {key_num}:\n{bot5.to_string(index=False)}\n"
            )
        except Exception:
            pass

    # ── Cross-tab summaries (all cat pairs × top numeric cols) ──────────────────
    cross_summaries = ""
    if len(useful_cats) >= 2 and numeric_cols:
        from itertools import combinations
        cat_pairs = list(combinations(useful_cats, 2))[:12]  # cap at 12 pairs
        for num_col in numeric_cols[:3]:
            for cat_a, cat_b in cat_pairs:
                try:
                    pivot = df.pivot_table(
                        values=num_col,
                        index=cat_a,
                        columns=cat_b,
                        aggfunc="mean",
                        fill_value=0
                    ).round(2)
                    if pivot.shape[0] <= 20 and pivot.shape[1] <= 15:
                        cross_summaries += (
                            f"\nAvg {num_col} by {cat_a} × {cat_b}:\n"
                            f"{pivot.to_string()}\n"
                        )
                except Exception:
                    pass

    # ── Categorical × categorical count/rate tables ───────────────────────────
    # Finds binary columns (Yes/No, 1/0, True/False) and cross-tabs them
    # against every other categorical column so Claude can answer questions
    # like "attrition rate by job role" or "churn rate by region."
    # Also computes avg of ALL numeric columns grouped by each binary column
    # (e.g. avg DistanceFromHome for Attrition=Yes vs No) so Claude can answer
    # any "does X differ between churned and retained employees?" question.
    cat_rate_summary = ""
    _binary_cols = [
        c for c in useful_cats
        if df[c].nunique() == 2
    ]
    for bin_col in _binary_cols[:3]:
        vals = df[bin_col].dropna().unique()
        # Identify the "positive" value (Yes / True / 1 / Churned etc.)
        _pos_val = None
        for v in vals:
            if str(v).lower() in ("yes", "true", "1", "churned", "left", "attrited"):
                _pos_val = v
                break
        if _pos_val is None:
            _pos_val = vals[0]  # fallback: first value

        # Numeric profile by binary outcome (ALL numeric cols)
        try:
            num_profile = df.groupby(bin_col)[numeric_cols].mean().round(2).T
            cat_rate_summary += (
                f"\nAvg of all numeric columns by {bin_col} "
                f"(use to answer 'does X differ between {bin_col} groups'):\n"
                f"{num_profile.to_string()}\n"
            )
        except Exception:
            pass

        for grp_col in useful_cats:
            if grp_col == bin_col:
                continue
            if df[grp_col].nunique() > 25:
                continue
            try:
                ct = df.groupby(grp_col)[bin_col].agg(
                    Total="count",
                    Positive=lambda x, pv=_pos_val: (x == pv).sum()
                )
                ct["Rate_%"] = (ct["Positive"] / ct["Total"] * 100).round(1)
                ct = ct.sort_values("Rate_%", ascending=False)
                cat_rate_summary += (
                    f"\n{bin_col} rate by {grp_col} "
                    f"('{_pos_val}' = positive):\n"
                    f"{ct.to_string()}\n"
                )
            except Exception:
                pass

    # ── Derived ratio summaries ────────────────────────────────────────────────
    ratio_summary = ""
    ratio_pairs = [
        (["profit", "margin"], ["sales", "revenue"]),
        (["cost", "expense", "spend"], ["sales", "revenue"]),
        (["discount"], ["sales", "revenue"]),
    ]
    for num_kws, den_kws in ratio_pairs:
        num_col = next(
            (c for c in numeric_cols if any(kw in c.lower() for kw in num_kws)), None
        )
        den_col = next(
            (c for c in numeric_cols
             if any(kw in c.lower() for kw in den_kws) and c != num_col), None
        )
        if not (num_col and den_col):
            continue
        overall_den = df[den_col].sum()
        if overall_den == 0:
            continue
        overall_ratio = df[num_col].sum() / overall_den * 100
        ratio_summary += (
            f"\nDerived ratio — {num_col} as % of {den_col}: "
            f"Overall = {overall_ratio:.2f}%\n"
        )
        for cat_col in useful_cats[:2]:
            try:
                g = df.groupby(cat_col)[[num_col, den_col]].sum()
                g["ratio_%"] = (g[num_col] / g[den_col].replace(0, float("nan")) * 100).round(2)
                ratio_summary += f"  by {cat_col}:\n{g.to_string()}\n"
            except Exception:
                pass

    # ── Derived metrics (Gross Profit, Margin %) ───────────────────────────────
    # When a dataset has revenue + cost columns but no profit column, compute
    # Gross Profit = Revenue - Cost and Margin % = Gross Profit / Revenue * 100,
    # then break them down by every categorical column so Claude can answer
    # profitability questions without refusing.
    derived_summary = ""
    _rev_col = next(
        (c for c in numeric_cols
         if any(kw in c.lower() for kw in ["sales", "revenue", "amount", "price"])),
        None
    )
    _cost_col = next(
        (c for c in numeric_cols
         if any(kw in c.lower() for kw in ["cost", "expense", "cogs"])
         and c != _rev_col),
        None
    )
    _profit_col_exists = any(
        kw in c.lower() for c in numeric_cols for kw in ["profit", "margin"]
    )

    if _rev_col and _cost_col and not _profit_col_exists:
        try:
            _gp = df[_rev_col] - df[_cost_col]
            _total_rev  = df[_rev_col].sum()
            _total_cost = df[_cost_col].sum()
            _total_gp   = _gp.sum()
            _overall_margin = (_total_gp / _total_rev * 100) if _total_rev else 0

            derived_summary += (
                f"\nDerived metrics (Gross Profit = {_rev_col} − {_cost_col}):\n"
                f"  Total {_rev_col}: {_total_rev:,.2f}\n"
                f"  Total {_cost_col}: {_total_cost:,.2f}\n"
                f"  Total Gross Profit: {_total_gp:,.2f}\n"
                f"  Overall Gross Margin %: {_overall_margin:.2f}%\n"
            )

            # Break down derived profit by every useful categorical column
            _df_tmp = df.copy()
            _df_tmp["_GrossProfit"] = _gp
            _df_tmp["_MarginPct"]   = (_gp / df[_rev_col].replace(0, float("nan")) * 100)

            for cat_col in useful_cats:
                try:
                    grp = _df_tmp.groupby(cat_col).agg(
                        Total_Revenue   =(_rev_col,      "sum"),
                        Total_Cost      =(_cost_col,     "sum"),
                        Gross_Profit    =("_GrossProfit","sum"),
                        Avg_Margin_Pct  =("_MarginPct",  "mean"),
                        Row_Count       =(_rev_col,      "count"),
                    ).round(2).sort_values("Total_Revenue", ascending=False)
                    grp["Margin_%"] = (
                        grp["Gross_Profit"] / grp["Total_Revenue"].replace(0, float("nan")) * 100
                    ).round(2)
                    derived_summary += (
                        f"\nGross Profit & Margin by {cat_col}:\n"
                        f"{grp.to_string()}\n"
                    )
                except Exception:
                    pass
        except Exception:
            pass

    # ── Numeric correlation matrix ─────────────────────────────────────────────
    # Lets Claude answer "is X related to Y?" questions across any two numeric
    # columns (e.g. DaysLateLast30 vs EngagementSurvey, Absences vs Salary).
    corr_summary = ""
    if len(numeric_cols) >= 2:
        try:
            corr_matrix = df[numeric_cols].corr().round(3)
            corr_summary = (
                "\nNumeric correlation matrix (Pearson r — use for "
                "questions about relationships between any two numeric columns):\n"
                f"{corr_matrix.to_string()}\n"
            )
        except Exception:
            pass

    # ── Numeric binned cross-tabs ──────────────────────────────────────────────
    # For count/score/days columns, bin them into ranges and compute stats on
    # other numeric columns — e.g. "avg EngagementSurvey for employees with
    # 0 vs 1-3 vs 4+ absences."
    binned_summary = ""
    _bin_candidates = [
        c for c in numeric_cols
        if any(kw in c.lower() for kw in
               ["absence", "late", "days", "count", "score",
                "rating", "tenure", "year", "age", "number"])
        and df[c].nunique() <= 50      # skip high-cardinality continuous cols
        and df[c].min() >= 0           # non-negative (counts/days/scores)
    ]
    for bin_col in _bin_candidates[:3]:
        for target_col in numeric_cols:
            if target_col == bin_col:
                continue
            try:
                n_bins = min(4, df[bin_col].nunique())
                if n_bins < 2:
                    continue
                binned = pd.cut(df[bin_col], bins=n_bins, duplicates="drop")
                grp = (
                    df.groupby(binned, observed=True)[target_col]
                    .agg(Count="count", Mean="mean", Std="std")
                    .round(2)
                )
                binned_summary += (
                    f"\nAvg {target_col} by {bin_col} ranges:\n"
                    f"{grp.to_string()}\n"
                )
            except Exception:
                pass

    # ── Time-series summaries ──────────────────────────────────────────────────
    # Find date columns, parse them, and build monthly + yearly totals so
    # Claude can answer trend questions ("monthly sales", "YoY growth", etc.)
    time_summaries = ""
    parsed_date_col = None

    date_col_candidates = [
        c for c in df.columns
        if any(kw in c.lower() for kw in ["date", "time", "month", "year", "period"])
    ]

    for col in date_col_candidates:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().sum() > 100:
                parsed_date_col = col
                date_series = parsed
                break
        except Exception:
            continue

    if parsed_date_col and numeric_cols:
        # Prefer a meaningful value column over quantity/count columns
        num_col = next(
            (c for c in numeric_cols if any(p in c.lower() for p in PRIORITY_NUM_COLS)),
            numeric_cols[0]   # fall back to first if no priority match
        )
        try:
            # Build a clean working dataframe with just what we need
            work = pd.DataFrame({
                "date": date_series,
                "value": df[num_col]
            }).dropna()

            # Monthly totals using string format (avoids Period object issues)
            work["month"] = work["date"].dt.strftime("%Y-%m")
            monthly = work.groupby("month")["value"].sum().round(2).sort_index()
            time_summaries += f"\nMonthly {num_col} (from {parsed_date_col}):\n{monthly.to_string()}\n"

            # Month-over-month growth (last 12 months shown)
            if len(monthly) > 1:
                mom = monthly.pct_change().mul(100).round(1).dropna()
                time_summaries += (
                    f"\nMonth-over-month {num_col} growth (%, last 12 periods):\n"
                    f"{mom.tail(12).to_string()}\n"
                )

            # Quarterly totals
            work["quarter"] = work["date"].dt.to_period("Q").astype(str)
            quarterly = work.groupby("quarter")["value"].sum().round(2).sort_index()
            time_summaries += f"\nQuarterly {num_col}:\n{quarterly.to_string()}\n"

            # Yearly totals
            work["year"] = work["date"].dt.year
            yearly = work.groupby("year")["value"].sum().round(2).sort_index()
            time_summaries += f"\nYearly {num_col}:\n{yearly.to_string()}\n"

            # Year-over-year growth
            if len(yearly) > 1:
                yoy = yearly.pct_change().mul(100).round(1).dropna()
                time_summaries += f"\nYear-over-year {num_col} growth (%):\n{yoy.to_string()}\n"

        except Exception:
            pass

    # ── Shipping / lead-time summaries ────────────────────────────────────────
    # Detect order-date + ship-date column pairs and pre-calculate delivery days
    # so Claude can answer "average shipping time by ship mode" accurately.
    shipping_summary = ""
    try:
        # Find an order-date column
        order_col = next(
            (c for c in df.columns if any(kw in c.lower() for kw in
             ["order date", "order_date", "orderdate", "purchase date"])),
            None
        )
        # Find a ship-date column (exclude the order-date column itself)
        ship_col = next(
            (c for c in df.columns if any(kw in c.lower() for kw in
             ["ship date", "ship_date", "shipdate", "delivery date",
              "delivered date", "fulfillment date"])
             and c != order_col),
            None
        )

        if order_col and ship_col:
            order_dates = pd.to_datetime(df[order_col], errors="coerce")
            ship_dates  = pd.to_datetime(df[ship_col],  errors="coerce")
            days_diff   = (ship_dates - order_dates).dt.days

            valid_mask = days_diff.notna() & (days_diff >= 0)
            if valid_mask.sum() > 10:
                overall_mean = days_diff[valid_mask].mean()
                overall_med  = days_diff[valid_mask].median()
                shipping_summary += (
                    f"\nShipping / Lead Time ('{order_col}' → '{ship_col}'):\n"
                    f"  Overall average: {overall_mean:.1f} days  |  "
                    f"Median: {overall_med:.1f} days  |  "
                    f"Valid rows: {valid_mask.sum():,}\n"
                )

                # Break down by any categorical column whose name hints at
                # shipping, plus the first two useful_cats (region, segment, etc.)
                ship_cats = [
                    c for c in df.columns
                    if any(kw in c.lower() for kw in
                           ["ship mode", "ship_mode", "shipmode", "carrier",
                            "delivery type", "shipping method", "service level"])
                ]
                # Append up to 2 of the general useful cats as additional breakdowns
                for c in useful_cats[:2]:
                    if c not in ship_cats:
                        ship_cats.append(c)

                work_ship = df[valid_mask].copy()
                work_ship["_days"] = days_diff[valid_mask]

                for cat in ship_cats[:4]:
                    if cat not in df.columns:
                        continue
                    try:
                        agg = (
                            work_ship.groupby(cat)["_days"]
                            .agg(Avg_Days="mean", Median_Days="median",
                                 Min_Days="min", Max_Days="max", Count="count")
                            .round(1)
                            .sort_values("Avg_Days")
                        )
                        shipping_summary += (
                            f"\nAverage shipping days by {cat}:\n"
                            f"{agg.to_string()}\n"
                        )
                    except Exception:
                        pass
    except Exception:
        pass

    return f"""
{label}:
Columns: {', '.join(df.columns.tolist())}
Rows: {df.shape[0]:,}
Date column detected: {parsed_date_col if parsed_date_col else "None"}
{unique_info}
Pre-calculated statistics (use these for accurate answers — do not guess):
{group_summaries}
{pct_stats}
{top_rows_summary}
{cross_summaries}
{cat_rate_summary}
{ratio_summary}
{derived_summary}
{corr_summary}
{binned_summary}
{time_summaries}
{shipping_summary}
Sample (first 5 rows):
{df.head(5).to_string()}
"""


@st.cache_data
def build_comparison_info(df1, df2):
    """
    Pre-calculate differences between two datasets so Claude can answer
    comparison questions accurately (e.g. this year vs last year).
    """
    num1 = {c for c in df1.select_dtypes(include="number").columns
            if not any(kw in c.lower() for kw in SKIP_NUM_COLS)}
    num2 = {c for c in df2.select_dtypes(include="number").columns
            if not any(kw in c.lower() for kw in SKIP_NUM_COLS)}
    common_num = sorted(num1 & num2)

    cat1 = set(df1.select_dtypes(include="object").columns)
    cat2 = set(df2.select_dtypes(include="object").columns)
    common_cat = sorted(cat1 & cat2)

    result = "\n--- PRE-CALCULATED COMPARISON (File 1 vs File 2) ---\n"

    # Overall totals
    if common_num:
        result += "\nOverall totals comparison:\n"
        for col in common_num[:4]:
            t1 = df1[col].sum()
            t2 = df2[col].sum()
            change = t2 - t1
            pct = (change / t1 * 100) if t1 != 0 else 0
            result += (
                f"  {col}: File1={t1:,.2f}  File2={t2:,.2f}  "
                f"Change={change:+,.2f} ({pct:+.1f}%)\n"
            )

    # Category-level breakdown comparison
    priority_cats = ["category", "segment", "region", "department", "sub-category"]
    for cat_col in common_cat:
        if not any(p in cat_col.lower() for p in priority_cats):
            continue
        for num_col in common_num[:2]:
            try:
                g1 = df1.groupby(cat_col)[num_col].sum().round(2).rename("File1")
                g2 = df2.groupby(cat_col)[num_col].sum().round(2).rename("File2")
                comp = pd.concat([g1, g2], axis=1).fillna(0)
                comp["Change"] = (comp["File2"] - comp["File1"]).round(2)
                comp["Change_%"] = ((comp["Change"] / comp["File1"].replace(0, float("nan"))) * 100).round(1)
                comp = comp.sort_values("File2", ascending=False)
                result += f"\n{num_col} by {cat_col} (File1 vs File2):\n{comp.to_string()}\n"
            except Exception:
                pass

    # Columns only in one file
    only_in_1 = sorted(cat1 - cat2)
    only_in_2 = sorted(cat2 - cat1)
    if only_in_1:
        result += f"\nColumns only in File 1: {', '.join(only_in_1)}\n"
    if only_in_2:
        result += f"\nColumns only in File 2: {', '.join(only_in_2)}\n"

    result += "\n--- END COMPARISON ---\n"
    return result


def get_summary(df, prompt, df2=None):
    """
    Generate the executive summary. Uses a simpler prompt than get_answer —
    no numbered-list instruction, no chart instruction, just prose.
    """
    df_info = build_df_info(df, "Dataset 1" if df2 is not None else "Dataset")
    if df2 is not None:
        df_info += build_df_info(df2, "Dataset 2")
        df_info += build_comparison_info(df, df2)
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": f"{prompt}\n\nData:\n{df_info}"
        }]
    )
    return message.content[0].text


def get_suggested_questions(columns):
    """Generate 6 suggested questions using only column names — no data context needed."""
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": (
                f"A business dataset has these columns: {', '.join(columns)}\n"
                "Suggest exactly 6 short, specific questions a business manager would ask "
                "about this data. Return only the 6 questions as a numbered list, nothing else. "
                "Each question must be under 10 words."
            )
        }]
    )
    return message.content[0].text


def get_advanced_questions(df):
    """
    Generate 8 consultant-level analytical questions tailored to the actual
    columns and data types in the uploaded file.
    """
    col_info = []
    num_cols = set(df.select_dtypes(include="number").columns)
    for col in df.columns:
        dtype    = "numeric" if col in num_cols else "categorical"
        n_unique = df[col].nunique()
        col_info.append(f"  {col} ({dtype}, {n_unique:,} unique values)")

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        messages=[{
            "role": "user",
            "content": (
                f"A business dataset has these columns:\n"
                + "\n".join(col_info) +
                "\n\nGenerate exactly 8 advanced, consultant-level analytical questions "
                "that a McKinsey analyst would ask about this specific dataset. "
                "Focus on: hidden patterns, risk concentration, efficiency gaps, "
                "cross-variable relationships, anomalies, and strategic implications. "
                "Every question must be directly answerable from the columns listed — "
                "do not reference columns that are not in the list. "
                "Return only the 8 questions as a numbered list, nothing else. "
                "Each question must be under 12 words."
            )
        }]
    )
    return message.content[0].text


def get_answer(df, question, df2=None):
    """Send question and data to Claude and return the answer."""
    df_info = build_df_info(df, "Dataset 1" if df2 is not None else "Dataset")
    if df2 is not None:
        df_info += build_df_info(df2, "Dataset 2")
        df_info += build_comparison_info(df, df2)

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": f"""You are a senior business data analyst with McKinsey-level expertise.
Here is the data:

{df_info}

Question: {question}

Instructions:
1. Answer using the pre-calculated statistics provided above — prefer those numbers over
   anything you derive yourself.
2. You MAY do simple arithmetic (sum two sub-totals, compute a ratio from two provided
   figures, etc.) when the pre-calculated data makes the inputs clear. Never estimate,
   assume, or invent numbers that are not derivable from the data above.
3. If a question genuinely cannot be answered from the data provided, say so in one sentence
   and explain specifically what additional pre-calculation would be needed — do not pad.
4. Format numbers clearly with commas (e.g. "2,261,536"). Use $ for currency values where appropriate.
5. Always name both parties explicitly in comparisons — never say "roughly X vs Y" without names.
6. Structure your answer as a numbered list of findings. Do not use markdown headers (lines starting with #) — use plain numbered points and bold text only.
7. End with a "Key Insight:" section that goes beyond the obvious.
   - Do NOT state things any manager already knows (e.g., "top performers perform well").
   - DO identify hidden risks, structural patterns, or strategic implications.
   - Think like a consultant advising the CEO: what decision does this number actually drive?

If the answer involves comparing multiple items or showing a trend, end your response with this exact separator on its own line, then the JSON on the next line:
---CHART---
{{"chart": {{"type": "bar", "labels": ["A","B","C"], "values": [100,200,300], "title": "Chart Title"}}}}

Chart type rules:
- Use "bar" for category comparisons (sales by region, top products, etc.)
- Use "line" for time-series data (monthly trend, quarterly growth, year-over-year)
Only include ---CHART--- if a chart would genuinely add value. Otherwise answer in plain text only."""
        }]
    )
    return message.content[0].text


@st.cache_data
def check_data_quality(df):
    """Check for common data quality issues and return a list of alert messages."""
    alerts = []

    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if not missing_cols.empty:
        for col, count in missing_cols.items():
            pct = (count / len(df)) * 100
            alerts.append(
                f"**{col}** has {count:,} missing values ({pct:.1f}% of rows)"
            )

    dupes = df.duplicated().sum()
    if dupes > 0:
        alerts.append(f"**{dupes:,} duplicate rows** detected in the dataset")

    for col in df.select_dtypes(include="object").columns:
        sample = df[col].dropna().head(100)
        numeric_count = sum(
            1 for v in sample
            if str(v).replace(".", "").replace(",", "").replace("-", "").isdigit()
        )
        if numeric_count > 80:
            alerts.append(
                f"**{col}** may contain numbers stored as text — "
                f"consider converting for accurate calculations"
            )

    return alerts


def try_render_chart(answer_text):
    """Extract chart JSON from the answer (after ---CHART--- separator) and render it.
    Returns (clean_text, chart_data) — chart_data is None if no chart was found."""
    if "---CHART---" in answer_text:
        parts = answer_text.split("---CHART---", 1)
        clean_text = parts[0].strip()
        json_block = parts[1].strip()
        # Strip any code fences around the JSON
        json_block = re.sub(r'^```(?:json)?\s*', '', json_block).strip()
        json_block = re.sub(r'\s*```$', '', json_block).strip()
        chart_data = None
        try:
            chart_data = json.loads(json_block)["chart"]
            labels = chart_data.get("labels", [])
            values = chart_data.get("values", [])
            title  = chart_data.get("title", "Chart")
            chart_type = chart_data.get("type", "bar")
            if labels and values and len(labels) == len(values):
                chart_df = pd.DataFrame({"Label": labels, "Value": values})
                if chart_type == "line":
                    fig = px.line(
                        chart_df, x="Label", y="Value", title=title, markers=True
                    )
                    fig.update_traces(line_color="#0066cc", marker_color="#0066cc")
                else:
                    fig = px.bar(
                        chart_df, x="Label", y="Value", title=title,
                        color="Value", color_continuous_scale="Blues"
                    )
                fig.update_layout(
                    xaxis_title="",
                    yaxis_title="Value",
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    font=dict(color="#1a1a2e", size=13),
                    title_font=dict(color="#0d1b2a", size=15),
                    xaxis=dict(gridcolor="#e8edf5", linecolor="#c5d5f5"),
                    yaxis=dict(gridcolor="#e8edf5", linecolor="#c5d5f5"),
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception:
            chart_data = None
        return clean_text, chart_data
    # No separator — strip any stray code fences and return as-is
    clean = re.sub(r'```.*?```', '', answer_text, flags=re.DOTALL).strip()
    return clean, None


@st.cache_data
def get_date_range(df):
    """Return a formatted date range string (e.g. 'Jan 2022 – Dec 2023') or empty string."""
    date_keywords = ["date", "time", "month", "year", "period"]
    for col in df.columns:
        if any(kw in col.lower() for kw in date_keywords):
            try:
                parsed = pd.to_datetime(df[col], errors="coerce").dropna()
                if len(parsed) > 0:
                    return f"{parsed.min().strftime('%b %Y')} – {parsed.max().strftime('%b %Y')}"
            except Exception:
                pass
    return ""


def fix_dollar_signs(text):
    """Replace dollar signs with HTML entity to prevent KaTeX rendering artifacts."""
    return text.replace('$', '&#36;')


def _md_to_html(text: str) -> str:
    """Convert Claude's markdown output to HTML with no external dependencies."""
    lines = text.split('\n')
    out = []
    in_ul = False
    in_ol = False

    def close_lists():
        nonlocal in_ul, in_ol
        if in_ul:
            out.append('</ul>')
            in_ul = False
        if in_ol:
            out.append('</ol>')
            in_ol = False

    for line in lines:
        if line.startswith('### '):
            close_lists()
            out.append(f'<h4 style="color:#0d1b2a;margin:0.8rem 0 0.3rem 0;">{line[4:]}</h4>')
        elif line.startswith('## '):
            close_lists()
            out.append(f'<h3 style="color:#0d1b2a;margin:0.8rem 0 0.3rem 0;">{line[3:]}</h3>')
        elif line.startswith('# '):
            close_lists()
            out.append(f'<h2 style="color:#0d1b2a;margin:0.8rem 0 0.3rem 0;">{line[2:]}</h2>')
        elif re.match(r'^[-*+] ', line):
            if in_ol:
                out.append('</ol>')
                in_ol = False
            if not in_ul:
                out.append('<ul style="margin:0.4rem 0;padding-left:1.4rem;">')
                in_ul = True
            out.append(f'<li style="color:#1a1a2e;margin-bottom:0.25rem;">{line[2:]}</li>')
        elif re.match(r'^\d+\. ', line):
            if in_ul:
                out.append('</ul>')
                in_ul = False
            if not in_ol:
                out.append('<ol style="margin:0.4rem 0;padding-left:1.4rem;">')
                in_ol = True
            _li_text = re.sub(r'^\d+\. ', '', line)
            out.append(f'<li style="color:#1a1a2e;margin-bottom:0.25rem;">{_li_text}</li>')
        elif line.strip() == '':
            close_lists()
            out.append('<div style="margin:0.4rem 0;"></div>')
        else:
            close_lists()
            out.append(f'<p style="color:#1a1a2e;margin:0.3rem 0;">{line}</p>')

    close_lists()
    result = '\n'.join(out)
    # Inline formatting
    result = re.sub(r'\*\*(.+?)\*\*', r'<strong style="color:#0d1b2a;">\1</strong>', result)
    result = re.sub(r'\*(.+?)\*', r'<em>\1</em>', result)
    result = re.sub(r'`(.+?)`', r'<code style="background:#f0f4fa;padding:1px 4px;border-radius:3px;">\1</code>', result)
    return result


def compute_deep_analytics(df: pd.DataFrame) -> dict:
    """
    Run automated analytics on a dataframe.
    Returns: {
        anomalies: {col: {count, pct, examples, mean, std}},
        drivers:   {target, correlations: {col: r}},
        trend:     {date_col, metric_col, direction, pct_change, r2,
                    historical: {dates, values},
                    forecast:   {dates, values}} | None
    }
    """
    result = {"anomalies": {}, "drivers": {}, "trend": None}

    # Exclude columns that are clearly identifiers/codes, not measurements.
    # Keep this list narrow — better to include a borderline column than to
    # silently drop legitimate data.
    _id_keywords = [
        "zip", "postal", "phone", "fax", "ssn", "ein",
        "upc", "isbn", "serial",
    ]
    # Also exclude columns whose name ends in " id" or " code" (exact suffix)
    def _is_id_col(col_name: str) -> bool:
        low = col_name.lower().strip()
        if any(kw in low for kw in _id_keywords):
            return True
        # Ends with "id" or "code" (with or without separator — catches EmpID, CustomerID, etc.)
        if re.search(r'id$|[_ ]code$', low):
            return True
        # Is literally just "id" or "code"
        if low in ("id", "code", "key", "index", "row"):
            return True
        return False

    numeric_cols = [
        c for c in df.select_dtypes(include="number").columns
        if not _is_id_col(c)
    ]

    # ── 1. Anomaly detection (Z-score > 2.5) ──────────────────────────────────
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) < 10:
            continue
        mean_v = col_data.mean()
        std_v  = col_data.std()
        if std_v == 0:
            continue
        z = (col_data - mean_v) / std_v
        outliers = col_data[z.abs() > 2.5]
        if len(outliers):
            result["anomalies"][col] = {
                "count":    len(outliers),
                "pct":      round(len(outliers) / len(col_data) * 100, 1),
                "examples": outliers.values[:5].tolist(),
                "mean":     round(mean_v, 2),
                "std":      round(std_v, 2),
                "z_scores": z[outliers.index].values[:5].tolist(),
                "all_z":    z.tolist(),          # for chart
                "all_vals": col_data.tolist(),   # for chart
                "all_idx":  col_data.index.tolist(),
            }

    # ── 2. Driver / correlation analysis (numeric + categorical) ──────────────
    # Find target metric
    target_kws = ["profit", "revenue", "sales", "income", "margin",
                  "score", "rating", "amount", "value"]
    target_col = None
    all_num = df.select_dtypes(include="number").columns.tolist()
    for kw in target_kws:
        for col in all_num:
            if kw in col.lower():
                target_col = col
                break
        if target_col:
            break
    if not target_col and all_num:
        target_col = all_num[-1]

    if target_col:
        drivers = {}  # col -> {"score": float, "type": "numeric"|"categorical"}

        # Numeric: Pearson correlation (signed)
        for col in numeric_cols:
            if col == target_col:
                continue
            try:
                r = df[target_col].corr(df[col])
                if not pd.isna(r):
                    drivers[col] = {"score": round(float(r), 3), "type": "numeric"}
            except Exception:
                pass

        # Categorical: eta-squared (how much variance in target each group explains)
        cat_cols = [
            c for c in df.select_dtypes(include=["object", "category"]).columns
            if 2 <= df[c].nunique() <= 20
        ]
        target_series = df[target_col].dropna()
        grand_mean = target_series.mean()
        ss_total = ((target_series - grand_mean) ** 2).sum()
        for col in cat_cols:
            try:
                groups = [grp[target_col].dropna()
                          for _, grp in df.groupby(col)
                          if len(grp[target_col].dropna()) > 0]
                ss_between = sum(
                    len(g) * (g.mean() - grand_mean) ** 2 for g in groups
                )
                eta2 = float(ss_between / ss_total) if ss_total > 0 else 0.0
                if eta2 > 0.01:  # only include if it explains >1% of variance
                    drivers[col] = {"score": round(eta2, 3), "type": "categorical"}
            except Exception:
                pass

        # Pre-compute regression slopes for numeric drivers (used by what-if simulator)
        for col, info in drivers.items():
            if info["type"] == "numeric":
                try:
                    valid = df[[col, target_col]].dropna()
                    if len(valid) > 1:
                        slope = float(np.polyfit(
                            valid[col].values.astype(float),
                            valid[target_col].values.astype(float), 1
                        )[0])
                        info["slope"]      = slope
                        info["col_mean"]   = float(df[col].mean())
                    else:
                        info["slope"] = 0.0
                        info["col_mean"] = 0.0
                except Exception:
                    info["slope"] = 0.0
                    info["col_mean"] = 0.0
            else:
                # For categorical: store group means
                try:
                    info["group_means"] = (
                        df.groupby(col)[target_col].mean()
                        .round(2).to_dict()
                    )
                except Exception:
                    info["group_means"] = {}

        result["drivers"] = {
            "target":       target_col,
            "target_mean":  round(float(df[target_col].mean()), 2),
            "target_total": round(float(df[target_col].sum()), 2),
            "n_rows":       len(df),
            "drivers": dict(
                sorted(drivers.items(), key=lambda x: abs(x[1]["score"]), reverse=True)
            ),
        }

    # ── 3. Trend & forecast ────────────────────────────────────────────────────
    # Only run trend analysis on transaction/event date columns, not attribute
    # dates (Hire Date, Birth Date, Review Date, etc.).
    # Two-gate filter:
    #   1. Keyword blocklist — catches obviously named attribute dates.
    #   2. Uniqueness ratio — if >60% of rows have a unique date value, the
    #      column is almost certainly a per-entity attribute (e.g. each employee
    #      has their own hire date), not a repeating transaction date.
    _attr_date_kws = [
        "hire", "birth", "born", "dob", "start", "end",
        "termination", "terminat", "fired", "expir",
        "anniversary", "join", "effective", "review",
        "promotion", "last", "next", "due",
    ]

    def _is_transaction_date(col: str) -> bool:
        low = col.lower()
        # Gate 1: blocked keyword
        if any(kw in low for kw in _attr_date_kws):
            return False
        # Gate 2: uniqueness ratio — attribute dates are nearly unique per row
        try:
            parsed = pd.to_datetime(df[col], errors="coerce").dropna()
            if len(parsed) == 0:
                return False
            unique_ratio = parsed.nunique() / len(parsed)
            if unique_ratio > 0.6:   # >60% unique → likely an attribute date
                return False
        except Exception:
            return False
        return True

    date_cols = [
        c for c in df.columns
        if ("date" in c.lower() or "time" in c.lower() or "month" in c.lower())
        and _is_transaction_date(c)
    ]
    if date_cols and numeric_cols:
        date_col = date_cols[0]
        try:
            df_t = df.copy()
            df_t[date_col] = pd.to_datetime(df_t[date_col], errors="coerce")
            df_t = df_t.dropna(subset=[date_col])

            metric_col = None
            for kw in ["sales", "revenue", "profit", "amount", "value", "income"]:
                for col in numeric_cols:
                    if kw in col.lower():
                        metric_col = col
                        break
                if metric_col:
                    break
            if not metric_col:
                metric_col = numeric_cols[0]

            df_t["_period"] = df_t[date_col].dt.to_period("M")
            grouped = (df_t.groupby("_period")[metric_col]
                       .sum().reset_index())
            grouped["_ts"] = grouped["_period"].dt.to_timestamp()
            grouped = grouped.sort_values("_ts")

            if len(grouped) >= 3:
                x = np.arange(len(grouped), dtype=float)
                y = grouped[metric_col].values.astype(float)
                coeffs  = np.polyfit(x, y, 1)
                slope   = coeffs[0]
                y_pred  = np.polyval(coeffs, x)
                ss_res  = np.sum((y - y_pred) ** 2)
                ss_tot  = np.sum((y - y.mean()) ** 2)
                r2      = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

                last_period = grouped["_period"].iloc[-1]
                fcast_periods = [last_period + i for i in range(1, 4)]
                fcast_dates   = [str(p.to_timestamp())[:10] for p in fcast_periods]
                fcast_vals    = [round(float(np.polyval(coeffs, len(grouped) + i)), 2)
                                 for i in range(3)]
                pct_chg = round((slope * len(grouped)) / y.mean() * 100, 1) if y.mean() else 0

                result["trend"] = {
                    "date_col":   date_col,
                    "metric_col": metric_col,
                    "direction":  "up" if slope > 0 else "down",
                    "pct_change": pct_chg,
                    "r2":         round(r2, 3),
                    "historical": {
                        "dates":  [str(d)[:10] for d in grouped["_ts"].tolist()],
                        "values": y.tolist(),
                    },
                    "forecast": {
                        "dates":  fcast_dates,
                        "values": fcast_vals,
                    },
                }
        except Exception:
            pass  # trend is optional — silently skip on any error

    return result


def compute_insight_cards(df: pd.DataFrame, da: dict) -> list:
    """
    Auto-generate 4–5 insight cards from the dataframe + deep analytics results.
    Returns a list of dicts: {icon, label, headline, detail, color, bg}
    No API calls — purely derived from pre-computed stats.
    """
    cards = []
    skip_cat = ["id", "date", "time", "postal", "code", "zip",
                "phone", "address", "row", "order"]

    drivers_info = da.get("drivers", {})
    target = drivers_info.get("target")

    def _is_currency(col):
        return any(k in col.lower() for k in
                   ["sales", "revenue", "profit", "income", "amount", "spend"])

    def _fmt_val(val, col):
        return f"${val:,.0f}" if _is_currency(col) else f"{val:,.1f}"

    # ── Card 1: Top Performer ─────────────────────────────────────────────────
    if target and target in df.columns:
        cat_cols = [
            c for c in df.select_dtypes(include="object").columns
            if not any(kw in c.lower() for kw in skip_cat)
            and df[c].nunique() <= 30
        ]
        # Prefer short, high-level categoricals
        priority = ["category", "region", "segment", "department", "channel"]
        cat_cols_sorted = (
            [c for c in cat_cols if any(p in c.lower() for p in priority)] +
            [c for c in cat_cols if not any(p in c.lower() for p in priority)]
        )
        if cat_cols_sorted:
            best_cat = cat_cols_sorted[0]
            grp = df.groupby(best_cat)[target].sum().sort_values(ascending=False)
            if len(grp) >= 2:
                top_name  = grp.index[0]
                top_val   = grp.iloc[0]
                total_val = grp.sum()
                top_pct   = top_val / total_val * 100 if total_val else 0
                cards.append({
                    "icon": "🏆",
                    "label": "Top Performer",
                    "headline": f"{top_name} leads {best_cat}",
                    "detail": (
                        f"{_fmt_val(top_val, target)} in {target} — "
                        f"{top_pct:.0f}% of total"
                    ),
                    "color": "#0066cc",
                    "bg": "#e8f0fe",
                })

    # ── Card 2: Trend Signal ──────────────────────────────────────────────────
    trend = da.get("trend")
    if trend:
        direction  = trend["direction"]
        pct        = trend["pct_change"]
        metric     = trend["metric_col"]
        conf       = ("strong" if trend["r2"] > 0.7
                      else ("moderate" if trend["r2"] > 0.4 else "weak"))
        arrow      = "↑" if direction == "up" else "↓"
        trend_word = "uptrending" if direction == "up" else "downtrending"
        cards.append({
            "icon": "📈" if direction == "up" else "📉",
            "label": "Trend Signal",
            "headline": f"{metric} is {trend_word}",
            "detail": (
                f"{arrow} {abs(pct):.1f}% over the full period "
                f"· {conf} trend (R²={trend['r2']})"
            ),
            "color": "#16a34a" if direction == "up" else "#e53e3e",
            "bg": "#e8f5e9" if direction == "up" else "#fde8e8",
        })

    # ── Card 3: Data Alert ────────────────────────────────────────────────────
    anomalies = da.get("anomalies", {})
    if anomalies:
        top_col, top_info = sorted(
            anomalies.items(), key=lambda x: x[1]["count"], reverse=True
        )[0]
        cards.append({
            "icon": "⚠️",
            "label": "Data Alert",
            "headline": f"{top_col} has outliers",
            "detail": (
                f"{top_info['count']} unusual values "
                f"({top_info['pct']}% of rows) — worth reviewing"
            ),
            "color": "#d97706",
            "bg": "#fffbeb",
        })
    else:
        cards.append({
            "icon": "✅",
            "label": "Data Quality",
            "headline": "No anomalies detected",
            "detail": "Values are consistent across all numeric columns",
            "color": "#16a34a",
            "bg": "#e8f5e9",
        })

    # ── Card 4: Key Driver ────────────────────────────────────────────────────
    if drivers_info.get("drivers"):
        top_drv_name, top_drv_info = list(drivers_info["drivers"].items())[0]
        tgt_name = drivers_info["target"]
        if top_drv_info["type"] == "numeric":
            direction_word = "positively" if top_drv_info["score"] > 0 else "negatively"
            drv_detail = (
                f"{direction_word.capitalize()} correlated "
                f"(r = {top_drv_info['score']:.2f}) with {tgt_name}"
            )
        else:
            score_pct = top_drv_info["score"] * 100
            drv_detail = (
                f"Accounts for {score_pct:.0f}% of variance "
                f"in {tgt_name} across groups"
            )
        cards.append({
            "icon": "🔑",
            "label": "Key Driver",
            "headline": f"{top_drv_name} drives {tgt_name}",
            "detail": drv_detail,
            "color": "#6366f1",
            "bg": "#ede9fe",
        })

    # ── Card 5: Dataset Scale ─────────────────────────────────────────────────
    n = len(df)
    if target and target in df.columns:
        total_v = df[target].sum()
        mean_v  = df[target].mean()
        cards.append({
            "icon": "📊",
            "label": "Dataset Scale",
            "headline": f"{n:,} rows · {_fmt_val(total_v, target)} total",
            "detail": f"Average {target} per row: {_fmt_val(mean_v, target)}",
            "color": "#3a4f70",
            "bg": "#f0f4fa",
        })

    return cards[:4]


def generate_report_html(filename, summary, history, filename2=None):
    """Generate a formatted HTML report of the session for download."""
    now = datetime.now().strftime("%B %d, %Y at %I:%M %p")

    rows = ""
    chart_scripts = ""
    chart_idx = 0

    for i, item in enumerate(history, 1):
        safe_q = html_lib.escape(item["question"])
        # Answers may contain $; html_lib.escape doesn't touch $, so they render fine
        safe_a = html_lib.escape(item["answer"])
        chart_html = ""
        cd = item.get("chart_data")
        if cd:
            cid = f"chart_{chart_idx}"
            chart_idx += 1
            labels_json = json.dumps(cd.get("labels", []))
            values_json = json.dumps(cd.get("values", []))
            title_json  = json.dumps(cd.get("title", ""))
            chart_type  = "line" if cd.get("type") == "line" else "bar"
            chart_html = f'<canvas id="{cid}" style="max-height:320px;margin:1rem 0"></canvas>'
            color = "'rgba(0,102,204,0.75)'" if chart_type == "bar" else "'#0066cc'"
            chart_scripts += f"""
            (function() {{
                var ctx = document.getElementById('{cid}').getContext('2d');
                new Chart(ctx, {{
                    type: '{chart_type}',
                    data: {{
                        labels: {labels_json},
                        datasets: [{{
                            label: {title_json},
                            data: {values_json},
                            backgroundColor: {color},
                            borderColor: '#0066cc',
                            borderWidth: 2,
                            tension: 0.3,
                            fill: false,
                            pointRadius: 4
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        plugins: {{ legend: {{ display: false }},
                                   title: {{ display: true, text: {title_json}, font: {{ size: 14 }} }} }},
                        scales: {{ y: {{ beginAtZero: false }} }}
                    }}
                }});
            }})();"""
        rows += f"""
        <div class="qa-block">
            <div class="question">Q{i}: {safe_q}</div>
            <div class="answer">{safe_a}</div>
            {chart_html}
        </div>
        """

    safe_filename = html_lib.escape(filename)
    safe_summary  = html_lib.escape(summary)
    file_label    = (f"{safe_filename} vs {html_lib.escape(filename2)}"
                     if filename2 else safe_filename)

    chart_js_tag = (
        '<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>'
        if chart_idx > 0 else ""
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
{chart_js_tag}
<style>
    body {{ font-family: Arial, sans-serif; max-width: 860px; margin: 40px auto;
           color: #333; line-height: 1.6; padding: 0 1rem; }}
    h1 {{ color: #0066cc; border-bottom: 2px solid #0066cc; padding-bottom: 0.5rem; }}
    h2 {{ color: #0066cc; margin-top: 2rem; }}
    .meta {{ color: #666; font-size: 0.9rem; margin-bottom: 2rem; }}
    .summary {{ background: #f0f7ff; padding: 1.2rem; border-radius: 6px;
                margin-bottom: 2rem; white-space: pre-wrap; }}
    .qa-block {{ margin-bottom: 1.8rem; border-bottom: 1px solid #eee;
                 padding-bottom: 1.5rem; }}
    .question {{ font-weight: bold; color: #0066cc; margin-bottom: 0.5rem; }}
    .answer {{ white-space: pre-wrap; }}
    .footer {{ margin-top: 3rem; color: #999; font-size: 0.8rem;
               border-top: 1px solid #eee; padding-top: 1rem; }}
</style>
</head>
<body>
    <h1>Business Data Analysis Report</h1>
    <div class="meta">
        <strong>File:</strong> {file_label} &nbsp;|&nbsp;
        <strong>Generated:</strong> {now}
    </div>
    <h2>Executive Summary</h2>
    <div class="summary">{safe_summary}</div>
    <h2>Analysis</h2>
    {rows if rows else "<p>No questions asked yet.</p>"}
    <div class="footer">
        Generated by AI Business Data Analyst &nbsp;·&nbsp; Powered by Anthropic Claude
    </div>
    <script>{chart_scripts}</script>
</body>
</html>"""


# ── File upload ────────────────────────────────────────────────────────────────
if mode == "Single File":
    uploaded_file = st.file_uploader(
        "Upload a CSV or Excel file to get started",
        type=["csv", "xlsx", "xls"]
    )
    uploaded_file2 = None
else:
    st.write("Upload two files to compare them — for example, this year vs last year.")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**File 1**")
        uploaded_file = st.file_uploader(
            "Upload first file", type=["csv", "xlsx", "xls"], key="file1"
        )
    with col2:
        st.write("**File 2**")
        uploaded_file2 = st.file_uploader(
            "Upload second file", type=["csv", "xlsx", "xls"], key="file2"
        )

# ── Landing state (no file uploaded yet) ──────────────────────────────────────
if uploaded_file is None and not st.session_state.use_sample:
    st.markdown("""
    <div class="hero-card">
        <h2>Your data has answers.<br>Just ask.</h2>
        <p>Upload any business spreadsheet and get instant, consultant-level insights in plain English.</p>
        <p style="color:rgba(255,255,255,0.4); font-size:0.82rem; margin-top:0.6rem;">CSV · Excel · Up to 10 MB</p>
        <div class="hero-features">
            <div class="hero-feature">📈 Executive summary</div>
            <div class="hero-feature">💬 Plain English Q&amp;A</div>
            <div class="hero-feature">📊 Auto-generated charts</div>
            <div class="hero-feature">🔀 Compare two datasets</div>
            <div class="hero-feature">📥 Downloadable report</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center; color:#9aaccc; font-size:0.85rem; margin-top:-0.5rem;'>"
        "— or —</p>",
        unsafe_allow_html=True,
    )
    _c1, _c2, _c3 = st.columns([1, 2, 1])
    with _c2:
        if st.button("🗂️ Load sample data to explore", use_container_width=True):
            st.session_state.use_sample = True
            st.rerun()

# Sample data: also offer "load sample" below the uploader when no file is uploaded
if uploaded_file is None and not st.session_state.use_sample:
    st.markdown(
        "<p style='text-align:center; color:#9aaccc; font-size:0.82rem; margin-top:0.3rem;'>"
        "No file yet? Try the sample dataset above.</p>",
        unsafe_allow_html=True,
    )

# ── Resolve which data source is active ───────────────────────────────────────
using_sample = st.session_state.get("use_sample", False) and uploaded_file is None

# ── Main analysis ──────────────────────────────────────────────────────────────
if uploaded_file is not None or using_sample:
    # ── Load data ────────────────────────────────────────────────────────────
    if using_sample:
        df = get_sample_data()
        effective_name = "Sample Retail Dataset (500 rows)"
        file_key = "sample_data_v1"
        df2 = None
    else:
        # File size check
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            st.error(
                f"**{uploaded_file.name}** is {file_size_mb:.1f} MB — "
                f"maximum allowed is {MAX_FILE_SIZE_MB} MB. Please upload a smaller file."
            )
            st.stop()
        if uploaded_file2 is not None:
            file2_size_mb = uploaded_file2.size / (1024 * 1024)
            if file2_size_mb > MAX_FILE_SIZE_MB:
                st.error(
                    f"**{uploaded_file2.name}** is {file2_size_mb:.1f} MB — "
                    f"maximum allowed is {MAX_FILE_SIZE_MB} MB. Please upload a smaller file."
                )
                st.stop()

        df = load_file(uploaded_file)
        if df.empty:
            st.error(f"**'{uploaded_file.name}' appears to be empty.** Upload a file with at least one row of data.")
            st.stop()

        df2 = load_file(uploaded_file2) if uploaded_file2 is not None else None
        if df2 is not None and df2.empty:
            st.error(f"**'{uploaded_file2.name}' appears to be empty.** Upload a file with at least one row of data.")
            st.stop()

        effective_name = uploaded_file.name

    if using_sample:
        st.info(
            "📊 **You're exploring the sample dataset** — a synthetic 500-row retail sales "
            "dataset with orders from 2022–2024. Try the suggested questions below, or ask "
            "anything! When ready, upload your own file using the sidebar.",
            icon=None,
        )
        _samp_col1, _samp_col2 = st.columns([4, 1])
        with _samp_col2:
            if st.button("✕ Clear sample", use_container_width=True):
                st.session_state.use_sample = False
                for _k in ("summary", "history", "suggestions",
                           "last_file", "last_suggestion_key"):
                    st.session_state.pop(_k, None)
                st.rerun()

    if not using_sample:
        if df2 is not None:
            st.success(
                f"✅ Files loaded: **{effective_name}** ({len(df):,} rows) "
                f"and **{uploaded_file2.name}** ({len(df2):,} rows)"
            )
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("File 1 Rows", f"{len(df):,}")
            m2.metric("File 2 Rows", f"{len(df2):,}")
            m3.metric("File 1 Columns", len(df.columns))
            m4.metric("File 2 Columns", len(df2.columns))
        else:
            st.success(f"✅ **{effective_name}** loaded successfully")
            date_range = get_date_range(df)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Rows", f"{len(df):,}")
            m2.metric("Columns", len(df.columns))
            num_cols = len([c for c in df.select_dtypes(include="number").columns
                            if not any(kw in c.lower() for kw in SKIP_NUM_COLS)])
            m3.metric("Numeric Columns", num_cols)
            m4.metric("Date Range", date_range if date_range else "N/A")
    else:
        date_range = get_date_range(df)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows", f"{len(df):,}")
        m2.metric("Columns", len(df.columns))
        num_cols = len([c for c in df.select_dtypes(include="number").columns
                        if not any(kw in c.lower() for kw in SKIP_NUM_COLS)])
        m3.metric("Numeric Columns", num_cols)
        m4.metric("Date Range", date_range if date_range else "N/A")

    # Data quality alerts
    alerts = check_data_quality(df)
    alerts2 = check_data_quality(df2) if df2 is not None else []
    # Label which file each alert belongs to in compare mode
    if df2 is not None:
        labeled_alerts = (
            [f"**File 1** — {a}" for a in alerts] +
            [f"**File 2** — {a}" for a in alerts2]
        )
    else:
        labeled_alerts = alerts
    if labeled_alerts:
        with st.expander(
            f"⚠️ {len(labeled_alerts)} data quality issue(s) found — click to review",
            expanded=False
        ):
            for alert in labeled_alerts:
                st.warning(alert)

    # Data preview
    if df2 is not None:
        st.write("### Data Preview")
        prev1, prev2 = st.columns(2)
        with prev1:
            st.write(f"**File 1:** {effective_name}")
            st.dataframe(df.head(10), use_container_width=True)
        with prev2:
            st.write(f"**File 2:** {uploaded_file2.name}")
            st.dataframe(df2.head(10), use_container_width=True)
    else:
        st.write("### Preview of Your Data")
        st.dataframe(df.head(20), use_container_width=True)

    # Stable file identifier — set above in sample/upload branches
    if not using_sample:
        file_key = (
            getattr(uploaded_file, "file_id", uploaded_file.name + str(uploaded_file.size)) +
            (getattr(uploaded_file2, "file_id", uploaded_file2.name + str(uploaded_file2.size))
             if uploaded_file2 else "")
        )
    # Suggestion key tracks both file AND selected industry template
    suggestion_key = file_key + "|" + industry

    # ── Deep analytics: compute once per file ─────────────────────────────────
    if st.session_state.get("last_analytics_file") != file_key:
        st.session_state.deep_analytics = compute_deep_analytics(df)
        st.session_state.last_analytics_file = file_key

    _da = st.session_state.get("deep_analytics", {})

    # ── Executive summary: regenerate only when file changes ──────────────────
    if "summary" not in st.session_state or st.session_state.get("last_file") != file_key:
        st.session_state.last_file = file_key
        st.session_state.history = []   # clear Q&A from any previously loaded file

        with st.spinner("Generating executive summary..."):
            if df2 is not None:
                summary_prompt = (
                    "You are a business analyst. Two datasets have been uploaded for comparison. "
                    "Write a 3-4 sentence executive summary highlighting the most important "
                    "differences between File 1 and File 2. Focus on what changed — which "
                    "totals went up or down, which categories shifted, and what the overall "
                    "trend is. Be specific with numbers. Use $ for currency values."
                )
            else:
                # Build analytics context to enrich the summary
                _analytics_context = ""
                _da_now = compute_deep_analytics(df)
                if _da_now.get("anomalies"):
                    top_anom = sorted(_da_now["anomalies"].items(),
                                      key=lambda x: x[1]["count"], reverse=True)[:2]
                    anom_str = "; ".join(
                        f"{col}: {info['count']} outliers ({info['pct']}% of rows)"
                        for col, info in top_anom
                    )
                    _analytics_context += f"\nAnomalies detected: {anom_str}."
                if _da_now.get("drivers", {}).get("drivers"):
                    _drv = _da_now["drivers"]
                    top_driver = list(_drv["drivers"].items())[0]
                    _analytics_context += (
                        f"\nStrongest driver of {_drv['target']}: "
                        f"{top_driver[0]} (impact score={top_driver[1]['score']})."
                    )
                if _da_now.get("trend"):
                    _tr = _da_now["trend"]
                    _analytics_context += (
                        f"\nTrend: {_tr['metric_col']} is trending "
                        f"{_tr['direction']} ({_tr['pct_change']:+.1f}% over the period, "
                        f"R²={_tr['r2']})."
                    )

                summary_prompt = (
                    "You are a business analyst. Look at this dataset and write a 3-4 sentence "
                    "executive summary for a business manager. Include: what the data covers, "
                    "the date range if available, total records, and 2-3 of the most interesting "
                    "patterns or insights you can immediately identify. Be specific with numbers. "
                    "Use $ for currency values where appropriate."
                    + (_analytics_context and
                       f"\n\nAdditional findings to reference where relevant:{_analytics_context}"
                       or "")
                )
            try:
                st.session_state.request_count += 1
                st.session_state.summary = get_summary(df, summary_prompt, df2)
            except Exception:
                st.session_state.request_count -= 1
                st.session_state.summary = (
                    "Summary unavailable — ask your own questions below."
                )

    # ── Suggested questions: regenerate when file OR industry template changes ─
    if st.session_state.get("last_suggestion_key") != suggestion_key:
        st.session_state.last_suggestion_key = suggestion_key

        if df2 is not None:
            st.session_state.suggestions = [
                "What changed the most between the two files?",
                "Which categories grew and which declined?",
                "What is the overall change in total revenue?",
                "Which region improved the most?",
                "Are there any categories only in one file?",
                "What is the percentage change by segment?"
            ]
            st.session_state.advanced_questions = []
        elif industry != "Auto-detect from data":
            st.session_state.suggestions = INDUSTRY_TEMPLATES[industry]
            st.session_state.advanced_questions = []
        else:
            with st.spinner("Generating suggested questions..."):
                try:
                    st.session_state.request_count += 1
                    raw = get_suggested_questions(df.columns.tolist())
                    lines = [
                        line.strip() for line in raw.strip().split("\n")
                        if line.strip()
                    ]
                    st.session_state.suggestions = [
                        line.lstrip("0123456789.-) ").strip()
                        for line in lines if len(line) > 5
                    ][:6]
                except Exception:
                    st.session_state.request_count -= 1
                    st.session_state.suggestions = [
                        "What are the top 5 items by value?",
                        "Which category performs best?",
                        "What is the overall total?",
                        "Which entry has the highest value?",
                        "Show a breakdown by category",
                        "What are the key trends?"
                    ]

            with st.spinner("Generating advanced questions..."):
                try:
                    st.session_state.request_count += 1
                    raw_adv = get_advanced_questions(df)
                    adv_lines = [
                        line.strip() for line in raw_adv.strip().split("\n")
                        if line.strip()
                    ]
                    st.session_state.advanced_questions = [
                        line.lstrip("0123456789.-) ").strip()
                        for line in adv_lines if len(line) > 5
                    ][:8]
                except Exception:
                    st.session_state.request_count -= 1
                    st.session_state.advanced_questions = []

    # Executive summary display (strip chart JSON and any markdown headers)
    summary_clean = st.session_state.summary.split("---CHART---")[0].strip()
    summary_clean = "\n".join(
        line for line in summary_clean.split("\n")
        if not line.strip().startswith("#")
    ).strip()
    summary_clean = fix_dollar_signs(summary_clean)
    st.markdown("""
    <div class="section-header" style="margin-top:0.5rem;">
        <div class="section-header-icon">📋</div>
        <div class="section-header-text">Executive Summary</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="summary-box">
        <p style="margin:0;">{summary_clean}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Auto-generated Insight Cards ─────────────────────────────────────────
    _insight_cards = compute_insight_cards(df, _da)
    if _insight_cards:
        st.markdown("""
        <div class="section-header" style="margin-top:1.4rem;">
            <div class="section-header-icon">💡</div>
            <div class="section-header-text">Auto-Generated Insights</div>
        </div>
        """, unsafe_allow_html=True)
        _ic_cols = st.columns(len(_insight_cards))
        for _i, _card in enumerate(_insight_cards):
            with _ic_cols[_i]:
                st.markdown(f"""
                <div style="
                    background: {_card['bg']};
                    border-radius: 16px;
                    padding: 1.2rem 1.3rem 1.1rem 1.3rem;
                    border-top: 4px solid {_card['color']};
                    box-shadow: 0 4px 16px rgba(0,0,0,0.07), 0 1px 4px rgba(0,0,0,0.04);
                    height: 100%;
                    min-height: 130px;
                ">
                    <div style="font-size:0.62rem;font-weight:700;text-transform:uppercase;
                                letter-spacing:0.12em;color:{_card['color']};
                                margin-bottom:0.5rem;">
                        {_card['icon']} {_card['label']}
                    </div>
                    <div style="font-size:0.92rem;font-weight:800;color:#0d1b2a;
                                line-height:1.25;margin-bottom:0.45rem;
                                letter-spacing:-0.01em;">
                        {_card['headline']}
                    </div>
                    <div style="font-size:0.76rem;color:#4a5a75;line-height:1.55;">
                        {_card['detail']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── Refresh Executive Summary button ────────────────────────────────────
    _history = st.session_state.get("history", [])
    _refresh_col1, _refresh_col2, _refresh_col3 = st.columns([3, 2, 3])
    with _refresh_col2:
        _refresh_label = (
            "🔄 Refresh summary with Q&A findings"
            if _history else "🔄 Refresh summary"
        )
        if st.button(_refresh_label, use_container_width=True, key="refresh_summary_btn"):
            _at_limit = st.session_state.request_count >= MAX_REQUESTS_PER_SESSION
            _last_t = st.session_state.get("last_request_time")
            _on_cooldown = (
                _last_t is not None and
                (datetime.now() - _last_t).total_seconds() < COOLDOWN_SECONDS
            ) if isinstance(_last_t, datetime) else False
            if _at_limit:
                st.error(f"Session limit of {MAX_REQUESTS_PER_SESSION} requests reached. Refresh the page.")
            elif _on_cooldown:
                st.warning("Please wait a moment before refreshing.")
            else:
                with st.spinner("Refreshing executive summary..."):
                    if _history and df2 is None:
                        # Incorporate last-3 Q&A findings into the updated summary
                        _recent = _history[-3:]
                        _qa_context = "\n\n".join(
                            f"Q: {item['question']}\nA: {item['answer'].split('---CHART---')[0].strip()[:400]}"
                            for item in _recent
                        )
                        _refresh_prompt = (
                            "You are a business analyst. Using the dataset statistics below "
                            "AND the following recent analytical findings, write an updated "
                            "4-5 sentence executive summary for a business manager. Incorporate "
                            "specific insights surfaced in the Q&A. Be concrete with numbers. "
                            "Use $ for currency values where appropriate.\n\n"
                            f"Recent findings:\n{_qa_context}"
                        )
                    elif df2 is not None:
                        _refresh_prompt = (
                            "You are a business analyst. Two datasets have been uploaded for "
                            "comparison. Write a refreshed 3-4 sentence executive summary "
                            "highlighting the most important differences. Be specific with "
                            "numbers. Use $ for currency values."
                        )
                    else:
                        _refresh_prompt = (
                            "You are a business analyst. Look at this dataset and write a "
                            "refreshed 3-4 sentence executive summary for a business manager. "
                            "Include the date range if available, total records, and 2-3 of "
                            "the most interesting patterns or insights. Be specific with numbers. "
                            "Use $ for currency values where appropriate."
                        )
                    try:
                        st.session_state.request_count += 1
                        st.session_state.last_request_time = datetime.now()
                        st.session_state.summary = get_summary(df, _refresh_prompt, df2)
                        st.rerun()
                    except Exception as _e:
                        st.session_state.request_count -= 1
                        st.warning(f"Could not refresh summary: {_e}")

    # ── Deep Analytics Section ───────────────────────────────────────────────
    with st.expander("🔬 Deep Analytics — Anomalies · Drivers · Forecast", expanded=True):
        da_col1, da_col2, da_col3 = st.columns(3)

        # ── Panel 1: Anomaly Detection ────────────────────────────────────────
        with da_col1:
            st.markdown("""
            <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;
                        letter-spacing:0.1em;color:#3a4f70;margin-bottom:0.5rem;">
                ⚠️ Anomaly Detection
            </div>""", unsafe_allow_html=True)
            if _da.get("anomalies"):
                # Show the column with the most outliers as a scatter chart
                top_col, top_info = sorted(
                    _da["anomalies"].items(),
                    key=lambda x: x[1]["count"], reverse=True
                )[0]
                all_vals = top_info["all_vals"]
                all_z    = top_info["all_z"]
                colors   = ["#e53e3e" if abs(z) > 2.5 else "#0066cc" for z in all_z]
                anom_fig = go.Figure()
                anom_fig.add_trace(go.Scatter(
                    x=list(range(len(all_vals))),
                    y=all_vals,
                    mode="markers",
                    marker=dict(color=colors, size=5, opacity=0.75),
                    hovertemplate="%{y:.2f}<extra></extra>",
                ))
                anom_fig.update_layout(
                    title=dict(text=f"{top_col}", font=dict(size=12, color="#0d1b2a")),
                    margin=dict(l=10, r=10, t=30, b=10),
                    height=200,
                    plot_bgcolor="white", paper_bgcolor="white",
                    font=dict(color="#1a1a2e", size=11),
                    xaxis=dict(showticklabels=False, gridcolor="#e8edf5"),
                    yaxis=dict(gridcolor="#e8edf5"),
                    showlegend=False,
                )
                st.plotly_chart(anom_fig, use_container_width=True, config={"displayModeBar": False})
                # Plain language explanation
                top_anom_col, top_anom_info = sorted(
                    _da["anomalies"].items(), key=lambda x: x[1]["count"], reverse=True
                )[0]
                st.markdown(
                    f'<div style="font-size:0.78rem;color:#1a1a2e;line-height:1.6;">'
                    f'<span style="color:#e53e3e;">●</span> Red dots are unusual values — '
                    f'far higher or lower than typical. '
                    f'<strong>{top_anom_col}</strong> has '
                    f'<strong>{top_anom_info["count"]} unusual entries</strong> '
                    f'worth reviewing. These could be data errors, one-time events, '
                    f'or hidden opportunities.</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div style="font-size:0.82rem;color:#16a34a;padding:0.5rem 0;">'
                    '✓ No unusual values detected — your data looks consistent.</div>',
                    unsafe_allow_html=True
                )

        # ── Panel 2: Driver Analysis ──────────────────────────────────────────
        with da_col2:
            st.markdown("""
            <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;
                        letter-spacing:0.1em;color:#3a4f70;margin-bottom:0.5rem;">
                📊 Key Drivers
            </div>""", unsafe_allow_html=True)
            if _da.get("drivers", {}).get("drivers"):
                drv   = _da["drivers"]
                items = list(drv["drivers"].items())[:7]
                cols_d  = [i[0] for i in items]
                # Use absolute value for bar length; color by type
                vals_d  = [abs(i[1]["score"]) for i in items]
                colors_d = [
                    "#0066cc" if i[1]["type"] == "numeric" else "#6366f1"
                    for i in items
                ]
                drv_fig = go.Figure(go.Bar(
                    x=vals_d,
                    y=cols_d,
                    orientation="h",
                    marker_color=colors_d,
                    hovertemplate="%{x:.2f}<extra></extra>",
                ))
                drv_fig.update_layout(
                    title=dict(
                        text=f"Impact on {drv['target']}",
                        font=dict(size=12, color="#0d1b2a")
                    ),
                    margin=dict(l=10, r=10, t=30, b=10),
                    height=200,
                    plot_bgcolor="white", paper_bgcolor="white",
                    font=dict(color="#1a1a2e", size=11),
                    xaxis=dict(gridcolor="#e8edf5", range=[0, 1]),
                    yaxis=dict(autorange="reversed"),
                    showlegend=False,
                )
                st.plotly_chart(drv_fig, use_container_width=True, config={"displayModeBar": False})
                top_drv_name = items[0][0]
                top_drv_info = items[0][1]
                if top_drv_info["type"] == "numeric":
                    if top_drv_info["score"] > 0:
                        drv_plain = (
                            f'When <strong>{top_drv_name}</strong> goes up, '
                            f'<strong>{drv["target"]}</strong> tends to go up too.'
                        )
                    else:
                        drv_plain = (
                            f'When <strong>{top_drv_name}</strong> goes up, '
                            f'<strong>{drv["target"]}</strong> tends to go down.'
                        )
                else:
                    drv_plain = (
                        f'<strong>{top_drv_name}</strong> is the category that '
                        f'most influences <strong>{drv["target"]}</strong> — '
                        f'different groups have very different averages.'
                    )
                st.markdown(
                    f'<div style="font-size:0.78rem;color:#1a1a2e;line-height:1.6;">'
                    f'Longer bar = stronger influence. {drv_plain}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div style="font-size:0.82rem;color:#6a80a0;padding:0.5rem 0;">'
                    'Could not identify a target metric or driver columns.</div>',
                    unsafe_allow_html=True
                )

        # ── Panel 3: Trend & Forecast ─────────────────────────────────────────
        with da_col3:
            st.markdown("""
            <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;
                        letter-spacing:0.1em;color:#3a4f70;margin-bottom:0.5rem;">
                📈 Trend & Forecast
            </div>""", unsafe_allow_html=True)
            if _da.get("trend"):
                tr = _da["trend"]
                hist_dates  = tr["historical"]["dates"]
                hist_vals   = tr["historical"]["values"]
                fcast_dates = tr["forecast"]["dates"]
                fcast_vals  = tr["forecast"]["values"]

                trend_fig = go.Figure()
                trend_fig.add_trace(go.Scatter(
                    x=hist_dates, y=hist_vals,
                    mode="lines+markers",
                    name="Actual",
                    line=dict(color="#0066cc", width=2),
                    marker=dict(size=4),
                    hovertemplate="%{y:,.0f}<extra>Actual</extra>",
                ))
                # Connect last actual to first forecast
                trend_fig.add_trace(go.Scatter(
                    x=[hist_dates[-1]] + fcast_dates,
                    y=[hist_vals[-1]]  + fcast_vals,
                    mode="lines+markers",
                    name="Forecast",
                    line=dict(color="#6366f1", width=2, dash="dash"),
                    marker=dict(size=4, symbol="diamond"),
                    hovertemplate="%{y:,.0f}<extra>Forecast</extra>",
                ))
                arrow = "↑" if tr["direction"] == "up" else "↓"
                trend_fig.update_layout(
                    title=dict(
                        text=f"{tr['metric_col']} {arrow} {abs(tr['pct_change']):.1f}%",
                        font=dict(size=12, color="#0d1b2a")
                    ),
                    margin=dict(l=10, r=10, t=30, b=10),
                    height=200,
                    plot_bgcolor="white", paper_bgcolor="white",
                    font=dict(color="#1a1a2e", size=11),
                    xaxis=dict(gridcolor="#e8edf5"),
                    yaxis=dict(gridcolor="#e8edf5"),
                    legend=dict(font=dict(size=10)),
                    showlegend=True,
                )
                st.plotly_chart(trend_fig, use_container_width=True, config={"displayModeBar": False})
                trend_color = "#16a34a" if tr["direction"] == "up" else "#e53e3e"
                trend_word  = "growing" if tr["direction"] == "up" else "declining"
                confidence  = "strong" if tr["r2"] > 0.7 else ("moderate" if tr["r2"] > 0.4 else "weak")
                st.markdown(
                    f'<div style="font-size:0.78rem;color:#1a1a2e;line-height:1.6;">'
                    f'<strong>{tr["metric_col"]}</strong> is '
                    f'<strong style="color:{trend_color};">{trend_word}</strong> — '
                    f'up {abs(tr["pct_change"]):.1f}% over the full period. '
                    f'The purple dashed line is a 3-month projection based on this trend. '
                    f'Confidence in this trend is <strong>{confidence}</strong>.</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div style="font-size:0.82rem;color:#6a80a0;padding:0.5rem 0;">'
                    'No date column found. Add a date field to your data '
                    'to see trends and projections here.</div>',
                    unsafe_allow_html=True
                )

    # ── What-If Simulator ────────────────────────────────────────────────────
    _drv_data = _da.get("drivers", {})
    if _drv_data.get("drivers"):
        with st.expander("🎯 What-If Simulator — adjust drivers and see projected impact", expanded=False):
            _target      = _drv_data["target"]
            _tgt_mean    = _drv_data["target_mean"]
            _tgt_total   = _drv_data["target_total"]
            _n           = _drv_data["n_rows"]
            _is_currency = any(k in _target.lower() for k in
                               ["sales", "revenue", "profit", "income", "margin", "amount"])

            def _fmt(val):
                return f"${val:,.0f}" if _is_currency else f"{val:,.1f}"

            # Separate numeric and categorical drivers, guarantee up to 2 of each
            _all_numeric_drivers = [
                (col, info) for col, info in _drv_data["drivers"].items()
                if info["type"] == "numeric"
            ]
            _all_cat_drivers = [
                (col, info) for col, info in _drv_data["drivers"].items()
                if info["type"] == "categorical"
            ]
            _top_drivers = (
                _all_numeric_drivers[:2] + _all_cat_drivers[:2]
                if _all_numeric_drivers
                else _all_cat_drivers[:4]
            )

            _has_sliders = bool(_all_numeric_drivers)
            _control_desc = (
                "Use the sliders and dropdowns below"
                if _has_sliders and _all_cat_drivers
                else ("Move the sliders below" if _has_sliders else "Use the dropdowns below")
            )
            _no_slider_note = (
                "" if _has_sliders else
                ' <span style="color:#9aaccc;">(This dataset has no numeric driver columns '
                '— add columns like Quantity or Discount to unlock sliders.)</span>'
            )

            st.markdown(
                f'<div style="font-size:0.82rem;color:#3a4f70;margin-bottom:0.75rem;">'
                f'{_control_desc} to explore how changes in key drivers '
                f'affect <strong>{_target}</strong>. '
                f'Current baseline: <strong>{_fmt(_tgt_total)}</strong> total '
                f'({_fmt(_tgt_mean)} per row).{_no_slider_note}</div>',
                unsafe_allow_html=True
            )
            _total_delta = 0.0
            _driver_impacts = []

            for _dcol, _dinfo in _top_drivers:
                _sim_col1, _sim_col2 = st.columns([2, 1])
                if _dinfo["type"] == "numeric":
                    with _sim_col1:
                        _pct = st.slider(
                            f"Change in **{_dcol}** (%)",
                            min_value=-50, max_value=50, value=0, step=5,
                            key=f"whatsim_{_dcol}"
                        )
                    _delta_driver = _dinfo["col_mean"] * _pct / 100.0
                    _delta_target = _dinfo.get("slope", 0) * _delta_driver * _n
                    _total_delta += _delta_target
                    _driver_impacts.append((_dcol, _pct, _delta_target, "numeric"))
                    with _sim_col2:
                        _color = "#16a34a" if _delta_target >= 0 else "#e53e3e"
                        _sign  = "+" if _delta_target >= 0 else ""
                        st.markdown(
                            f'<div style="padding:0.4rem 0;font-size:0.8rem;color:{_color};">'
                            f'<strong>{_sign}{_fmt(_delta_target)}</strong> impact</div>',
                            unsafe_allow_html=True
                        )
                else:
                    # Categorical: dropdown
                    _group_means = _dinfo.get("group_means", {})
                    _options = ["(no change)"] + sorted(_group_means.keys())
                    with _sim_col1:
                        _sel = st.selectbox(
                            f"Shift mix toward **{_dcol}**:",
                            _options,
                            key=f"whatsim_{_dcol}"
                        )
                    if _sel != "(no change)" and _sel in _group_means:
                        _cat_mean = _group_means[_sel]
                        _delta_mean = _cat_mean - _tgt_mean
                        _delta_target = _delta_mean * _n
                        _total_delta += _delta_target
                        _driver_impacts.append((_dcol, _sel, _delta_target, "categorical"))
                        with _sim_col2:
                            _color = "#16a34a" if _delta_target >= 0 else "#e53e3e"
                            _sign  = "+" if _delta_target >= 0 else ""
                            st.markdown(
                                f'<div style="padding:0.4rem 0;font-size:0.8rem;color:{_color};">'
                                f'<strong>{_sign}{_fmt(_delta_target)}</strong> impact</div>',
                                unsafe_allow_html=True
                            )
                    else:
                        with _sim_col2:
                            st.markdown(
                                '<div style="padding:0.4rem 0;font-size:0.8rem;color:#aaa;">—</div>',
                                unsafe_allow_html=True
                            )

            # ── Result card ─────────────────────────────────────────────────
            _new_total  = _tgt_total + _total_delta
            _pct_impact = (_total_delta / _tgt_total * 100) if _tgt_total else 0
            _card_color = "#e8f5e9" if _total_delta >= 0 else "#fde8e8"
            _bar_color  = "#16a34a" if _total_delta >= 0 else "#e53e3e"
            _arrow      = "▲" if _total_delta >= 0 else "▼"
            _sign       = "+" if _total_delta >= 0 else ""

            st.markdown(
                f"""
                <div style="background:{_card_color};border-radius:14px;
                            padding:1.1rem 1.4rem;margin-top:1rem;
                            border-left:5px solid {_bar_color};">
                    <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;
                                letter-spacing:0.1em;color:{_bar_color};margin-bottom:0.4rem;">
                        Projected outcome
                    </div>
                    <div style="font-size:1.5rem;font-weight:900;color:#0d1b2a;
                                letter-spacing:-0.02em;">
                        {_fmt(_new_total)}
                        <span style="font-size:1rem;font-weight:700;color:{_bar_color};">
                            {_arrow} {_sign}{_fmt(_total_delta)} ({_sign}{_pct_impact:.1f}%)
                        </span>
                    </div>
                    <div style="font-size:0.78rem;color:#3a4f70;margin-top:0.3rem;">
                        Estimated total <strong>{_target}</strong> if these changes
                        were applied across all {_n:,} rows.
                        This is a directional estimate based on observed relationships in the data.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Question input
    st.markdown("""
    <div class="section-header">
        <div class="section-header-icon">💬</div>
        <div class="section-header-text">Ask a Question</div>
    </div>
    <div class="question-label">Suggested questions</div>
    """, unsafe_allow_html=True)

    suggestions = st.session_state.get("suggestions", [])
    cols = st.columns(3)
    for i, suggestion in enumerate(suggestions):
        if cols[i % 3].button(suggestion, key=f"suggestion_{i}"):
            st.session_state.question_input = suggestion

    # Advanced analysis questions — AI-generated and tailored to this dataset
    _adv_qs = st.session_state.get("advanced_questions", [])
    if _adv_qs:
        with st.expander("🔍 Advanced Analysis — click to expand"):
            st.caption(
                "Consultant-level questions generated specifically for your data — "
                "designed to surface hidden patterns and strategic insights."
            )
            adv_cols = st.columns(2)
            for i, q in enumerate(_adv_qs):
                if adv_cols[i % 2].button(q, key=f"advanced_{i}"):
                    st.session_state.question_input = q

    question = st.text_input(
        "Type your question or request here:",
        placeholder="e.g. Which region had the highest sales? or Show me the monthly sales trend.",
        key="question_input"
    )

    # Show usage counter
    remaining = MAX_REQUESTS_PER_SESSION - st.session_state.request_count
    if remaining <= 5:
        st.caption(f"⚠️ {remaining} analysis request(s) remaining this session.")

    if st.button("Analyze", type="primary") and question:
        # Check session limit
        if st.session_state.request_count >= MAX_REQUESTS_PER_SESSION:
            st.error(
                f"You've reached the {MAX_REQUESTS_PER_SESSION}-request limit for this session. "
                f"Refresh the page to start a new session."
            )
        # Check cooldown
        elif (
            st.session_state.last_request_time is not None and
            (datetime.now() - st.session_state.last_request_time).total_seconds() < COOLDOWN_SECONDS
        ):
            elapsed = (datetime.now() - st.session_state.last_request_time).total_seconds()
            st.warning(
                f"Please wait {int(COOLDOWN_SECONDS - elapsed) + 1} second(s) before submitting again."
            )
        else:
            with st.spinner("Analyzing your data..."):
                try:
                    st.session_state.last_request_time = datetime.now()
                    st.session_state.request_count += 1
                    raw_answer = get_answer(df, question, df2)
                    clean_answer, chart_data = try_render_chart(raw_answer)
                    _encoded = base64.b64encode(clean_answer.encode()).decode()
                    st.markdown("""
                    <div class="section-header" style="margin-top:1.2rem;">
                        <div class="section-header-icon">✨</div>
                        <div class="section-header-text">AI Answer</div>
                    </div>
                    """, unsafe_allow_html=True)
                    _answer_html = _md_to_html(fix_dollar_signs(clean_answer))
                    st.markdown(
                        f'<div class="answer-card">{_answer_html}</div>',
                        unsafe_allow_html=True
                    )
                    components.html(
                        f"""<button onclick="
                            navigator.clipboard.writeText(atob('{_encoded}')).then(()=>{{
                                this.textContent='✓ Copied!';
                                this.style.color='#16a34a';
                                this.style.borderColor='#bbf7d0';
                                setTimeout(()=>{{this.textContent='📋 Copy';
                                    this.style.color='#0066cc';
                                    this.style.borderColor='#dce8f8';}},2000);
                            }});"
                            style="background:transparent;border:1.5px solid #dce8f8;
                                   border-radius:20px;padding:3px 12px;font-size:11px;
                                   color:#0066cc;cursor:pointer;font-family:inherit;
                                   margin-top:2px;">
                            📋 Copy
                        </button>""",
                        height=36,
                    )
                    st.session_state.history.append({
                        "question": question,
                        "answer": clean_answer,       # unescaped — used in HTML report
                        "chart_data": chart_data      # None if no chart
                    })
                except Exception as e:
                    st.session_state.request_count -= 1  # don't count failed requests
                    st.warning("I couldn't answer that. Try rephrasing your question.")
                    st.caption(f"Technical detail: {str(e)}")

    # Conversation history — exclude the most recent item (already shown above)
    _past_history = st.session_state.history[:-1]
    if _past_history:
        st.markdown("""
        <div class="section-header">
            <div class="section-header-icon">🕐</div>
            <div class="section-header-text">Previous Questions</div>
        </div>
        """, unsafe_allow_html=True)
        for item in reversed(_past_history[-5:]):
            _a_preview = fix_dollar_signs(item["answer"])[:320]
            if len(item["answer"]) > 320:
                _a_preview += "…"
            st.markdown(f"""
            <div class="history-item">
                <div class="history-q">
                    <span style="color:#0066cc;margin-top:1px;">▸</span>
                    {item['question']}
                </div>
                <div class="history-a">{_a_preview}</div>
            </div>
            """, unsafe_allow_html=True)

        # Download report
        st.divider()
        report_html = generate_report_html(
            effective_name,
            st.session_state.get("summary", ""),
            st.session_state.history,
            filename2=uploaded_file2.name if uploaded_file2 is not None else None
        )
        st.download_button(
            label="📥 Download Analysis Report",
            data=report_html,
            file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
            mime="text/html"
        )
        st.caption(
            "Tip: Open the downloaded file in any browser — charts are included. "
            "To save as PDF, press Cmd+P (Mac) or Ctrl+P (Windows) and choose 'Save as PDF'."
        )
