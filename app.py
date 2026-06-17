import streamlit as st
import pandas as pd
import os
import anthropic
import plotly.express as px
import json
import re
import html as html_lib
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ── Anthropic client (initialized once) ───────────────────────────────────────
_api_key = os.getenv("ANTHROPIC_API_KEY")
if not _api_key:
    st.error(
        "**ANTHROPIC_API_KEY not found.** "
        "Create a `.env` file in the project folder with:\n\n"
        "```\nANTHROPIC_API_KEY=sk-ant-your-key-here\n```\n\n"
        "Get a key at [console.anthropic.com](https://console.anthropic.com)."
    )
    st.stop()
client = anthropic.Anthropic(api_key=_api_key)

# ── Rate limiting config ───────────────────────────────────────────────────────
MAX_REQUESTS_PER_SESSION = 25   # max AI calls per browser session
COOLDOWN_SECONDS = 3            # minimum seconds between requests
MAX_FILE_SIZE_MB = 10           # max upload size in MB

# ── Session state ──────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "question_input" not in st.session_state:
    st.session_state.question_input = ""
if "request_count" not in st.session_state:
    st.session_state.request_count = 0
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
    page_title="AI Business Analyst",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
    /* ── Global ── */
    .main { padding: 1.5rem 2rem; }
    h1 { font-size: 1.9rem !important; font-weight: 700 !important; color: #0d1b2a !important; }
    h3 { font-size: 1.05rem !important; font-weight: 600 !important;
         color: #0d1b2a !important; text-transform: uppercase;
         letter-spacing: 0.05em; margin-top: 1.5rem !important; }

    /* ── Buttons ── */
    .stButton > button {
        background-color: #0066cc;
        color: white;
        border-radius: 20px;
        padding: 0.35rem 1rem;
        font-size: 0.8rem;
        font-weight: 500;
        border: none;
        transition: background 0.2s;
    }
    .stButton > button:hover { background-color: #0052a3; }

    /* ── Primary Get Answer button ── */
    .stButton > button[kind="primary"] {
        background-color: #0066cc;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-size: 0.95rem;
        font-weight: 600;
    }

    /* ── Answer box ── */
    .answer-box {
        background: #f8faff;
        border-left: 4px solid #0066cc;
        border-radius: 0 8px 8px 0;
        padding: 1.2rem 1.5rem;
        margin-top: 0.75rem;
        line-height: 1.7;
        color: #1a1a2e;
        white-space: pre-wrap;
    }

    /* ── Executive summary ── */
    .summary-box {
        background: linear-gradient(135deg, #e8f0fe 0%, #f0f4ff 100%);
        border-radius: 10px;
        padding: 1.1rem 1.4rem;
        border: 1px solid #c5d5f5;
        font-size: 0.95rem;
        line-height: 1.65;
        color: #1a1a2e;
    }

    /* ── Metric cards ── */
    [data-testid="metric-container"] {
        background: #f8faff;
        border: 1px solid #e0e8f5;
        border-radius: 10px;
        padding: 0.8rem 1rem;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #0d1b2a;
    }
    [data-testid="stSidebar"] * { color: #e8edf5 !important; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stRadio label { color: #b0bcd4 !important; }
    [data-testid="stSidebar"] hr { border-color: #2a3f5f !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("AI Business Analyst")
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
    st.write("**Built by:** Connor Lewis")
    st.write("**Stack:** Claude API · Streamlit · Plotly")

# ── Main heading ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 0.5rem 0 1.5rem 0; border-bottom: 2px solid #e0e8f5; margin-bottom: 1.5rem;">
    <h1 style="margin-bottom: 0.25rem;">📊 AI Business Data Analyst</h1>
    <p style="color: #5a6a85; font-size: 1rem; margin: 0;">
        Upload any business spreadsheet and ask questions in plain English.
        Powered by Anthropic Claude.
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
def build_df_info(df, label="Dataset"):
    """
    Build a concise summary with pre-calculated group totals.

    Priority columns (Category, Segment, Region, etc.) are always included
    first regardless of their position in the dataframe, so Claude always
    has the most business-relevant breakdowns available.
    """

    # ── Numeric columns to aggregate (skip ID-like fields) ────────────────────
    skip_num = ["id", "row", "postal", "code", "zip"]
    priority_num_names = ["sales", "revenue", "profit", "amount", "total",
                          "price", "cost", "spend", "income", "earnings", "value"]
    all_num = [
        c for c in df.select_dtypes(include="number").columns
        if not any(kw in c.lower() for kw in skip_num)
    ]
    # Sort so priority columns come first, then the rest in original order
    numeric_cols = (
        [c for c in all_num if any(p in c.lower() for p in priority_num_names)] +
        [c for c in all_num if not any(p in c.lower() for p in priority_num_names)]
    )

    # ── Categorical columns: priority-first ordering ───────────────────────────
    # These names are checked with exact case-insensitive matching first,
    # so "Category" is never buried behind less-useful columns.
    priority_names = [
        "category", "segment", "region", "sub-category", "subcategory",
        "ship mode", "state", "country", "department", "product name",
        "customer name", "channel", "brand", "division", "team"
    ]

    skip_cat = ["id", "date", "time", "postal", "code", "zip",
                "phone", "address", "row", "order", "name"]

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
            if not any(kw in c.lower() for kw in skip_cat):
                useful_cats.append(c)

    # ── Build group summaries (categorical) ───────────────────────────────────
    group_summaries = ""
    for cat_col in useful_cats[:5]:           # top 5 categorical columns
        for num_col in numeric_cols[:3]:       # top 3 numeric columns
            try:
                summary = (
                    df.groupby(cat_col)[num_col]
                    .agg(Total="sum", Average="mean", Std_Dev="std",
                         Min="min", Max="max", Count="count")
                    .round(2)
                    .sort_values("Total", ascending=False)
                    .head(10)
                )
                # Add % of total so Claude never needs to calculate it
                grand_total = summary["Total"].sum()
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
        priority_num = ["sales", "revenue", "amount", "total", "profit", "price",
                        "cost", "spend", "income", "earnings", "value"]
        num_col = next(
            (c for c in numeric_cols if any(p in c.lower() for p in priority_num)),
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

    return f"""
{label}:
Columns: {', '.join(df.columns.tolist())}
Rows: {df.shape[0]:,}
Date column detected: {parsed_date_col if parsed_date_col else "None"}

Pre-calculated statistics (use these for accurate answers — do not guess):
{group_summaries}
{time_summaries}
Sample (first 5 rows):
{df.head(5).to_string()}
"""


@st.cache_data
def build_comparison_info(df1, df2):
    """
    Pre-calculate differences between two datasets so Claude can answer
    comparison questions accurately (e.g. this year vs last year).
    """
    skip_num = ["id", "row", "postal", "code", "zip"]

    num1 = {c for c in df1.select_dtypes(include="number").columns
            if not any(kw in c.lower() for kw in skip_num)}
    num2 = {c for c in df2.select_dtypes(include="number").columns
            if not any(kw in c.lower() for kw in skip_num)}
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


def get_answer(df, question, df2=None):
    """Send question and data to Claude and return the answer."""
    df_info = build_df_info(df, "Dataset 1")
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
1. Answer using ONLY the pre-calculated statistics provided above — never guess or estimate.
2. Do NOT perform any arithmetic yourself. Every number you state must be copied directly
   from the pre-calculated data above. Percentages, totals, and differences are already
   provided — use them as-is. If a number is not pre-calculated, do not include it.
3. Format numbers clearly: write "25,043" not "$25,043" (do not use dollar signs at all).
4. Always name both parties explicitly in comparisons — never say "roughly X vs Y" without names.
5. Structure your answer as a numbered list of findings.
6. End with a "Key Insight:" section that goes beyond the obvious.
   - Do NOT state things any manager already knows (e.g., "top performers perform well").
   - DO identify hidden risks, structural patterns, or strategic implications.
   - Think like a consultant advising the CEO: what decision does this number actually drive?

If the answer involves comparing multiple items (top products, regional breakdown, etc.),
end your response with this exact separator on its own line, then the JSON on the next line:
---CHART---
{{"chart": {{"type": "bar", "labels": ["A","B","C"], "values": [100,200,300], "title": "Chart Title"}}}}

Only include the ---CHART--- separator and JSON if a chart would genuinely add value. Otherwise answer in plain text only."""
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
    """Extract chart JSON from the answer (after ---CHART--- separator) and render it."""
    if "---CHART---" in answer_text:
        parts = answer_text.split("---CHART---", 1)
        clean_text = parts[0].strip()
        json_block = parts[1].strip()
        # Strip any code fences around the JSON
        json_block = re.sub(r'^```(?:json)?\s*', '', json_block).strip()
        json_block = re.sub(r'\s*```$', '', json_block).strip()
        try:
            chart_data = json.loads(json_block)["chart"]
            labels = chart_data.get("labels", [])
            values = chart_data.get("values", [])
            title = chart_data.get("title", "Chart")
            if labels and values and len(labels) == len(values):
                chart_df = pd.DataFrame({"Category": labels, "Value": values})
                fig = px.bar(
                    chart_df,
                    x="Category",
                    y="Value",
                    title=title,
                    color="Value",
                    color_continuous_scale="Blues"
                )
                fig.update_layout(xaxis_title="Category", yaxis_title="Value")
                st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass
        return clean_text
    # No separator — strip any stray code fences and return as-is
    return re.sub(r'```.*?```', '', answer_text, flags=re.DOTALL).strip()


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


def generate_report_html(filename, summary, history):
    """Generate a formatted HTML report of the session for download."""
    now = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    rows = ""
    for i, item in enumerate(history, 1):
        safe_q = html_lib.escape(item['question'])
        safe_a = html_lib.escape(item['answer'])
        rows += f"""
        <div class="qa-block">
            <div class="question">Q{i}: {safe_q}</div>
            <div class="answer">{safe_a}</div>
        </div>
        """
    safe_filename = html_lib.escape(filename)
    safe_summary = html_lib.escape(summary)
    return f"""
    <html>
    <head>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 820px; margin: 40px auto; color: #333; line-height: 1.6; }}
        h1 {{ color: #0066cc; border-bottom: 2px solid #0066cc; padding-bottom: 0.5rem; }}
        h2 {{ color: #0066cc; margin-top: 2rem; }}
        .meta {{ color: #666; font-size: 0.9rem; margin-bottom: 2rem; }}
        .summary {{ background: #f0f7ff; padding: 1.2rem; border-radius: 6px; margin-bottom: 2rem; white-space: pre-wrap; }}
        .qa-block {{ margin-bottom: 1.5rem; border-bottom: 1px solid #eee; padding-bottom: 1.5rem; }}
        .question {{ font-weight: bold; color: #0066cc; margin-bottom: 0.5rem; }}
        .answer {{ white-space: pre-wrap; }}
        .footer {{ margin-top: 3rem; color: #999; font-size: 0.8rem; border-top: 1px solid #eee; padding-top: 1rem; }}
    </style>
    </head>
    <body>
        <h1>Business Data Analysis Report</h1>
        <div class="meta">
            <strong>File:</strong> {safe_filename} &nbsp;|&nbsp;
            <strong>Generated:</strong> {now}
        </div>
        <h2>Executive Summary</h2>
        <div class="summary">{safe_summary}</div>
        <h2>Analysis</h2>
        {rows if rows else "<p>No questions asked yet.</p>"}
        <div class="footer">
            Generated by AI Business Data Analyst &nbsp;·&nbsp; Powered by Anthropic Claude
        </div>
    </body>
    </html>
    """


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

# ── Main analysis ──────────────────────────────────────────────────────────────
if uploaded_file is not None:
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

    if df2 is not None:
        st.success(
            f"✅ Files loaded: **{uploaded_file.name}** ({len(df):,} rows) "
            f"and **{uploaded_file2.name}** ({len(df2):,} rows)"
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("File 1 Rows", f"{len(df):,}")
        m2.metric("File 2 Rows", f"{len(df2):,}")
        m3.metric("File 1 Columns", len(df.columns))
        m4.metric("File 2 Columns", len(df2.columns))
    else:
        st.success(f"✅ **{uploaded_file.name}** loaded successfully")
        date_range = get_date_range(df)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows", f"{len(df):,}")
        m2.metric("Columns", len(df.columns))
        num_cols = len(df.select_dtypes(include="number").columns)
        m3.metric("Numeric Columns", num_cols)
        m4.metric("Date Range", date_range if date_range else "N/A")

    # Data quality alerts
    alerts = check_data_quality(df)
    if alerts:
        with st.expander(
            f"⚠️ {len(alerts)} data quality issue(s) found — click to review",
            expanded=False
        ):
            for alert in alerts:
                st.warning(alert)
    else:
        st.success("✅ No data quality issues detected.")

    # Data preview
    st.write("### Preview of Your Data")
    st.dataframe(df.head(20), use_container_width=True)

    # Executive summary and suggested questions (only regenerate when file changes)
    file_key = (
        uploaded_file.name + str(uploaded_file.size) +
        (uploaded_file2.name + str(uploaded_file2.size) if uploaded_file2 else "")
    )

    if "summary" not in st.session_state or st.session_state.get("last_file") != file_key:
        st.session_state.last_file = file_key

        with st.spinner("Generating executive summary..."):
            if df2 is not None:
                summary_prompt = """
                You are a business analyst. Two datasets have been uploaded for comparison.
                Write a 3-4 sentence executive summary highlighting the most important
                differences between File 1 and File 2. Focus on what changed — which
                totals went up or down, which categories shifted, and what the overall
                trend is. Be specific with numbers. Do not use dollar signs.
                """
            else:
                summary_prompt = """
                You are a business analyst. Look at this dataset and write a 3-4 sentence
                executive summary for a business manager. Include: what the data covers,
                the date range if available, total records, and 2-3 of the most interesting
                patterns or insights you can immediately identify. Be specific with numbers.
                Do not use dollar signs — write amounts as plain numbers.
                """
            try:
                st.session_state.request_count += 1
                st.session_state.summary = get_answer(df, summary_prompt, df2)
            except Exception:
                st.session_state.request_count -= 1
                st.session_state.summary = (
                    "Summary unavailable — ask your own questions below."
                )

        # Suggested questions: comparison-specific or auto-generate
        if df2 is not None:
            st.session_state.suggestions = [
                "What changed the most between the two files?",
                "Which categories grew and which declined?",
                "What is the overall change in total revenue?",
                "Which region improved the most?",
                "Are there any categories only in one file?",
                "What is the percentage change by segment?"
            ]
        elif industry != "Auto-detect from data":
            st.session_state.suggestions = INDUSTRY_TEMPLATES[industry]
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

    # Executive summary display (strip any chart JSON Claude may have included)
    summary_clean = st.session_state.summary.split("---CHART---")[0].strip()
    summary_clean = fix_dollar_signs(summary_clean)
    st.markdown(f"""
    <div class="summary-box">
        <strong style="font-size:0.8rem; text-transform:uppercase; letter-spacing:0.07em;
                       color:#0066cc;">Executive Summary</strong>
        <p style="margin: 0.5rem 0 0 0;">{summary_clean}</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Question input
    st.write("### Ask a Question")
    st.write("**Or try one of these:**")

    suggestions = st.session_state.get("suggestions", [])
    cols = st.columns(3)
    for i, suggestion in enumerate(suggestions):
        if cols[i % 3].button(suggestion, key=f"suggestion_{i}"):
            st.session_state.question_input = suggestion

    # Advanced analysis questions
    with st.expander("🔍 Advanced Analysis — click to expand"):
        st.caption(
            "Consultant-level questions designed to surface hidden patterns "
            "and strategic insights."
        )
        advanced_questions = [
            "What is our revenue concentration risk across customers?",
            "Which customers have high sales but low profitability?",
            "Where are we leaving money on the table?",
            "What is the 80/20 breakdown of our revenue?",
            "Which segments are growing and which are declining?",
            "What does our profit efficiency look like across categories?",
            "Which products have the worst profit margin despite high sales volume?",
            "Are there any outliers that are distorting our averages?",
            "What would happen to total revenue if we lost our top 3 customers?",
            "Which category has the highest variance in order size?"
        ]
        adv_cols = st.columns(2)
        for i, q in enumerate(advanced_questions):
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
                    clean_answer = try_render_chart(raw_answer)
                    display_answer = fix_dollar_signs(clean_answer)
                    st.markdown("### Answer")
                    st.markdown(
                        f'<div class="answer-box">{display_answer}</div>',
                        unsafe_allow_html=True
                    )
                    st.session_state.history.append({
                        "question": question,
                        "answer": clean_answer   # unescaped — used in HTML report
                    })
                except Exception as e:
                    st.session_state.request_count -= 1  # don't count failed requests
                    st.warning("I couldn't answer that. Try rephrasing your question.")
                    st.caption(f"Technical detail: {str(e)}")

    # Conversation history
    if st.session_state.history:
        st.divider()
        st.write("### Previous Questions")
        for item in reversed(st.session_state.history[-5:]):
            with st.expander(f"Q: {item['question']}"):
                st.markdown(fix_dollar_signs(item["answer"]))

        # Download report
        st.divider()
        report_html = generate_report_html(
            uploaded_file.name,
            st.session_state.get("summary", ""),
            st.session_state.history
        )
        st.download_button(
            label="📥 Download Analysis Report",
            data=report_html,
            file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
            mime="text/html"
        )
        st.caption(
            "Tip: Open the downloaded file in your browser, "
            "then press Cmd+P and choose 'Save as PDF'. "
            "Note: charts are not included in the downloaded report."
        )
