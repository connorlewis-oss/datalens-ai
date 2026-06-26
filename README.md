ARIA Business Intelligence Solution
AI Research & Insight Analyst
ARIA is an AI-powered business intelligence tool built with Streamlit and Claude (Anthropic). Upload a spreadsheet or connect a Google Sheet and get instant, consultant-level analysis in plain English — no SQL, no dashboards, no data team required.


Features
Data input

Upload CSV or Excel files (up to 10 MB)
Connect a live Google Sheet — paste the URL and hit Refresh to pull the latest data
Compare two datasets side by side
Built-in retail demo dataset (no upload needed)

Analysis

Executive summary generated on file load
Plain English Q&A — ask anything about your data
Auto-generated charts (bar, line, scatter)
Anomaly detection — flags outliers automatically
Driver analysis — identifies what's most correlated with your key metric
Trend & forecast — projects the next 3 periods from historical data
What-if simulator — adjust a variable and see the projected impact
Statistical significance testing between groups

UX

AI-generated suggested questions grounded in your actual column names
Follow-up questions after every answer
Conversation thread — scroll back through the full session
Confidence indicator on every answer (High / Medium / Low)
Copy button on answers
Column profiling panel
Data cleaning suggestions (missing values, text-as-numbers)
PDF report export and HTML report download
Onboarding tour for new users


Setup — Local
1. Clone the repo

git clone https://github.com/YOUR_USERNAME/ai-data-analyst.git

cd ai-data-analyst

2. Install dependencies

pip install -r requirements.txt

3. Add your Anthropic API key

Create a .env file in the project root:

ANTHROPIC_API_KEY=sk-ant-your-key-here

Get a key at console.anthropic.com. Never commit this file — it is already in .gitignore.

4. Run the app

python3 -m streamlit run app.py

The app opens at http://localhost:8501.


Google Sheets Connector
ARIA can read live data directly from Google Sheets — no file upload needed.

Open your Google Sheet
Click Share → Change to "Anyone with the link" → Viewer
Copy the URL
In ARIA, click the 🔗 Google Sheets tab, paste the URL, and click Load Sheet
Use the 🔄 Refresh button anytime to pull the latest data from the sheet

Note: the sheet must be publicly viewable for the connector to work. Private sheet support (via OAuth) is on the roadmap.


Deployment — Streamlit Cloud
Push the repo to GitHub (never include your .env file)
Go to share.streamlit.io and connect the repo
In Advanced settings → Secrets, add:

ANTHROPIC_API_KEY = "sk-ant-your-key-here"

Deploy — Streamlit Cloud installs requirements.txt automatically


Project Structure
ai-data-analyst/

├── app.py              # Main application

├── requirements.txt    # Python dependencies

├── .env                # API key (local only — never commit)

├── .gitignore

└── README.md


Dependencies
Package
Purpose
streamlit
Web app framework
anthropic
Claude API client
pandas
Data loading and processing
plotly
Interactive charts
scipy
Statistical significance tests
fpdf2
PDF report generation
tabulate
Data formatting for AI prompts
python-dotenv
Load .env file locally



Built by
Connor Lewis — built with Claude (Anthropic) via the ARIA AI Research & Insight Analyst project.


