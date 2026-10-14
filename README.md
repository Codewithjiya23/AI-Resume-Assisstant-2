# 📄 Resume ATS Checker

Upload a resume (PDF, DOCX or TXT) and get an **estimated ATS score**, a category breakdown, prioritised improvements, keyword gaps and bullet-point rewrites. Paste an optional job description to also get a **job-match score**.

Built with **Streamlit** and **Google Gemini Flash**.

> ⚠️ No ATS publishes a real score. The number here is an AI-based estimate of how readable and well-targeted your resume is. Use it as guidance, not a guarantee.

## Features
- PDF / DOCX / TXT upload (max 5 MB), including text inside DOCX tables
- Overall score computed in code from weighted categories: formatting, sections, keywords, impact, readability (+ job match)
- Prioritised fixes (High / Medium / Low), strengths, found vs. missing keywords
- Bullet rewrites with placeholders (e.g. `[X%]`) so nothing is invented
- Instant rule-based checks (email, phone, links, length, sections)
- Downloadable Markdown report

## Run locally
```bash
git clone https://github.com/<your-username>/resume-ats-checker.git
cd resume-ats-checker
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Get a free Gemini API key at https://aistudio.google.com/apikey, then provide it in one of three ways:

1. **Sidebar box** in the app (easiest, session only)
2. **Environment variable:** `export GEMINI_API_KEY="your-key"`
3. **Secrets file:** create `.streamlit/secrets.toml`
   ```toml
   GEMINI_API_KEY = "your-key"
   # GEMINI_MODEL = "gemini-3.8-flash"   # optional override
   ```

> Never commit `.streamlit/secrets.toml` or any API key to GitHub.

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub (public or private).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app** → choose your repo, branch `main`, main file `app.py`.
4. Open **Advanced settings → Secrets** and paste:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```
5. Click **Deploy**.

**Note:** if you put your key in Secrets, every visitor uses *your* quota. For a public demo, either leave the secret empty (users enter their own key in the sidebar) or monitor usage.

## Configuration
| Setting | Where | Default |
|---|---|---|
| `GEMINI_API_KEY` | secrets / env / sidebar | none |
| `GEMINI_MODEL` | secrets / env | `gemini-3.8-flash` |

## Limitations
- Scanned/image-only PDFs can't be read (an ATS can't read them either).
- Layout problems (columns, text boxes) are inferred from extracted text, not from rendering the page.
- Scores may vary slightly between runs.
- Google retires Gemini models often. If you see a "model not found" error, set `GEMINI_MODEL` to a current Flash model listed at https://ai.google.dev/gemini-api/docs/models (the app also falls back to `gemini-3.6-flash` and `gemini-3.5-flash-lite` automatically).

## Project structure
```
app.py             # Streamlit app + Gemini logic
requirements.txt   # Python dependencies
README.md
```
