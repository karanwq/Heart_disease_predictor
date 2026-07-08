import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, url_for
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

# sentence-transformers and faiss removed to stay within Render free-tier memory limit
faiss = None
SentenceTransformer = None

try:
    from groq import Groq
except ImportError:
    Groq = None

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
MODEL_PATH  = BASE_DIR / "health.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
DATA_PATH   = BASE_DIR / "heart.csv"
PDF_DIR     = BASE_DIR / "rag_pdfs"

# ── Config ─────────────────────────────────────────────────────────────────────
CHUNK_WORDS         = 120
CHUNK_OVERLAP       = 30
SEMANTIC_MODEL_NAME = os.environ.get("SEMANTIC_MODEL_NAME", "all-MiniLM-L6-v2")
GROQ_API_KEY        = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_MODEL_NAME     = os.environ.get("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
MODEL_AUC           = os.environ.get("MODEL_AUC", "").strip() or None

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

FIELD_RANGES = {
    "age":      (1,   120),
    "sex":      (0,   1),
    "cp":       (0,   3),
    "trestbps": (80,  200),
    "chol":     (100, 600),
    "fbs":      (0,   1),
    "restecg":  (0,   2),
    "thalach":  (60,  220),
    "exang":    (0,   1),
    "oldpeak":  (0,   7),
    "slope":    (0,   2),
    "ca":       (0,   4),
    "thal":     (1,   3),
}

FIELDS = [
    {"name": "age",      "label": "Age",             "hint": "years",        "placeholder": "54",  "min": "1",   "max": "120", "step": "1"},
    {"name": "sex",      "label": "Sex",             "hint": "0=F, 1=M",     "placeholder": "1",   "min": "0",   "max": "1",   "step": "1"},
    {"name": "cp",       "label": "Chest Pain",      "hint": "0–3",          "placeholder": "0",   "min": "0",   "max": "3",   "step": "1"},
    {"name": "trestbps", "label": "Resting BP",      "hint": "mm Hg",        "placeholder": "130", "min": "80",  "max": "200", "step": "1"},
    {"name": "chol",     "label": "Cholesterol",     "hint": "mg/dl",        "placeholder": "220", "min": "100", "max": "600", "step": "1"},
    {"name": "fbs",      "label": "Fasting Sugar",   "hint": ">120 mg/dl",   "placeholder": "0",   "min": "0",   "max": "1",   "step": "1"},
    {"name": "restecg",  "label": "Rest ECG",        "hint": "0–2",          "placeholder": "1",   "min": "0",   "max": "2",   "step": "1"},
    {"name": "thalach",  "label": "Max Heart Rate",  "hint": "bpm",          "placeholder": "150", "min": "60",  "max": "220", "step": "1"},
    {"name": "exang",    "label": "Exercise Angina", "hint": "0=No, 1=Yes",  "placeholder": "0",   "min": "0",   "max": "1",   "step": "1"},
    {"name": "oldpeak",  "label": "ST Depression",   "hint": "oldpeak",      "placeholder": "1.0", "min": "0",   "max": "7",   "step": "0.1"},
    {"name": "slope",    "label": "Slope",           "hint": "0–2",          "placeholder": "1",   "min": "0",   "max": "2",   "step": "1"},
    {"name": "ca",       "label": "Major Vessels",   "hint": "0–4",          "placeholder": "0",   "min": "0",   "max": "4",   "step": "1"},
    {"name": "thal",     "label": "Thalassemia",     "hint": "1–3",          "placeholder": "2",   "min": "1",   "max": "3",   "step": "1"},
]

SYSTEM_PROMPT = """You are a knowledgeable, empathetic heart-health assistant built into
a cardiovascular risk tool. Your job is to help users understand their results and
general heart-health concepts.

Rules:
- Base answers on the provided medical context when relevant.
- Be concise (3-5 sentences max unless the user asks for more detail).
- Never diagnose or prescribe — always recommend seeing a doctor for personal advice.
- If a question is completely unrelated to health or the tool, politely redirect.
- Use plain English; avoid jargon unless the user clearly prefers clinical terms.
"""

# Fallback knowledge when no PDFs are loaded
BUILTIN_DOCS = [
    "High blood pressure can occur due to stress, high salt intake, lack of exercise, or obesity.",
    "Cholesterol buildup in arteries reduces blood flow and increases heart disease risk.",
    "Smoking damages blood vessels and increases blood pressure.",
    "Regular exercise helps lower blood pressure and improves heart health.",
    "Obesity increases strain on the heart and raises blood pressure.",
    "Excess salt intake causes water retention which increases blood pressure.",
    "Diabetes damages blood vessels and contributes to heart disease.",
    "Chest pain, exercise angina, and abnormal ECG findings can be warning signs that need medical review.",
    "A resting heart rate above 100 bpm or irregular rhythm may indicate cardiac stress.",
    "ST depression on an ECG during exercise is a marker for reduced coronary blood flow.",
    "Thalassemia and other blood disorders can increase cardiovascular strain over time.",
    "The number of major vessels colored by fluoroscopy is a direct marker of coronary artery disease severity.",
]

STOP_WORDS = {
    "a","about","above","after","again","against","all","am","an","and","any","are","as","at",
    "be","because","been","before","being","below","between","both","but","by","can","could",
    "did","do","does","doing","down","during","each","few","for","from","further","had","has",
    "have","having","he","her","here","hers","herself","him","himself","his","how","i","if",
    "in","into","is","it","its","itself","just","me","more","most","my","myself","no","nor",
    "not","now","of","off","on","once","only","or","other","our","ours","ourselves","out",
    "over","own","same","she","should","so","some","such","than","that","the","their","theirs",
    "them","themselves","then","there","these","they","this","those","through","to","too",
    "under","until","up","very","was","we","were","what","when","where","which","while","who",
    "whom","why","will","with","you","your","yours","yourself","yourselves",
}

# ── Model loading ──────────────────────────────────────────────────────────────
def train_and_save_model():
    df = pd.read_csv(DATA_PATH)
    X, y = df[FEATURE_NAMES], df["target"]
    stratify = y if y.nunique() > 1 else None
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
    sc = StandardScaler()
    X_train_s = sc.fit_transform(X_train)
    # Matches notebook: 200 estimators, min_samples_leaf=2
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf.fit(X_train_s, y_train)
    with MODEL_PATH.open("wb") as f: pickle.dump(rf, f)
    with SCALER_PATH.open("wb") as f: pickle.dump(sc, f)
    return rf, sc, "Trained from heart.csv (200 trees)."

def load_model():
    if MODEL_PATH.exists() and SCALER_PATH.exists():
        with MODEL_PATH.open("rb") as f: rf = pickle.load(f)
        with SCALER_PATH.open("rb") as f: sc = pickle.load(f)
        return rf, sc, "Loaded saved health.pkl and scaler.pkl."
    if DATA_PATH.exists():
        return train_and_save_model()
    return None, None, "Missing model files. Add health.pkl + scaler.pkl, or heart.csv to train on startup."

# ── PDF / RAG ──────────────────────────────────────────────────────────────────
def chunk_text(text):
    words = re.findall(r"\S+", text)
    if not words: return []
    step, chunks = max(1, CHUNK_WORDS - CHUNK_OVERLAP), []
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + CHUNK_WORDS]).strip()
        if chunk: chunks.append(chunk)
        if start + CHUNK_WORDS >= len(words): break
    return chunks

def tokenize(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in STOP_WORDS]

def load_pdf_documents():
    if PdfReader is None or not PDF_DIR.exists(): return []
    chunks = []
    for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
        try: reader = PdfReader(str(pdf_path))
        except Exception: continue
        for pg_num, page in enumerate(reader.pages, 1):
            text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
            for i, chunk in enumerate(chunk_text(text), 1):
                chunks.append({"text": chunk, "source": f"{pdf_path.name}, p{pg_num} chunk {i}", "tokens": tokenize(chunk)})
    return chunks

def build_semantic_retriever(documents):
    # Disabled: sentence-transformers exceeds Render free-tier memory
    return None, None

def retrieve(query, k=5):
    """Semantic retrieval with keyword fallback."""
    if SEMANTIC_MODEL is not None and SEMANTIC_INDEX is not None and PDF_DOCUMENTS:
        try:
            q = SEMANTIC_MODEL.encode([query], convert_to_numpy=True)
            q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
            _, idxs = SEMANTIC_INDEX.search(q.astype("float32"), k)
            return [PDF_DOCUMENTS[i]["text"] for i in idxs[0] if 0 <= i < len(PDF_DOCUMENTS)]
        except Exception:
            pass

    # Keyword fallback over PDFs + built-ins
    words = set(tokenize(query))
    sources = list(PDF_DOCUMENTS) + [{"text": d, "source": "built-in", "tokens": tokenize(d)} for d in BUILTIN_DOCS]
    ranked = sorted(
        [(len(words & set(d.get("tokens") or tokenize(d["text"]))), d["text"]) for d in sources if words & set(d.get("tokens") or tokenize(d["text"]))],
        reverse=True
    )
    return [text for _, text in ranked[:k]] or BUILTIN_DOCS[:k]

# ── Groq LLM ───────────────────────────────────────────────────────────────────
def load_groq_client():
    if not GROQ_API_KEY or Groq is None: return None
    try: return Groq(api_key=GROQ_API_KEY)
    except Exception: return None

def ask_llm(user_message, history=None, rag_context="", prediction_context=""):
    if GROQ_CLIENT is None:
        return "AI assistant unavailable — set GROQ_API_KEY in Render environment variables."

    enriched = user_message
    if rag_context:
        enriched = f"[Relevant medical context]\n{rag_context}\n\n[User question]\n{user_message}"
    if prediction_context:
        enriched = f"[Current prediction]\n{prediction_context}\n\n" + enriched

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages += history[-20:]   # keep last 20 turns max
    messages.append({"role": "user", "content": enriched})

    response = GROQ_CLIENT.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=messages,
        max_tokens=512,
        temperature=0.5,
    )
    return response.choices[0].message.content.strip()

# ── Input validation ───────────────────────────────────────────────────────────
def parse_and_validate(form_data):
    values, errors = [], []
    for col in FEATURE_NAMES:
        raw = form_data.get(col, "").strip()
        try:
            val = float(raw)
        except ValueError:
            errors.append(f"{col}: must be a number")
            values.append(0.0)
            continue
        lo, hi = FIELD_RANGES.get(col, (-1e9, 1e9))
        if not (lo <= val <= hi):
            errors.append(f"{col}: {val} out of range [{lo}–{hi}]")
        values.append(val)
    return values, errors

def feature_summary(values):
    labels = {
        "age": "Age", "sex": "Sex", "cp": "Chest pain type",
        "trestbps": "Resting BP", "chol": "Cholesterol", "fbs": "Fasting blood sugar>120",
        "restecg": "Rest ECG", "thalach": "Max heart rate", "exang": "Exercise angina",
        "oldpeak": "ST depression", "slope": "ST slope", "ca": "Major vessels", "thal": "Thalassemia",
    }
    return ", ".join(f"{labels[k]}={v}" for k, v in zip(FEATURE_NAMES, values))

# ── HTML Template ──────────────────────────────────────────────────────────────
HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Heart AI Predictor</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
:root{--red:#e24b4a;--red-dk:#a32d2d;--teal:#1d9e75;--amber:#ba7517;--ink:#2e2d2b;--paper:#f4f2ec;--card:#fff;--muted:rgba(46,45,43,.45);--border:rgba(46,45,43,.1);--field:#f7f6f2}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{min-height:100vh;overflow-x:hidden;font-family:'DM Sans',sans-serif;background:var(--paper);color:var(--ink)}
body::before{content:"";position:fixed;inset:0;z-index:0;pointer-events:none;background:radial-gradient(ellipse 80% 60% at 20% 10%,rgba(226,75,74,.07) 0%,transparent 60%),radial-gradient(ellipse 60% 50% at 80% 80%,rgba(29,158,117,.06) 0%,transparent 55%)}
.page{position:relative;z-index:1;width:min(840px,calc(100% - 48px));margin:0 auto;padding:3rem 0 5rem}
header{text-align:center;margin-bottom:3rem;animation:fadeDown .7s ease both}
.heart{width:32px;height:32px;margin-bottom:1.25rem;animation:beat 1.6s ease-in-out infinite}
.eyebrow{margin-bottom:1rem;display:flex;align-items:center;justify-content:center;gap:10px;color:var(--red);font-size:11px;font-weight:600;letter-spacing:.25em;text-transform:uppercase}
.eyebrow::before,.eyebrow::after{content:"";display:block;width:40px;height:1px;background:var(--red);opacity:.5}
h1{margin-bottom:.75rem;color:var(--ink);font-family:'DM Serif Display',serif;font-size:clamp(2.4rem,5vw,3.6rem);font-weight:400;line-height:1.1}
h1 em{color:var(--red);font-style:italic}
.sub{max-width:380px;margin:0 auto;color:var(--muted);font-size:15px;font-weight:300;line-height:1.6}
.panel{width:100%;overflow:hidden;background:var(--card);border:1px solid var(--border);border-radius:24px;box-shadow:0 4px 24px rgba(46,45,43,.07);animation:fadeUp .8s .15s ease both}
.sec-label{padding:1.25rem 2rem .75rem;border-bottom:1px solid var(--border);color:var(--muted);font-size:10px;font-weight:600;letter-spacing:.18em;text-transform:uppercase;display:flex;align-items:center;gap:7px}
.sec-label::before{content:'';width:6px;height:6px;border-radius:50%;background:var(--red);opacity:.7;flex-shrink:0}
form{padding:1.5rem 2rem 2rem}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px;margin-bottom:1.75rem}
.field{display:flex;flex-direction:column;gap:5px}
label{color:var(--muted);font-size:10px;font-weight:600;letter-spacing:.07em;text-transform:uppercase}
label small{font-size:9px;font-weight:300;opacity:.75;text-transform:none}
input{width:100%;min-height:40px;padding:10px 14px;border:1px solid var(--border);border-radius:10px;background:var(--field);color:var(--ink);font:400 14px 'DM Sans',sans-serif;outline:none;transition:border-color .2s,background .2s}
input:focus{border-color:rgba(226,75,74,.5);background:#fff}
.error-banner{margin:0 2rem 1rem;padding:10px 14px;border-radius:10px;background:rgba(226,75,74,.1);color:var(--red-dk);border:1px solid rgba(226,75,74,.25);font-size:13px}
.predict{width:100%;min-height:48px;padding:15px 28px;border:0;border-radius:12px;background:var(--red);color:#fff;font:600 14px 'DM Sans',sans-serif;letter-spacing:.06em;text-transform:uppercase;cursor:pointer;transition:transform .15s,opacity .2s}
.predict:hover{opacity:.88;transform:translateY(-1px)}
.result{padding:1.75rem 2rem 2rem;border-top:1px solid var(--border)}
.result-title{font-family:'DM Serif Display',serif;font-size:1.6rem;font-weight:400;line-height:1.3;margin-bottom:1.25rem}
.risk-meta{display:flex;justify-content:space-between;align-items:baseline;gap:16px;margin-bottom:10px;font-size:11px;font-weight:500;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}
.score{color:var(--ink);font-family:'DM Serif Display',serif;font-size:2rem;line-height:1;letter-spacing:0;text-transform:none}
.track{position:relative;height:8px;overflow:hidden;border-radius:100px;background:rgba(46,45,43,.08)}
.fill{height:100%;width:0%;border-radius:100px;background:linear-gradient(90deg,var(--teal) 0%,var(--amber) 55%,var(--red) 100%);transition:width 1.2s cubic-bezier(.4,0,.2,1)}
.risk-scale{display:flex;justify-content:space-between;margin-top:6px}
.risk-scale span{color:var(--muted);font-size:10px;letter-spacing:.04em}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0}
.chip{padding:5px 14px;border-radius:100px;font-size:12px;font-weight:500;letter-spacing:.04em}
.chip-low{background:rgba(29,158,117,.1);color:#0f6e56;border:1px solid rgba(29,158,117,.25)}
.chip-mid{background:rgba(186,117,23,.1);color:#7a4e0c;border:1px solid rgba(186,117,23,.25)}
.chip-high{background:rgba(226,75,74,.1);color:var(--red-dk);border:1px solid rgba(226,75,74,.25)}
.insight{font-size:13px;line-height:1.65;color:var(--ink);background:var(--field);border-radius:10px;padding:12px 14px;border:1px solid var(--border);white-space:pre-line}
.note{max-width:500px;margin:2.5rem auto 0;text-align:center;color:var(--muted);font-size:12px;font-weight:300;line-height:1.7}
/* Chat */
#chat-toggle{position:fixed;right:28px;bottom:28px;z-index:1000;width:56px;height:56px;border:0;border-radius:50%;display:flex;align-items:center;justify-content:center;background:var(--red);color:#fff;cursor:pointer;box-shadow:0 4px 18px rgba(226,75,74,.35);transition:transform .2s,box-shadow .2s}
#chat-toggle:hover{transform:scale(1.08)}
#chat-window{position:fixed;right:28px;bottom:96px;z-index:999;width:340px;max-height:520px;display:none;flex-direction:column;overflow:hidden;background:#fff;border:1px solid var(--border);border-radius:20px;box-shadow:0 8px 40px rgba(46,45,43,.13)}
#chat-window.open{display:flex;animation:fadeUp .25s ease both}
.chat-header{display:flex;align-items:center;gap:10px;padding:14px 18px;background:var(--red)}
.chat-header-icon{width:30px;height:30px;border-radius:50%;display:flex;align-items:center;justify-content:center;flex-shrink:0;background:rgba(255,255,255,.2)}
.chat-header-text h3{color:#fff;font-size:13px;font-weight:600;line-height:1.2}
.chat-header-text p{color:rgba(255,255,255,.7);font-size:11px;font-weight:300}
#chat-messages{flex:1;overflow-y:auto;padding:14px 14px 8px;display:flex;flex-direction:column;gap:10px;background:#f7f6f2}
.msg{max-width:85%;padding:9px 13px;border-radius:14px;font-size:13px;line-height:1.55;white-space:pre-line}
.msg.bot{align-self:flex-start;background:#fff;color:var(--ink);border:1px solid var(--border);border-bottom-left-radius:3px}
.msg.user{align-self:flex-end;background:var(--red);color:#fff;border-bottom-right-radius:3px}
.msg.typing{align-self:flex-start;background:#fff;color:var(--muted);border:1px solid var(--border);font-style:italic;font-size:12px}
.chat-input-row{display:flex;gap:8px;padding:10px 12px;background:#fff;border-top:1px solid var(--border)}
#chat-input{flex:1;resize:none;padding:9px 12px;border:1px solid var(--border);border-radius:10px;background:var(--field);color:var(--ink);font:400 13px 'DM Sans',sans-serif;outline:none}
#chat-input:focus{border-color:rgba(226,75,74,.45);background:#fff}
#chat-send{width:36px;height:36px;align-self:flex-end;flex-shrink:0;border:0;border-radius:10px;display:flex;align-items:center;justify-content:center;background:var(--red);color:#fff;cursor:pointer;transition:opacity .2s}
#chat-send:hover{opacity:.85}
@keyframes beat{0%,100%{transform:scale(1)}20%{transform:scale(1.18)}34%{transform:scale(1)}}
@keyframes fadeDown{from{opacity:0;transform:translateY(-16px)}to{opacity:1;transform:translateY(0)}}
@keyframes fadeUp{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:translateY(0)}}
@media(max-width:560px){form,.result{padding:1.1rem 1.2rem 1.4rem}.sec-label{padding:1.1rem 1.2rem .65rem}.grid{grid-template-columns:1fr 1fr}#chat-window{width:calc(100vw - 32px);right:16px}}
@media(max-width:380px){.grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<main class="page">
  <header>
    <svg class="heart" viewBox="0 0 32 32"><path d="M16 28S3 20.5 3 11.5C3 7.36 6.13 4 10 4c2.35 0 4.43 1.21 6 3 1.57-1.79 3.65-3 6-3 3.87 0 7 3.36 7 7.5C29 20.5 16 28 16 28Z" fill="#e24b4a"/></svg>
    <div class="eyebrow">AI Clinical Assessment</div>
    <h1>Heart Risk<br><em>Predictor</em></h1>
    <p class="sub">Enter patient clinical data below for an AI-powered cardiovascular risk assessment.</p>
  </header>

  <section class="panel">
    <div class="sec-label">Patient Clinical Parameters</div>
    <form action="/predict" method="POST">
      <div class="form-body" style="padding:1.5rem 2rem 2rem">
        <div class="grid">
          {% for f in fields %}
          <div class="field">
            <label for="{{ f.name }}">{{ f.label }} <small>{{ f.hint }}</small></label>
            <input id="{{ f.name }}" name="{{ f.name }}" type="number"
                   step="{{ f.step }}" min="{{ f.min }}" max="{{ f.max }}"
                   placeholder="{{ f.placeholder }}"
                   value="{{ values.get(f.name, '') }}" required>
          </div>
          {% endfor %}
        </div>
        <button class="predict" type="submit">Run Prediction &rarr;</button>
      </div>
    </form>

    {% if error %}
    <div class="error-banner">{{ error }}</div>
    {% endif %}

    {% if prediction_text %}
    <div class="sec-label">Assessment Result</div>
    <div class="result">
      <p class="result-title">{{ prediction_text }}</p>
      <div class="risk-meta">
        <span>Cardiovascular Risk Score</span>
        <span class="score">{{ risk }}%</span>
      </div>
      <div class="track"><div class="fill" id="riskFill"></div></div>
      <div class="risk-scale"><span>Low</span><span>Moderate</span><span>High</span></div>
      <div class="chips">
        {% if risk < 30 %}
          <span class="chip chip-low">Low Risk</span>
          <span class="chip chip-low">Routine Monitoring</span>
        {% elif risk < 65 %}
          <span class="chip chip-mid">Moderate Risk</span>
          <span class="chip chip-mid">Follow-Up Recommended</span>
        {% else %}
          <span class="chip chip-high">High Risk</span>
          <span class="chip chip-high">Urgent Review Advised</span>
        {% endif %}
      </div>
      {% if insight %}<div class="insight">{{ insight }}</div>{% endif %}
    </div>
    {% endif %}
  </section>

  <p class="note"><strong>Clinical disclaimer.</strong> This tool provides a statistical estimate only and is not a substitute for professional medical diagnosis. Always consult a qualified cardiologist.</p>
</main>

<!-- Chat button -->
<button id="chat-toggle" title="Ask a health question" aria-label="Open chat">
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none"><path d="M21 15c0 .53-.21 1.04-.59 1.41-.37.38-.88.59-1.41.59H7l-4 4V5c0-.53.21-1.04.59-1.41C3.96 3.21 4.47 3 5 3h14c.53 0 1.04.21 1.41.59.38.37.59.88.59 1.41v10z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
</button>

<section id="chat-window" aria-label="Heart health assistant">
  <div class="chat-header">
    <div class="chat-header-icon">
      <svg width="16" height="16" viewBox="0 0 32 32" fill="none"><path d="M16 28C16 28 3 20.5 3 11.5C3 7.36 6.13 4 10 4C12.35 4 14.43 5.21 16 7C17.57 5.21 19.65 4 22 4C25.87 4 29 7.36 29 11.5C29 20.5 16 28 16 28Z" fill="white"/></svg>
    </div>
    <div class="chat-header-text">
      <h3>Heart Health Assistant</h3>
      <p>Ask about your results or heart health</p>
    </div>
  </div>
  <div id="chat-messages">
    <div class="msg bot">Hi! I can answer questions about heart health, risk factors, and your results. What would you like to know?</div>
  </div>
  <div class="chat-input-row">
    <textarea id="chat-input" rows="1" placeholder="Ask about cholesterol, BP, your results…"></textarea>
    <button id="chat-send" aria-label="Send">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none"><path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
    </button>
  </div>
</section>

<script>
  // Animate risk bar
  var risk = parseFloat("{{ risk if risk else 0 }}") || 0;
  var bar = document.getElementById("riskFill");
  if (bar) setTimeout(() => bar.style.width = risk + "%", 300);

  // Chat toggle
  document.getElementById("chat-toggle").addEventListener("click", () =>
    document.getElementById("chat-window").classList.toggle("open"));

  // Auto-resize textarea
  var inp = document.getElementById("chat-input");
  inp.addEventListener("input", () => {
    inp.style.height = "auto";
    inp.style.height = Math.min(inp.scrollHeight, 90) + "px";
  });

  // Conversation history (kept in browser, sent to /chat each time)
  var chatHistory = [];

  async function sendMessage() {
    var text = inp.value.trim();
    if (!text) return;
    var msgs = document.getElementById("chat-messages");

    var u = document.createElement("div");
    u.className = "msg user"; u.textContent = text;
    msgs.appendChild(u);

    var t = document.createElement("div");
    t.className = "msg typing"; t.id = "typing"; t.textContent = "Thinking…";
    msgs.appendChild(t);
    msgs.scrollTop = msgs.scrollHeight;

    inp.value = ""; inp.style.height = "auto";

    try {
      var res = await fetch("/chat", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ message: text, history: chatHistory })
      });
      var data = await res.json();
      document.getElementById("typing")?.remove();

      var b = document.createElement("div");
      b.className = "msg bot";
      b.textContent = data.reply || "Sorry, I could not find an answer.";
      msgs.appendChild(b);
      msgs.scrollTop = msgs.scrollHeight;

      // Update history
      chatHistory.push({role: "user", content: text});
      chatHistory.push({role: "assistant", content: data.reply});
      if (chatHistory.length > 20) chatHistory = chatHistory.slice(-20);
    } catch {
      document.getElementById("typing")?.remove();
      var e = document.createElement("div");
      e.className = "msg bot"; e.textContent = "Connection error — please try again.";
      msgs.appendChild(e);
      msgs.scrollTop = msgs.scrollHeight;
    }
  }

  document.getElementById("chat-send").addEventListener("click", sendMessage);
  inp.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });
</script>
</body>
</html>
"""

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__)

model, scaler, model_mode = load_model()
PDF_DOCUMENTS = load_pdf_documents()
SEMANTIC_MODEL, SEMANTIC_INDEX = build_semantic_retriever(PDF_DOCUMENTS)
GROQ_CLIENT = load_groq_client()


@app.route("/")
def home():
    return render_template("index.html", prediction_text=None, risk=0,
                           insight=None, error=None, model_auc=MODEL_AUC)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return redirect(url_for("home"))

    form = request.form.to_dict()
    user_input, errors = parse_and_validate(form)

    if errors:
        return render_template("index.html", prediction_text=None, risk=0,
                               insight=None,
                               error="Input error: " + " | ".join(errors),
                               model_auc=MODEL_AUC), 400

    if model is None or scaler is None:
        return render_template("index.html", prediction_text=None, risk=0,
                               insight=None, error=model_mode,
                               model_auc=MODEL_AUC), 503

    scaled = scaler.transform([user_input])
    pred   = int(model.predict(scaled)[0])
    prob   = round(float(model.predict_proba(scaled)[0][1]) * 100, 1)
    result = "Heart Disease Detected" if pred == 1 else "No Heart Disease Detected"

    # AI insight (matches notebook behaviour)
    feat_str = feature_summary(user_input)
    pred_ctx = f"Prediction: {result} | Risk score: {prob}% | Patient data: {feat_str}"
    rag_ctx  = "\n".join(retrieve("heart disease risk factors causes", k=4))

    try:
        insight = ask_llm(
            user_message=(
                "In 2-3 sentences, explain what the prediction result means for this "
                "patient and the key factors that likely drove it. Be direct and clear."
            ),
            history=[],
            rag_context=rag_ctx,
            prediction_context=pred_ctx,
        )
    except Exception as exc:
        insight = f"(AI insight unavailable: {exc})"

    return render_template("index.html", prediction_text=result, risk=prob,
                           insight=insight, error=None, model_auc=MODEL_AUC)


@app.route("/chat", methods=["POST"])
def chat():
    try:
        payload  = request.get_json(silent=True) or {}
        user_msg = payload.get("message", "").strip()
        history  = payload.get("history", [])   # browser sends full history

        if not user_msg:
            return jsonify(reply="Please ask something.")

        rag_ctx = "\n".join(retrieve(user_msg, k=5))

        try:
            reply = ask_llm(user_message=user_msg, history=history, rag_context=rag_ctx)
        except Exception as exc:
            reply = f"Something went wrong: {exc}"

        return jsonify(reply=reply)
    except Exception:
        return jsonify(reply="Something went wrong — please try again.")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
