import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, redirect, render_template_string, request, url_for
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError:
    faiss = None
    SentenceTransformer = None

try:
    from transformers import pipeline
except ImportError:
    pipeline = None


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "health.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
DATA_PATH = BASE_DIR / "heart.csv"
PDF_DIR = BASE_DIR / "rag_pdfs"
CHUNK_WORDS = 120
CHUNK_OVERLAP = 30
SEMANTIC_MODEL_NAME = os.environ.get("SEMANTIC_MODEL_NAME", "all-MiniLM-L6-v2")
GENERATOR_MODEL_NAME = os.environ.get("GENERATOR_MODEL_NAME", "google/flan-t5-large")
ENABLE_GENERATOR = os.environ.get("ENABLE_GENERATOR", "0") == "1"
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "could", "did", "do", "does", "doing",
    "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more",
    "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same",
    "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs",
    "them", "themselves", "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when",
    "where", "which", "while", "who", "whom", "why", "will", "with", "you", "your",
    "yours", "yourself", "yourselves",
}

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

DOCUMENTS = [
    "High blood pressure can occur due to stress, high salt intake, lack of exercise, or obesity.",
    "Cholesterol buildup in arteries reduces blood flow and increases heart disease risk.",
    "Smoking damages blood vessels and increases blood pressure.",
    "Regular exercise helps lower blood pressure and improves heart health.",
    "Obesity increases strain on the heart and raises blood pressure.",
    "Excess salt intake causes water retention which increases blood pressure.",
    "Diabetes damages blood vessels and contributes to heart disease.",
    "Chest pain, exercise angina, and abnormal ECG findings can be warning signs that need medical review.",
]
PDF_DOCUMENTS = []

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Heart AI Predictor</title>
  <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
  <style>
    :root { --red: #e24b4a; --teal: #1d9e75; --amber: #ba7517; --ink: #2e2d2b; --paper: #f4f2ec; }
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { min-height: 100vh; overflow-x: hidden; font-family: 'DM Sans', sans-serif; background: var(--paper); color: var(--ink); }
    body::before { content: ""; position: fixed; inset: 0; z-index: 0; pointer-events: none; background: radial-gradient(ellipse 80% 60% at 20% 10%, rgba(226,75,74,.07) 0%, transparent 60%), radial-gradient(ellipse 60% 50% at 80% 80%, rgba(29,158,117,.06) 0%, transparent 55%), radial-gradient(ellipse 40% 40% at 60% 30%, rgba(186,117,23,.04) 0%, transparent 50%); }
    .page { position: relative; z-index: 1; width: min(820px, calc(100% - 48px)); margin: 0 auto; padding: 3rem 0 5rem; }
    header { text-align: center; margin-bottom: 3rem; animation: fadeDown .7s ease both; }
    .heart { width: 32px; height: 32px; margin-bottom: 1.25rem; animation: beat 1.6s ease-in-out infinite; }
    .eyebrow { margin-bottom: 1rem; display: flex; align-items: center; justify-content: center; gap: 10px; color: var(--red); font-size: 11px; font-weight: 600; letter-spacing: .25em; text-transform: uppercase; }
    .eyebrow::before, .eyebrow::after { content: ""; display: block; width: 40px; height: 1px; background: var(--red); opacity: .5; }
    h1 { margin-bottom: .75rem; color: var(--ink); font-family: 'DM Serif Display', serif; font-size: clamp(2.4rem, 5vw, 3.6rem); font-weight: 400; line-height: 1.1; }
    h1 em { color: var(--red); font-style: italic; }
    .sub { max-width: 380px; margin: 0 auto; color: rgba(46,45,43,.5); font-size: 15px; font-weight: 300; line-height: 1.6; }
    .ecg-line { width: 100%; max-width: 340px; margin: .75rem auto 0; opacity: .18; }
    .panel { width: 100%; overflow: hidden; background: #fff; border: 1px solid rgba(46,45,43,.08); border-radius: 24px; box-shadow: 0 4px 24px rgba(46,45,43,.07); animation: fadeUp .8s .15s ease both; }
    .label { padding: 1.5rem 2rem .75rem; border-bottom: 1px solid rgba(46,45,43,.06); color: rgba(46,45,43,.4); font-size: 10px; font-weight: 600; letter-spacing: .18em; text-transform: uppercase; }
    form { padding: 1.5rem 2rem 2rem; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; margin-bottom: 1.75rem; }
    label { display: flex; flex-direction: column; gap: 6px; margin-bottom: 6px; color: rgba(46,45,43,.5); font-size: 11px; font-weight: 500; letter-spacing: .06em; text-transform: uppercase; }
    label span { color: rgba(46,45,43,.35); font-size: 10px; font-weight: 300; text-transform: none; }
    input { width: 100%; min-height: 40px; padding: 10px 14px; border: 1px solid rgba(46,45,43,.14); border-radius: 10px; background: #f7f6f2; color: var(--ink); font: 400 14px 'DM Sans', sans-serif; outline: none; transition: border-color .2s, background .2s; }
    input::placeholder { color: rgba(46,45,43,.3); }
    input:hover:not(:focus) { border-color: rgba(46,45,43,.25); }
    input:focus { border-color: rgba(226,75,74,.5); background: #fff; }
    .predict { width: 100%; min-height: 48px; padding: 15px 28px; border: 0; border-radius: 12px; background: var(--red); color: #fff; font: 600 14px 'DM Sans', sans-serif; letter-spacing: .06em; text-transform: uppercase; cursor: pointer; transition: transform .15s, opacity .2s; }
    .predict:hover { opacity: .88; transform: translateY(-1px); }
    .result { padding: 1.75rem 2rem 2rem; border-top: 1px solid rgba(46,45,43,.07); }
    .headline { margin-bottom: 1.25rem; color: var(--ink); font-family: 'DM Serif Display', serif; font-size: 1.6rem; font-weight: 400; line-height: 1.3; }
    .meta { display: flex; justify-content: space-between; align-items: baseline; gap: 16px; margin-bottom: 10px; color: rgba(46,45,43,.4); font-size: 11px; font-weight: 500; letter-spacing: .1em; text-transform: uppercase; }
    .score { color: var(--ink); font-family: 'DM Serif Display', serif; font-size: 2rem; line-height: 1; letter-spacing: 0; text-transform: none; }
    .track { position: relative; height: 8px; overflow: hidden; border-radius: 100px; background: rgba(46,45,43,.08); }
    .fill { height: 100%; width: 0%; border-radius: 100px; background: linear-gradient(90deg, var(--teal) 0%, var(--amber) 55%, var(--red) 100%); transition: width 1.2s cubic-bezier(.4,0,.2,1); }
    .risk-scale { display: flex; justify-content: space-between; margin-top: 6px; }
    .risk-scale span { color: rgba(46,45,43,.35); font-size: 10px; letter-spacing: .04em; }
    .chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
    .chip { padding: 5px 14px; border-radius: 100px; font-size: 12px; font-weight: 500; letter-spacing: .04em; }
    .chip-low { background: rgba(29,158,117,.1); color: #0f6e56; border: 1px solid rgba(29,158,117,.25); }
    .chip-medium { background: rgba(186,117,23,.1); color: #8a5a0f; border: 1px solid rgba(186,117,23,.25); }
    .chip-high { background: rgba(226,75,74,.1); color: #a32d2d; border: 1px solid rgba(226,75,74,.25); }
    .mode, .error { margin: 0 2rem 1rem; padding: 11px 13px; border-radius: 10px; font-size: 13px; line-height: 1.45; }
    .mode { background: rgba(186,117,23,.1); color: #8a5a0f; border: 1px solid rgba(186,117,23,.25); }
    .error { background: rgba(226,75,74,.1); color: #a32d2d; border: 1px solid rgba(226,75,74,.25); }
    .note { max-width: 500px; margin: 2.5rem auto 0; text-align: center; color: rgba(46,45,43,.35); font-size: 12px; font-weight: 300; line-height: 1.7; }
    #chat-toggle { position: fixed; right: 28px; bottom: 28px; z-index: 1000; width: 56px; height: 56px; border: 0; border-radius: 50%; display: flex; align-items: center; justify-content: center; background: var(--red); color: #fff; cursor: pointer; box-shadow: 0 4px 18px rgba(226,75,74,.35); transition: transform .2s, box-shadow .2s; }
    #chat-toggle:hover { transform: scale(1.08); box-shadow: 0 6px 24px rgba(226,75,74,.45); }
    #chat-window { position: fixed; right: 28px; bottom: 96px; z-index: 999; width: 340px; max-height: 500px; display: none; flex-direction: column; overflow: hidden; background: #fff; border: 1px solid rgba(46,45,43,.1); border-radius: 20px; box-shadow: 0 8px 40px rgba(46,45,43,.13); animation: fadeUp .25s ease both; }
    #chat-window.open { display: flex; }
    .chat-header { display: flex; align-items: center; gap: 10px; padding: 14px 18px; background: var(--red); }
    .chat-header-icon { width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center; flex-shrink: 0; background: rgba(255,255,255,.2); }
    .chat-header-text h3 { color: #fff; font-size: 13px; font-weight: 600; line-height: 1.2; }
    .chat-header-text p { color: rgba(255,255,255,.7); font-size: 11px; font-weight: 300; }
    #chat-messages { flex: 1; overflow-y: auto; padding: 14px 14px 8px; display: flex; flex-direction: column; gap: 10px; background: #f7f6f2; }
    .msg { max-width: 85%; padding: 9px 13px; border-radius: 14px; font-size: 13px; line-height: 1.5; white-space: pre-line; }
    .msg.bot { align-self: flex-start; background: #fff; color: var(--ink); border: 1px solid rgba(46,45,43,.1); border-bottom-left-radius: 4px; }
    .msg.user { align-self: flex-end; background: var(--red); color: #fff; border-bottom-right-radius: 4px; }
    .msg.typing { align-self: flex-start; background: #fff; color: rgba(46,45,43,.4); border: 1px solid rgba(46,45,43,.1); font-style: italic; font-size: 12px; }
    .chat-input-row { display: flex; gap: 8px; padding: 10px 12px; background: #fff; border-top: 1px solid rgba(46,45,43,.07); }
    #chat-input { flex: 1; resize: none; padding: 9px 12px; border: 1px solid rgba(46,45,43,.14); border-radius: 10px; background: #f7f6f2; color: var(--ink); font: 400 13px 'DM Sans', sans-serif; outline: none; }
    #chat-input:focus { border-color: rgba(226,75,74,.45); background: #fff; }
    #chat-send { width: 36px; height: 36px; align-self: flex-end; flex-shrink: 0; border: 0; border-radius: 10px; display: flex; align-items: center; justify-content: center; background: var(--red); color: #fff; cursor: pointer; transition: opacity .2s; }
    #chat-send:hover { opacity: .85; }
    @keyframes beat { 0%, 100% { transform: scale(1); } 20% { transform: scale(1.18); } 34% { transform: scale(1); } }
  </style>
</head>
<body>
  <main class="page">
    <header>
      <svg class="heart" viewBox="0 0 32 32" aria-hidden="true"><path d="M16 28S3 20.5 3 11.5C3 7.36 6.13 4 10 4c2.35 0 4.43 1.21 6 3 1.57-1.79 3.65-3 6-3 3.87 0 7 3.36 7 7.5C29 20.5 16 28 16 28Z" fill="#e24b4a"/></svg>
      <div class="eyebrow">AI Clinical Assessment</div>
      <h1>Heart Risk<br><em>Predictor</em></h1>
      <p class="sub">Enter patient clinical data below for an AI-powered cardiovascular risk assessment.</p>
      <svg class="ecg-line" viewBox="0 0 340 28" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
        <polyline points="0,14 40,14 52,14 60,2 68,26 74,6 82,22 90,14 130,14 170,14 210,14 250,14 290,14 340,14" fill="none" stroke="#e24b4a" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
    </header>

    <section class="panel">
      <div class="label">Patient Clinical Parameters</div>
      <form action="/predict" method="POST">
        <div class="grid">
          {% for field in fields %}
          <div>
            <label for="{{ field.name }}">{{ field.label }} <span>{{ field.hint }}</span></label>
            <input id="{{ field.name }}" name="{{ field.name }}" type="number" step="{{ field.step }}" min="{{ field.min }}" max="{{ field.max }}" placeholder="{{ field.placeholder }}" value="{{ values.get(field.name, '') }}" required>
          </div>
          {% endfor %}
        </div>
        <button class="predict" type="submit">Run Prediction &rarr;</button>
      </form>
        {% if error %}<div class="error">{{ error }}</div>{% endif %}

      {% if prediction_text %}
      <div class="result">
        <p class="headline">{{ prediction_text }}</p>
        <div class="meta"><span>Cardiovascular Risk Score</span><span class="score">{{ risk }}%</span></div>
        <div class="track"><div class="fill" id="riskFill"></div></div>
        <div class="risk-scale"><span>Low</span><span>Moderate</span><span>High</span></div>
        <div class="chips">
          {% if risk < 30 %}
          <span class="chip chip-low">Low Risk</span><span class="chip chip-low">Routine Monitoring</span>
          {% elif risk < 65 %}
          <span class="chip chip-medium">Moderate Risk</span><span class="chip chip-medium">Follow-Up Recommended</span>
          {% else %}
          <span class="chip chip-high">High Risk</span><span class="chip chip-high">Urgent Review Advised</span>
          {% endif %}
        </div>
      </div>
      {% endif %}
    </section>

    <p class="note">This tool provides a statistical estimate only and is not a substitute for professional medical diagnosis.</p>
  </main>

  <button id="chat-toggle" type="button" title="Ask a health question" aria-label="Open chat">
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" aria-hidden="true" xmlns="http://www.w3.org/2000/svg">
      <path d="M21 15C21 15.5304 20.7893 16.0391 20.4142 16.4142C20.0391 16.7893 19.5304 17 19 17H7L3 21V5C3 4.46957 3.21071 3.96086 3.58579 3.58579C3.96086 3.21071 4.46957 3 5 3H19C19.5304 3 20.0391 3.21071 20.4142 3.58579C20.7893 3.96086 21 4.46957 21 5V15Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
  </button>
  <section id="chat-window" aria-label="Heart health assistant">
    <div class="chat-header">
      <div class="chat-header-icon">
        <svg width="16" height="16" viewBox="0 0 32 32" fill="none" aria-hidden="true">
          <path d="M16 28C16 28 3 20.5 3 11.5C3 7.36 6.13 4 10 4C12.35 4 14.43 5.21 16 7C17.57 5.21 19.65 4 22 4C25.87 4 29 7.36 29 11.5C29 20.5 16 28 16 28Z" fill="white"/>
        </svg>
      </div>
      <div class="chat-header-text">
        <h3>Heart Health Assistant</h3>
        <p>Powered by RAG - Ask anything</p>
      </div>
    </div>

    <div id="chat-messages">
      <div class="msg bot">Hi! I can answer questions about heart health, risk factors, and your results. What would you like to know?</div>
    </div>

    <div class="chat-input-row">
      <textarea id="chat-input" rows="1" placeholder="Ask about cholesterol, BP, risk factors..."></textarea>
      <button id="chat-send" type="button" aria-label="Send">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path d="M22 2L11 13" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M22 2L15 22L11 13L2 9L22 2Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </button>
    </div>
  </section>

  <script>
    var risk = parseFloat("{{ risk if risk else 0 }}") || 0;
    var riskBar = document.getElementById("riskFill");
    if (riskBar) {
      setTimeout(function() { riskBar.style.width = risk + "%"; }, 300);
    }

    var chatWindow = document.getElementById("chat-window");
    var chatInput = document.getElementById("chat-input");
    var messages = document.getElementById("chat-messages");

    document.getElementById("chat-toggle").addEventListener("click", function() {
      chatWindow.classList.toggle("open");
    });

    chatInput.addEventListener("input", function() {
      this.style.height = "auto";
      this.style.height = Math.min(this.scrollHeight, 100) + "px";
    });

    async function sendMessage() {
      var text = chatInput.value.trim();
      if (!text) return;

      var userMsg = document.createElement("div");
      userMsg.className = "msg user";
      userMsg.textContent = text;
      messages.appendChild(userMsg);

      chatInput.value = "";
      chatInput.style.height = "auto";

      var pending = document.createElement("div");
      pending.className = "msg typing";
      pending.id = "typing-indicator";
      pending.textContent = "Thinking...";
      messages.appendChild(pending);
      messages.scrollTop = messages.scrollHeight;

      try {
        var res = await fetch("/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: text })
        });
        var data = await res.json();
        pending.className = "msg bot";
        pending.removeAttribute("id");
        pending.textContent = data.reply || "Sorry, I could not find an answer.";
      } catch {
        pending.className = "msg bot";
        pending.removeAttribute("id");
        pending.textContent = "Connection error. Please try again.";
      }

      messages.scrollTop = messages.scrollHeight;
    }

    document.getElementById("chat-send").addEventListener("click", sendMessage);
    chatInput.addEventListener("keydown", function(event) {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
      }
    });
  </script>
</body>
</html>
"""

FIELDS = [
    {"name": "age", "label": "Age", "hint": "years", "placeholder": "54", "min": "1", "max": "120", "step": "1"},
    {"name": "sex", "label": "Sex", "hint": "0=F, 1=M", "placeholder": "1", "min": "0", "max": "1", "step": "1"},
    {"name": "cp", "label": "Chest Pain", "hint": "0-3", "placeholder": "0", "min": "0", "max": "3", "step": "1"},
    {"name": "trestbps", "label": "Resting BP", "hint": "mm Hg", "placeholder": "120", "min": "1", "max": "260", "step": "1"},
    {"name": "chol", "label": "Cholesterol", "hint": "mg/dl", "placeholder": "220", "min": "1", "max": "700", "step": "1"},
    {"name": "fbs", "label": "Fasting Sugar", "hint": "0/1", "placeholder": "0", "min": "0", "max": "1", "step": "1"},
    {"name": "restecg", "label": "Rest ECG", "hint": "0-2", "placeholder": "1", "min": "0", "max": "2", "step": "1"},
    {"name": "thalach", "label": "Max Heart Rate", "hint": "bpm", "placeholder": "150", "min": "1", "max": "260", "step": "1"},
    {"name": "exang", "label": "Exercise Angina", "hint": "0/1", "placeholder": "0", "min": "0", "max": "1", "step": "1"},
    {"name": "oldpeak", "label": "ST Depression", "hint": "oldpeak", "placeholder": "1.2", "min": "0", "max": "10", "step": "0.1"},
    {"name": "slope", "label": "Slope", "hint": "0-2", "placeholder": "1", "min": "0", "max": "2", "step": "1"},
    {"name": "ca", "label": "Major Vessels", "hint": "0-4", "placeholder": "0", "min": "0", "max": "4", "step": "1"},
    {"name": "thal", "label": "Thalassemia", "hint": "1-3", "placeholder": "2", "min": "1", "max": "3", "step": "1"},
]


def train_and_save_model():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_NAMES]
    y = df["target"]
    stratify = y if y.nunique() > 1 else None
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)

    fitted_scaler = StandardScaler()
    X_train_scaled = fitted_scaler.fit_transform(X_train)
    fitted_model = RandomForestClassifier(n_estimators=100, random_state=42)
    fitted_model.fit(X_train_scaled, y_train)

    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(fitted_model, model_file)
    with SCALER_PATH.open("wb") as scaler_file:
        pickle.dump(fitted_scaler, scaler_file)

    return fitted_model, fitted_scaler, "Trained from heart.csv."


def load_model():
    if MODEL_PATH.exists() and SCALER_PATH.exists():
        with MODEL_PATH.open("rb") as model_file:
            fitted_model = pickle.load(model_file)
        with SCALER_PATH.open("rb") as scaler_file:
            fitted_scaler = pickle.load(scaler_file)
        return fitted_model, fitted_scaler, "Loaded saved health.pkl and scaler.pkl."

    if DATA_PATH.exists():
        return train_and_save_model()

    return (
        None,
        None,
        "Missing model files. Add health.pkl and scaler.pkl, or add heart.csv so the app can train a real model.",
    )


def predict_probability(values):
    if model is None or scaler is None:
        raise RuntimeError(model_mode)

    scaled = scaler.transform([values])
    prediction = int(model.predict(scaled)[0])
    probability = float(model.predict_proba(scaled)[0][1])
    return prediction, probability


def chunk_text(text):
    words = re.findall(r"\S+", text)
    if not words:
        return []

    step = max(1, CHUNK_WORDS - CHUNK_OVERLAP)
    chunks = []
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + CHUNK_WORDS]).strip()
        if chunk:
            chunks.append(chunk)
        if start + CHUNK_WORDS >= len(words):
            break
    return chunks


def tokenize(text):
    return [
        word
        for word in re.findall(r"[a-z0-9]+", text.lower())
        if len(word) > 2 and word not in STOP_WORDS
    ]


def load_pdf_documents():
    if PdfReader is None or not PDF_DIR.exists():
        return []

    chunks = []
    for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
        try:
            reader = PdfReader(str(pdf_path))
        except Exception:
            continue

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            cleaned = re.sub(r"\s+", " ", text).strip()
            for index, chunk in enumerate(chunk_text(cleaned), start=1):
                chunks.append({
                    "text": chunk,
                    "source": f"{pdf_path.name}, page {page_number}, chunk {index}",
                    "tokens": tokenize(chunk),
                })
    return chunks


def build_semantic_retriever(documents):
    if not documents or faiss is None or SentenceTransformer is None:
        return None, None

    try:
        embed_model = SentenceTransformer(SEMANTIC_MODEL_NAME)
        texts = [document["text"] for document in documents]
        embeddings = embed_model.encode(texts, convert_to_numpy=True)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-12)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype("float32"))
        return embed_model, index
    except Exception:
        return None, None


def load_generator():
    if not ENABLE_GENERATOR or pipeline is None:
        return None

    try:
        return pipeline("text2text-generation", model=GENERATOR_MODEL_NAME)
    except Exception:
        return None


def retrieve_documents(query, limit=3, use_built_in=True):
    if SEMANTIC_MODEL is not None and SEMANTIC_INDEX is not None and PDF_DOCUMENTS:
        try:
            query_embedding = SEMANTIC_MODEL.encode([query], convert_to_numpy=True)
            query_embedding = query_embedding / np.maximum(
                np.linalg.norm(query_embedding, axis=1, keepdims=True),
                1e-12,
            )
            _, indexes = SEMANTIC_INDEX.search(query_embedding.astype("float32"), limit)
            return [
                {
                    "text": PDF_DOCUMENTS[index]["text"],
                    "source": PDF_DOCUMENTS[index]["source"],
                }
                for index in indexes[0]
                if 0 <= index < len(PDF_DOCUMENTS)
            ]
        except Exception:
            pass

    words = set(tokenize(query))
    if not words:
        return []

    ranked = []
    sources = list(PDF_DOCUMENTS)
    if use_built_in:
        sources += [
            {"text": document, "source": "built-in", "tokens": tokenize(document)}
            for document in DOCUMENTS
        ]

    for document in sources:
        doc_words = set(document.get("tokens") or tokenize(document["text"]))
        overlap = words & doc_words
        score = len(overlap)
        if score:
            ranked.append((score, document["text"], document["source"]))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [
        {"text": text, "source": source}
        for score, text, source in ranked
    ][:limit]


def generate_pdf_reply(user_message, context):
    context_text = "\n".join(item["text"] for item in context)

    if GENERATOR is not None:
        prompt = f"""
You are a helpful medical assistant.

Context:
{context_text}

Question: {user_message}

Give 4 bullet points explaining causes, risks, and advice.
"""
        try:
            output = GENERATOR(prompt, max_new_tokens=150, temperature=0.7)
            reply = output[0].get("generated_text", "").strip()
            if "Answer:" in reply:
                reply = reply.split("Answer:")[-1].strip()
            if reply:
                return reply
        except Exception:
            pass

    bullets = []
    for item in context[:4]:
        sentence = re.split(r"(?<=[.!?])\s+", item["text"].strip())[0]
        bullets.append(f"- {sentence} ({item['source']})")

    bullets.append("- This is general education only. For symptoms, diagnosis, or treatment, talk with a qualified clinician.")
    return "\n".join(bullets)


app = Flask(__name__)
model, scaler, model_mode = load_model()
PDF_DOCUMENTS = load_pdf_documents()
SEMANTIC_MODEL, SEMANTIC_INDEX = build_semantic_retriever(PDF_DOCUMENTS)
GENERATOR = load_generator()


@app.route("/")
def home():
    return render_template_string(
        HTML_TEMPLATE,
        fields=FIELDS,
        values={},
        prediction_text=None,
        risk=None,
        mode_message=model_mode,
        error=None,
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return redirect(url_for("home"))

    values = request.form.to_dict()
    try:
        user_input = [float(values[name]) for name in FEATURE_NAMES]
    except (KeyError, ValueError):
        return render_template_string(
            HTML_TEMPLATE,
            fields=FIELDS,
            values=values,
            prediction_text=None,
            risk=None,
            mode_message=model_mode,
            error="Please enter valid numeric values for every field.",
        ), 400

    try:
        prediction, probability = predict_probability(user_input)
    except RuntimeError as exc:
        return render_template_string(
            HTML_TEMPLATE,
            fields=FIELDS,
            values=values,
            prediction_text=None,
            risk=None,
            mode_message=None,
            error=str(exc),
        ), 503

    result = "Heart Disease" if prediction == 1 else "No Heart Disease"

    return render_template_string(
        HTML_TEMPLATE,
        fields=FIELDS,
        values=values,
        prediction_text=result,
        risk=round(probability * 100, 2),
        mode_message=model_mode,
        error=None,
    )


@app.route("/chat", methods=["POST"])
def chat():
    try:
        payload = request.get_json(silent=True) or {}
        user_message = payload.get("message", "").strip()
        if not user_message:
            return jsonify({"reply": "Please ask something"})

        user_message_lower = user_message.lower()

        if any(word in user_message_lower for word in ["hello", "hi", "hey"]):
            return jsonify({"reply": "Hey! I can help you with heart health questions."})

        if "thank" in user_message_lower:
            return jsonify({"reply": "You're welcome"})

        if any(word in user_message_lower for word in ["bye", "goodbye"]):
            return jsonify({"reply": "Take care! Stay healthy"})

        if "who are you" in user_message_lower:
            return jsonify({"reply": "I'm your AI health assistant."})

        context = retrieve_documents(
            f"heart disease explanation about {user_message}",
            limit=6,
            use_built_in=not PDF_DOCUMENTS,
        )
        if not context:
            if PDF_DOCUMENTS:
                return jsonify({
                    "reply": (
                        "I could not find that answer in the PDF knowledge source. "
                        "Try asking with keywords from the document, or add a PDF that covers this topic."
                    )
                })

            return jsonify({
                "reply": "No PDF knowledge source is loaded. Add a PDF to the rag_pdfs folder and restart the app."
            })

        return jsonify({"reply": generate_pdf_reply(user_message, context)})
    except Exception:
        return jsonify({"reply": "Something went wrong"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
