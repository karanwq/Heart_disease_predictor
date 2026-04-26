import os
import pickle
import re
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template_string, request
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "health.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
DATA_PATH = BASE_DIR / "heart.csv"

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
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100vh; font-family: 'DM Sans', sans-serif; background: var(--paper); color: var(--ink); }
    body::before { content: ""; position: fixed; inset: 0; pointer-events: none; background: radial-gradient(ellipse at 18% 8%, rgba(226,75,74,.08), transparent 55%), radial-gradient(ellipse at 88% 80%, rgba(29,158,117,.08), transparent 50%); }
    .page { position: relative; width: min(920px, calc(100% - 32px)); margin: 0 auto; padding: 42px 0 84px; }
    header { text-align: center; margin-bottom: 28px; }
    .heart { width: 34px; height: 34px; animation: beat 1.7s ease-in-out infinite; }
    .eyebrow { margin-top: 10px; color: var(--red); font-size: 11px; font-weight: 700; letter-spacing: .22em; text-transform: uppercase; }
    h1 { margin: 10px 0 8px; font-family: 'DM Serif Display', serif; font-size: clamp(2.4rem, 7vw, 4rem); font-weight: 400; line-height: .95; }
    h1 em { color: var(--red); }
    .sub { margin: 0 auto; max-width: 520px; color: rgba(46,45,43,.62); line-height: 1.6; }
    .panel { background: #fff; border: 1px solid rgba(46,45,43,.1); border-radius: 8px; box-shadow: 0 14px 45px rgba(46,45,43,.08); overflow: hidden; }
    .label { padding: 18px 24px 10px; border-bottom: 1px solid rgba(46,45,43,.07); color: rgba(46,45,43,.48); font-size: 10px; font-weight: 700; letter-spacing: .18em; text-transform: uppercase; }
    form { padding: 22px 24px 24px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }
    label { display: block; margin-bottom: 6px; color: rgba(46,45,43,.55); font-size: 11px; font-weight: 700; letter-spacing: .05em; text-transform: uppercase; }
    label span { color: rgba(46,45,43,.35); font-weight: 500; text-transform: none; }
    input { width: 100%; min-height: 42px; padding: 9px 12px; border: 1px solid rgba(46,45,43,.16); border-radius: 8px; background: #f8f7f3; color: var(--ink); font: inherit; }
    input:focus { outline: none; border-color: rgba(226,75,74,.58); background: #fff; }
    .predict { width: 100%; margin-top: 18px; min-height: 48px; border: 0; border-radius: 8px; background: var(--red); color: #fff; font-weight: 700; letter-spacing: .05em; text-transform: uppercase; cursor: pointer; }
    .result { padding: 22px 24px 26px; border-top: 1px solid rgba(46,45,43,.08); }
    .headline { margin: 0 0 16px; font-family: 'DM Serif Display', serif; font-size: 1.7rem; }
    .meta { display: flex; justify-content: space-between; align-items: baseline; gap: 16px; margin-bottom: 8px; }
    .score { font-family: 'DM Serif Display', serif; font-size: 2.1rem; }
    .track { height: 9px; overflow: hidden; border-radius: 999px; background: rgba(46,45,43,.09); }
    .fill { height: 100%; width: {{ risk if risk else 0 }}%; background: linear-gradient(90deg, var(--teal), var(--amber), var(--red)); }
    .chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
    .chip { padding: 6px 12px; border-radius: 999px; font-size: 12px; background: rgba(226,75,74,.1); color: #9b2d2d; }
    .mode, .error { margin-top: 14px; padding: 11px 13px; border-radius: 8px; font-size: 13px; line-height: 1.45; }
    .mode { background: rgba(186,117,23,.1); color: #7d520f; }
    .error { background: rgba(226,75,74,.12); color: #9b2d2d; }
    .note { max-width: 560px; margin: 22px auto 0; text-align: center; color: rgba(46,45,43,.48); font-size: 12px; line-height: 1.6; }
    #chat-toggle { position: fixed; right: 24px; bottom: 24px; width: 56px; height: 56px; border: 0; border-radius: 50%; background: var(--red); color: #fff; font-size: 24px; cursor: pointer; box-shadow: 0 12px 32px rgba(226,75,74,.34); }
    #chat { position: fixed; right: 24px; bottom: 92px; width: min(360px, calc(100% - 32px)); max-height: 520px; display: none; flex-direction: column; overflow: hidden; background: #fff; border: 1px solid rgba(46,45,43,.12); border-radius: 8px; box-shadow: 0 20px 60px rgba(46,45,43,.18); }
    #chat.open { display: flex; }
    .chat-head { padding: 14px 16px; background: var(--red); color: #fff; font-weight: 700; }
    #messages { min-height: 220px; max-height: 360px; overflow-y: auto; padding: 14px; background: #f8f7f3; }
    .msg { width: fit-content; max-width: 86%; margin: 0 0 10px; padding: 9px 12px; border-radius: 8px; font-size: 13px; line-height: 1.45; white-space: pre-line; }
    .bot { background: #fff; border: 1px solid rgba(46,45,43,.1); }
    .user { margin-left: auto; background: var(--red); color: #fff; }
    .chat-row { display: flex; gap: 8px; padding: 10px; border-top: 1px solid rgba(46,45,43,.08); }
    #chat-input { flex: 1; }
    #send { width: 42px; border: 0; border-radius: 8px; background: var(--red); color: #fff; cursor: pointer; }
    @keyframes beat { 0%, 100% { transform: scale(1); } 20% { transform: scale(1.18); } 34% { transform: scale(1); } }
  </style>
</head>
<body>
  <main class="page">
    <header>
      <svg class="heart" viewBox="0 0 32 32" aria-hidden="true"><path d="M16 28S3 20.5 3 11.5C3 7.36 6.13 4 10 4c2.35 0 4.43 1.21 6 3 1.57-1.79 3.65-3 6-3 3.87 0 7 3.36 7 7.5C29 20.5 16 28 16 28Z" fill="#e24b4a"/></svg>
      <div class="eyebrow">AI Clinical Assessment</div>
      <h1>Heart Risk<br><em>Predictor</em></h1>
      <p class="sub">Deployed Flask version of the health notebook with a prediction form and heart-health chatbot.</p>
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
        <button class="predict" type="submit">Run Prediction</button>
        {% if mode_message %}<div class="mode">{{ mode_message }}</div>{% endif %}
        {% if error %}<div class="error">{{ error }}</div>{% endif %}
      </form>

      {% if prediction_text %}
      <div class="result">
        <p class="headline">{{ prediction_text }}</p>
        <div class="meta"><span>Cardiovascular Risk Score</span><span class="score">{{ risk }}%</span></div>
        <div class="track"><div class="fill"></div></div>
        <div class="chips">
          {% if risk < 30 %}
          <span class="chip">Low Risk</span><span class="chip">Routine Monitoring</span>
          {% elif risk < 65 %}
          <span class="chip">Moderate Risk</span><span class="chip">Follow-Up Recommended</span>
          {% else %}
          <span class="chip">High Risk</span><span class="chip">Medical Review Advised</span>
          {% endif %}
        </div>
      </div>
      {% endif %}
    </section>

    <p class="note">This tool provides a statistical estimate only and is not a substitute for professional medical diagnosis.</p>
  </main>

  <button id="chat-toggle" aria-label="Open chat">?</button>
  <section id="chat" aria-label="Heart health assistant">
    <div class="chat-head">Heart Health Assistant</div>
    <div id="messages"><div class="msg bot">Hi! Ask about blood pressure, cholesterol, exercise, or heart-disease risk factors.</div></div>
    <div class="chat-row">
      <input id="chat-input" type="text" placeholder="Ask a heart health question">
      <button id="send" type="button">Go</button>
    </div>
  </section>

  <script>
    const chat = document.getElementById("chat");
    const messages = document.getElementById("messages");
    const input = document.getElementById("chat-input");
    document.getElementById("chat-toggle").addEventListener("click", () => chat.classList.toggle("open"));
    async function sendMessage() {
      const text = input.value.trim();
      if (!text) return;
      messages.insertAdjacentHTML("beforeend", `<div class="msg user"></div>`);
      messages.lastElementChild.textContent = text;
      input.value = "";
      const pending = document.createElement("div");
      pending.className = "msg bot";
      pending.textContent = "Thinking...";
      messages.appendChild(pending);
      messages.scrollTop = messages.scrollHeight;
      try {
        const res = await fetch("/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message: text }) });
        const data = await res.json();
        pending.textContent = data.reply || "I could not find an answer.";
      } catch {
        pending.textContent = "Connection error. Please try again.";
      }
      messages.scrollTop = messages.scrollHeight;
    }
    document.getElementById("send").addEventListener("click", sendMessage);
    input.addEventListener("keydown", (event) => { if (event.key === "Enter") sendMessage(); });
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

    return None, None, "No health.pkl/scaler.pkl or heart.csv found, so predictions use fallback demo scoring."


def fallback_probability(values):
    data = dict(zip(FEATURE_NAMES, values))
    score = 8
    score += max(0, data["age"] - 45) * 0.55
    score += 8 if data["sex"] == 1 else 3
    score += data["cp"] * 7
    score += max(0, data["trestbps"] - 120) * 0.18
    score += max(0, data["chol"] - 200) * 0.08
    score += data["fbs"] * 5
    score += data["restecg"] * 4
    score += max(0, 150 - data["thalach"]) * 0.16
    score += data["exang"] * 11
    score += data["oldpeak"] * 5
    score += data["slope"] * 3
    score += data["ca"] * 8
    score += max(0, data["thal"] - 1) * 7
    return min(max(score / 100, 0.03), 0.97)


def predict_probability(values):
    if model is None or scaler is None:
        probability = fallback_probability(values)
        return int(probability >= 0.5), probability

    scaled = scaler.transform([values])
    prediction = int(model.predict(scaled)[0])
    probability = float(model.predict_proba(scaled)[0][1])
    return prediction, probability


def retrieve_documents(query, limit=3):
    words = set(re.findall(r"[a-z0-9]+", query.lower()))
    ranked = []
    for document in DOCUMENTS:
        doc_words = set(re.findall(r"[a-z0-9]+", document.lower()))
        ranked.append((len(words & doc_words), document))
    ranked.sort(reverse=True)
    matches = [document for score, document in ranked if score > 0]
    return (matches or DOCUMENTS[:limit])[:limit]


app = Flask(__name__)
model, scaler, model_mode = load_model()


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


@app.route("/predict", methods=["POST"])
def predict():
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

    prediction, probability = predict_probability(user_input)
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
    payload = request.get_json(silent=True) or {}
    user_message = payload.get("message", "").strip()
    if not user_message:
        return jsonify({"reply": "Please ask a heart-health question."})

    context = retrieve_documents(user_message)
    bullets = "\n".join(f"- {item}" for item in context)
    reply = (
        f"{bullets}\n"
        "- This is general education only. For symptoms, diagnosis, or treatment, talk with a qualified clinician."
    )
    return jsonify({"reply": reply})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
