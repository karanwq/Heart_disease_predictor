import os
import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, render_template_string, request
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "health.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
DATA_PATH = BASE_DIR / "heart.csv"

FEATURE_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
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
    :root {
      --red-400: #E24B4A;
      --teal-400: #1D9E75;
      --amber-400: #BA7517;
    }
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'DM Sans', sans-serif;
      background: #0c0a09;
      color: #f5f0eb;
      min-height: 100vh;
      overflow-x: hidden;
    }
    body::before {
      content: '';
      position: fixed;
      inset: 0;
      background:
        radial-gradient(ellipse 80% 60% at 20% 10%, rgba(162, 45, 45, 0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 80% 80%, rgba(15, 110, 86, 0.12) 0%, transparent 55%),
        radial-gradient(ellipse 40% 40% at 60% 30%, rgba(186, 117, 23, 0.08) 0%, transparent 50%);
      pointer-events: none;
      z-index: 0;
    }
    .page-wrap {
      position: relative;
      z-index: 1;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 3rem 1.5rem 5rem;
    }
    .header {
      text-align: center;
      margin-bottom: 3rem;
      animation: fadeDown 0.7s ease both;
    }
    .header-eyebrow {
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.25em;
      text-transform: uppercase;
      color: var(--red-400);
      margin-bottom: 1rem;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
    }
    .header-eyebrow::before,
    .header-eyebrow::after {
      content: '';
      display: block;
      width: 40px;
      height: 1px;
      background: var(--red-400);
      opacity: 0.5;
    }
    .header h1 {
      font-family: 'DM Serif Display', serif;
      font-size: clamp(2.4rem, 5vw, 3.6rem);
      font-weight: 400;
      line-height: 1.1;
      margin-bottom: 0.75rem;
    }
    .header h1 em { font-style: italic; color: var(--red-400); }
    .header-sub {
      font-size: 15px;
      color: rgba(245, 240, 235, 0.45);
      font-weight: 300;
      max-width: 380px;
      margin: 0 auto;
      line-height: 1.6;
    }
    .heart-icon {
      display: inline-block;
      width: 32px;
      height: 32px;
      margin-bottom: 1.25rem;
      animation: heartbeat 1.6s ease-in-out infinite;
    }
    .ecg-line {
      width: 100%;
      max-width: 340px;
      margin: 0.75rem auto 0;
      opacity: 0.18;
    }
    .card {
      width: 100%;
      max-width: 820px;
      background: rgba(245, 240, 235, 0.04);
      border: 0.5px solid rgba(245, 240, 235, 0.1);
      border-radius: 24px;
      overflow: hidden;
      animation: fadeUp 0.8s 0.15s ease both;
    }
    .section-label {
      font-size: 10px;
      font-weight: 600;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: rgba(245, 240, 235, 0.35);
      padding: 1.5rem 2rem 0.75rem;
      border-bottom: 0.5px solid rgba(245, 240, 235, 0.06);
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .section-label .dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: var(--red-400);
      opacity: 0.7;
    }
    .form-body { padding: 1.5rem 2rem 2rem; }
    .fields-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 12px;
      margin-bottom: 1.75rem;
    }
    .field-wrap {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .field-label {
      font-size: 11px;
      font-weight: 500;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: rgba(245, 240, 235, 0.4);
    }
    .field-hint {
      font-size: 10px;
      color: rgba(245, 240, 235, 0.25);
      font-weight: 300;
    }
    input {
      width: 100%;
      padding: 10px 14px;
      border-radius: 10px;
      border: 0.5px solid rgba(245, 240, 235, 0.12);
      background: rgba(245, 240, 235, 0.05);
      color: #f5f0eb;
      font-family: 'DM Sans', sans-serif;
      font-size: 14px;
      outline: none;
      transition: border-color 0.2s, background 0.2s;
      -webkit-appearance: none;
    }
    input::placeholder { color: rgba(245, 240, 235, 0.2); }
    input:focus {
      border-color: rgba(226, 75, 74, 0.5);
      background: rgba(245, 240, 235, 0.08);
    }
    .btn-predict {
      width: 100%;
      padding: 15px 28px;
      border-radius: 12px;
      border: none;
      background: var(--red-400);
      color: #fff;
      font-family: 'DM Sans', sans-serif;
      font-size: 14px;
      font-weight: 600;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      cursor: pointer;
    }
    .divider {
      height: 0.5px;
      background: rgba(245, 240, 235, 0.07);
      margin: 0 2rem;
    }
    .result-section { padding: 1.75rem 2rem 2rem; }
    .result-headline {
      font-family: 'DM Serif Display', serif;
      font-size: 1.6rem;
      font-weight: 400;
      margin-bottom: 1.25rem;
      line-height: 1.3;
    }
    .risk-meter-wrap { margin-bottom: 1.5rem; }
    .risk-meta {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      margin-bottom: 10px;
    }
    .risk-meta-label {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: rgba(245, 240, 235, 0.35);
      font-weight: 500;
    }
    .risk-percentage {
      font-family: 'DM Serif Display', serif;
      font-size: 2rem;
      line-height: 1;
    }
    .risk-percentage span {
      font-family: 'DM Sans', sans-serif;
      font-size: 13px;
      color: rgba(245, 240, 235, 0.4);
      font-weight: 300;
      margin-left: 3px;
    }
    .risk-track {
      height: 8px;
      background: rgba(245, 240, 235, 0.08);
      border-radius: 100px;
      overflow: hidden;
    }
    .risk-fill {
      height: 100%;
      width: 0;
      border-radius: 100px;
      background: linear-gradient(90deg, var(--teal-400) 0%, var(--amber-400) 55%, var(--red-400) 100%);
      transition: width 1.2s cubic-bezier(0.4, 0, 0.2, 1);
      position: relative;
    }
    .risk-fill::after {
      content: '';
      position: absolute;
      right: 0;
      top: 50%;
      transform: translateY(-50%);
      width: 14px;
      height: 14px;
      border-radius: 50%;
      background: #fff;
      box-shadow: 0 0 0 3px rgba(255,255,255,0.2);
    }
    .risk-scale {
      display: flex;
      justify-content: space-between;
      margin-top: 6px;
    }
    .risk-scale-item {
      font-size: 10px;
      color: rgba(245, 240, 235, 0.25);
      letter-spacing: 0.04em;
    }
    .status-chips {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    .chip {
      padding: 5px 14px;
      border-radius: 100px;
      font-size: 12px;
      font-weight: 500;
      letter-spacing: 0.04em;
    }
    .chip-low {
      background: rgba(29, 158, 117, 0.15);
      color: #5DCAA5;
      border: 0.5px solid rgba(29, 158, 117, 0.3);
    }
    .chip-medium {
      background: rgba(186, 117, 23, 0.15);
      color: #FAC775;
      border: 0.5px solid rgba(186, 117, 23, 0.3);
    }
    .chip-high {
      background: rgba(226, 75, 74, 0.15);
      color: #F09595;
      border: 0.5px solid rgba(226, 75, 74, 0.3);
    }
    .error-message {
      margin-top: 1rem;
      padding: 12px 14px;
      border-radius: 12px;
      background: rgba(226, 75, 74, 0.12);
      border: 0.5px solid rgba(226, 75, 74, 0.3);
      color: #f5c2c1;
      font-size: 14px;
    }
    .footer-note {
      margin-top: 2.5rem;
      text-align: center;
      font-size: 12px;
      color: rgba(245, 240, 235, 0.2);
      line-height: 1.7;
      max-width: 500px;
    }
    .footer-note strong {
      color: rgba(245, 240, 235, 0.35);
      font-weight: 500;
    }
    @keyframes fadeDown {
      from { opacity: 0; transform: translateY(-18px); }
      to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeUp {
      from { opacity: 0; transform: translateY(22px); }
      to { opacity: 1; transform: translateY(0); }
    }
    @keyframes heartbeat {
      0%, 100% { transform: scale(1); }
      14% { transform: scale(1.2); }
      28% { transform: scale(1); }
      42% { transform: scale(1.12); }
      70% { transform: scale(1); }
    }
    @media (max-width: 580px) {
      .form-body { padding: 1.25rem 1.25rem 1.5rem; }
      .section-label { padding: 1.25rem 1.25rem 0.75rem; }
      .divider { margin: 0 1.25rem; }
      .result-section { padding: 1.5rem 1.25rem; }
      .fields-grid { grid-template-columns: 1fr 1fr; gap: 10px; }
    }
    @media (max-width: 400px) {
      .fields-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="page-wrap">
    <header class="header">
      <svg class="heart-icon" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M16 28C16 28 3 20.5 3 11.5C3 7.36 6.13 4 10 4C12.35 4 14.43 5.21 16 7C17.57 5.21 19.65 4 22 4C25.87 4 29 7.36 29 11.5C29 20.5 16 28 16 28Z" fill="#E24B4A"/>
      </svg>
      <div class="header-eyebrow">AI Clinical Assessment</div>
      <h1>Heart Risk<br><em>Predictor</em></h1>
      <p class="header-sub">Enter patient clinical data below for an AI-powered cardiovascular risk assessment.</p>
      <svg class="ecg-line" viewBox="0 0 340 28" xmlns="http://www.w3.org/2000/svg">
        <polyline points="0,14 40,14 52,14 60,2 68,26 74,6 82,22 90,14 130,14 170,14 210,14 250,14 290,14 340,14" fill="none" stroke="#E24B4A" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
    </header>

    <div class="card">
      <div class="section-label">
        <span class="dot"></span>
        Patient Clinical Parameters
      </div>

      <form action="/predict" method="POST">
        <div class="form-body">
          <div class="fields-grid">
            <div class="field-wrap">
              <label class="field-label" for="age">Age <span class="field-hint">years</span></label>
              <input type="number" id="age" name="age" placeholder="e.g. 54" min="1" max="120" required>
            </div>
            <div class="field-wrap">
              <label class="field-label" for="sex">Sex <span class="field-hint">0=F, 1=M</span></label>
              <input type="number" id="sex" name="sex" placeholder="0 or 1" min="0" max="1" required>
            </div>
            <div class="field-wrap">
              <label class="field-label" for="cp">Chest Pain <span class="field-hint">type 0-3</span></label>
              <input type="number" id="cp" name="cp" placeholder="0-3" min="0" max="3" required>
            </div>
            <div class="field-wrap">
              <label class="field-label" for="trestbps">Resting BP <span class="field-hint">mm Hg</span></label>
              <input type="number" id="trestbps" name="trestbps" placeholder="e.g. 120" required>
            </div>
            <div class="field-wrap">
              <label class="field-label" for="chol">Cholesterol <span class="field-hint">mg/dl</span></label>
              <input type="number" id="chol" name="chol" placeholder="e.g. 220" required>
            </div>
            <div class="field-wrap">
              <label class="field-label" for="fbs">Fasting Blood Sugar <span class="field-hint">&gt;120 mg/dl</span></label>
              <input type="number" id="fbs" name="fbs" placeholder="0 or 1" min="0" max="1" required>
            </div>
            <div class="field-wrap">
              <label class="field-label" for="restecg">Rest ECG <span class="field-hint">0-2</span></label>
              <input type="number" id="restecg" name="restecg" placeholder="0-2" min="0" max="2" required>
            </div>
            <div class="field-wrap">
              <label class="field-label" for="thalach">Max Heart Rate <span class="field-hint">bpm</span></label>
              <input type="number" id="thalach" name="thalach" placeholder="e.g. 150" required>
            </div>
            <div class="field-wrap">
              <label class="field-label" for="exang">Exercise Angina <span class="field-hint">0=No, 1=Yes</span></label>
              <input type="number" id="exang" name="exang" placeholder="0 or 1" min="0" max="1" required>
            </div>
            <div class="field-wrap">
              <label class="field-label" for="oldpeak">ST Depression <span class="field-hint">Oldpeak</span></label>
              <input type="number" id="oldpeak" name="oldpeak" placeholder="e.g. 1.2" step="0.1" required>
            </div>
            <div class="field-wrap">
              <label class="field-label" for="slope">Slope <span class="field-hint">0-2</span></label>
              <input type="number" id="slope" name="slope" placeholder="0-2" min="0" max="2" required>
            </div>
            <div class="field-wrap">
              <label class="field-label" for="ca">Major Vessels <span class="field-hint">0-4</span></label>
              <input type="number" id="ca" name="ca" placeholder="0-4" min="0" max="4" required>
            </div>
            <div class="field-wrap">
              <label class="field-label" for="thal">Thalassemia <span class="field-hint">1-3</span></label>
              <input type="number" id="thal" name="thal" placeholder="1-3" min="1" max="3" required>
            </div>
          </div>

          <button type="submit" class="btn-predict">Run Prediction</button>

          {% if error %}
          <div class="error-message">{{ error }}</div>
          {% endif %}
        </div>
      </form>

      {% if prediction_text %}
      <div class="divider"></div>

      <div class="section-label">
        <span class="dot"></span>
        Assessment Result
      </div>

      <div class="result-section">
        <p class="result-headline">{{ prediction_text }}</p>
        <div class="risk-meter-wrap">
          <div class="risk-meta">
            <span class="risk-meta-label">Cardiovascular Risk Score</span>
            <div class="risk-percentage">{{ risk }}<span>%</span></div>
          </div>
          <div class="risk-track">
            <div class="risk-fill" id="riskFill"></div>
          </div>
          <div class="risk-scale">
            <span class="risk-scale-item">Low</span>
            <span class="risk-scale-item">Moderate</span>
            <span class="risk-scale-item">High</span>
          </div>
        </div>

        <div class="status-chips">
          {% if risk < 30 %}
          <span class="chip chip-low">Low Risk</span>
          <span class="chip chip-low">Routine Monitoring</span>
          {% elif risk < 65 %}
          <span class="chip chip-medium">Moderate Risk</span>
          <span class="chip chip-medium">Follow-Up Recommended</span>
          {% else %}
          <span class="chip chip-high">High Risk</span>
          <span class="chip chip-high">Urgent Review Advised</span>
          {% endif %}
        </div>
      </div>
      {% endif %}
    </div>

    <p class="footer-note">
      <strong>Clinical use disclaimer.</strong> This tool provides a statistical estimate only and is not a substitute for professional medical diagnosis.
    </p>
  </div>

  <script>
    var risk = parseFloat("{{ risk if risk is not none else 0 }}") || 0;
    var bar = document.getElementById("riskFill");
    if (bar) {
      setTimeout(function() { bar.style.width = risk + "%"; }, 300);
    }
  </script>
</body>
</html>
"""


def train_and_save_model():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Missing model artifacts and heart.csv. Commit health.pkl and scaler.pkl, or add heart.csv."
        )

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_NAMES]
    y = df["target"]

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(model, model_file)

    with SCALER_PATH.open("wb") as scaler_file:
        pickle.dump(scaler, scaler_file)

    return model, scaler


def load_artifacts():
    if MODEL_PATH.exists() and SCALER_PATH.exists():
        with MODEL_PATH.open("rb") as model_file:
            model = pickle.load(model_file)
        with SCALER_PATH.open("rb") as scaler_file:
            scaler = pickle.load(scaler_file)
        return model, scaler

    return train_and_save_model()


app = Flask(__name__)
model, scaler = load_artifacts()


@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE, prediction_text=None, risk=None, error=None)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        user_input = [float(request.form[name]) for name in FEATURE_NAMES]
        user_scaled = scaler.transform([user_input])
        prediction = int(model.predict(user_scaled)[0])
        probability = float(model.predict_proba(user_scaled)[0][1])
        result = "Heart Disease" if prediction == 1 else "No Heart Disease"

        return render_template_string(
            HTML_TEMPLATE,
            prediction_text=result,
            risk=round(probability * 100, 2),
            error=None,
        )
    except KeyError as exc:
        return render_template_string(
            HTML_TEMPLATE,
            prediction_text=None,
            risk=None,
            error=f"Missing field: {exc.args[0]}",
        ), 400
    except ValueError:
        return render_template_string(
            HTML_TEMPLATE,
            prediction_text=None,
            risk=None,
            error="Please enter valid numeric values for all fields.",
        ), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
