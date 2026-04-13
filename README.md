# 🫀 Heart Disease Predictor — AI Clinical Assessment Tool

An AI-powered cardiovascular risk prediction web app built in **Google Colab**, using machine learning and explainability techniques to assess the likelihood of heart disease from patient clinical data.

---

## 📌 Project Overview

This project uses the **UCI Heart Disease dataset** (`heart.csv` from Kaggle) to train a **Random Forest Classifier** that predicts whether a patient has heart disease based on 13 clinical parameters. The model is served through a **Flask web interface** and deployed publicly via **ngrok**, all within Google Colab.

---

## 🚀 Features

- ✅ Trained Random Forest Classifier with 100 estimators
- ✅ StandardScaler for feature normalization
- ✅ SHAP (SHapley Additive exPlanations) for model interpretability
- ✅ Flask web app with a polished, responsive UI
- ✅ Real-time cardiovascular risk percentage score
- ✅ Risk classification: **Low / Moderate / High**
- ✅ Model and scaler saved as `.pkl` files for reuse
- ✅ Live public URL via ngrok tunnel

---

## 🗂️ Project Structure

```
heart-disease-predictor/
│
├── updated_health.py       # Main Colab notebook (exported as .py)
├── heart.csv               # Dataset (from Kaggle)
├── health.pkl              # Saved trained model
├── scaler.pkl              # Saved StandardScaler
└── templates/
    └── index.html          # Flask HTML web interface (auto-generated)
```

---

## 📊 Dataset

- **Source:** [Kaggle — Heart Disease UCI](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci)
- **File:** `heart.csv`
- **Target column:** `target` (1 = Heart Disease, 0 = No Heart Disease)

### Input Features

| Feature      | Description                                      | Range / Unit     |
|--------------|--------------------------------------------------|------------------|
| `age`        | Age of the patient                               | Years            |
| `sex`        | Sex (0 = Female, 1 = Male)                       | 0 or 1           |
| `cp`         | Chest pain type                                  | 0–3              |
| `trestbps`   | Resting blood pressure                           | mm Hg            |
| `chol`       | Serum cholesterol                                | mg/dl            |
| `fbs`        | Fasting blood sugar > 120 mg/dl                  | 0 or 1           |
| `restecg`    | Resting ECG results                              | 0–2              |
| `thalach`    | Maximum heart rate achieved                      | bpm              |
| `exang`      | Exercise induced angina                          | 0 = No, 1 = Yes  |
| `oldpeak`    | ST depression induced by exercise                | Numeric          |
| `slope`      | Slope of peak exercise ST segment               | 0–2              |
| `ca`         | Number of major vessels colored by fluoroscopy  | 0–4              |
| `thal`       | Thalassemia type                                 | 1–3              |

---

## 🧠 Machine Learning Pipeline

```
heart.csv
    │
    ▼
Data Loading (pandas)
    │
    ▼
Train/Test Split (80/20)
    │
    ▼
Feature Scaling (StandardScaler)
    │
    ▼
Random Forest Classifier (n_estimators=100)
    │
    ▼
SHAP TreeExplainer (model interpretability)
    │
    ▼
Saved as health.pkl + scaler.pkl
    │
    ▼
Flask Web App → ngrok → Public URL
```

---

## 🖥️ Web Interface

The web app is built with **Flask** and features:

- A clean dark-themed UI with animated ECG line and heartbeat icon
- Input fields for all 13 clinical parameters
- On submission, displays:
  - **Prediction:** Heart Disease / No Heart Disease
  - **Risk Score:** percentage probability (e.g., 73.5%)
  - **Animated risk bar** (Low / Moderate / High)
  - **Status chips** with recommended action (e.g., "Urgent Review Advised")

> ⚠️ **Disclaimer:** This tool provides a statistical estimate only and is **not a substitute** for professional medical diagnosis. Always consult a qualified cardiologist.

---

## ⚙️ How to Run (Google Colab)

1. **Upload files** to your Colab session:
   - `heart.csv`
   - `updated_health.py` (or open the `.ipynb` directly)

2. **Install dependencies:**
   ```bash
   pip install flask pyngrok shap
   ```

3. **Add your ngrok auth token:**
   ```python
   !ngrok config add-authtoken "YOUR_NGROK_AUTH_TOKEN"
   ```

4. **Run all cells** in order. The script will:
   - Train and save the model
   - Generate the HTML template
   - Start the Flask server on port 5000
   - Print a public ngrok URL to access the app

5. **Open the ngrok URL** in your browser to use the web interface.

---

## 📦 Dependencies

| Library        | Purpose                          |
|----------------|----------------------------------|
| `pandas`       | Data loading and manipulation    |
| `scikit-learn` | ML model, scaler, train/test split |
| `shap`         | Model explainability (SHAP values) |
| `flask`        | Web framework                    |
| `pyngrok`      | Expose local server publicly     |
| `pickle`       | Save and load model artifacts    |
| `numpy`        | Numerical operations             |

---

## 📈 Model Details

| Parameter         | Value                    |
|-------------------|--------------------------|
| Algorithm         | Random Forest Classifier |
| Number of Trees   | 100                      |
| Test Size         | 20%                      |
| Feature Scaling   | StandardScaler           |
| Explainability    | SHAP TreeExplainer       |
| Output            | Binary (0 / 1) + Probability |

---

## 🔮 Future Improvements

- [ ] Add cross-validation and display accuracy/F1 score metrics
- [ ] Integrate SHAP force plots into the web UI for per-prediction explanation
- [ ] Add support for CSV batch predictions
- [ ] Deploy to a permanent cloud platform (Render, Hugging Face Spaces, etc.)
- [ ] Add data visualization / EDA section in the notebook

---

## 👨‍💻 Author

Built as a machine learning project using **Google Colab**.  
Dataset sourced from **Kaggle** — UCI Heart Disease dataset.

---

## 📄 License

This project is for educational purposes only. Not intended for clinical use.
