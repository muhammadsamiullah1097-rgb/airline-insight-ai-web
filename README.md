# AirlineInsight AI — Full-Stack ML Web App

A small machine learning web app that predicts:

1. **Flight arrival delay** (a number) — Linear Regression
2. **Passenger satisfaction** (a category) — Logistic Regression

Dataset: [Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction) (Kaggle) — 129,880 survey records. The CSV files are **included in this repo** (`data/train.csv` + `data/test.csv`), so nothing needs to be downloaded separately.

## 🌐 Live Demo

Open the web app directly in your browser — no installation needed:

**https://muhammadsamiullah1097-rgb.github.io/airline-insight-ai-web/**

The live page runs the *same trained models* inside the browser using JavaScript
(the models' learned numbers are written into the page), so it gives exactly
the same answers as the Python backend.

## 📊 Model Scores

| Model | Task | Score |
|---|---|---|
| Linear Regression | Arrival Delay | R² = 0.9302 · MAE = 5.27 min |
| Logistic Regression | Satisfaction | Accuracy = 87.04% · ROC-AUC = 0.9248 |

## 📁 The 3 parts of this project

1. **ML models** — `train_models.py` trains both models with scikit-learn and saves them in `models/` using **joblib**.
2. **Backend** — `app/main.py` (FastAPI) loads both models and gives two POST endpoints that return JSON:
   - `POST /predict/regression`
   - `POST /predict/classification`
3. **Frontend** — one HTML page with a form for each model:
   - `app/static/index.html` — served by the FastAPI backend
   - `index.html` (top level) — the live GitHub Pages version

## ▶️ How to run the full app locally (FastAPI version)

```bash
# 1. Nothing to download - the dataset is already in data/
#    (train.csv + test.csv are included in this repo)

# 2. Install the libraries
pip install -r requirements.txt

# 3. Train the models (creates the .pkl files in models/)
python train_models.py

# 4. Start the server
uvicorn app.main:app --reload

# 5. Open in your browser:  http://127.0.0.1:8000
```

## 📂 Project structure

```
index.html             <- Live web page (GitHub Pages)
app/main.py            <- FastAPI backend (2 POST endpoints)
app/static/index.html  <- Frontend page served by FastAPI
train_models.py        <- Trains + saves both models (joblib)
models/model_info.json <- Model scores shown on the page
data/README.md         <- Where to get the dataset
VIVA_GUIDE.md          <- Line-by-line explanation of all the code
```
