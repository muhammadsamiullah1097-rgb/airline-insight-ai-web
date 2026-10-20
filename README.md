# ✈️ AirlineInsight AI — Full-Stack ML Web App

A small machine learning web app that predicts:

1. **Flight arrival delay** (a number) — Linear Regression
2. **Passenger satisfaction** (a category) — Logistic Regression

Built with **scikit-learn**, **FastAPI** and a single **HTML/CSS/JavaScript** page.

---

## 🌐 Live Demo (no installation needed)

**https://muhammadsamiullah1097-rgb.github.io/airline-insight-ai-web/**

Open the link and both predictors work right in your browser.
The live page runs the *same trained models* — their learned numbers are written
into the page as JavaScript, so its answers are exactly the same as the
Python backend's (verified). The full FastAPI version below works the same way,
with the real `.pkl` models.

## ⚡ Quick Start (download → run in 3 commands)

The dataset is **already included** in this repo (`data/train.csv` +
`data/test.csv`) — nothing else to download.

```bash
pip install -r requirements.txt   # 1. install the libraries
python train_models.py            # 2. train the models (makes models/*.pkl)
uvicorn app.main:app --reload     # 3. start the server
```

Then open **http://127.0.0.1:8000** in your browser.

> Downloaded this project as a ZIP from GitHub? Just extract it and follow
> the same 3 commands — the CSV files are already inside `data/`, so the
> system is fully workable right away.

---

## 📊 Dataset

[Airline Passenger Satisfaction — Kaggle](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)

- **129,880 records** in total, in the ready-made split that comes with the dataset:
  - `data/train.csv` — **103,904 rows** → the models are *trained* (learn) on this file
  - `data/test.csv` — **25,976 rows** → the models are *tested* (scored) on this file
- Each record has passenger details, flight distance, delays and 14 service
  ratings (0–5), plus the final `satisfaction` label.
- Both files are included in this repo on purpose, so the project depends only
  on its own files and runs anywhere without a separate dataset download.

## 🤖 The Two Models

| # | Task | Model | Input | Output | Score |
|---|------|-------|-------|--------|-------|
| 1 | Regression | **Linear Regression** | Departure Delay, Flight Distance | Arrival Delay (minutes) | **R² = 0.9302**, MAE = 5.27 min |
| 2 | Classification | **Logistic Regression** | 14 service ratings, Age, Flight Distance, Gender, Customer Type, Type of Travel, Class | `satisfied` / `neutral or dissatisfied` | **Accuracy = 87.04%**, ROC-AUC = 0.9248 |

- Both models are trained by `train_models.py` and saved with **joblib** into `models/`.
- Training uses **train.csv**, scoring uses **test.csv** — `python train_models.py` prints both scores.
- In simple words, the regression model learned: `Arrival Delay ≈ 0.92 + 0.98 × Departure Delay`
  — a flight that leaves late arrives almost equally late.
- The classifier first prepares the data (numbers scaled with `StandardScaler`,
  text columns converted with `OneHotEncoder`), then Logistic Regression predicts
  the category together with a confidence probability.

## 📁 The 3 Parts of This Project

1. **ML models** — `train_models.py`
2. **Backend (FastAPI)** — `app/main.py` loads both models and gives two POST endpoints that return JSON:
   - `POST /predict/regression` → `{ "predicted_arrival_delay_minutes": 30.1 }`
   - `POST /predict/classification` → `{ "prediction": "satisfied", "probability_satisfied": 0.9914 }`
   - Plus `GET /model-info` → the model scores shown in the website header.
3. **Frontend (HTML)** — one page, one form per model:
   - `app/static/index.html` — served by the FastAPI backend
   - `index.html` (top level) — the live GitHub Pages version

## 📂 Project Structure

```
airline-insight-ai-web/
├── index.html             # Live web page (GitHub Pages)
├── README.md
├── VIVA_GUIDE.md          # Line-by-line explanation of all the code
├── train_models.py        # Trains + saves both models (joblib)
├── requirements.txt       # Python libraries needed
├── data/
│   ├── train.csv          # 103,904 rows (included)
│   ├── test.csv           # 25,976 rows (included)
│   └── README.md
├── models/
│   └── model_info.json    # Model scores shown on the page
│                          # (.pkl models are created by train_models.py)
└── app/
    ├── main.py            # FastAPI backend (2 POST endpoints)
    └── static/
        └── index.html     # Frontend served by FastAPI
```

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named ...` | Run `pip install -r requirements.txt` again |
| `FileNotFoundError: models/regression_model.pkl` | Run `python train_models.py` once — it creates the `.pkl` models from the included CSV files |
| Port 8000 already in use | Run `uvicorn app.main:app --port 8001` and open http://127.0.0.1:8001 |
| Page opens but predictions fail (local version) | The FastAPI server must be running — the frontend sends the forms to it |

## ⚠️ Honest Notes

- The delay model mostly uses the **departure delay**, so it is an *estimate at/after departure*, not a forecast days before the flight.
- Satisfaction is predicted from the passenger's **own survey ratings**, so the classifier best shows *which services drive satisfaction*.
- The `.pkl` model files are binary and are not stored in this GitHub repo — `train_models.py` recreates them from the included dataset in a few seconds.

## 🛠️ Tech Used
Python · pandas · scikit-learn · joblib · FastAPI · HTML/CSS/JavaScript
