"""
AirlineInsight AI - Model Training Script
==========================================
This script trains BOTH models for the assignment:

  Model 1 (Regression)     : Linear Regression
                             Predicts "Arrival Delay in Minutes" (a number)

  Model 2 (Classification) : Logistic Regression
                             Predicts passenger "satisfaction" (a category)

Run it with:  python train_models.py
It saves both trained models in the "models" folder using joblib.
"""

# ------------------------------------------------------------------
# STEP 0: Import the libraries we need
# ------------------------------------------------------------------
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ------------------------------------------------------------------
# STEP 1: Load the dataset (train file + test file given by Kaggle)
# ------------------------------------------------------------------
train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

print("Train rows:", len(train_data))
print("Test rows :", len(test_data))

# ==================================================================
# MODEL 1: LINEAR REGRESSION  ->  Predict Arrival Delay
# ==================================================================
# Idea in simple words:
# If a flight leaves late, it also arrives late.
# So we use "Departure Delay" and "Flight Distance" to predict
# "Arrival Delay".

# The 2 input columns (features) and the 1 output column (target)
REG_FEATURES = ["Departure Delay in Minutes", "Flight Distance"]
REG_TARGET = "Arrival Delay in Minutes"

# Some rows have no arrival delay recorded, so we remove them.
train_reg = train_data.dropna(subset=[REG_TARGET])
test_reg = test_data.dropna(subset=[REG_TARGET])

# Split inputs (X) and answers (y)
X_train = train_reg[REG_FEATURES]
y_train = train_reg[REG_TARGET]
X_test = test_reg[REG_FEATURES]
y_test = test_reg[REG_TARGET]

# Create the model and train it (it "learns" from the train data)
regression_model = LinearRegression()
regression_model.fit(X_train, y_train)

# Ask the model to predict on the test data, then check the score
reg_predictions = regression_model.predict(X_test)
reg_r2 = r2_score(y_test, reg_predictions)              # R2 score (1.0 = perfect)
reg_mae = mean_absolute_error(y_test, reg_predictions)  # average error in minutes

print("\n--- Model 1: Linear Regression (Arrival Delay) ---")
print("R2 Score :", round(reg_r2, 4))
print("MAE      :", round(reg_mae, 2), "minutes")
print("Learned formula: Arrival Delay =",
      round(regression_model.intercept_, 2),
      "+", round(regression_model.coef_[0], 2), "x Departure Delay",
      "+", round(regression_model.coef_[1], 4), "x Flight Distance")

# Save the trained regression model
joblib.dump(regression_model, "models/regression_model.pkl")
print("Saved: models/regression_model.pkl")

# ==================================================================
# MODEL 2: LOGISTIC REGRESSION  ->  Predict Passenger Satisfaction
# ==================================================================
# Idea in simple words:
# A passenger who rates the services high (wifi, food, seat, ...)
# is most likely "satisfied". We use the service ratings plus a few
# passenger/flight details to predict the satisfaction category.

# The 14 service rating columns (each rated 0 to 5 by the passenger)
RATING_COLUMNS = [
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Inflight service",
    "Cleanliness",
]

# Number columns: the ratings + age + flight distance
NUMERIC_FEATURES = RATING_COLUMNS + ["Age", "Flight Distance"]

# Text columns (words, not numbers) - the model cannot read words,
# so these will be converted to numbers by OneHotEncoder below.
TEXT_FEATURES = ["Gender", "Customer Type", "Type of Travel", "Class"]

ALL_FEATURES = NUMERIC_FEATURES + TEXT_FEATURES

# Change the answer column into simple 0 / 1 numbers:
# "satisfied" = 1, "neutral or dissatisfied" = 0
y_train = (train_data["satisfaction"] == "satisfied").astype(int)
y_test = (test_data["satisfaction"] == "satisfied").astype(int)
X_train = train_data[ALL_FEATURES]
X_test = test_data[ALL_FEATURES]

# Preprocessing (prepare the data before the model sees it):
#  - Number columns -> StandardScaler makes all numbers a similar size
#  - Text columns    -> OneHotEncoder converts words into 0/1 columns
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC_FEATURES),
    ("cat", OneHotEncoder(handle_unknown="ignore"), TEXT_FEATURES),
])

# Pipeline = preprocessing + model, joined like steps of one machine.
# The data flows: raw input -> preprocessor -> Logistic Regression.
classification_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000)),
])

# Train the classification model
classification_model.fit(X_train, y_train)

# Predict on the test data and check the scores
class_predictions = classification_model.predict(X_test)
class_probabilities = classification_model.predict_proba(X_test)[:, 1]
class_accuracy = accuracy_score(y_test, class_predictions)
class_roc = roc_auc_score(y_test, class_probabilities)

print("\n--- Model 2: Logistic Regression (Satisfaction) ---")
print("Accuracy :", round(class_accuracy, 4))
print("ROC-AUC  :", round(class_roc, 4))

# Save the trained classification model
joblib.dump(classification_model, "models/classification_model.pkl")
print("Saved: models/classification_model.pkl")

# ==================================================================
# STEP LAST: Save the scores in a small JSON file.
# The website reads this file to show the scores on the page.
# ==================================================================
model_info = {
    "regression": {
        "model": "Linear Regression",
        "predicts": "Arrival Delay in Minutes",
        "r2_score": round(reg_r2, 4),
        "mae_minutes": round(reg_mae, 2),
    },
    "classification": {
        "model": "Logistic Regression",
        "predicts": "Passenger Satisfaction",
        "accuracy": round(class_accuracy, 4),
        "roc_auc": round(class_roc, 4),
    },
}

with open("models/model_info.json", "w") as file:
    json.dump(model_info, file, indent=2)

print("\nSaved: models/model_info.json")
print("All done! Both models are trained and saved.")
