"""
AirlineInsight AI - FastAPI Backend
====================================
This file is the backend (server) of the project.

What it does, in simple words:
  1. Loads the two trained models (saved with joblib).
  2. Gives two POST endpoints:
       POST /predict/regression      -> predicts arrival delay (a number)
       POST /predict/classification  -> predicts satisfaction (a category)
  3. Serves the website (frontend/index page) at  http://127.0.0.1:8000

Run it with:  uvicorn app.main:app --reload
"""

# ------------------------------------------------------------------
# Import the libraries we need
# ------------------------------------------------------------------
import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ------------------------------------------------------------------
# Load the trained models (only once, when the server starts)
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # project folder

regression_model = joblib.load(BASE_DIR / "models" / "regression_model.pkl")
classification_model = joblib.load(BASE_DIR / "models" / "classification_model.pkl")

# Load the model scores (R2, accuracy...) to show them on the website
with open(BASE_DIR / "models" / "model_info.json") as file:
    model_info = json.load(file)

# Create the FastAPI app
app = FastAPI(title="AirlineInsight AI")


# ------------------------------------------------------------------
# Input shapes: these classes define what data each form must send.
# FastAPI automatically checks the data using these rules.
# ------------------------------------------------------------------
class RegressionInput(BaseModel):
    departure_delay: float = Field(ge=0, le=1500)   # minutes, 0 or more
    flight_distance: float = Field(gt=0, le=10000)  # miles, more than 0


class ClassificationInput(BaseModel):
    # Passenger / flight details
    age: int = Field(ge=1, le=100)
    flight_distance: float = Field(gt=0, le=10000)
    gender: str            # "Male" or "Female"
    customer_type: str     # "Loyal Customer" or "dis-loyal Customer"
    travel_type: str       # "Business travel" or "Personal Travel"
    flight_class: str      # "Eco", "Eco Plus" or "Business"

    # The 14 service ratings (each between 0 and 5)
    inflight_wifi: int = Field(ge=0, le=5)
    time_convenient: int = Field(ge=0, le=5)
    online_booking: int = Field(ge=0, le=5)
    gate_location: int = Field(ge=0, le=5)
    food_and_drink: int = Field(ge=0, le=5)
    online_boarding: int = Field(ge=0, le=5)
    seat_comfort: int = Field(ge=0, le=5)
    inflight_entertainment: int = Field(ge=0, le=5)
    onboard_service: int = Field(ge=0, le=5)
    leg_room: int = Field(ge=0, le=5)
    baggage_handling: int = Field(ge=0, le=5)
    checkin_service: int = Field(ge=0, le=5)
    inflight_service: int = Field(ge=0, le=5)
    cleanliness: int = Field(ge=0, le=5)


# ------------------------------------------------------------------
# Home page: show the website (the HTML frontend)
# ------------------------------------------------------------------
@app.get("/")
def home():
    return FileResponse(BASE_DIR / "app" / "static" / "index.html")


# Small extra endpoint: the website uses it to show the model scores
@app.get("/model-info")
def get_model_info():
    return model_info


# ------------------------------------------------------------------
# ENDPOINT 1: POST /predict/regression
# Input : departure delay + flight distance
# Output: predicted arrival delay in minutes (JSON)
# ------------------------------------------------------------------
@app.post("/predict/regression")
def predict_regression(data: RegressionInput):
    # Put the input into a small table with the SAME column names
    # that were used during training (the model expects these names)
    input_data = pd.DataFrame([{
        "Departure Delay in Minutes": data.departure_delay,
        "Flight Distance": data.flight_distance,
    }])

    # Ask the model for the prediction.
    # A flight cannot arrive "before time" by a negative delay in
    # this simple app, so we do not allow a value below 0.
    prediction = regression_model.predict(input_data)[0]
    prediction = max(0.0, round(float(prediction), 1))

    # Return the answer as JSON
    return {"predicted_arrival_delay_minutes": prediction}


# ------------------------------------------------------------------
# ENDPOINT 2: POST /predict/classification
# Input : passenger details + 14 service ratings
# Output: "satisfied" or "neutral or dissatisfied" + probability
# ------------------------------------------------------------------
@app.post("/predict/classification")
def predict_classification(data: ClassificationInput):
    # Again, build a table with the SAME column names used in training
    input_data = pd.DataFrame([{
        "Gender": data.gender,
        "Customer Type": data.customer_type,
        "Age": data.age,
        "Type of Travel": data.travel_type,
        "Class": data.flight_class,
        "Flight Distance": data.flight_distance,
        "Inflight wifi service": data.inflight_wifi,
        "Departure/Arrival time convenient": data.time_convenient,
        "Ease of Online booking": data.online_booking,
        "Gate location": data.gate_location,
        "Food and drink": data.food_and_drink,
        "Online boarding": data.online_boarding,
        "Seat comfort": data.seat_comfort,
        "Inflight entertainment": data.inflight_entertainment,
        "On-board service": data.onboard_service,
        "Leg room service": data.leg_room,
        "Baggage handling": data.baggage_handling,
        "Checkin service": data.checkin_service,
        "Inflight service": data.inflight_service,
        "Cleanliness": data.cleanliness,
    }])

    # predict() gives the category: 1 = satisfied, 0 = not satisfied
    prediction = classification_model.predict(input_data)[0]

    # predict_proba() gives how confident the model is (0 to 1)
    probability = classification_model.predict_proba(input_data)[0][1]

    if prediction == 1:
        label = "satisfied"
    else:
        label = "neutral or dissatisfied"

    # Return the answer as JSON
    return {
        "prediction": label,
        "probability_satisfied": round(float(probability), 4),
    }
