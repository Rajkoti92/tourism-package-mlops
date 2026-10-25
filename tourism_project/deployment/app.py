"""Streamlit app for Tourism Package Purchase Prediction."""
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import json

st.set_page_config(page_title="Tourism Package Predictor", page_icon="✈️", layout="centered")

BASE_DIR = os.path.dirname(__file__)

# Load preprocessing config and XGBoost model using pure Booster (no sklearn needed)
with open(os.path.join(BASE_DIR, "preprocessing_config.json")) as f:
    PREPROCESS_CONFIG = json.load(f)

MODEL = xgb.Booster()
MODEL.load_model(os.path.join(BASE_DIR, "xgboost_model.json"))

NUM_COLS = PREPROCESS_CONFIG["numeric_columns"]
NUM_FILL = PREPROCESS_CONFIG["numeric_fill_values"]
CAT_COLS = PREPROCESS_CONFIG["categorical_columns"]
CAT_FILL = PREPROCESS_CONFIG["categorical_fill_values"]
CAT_CATEGORIES = PREPROCESS_CONFIG["categories"]
FEATURE_ORDER = PREPROCESS_CONFIG["feature_names_out"]


def preprocess(df):
    """Replicate the sklearn ColumnTransformer preprocessing in pure pandas."""
    result = pd.DataFrame()
    # Numeric: fill missing with median, keep as-is
    for col in NUM_COLS:
        result[f"num__{col}"] = df[col].fillna(NUM_FILL[col])
    # Categorical: fill missing with mode, then one-hot encode (drop='first')
    for col in CAT_COLS:
        filled = df[col].fillna(CAT_FILL[col])
        categories = CAT_CATEGORIES[col]
        # drop='first' means skip the first category
        for cat in categories[1:]:
            result[f"cat__{col}_{cat}"] = (filled == cat).astype(float)
    return result[FEATURE_ORDER].values.astype(np.float32)


st.title("✈️ Tourism Package Purchase Predictor")
st.write("Predict whether a customer will purchase the Wellness Tourism Package.")

st.subheader("Customer Details")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    num_persons = st.number_input("Number of Persons Visiting", min_value=1, max_value=5, value=2)
    preferred_star = st.selectbox("Preferred Property Star", [3, 4, 5])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])

with col2:
    num_trips = st.number_input("Number of Trips (annual)", min_value=1, max_value=10, value=2)
    passport = st.selectbox("Passport", [0, 1], format_func=lambda x: "Yes" if x else "No")
    pitch_score = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
    own_car = st.selectbox("Own Car", [0, 1], format_func=lambda x: "Yes" if x else "No")
    num_children = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly Income", min_value=5000, max_value=100000, value=20000)

st.subheader("Interaction Details")
col3, col4 = st.columns(2)

with col3:
    duration_of_pitch = st.number_input("Duration of Pitch (min)", min_value=1, max_value=60, value=10)
    num_followups = st.number_input("Number of Follow-ups", min_value=1, max_value=6, value=3)

with col4:
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])

input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_persons,
    "NumberOfFollowups": float(num_followups),
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": float(preferred_star),
    "MaritalStatus": marital_status,
    "NumberOfTrips": float(num_trips),
    "Passport": passport,
    "PitchSatisfactionScore": pitch_score,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": float(num_children),
    "Designation": designation,
    "MonthlyIncome": monthly_income,
}])

if st.button("Predict", type="primary"):
    features = preprocess(input_data)
    dmatrix = xgb.DMatrix(features, feature_names=FEATURE_ORDER)
    prob_positive = MODEL.predict(dmatrix)[0]
    prediction = int(prob_positive > 0.5)
    if prediction == 1:
        st.success(f"The customer is **likely to purchase** the package. (Confidence: {prob_positive:.1%})")
    else:
        st.info(f"The customer is **unlikely to purchase** the package. (Confidence: {1 - prob_positive:.1%})")
