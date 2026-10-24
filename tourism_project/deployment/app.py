"""Streamlit app for Tourism Package Purchase Prediction."""
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Tourism Package Predictor", page_icon="✈️", layout="centered")

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "tourism_model.joblib")
model = joblib.load(MODEL_PATH)

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

# Build the input dataframe
input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_persons,
    "NumberOfFollowups": num_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": num_trips,
    "Passport": passport,
    "PitchSatisfactionScore": pitch_score,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": num_children,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
}])

if st.button("Predict", type="primary"):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]
    if prediction == 1:
        st.success(f"The customer is **likely to purchase** the package. (Confidence: {proba[1]:.1%})")
    else:
        st.info(f"The customer is **unlikely to purchase** the package. (Confidence: {proba[0]:.1%})")
