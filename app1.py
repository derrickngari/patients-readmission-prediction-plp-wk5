import streamlit as st
import pandas as pd
import pickle
import streamlit_shadcn_ui as ui

# Load model and expected feature names
model = pickle.load(open('readmission_model.pkl', 'rb'))
expected_features = model.feature_names_in_  # Ensures alignment

# Page config
st.set_page_config(page_title="Patient Readmission Prediction", page_icon="🏥", layout="wide")
st.title("Patient Readmission Prediction")

# Initialize session state defaults
defaults = {
    "gender": "Male",
    "diabetes": "No",
    "hypertension": "No",
    "discharge": "Home",
    "age": 30,
    "bmi": 25.0,
    "meds": 10,
    "visits": 2
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# UI Card to group inputs
with ui.card(key="form_card"):
    st.subheader("🧑‍⚕️ Patient Demographics")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("gender")
        ui.select(options=["Male", "Female"], label="Gender", key="gender")
        st.subheader("diabetes?")
        ui.select(options=["Yes", "No"], label="Diabetes", key="diabetes")

    with col2:
        st.subheader("hypertension?")
        ui.select(options=["Yes", "No"], label="Hypertension", key="hypertension")
        st.subheader("discharged destination")
        ui.select(options=["Nursing_Facility", "Home", "Rehab"], label="Discharge Destination", key="discharge")

    st.subheader("📊 Health Metrics")
    age = st.slider("Age", min_value=0, max_value=120, value=30)
    bmi = st.slider("BMI", min_value=10.0, max_value=50.0, value=25.0)
    num_medications = st.slider("Number of Medications", min_value=0, max_value=50, value=10)
    visits = st.slider("Number of visits", min_value=0, max_value=20, value=2)

predict_btn = ui.button(text="🔍 Predict Readmission", key="predict_btn")

# Prediction logic
if predict_btn:
    # Raw input dictionary
    input_data = {
        'gender': st.session_state.gender,
        'diabetes': st.session_state.diabetes,
        'hypertension': st.session_state.hypertension,
        'discharge_destination': st.session_state.discharge,
        'age': age,
        'bmi': bmi,
        'num_medications': num_medications,
        'visits': visits
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # One-hot encode categorical variables
    input_encoded = pd.get_dummies(input_df)

    # Align with model's expected features
    input_aligned = input_encoded.reindex(columns=expected_features, fill_value=0)

    # Make prediction
    prediction = model.predict(input_aligned)
    result = "🔁 Patient has a high chance of being Readmitted" if prediction[0] == 1 else "✅ Patient will not be Readmitted"

    # Show result
    # Show result with icons
    st.write("Prediction Result:", result)


# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; font-size: 14px; color: gray;'>
        © 2025 <strong><a href="https://github.com/maxallday" target="_blank">Maxallday</a></strong>
        &nbsp;|&nbsp; Special thanks to <a href="https://github.com/derrickngari" target="_blank">Ngari</a> and <a href="https://github.com/Bossy-V-Osinde" target="_blank">Osinde</a> for their contributions.
        "tout est facile"
        <br>    
    </div>
    """,
    unsafe_allow_html=True
)
