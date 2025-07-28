import streamlit as st
import pandas as pd
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.model import train_model, evaluate_model
from src.utils import plot_actual_vs_predicted

st.set_page_config(page_title="🎓 UCLA Admission Predictor", layout="wide")
st.title("🎓 UCLA Admission Predictor (MLP Neural Network)")

# Load and preview data
df = load_data("data/Admission.csv")
st.write("### Dataset Sample", df.head())

# Preprocess data
X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)

# Train model
model = train_model(X_train, y_train)

# Evaluate model
r2, mse, predictions = evaluate_model(model, X_test, y_test)
st.write("### 🔍 Model Performance")
st.metric("R² Score", round(r2, 3))
st.metric("Mean Squared Error", round(mse, 4))

# Plot predictions
fig = plot_actual_vs_predicted(y_test, predictions)
st.pyplot(fig)

# User input prediction form
st.write("### 📋 Try Predicting Admission Chance")

# Define which fields should be integers
int_fields = ['GRE_Score', 'TOEFL_Score', 'University_Rating', 'Research']

with st.form("predict_form"):
    inputs = {}
    for col in feature_names:
        default = float(df[col].mean())

        if col in int_fields:
            inputs[col] = st.number_input(col, value=int(default), step=1)
        else:
            inputs[col] = st.number_input(col, value=round(default, 2), step=0.1)

    submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame([inputs])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        st.success(f"🎯 Estimated Chance of Admission: **{prediction:.2f}**")