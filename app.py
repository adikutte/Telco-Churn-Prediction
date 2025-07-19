import streamlit as st
import pandas as pd
import pickle

# Load model
with open('churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.set_page_config(page_title="Telco Customer Churn Prediction", layout="wide")
st.title("Telco Customer Churn Prediction")
st.markdown("### Enter Customer Details")

# Equal layout with balanced fields
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["No", "Yes"])
    dependent = st.selectbox("Has Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phoneservice = st.selectbox("Phone Service", ["Yes", "No"])

with col2:
    multiple = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    onlinesec = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    onlineback = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    techsup = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

with col3:
    tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly = st.slider("Monthly Charges", 0.0, 150.0, 50.0)
    total = st.slider("Total Charges", 0.0, 10000.0, 500.0)

# Input data
input_data = pd.DataFrame([{
    'gender': 0, 'SeniorCitizen': 0, 'Partner': 0, 'Dependents': 0,
    'tenure': 0, 'PhoneService': 0, 'PaperlessBilling': 0,
    'MonthlyCharges': 0.0, 'TotalCharges': 0.0,
    'InternetService_DSL': 0, 'InternetService_Fiber optic': 0, 'InternetService_No': 0,
    'OnlineSecurity_No': 0, 'OnlineSecurity_No internet service': 0, 'OnlineSecurity_Yes': 0,
    'OnlineBackup_No': 0, 'OnlineBackup_No internet service': 0, 'OnlineBackup_Yes': 0,
    'DeviceProtection_No': 0, 'DeviceProtection_No internet service': 0, 'DeviceProtection_Yes': 0,
    'TechSupport_No': 0, 'TechSupport_No internet service': 0, 'TechSupport_Yes': 0,
    'StreamingTV_No': 0, 'StreamingTV_No internet service': 0, 'StreamingTV_Yes': 0,
    'StreamingMovies_No': 0, 'StreamingMovies_No internet service': 0, 'StreamingMovies_Yes': 0,
    'Contract_Month-to-month': 0, 'Contract_One year': 0, 'Contract_Two year': 0,
    'PaymentMethod_Bank transfer (automatic)': 0, 'PaymentMethod_Credit card (automatic)': 0,
    'PaymentMethod_Electronic check': 0, 'PaymentMethod_Mailed check': 0,
    'MultipleLines_No': 0, 'MultipleLines_No phone service': 0, 'MultipleLines_Yes': 0
}])

# Update values
input_data.at[0, 'gender'] = 1 if gender == 'Male' else 0
input_data.at[0, 'SeniorCitizen'] = 1 if senior == 'Yes' else 0
input_data.at[0, 'Partner'] = 1 if partner == 'Yes' else 0
input_data.at[0, 'Dependents'] = 1 if dependent == 'Yes' else 0
input_data.at[0, 'tenure'] = tenure
input_data.at[0, 'PhoneService'] = 1 if phoneservice == 'Yes' else 0
input_data.at[0, 'PaperlessBilling'] = 1 if paperless == 'Yes' else 0
input_data.at[0, 'MonthlyCharges'] = monthly
input_data.at[0, 'TotalCharges'] = total
input_data.at[0, f'InternetService_{internet}'] = 1
input_data.at[0, f'OnlineSecurity_{onlinesec}'] = 1
input_data.at[0, f'OnlineBackup_{onlineback}'] = 1
input_data.at[0, f'DeviceProtection_{device}'] = 1
input_data.at[0, f'TechSupport_{techsup}'] = 1
input_data.at[0, f'StreamingTV_{tv}'] = 1
input_data.at[0, f'StreamingMovies_{movies}'] = 1
input_data.at[0, f'Contract_{contract}'] = 1
input_data.at[0, f'PaymentMethod_{payment}'] = 1
input_data.at[0, f'MultipleLines_{multiple}'] = 1

# Match training columns
input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

# Predict
if st.button("Predict Churn"):
    result = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1] * 100
    st.markdown("### Prediction Result")
    if result == 1:
        st.error(f"Customer is likely to churn.\nProbability: {prob:.2f}%")
    else:
        st.success(f"Customer is not likely to churn.\nProbability: {100 - prob:.2f}%")


