import streamlit as st


import streamlit as st

from src.predict import predict_single_customer


st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="centered"
)

st.title("Customer Churn Prediction App")
st.write(
    "Predict whether a customer is likely to churn based on their profile and service usage."
)

st.divider()

# ------------------------------------------------------
# Customer Inputs
# ------------------------------------------------------

st.subheader("Customer Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["0", "1"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)

with col2:
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiline = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )
    internet = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )
    contract = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"]
    )
    monthly_charges = st.slider(
        "Monthly Charges ($)",
        20.0, 120.0, 70.0
    )

st.subheader("Internet Services")

col3, col4 = st.columns(2)

with col3:
    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )
    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )
    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

with col4:
    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )
    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )
    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

st.subheader("Billing")

paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

# ------------------------------------------------------
# Prediction
# ------------------------------------------------------

st.divider()

if st.button("🔍 Predict Churn"):
    customer_data = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiline,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges
    }

    with st.spinner("Predicting..."):
        result = predict_single_customer(customer_data)

    churn_prob = result["churn_probability"]
    prediction = result["prediction"]

    st.subheader("Prediction Result")

    st.metric(
        label="Churn Probability",
        value=f"{churn_prob * 100:.2f}%"
    )

    if prediction == "Churn":
        st.error(
            "High risk of churn. Consider retention offers or proactive support."
        )
    else:
        st.success(
            "Low churn risk. Customer is likely to stay."
        )
