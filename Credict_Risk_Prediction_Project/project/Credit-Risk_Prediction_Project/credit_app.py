import streamlit as st
import boto3
import json
import os

# Load endpoint name
if os.path.exists("endpoint.txt"):
    with open("endpoint.txt", "r") as f:
        ENDPOINT_NAME = f.read().strip()
else:
    # fallback: get latest endpoint from SageMaker
    sm = boto3.client("sagemaker")
    endpoints = sm.list_endpoints(SortBy="CreationTime", SortOrder="Descending")
    ENDPOINT_NAME = endpoints["Endpoints"][0]["EndpointName"]

runtime = boto3.client("sagemaker-runtime")

st.set_page_config(page_title="Credit Risk Predictor", layout="centered")
st.title("💳 Credit Risk Prediction")
st.write("Enter applicant details to predict the probability of loan default.")

with st.form("credit_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Annual Income ($)", min_value=5000, max_value=500000, value=50000)
    loan_amount = st.number_input("Loan Amount ($)", min_value=1000, max_value=200000, value=15000)
    submitted = st.form_submit_button("Predict Risk")

if submitted:
    features = [age, income, loan_amount]
    payload = json.dumps({"features": features})

    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=payload
    )
    result = json.loads(response["Body"].read().decode())

    prediction = result["prediction"][0]

    st.subheader("📊 Prediction Result")
    if "probabilities" in result:
        probabilities = result["probabilities"][0]
        st.write(f"**Prediction:** {'Default (1)' if prediction == 1 else 'No Default (0)'}")
        st.write(f"**Probabilities:** Default = {probabilities[1]:.2f}, No Default = {probabilities[0]:.2f}")
        st.progress(probabilities[1])
        if prediction == 1:
            st.error("⚠️ High Risk: The applicant may not repay the loan.")
        else:
            st.success("✅ Low Risk: The applicant is likely to repay the loan.")
    else:
        st.write(f"**Prediction:** {'Default (1)' if prediction == 1 else 'No Default (0)'}")
