import joblib
import pandas as pd

# Load Saved Model

MODEL_PATH = "models/churn_model.pkl"

model = joblib.load(MODEL_PATH)

print("=" * 60)
print("Customer Churn Prediction")
print("=" * 60)

# Sample Customer Data

sample_customer = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 75.50,
    "TotalCharges": 906.00,
}

# Convert dictionary to DataFrame
customer_df = pd.DataFrame([sample_customer])

print("\nCustomer Details:")
print(customer_df)

# Predict

prediction = model.predict(customer_df)[0]

# Prediction Probability

probability = model.predict_proba(customer_df)[0]

# Display Result

print("\nPrediction Result")
print("-" * 40)

if prediction == 1:
    print("Customer is likely to CHURN.")
else:
    print("Customer is NOT likely to churn.")

print(f"\nProbability of NOT Churning : {probability[0]:.4f}")
print(f"Probability of Churning     : {probability[1]:.4f}")

print("\nPrediction Completed Successfully.")