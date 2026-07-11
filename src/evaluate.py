import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# Load Processed Dataset


DATA_PATH = "data/processed/churn_processed.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("Customer Churn Model Evaluation")
print("=" * 60)

print(f"\nDataset Shape : {df.shape}")


# Features and Target


X = df.drop("Churn", axis=1)

y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


# Train Test Split


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Load Saved Model


MODEL_PATH = "models/churn_model.pkl"

model = joblib.load(MODEL_PATH)

print("\nModel Loaded Successfully")


# Predictions


predictions = model.predict(X_test)


# Evaluation Metrics


accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("\n" + "=" * 60)
print("Evaluation Results")
print("=" * 60)

print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


# Classification Report


print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        predictions,
        target_names=["No", "Yes"]
    )
)

# Confusion Matrix


print("Confusion Matrix\n")

print(confusion_matrix(y_test, predictions))

print("\nEvaluation Completed Successfully.")