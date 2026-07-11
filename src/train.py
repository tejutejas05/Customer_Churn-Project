import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from xgboost import XGBClassifier

# Load Dataset

DATA_PATH = "data/processed/churn_processed.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("Customer Churn Model Training")
print("=" * 60)

print(f"\nDataset Shape : {df.shape}")

# Features & Target

X = df.drop("Churn", axis=1)

y = df["Churn"]

# Convert target

y = y.map({
    "No": 0,
    "Yes": 1
})

# Numerical & Categorical Columns

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("\nCategorical Columns")
print(categorical_columns)

print("\nNumerical Columns")
print(numerical_columns)

# Preprocessor

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns,
        )
    ],
    remainder="passthrough",
)

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# MLflow

mlflow.set_experiment("Customer Churn XGBoost")

# Hyperparameter Values

n_estimators_list = [50, 100, 150, 200, 300]

best_f1 = 0
best_model = None

os.makedirs("models", exist_ok=True)

# Training Loop

for n in n_estimators_list:

    print("\n" + "=" * 60)
    print(f"Training XGBoost (n_estimators={n})")
    print("=" * 60)

    with mlflow.start_run(run_name=f"XGB_{n}_Trees"):

        model = XGBClassifier(
            n_estimators=n,
            learning_rate=0.1,
            max_depth=4,
            random_state=42,
            eval_metric="logloss",
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", model),
            ]
        )

        # Train

        pipeline.fit(X_train, y_train)

        # Predict

        predictions = pipeline.predict(X_test)

        # Metrics

        accuracy = accuracy_score(
            y_test,
            predictions,
        )

        precision = precision_score(
            y_test,
            predictions,
        )

        recall = recall_score(
            y_test,
            predictions,
        )

        f1 = f1_score(
            y_test,
            predictions,
        )

        print(f"Accuracy  : {accuracy:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1 Score  : {f1:.4f}")

        print("\nConfusion Matrix")

        print(confusion_matrix(
            y_test,
            predictions,
        ))

        # MLflow Logs

        mlflow.log_param("Algorithm", "XGBoost")
        mlflow.log_param("n_estimators", n)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("max_depth", 4)

        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("F1 Score", f1)

        mlflow.xgboost.log_model(
        model.get_booster(),
        name="model"
        )

        if f1 > best_f1:

            best_f1 = f1
            best_model = pipeline

# Save Best Model

joblib.dump(
    best_model,
    "models/churn_model.pkl"
)

print("\n" + "=" * 60)
print("Training Completed")
print("=" * 60)

print(f"\nBest F1 Score : {best_f1:.4f}")

print("\nBest Model Saved Successfully")

print("\nLocation : models/churn_model.pkl")

print("\nFinished")