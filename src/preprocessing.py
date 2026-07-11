import os
import pandas as pd

# STEP 1: Load Dataset

DATA_PATH = "data/raw/Telco-Customer-Churn.csv"

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

# STEP 2: Basic Information

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

# STEP 3: Missing Values

print("\nMissing Values:")
print(df.isnull().sum())

# STEP 4: Check TotalCharges

print("\nData Type of TotalCharges:")
print(df["TotalCharges"].dtype)

blank_values = (df["TotalCharges"] == " ").sum()

print(f"\nBlank values in TotalCharges: {blank_values}")

# STEP 5: Convert TotalCharges to Numeric

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

print("\nMissing values after conversion:")
print(df["TotalCharges"].isnull().sum())

# Fill missing values with median
df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# STEP 6: Remove Duplicate Rows

duplicates = df.duplicated().sum()

print(f"\nDuplicate Rows: {duplicates}")

if duplicates > 0:
    df.drop_duplicates(inplace=True)

# STEP 7: Drop Unnecessary Column

print("\nDropping customerID column...")

df.drop("customerID", axis=1, inplace=True)

# STEP 8: Save Cleaned Dataset

os.makedirs("data/processed", exist_ok=True)

OUTPUT_PATH = "data/processed/churn_processed.csv"

df.to_csv(OUTPUT_PATH, index=False)

print("\nCleaned dataset saved successfully.")

# STEP 9: Final Check

print("\nFinal Dataset Shape:")
print(df.shape)

print("\nFinal Data Types:")
print(df.dtypes)

print("\nFirst 5 Rows of Cleaned Dataset:")
print(df.head())

print("\nPreprocessing Completed Successfully!")