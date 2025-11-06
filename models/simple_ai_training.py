# fixed_ai_training
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

print("🚀 AI TRAINING WITH YOUR ACTUAL DATASET COLUMNS")
print("=" * 50)

# 1. Load your combined dataset
df = pd.read_csv("final_combined_dataset.csv")
print(f"✅ Dataset loaded: {df.shape}")

# 2. Show ALL columns we have
print(f"📊 ALL COLUMNS IN YOUR DATASET:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")

# 3. Check for existing target column or create one
target_column = None
potential_targets = ["loan_status", "default", "fraud", "risk", "status", "approved"]

for col in df.columns:
    if any(target in col.lower() for target in potential_targets):
        target_column = col
        print(f"🎯 FOUND EXISTING TARGET: {target_column}")
        break

if target_column is None:
    # Create target based on loan risk factors (using YOUR actual columns)
    print("⚙️ Creating target variable based on your dataset columns...")

    # Use the columns from your error message
    if "income_annum" in df.columns and "loan_amount" in df.columns:
        # Higher loan-to-income ratio = higher risk
        df["will_default"] = (df["loan_amount"] > df["income_annum"] * 0.5).astype(int)
        target_column = "will_default"
    elif "cibil_score" in df.columns:
        # Lower credit score = higher risk
        df["will_default"] = (df["cibil_score"] < 600).astype(int)
        target_column = "will_default"
    else:
        # Use first suitable column
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            median_val = df[numeric_cols[0]].median()
            df["will_default"] = (df[numeric_cols[0]] > median_val).astype(int)
            target_column = "will_default"

print(f"🎯 Using target: {target_column}")
print(f"Default distribution: {df[target_column].value_counts().to_dict()}")

# 4. Select features (using YOUR actual column names)
# Based on the error, your dataset has these columns:
feature_cols = [
    "bank_asset_value",
    "cibil_score",
    "commercial_assets_value",
    "income_annum",
    "loan_amount",
    "loan_term",
    "residential_assets_value",
]

# Only use columns that actually exist in your dataset
available_features = [col for col in feature_cols if col in df.columns]
print(f"🤖 Using available features: {available_features}")

if len(available_features) < 2:
    # Fallback: use all numeric columns except target
    available_features = [
        col
        for col in df.columns
        if col != target_column and pd.api.types.is_numeric_dtype(df[col])
    ]
    print(f"🔧 Fallback to all numeric features: {available_features}")

X = df[available_features]
y = df[target_column]

print(f"📊 Final training data: X={X.shape}, y={y.shape}")

# 5. Train the model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Check accuracy
accuracy = model.score(X_test, y_test)
print(f"🎯 MODEL ACCURACY: {accuracy:.2%}")

# 7. Save the model AND feature names
model_data = {
    "model": model,
    "feature_columns": available_features,
    "target_column": target_column,
}
joblib.dump(model_data, "trained_loan_model.joblib")
print("💾 Model saved as 'trained_loan_model.joblib'")

# 8. Show feature importance
if hasattr(model, "feature_importances_"):
    print("\n📊 FEATURE IMPORTANCE:")
    importance_df = pd.DataFrame(
        {"feature": available_features, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    print(importance_df)

print("\n✅ AI TRAINING COMPLETE! Ready for predictions! 🎉")
