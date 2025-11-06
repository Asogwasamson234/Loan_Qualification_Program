# Combines the all the dataset
import joblib
import pandas as pd


def predict_loan_application():
    """Use the trained AI to predict loan defaults with YOUR actual columns"""

    # Load the trained model and feature info
    model_data = joblib.load("trained_loan_model.joblib")
    model = model_data["model"]
    feature_columns = model_data["feature_columns"]
    target_column = model_data.get("target_column", "will_default")

    print("🤖 AI LOAN PREDICTION SYSTEM")
    print("=" * 50)
    print("Using your actual dataset columns:")
    for i, col in enumerate(feature_columns, 1):
        print(f"  {i}. {col}")
    print()

    # Get application details using YOUR actual column names
    print("Enter loan application details:")
    application = {}

    for col in feature_columns:
        if col == "income_annum":
            application[col] = float(input("Annual Income (income_annum): $"))
        elif col == "loan_amount":
            application[col] = float(input("Loan Amount: $"))
        elif col == "cibil_score":
            application[col] = float(input("Credit Score (cibil_score): "))
        elif col == "bank_asset_value":
            application[col] = float(input("Bank Asset Value: $"))
        elif col == "commercial_assets_value":
            application[col] = float(input("Commercial Assets Value: $"))
        elif col == "residential_assets_value":
            application[col] = float(input("Residential Assets Value: $"))
        elif col == "loan_term":
            application[col] = float(input("Loan Term (months): "))
        else:
            application[col] = float(input(f"{col}: "))

    # Create DataFrame with EXACT same columns as training
    app_df = pd.DataFrame([application])[feature_columns]

    # Make prediction
    prediction = model.predict(app_df)[0]
    probability = model.predict_proba(app_df)[0][1]

    print(f"\n🎯 AI PREDICTION RESULTS:")
    print(f"Default Probability: {probability:.2%}")
    print(f"Recommendation: {'APPROVE' if prediction == 0 else 'REJECT'}")
    print(
        f"Risk Level: {'LOW' if probability < 0.3 else 'MEDIUM' if probability < 0.7 else 'HIGH'}"
    )


if __name__ == "__main__":
    predict_loan_application()
