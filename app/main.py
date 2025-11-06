
import pandas as pd
import joblib
import numpy as np


class LoanPredictionService:
    def __init__(self):
        self.default_model = None
        self.interest_model = None
        self.feature_columns = None
        self.load_models()

    def load_models(self):
        """Load the trained AI models"""
        try:
            self.default_model = joblib.load("models/default_classifier.joblib")
            self.interest_model = joblib.load("models/interest_regressor.joblib")
            self.feature_columns = joblib.load("models/feature_columns.joblib")
            print("✅ AI Models loaded successfully!")
        except FileNotFoundError as e:
            print(f"❌ Models not found: {e}. Please train models first.")
        except Exception as e:
            # Scoped fallback for unexpected errors (corrupt file, unpickling error, etc.)
            print(f"❌ Failed to load models: {e}")
            # Optionally: raise  # re-raise if you want the app to fail fast

    def predict_loan_application(self, application_data):
        """Make predictions for a new loan application"""
        # Create feature vector
        features = [application_data[col] for col in self.feature_columns]
        features_array = np.array(features).reshape(1, -1)

        # Make predictions
        default_prob = self.default_model.predict_proba(features_array)[0][1]
        default_prediction = self.default_model.predict(features_array)[0]
        interest_rate = self.interest_model.predict(features_array)[0]

        # Decision logic
        if default_prob > 0.7:
            decision = "REJECT"
            reason = "High default risk"
        elif default_prob > 0.4:
            decision = "APPROVE WITH CAUTION"
            reason = "Moderate default risk"
        else:
            decision = "APPROVE"
            reason = "Low default risk"

        return {
            "decision": decision,
            "default_probability": f"{default_prob:.2%}",
            "suggested_interest_rate": f"{interest_rate:.2f}%",
            "reason": reason,
            "risk_level": "HIGH"
            if default_prob > 0.6
            else "MEDIUM"
            if default_prob > 0.3
            else "LOW",
        }


def main():
    """Main interactive application"""
    print("🤖 AI LOAN APPROVAL SYSTEM")
    print("=" * 40)

    service = LoanPredictionService()

    while True:
        print("\n📝 Enter Loan Application Details:")

        try:
            # Collect application data
            app_data = {
                "age": int(input("Age: ")),
                "income": int(input("Annual Income: $")),
                "credit_score": int(input("Credit Score (300-850): ")),
                "loan_amount": int(input("Loan Amount: $")),
                "loan_term": int(input("Loan Term (months): ")),
                "employment_years": int(input("Years Employed: ")),
                "debt_to_income": float(input("Debt-to-Income Ratio (0.1-0.8): ")),
                "existing_loans": int(input("Number of Existing Loans: ")),
                "savings_balance": int(input("Savings Balance: $")),
            }

            # Get AI prediction
            result = service.predict_loan_application(app_data)

            # Display results
            print("\n" + "=" * 40)
            print("🎯 AI DECISION RESULTS:")
            print(f"Decision: {result['decision']}")
            print(f"Default Probability: {result['default_probability']}")
            print(f"Suggested Interest Rate: {result['suggested_interest_rate']}")
            print(f"Risk Level: {result['risk_level']}")
            print(f"Reason: {result['reason']}")
            print("=" * 40)

        except ValueError:
            print("❌ Invalid input! Please enter numbers only.")
        except Exception as e:
            print(f"❌ Error: {e}")

        # Ask to continue
        continue_app = input("\nProcess another application? (y/n): ").lower()
        if continue_app != "y":
            print("Thank you for using AI Loan Approval System! 👋")
            break


if __name__ == "__main__":
    main()
