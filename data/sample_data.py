
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification


def generate_loan_data():
    """
    Generate realistic loan application data
    """
    np.random.seed(42)

    n_samples = 5000

    # Create realistic features for loan applicants
    data = {
        "age": np.random.randint(18, 70, n_samples),
        "income": np.random.normal(50000, 20000, n_samples).astype(int),
        "credit_score": np.random.randint(300, 850, n_samples),
        "loan_amount": np.random.randint(5000, 100000, n_samples),
        "loan_term": np.random.choice([12, 24, 36, 48, 60], n_samples),
        "employment_years": np.random.exponential(5, n_samples).astype(int),
        "debt_to_income": np.random.uniform(0.1, 0.8, n_samples),
        "existing_loans": np.random.randint(0, 5, n_samples),
        "savings_balance": np.random.exponential(10000, n_samples).astype(int),
    }

    df = pd.DataFrame(data)

    # Create realistic target variable (will they default?)
    # Higher risk if: low credit score, high debt, low income, young age
    risk_score = (
        (850 - df["credit_score"]) / 550  # Credit score impact
        + df["debt_to_income"]  # Debt burden
        + (df["loan_amount"] / df["income"])  # Loan size relative to income
        + (1 - (df["age"] / 70))  # Age (younger = higher risk)
        + (df["existing_loans"] * 0.2)  # Existing loans
        - (df["savings_balance"] / 50000)  # Savings cushion
    )

    # Convert risk score to probability and create binary target
    default_probability = 1 / (1 + np.exp(-risk_score))
    df["will_default"] = (default_probability > 0.6).astype(int)

    # Calculate suggested interest rate based on risk
    df["suggested_interest_rate"] = np.where(
        df["will_default"] == 1,
        np.random.uniform(12, 25, n_samples),  # High risk: 12-25%
        np.random.uniform(5, 12, n_samples),  # Low risk: 5-12%
    )

    print(f"✅ Generated {len(df)} loan applications")
    print(f"Default rate: {df['will_default'].mean():.2%}")

    return df


# Generate and save sample data
if __name__ == "__main__":
    loan_data = generate_loan_data()
    loan_data.to_csv("data/loan_dataset.csv", index=False)
    print("📁 Data saved to 'data/loan_dataset.csv'")
