import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt


class LoanAITrainer:
    def __init__(self):
        self.model = None
        self.feature_columns = None

    def load_combined_data(self, file_path="final_combined_dataset.csv"):
        """
        Load the combined dataset and prepare it for AI training
        """
        print("📊 LOADING COMBINED DATASET FOR AI TRAINING")
        print("=" * 50)

        try:
            self.df = pd.read_csv(file_path)
            print(f"✅ Dataset loaded: {self.df.shape}")
            print(f"Columns: {self.df.columns.tolist()}")

            # Show basic info
            print(f"\n📈 DATASET INFO:")
            print(f"Total records: {len(self.df):,}")
            print(f"Total features: {len(self.df.columns)}")
            print(f"\nData types:")
            print(self.df.dtypes.value_counts())

            return True

        except FileNotFoundError:
            print(f"❌ File {file_path} not found!")
            return False

    def explore_dataset(self):
        """
        Explore the dataset to understand its structure
        """
        print("\n🔍 EXPLORING DATASET STRUCTURE")
        print("=" * 50)

        # Show first few rows
        print("First 5 rows:")
        print(self.df.head())

        # Check for missing values
        print(f"\n📊 MISSING VALUES:")
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100

        for col, missing_count in missing_data.items():
            if missing_count > 0:
                print(f"  {col}: {missing_count} ({missing_percent[col]:.1f}%)")

        if missing_data.sum() == 0:
            print("  ✅ No missing values!")

        # Look for potential target columns
        print(f"\n🎯 LOOKING FOR TARGET VARIABLES:")
        target_keywords = [
            "default",
            "fraud",
            "risk",
            "approve",
            "status",
            "class",
            "target",
            "label",
        ]

        potential_targets = []
        for col in self.df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in target_keywords):
                potential_targets.append(col)
                print(f"  ✅ Potential target: {col}")

        if not potential_targets:
            print("  ⚠️  No obvious target columns found. We'll need to create one.")

        return potential_targets

    def prepare_features_and_target(self, target_column=None):
        """
        Prepare features (X) and target (y) for model training
        """
        print("\n🛠️ PREPARING FEATURES AND TARGET")
        print("=" * 50)

        # If no target column specified, try to find or create one
        if target_column is None:
            target_column = self.find_or_create_target()

        # Select features (exclude target and non-numeric columns)
        exclude_columns = [target_column, "id", "customer_id", "application_id"]

        self.feature_columns = [
            col
            for col in self.df.columns
            if col not in exclude_columns
            and pd.api.types.is_numeric_dtype(self.df[col])
        ]

        print(f"Selected {len(self.feature_columns)} features:")
        for col in self.feature_columns:
            print(f"  📊 {col}")

        # Prepare X and y
        X = self.df[self.feature_columns]
        y = self.df[target_column]

        print(f"\n🎯 FINAL DATA SHAPE:")
        print(f"Features (X): {X.shape}")
        print(f"Target (y): {y.shape}")
        print(f"Target distribution:")
        print(y.value_counts())

        return X, y, target_column

    def find_or_create_target(self):
        """
        Find or create a target variable for loan default prediction
        """
        print("🔍 Finding or creating target variable...")

        # First, try to find existing target columns
        target_keywords = ["default", "fraud", "risk", "status"]
        for col in self.df.columns:
            if any(keyword in col.lower() for keyword in target_keywords):
                print(f"✅ Using existing target: {col}")
                return col

        # If no target found, create a synthetic one based on risk factors
        print("⚙️ Creating synthetic target variable based on risk factors...")

        # Common risk factors for loan defaults
        risk_factors = []

        # Look for columns that indicate risk
        risk_columns = {
            "income": ["income", "salary", "annual_income"],
            "debt": ["debt", "loan_amount", "credit_utilization"],
            "score": ["credit_score", "score", "rating"],
            "age": ["age", "employment_years", "experience"],
        }

        for factor, keywords in risk_columns.items():
            for col in self.df.columns:
                if any(
                    keyword in col.lower() for keyword in keywords
                ) and pd.api.types.is_numeric_dtype(self.df[col]):
                    risk_factors.append(col)
                    break

        if len(risk_factors) >= 2:
            print(f"✅ Using risk factors: {risk_factors}")

            # Create a simple risk score (you can customize this)
            risk_score = 0
            for col in risk_factors:
                if "income" in col.lower() or "salary" in col.lower():
                    # Higher income = lower risk
                    risk_score -= self.df[col] / self.df[col].max()
                elif "debt" in col.lower() or "loan" in col.lower():
                    # Higher debt = higher risk
                    risk_score += self.df[col] / self.df[col].max()
                elif "score" in col.lower() and "credit" in col.lower():
                    # Higher credit score = lower risk
                    risk_score -= self.df[col] / self.df[col].max()
                else:
                    # For other numeric columns
                    risk_score += self.df[col] / self.df[col].max()

            # Convert risk score to binary target (will_default)
            threshold = risk_score.median()
            self.df["will_default"] = (risk_score > threshold).astype(int)

            print(f"✅ Created target 'will_default'")
            print(f"Default rate: {self.df['will_default'].mean():.2%}")

            return "will_default"
        else:
            # Last resort: create random target for demonstration
            print("⚠️ Creating random target for demonstration...")
            self.df["will_default"] = np.random.choice(
                [0, 1], size=len(self.df), p=[0.8, 0.2]
            )
            return "will_default"

    def train_model(self, X, y):
        """
        Train the AI model
        """
        print("\n🤖 TRAINING AI MODEL")
        print("=" * 50)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"Training set: {X_train.shape}")
        print(f"Testing set: {X_test.shape}")

        # Train Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=100, random_state=42, max_depth=10
        )

        self.model.fit(X_train, y_train)

        # Make predictions
        y_pred = self.model.predict(X_test)

        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n🎯 MODEL PERFORMANCE:")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"\nDetailed Report:")
        print(classification_report(y_test, y_pred))

        # Feature importance
        self.plot_feature_importance(X.columns)

        return accuracy

    def plot_feature_importance(self, feature_names):
        """
        Plot feature importance
        """
        importance = self.model.feature_importances_
        indices = np.argsort(importance)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title("Feature Importance in Loan Default Prediction")
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(
            range(len(importance)), [feature_names[i] for i in indices], rotation=45
        )
        plt.tight_layout()
        plt.show()

        print("\n📊 TOP 5 MOST IMPORTANT FEATURES:")
        for i in range(min(5, len(importance))):
            print(
                f"  {i + 1}. {feature_names[indices[i]]}: {importance[indices[i]]:.3f}"
            )

    def save_model(self, filename="loan_default_model.joblib"):
        """
        Save the trained model
        """
        if self.model is not None:
            joblib.dump(
                {
                    "model": self.model,
                    "feature_columns": self.feature_columns,
                    "feature_importance": self.model.feature_importances_,
                },
                filename,
            )
            print(f"💾 Model saved as: {filename}")
        else:
            print("❌ No model to save!")


# 🚀 COMPLETE AI TRAINING PIPELINE
def main():
    """
    Complete AI training using the combined dataset
    """
    print("🤖 AI LOAN DEFAULT PREDICTION SYSTEM")
    print("=" * 60)

    # Initialize trainer
    trainer = LoanAITrainer()

    # Step 1: Load the combined dataset
    if not trainer.load_combined_data("final_combined_dataset.csv"):
        # Try alternative names
        for file in [
            "combined_data.csv",
            "manual_combined_data.csv",
            "debug_combined_data.csv",
        ]:
            if os.path.exists(file):
                print(f"🔄 Trying alternative: {file}")
                if trainer.load_combined_data(file):
                    break
        else:
            print("❌ No combined dataset found!")
            return

    # Step 2: Explore the dataset
    trainer.explore_dataset()

    # Step 3: Prepare features and target
    X, y, target_name = trainer.prepare_features_and_target()

    # Step 4: Train the AI model
    accuracy = trainer.train_model(X, y)

    # Step 5: Save the model
    trainer.save_model()

    print(f"\n🎉 AI MODEL TRAINING COMPLETE!")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Model saved: 'loan_default_model.joblib'")
    print(f"Ready for predictions! 🚀")


# Run the AI training
if __name__ == "__main__":
    main()
