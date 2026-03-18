from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os

from data_preprocessing import load_and_clean_data


def train():
    # Load data
    df = load_and_clean_data("data/Telco-Customer-Churn.csv")

    # Split features and target
    X = df.drop("MonthlyCharges", axis=1)
    y = df["MonthlyCharges"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    print("R2 Score:", r2_score(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))

    # Ensure models folder exists
    os.makedirs("models", exist_ok=True)

    # Save model
    joblib.dump(model, "models/telecom_model.pkl")
    print("✅ Model saved successfully!")


if __name__ == "__main__":
    train()