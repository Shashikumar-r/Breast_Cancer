import os

import joblib
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)


def main():

    print("Loading Breast Cancer Wisconsin Dataset...")

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------
    data = load_breast_cancer()

    X = pd.DataFrame(
        data.data,
        columns=data.feature_names
    )

    y = pd.Series(
        data.target,
        name="target"
    )

    print(f"Dataset shape: {X.shape}")

    # ---------------------------------------------------------
    # Train-test split
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # ---------------------------------------------------------
    # Save test data
    # ---------------------------------------------------------
    test_df = X_test.copy()
    test_df["target"] = y_test

    test_df.to_csv(
        "test_data.csv",
        index=False
    )

    print("Saved test_data.csv")

    # ---------------------------------------------------------
    # Feature scaling
    # ---------------------------------------------------------
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------------------------------------------------------
    # Create model directory
    # ---------------------------------------------------------
    os.makedirs("model", exist_ok=True)

    joblib.dump(
        scaler,
        "model/scaler.pkl"
    )

    # ---------------------------------------------------------
    # Define six models
    # ---------------------------------------------------------
    models = {
        "Logistic Regression":
            LogisticRegression(
                max_iter=10000,
                random_state=42
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                random_state=42
            ),

        "kNN":
            KNeighborsClassifier(),

        "Naive Bayes":
            GaussianNB(),

        "Random Forest (Ensemble)":
            RandomForestClassifier(
                random_state=42
            ),

        "Support Vector Machine":
            SVC(
                probability=True,
                random_state=42
            ),
    }

    results = []

    print("\nTraining and Evaluating Models...")

    # ---------------------------------------------------------
    # Train and evaluate
    # ---------------------------------------------------------
    for name, model in models.items():

        if name in {
            "Decision Tree",
            "Random Forest (Ensemble)",
        }:
            X_train_input = X_train
            X_test_input = X_test
        else:
            X_train_input = X_train_scaled
            X_test_input = X_test_scaled

        model.fit(
            X_train_input,
            y_train
        )

        y_pred = model.predict(
            X_test_input
        )

        y_prob = model.predict_proba(
            X_test_input
        )[:, 1]

        # -----------------------------------------------------
        # Metrics
        # -----------------------------------------------------
        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        auc = roc_auc_score(
            y_test,
            y_prob
        )

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        mcc = matthews_corrcoef(
            y_test,
            y_pred
        )

        results.append(
            {
                "ML Model Name": name,
                "Accuracy": round(accuracy, 4),
                "AUC": round(auc, 4),
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1": round(f1, 4),
                "MCC": round(mcc, 4),
            }
        )

        # -----------------------------------------------------
        # Save model
        # -----------------------------------------------------
        safe_name = name.replace(
            " ",
            "_"
        )

        model_path = (
            f"model/{safe_name}.pkl"
        )

        joblib.dump(
            model,
            model_path
        )

        print(
            f"Saved model: {model_path}"
        )

    # ---------------------------------------------------------
    # Final results
    # ---------------------------------------------------------
    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        "model_results.csv",
        index=False
    )

    print("\n--- Model Evaluation Results ---")
    print(
        results_df.to_string(
            index=False
        )
    )

    print("\nSaved model_results.csv")


if __name__ == "__main__":
    main()