import os
import pandas as pd
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

def main():
    print("Loading Breast Cancer Wisconsin Dataset...")
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save test data to CSV (Required for Streamlit App upload)
    test_df = X_test.copy()
    test_df['target'] = y_test
    test_df.to_csv('test_data.csv', index=False)
    print("Saved test_data.csv")

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create model directory
    os.makedirs('model', exist_ok=True)
    joblib.dump(scaler, 'model/scaler.pkl')

    # Define 6 Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=10000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "kNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(random_state=42),
        "Support Vector Machine": SVC(probability=True, random_state=42)
    }

    results = []
    print("Training and Evaluating Models...")
    
    for name, model in models.items():
        # Tree-based models don't strictly require scaling
        if name in ["Decision Tree", "Random Forest (Ensemble)"]:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            
        # Calculate Metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        results.append({
            "ML Model Name": name, "Accuracy": round(acc, 4), "AUC": round(auc, 4),
            "Precision": round(prec, 4), "Recall": round(rec, 4), "F1": round(f1, 4), "MCC": round(mcc, 4)
        })
        
        # Save Model
        safe_name = name.replace(" ", "_")
        joblib.dump(model, f'model/{safe_name}.pkl')
        print(f"Saved model: {safe_name}.pkl")

    # Print Final Results
    results_df = pd.DataFrame(results)
    print("\n--- Model Evaluation Results ---")
    print(results_df.to_markdown(index=False))

if __name__ == "__main__":
    main()