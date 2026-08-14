import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="ML Classification App", layout="wide")

st.title("🤖 Machine Learning Classification Web App")
st.markdown("""
This application demonstrates multiple classification models trained on the **Breast Cancer Wisconsin (Diagnostic) Dataset**. 
Upload the test dataset to evaluate model performance, view metrics, and analyze the confusion matrix.
""")

# --- NEW FEATURE: Download Button for Evaluator Convenience ---
st.sidebar.header("📥 Need Test Data?")
st.sidebar.markdown("Click below to download the sample test data used for this project:")

if os.path.exists("test_data.csv"):
    with open("test_data.csv", "rb") as file:
        st.sidebar.download_button(
            label="Download Sample test_data.csv",
            data=file,
            file_name="test_data.csv",
            mime="text/csv"
        )
else:
    st.sidebar.warning("test_data.csv not found in repository.")
# ----------------------------------------------------------------

st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. Download the sample data using the button above (if needed).
2. Upload the `test_data.csv` file below.
3. Select a trained ML model from the dropdown.
4. Click **Evaluate Model** to see predictions and metrics.
""")

model_files = {
    "Logistic Regression": "model/Logistic_Regression.pkl",
    "Decision Tree": "model/Decision_Tree.pkl",
    "kNN": "model/kNN.pkl",
    "Naive Bayes": "model/Naive_Bayes.pkl",
    "Random Forest (Ensemble)": "model/Random_Forest_(Ensemble).pkl",
    "Support Vector Machine": "model/Support_Vector_Machine.pkl"
}

uploaded_file = st.file_uploader("Upload Test Data (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Dataset Preview")
    st.dataframe(df.head())
    
    target_col = 'target' if 'target' in df.columns else df.columns[-1]
    X_test = df.drop(columns=[target_col])
    y_test = df[target_col]
    
    try:
        scaler = joblib.load('model/scaler.pkl')
        X_test_scaled = scaler.transform(X_test)
    except Exception as e:
        st.error(f"Error loading scaler: {e}")
        st.stop()
        
    selected_model_name = st.selectbox("Select ML Model", list(model_files.keys()))
    
    if st.button("Evaluate Model"):
        model_path = model_files[selected_model_name]
        if not os.path.exists(model_path):
            st.error(f"Model file not found. Ensure models are saved in the 'model/' directory.")
        else:
            model = joblib.load(model_path)
            
            if selected_model_name in ["Decision Tree", "Random Forest (Ensemble)"]:
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_pred = model.predict(X_test_scaled)
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
                
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            mcc = matthews_corrcoef(y_test, y_pred)
            
            st.subheader("📊 Evaluation Metrics")
            metrics_df = pd.DataFrame({
                "Metric": ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"],
                "Score": [f"{acc:.4f}", f"{auc:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}", f"{mcc:.4f}"]
            })
            st.dataframe(metrics_df, hide_index=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🧮 Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)
            
            with col2:
                st.subheader("📝 Classification Report")
                st.text(classification_report(y_test, y_pred))
else:
    st.info("👆 Please upload a CSV file to get started. (You can download the sample file from the sidebar!)")