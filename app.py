import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

st.set_page_config(
    page_title="ML Classification App",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Machine Learning Classification Web App")

st.markdown(
    """
    This application demonstrates six classification models trained on the
    **Breast Cancer Wisconsin (Diagnostic) Dataset**.

    Upload the test CSV file, select a machine learning model, and evaluate
    its performance using Accuracy, AUC, Precision, Recall, F1 Score, and MCC.
    """
)

# ---------------------------------------------------------
# Model files
# ---------------------------------------------------------
model_files = {
    "Logistic Regression": "model/Logistic_Regression.pkl",
    "Decision Tree": "model/Decision_Tree.pkl",
    "kNN": "model/kNN.pkl",
    "Naive Bayes": "model/Naive_Bayes.pkl",
    "Random Forest (Ensemble)": "model/Random_Forest_(Ensemble).pkl",
    "Support Vector Machine": "model/Support_Vector_Machine.pkl",
}

tree_based_models = {
    "Decision Tree",
    "Random Forest (Ensemble)",
}

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.header("📥 Test Data")

if os.path.exists("test_data.csv"):
    with open("test_data.csv", "rb") as file:
        st.sidebar.download_button(
            label="Download Sample test_data.csv",
            data=file,
            file_name="test_data.csv",
            mime="text/csv",
        )
else:
    st.sidebar.warning(
        "test_data.csv not found in the repository."
    )

st.sidebar.header("Instructions")
st.sidebar.markdown(
    """
    Download the sample data using the button above (if needed).
    1. Upload `test_data.csv`.
    2. Select a trained ML model.
    3. Click **Evaluate Model**.
    4. View the evaluation metrics and confusion matrix.
    """
)

# ---------------------------------------------------------
# Upload test data
# ---------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Test Data (CSV)",
    type=["csv"]
)

if uploaded_file is None:
    st.info(
        "👆 Please upload the test CSV file to evaluate the models."
    )
    st.stop()

# ---------------------------------------------------------
# Read uploaded CSV
# ---------------------------------------------------------
try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Unable to read the uploaded CSV file: {e}")
    st.stop()

st.subheader("📄 Uploaded Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

# ---------------------------------------------------------
# Validate target column
# ---------------------------------------------------------
if "target" not in df.columns:
    st.error(
        "The uploaded CSV must contain a 'target' column."
    )
    st.stop()

target_col = "target"

X_test = df.drop(columns=[target_col])
y_test = df[target_col]

# ---------------------------------------------------------
# Load scaler
# ---------------------------------------------------------
scaler_path = "model/scaler.pkl"

if not os.path.exists(scaler_path):
    st.error(
        "model/scaler.pkl was not found. "
        "Please make sure the model directory is uploaded to GitHub."
    )
    st.stop()

try:
    scaler = joblib.load(scaler_path)
    X_test_scaled = scaler.transform(X_test)
except Exception as e:
    st.error(
        f"Error loading or applying the scaler: {e}"
    )
    st.stop()

# ---------------------------------------------------------
# Model selection
# ---------------------------------------------------------
selected_model_name = st.selectbox(
    "Select ML Model",
    list(model_files.keys())
)

# ---------------------------------------------------------
# Evaluate model
# ---------------------------------------------------------
if st.button("🚀 Evaluate Model", type="primary"):

    model_path = model_files[selected_model_name]

    if not os.path.exists(model_path):
        st.error(
            f"Model file not found: {model_path}"
        )
        st.stop()

    try:
        model = joblib.load(model_path)

        # Tree models use unscaled features
        if selected_model_name in tree_based_models:
            X_input = X_test
        else:
            X_input = X_test_scaled

        y_pred = model.predict(X_input)
        y_prob = model.predict_proba(X_input)[:, 1]

        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        st.success(
            f"Evaluation completed successfully for **{selected_model_name}**."
        )

        st.subheader("📊 Evaluation Metrics")

        metrics_df = pd.DataFrame(
            {
                "Metric": [
                    "Accuracy",
                    "AUC",
                    "Precision",
                    "Recall",
                    "F1 Score",
                    "MCC",
                ],
                "Score": [
                    round(acc, 4),
                    round(auc, 4),
                    round(prec, 4),
                    round(rec, 4),
                    round(f1, 4),
                    round(mcc, 4),
                ],
            }
        )

        st.dataframe(
            metrics_df,
            hide_index=True,
            use_container_width=True,
        )

        # -------------------------------------------------
        # Confusion Matrix
        # -------------------------------------------------
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🧮 Confusion Matrix")

            cm = confusion_matrix(y_test, y_pred)

            fig, ax = plt.subplots(figsize=(5, 4))

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                ax=ax,
            )

            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(selected_model_name)

            st.pyplot(fig)
            plt.close(fig)

        # -------------------------------------------------
        # Classification Report
        # -------------------------------------------------
        with col2:
            st.subheader("📝 Classification Report")

            report = classification_report(
                y_test,
                y_pred,
                zero_division=0
            )

            st.text(report)

    except Exception as e:
        st.error(
            f"An error occurred while evaluating the model: {e}"
        )