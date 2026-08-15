import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    confusion_matrix, classification_report
)

# Import the 5 required Machine Learning models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="WDBC ML Pipeline Web UI", layout="wide")
st.title("🩺 Breast Cancer Diagnostic (WDBC) Machine Learning Application")

DATA_FILE = "wdbc.data"

@st.cache_resource
def train_base_pipeline():
    """Reads wdbc.data with explicit column labels, preprocesses, and trains all 5 baseline models."""
    if not os.path.exists(DATA_FILE):
        return None, None, None, None
        
    column_names = [
        'id', 'diagnosis',
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
        'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
        'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
        'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ]
    
    df = pd.read_csv(DATA_FILE, header=None, names=column_names)
    
    df_ml = df.drop(columns=['id'])
    le = LabelEncoder()
    df_ml['diagnosis'] = le.fit_transform(df_ml['diagnosis']) # M -> 1, B -> 0
    
    X = df_ml.drop(columns=['diagnosis'])
    y = df_ml['diagnosis']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    
    trained_models = {
        "Logistic Regression": LogisticRegression(max_iter=10000, random_state=42),
        "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
        "K-Nearest Neighbor Classifier": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes Classifier (Gaussian)": GaussianNB(),
        "Ensemble Model (Random Forest)": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    for name, model in trained_models.items():
        model.fit(X_train_scaled, y_train)
        
    return trained_models, scaler, X_test_scaled, y_test, le

# Load pipeline framework assets
pipeline_data = train_base_pipeline()

if pipeline_data[0] is None:
    st.error(f"❌ Missing base training asset '{DATA_FILE}' in your project directory root. Please place it next to app.py.")
else:
    models, scaler, default_X_test, default_y_test, le = pipeline_data
    
    # -------------------------------------------------------------------------
    # FEATURE A: DATASET UPLOAD OPTION (CSV) WITH AUTO-CLEANING
    # -------------------------------------------------------------------------
    st.header("📂 a. Dataset Upload Option (CSV Test Data)")
    st.info("💡 Upload your `test_data.csv` file. Columns containing spaces or 'error' are cleaned automatically.")
    
    uploaded_file = st.file_uploader("Upload test dataset CSV file:", type=["csv"])
    
    X_eval = default_X_test
    y_eval = default_y_test.values
    
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            
            uploaded_y = None
            if 'diagnosis' in uploaded_df.columns:
                diag_col = uploaded_df['diagnosis']
                if diag_col.dtype == object or diag_col.astype(str).str.contains('B|M|b|m').any():
                    uploaded_y = diag_col.map({'B': 1, 'M': 1, 'b': 0, 'm': 0}).values
                    if pd.isna(uploaded_y).any():
                        uploaded_y = le.transform(diag_col.astype(str))
                else:
                    uploaded_y = diag_col.values.astype(int)
                uploaded_df = uploaded_df.drop(columns=['diagnosis'])
            
            if 'id' in uploaded_df.columns:
                uploaded_df = uploaded_df.drop(columns=['id'])
                
            # Clean incoming column names (spaces and error mismatches)
            uploaded_df.columns = uploaded_df.columns.str.strip().str.lower()
            uploaded_df.columns = uploaded_df.columns.str.replace(' ', '_')
            uploaded_df.columns = uploaded_df.columns.str.replace('_error', '_se')

            if uploaded_df.shape[1] == 30:
                X_eval = pd.DataFrame(scaler.transform(uploaded_df), columns=default_X_test.columns)
                if uploaded_y is not None and len(uploaded_y) == len(uploaded_df):
                    y_eval = np.array(uploaded_y, dtype=int)
                else:
                    y_eval = np.zeros(len(uploaded_df), dtype=int)
                    st.info(f"ℹ️ No targets found in CSV. Using dummy markers for {len(uploaded_df)} samples.")
                st.success(f"✅ Successfully processed custom test array with {len(uploaded_df)} rows.")
            else:
                st.warning(f"⚠️ Shape mismatch: Expected 30 features, got {uploaded_df.shape[1]}. Using default cache.")
        except Exception as e:
            st.error(f"❌ Upload parsing error: {str(e)}")

    # -------------------------------------------------------------------------
    # COMPREHENSIVE COMPARATIVE PERFORMANCE MATRIX (All Models)
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.header("📊 Comparative Evaluation Performance Matrix")
    st.write("Scorecard evaluating all implemented classifiers simultaneously on the active evaluation dataset:")

    metrics_summary = []
    for name, model in models.items():
        y_pred = model.predict(X_eval)
        try:
            y_prob = model.predict_proba(X_eval)[:, 1]
            auc = roc_auc_score(y_eval, y_prob)
        except (AttributeError, ValueError):
            auc = 0.0

        metrics_summary.append({
            "Model Name": name,
            "Accuracy": round(accuracy_score(y_eval, y_pred), 4),
            "AUC Score": round(auc, 4),
            "Precision": round(precision_score(y_eval, y_pred, zero_division=0), 4),
            "Recall": round(recall_score(y_eval, y_pred, zero_division=0), 4),
            "F1 Score": round(f1_score(y_eval, y_pred, zero_division=0), 4),
            "MCC Score": round(matthews_corrcoef(y_eval, y_pred), 4)
        })

    results_table = pd.DataFrame(metrics_summary)
    st.dataframe(results_table, use_container_width=True)
    
    # Download button for performance scorecard
    st.download_button(
        label="📥 Download Performance Matrix CSV",
        data=results_table.to_csv(index=False),
        file_name="streamlit_evaluation_report.csv",
        mime="text/csv"
    )

    # -------------------------------------------------------------------------
    # SINGLE MODEL DEEP INSPECTION DROPDOWN
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.header("⚙️ Single Model Deep Inspection")
    selected_model_name = st.selectbox(
        "Choose a model to examine breakdown statistics:",
        options=list(models.keys())
    )
    
    active_model = models[selected_model_name]
    y_pred_single = active_model.predict(X_eval)
    
    left_col, right_col = st.columns(2)
    with left_col:
        st.subheader("Confusion Matrix")
        st.write(confusion_matrix(y_eval, y_pred_single))
        
    with right_col:
        st.subheader("Classification Report")
        report = classification_report(y_eval, y_pred_single, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).transpose())
