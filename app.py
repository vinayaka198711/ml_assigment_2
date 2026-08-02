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

st.set_page_config(page_title="WDBC ML Pipeline", layout="wide")
st.title("🩺 Breast Cancer Diagnostic (WDBC) Machine Learning Application")

# Global baseline training config
DATA_FILE = "wdbc.data"

@st.cache_resource
def train_base_pipeline():
    """Trains all 5 baseline models using descriptive feature names to match test_data.csv."""
    if not os.path.exists(DATA_FILE):
        return None, None, None, None
        
    # Define exact descriptive feature names to align perfectly with test_data.csv
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
    
    # Preprocess dimensions
    df_ml = df.drop(columns=['id'])
    le = LabelEncoder()
    df_ml['diagnosis'] = le.fit_transform(df_ml['diagnosis']) # M -> 1, B -> 0
    
    X = df_ml.drop(columns=['diagnosis'])
    y = df_ml['diagnosis']
    
    # Partition datasets (80/20 train/test split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    
    # Keep X_train and X_test as DataFrames with explicit column names
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    
    trained_models = {
        "Logistic Regression": LogisticRegression(max_iter=10000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "K-Nearest Neighbor": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes (Gaussian)": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    for name, model in trained_models.items():
        model.fit(X_train_scaled, y_train)
        
    return trained_models, scaler, X_test_scaled, y_test

# Load/train the underlying pipeline framework
models, scaler, default_X_test, default_y_test = train_base_pipeline()

if models is None:
    st.error(f"❌ Missing base training asset '{DATA_FILE}' in your project directory root. Please upload it to your GitHub Repository.")
else:
    # -------------------------------------------------------------------------
    # FEATURE A: DATASET UPLOAD OPTION (CSV)
    # -------------------------------------------------------------------------
    st.header("📂 a. Dataset Upload Option (CSV Test Data Only)")
    st.info("💡 **Capacity Optimization Rule**: Streamlit cloud storage tiers use limited tracking allocations. Upload a tailored `test_data.csv` file featuring structural test inputs below.")
    
    uploaded_file = st.file_uploader("Upload your test dataset CSV file:", type=["csv"])
    
    # Set fallback variables if no file is uploaded
    X_eval = default_X_test
    y_eval = default_y_test
    
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            
            # Extract diagnosis if present in the uploaded test set
            if 'diagnosis' in uploaded_df.columns:
                if uploaded_df['diagnosis'].dtype == object:
                    le = LabelEncoder()
                    uploaded_df['diagnosis'] = le.fit_transform(uploaded_df['diagnosis'])
                y_eval = uploaded_df['diagnosis'].values
                uploaded_df = uploaded_df.drop(columns=['diagnosis'])
            
            if 'id' in uploaded_df.columns:
                uploaded_df = uploaded_df.drop(columns=['id'])
                
            # FIXED LOGIC: Extract column count using .shape[1] instead of the tuple object
            if uploaded_df.shape[1] == 30:
                # Transform into a DataFrame to preserve feature column tracking names
                X_eval = pd.DataFrame(scaler.transform(uploaded_df), columns=default_X_test.columns)
                st.success(f"✅ Successfully loaded custom test array with {uploaded_df.shape[0]} samples.")
            else:
                st.warning(f"⚠️ Column shape mismatch. Upload features {uploaded_df.shape[1]} attributes instead of the requested 30 dimensions. Falling back to internal test set cache.")
        except Exception as e:
            st.error(f"❌ Upload parsing failure: {str(e)}")

    # -------------------------------------------------------------------------
    # FEATURE B: MODEL SELECTION DROPDOWN
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.header("⚙️ b. Model Selection Dropdown")
    selected_model_name = st.selectbox(
        "Choose an implemented machine learning architecture to evaluate:",
        options=list(models.keys())
    )
    
    # Run predictions using selected active model
    active_model = models[selected_model_name]
    y_pred = active_model.predict(X_eval)
    y_prob = active_model.predict_proba(X_eval)[:, 1]

    # -------------------------------------------------------------------------
    # FEATURE C: DISPLAY OF EVALUATION METRICS
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.header(f"📊 c. Evaluation Metrics Matrix for {selected_model_name}")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Accuracy", f"{accuracy_score(y_eval, y_pred):.4f}")
    col2.metric("AUC Score", f"{roc_auc_score(y_eval, y_prob):.4f}")
    col3.metric("Precision", f"{precision_score(y_eval, y_pred, zero_division=0):.4f}")
    col4.metric("Recall", f"{recall_score(y_eval, y_pred, zero_division=0):.4f}")
    col5.metric("F1 Score", f"{f1_score(y_eval, y_pred, zero_division=0):.4f}")
    col6.metric("MCC Score", f"{matthews_corrcoef(y_eval, y_pred):.4f}")

    # -------------------------------------------------------------------------
    # FEATURE D: CONFUSION MATRIX & CLASSIFICATION REPORT
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.header("🔬 d. Advanced Diagnostic Analysis Reports")
    
    left_report_col, right_report_col = st.columns(2)
    
    with left_report_col:
        st.subheader("📋 Classification Report Summary")
        text_report = classification_report(
            y_eval, y_pred, 
            target_names=["Benign (0)", "Malignant (1)"], 
            zero_division=0
        )
        st.code(text_report, language="text")
        
    with right_report_col:
        st.subheader("🧩 Confusion Matrix Heatmap Representation")
        cm = confusion_matrix(y_eval, y_pred)
        
        # Build stylized visualization table dataframe
        cm_df = pd.DataFrame(
            cm,
            index=["Actual Benign (0)", "Actual Malignant (1)"],
            columns=["Predicted Benign (0)", "Predicted Malignant (1)"]
        )
        st.dataframe(cm_df.style.background_gradient(cmap="Blues"), use_container_width=True)
        
        # Breakdown raw counts
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
        st.markdown(f"""
        * **True Negatives (TN)**: `{tn}` instances correctly identified as Benign.
        * **False Positives (FP)**: `{fp}` samples incorrectly flagged as Malignant.
        * **False Negatives (FN)**: `{fn}` malignant cases missed by the classifier pipeline.
        * **True Positives (TP)**: `{tp}` instances correctly identified as Malignant.
        """)
