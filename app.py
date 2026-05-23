import streamlit as st
import pandas as pd
import numpy as np
import catboost as cb
import joblib
import shap
import warnings

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="ICAS Diagnostics AI", page_icon="🧠", layout="wide")

# --- LOAD ASSETS (Cached so it only loads once) ---
@st.cache_resource
def load_models():
    model = joblib.load('final_tcd_model_catboost.joblib')
    scaler = joblib.load('scaler_rules_catboost.joblib')
    features = joblib.load('final_features_catboost.joblib')
    explainer = shap.TreeExplainer(model)
    return model, scaler, features, explainer

try:
    final_model, scaler_rules, final_features, final_explainer = load_models()
    models_loaded = True
except Exception as e:
    st.error(f"Error loading model files. Make sure the .joblib files are in the same folder as this script. Error: {e}")
    models_loaded = False

# --- MAIN UI ---
st.title("🧠 Intracranial Arterial Stenosis (ICAS) AI Classifier")
st.markdown("""
Welcome to the clinical deployment prototype. This tool utilizes the optimized **CatBoost architecture** to predict the probability of stenosis based on Transcranial Doppler (TCD) hemodynamics across **7 arterial segments**.
""")

if models_loaded:
    # --- SIDEBAR FOR PATIENT INPUTS ---
    st.sidebar.header("📋 Patient TCD Data")
    st.sidebar.markdown("Enter the specific hemodynamic parameters below:")

    # Demographics
    age = st.sidebar.number_input("Age (Years)", min_value=18, max_value=100, value=50)
    gender = st.sidebar.selectbox("Gender", options=['M', 'F'])

    st.sidebar.divider()

    # Hemodynamics
    vessel_options = [
        'L-MCA', 'R-MCA', 'L-ACA', 'R-ACA', 'L-VA', 'R-VA', 'BA', 
        'L-OA', 'R-OA', 'L-Siphon', 'R-Siphon', 'L-ICA', 'R-ICA'
    ]
    vessel = st.sidebar.selectbox("Arterial Segment", options=vessel_options)
    depth = st.sidebar.number_input("Insonation Depth (mm)", min_value=30, max_value=120, value=50)
    mfv = st.sidebar.number_input("Mean Flow Velocity (MFV - cm/s)", min_value=-100.0, max_value=300.0, value=55.0)
    pi = st.sidebar.number_input("Pulsatility Index (PI)", min_value=0.1, max_value=3.0, value=0.90)

    # --- PREDICTION BUTTON ---
    if st.sidebar.button("Run AI Diagnostics", type="primary"):
        
        with st.spinner("Analyzing hemodynamic parameters..."):
            # 1. Raw Input
            raw_df = pd.DataFrame({
                'MFV': [mfv], 'PI': [pi], 'Age': [age], 
                'Blood Vessel': [vessel], 'Depth': [depth], 'Gender': [gender]
            })
            
            # 2. One-Hot Encoding
            categorical_cols = ['Blood Vessel', 'Gender', 'Depth']
            df_encoded = pd.get_dummies(raw_df, columns=categorical_cols)
            
            # 3. Alignment (Top 15 Features)
            X_input = pd.DataFrame(0, index=[0], columns=final_features)
            for col in df_encoded.columns:
                if col in X_input.columns:
                    X_input[col] = df_encoded[col]

            # 4. Scaling
            temp_scale_df = raw_df[['MFV', 'PI', 'Age']].copy()
            scaled_values = scaler_rules.transform(temp_scale_df)
            
            if 'MFV' in X_input.columns: X_input['MFV'] = scaled_values[0][0]
            if 'PI' in X_input.columns:  X_input['PI'] = scaled_values[0][1]
            if 'Age' in X_input.columns: X_input['Age'] = scaled_values[0][2]

            # 5. Prediction (CatBoost specific syntax)
            prediction_proba = final_model.predict_proba(X_input)[0][1]
            prediction_label = 1 if prediction_proba > 0.5 else 0
            
            # 6. SHAP Interpretability
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                shap_values = final_explainer.shap_values(X_input)
                
            if isinstance(shap_values, list): shap_values = shap_values[1]
            
            top_idx = np.argmax(np.abs(shap_values[0]))
            top_feature = final_features[top_idx]

        # --- DISPLAY RESULTS ---
        st.subheader("Diagnostic Output")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction_label == 1:
                st.error("🚨 ABNORMAL (Stenosis Detected)")
            else:
                st.success("✅ NORMAL (Healthy Flow)")
                
        with col2:
            st.metric(label="Risk Probability", value=f"{prediction_proba * 100:.1f}%")
            
        with col3:
            st.info(f"🔍 Primary Driver: **{top_feature}**")

        # Provide a short text breakdown
        st.markdown("---")
        st.markdown("### 📊 Explainable AI (SHAP) Summary")
        if prediction_label == 1:
            st.write(f"The model classified this segment as **Abnormal** with a {prediction_proba * 100:.1f}% confidence. The decision was primarily driven by the patient's **{top_feature}**, which pushed the algorithm's risk threshold into the pathological zone.")
        else:
            st.write(f"The model classified this segment as **Normal**. The physiological measurements, primarily the **{top_feature}**, stabilized the prediction within healthy baseline parameters.")