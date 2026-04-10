import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from tensorflow.keras.models import load_model
from keras.models import load_model

#===========================================
# Page Configuration
#===========================================
st.set_page_config(
    page_title="ANN Customer Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for Premium Dark Mode theme
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #f1f5f9;
    }

    /* Set background of the entire app to a deep dark slate */
    .stApp {
        background-color: #0f172a;
    }

    /* Hero Section Header - Dark Gradient */
    .hero-container {
        background: linear-gradient(135deg, #1e1b4b 0%, #1e293b 100%);
        padding: 4rem 2rem;
        border-radius: 1rem;
        color: #f1f5f9;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.5);
        border: 1px solid #334155;
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #818cf8 0%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        font-size: 1.15rem;
        color: #94a3b8;
        max-width: 750px;
        margin: 0 auto;
        font-weight: 400;
    }

    /* Card Box for grouping inputs in Dark Mode */
    .input-card {
        background: #1e293b;
        padding: 2.2rem;
        border-radius: 1.2rem;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.3);
        border: 1px solid #334155;
        margin-bottom: 1.5rem;
    }

    .input-card h4 {
        color: #818cf8;
        margin-top: 0;
        margin-bottom: 1.8rem;
        font-size: 1.3rem;
        border-bottom: 1px solid #334155;
        padding-bottom: 0.8rem;
        font-weight: 700;
    }

    /* Streamlit Input Overrides for Dark Mode */
    .stSelectbox div[data-baseweb="select"], .stNumberInput input, .stSlider {
        color: #f1f5f9 !important;
    }
    
    /* Button Styling - Vibrant Indigo */
    div.stButton > button {
        background: linear-gradient(90deg, #4f46e5 0%, #6366f1 100%) !important;
        color: white !important;
        border-radius: 0.7rem !important;
        padding: 0.8rem 2rem !important;
        font-weight: 700 !important;
        border: none !important;
        width: 100% !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 20px -5px rgba(79, 70, 229, 0.5);
        filter: brightness(1.1);
    }

    /* Result Panels in Dark Mode */
    .result-box {
        padding: 1.8rem;
        border-radius: 1rem;
        text-align: center;
        font-weight: 600;
        margin-top: 1.5rem;
        background: #0f172a;
        box-shadow: inset 0 2px 4px 0 rgb(0 0 0 / 0.06);
    }
    
    .high-risk {
        border: 1px solid #ef4444;
        color: #fca5a5;
        background: rgba(239, 68, 68, 0.1);
    }
    
    .low-risk {
        border: 1px solid #10b981;
        color: #6ee7b7;
        background: rgba(16, 185, 129, 0.1);
    }

    /* Label Styling */
    label {
        color: #94a3b8 !important;
        font-weight: 500 !important;
        margin-bottom: 0.5rem !important;
    }

    /* Footer Overrides */
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Load the Train model and encoders
@st.cache_resource
def load_assets():
    model = load_model('Practices/01_ANN Classification/04_model.h5')
    with open('Practices/01_ANN Classification/01_encoded_gender.pkl', 'rb') as f:
        encoded_gender = pickle.load(f)
    with open('Practices/01_ANN Classification/02_encoded_geo.pkl', 'rb') as f:
        encoded_geo = pickle.load(f)
    with open('Practices/01_ANN Classification/03_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, encoded_gender, encoded_geo, scaler

try:
    model, encoded_gender, encoded_geo, scaler = load_assets()
except Exception as e:
    st.error(f"Error loading model assets: {e}")
    st.stop()

# Header Section
st.markdown("""
    <div class="hero-container">
        <div class="hero-title">Customer Churn Intelligence</div>
        <div class="hero-subtitle">Predict customer behavior with high-precision ANN classification. Optimize retention and drive business growth through data-driven insights.</div>
    </div>
""", unsafe_allow_html=True)

# Main Form
col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.markdown('<div class="input-card"><h4>👤 Demographics</h4>', unsafe_allow_html=True)
    geography = st.selectbox('Geography', encoded_geo.categories_[0])
    gender = st.selectbox('Gender', encoded_gender.classes_)
    age = st.slider('Age', min_value=18, max_value=92, value=35)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="input-card"><h4>📉 Activity & Engagement</h4>', unsafe_allow_html=True)
    tenure = st.slider('Tenure (Years)', min_value=0, max_value=10, value=5)
    num_of_product = st.slider('Number of Products', min_value=1, max_value=6, value=2)
    has_cr_card = st.selectbox('Has Credit Card', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    is_active_memb = st.selectbox('Is Active Member', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="input-card"><h4>💰 Financial Status</h4>', unsafe_allow_html=True)
    credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=650)
    balance = st.number_input('Account Balance ($)', min_value=0.0, step=1000.0, value=50000.0)
    estimated_salary = st.number_input('Estimated Annual Salary ($)', min_value=0.0, step=1000.0, value=75000.0)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
    
    if st.button('Analyze Retention Risk'):
        # Prepare data
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Gender': [encoded_gender.transform([gender])[0]],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_product],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active_memb],
            'EstimatedSalary': [estimated_salary]
        })

        # Geo encoded
        geo_encoded = encoded_geo.transform([[geography]]).toarray()
        geo_encoded_df = pd.DataFrame(geo_encoded, columns=encoded_geo.get_feature_names_out(['Geography']))

        ## Combine data
        input_df = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

        ## Scale the input data
        input_data_scaled = scaler.transform(input_df)

        ## Prediction
        prediction = model.predict(input_data_scaled)
        prediction_prob = prediction[0][0]

        # Results Display
        st.markdown(f"<h3 style='text-align: center; color: #1e293b; margin-top: 1rem;'>Churn Probability: {prediction_prob*100:.2f}%</h3>", unsafe_allow_html=True)
        
        if prediction_prob >= 0.5:
            st.markdown(f"""
                <div class="result-box high-risk">
                    <div style="font-size: 2rem;">🔥</div>
                    <div>HIGH CHURN RISK</div>
                    <div style="font-weight: 400; font-size: 0.9rem; margin-top: 5px;">This customer is likely to leave. Tactical intervention is advised.</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="result-box low-risk">
                    <div style="font-size: 2rem;">✅</div>
                    <div>LOW CHURN RISK</div>
                    <div style="font-weight: 400; font-size: 0.9rem; margin-top: 5px;">This customer is currently stable. Maintain routine engagement.</div>
                </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 3rem; color: #94a3b8; font-size: 0.8rem; border-top: 1px solid #e2e8f0; padding-top: 1rem;">
    ANN Classification Engine v1.0 | Finance Solutions Inc.
</div>
""", unsafe_allow_html=True)
